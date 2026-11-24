"""
Module: src/vision/burn_analyzer.py
Mục đích: Bộ phân tích vết bỏng toàn diện (Clinical Burn Analyzer):
1. Phân đoạn vết bỏng bằng U-Net (ResNet-34) để sinh mặt nạ (Mask).
2. Lọc nhiễu hình thái học (Morphological Cleaning).
3. Tính toán diện tích vết thương:
   - Số lượng pixel tổn thương.
   - Phần trăm diện tích trên khung hình (% Frame Area).
   - Quy đổi ra cm² thực tế (thông qua Calibration Factor / Thước chuẩn).
   - Ước tính phần trăm tổn thương cơ thể (% TBSA) theo vùng giải phẫu (Luật số 9 Wallace).
4. Tính toán lượng dịch bù Parkland cho ca bỏng nặng.
5. Cắt ảnh ROI và Phân loại mức độ bỏng (Độ 1, Độ 2, Độ 3) bằng ResNet-34.
6. Trích xuất ảnh trực quan hóa (Overlay màu bán trong suốt, Contour, Bounding Box).
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple, List, Union
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Đường dẫn project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    SEGMENTATION_MODELS,
    CLASSIFICATION_MODELS,
    DEFAULT_SEG_MODEL,
    DEFAULT_CLS_MODEL,
)
from src.vision.models import build_segmentation_model, build_classification_model
from src.vision.postprocessing import clean_binary_mask, calculate_burn_area_percentage, get_burn_bounding_box, crop_roi
from src.utils.gradcam import GradCAM, overlay_cam_on_image


# Bảng tỷ lệ diện tích cơ thể theo Luật số 9 của Wallace (Rule of Nines)
RULE_OF_NINES = {
    "head_neck": 9.0,         # Đầu - Mặt - Cổ
    "left_arm": 9.0,          # Toàn bộ tay trái
    "right_arm": 9.0,         # Toàn bộ tay phải
    "chest_abdomen": 18.0,    # Ngực - Bụng trước
    "back_buttocks": 18.0,    # Lưng - Mông sau
    "left_leg": 18.0,         # Toàn bộ chân trái
    "right_leg": 18.0,        # Toàn bộ chân phải
    "perineum": 1.0,          # Bộ phận sinh dục
    "unknown": 10.0           # Mặc định vùng ước lượng
}

CLASS_NAMES = ["Độ 1 (Superficial)", "Độ 2 (Partial Thickness)", "Độ 3 (Full Thickness)"]


class BurnAnalyzer:
    """
    Pipeline tích hợp hoàn chỉnh phân tích tổn thương do bỏng từ ảnh chụp y tế.
    """
    def __init__(
        self,
        seg_checkpoint: Optional[str] = None,
        cls_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        seg_model_key: str = DEFAULT_SEG_MODEL,
        cls_model_key: str = DEFAULT_CLS_MODEL,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.seg_model_key = seg_model_key if seg_model_key in SEGMENTATION_MODELS else DEFAULT_SEG_MODEL
        self.cls_model_key = cls_model_key if cls_model_key in CLASSIFICATION_MODELS else DEFAULT_CLS_MODEL

        self.seg_cfg = SEGMENTATION_MODELS[self.seg_model_key]
        self.cls_cfg = CLASSIFICATION_MODELS[self.cls_model_key]

        self.seg_checkpoint = seg_checkpoint or self.seg_cfg["checkpoint"]
        self.cls_checkpoint = cls_checkpoint or self.cls_cfg["checkpoint"]

        self.seg_model = None
        self.cls_model = None

        self._init_transforms()

    def _init_transforms(self):
        # Tiền xử lý cho Segmentation (Resize 256x256, chuẩn hóa ImageNet)
        self.seg_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Tiền xử lý cho Classification (Resize 224x224, chuẩn hóa ImageNet)
        self.cls_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def load_models(self):
        """Khởi tạo và nạp trọng số cho cả 2 mô hình theo cấu hình từ config.py"""
        # 1. Nạp mô hình phân đoạn (U-Net / EA-UNet / Attention U-Net)
        if os.path.exists(self.seg_checkpoint):
            ckpt = torch.load(self.seg_checkpoint, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            arch = self.seg_cfg.get("arch", "unet")
            if any("core_branch" in k or "gated_fusion" in k for k in state_dict.keys()):
                arch = "ea_unet"
            self.seg_model = build_segmentation_model(
                arch=arch,
                backbone=self.seg_cfg["backbone"],
                attention=self.seg_cfg.get("attention", "none"),
                classes=1,
                pretrained=False,
            )
            self.seg_model.load_state_dict(state_dict)
            self.seg_model.to(self.device)
            self.seg_model.eval()
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint phân đoạn: {self.seg_checkpoint}")

        # 2. Nạp mô hình phân loại (ResNet-34 / ConvNeXt-Tiny)
        if os.path.exists(self.cls_checkpoint):
            self.cls_model = build_classification_model(
                model_name=self.cls_cfg["model_name"],
                num_classes=3,
                pretrained=False,
            )
            ckpt = torch.load(self.cls_checkpoint, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.cls_model.load_state_dict(state_dict)
            self.cls_model.to(self.device)
            self.cls_model.eval()
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint phân loại: {self.cls_checkpoint}")

    def predict_mask(self, image_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán mặt nạ phân đoạn nhị phân từ ảnh RGB gốc.
        Trả về mask nhị phân kích thước gốc của ảnh (0: nền, 255: vết bỏng).
        """
        if self.seg_model is None:
            self.load_models()

        h_orig, w_orig = image_rgb.shape[:2]
        transformed = self.seg_transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.seg_model(transformed)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            mask_256 = (probs > threshold).astype(np.uint8) * 255

        # Resize về kích thước ban đầu của ảnh
        mask_orig = cv2.resize(mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        # Khử nhiễu hình thái học
        cleaned_mask = clean_binary_mask(mask_orig, min_area=80, kernel_size=5)
        return cleaned_mask

    def calculate_metrics(
        self,
        mask: np.ndarray,
        pixels_per_cm: Optional[float] = None,
        body_part: str = "chest_abdomen",
        weight_kg: Optional[float] = None,
        predicted_class_id: int = 1
    ) -> Dict[str, Any]:
        """
        Tính toán toàn bộ thông số diện tích y khoa và phác đồ bù dịch.
        
        Tham số:
        - mask: Mặt nạ nhị phân (255 là vùng bỏng).
        - pixels_per_cm: Hệ số quy đổi (số pixel tương ứng với 1 cm thực tế).
        - body_part: Vùng cơ thể theo Luật số 9 (head_neck, left_arm, chest_abdomen, etc.).
        - weight_kg: Cân nặng bệnh nhân (dùng tính công thức Parkland).
        - predicted_class_id: Cấp độ bỏng dự đoán (0: Độ 1, 1: Độ 2, 2: Độ 3).
        """
        h, w = mask.shape[:2]
        total_pixels = h * w
        burn_pixels = int(np.count_nonzero(mask > 127))

        # 1. Tỷ lệ phần trăm diện tích trên khung hình ảnh
        pct_frame = calculate_burn_area_percentage(mask)

        # 2. Diện tích thực tế quy đổi cm² (nếu có thước chuẩn)
        area_cm2 = None
        if pixels_per_cm and pixels_per_cm > 0:
            area_cm2 = round(burn_pixels / (pixels_per_cm ** 2), 2)

        # 3. Ước tính phần trăm tổng diện tích bề mặt cơ thể (% TBSA)
        # Chuẩn y khoa: Nếu có thước chuẩn (area_cm2), tính chính xác theo diện tích cơ thể (BSA Mosteller)
        # Nếu không có thước đo, mới ước lượng tương đối theo tỷ lệ diện tích khung ảnh (pct_frame)
        region_weight = float(RULE_OF_NINES.get(body_part, 9.0))
        if area_cm2 and area_cm2 > 0:
            patient_weight = float(weight_kg if (weight_kg and weight_kg > 0) else 68.0)
            # Công thức Mosteller tính diện tích bề mặt cơ thể BSA (m2 -> cm2, giả định chiều cao chuẩn 170cm)
            bsa_cm2 = float(np.sqrt((170.0 * patient_weight) / 3600.0) * 10000.0)
            tbsa_from_area = float((area_cm2 / bsa_cm2) * 100.0)
            estimated_tbsa = float(round(min(region_weight, max(0.001, tbsa_from_area)), 3))
        else:
            # Fallback: Giả định tỷ lệ khung hình
            estimated_tbsa = float(round(min(region_weight, (pct_frame / 100.0) * region_weight), 2))

        # 4. Tính toán phác đồ bù dịch Parkland (Chuẩn Bộ Y tế & Hiệp hội Bỏng Hoa Kỳ ABA)
        # Chỉ định truyền dịch tĩnh mạch KHI VÀ CHỈ KHI:
        # - Bỏng Độ 2 hoặc Độ 3 với diện tích TBSA >= 10-15% (có nguy cơ sốc bỏng)
        # - TUYỆT ĐỐI KHÔNG TRUYỀN DỊCH TĨNH MẠCH KHI:
        #   + Bỏng Độ 1 (Superficial): Không tổn thương mao mạch, không gây thất thoát thể tích tuần hoàn.
        #   + Bỏng Độ 2 / Độ 3 diện tích nhỏ (< 10% TBSA): Không có nguy cơ sốc bỏng.
        parkland_plan = None
        if weight_kg and weight_kg > 0 and estimated_tbsa > 0:
            total_fluid_ml = float(round(4.0 * float(weight_kg) * estimated_tbsa, 1))
            fluid_first_8h = float(round(total_fluid_ml * 0.5, 1))
            rate_first_8h = float(round(fluid_first_8h / 8.0, 1))      # ml/h
            rate_next_16h = float(round((total_fluid_ml * 0.5) / 16.0, 1))  # ml/h

            is_degree_1 = bool(predicted_class_id == 0)
            is_indicated = bool((not is_degree_1) and (estimated_tbsa >= 10.0))
            is_urgent = bool((not is_degree_1) and (estimated_tbsa >= 15.0))

            if is_degree_1:
                guidance = "Bỏng Độ 1 (thượng bì) không gây thất thoát huyết tương. Theo hướng dẫn của Bộ Y tế, KHÔNG áp dụng công thức Parkland và KHÔNG truyền dịch tĩnh mạch. Chỉ cần uống nước / Oresol bù điện giải."
            elif not is_indicated:
                guidance = f"Diện tích tổn thương nhỏ ({estimated_tbsa}% TBSA), không có nguy cơ sốc giảm thể tích. TUYỆT ĐỐI KHÔNG truyền dịch tĩnh mạch để tránh quá tải tuần hoàn. Bù nước đường uống (Oresol) và tập trung xử trí tại chỗ."
            else:
                guidance = f"Chỉ định cấp cứu: Diện tích tổn thương rộng ({estimated_tbsa}% TBSA), nguy cơ sốc bỏng cao. Bắt buộc truyền Ringer Lactate hồi sức theo phác đồ Parkland."

            parkland_plan = {
                "indicated": bool(is_indicated),
                "total_fluid_24h_ml": float(total_fluid_ml) if is_indicated else 0.0,
                "calculated_raw_ml": float(total_fluid_ml),
                "first_8h_ml": float(fluid_first_8h) if is_indicated else 0.0,
                "rate_first_8h_ml_h": float(rate_first_8h) if is_indicated else 0.0,
                "rate_next_16h_ml_h": float(rate_next_16h) if is_indicated else 0.0,
                "fluid_type": "Dung dịch Ringer Lactate" if is_indicated else "Bù dịch đường uống (Oresol / Nước ấm)",
                "urgent_triage": bool(is_urgent),
                "clinical_guidance": guidance
            }

        return {
            "image_dimensions": f"{w}x{h}",
            "total_pixels": int(total_pixels),
            "burn_pixels": int(burn_pixels),
            "pct_frame_area": float(pct_frame),
            "area_cm2": float(area_cm2) if area_cm2 is not None else None,
            "body_part": body_part,
            "estimated_tbsa_pct": float(estimated_tbsa),
            "parkland_plan": parkland_plan
        }

    def classify_severity(self, image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Cắt ROI theo vết bỏng và phân loại cấp độ bằng ResNet-34.
        """
        if self.cls_model is None:
            self.load_models()

        box = get_burn_bounding_box(mask, padding_ratio=0.15)
        roi_img = crop_roi(image_rgb, box, target_size=(224, 224), make_square=True)

        transformed = self.cls_transform(image=roi_img)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cls_model(transformed)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        conf_pct = float(round(confidence * 100.0, 2))
        is_abstaining = bool(conf_pct < 60.0)

        # Trích xuất Grad-CAM (hỗ trợ cả ResNet-34 và ConvNeXt-Tiny)
        gradcam_overlay = None
        try:
            target_layer = None
            if hasattr(self.cls_model, "layer4"):
                target_layer = self.cls_model.layer4[-1]
            elif hasattr(self.cls_model, "stages"):
                target_layer = self.cls_model.stages[-1]

            if target_layer is not None:
                cam_extractor = GradCAM(self.cls_model, target_layer)
                # Forward pass có gradients
                transformed.requires_grad = True
                heatmap = cam_extractor.generate_heatmap(transformed, class_idx=pred_idx)
                gradcam_overlay = overlay_cam_on_image(roi_img, heatmap, alpha=0.5)
                cam_extractor.remove_hooks()
        except Exception:
            # Fallback nếu model cấu trúc khác
            pass

        return {
            "predicted_class_id": pred_idx,
            "predicted_degree": CLASS_NAMES[pred_idx],
            "confidence": conf_pct,
            "is_abstaining": is_abstaining,
            "ethical_status": "ABSTAIN" if is_abstaining else "ACCEPT",
            "abstain_message": (
                f"Độ tin cậy của AI chỉ đạt {conf_pct}% (dưới ngưỡng an toàn 60.0%). "
                "Theo nguyên tắc đạo đức y tế 'Không gây hại', AI TỪ CHỐI đưa ra phác đồ tự điều trị. Cần bác sĩ khám trực tiếp!"
                if is_abstaining else "Độ tin cậy đạt chuẩn an toàn y tế."
            ),
            "probabilities": {
                "Độ 1 (Superficial)": round(float(probs[0]) * 100.0, 2),
                "Độ 2 (Partial Thickness)": round(float(probs[1]) * 100.0, 2),
                "Độ 3 (Full Thickness)": round(float(probs[2]) * 100.0, 2)
            },
            "bounding_box": box,
            "roi_image": roi_img,
            "gradcam_image": gradcam_overlay
        }

    def generate_visualizations(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        box: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Tạo các hình ảnh minh họa y khoa:
        1. Mask Overlay (đè màu đỏ bán trong suốt lên vết bỏng).
        2. Contour Outline (vẽ đường viền màu vàng ôm quanh vết bỏng và Bounding Box).
        """
        overlay = image_rgb.copy()
        burn_indices = mask > 127
        # Tô màu đỏ rực (255, 0, 0) với độ trong suốt 40%
        overlay[burn_indices] = (0.6 * overlay[burn_indices] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

        # Vẽ đường viền Contour
        contours_img = image_rgb.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (255, 215, 0), 2)  # Màu vàng viền 2px

        # Vẽ Bounding Box nếu có
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(contours_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Hộp chữ nhật xanh lá

        return {
            "overlay": overlay,
            "contours": contours_img
        }

    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        pixels_per_cm: Optional[float] = None,
        body_part: str = "chest_abdomen",
        weight_kg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Hàm chính chạy toàn bộ Pipeline End-to-End từ ảnh đầu vào.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            bgr = cv2.imread(image_input)
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_input

        # 1. Dự đoán Mask
        mask = self.predict_mask(image_rgb)

        # 2. Phân loại mức độ bỏng trên ROI trước để xác định độ sâu
        classification = self.classify_severity(image_rgb, mask)

        # 3. Tính diện tích và phác đồ dựa trên độ sâu lâm sàng
        metrics = self.calculate_metrics(
            mask=mask,
            pixels_per_cm=pixels_per_cm,
            body_part=body_part,
            weight_kg=weight_kg,
            predicted_class_id=classification.get("predicted_class_id", 1)
        )

        # 4. Sinh ảnh trực quan hóa
        visuals = self.generate_visualizations(image_rgb, mask, classification["bounding_box"])

        return {
            "metrics": metrics,
            "classification": classification,
            "visualizations": visuals,
            "mask": mask
        }
