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
from src.vision.anatomical_tbsa import (
    ANATOMICAL_REGIONS,
    RULE_OF_NINES,
    compute_anatomical_tbsa,
)
from src.utils.gradcam import GradCAM, overlay_cam_on_image

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
        # 1. Nạp mô hình phân đoạn (U-Net / Attention U-Net)
        if os.path.exists(self.seg_checkpoint):
            self.seg_model = build_segmentation_model(
                arch=self.seg_cfg["arch"],
                backbone=self.seg_cfg["backbone"],
                attention=self.seg_cfg["attention"],
                classes=1,
                pretrained=False,
            )
            ckpt = torch.load(self.seg_checkpoint, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
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
        predicted_class_id: int = 1,
        image_rgb: Optional[np.ndarray] = None,
        region_coverage: float = 1.0,
        calibration_mode: str = "auto",
        patient_age: Optional[int] = None,
        burn_time_hours: Optional[float] = None,
        height_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán toàn bộ thông số diện tích y khoa (chuẩn hóa giải phẫu Rule of Nines /
        Palmar 1% / Lund-Browder trên diện tích vùng da chụp được) và phác đồ bù dịch Parkland.
        """
        h, w = mask.shape[:2]
        tbsa_info = compute_anatomical_tbsa(
            burn_mask=mask,
            image_rgb=image_rgb,
            body_part=body_part,
            region_coverage=region_coverage,
            calibration_mode=calibration_mode,
            pixels_per_cm=pixels_per_cm,
            weight_kg=weight_kg,
            patient_age=patient_age,
            height_cm=height_cm,
        )

        total_pixels = tbsa_info["total_pixels"]
        burn_pixels = tbsa_info["burn_pixels"]
        pct_frame = tbsa_info["pct_frame_area"]
        area_cm2 = tbsa_info["area_cm2"]
        estimated_tbsa = tbsa_info["estimated_tbsa_pct"]

        # Tính toán phác đồ bù dịch Parkland (Chuẩn Bộ Y tế & Hiệp hội Bỏng Hoa Kỳ ABA)
        # Bắt buộc dùng % TBSA đã chuẩn hóa giải phẫu (TUYỆT ĐỐI KHÔNG dùng % khung ảnh!)
        parkland_plan = None
        elapsed_h = float(burn_time_hours) if (burn_time_hours is not None and burn_time_hours >= 0) else 0.0
        rem_first_8h = float(max(0.5, 8.0 - elapsed_h)) if elapsed_h < 8.0 else 0.5

        if weight_kg and weight_kg > 0 and estimated_tbsa > 0:
            total_fluid_ml = float(round(4.0 * float(weight_kg) * estimated_tbsa, 1))
            fluid_first_8h = float(round(total_fluid_ml * 0.5, 1))
            rate_first_8h = float(round(fluid_first_8h / rem_first_8h, 1))  # ml/h trên số giờ còn lại của 8h đầu
            rate_first_8h_std = float(round(fluid_first_8h / 8.0, 1))
            rate_next_16h = float(round((total_fluid_ml * 0.5) / 16.0, 1))  # ml/h

            is_pediatric = bool(patient_age is not None and 0 <= int(patient_age) < 15)
            shock_threshold = 10.0
            urgent_threshold = 10.0 if is_pediatric else 15.0

            is_degree_1 = bool(predicted_class_id == 0)
            is_indicated = bool((not is_degree_1) and (estimated_tbsa >= shock_threshold))
            is_urgent = bool((not is_degree_1) and (estimated_tbsa >= urgent_threshold))

            norm_note = (
                f"Đã chuẩn hóa từ {pct_frame}% khung ảnh → {tbsa_info['pct_burn_of_skin']}% diện tích da "
                f"vùng '{tbsa_info['body_part_name']}' ({tbsa_info['region_weight_pct']}%) = {estimated_tbsa}% TBSA."
            )

            if is_degree_1:
                guidance = (
                    "Bỏng Độ 1 (thượng bì) không gây thất thoát huyết tương. Theo hướng dẫn của Bộ Y tế, "
                    "KHÔNG áp dụng công thức Parkland và KHÔNG truyền dịch tĩnh mạch. Chỉ cần uống nước / Oresol bù điện giải."
                )
            elif not is_indicated:
                guidance = (
                    f"Diện tích tổn thương sau chuẩn hóa giải phẫu là {estimated_tbsa}% TBSA "
                    f"(< {shock_threshold:.0f}% TBSA), không có nguy cơ sốc giảm thể tích. "
                    f"TUYỆT ĐỐI KHÔNG truyền dịch tĩnh mạch để tránh quá tải tuần hoàn. Bù nước đường uống (Oresol) và xử trí tại chỗ."
                )
            else:
                guidance = (
                    f"Chỉ định cấp cứu: Diện tích tổn thương rộng ({estimated_tbsa}% TBSA sau chuẩn hóa giải phẫu), "
                    f"nguy cơ sốc bỏng cao. Bắt buộc truyền Ringer Lactate hồi sức theo phác đồ Parkland."
                )

            parkland_plan = {
                "indicated": bool(is_indicated),
                "total_fluid_24h_ml": float(total_fluid_ml) if is_indicated else 0.0,
                "calculated_raw_ml": float(total_fluid_ml),
                "first_8h_ml": float(fluid_first_8h) if is_indicated else 0.0,
                "remaining_first_8h_hours": float(round(rem_first_8h, 1)),
                "rate_first_8h_ml_h": float(rate_first_8h) if is_indicated else 0.0,
                "rate_first_8h_standard_ml_h": float(rate_first_8h_std) if is_indicated else 0.0,
                "rate_next_16h_ml_h": float(rate_next_16h) if is_indicated else 0.0,
                "fluid_type": "Dung dịch Ringer Lactate" if is_indicated else "Bù dịch đường uống (Oresol / Nước ấm)",
                "urgent_triage": bool(is_urgent),
                "anatomical_normalization_note": norm_note,
                "clinical_guidance": guidance,
            }

        return {
            "image_dimensions": f"{w}x{h}",
            "total_pixels": int(total_pixels),
            "burn_pixels": int(burn_pixels),
            "skin_pixels": int(tbsa_info["skin_pixels"]),
            "pct_frame_area": float(pct_frame),
            "pct_skin_in_frame": float(tbsa_info["pct_skin_in_frame"]),
            "pct_burn_of_skin": float(tbsa_info["pct_burn_of_skin"]),
            "area_cm2": float(area_cm2) if area_cm2 is not None else None,
            "estimated_area_cm2": float(tbsa_info["estimated_area_cm2"]) if tbsa_info["estimated_area_cm2"] is not None else None,
            "area_is_estimated": bool(tbsa_info["area_is_estimated"]),
            "bsa_cm2": float(tbsa_info["bsa_cm2"]),
            "body_part": body_part,
            "body_part_name": tbsa_info["body_part_name"],
            "region_weight_pct": float(tbsa_info["region_weight_pct"]),
            "region_coverage": float(tbsa_info["region_coverage"]),
            "effective_visible_region_pct": float(tbsa_info["effective_visible_region_pct"]),
            "calibration_mode": tbsa_info["calibration_mode"],
            "atlas_method": tbsa_info["atlas_method"],
            "tbsa_method": tbsa_info["tbsa_method"],
            "formula_summary": tbsa_info["formula_summary"],
            "anatomical_tbsa_pct": float(tbsa_info["anatomical_tbsa_pct"]),
            "estimated_tbsa_pct": float(estimated_tbsa),
            "is_closeup_detected": bool(tbsa_info["is_closeup_detected"]),
            "framing_warning": tbsa_info["framing_warning"],
            "burn_time_hours": float(elapsed_h) if burn_time_hours is not None else None,
            "parkland_plan": parkland_plan,
            "_skin_mask": tbsa_info["skin_mask"],
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
        box: Optional[List[int]] = None,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Tạo các hình ảnh minh họa y khoa:
        1. Mask Overlay (đè màu đỏ bán trong suốt lên vết bỏng).
        2. Contour Outline:
           - Viền xanh lam (Cyan): Ranh giới vùng da bộ phận chụp được (loại bỏ phông nền).
           - Viền vàng (Gold): Ranh giới vết bỏng U-Net.
           - Khung chữ nhật xanh lá: Vùng cắt ROI đưa vào ResNet-34.
        """
        overlay = image_rgb.copy()
        burn_indices = mask > 127
        # Tô màu đỏ rực (255, 0, 0) với độ trong suốt 40%
        overlay[burn_indices] = (0.6 * overlay[burn_indices] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

        # Vẽ đường viền Contour
        contours_img = image_rgb.copy()
        if skin_mask is not None:
            skin_ratio = np.count_nonzero(skin_mask > 127) / max(1, skin_mask.size)
            if 0.01 < skin_ratio < 0.97:
                skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contours_img, skin_contours, -1, (0, 212, 255), 2)  # Viền xanh lam vùng da

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (255, 215, 0), 2)  # Màu vàng viền 2px vết bỏng

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
        weight_kg: Optional[float] = None,
        region_coverage: float = 1.0,
        calibration_mode: str = "auto",
        patient_age: Optional[int] = None,
        burn_time_hours: Optional[float] = None,
        height_cm: Optional[float] = None,
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

        # 3. Tính diện tích chuẩn hóa giải phẫu và phác đồ dựa trên độ sâu lâm sàng
        metrics = self.calculate_metrics(
            mask=mask,
            pixels_per_cm=pixels_per_cm,
            body_part=body_part,
            weight_kg=weight_kg,
            predicted_class_id=classification.get("predicted_class_id", 1),
            image_rgb=image_rgb,
            region_coverage=region_coverage,
            calibration_mode=calibration_mode,
            patient_age=patient_age,
            burn_time_hours=burn_time_hours,
            height_cm=height_cm,
        )
        skin_mask = metrics.pop("_skin_mask", None)

        # 4. Sinh ảnh trực quan hóa (kèm ranh giới vùng da bộ phận)
        visuals = self.generate_visualizations(
            image_rgb,
            mask,
            classification["bounding_box"],
            skin_mask=skin_mask,
        )

        return {
            "metrics": metrics,
            "classification": classification,
            "visualizations": visuals,
            "mask": mask,
            "skin_mask": skin_mask,
        }
