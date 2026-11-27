"""
Module: src/vision/burn_analyzer.py
Mục đích: Bộ phân tích vết bỏng toàn diện nhận thức Độ bất định Kép (Dual-Uncertainty Clinical Burn Analyzer):
1. Phân đoạn vết bỏng bằng 5-Fold U-Net (ResNet-34) (+4x TTA) để sinh mặt nạ (Mask) & Bản đồ Độ bất định Biên từng pixel.
2. Lọc nhiễu hình thái học (Morphological Cleaning).
3. Tính toán diện tích vết thương kèm Khoảng tin cậy (Spatial Uncertainty Interval):
   - Số lượng pixel tổn thương & % Frame Area [low, high].
   - Quy đổi ra cm² thực tế (nếu có thước chuẩn).
   - Ước tính khoảng % TBSA theo Luật số 9 Wallace & Cảnh báo ranh giới sốc bỏng 10% TBSA.
4. Tính toán phác đồ bù dịch Parkland kèm khoảng an toàn thể tích 24h (ml).
5. Cắt ảnh ROI và Phân loại mức độ bỏng (Độ 1, Độ 2, Độ 3) bằng 5-Fold Asymmetric R-EDL + Selective Clinical Triage.
6. Trích xuất ảnh trực quan hóa (Overlay, Contour, Pixel Uncertainty Heatmap, Grad-CAM).
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
from src.vision.train_seg import forward_seg_tta
from src.vision.edl_loss import logits_to_evidence, get_edl_metrics
from src.vision.spatial_uncertainty import (
    compute_pixel_uncertainty_maps,
    summarize_boundary_uncertainty,
    compute_clinical_tbsa_and_parkland_interval,
    render_uncertainty_heatmap_rgb,
)
from src.vision.postprocessing import clean_binary_mask, calculate_burn_area_percentage, get_burn_bounding_box, crop_roi
from src.utils.gradcam import GradCAM, overlay_cam_on_image


RULE_OF_NINES = {
    "head_neck": 9.0,
    "left_arm": 9.0,
    "right_arm": 9.0,
    "chest_abdomen": 18.0,
    "back_buttocks": 18.0,
    "left_leg": 18.0,
    "right_leg": 18.0,
    "perineum": 1.0,
    "unknown": 10.0,
}

CLASS_NAMES = ["Độ 1 (Superficial)", "Độ 2 (Partial Thickness)", "Độ 3 (Full Thickness)"]


class BurnAnalyzer:
    """
    Pipeline tích hợp hoàn chỉnh phân tích tổn thương bỏng nhận thức độ bất định kép (Dual-Uncertainty).
    Tự động hỗ trợ cả 5-Fold Ensemble lẫn 1-Model đơn lẻ (tương thích ngược 100%).
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
        self.seg_models: List[torch.nn.Module] = []
        self.cls_model = None
        self.cls_models: List[torch.nn.Module] = []

        self._last_prob_stack: Optional[np.ndarray] = None
        self._last_u_maps: Optional[Dict[str, np.ndarray]] = None

        self._init_transforms()

    def _init_transforms(self):
        self.seg_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.cls_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _load_single_seg(self, path: str) -> torch.nn.Module:
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        arch = self.seg_cfg.get("arch", "unet")
        if any("core_branch" in k or "gated_fusion" in k for k in state_dict.keys()):
            arch = "ea_unet"
        m = build_segmentation_model(
            arch=arch,
            backbone=self.seg_cfg["backbone"],
            attention=self.seg_cfg.get("attention", "none"),
            classes=1,
            pretrained=False,
        )
        m.load_state_dict(state_dict)
        m.to(self.device).eval()
        return m

    def _load_single_cls(self, path: str) -> torch.nn.Module:
        m = build_classification_model(
            model_name=self.cls_cfg["model_name"],
            num_classes=3,
            pretrained=False,
        )
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state_dict)
        m.to(self.device).eval()
        return m

    def load_models(self):
        """Nạp mô hình Phân đoạn và Phân loại (ưu tiên 5-Fold Ensemble nếu đủ file, tự động fallback 1-model)."""
        self.seg_models = []
        kfold_seg_paths = self.seg_cfg.get("checkpoints", [])
        if kfold_seg_paths and all(os.path.exists(p) for p in kfold_seg_paths):
            self.seg_models = [self._load_single_seg(p) for p in kfold_seg_paths]
            self.seg_model = self.seg_models[0]
        elif os.path.exists(self.seg_checkpoint):
            self.seg_model = self._load_single_seg(self.seg_checkpoint)
            self.seg_models = [self.seg_model]
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint phân đoạn: {self.seg_checkpoint}")

        self.cls_models = []
        kfold_cls_paths = self.cls_cfg.get("checkpoints", [])
        if kfold_cls_paths and all(os.path.exists(p) for p in kfold_cls_paths):
            self.cls_models = [self._load_single_cls(p) for p in kfold_cls_paths]
            self.cls_model = self.cls_models[0]
        elif os.path.exists(self.cls_checkpoint):
            self.cls_model = self._load_single_cls(self.cls_checkpoint)
            self.cls_models = [self.cls_model]
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint phân loại: {self.cls_checkpoint}")

    def predict_mask(self, image_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán mặt nạ phân đoạn nhị phân từ ảnh RGB gốc (kèm lưu bộ nhớ đệm bản đồ độ bất định không gian).
        Trả về mask nhị phân kích thước gốc của ảnh (0: nền, 255: vết bỏng).
        """
        if not self.seg_models and self.seg_model is None:
            self.load_models()
        active_seg_models = self.seg_models if self.seg_models else ([self.seg_model] if self.seg_model is not None else [])
        if not active_seg_models:
            raise RuntimeError("Chưa nạp được mô hình phân đoạn nào.")

        h_orig, w_orig = image_rgb.shape[:2]
        transformed = self.seg_transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)
        use_tta = bool(self.seg_cfg.get("use_tta", False))

        fold_probs = []
        with torch.no_grad():
            for m in active_seg_models:
                if use_tta:
                    logits = forward_seg_tta(m, transformed)
                else:
                    out = m(transformed)
                    logits = out[0] if isinstance(out, tuple) else out
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                fold_probs.append(probs)

        self._last_prob_stack = np.stack(fold_probs, axis=0)
        self._last_u_maps = compute_pixel_uncertainty_maps(self._last_prob_stack)

        mask_256 = (self._last_u_maps["mean_prob"] > threshold).astype(np.uint8) * 255
        mask_orig = cv2.resize(mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        cleaned_mask = clean_binary_mask(mask_orig, min_area=80, kernel_size=5)
        return cleaned_mask

    def calculate_metrics(
        self,
        mask: np.ndarray,
        pixels_per_cm: Optional[float] = None,
        body_part: str = "chest_abdomen",
        weight_kg: Optional[float] = None,
        predicted_class_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Tính toán thông số diện tích y khoa, khoảng độ bất định diện tích (TBSA Interval) và phác đồ bù dịch Parkland.
        """
        h, w = mask.shape[:2]
        total_pixels = h * w
        burn_pixels = int(np.count_nonzero(mask > 127))
        pct_frame = calculate_burn_area_percentage(mask)

        area_cm2 = None
        if pixels_per_cm and pixels_per_cm > 0:
            area_cm2 = round(burn_pixels / (pixels_per_cm ** 2), 2)

        region_weight = float(RULE_OF_NINES.get(body_part, 9.0))
        if area_cm2 and area_cm2 > 0:
            patient_weight = float(weight_kg if (weight_kg and weight_kg > 0) else 68.0)
            bsa_cm2 = float(np.sqrt((170.0 * patient_weight) / 3600.0) * 10000.0)
            tbsa_from_area = float((area_cm2 / bsa_cm2) * 100.0)
            estimated_tbsa = float(round(min(region_weight, max(0.001, tbsa_from_area)), 3))
        else:
            estimated_tbsa = float(round(min(region_weight, (pct_frame / 100.0) * region_weight), 2))

        # Lượng hóa khoảng bất định diện tích & khoảng Parkland từ bản đồ xác suất (Hướng 1)
        prob_source = self._last_prob_stack if self._last_prob_stack is not None else (mask > 127).astype(np.float32)[np.newaxis, ...]
        spatial_interval = compute_clinical_tbsa_and_parkland_interval(
            prob_source,
            region_weight=region_weight,
            weight_kg=weight_kg,
            predicted_class_id=predicted_class_id,
            pixels_per_cm=pixels_per_cm,
            orig_hw=(h, w),
        )
        boundary_stats = summarize_boundary_uncertainty(prob_source)

        parkland_plan = None
        if weight_kg and weight_kg > 0 and estimated_tbsa > 0:
            total_fluid_ml = float(round(4.0 * float(weight_kg) * estimated_tbsa, 1))
            fluid_first_8h = float(round(total_fluid_ml * 0.5, 1))
            rate_first_8h = float(round(fluid_first_8h / 8.0, 1))
            rate_next_16h = float(round((total_fluid_ml * 0.5) / 16.0, 1))

            is_degree_1 = bool(predicted_class_id == 0)
            is_indicated = bool((not is_degree_1) and (estimated_tbsa >= 10.0))
            is_urgent = bool((not is_degree_1) and (estimated_tbsa >= 15.0))
            borderline_warn = bool(spatial_interval["borderline_10pct_warning"])

            if is_degree_1:
                guidance = (
                    "Bỏng Độ 1 (thượng bì) không gây thất thoát huyết tương. Theo hướng dẫn của Bộ Y tế, "
                    "KHÔNG áp dụng công thức Parkland và KHÔNG truyền dịch tĩnh mạch. Chỉ cần uống nước / Oresol bù điện giải."
                )
            elif not is_indicated:
                guidance = (
                    f"Diện tích tổn thương nhỏ ({estimated_tbsa}% TBSA), không có nguy cơ sốc giảm thể tích. "
                    "TUYỆT ĐỐI KHÔNG truyền dịch tĩnh mạch để tránh quá tải tuần hoàn. Bù nước đường uống (Oresol) và xử trí tại chỗ."
                )
                if borderline_warn:
                    tbsa_lo, tbsa_hi = spatial_interval["tbsa_interval_pct"]
                    guidance += (
                        f" [CẢNH BÁO RANH GIỚI BẤT ĐỊNH]: Khoảng diện tích tin cậy [{tbsa_lo}% – {tbsa_hi}% TBSA] "
                        "chạm ngưỡng 10% TBSA, đề nghị bác sĩ đo kiểm tra lại diện tích trực tiếp!"
                    )
            else:
                pk_int = spatial_interval.get("parkland_interval")
                range_note = (
                    f" (Khoảng tham chiếu theo độ bất định biên: {pk_int['raw_24h_low_ml']} – {pk_int['raw_24h_high_ml']} ml/24h)."
                    if pk_int else ""
                )
                guidance = (
                    f"Chỉ định cấp cứu: Diện tích tổn thương rộng ({estimated_tbsa}% TBSA), nguy cơ sốc bỏng cao. "
                    f"Bắt buộc truyền Ringer Lactate hồi sức theo phác đồ Parkland{range_note}"
                )

            parkland_plan = {
                "indicated": bool(is_indicated),
                "total_fluid_24h_ml": float(total_fluid_ml) if is_indicated else 0.0,
                "calculated_raw_ml": float(total_fluid_ml),
                "first_8h_ml": float(fluid_first_8h) if is_indicated else 0.0,
                "rate_first_8h_ml_h": float(rate_first_8h) if is_indicated else 0.0,
                "rate_next_16h_ml_h": float(rate_next_16h) if is_indicated else 0.0,
                "fluid_type": "Dung dịch Ringer Lactate" if is_indicated else "Bù dịch đường uống (Oresol / Nước ấm)",
                "urgent_triage": bool(is_urgent),
                "borderline_10pct_warning": borderline_warn,
                "uncertainty_interval_ml": spatial_interval.get("parkland_interval"),
                "clinical_guidance": guidance,
            }

        return {
            "image_dimensions": f"{w}x{h}",
            "total_pixels": int(total_pixels),
            "burn_pixels": int(burn_pixels),
            "pct_frame_area": float(pct_frame),
            "pct_frame_interval": spatial_interval["pct_frame_interval"],
            "area_cm2": float(area_cm2) if area_cm2 is not None else None,
            "body_part": body_part,
            "estimated_tbsa_pct": float(estimated_tbsa),
            "tbsa_interval_pct": spatial_interval["tbsa_interval_pct"],
            "tbsa_half_width_pct": spatial_interval["tbsa_half_width_pct"],
            "borderline_10pct_warning": spatial_interval["borderline_10pct_warning"],
            "boundary_entropy": round(boundary_stats["boundary_entropy"], 4),
            "parkland_plan": parkland_plan,
        }

    def classify_severity(self, image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Cắt ROI theo vết bỏng và phân loại cấp độ (hỗ trợ cả 5-Fold Asymmetric R-EDL Evidential Fusion & Softmax).
        """
        if not self.cls_models and self.cls_model is None:
            self.load_models()
        active_cls_models = self.cls_models if self.cls_models else ([self.cls_model] if self.cls_model is not None else [])
        if not active_cls_models:
            raise RuntimeError("Chưa nạp được mô hình phân loại nào.")

        box = get_burn_bounding_box(mask, padding_ratio=0.15)
        roi_img = crop_roi(image_rgb, box, target_size=(224, 224), make_square=True)
        transformed = self.cls_transform(image=roi_img)["image"].unsqueeze(0).to(self.device)

        is_edl = (self.cls_cfg.get("loss_type", "softmax") == "edl")
        edl_act = self.cls_cfg.get("edl_activation", "exp")
        u_thresh = float(self.cls_cfg.get("uncertainty_threshold", 0.1180))
        deg3_c3 = float(self.cls_cfg.get("deg3_cost_c3", 1.0))

        with torch.no_grad():
            if is_edl:
                ev_sum = None
                for m in active_cls_models:
                    ev = logits_to_evidence(m(transformed), mode=edl_act)
                    ev_sum = ev if ev_sum is None else (ev_sum + ev)
                mean_ev = ev_sum / float(len(active_cls_models))
                probs_t, u_t = get_edl_metrics(mean_ev, num_classes=3)
                probs = probs_t.squeeze().cpu().numpy()
                epistemic_u = float(u_t.squeeze().cpu().item())
            else:
                prob_sum = None
                for m in active_cls_models:
                    pr = torch.softmax(m(transformed), dim=1)
                    prob_sum = pr if prob_sum is None else (prob_sum + pr)
                probs = (prob_sum / float(len(active_cls_models))).squeeze().cpu().numpy()
                epistemic_u = float(1.0 - np.max(probs))

        # Áp dụng hậu hiệu chuẩn chi phí Bayes Độ 3 nếu được cấu hình
        decision_probs = probs.copy()
        if abs(deg3_c3 - 1.0) > 1e-5:
            decision_probs[2] *= deg3_c3
            decision_probs = decision_probs / np.clip(np.sum(decision_probs), 1e-8, None)

        pred_idx = int(np.argmax(decision_probs))
        confidence = float(probs[pred_idx])
        conf_pct = float(round(confidence * 100.0, 2))

        if is_edl:
            is_abstaining = bool(epistemic_u > u_thresh or conf_pct < 60.0)
            abstain_msg = (
                f"Độ bất định nhận thức R-EDL u = {epistemic_u:.4f} vượt ngưỡng phân luồng an toàn (τ = {u_thresh:.4f}) "
                f"hoặc độ tin cậy thấp ({conf_pct}%). Hệ thống kích hoạt SELECTIVE CLINICAL TRIAGE: "
                "TỪ CHỐI tự kết luận và chuyển bác sĩ chuyên khoa hội chẩn trực tiếp!"
                if is_abstaining else
                f"Độ bất định nhận thức thấp (u = {epistemic_u:.4f} ≤ τ = {u_thresh:.4f}), đạt chuẩn an toàn lâm sàng."
            )
        else:
            is_abstaining = bool(conf_pct < 60.0)
            abstain_msg = (
                f"Độ tin cậy của AI chỉ đạt {conf_pct}% (dưới ngưỡng an toàn 60.0%). "
                "Theo nguyên tắc đạo đức y tế 'Không gây hại', AI TỪ CHỐI đưa ra phác đồ tự điều trị. Cần bác sĩ khám trực tiếp!"
                if is_abstaining else "Độ tin cậy đạt chuẩn an toàn y tế."
            )

        gradcam_overlay = None
        try:
            rep_model = active_cls_models[0]
            target_layer = None
            if hasattr(rep_model, "layer4"):
                target_layer = rep_model.layer4[-1]
            elif hasattr(rep_model, "stages"):
                target_layer = rep_model.stages[-1]

            if target_layer is not None:
                cam_extractor = GradCAM(rep_model, target_layer)
                inp_cam = transformed.clone().detach().requires_grad_(True)
                heatmap = cam_extractor.generate_heatmap(inp_cam, class_idx=pred_idx)
                gradcam_overlay = overlay_cam_on_image(roi_img, heatmap, alpha=0.5)
                cam_extractor.remove_hooks()
        except Exception:
            pass

        return {
            "predicted_class_id": pred_idx,
            "predicted_degree": CLASS_NAMES[pred_idx],
            "confidence": conf_pct,
            "epistemic_uncertainty": round(epistemic_u, 4),
            "uncertainty_threshold": round(u_thresh, 4) if is_edl else None,
            "ensemble_size": len(active_cls_models),
            "is_edl": is_edl,
            "is_abstaining": is_abstaining,
            "ethical_status": "ABSTAIN" if is_abstaining else "ACCEPT",
            "abstain_message": abstain_msg,
            "probabilities": {
                "Độ 1 (Superficial)": round(float(probs[0]) * 100.0, 2),
                "Độ 2 (Partial Thickness)": round(float(probs[1]) * 100.0, 2),
                "Độ 3 (Full Thickness)": round(float(probs[2]) * 100.0, 2),
            },
            "bounding_box": box,
            "roi_image": roi_img,
            "gradcam_image": gradcam_overlay,
        }

    def generate_visualizations(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        box: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        overlay = image_rgb.copy()
        burn_indices = mask > 127
        overlay[burn_indices] = (0.6 * overlay[burn_indices] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

        contours_img = image_rgb.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (255, 215, 0), 2)

        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(contours_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        u_heatmap = None
        if self._last_u_maps is not None:
            u_heatmap = render_uncertainty_heatmap_rgb(image_rgb, self._last_u_maps["predictive_entropy"])

        return {
            "overlay": overlay,
            "contours": contours_img,
            "uncertainty_heatmap": u_heatmap,
        }

    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        pixels_per_cm: Optional[float] = None,
        body_part: str = "chest_abdomen",
        weight_kg: Optional[float] = None,
    ) -> Dict[str, Any]:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            bgr = cv2.imread(image_input)
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_input

        mask = self.predict_mask(image_rgb)
        classification = self.classify_severity(image_rgb, mask)
        metrics = self.calculate_metrics(
            mask=mask,
            pixels_per_cm=pixels_per_cm,
            body_part=body_part,
            weight_kg=weight_kg,
            predicted_class_id=classification.get("predicted_class_id", 1),
        )
        visuals = self.generate_visualizations(image_rgb, mask, classification["bounding_box"])

        return {
            "metrics": metrics,
            "classification": classification,
            "visualizations": visuals,
            "mask": mask,
        }
