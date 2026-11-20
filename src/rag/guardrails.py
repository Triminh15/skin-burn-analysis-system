"""
Module: src/rag/guardrails.py
Mục đích: Chốt chặn Đạo đức Y tế & An toàn Lâm sàng (Medical Ethics & Clinical Guardrails)
Tuân thủ nguyên tắc 'Primum non nocere' (Trước hết là không gây hại):
1. Chốt chặn độ tin cậy thấp (Low-confidence Abstention / Selective Classification):
   Khi độ tin cậy của AI < 60%, AI từ chối đưa ra phác đồ tự điều trị để tránh đoán bừa.
2. Chốt chặn chống ngộ nhận bỏng nặng:
   Bỏng Độ 3 hoặc TBSA >= 10% bắt buộc chuyển viện, cấm tự ý điều trị tại nhà.
3. Chặn các bài thuốc dân gian độc hại (Negative Rejection Filter).
"""

from typing import Dict, Any, Optional, Tuple

CONFIDENCE_THRESHOLD = 60.0  # Ngưỡng tin cậy an toàn tối thiểu (%)
DANGEROUS_REMEDIES = [
    "kem đánh răng", "mỡ trăn", "lòng đỏ trứng", "nước mắm",
    "lá cây", "tro bếp", "vôi bột", "thuốc lá", "dầu hỏa", "nước đá"
]

def check_confidence_abstention(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """
    Kiểm tra mức độ tin cậy của mô hình theo nguyên tắc đạo đức y tế:
    Nếu confidence < threshold, AI từ chối khẳng định chẩn đoán.
    """
    is_abstaining = float(confidence) < float(threshold)
    return {
        "is_abstaining": is_abstaining,
        "confidence": round(float(confidence), 1),
        "threshold": threshold,
        "status": "ABSTAIN" if is_abstaining else "ACCEPT",
        "ethical_message": (
            f"⚠️ CHỐT CHẶN ĐẠO ĐỨC Y TẾ (AI ABSTENTION): Độ tin cậy dự đoán chỉ đạt {confidence:.1f}% "
            f"(dưới ngưỡng an toàn {threshold}%). Để bảo vệ người bệnh, AI TỪ CHỐI đưa ra phác đồ điều trị. "
            f"Bắt buộc người bệnh phải đến cơ sở y tế để bác sĩ chuyên khoa thăm khám trực tiếp!"
            if is_abstaining else "Độ tin cậy nằm trong ngưỡng an toàn cho phép tham khảo lâm sàng."
        )
    }

def filter_dangerous_remedies(query: str) -> Optional[str]:
    """
    Bộ lọc tiêu cực (Negative Rejection Filter): Chặn đứng các câu hỏi về mẹo dân gian nguy hại.
    """
    q_lower = query.lower()
    for remedy in DANGEROUS_REMEDIES:
        if remedy in q_lower:
            return (
                f"🚨 CẢNH BÁO NGUY HIỂM Y TẾ: Tuyệt đối **KHÔNG ĐƯỢC BÔI {remedy.upper()}** lên vết bỏng!\n"
                f"- **Cơ chế gây hại:** Việc bôi {remedy} làm bít tắc bề mặt thoát nhiệt, giữ nhiệt độ cao gây bỏng sâu hơn, "
                f"đồng thời đưa hàng triệu vi khuẩn vào mô da hở gây hoại tử, nhiễm trùng huyết và sốc nhiễm khuẩn nguy kịch.\n"
                f"- **Xử trí đúng:** Chỉ ngâm rửa bằng nước sạch mát (16°C - 20°C) trong 15-20 phút và băng gạc vô trùng."
            )
    return None

def validate_clinical_safety(degree: str, tbsa: float, confidence: float) -> Dict[str, Any]:
    """
    Đánh giá an toàn lâm sàng tổng hợp cho toàn bộ phiên khám.
    """
    abstain_info = check_confidence_abstention(confidence)
    is_severe = ("3" in degree) or (tbsa >= 10.0)
    
    allow_home_care = (not abstain_info["is_abstaining"]) and (not is_severe)
    
    return {
        "abstain": abstain_info["is_abstaining"],
        "confidence": confidence,
        "is_severe": is_severe,
        "allow_home_care": allow_home_care,
        "abstain_message": abstain_info["ethical_message"],
        "recommendation": (
            "Chuyển khám chuyên khoa Bỏng khẩn cấp." if is_severe else
            ("Tạm hoãn tư vấn điều trị - Cần thăm khám trực tiếp." if abstain_info["is_abstaining"] else
             "Có thể theo dõi và chăm sóc tại chỗ theo phác đồ Bộ Y tế.")
        )
    }
