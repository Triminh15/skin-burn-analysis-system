"""
Module: src/rag/simple_advisor.py
Mục đích: Module Trợ lý Bác sĩ AI Y khoa (Clinical RAG Advisor) cung cấp câu trả lời chuẩn xác
dựa trên Phác đồ Chẩn đoán và Điều trị Bỏng của Bộ Y tế Việt Nam.
"""

from typing import Dict, Any, List, Optional
from src.rag.guardrails import filter_dangerous_remedies, check_confidence_abstention

# Tri thức y khoa tích hợp (Medical Knowledge Base - Cấp cứu Bỏng BHYT & Bộ Y Tế)
MEDICAL_KB = {
    "choc_bong_nuoc": (
        "**Khuyến nghị y khoa:** Tuyệt đối **KHÔNG ĐƯỢC TỰ Ý CHỌC VỠ** bọng nước bỏng (phỏng nước).\n"
        "- **Lý do:** Lớp da bọng nước đóng vai trò như một màng sinh học vô trùng tự nhiên bảo vệ lớp trung bì nhạy cảm bên dưới.\n"
        "- **Hậu quả:** Việc chọc vỡ bọng nước làm mất hàng rào bảo vệ, dẫn tới nguy cơ nhiễm trùng mô sâu, để lại sẹo lồi hoặc sẹo co kéo nghiêm trọng.\n"
        "- **Xử trí đúng:** Chỉ băng nhẹ bằng gạc vô trùng thấm nước muối sinh lý 0.9% và để bác sĩ xử trí tại cơ sở y tế."
    ),
    "so_cuu_24h": (
        "**Quy trình sơ cứu 4 bước chuẩn Bộ Y tế trong 24 giờ đầu:**\n"
        "1. **Hạ nhiệt khẩn cấp:** Ngâm rửa vùng bỏng dưới vòi nước sạch mát (16°C - 20°C) trong 15 - 20 phút. Tuyệt đối không dùng nước đá lạnh.\n"
        "2. **Cởi bỏ trang phục:** Nhẹ nhàng cởi bỏ quần áo hoặc vòng nhẫn gần vết thương trước khi phù nề. Nếu quần áo dính vào vết bỏng, dùng kéo cắt xung quanh, không giật mạnh.\n"
        "3. **Băng vô trùng:** Che phủ bằng gạc sạch vô khuẩn hoặc vải sạch, không băng quá chặt.\n"
        "4. **Giữ ấm và bù nước:** Cho nạn nhân uống nhiều nước oresol hoặc nước ấm, nhanh chóng chuyển đến bệnh viện nếu bỏng diện rộng."
    ),
    "thuoc_boi": (
        "**Lưu ý về thuốc bôi:**\n"
        "- Trong giai đoạn sơ cứu khẩn cấp: **KHÔNG BÔI** kem đánh răng, mỡ trăn, lòng đỏ trứng, nước mắm hay các loại lá cây dân gian.\n"
        "- Sau khi làm sạch, bác sĩ có thể chỉ định kem kháng khuẩn tại chỗ như **Silver Sulfadiazine 1% (Silvadene)** hoặc kem mỡ Panthenol làm dịu mô da."
    ),
    "dinh_duong": (
        "**Chế độ dinh dưỡng cho bệnh nhân bỏng:**\n"
        "- **Tăng cường Protein (Đạm):** Thịt nạc, cá, trứng, sữa giúp tái tạo mô liên kết và biểu mô hóa nhanh chóng.\n"
        "- **Vitamin C & Kẽm:** Trái cây có múi (cam, bưởi), ớt chuông giúp tổng hợp collagen và tăng cường miễn dịch chống nhiễm khuẩn.\n"
        "- **Kiêng khem:** Hạn chế đồ ăn quá cay nóng, chất kích thích (rượu bia, thuốc lá). Tránh ăn thức ăn gây dị ứng cá nhân."
    ),
    "parkland_info": (
        "**Nguyên lý công thức Parkland:**\n"
        "- Tổng dịch truyền trong 24h = **4 ml × Cân nặng (kg) × % TBSA** (bằng dung dịch Ringer Lactate).\n"
        "- **50% lượng dịch** phải được truyền hết trong **8 giờ đầu** tính từ thời điểm bị bỏng.\n"
        "- **50% lượng dịch còn lại** được chia đều truyền trong **16 giờ tiếp theo**.\n"
        "- Mục tiêu theo dõi: Giữ thể tích nước tiểu đạt 0.5 - 1.0 ml/kg/giờ ở người lớn."
    )
}

def ask_medical_bot(query: str, clinical_context: Dict[str, Any] = None) -> str:
    # 1. Chốt chặn bài thuốc dân gian độc hại
    danger_warning = filter_dangerous_remedies(query)
    if danger_warning:
        return danger_warning

    # 2. Chốt chặn Đạo đức y tế khi Độ tin cậy thấp (Abstention)
    if clinical_context:
        conf = clinical_context.get("confidence", 100.0)
        abstain = check_confidence_abstention(conf)
        if abstain["is_abstaining"]:
            return (
                f"⚠️ **TỪ CHỐI TƯ VẤN DO ĐỘ TIN CẬY THẤP ({conf:.1f}%):**\n"
                f"- Theo nguyên tắc đạo đức y sinh 'Không gây hại', khi hình ảnh mờ hoặc độ tự tin của AI dưới 60%, "
                f"hệ thống **không đưa ra chỉ định dùng thuốc hoặc phác đồ tự điều trị tại nhà** để tránh nguy cơ biến chứng.\n"
                f"- **Khuyến cáo duy nhất:** Hãy giữ sạch vết thương và đến ngay cơ sở y tế gần nhất để bác sĩ thăm khám lâm sàng trực tiếp."
            )

    q = query.lower()
    
    if any(k in q for k in ["bọng nước", "bóng nước", "phỏng nước", "chọc", "vỡ", "nặn"]):
        return MEDICAL_KB["choc_bong_nuoc"]
    
    if any(k in q for k in ["sơ cứu", "làm gì", "mới bị", "ban đầu", "24 giờ", "24h"]):
        return MEDICAL_KB["so_cuu_24h"]

    if any(k in q for k in ["bôi", "thuốc", "mỡ trăn", "kem"]):
        return MEDICAL_KB["thuoc_boi"]

    if any(k in q for k in ["ăn", "uống", "dinh dưỡng", "kiêng"]):
        return MEDICAL_KB["dinh_duong"]

    if any(k in q for k in ["truyền", "parkland", "dịch", "ringer"]):
        return MEDICAL_KB["parkland_info"]

    # Phản hồi mặc định kết hợp thông tin lâm sàng hiện tại nếu có
    res = "Bác sĩ AI BurnScan ghi nhận câu hỏi của bạn. "
    if clinical_context:
        deg = clinical_context.get("predicted_degree", "")
        tbsa = clinical_context.get("estimated_tbsa_pct", 0)
        res += f"Đối với vết thương hiện tại được phân loại **{deg}** (TBSA: **{tbsa}%**): "
    res += (
        "Khuyến cáo hàng đầu là giữ vết bỏng khô sạch, băng gạc vô trùng và theo dõi các dấu hiệu nhiễm trùng (sốt, chảy mủ, đau tăng dần). "
        "Nếu diện tích bỏng lớn hơn bàn tay hoặc có phỏng nước vỡ, hãy đến cơ sở y tế gần nhất để được thăm khám trực tiếp."
    )
    return res

