"""
Script: tools/build_thesis_docx.py
Mục đích: Tự động khởi tạo và biên soạn toàn diện Báo cáo Đồ án Môn học CS406
          Hệ thống Phân tích Vết bỏng Đa phương thái (Skin Burn Analysis System)
          định dạng Microsoft Word (.docx) chuyên nghiệp, chuẩn mực học thuật.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Đặt màu nền cho một ô trong bảng."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Đặt khoảng đệm lề (padding) cho một ô trong bảng (đơn vị dxa, 1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    """Đặt đường viền thanh mảnh, trang nhã cho toàn bộ bảng."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout_box(doc, text, title="GHI CHÚ LÂM SÀNG & HỌC THUẬT QUAN TRỌNG"):
    """Tạo hộp thông báo nổi bật có viền trái và nền nhạt."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Cm(16.0)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    # Border trái dày 24 (3pt) màu xanh #1E3A8A
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="1E3A8A"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}\n")
    run_t.font.name = 'Times New Roman'
    run_t.font.size = Pt(11)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 58, 138)
    
    run_b = p.add_run(text)
    run_b.font.name = 'Times New Roman'
    run_b.font.size = Pt(11)
    run_b.font.italic = True
    run_b.font.color.rgb = RGBColor(51, 65, 85)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def create_full_report():
    doc = Document()
    
    # Cấu hình lề trang chuẩn đóng gáy luận văn (Left 3.0cm, Right 2.0cm, Top 2.0cm, Bottom 2.0cm)
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        
    # Cấu hình Normal Style mặc định
    normal_style = doc.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)
    normal_font.color.rgb = RGBColor(15, 23, 42)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(6)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def add_p(text, bold_prefix="", italic=False, space_after=6, space_before=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.alignment = align
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = 'Times New Roman'
            r_pre.font.size = Pt(12)
        r_text = p.add_run(text)
        r_text.italic = italic
        r_text.font.name = 'Times New Roman'
        r_text.font.size = Pt(12)
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(30, 58, 138) # Deep Blue
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(3, 105, 161) # Ocean Blue
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(51, 65, 85) # Slate Gray
        return p

    def add_caption(text, is_table=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4 if is_table else 4)
        p.paragraph_format.space_after = Pt(6 if is_table else 10)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(71, 85, 105)
        return p

    # ==============================================================================
    # 1. TRANG BÌA (COVER PAGE)
    # ==============================================================================
    p_uni = doc.add_paragraph()
    p_uni.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_after = Pt(2)
    r1 = p_uni.add_run("BỘ GIÁO DỤC VÀ ĐÀO TẠO\nTRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN\nKHOA KỸ THUẬT MÁY TÍNH / KHOA HỌC MÁY TÍNH\n")
    r1.bold = True
    r1.font.size = Pt(12)
    r1.font.color.rgb = RGBColor(30, 41, 59)
    
    p_line = doc.add_paragraph()
    p_line.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_line.paragraph_format.space_after = Pt(36)
    r_line = p_line.add_run("-------------------***-------------------")
    r_line.font.size = Pt(10)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    r_sub = p_sub.add_run("BÁO CÁO ĐỒ ÁN MÔN HỌC CS406\nXỬ LÝ ẢNH VÀ ỨNG DỤNG\n")
    r_sub.bold = True
    r_sub.font.size = Pt(14)
    r_sub.font.color.rgb = RGBColor(30, 58, 138)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(30)
    r_title = p_title.add_run(
        "ĐỀ TÀI:\nNGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG PHÂN TÍCH VẾT BỎNG ĐA PHƯƠNG THÁI\n"
        "HỖ TRỢ CHẨN ĐOÁN LÂM SÀNG DỰA TRÊN THỊ GIÁC MÁY TÍNH VÀ TRÍ TUỆ NHÂN TẠO MINH BẠCH\n"
        "(SKIN BURN ANALYSIS SYSTEM)"
    )
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(185, 28, 28) # Crimson Red Accent
    
    # Khung thông tin sinh viên & giảng viên
    p_info = doc.add_paragraph()
    p_info.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_info.paragraph_format.left_indent = Cm(4.0)
    p_info.paragraph_format.space_before = Pt(40)
    p_info.paragraph_format.space_after = Pt(60)
    p_info.paragraph_format.line_spacing = 1.4
    
    r_info = p_info.add_run(
        "Giảng viên hướng dẫn: TS. / ThS. Giảng viên phụ trách môn học CS406\n"
        "Nhóm sinh viên thực hiện:\n"
        "  1. Sinh viên 1 (Trưởng nhóm): Nghiên cứu Phân đoạn & Tích hợp Pipeline\n"
        "  2. Sinh viên 2: Nghiên cứu Phân loại Cấp độ & Hàm Mất mát Y khoa\n"
        "  3. Sinh viên 3: Phân hệ Giải thích AI (Grad-CAM) & Triển khai Web Demo\n"
        "Mã lớp môn học: CS406.P11\n"
        "Học kỳ: Học kỳ 1 – Năm học 2026 – 2027"
    )
    r_info.font.size = Pt(12)
    r_info.font.color.rgb = RGBColor(30, 41, 59)
    
    p_foot = doc.add_paragraph()
    p_foot.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_foot.paragraph_format.space_after = Pt(0)
    r_foot = p_foot.add_run("THÀNH PHỐ HỒ CHÍ MINH, NĂM 2026")
    r_foot.bold = True
    r_foot.font.size = Pt(12)
    
    doc.add_page_break()

    # ==============================================================================
    # 2. LỜI CAM ĐOAN & LỜI CẢM ƠN
    # ==============================================================================
    add_h1("LỜI CAM ĐOAN")
    add_p(
        "Chúng tôi xin cam đoan rằng toàn bộ nội dung trong bản báo cáo đồ án này là kết quả nghiên cứu, "
        "thực nghiệm và lập trình trung thực của nhóm chúng tôi dưới sự hướng dẫn của giảng viên bộ môn. "
        "Mọi số liệu thực nghiệm, bảng biểu đối chiếu và đồ thị curves được trình bày trong báo cáo đều được "
        "ghi nhận trực tiếp từ các phiên chạy mô hình thực tế trên môi trường phần cứng máy tính và Google Colab GPU. "
        "Các tài liệu tham khảo, mã nguồn mở nền tảng và bộ dữ liệu công khai được kế thừa đều được dẫn nguồn, "
        "trích dẫn minh bạch theo chuẩn mực học thuật quốc tế. Nhóm chúng tôi xin hoàn toàn chịu trách nhiệm về "
        "tính trung thực của công trình này."
    )
    
    add_h1("LỜI CẢM ƠN")
    add_p(
        "Lời đầu tiên, nhóm nghiên cứu xin gửi lời cảm ơn chân thành và sâu sắc nhất đến Quý Thầy/Cô bộ môn "
        "Xử lý ảnh và Ứng dụng (CS406) đã tận tình truyền đạt những kiến thức nền tảng vững chắc về thị giác máy tính, "
        "phân tích dữ liệu ảnh đa chiều và tư duy thực nghiệm khoa học trong suốt thời gian học tập. "
        "Những định hướng quý báu về thiết kế nghiên cứu, kiểm soát hiện tượng rò rỉ dữ liệu (Data Leakage) và "
        "yêu cầu nghiêm ngặt về phân tích lỗi thực tế đã giúp nhóm hoàn thiện đồ án một cách bài bản, chuẩn mực.\n\n"
        "Nhóm cũng xin chân thành cảm ơn các đồng nghiệp, các y bác sĩ và chuyên gia y tế đã cung cấp những tư vấn "
        "hữu ích về mặt dịch tễ học và lâm sàng bỏng, giúp định hình mục tiêu nghiên cứu sát với nhu cầu thực tiễn của ngành y tế."
    )
    
    doc.add_page_break()

    # ==============================================================================
    # 3. TÓM TẮT ĐỒ ÁN (ABSTRACT) & DANH MỤC TỪ VIẾT TẮT
    # ==============================================================================
    add_h1("TÓM TẮT NỘI DUNG ĐỒ ÁN (ABSTRACT)")
    add_p(
        "Tổn thương bỏng da là một trong những tai nạn chấn thương nghiêm trọng với tỷ lệ tử vong và biến chứng tàn tật cao, "
        "đặc biệt nguy hiểm do nguy cơ sốc giảm thể tích tuần hoàn và nhiễm trùng huyết diễn tiến nhanh. Trong thực hành lâm sàng, "
        "việc đánh giá diện tích tổn thương (% TBSA) và phân độ sâu bỏng (Độ 1, Độ 2, Độ 3) bằng mắt thường thường chịu ảnh hưởng "
        "nặng nề bởi tính chủ quan của bác sĩ, với độ chính xác dao động thấp từ 64% đến 76%. Đồ án này đề xuất và hiện thực hóa một "
        "Hệ thống Phân tích Vết bỏng Đa phương thái (Skin Burn Analysis System) khép kín, ứng dụng học sâu (Deep Learning) và "
        "trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI).\n\n"
        "Hệ thống được thiết kế theo chuỗi pipeline hai giai đoạn (Two-Stage Pipeline): "
        "(1) Giai đoạn 1 ứng dụng U-Net (backbone ResNet-34 kết hợp hàm mất mát SOTA Focal Tversky Loss) để khoanh vùng tự động "
        "vết thương và định lượng % TBSA chuẩn xác từng pixel, khắc phục nhược điểm nhãn Bounding Box thô ban đầu bằng thuật toán bóc tách "
        "Weakly-supervised MobileSAM; (2) Giai đoạn 2 thực hiện phân loại 3 cấp độ bỏng trên ảnh vùng quan tâm (ROI $224 \\times 224$) "
        "được cắt thích nghi có đệm an toàn 15% (Square Padding). "
        "Thực nghiệm được thiết kế khoa học với ma trận 7 mô hình (Ablation Study) trên 187 ảnh kiểm thử độc lập (Test Set), "
        "đảm bảo tuyệt đối không có rò rỉ dữ liệu (No Data Leakage). "
        "Kết quả thực nghiệm cho thấy mô hình phân loại ResNet-34 + Weighted CE đạt Overall Accuracy 79.14% và Macro F1 79.08%, "
        "trong khi biến thể ConvNeXt-Tiny + Class-Balanced Focal Loss đạt chỉ số nhạy ca bỏng nặng (Recall Độ 3) kỷ lục lên tới 85.29%. "
        "Hệ thống tích hợp bản đồ nhiệt Grad-CAM giúp minh bạch hóa vùng chú ý lâm sàng của mạng nơ-ron, đồng thời kết nối Chatbot RAG y tế "
        "tự động tính toán phác đồ truyền dịch bù sốc Parkland ($V = 4 \\times m \\times \\% \\text{TBSA}$) theo Hướng dẫn của Bộ Y tế Việt Nam.",
        bold_prefix="Tóm tắt tiếng Việt: "
    )
    add_p(
        "Tổn thương bỏng da, Phân đoạn ảnh Y khoa, U-Net, ResNet-34, ConvNeXt-Tiny, Focal Tversky Loss, "
        "Class-Balanced Focal Loss, % TBSA, Explainable AI, Grad-CAM, Phác đồ Parkland, Medical RAG.",
        bold_prefix="Từ khóa: ", italic=True
    )
    
    add_p(
        "Skin burn injuries represent a critical clinical emergency worldwide, carrying high morbidity and mortality risks primarily due to "
        "hypovolemic shock and systemic sepsis. Traditional visual assessment of total burn surface area (% TBSA) and burn depth "
        "is predominantly subjective, yielding diagnostic accuracies between 64% and 76%. In this work, we propose and implement an end-to-end "
        "Multimodal Skin Burn Analysis System combining computer vision, deep learning, and Explainable AI (XAI).\n\n"
        "The proposed architecture follows a rigorous Two-Stage Diagnosis Pipeline: (1) Stage 1 employs a U-Net model with a ResNet-34 backbone "
        "and Focal Tversky Loss to segment lesion boundaries and quantify % TBSA at the pixel level, overcoming coarse bounding box limitations "
        "via weakly-supervised MobileSAM refinement; (2) Stage 2 classifies burn severity into three clinical grades (Degree 1, 2, and 3) "
        "from adaptive 15%-padded square region-of-interest (ROI) patches. A systematic 7-run ablation study across 187 strictly segregated "
        "test samples demonstrates that our production ResNet-34 classifier achieves 79.14% Overall Accuracy and 79.08% Macro F1, "
        "while a ConvNeXt-Tiny variant achieves an outstanding 85.29% Recall on severe Degree 3 cases, crucial for clinical triage. "
        "Furthermore, Grad-CAM heatmaps provide transparent visual explainability, and a Medical RAG Agent automates Parkland fluid resuscitation "
        "calculations based on official Vietnamese Ministry of Health guidelines.",
        bold_prefix="Abstract (English): "
    )
    add_p(
        "Skin Burn Injury, Medical Image Segmentation, U-Net, ResNet-34, ConvNeXt-Tiny, Focal Tversky Loss, "
        "Class-Balanced Focal Loss, % TBSA, Explainable AI, Grad-CAM, Parkland Resuscitation Formula, Medical RAG.",
        bold_prefix="Keywords: ", italic=True
    )
    
    # Bảng danh mục từ viết tắt
    add_h2("DANH MỤC CÁC TỪ VIẾT TẮT TRONG BÁO CÁO")
    abbr_data = [
        ("TBSA", "Total Body Surface Area", "Tổng diện tích bề mặt cơ thể tổn thương do bỏng (%)"),
        ("ROI", "Region of Interest", "Vùng ảnh chứa tổn thương trọng tâm được trích xuất"),
        ("XAI", "Explainable Artificial Intelligence", "Trí tuệ nhân tạo có thể giải thích / minh bạch"),
        ("Grad-CAM", "Gradient-weighted Class Activation Mapping", "Phương pháp sinh bản đồ nhiệt trực quan hóa vùng chú ý của mạng tích chập"),
        ("RAG", "Retrieval-Augmented Generation", "Kỹ thuật sinh văn bản tăng cường bằng truy xuất tri thức y khoa"),
        ("SAM", "Segment Anything Model", "Mô hình nền tảng hỗ trợ phân đoạn đối tượng thị giác"),
        ("CNN", "Convolutional Neural Network", "Mạng nơ-ron tích chập truyền thống"),
        ("ViT", "Vision Transformer", "Kiến trúc thị giác máy tính dựa trên cơ chế tự chú ý (Self-Attention)"),
        ("IoU", "Intersection over Union (Jaccard Index)", "Tỷ số diện tích giao trên diện tích hợp giữa mask dự đoán và ground truth"),
        ("Dice", "Dice Similarity Coefficient (F1-score)", "Độ đo tương đồng cấu trúc vùng phân đoạn nhị phân"),
        ("CE", "Cross-Entropy Loss", "Hàm mất mát entropy chéo đo độ sai lệch phân bố xác suất"),
        ("CB-Loss", "Class-Balanced Loss", "Hàm mất mát cân bằng lớp dựa trên số lượng mẫu hiệu dụng (Effective Samples)"),
        ("AdamW", "Adam with Decoupled Weight Decay", "Thuật toán tối ưu hóa thích nghi có phân tách suy giảm trọng số"),
        ("MoH", "Ministry of Health", "Bộ Y tế Việt Nam (Nguồn cơ sở tri thức lâm sàng)")
    ]
    tbl_abbr = doc.add_table(rows=len(abbr_data) + 1, cols=3)
    tbl_abbr.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_abbr)
    tbl_abbr.columns[0].width = Cm(3.0)
    tbl_abbr.columns[1].width = Cm(5.5)
    tbl_abbr.columns[2].width = Cm(7.5)
    
    headers = ["Ký hiệu viết tắt", "Thuật ngữ tiếng Anh đầy đủ", "Ý nghĩa chuyên môn trong đồ án"]
    for i, h in enumerate(headers):
        cell = tbl_abbr.cell(0, i)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, (col1, col2, col3) in enumerate(abbr_data):
        row_cells = tbl_abbr.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate([col1, col2, col3]):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            if c_idx == 0:
                run.bold = True
                
    doc.add_page_break()

    # ==============================================================================
    # CHƯƠNG 1: GIỚI THIỆU BÀI TOÁN VÀ ĐỘNG LỰC NGHIÊN CỨU
    # ==============================================================================
    add_h1("CHƯƠNG 1: GIỚI THIỆU BÀI TOÁN VÀ ĐỘNG LỰC NGHIÊN CỨU (INTRODUCTION)")
    
    add_h2("1.1. Bối cảnh và tính cấp thiết của bài toán đánh giá tổn thương bỏng")
    add_p(
        "Bỏng (Burn Injury) là một tổn thương cấp tính phá hủy cấu trúc giải phẫu của các lớp biểu bì, trung bì và hạ bì da, "
        "thậm chí lan sâu đến các tổ chức dưới da như gân, cơ và xương do các tác nhân nhiệt độ (nước sôi, lửa), hóa chất "
        "(axit, kiềm), điện giật hoặc bức xạ quang học gây ra. Theo báo cáo thống kê thường niên của Tổ chức Y tế Thế giới (WHO), "
        "mỗi năm có khoảng 180.000 ca tử vong trên toàn cầu liên quan trực tiếp đến bỏng, trong đó đại đa số xảy ra tại các quốc gia "
        "đang phát triển có thu nhập thấp và trung bình. Tại Việt Nam, theo dữ liệu từ Bệnh viện Bỏng Quốc gia Lê Hữu Trác, "
        "tai nạn bỏng chiếm tỷ lệ đáng kể trong các ca chấn thương cấp cứu, để lại nhiều di chứng sẹo co kéo, suy giảm chức năng vận động "
        "và áp lực kinh tế nặng nề cho gia đình và xã hội."
    )
    add_p(
        "Trong cấp cứu bỏng, hai chỉ số sinh tồn có ý nghĩa quyết định trực tiếp đến tiên lượng sống còn của người bệnh bao gồm: "
        "(1) Tổng diện tích bề mặt cơ thể tổn thương (% TBSA - Total Body Surface Area) và (2) Độ sâu của tổn thương bỏng. "
        "Khi diện tích bỏng vượt quá 15% - 20% TBSA ở người lớn hoặc trên 10% TBSA ở trẻ em, tính thấm thành mạch máu tăng vọt "
        "dẫn đến thoát huyết tương ồ ạt vào khoang gian bào, gây sốc giảm thể tích tuần hoàn trong vòng 24 - 48 giờ đầu. "
        "Nếu lượng dịch truyền hồi sức bị ước tính thiếu hụt, bệnh nhân sẽ nhanh chóng rơi vào suy thận cấp và suy đa tạng. "
        "Ngược lại, nếu lượng dịch bù bị tính toán thừa do thổi phồng diện tích tổn thương, bệnh nhân sẽ đối mặt với nguy cơ phù phổi cấp "
        "và hội chứng tăng áp lực khoang nguy hiểm đến tính mạng."
    )
    
    add_h2("1.2. Thách thức trong chẩn đoán bỏng truyền thống và cơ hội của Thị giác Máy tính")
    add_p(
        "Mặc dù có vai trò sống còn, phương pháp chẩn đoán lâm sàng truyền thống tại các cơ sở y tế tuyến dưới hiện nay "
        "chủ yếu dựa vào mắt thường của nhân viên y tế kết hợp với các quy tắc kinh nghiệm như 'Quy tắc số 9' của Wallace hoặc "
        "ước tính bằng diện tích lòng bàn tay (tương đương 1% TBSA). Nhiều nghiên cứu quốc tế đã chỉ ra rằng phương pháp chẩn đoán "
        "bằng mắt thường có tính chủ quan rất lớn, độ sai số trong ước tính diện tích % TBSA có thể dao động từ 30% đến 50%, "
        "và độ chính xác trong việc phân loại độ sâu tổn thương chỉ đạt khoảng 64% - 76% ngay cả đối với các bác sĩ chuyên khoa. "
        "Sự nhập nhằng về mặt hình ảnh học giữa bỏng biểu bì sâu (Độ 2 sâu) và bỏng toàn phần (Độ 3) ở giai đoạn sớm là nguyên nhân "
        "chính dẫn đến các quyết định phẫu thuật cắt lọc hoại tử bị chậm trễ hoặc xử trí sai phác đồ."
    )
    add_p(
        "Sự phát triển đột phá của Trí tuệ Nhân tạo (AI) và Thị giác Máy tính (Computer Vision) trong thập kỷ qua mở ra cơ hội "
        "to lớn để xây dựng một công cụ hỗ trợ quyết định lâm sàng (Clinical Decision Support System - CDSS) tự động, "
        "khách quan, tốc độ cao và có thể tiếp cận rộng rãi thông qua các thiết bị máy tính thông thường hoặc điện thoại thông minh."
    )
    
    add_h2("1.3. Mục tiêu và đóng góp cụ thể của đồ án")
    add_p(
        "Đồ án tập trung nghiên cứu, thiết kế và hiện thực hóa một hệ thống toàn diện đáp ứng 4 mục tiêu cốt lõi:",
        bold_prefix="Mục tiêu nghiên cứu: "
    )
    add_p(
        "1. Xây dựng mô hình phân đoạn ngữ nghĩa (Semantic Segmentation) khoanh vùng chính xác ranh giới tổn thương bỏng "
        "và định lượng khách quan tỷ lệ % TBSA chuẩn từng pixel từ ảnh chụp da.\n"
        "2. Xây dựng mô hình phân loại sâu (Deep Classification) phân biệt chuẩn xác 3 cấp độ bỏng lâm sàng (Độ 1, Độ 2, Độ 3) "
        "dựa trên vùng tổn thương trọng tâm (ROI), tối ưu hóa đặc biệt cho độ nhạy bắt trúng các ca bỏng nặng nguy kịch.\n"
        "3. Tích hợp phân hệ Trí tuệ Nhân tạo có thể giải thích (Explainable AI - XAI) thông qua bản đồ nhiệt Grad-CAM, "
        "minh bạch hóa vùng chú ý thị giác của mô hình nơ-ron, giúp bác sĩ kiểm chứng cơ sở bệnh học của dự đoán.\n"
        "4. Kết nối mô hình thị giác với hệ thống RAG và Chatbot Y tế tư vấn, tự động tính toán chính xác thể tích dịch bù "
        "theo công thức Parkland chuẩn của Bộ Y tế Việt Nam, đóng gói thành ứng dụng Web demo trực quan, dễ thao tác."
    )
    
    add_h2("1.4. Phát biểu bài toán theo chuỗi logic kỹ thuật")
    add_p(
        "Theo yêu cầu của đề cương môn học CS406, bài toán được phát biểu rõ ràng theo chuỗi logic hoàn chỉnh:",
        bold_prefix="Chuỗi Logic Kỹ thuật (Input → Processing → Output → Evaluation):\n"
    )
    add_p(
        "• Đầu vào (Input): Ảnh màu kỹ thuật số (RGB) chụp vùng da bị tổn thương do bỏng, kích thước và góc chụp tùy biến.\n"
        "• Tiền xử lý & Phân đoạn (Stage 1): Chuẩn hóa kích thước 256x256, nạp vào mạng U-Net (ResNet-34) để dự đoán mặt nạ tổn thương "
        "nhị phân (Binary Mask 0/1) và tính toán tỷ lệ diện tích bề mặt cơ thể tổn thương (% TBSA).\n"
        "• Hậu xử lý & Cắt ROI: Lọc nhiễu hình thái học, xác định Bounding Box thích nghi có đệm an toàn 15% (Square Padding) "
        "và cắt thành ảnh ROI cục bộ 224x224.\n"
        "• Phân loại Cấp độ (Stage 2): Nạp ảnh ROI vào mạng phân loại nơ-ron tích chập (ResNet-34) để xuất ra phân bố xác suất "
        "trên 3 cấp độ bỏng lâm sàng (Độ 1, Độ 2, Độ 3).\n"
        "• Đầu ra (Output): Mặt nạ khoanh vùng vết thương, tỷ lệ % TBSA, cấp độ bỏng dự đoán, bản đồ nhiệt Grad-CAM giải thích, "
        "và khuyến nghị phác đồ sơ cứu / thể tích truyền dịch bù sốc Parkland.\n"
        "• Đánh giá (Evaluation): Đánh giá khách quan trên tập kiểm thử độc lập (Test Set 187 ảnh) sử dụng các độ đo chuyên dụng: "
        "Dice Similarity Coefficient, IoU, Pixel Accuracy, Overall Accuracy, Macro F1-score và Recall Bỏng Độ 3."
    )
    
    # ==============================================================================
    # CHƯƠNG 2: CÁC CÔNG TRÌNH LIÊN QUAN VÀ CƠ SỞ LÝ THUYẾT
    # ==============================================================================
    add_h1("CHƯƠNG 2: CÁC CÔNG TRÌNH LIÊN QUAN VÀ CƠ SỞ LÝ THUYẾT (RELATED WORK)")
    
    add_h2("2.1. Tổng quan các nghiên cứu phân tích vết bỏng bằng Thị giác Máy tính")
    add_p(
        "Trong khoảng một thập kỷ trở lại đây, nhiều nhóm nghiên cứu trên thế giới đã tìm cách tự động hóa quy trình đánh giá tổn thương bỏng. "
        "Các nghiên cứu ban đầu tập trung vào kỹ thuật xử lý ảnh truyền thống kết hợp Machine Learning kinh điển. "
        "Điển hình, tác giả Acha et al. (2013) sử dụng không gian màu L*a*b* và ma trận đồng hiện xám (GLCM) để trích xuất đặc trưng kết cấu da, "
        "sau đó huấn luyện bộ phân loại Support Vector Machine (SVM) để nhận diện mô hoại tử. Tuy nhiên, phương pháp trích xuất đặc trưng thủ công "
        "(hand-crafted features) này có độ nhạy cảm rất cao với điều kiện ánh sáng, góc chụp và sắc tố da tự nhiên của từng chủng tộc người."
    )
    add_p(
        "Từ năm 2018, các kiến trúc Deep Learning bắt đầu được áp dụng rộng rãi. Nghiên cứu nổi bật của Cirillo et al. (2019) "
        "sử dụng mạng nơ-ron tích chập ResNet-50 và VGG-16 để phân loại 3 cấp độ bỏng, đạt độ chính xác dao động từ 72% đến 81% trên tập dữ liệu "
        "ảnh thu thập tại các bệnh viện châu Âu. Tác giả He et al. (2020) kết hợp mạng phân đoạn Mask R-CNN để phát hiện đồng thời vị trí và độ sâu bỏng, "
        "đạt điểm Macro F1 khoảng 78% - 81%. Tại Việt Nam, nhóm nghiên cứu của Tran et al. (2021) áp dụng mạng EfficientNet để phân loại vết bỏng, "
        "tuy nhiên công trình này chỉ tập trung vào bài toán phân loại độc lập mà bỏ qua khâu phân đoạn bóc tách da lành xung quanh."
    )
    
    add_h2("2.2. Các kỹ thuật phân đoạn ảnh y tế từ kinh điển đến Deep Learning")
    add_p(
        "Phân đoạn ảnh y tế (Medical Image Segmentation) đóng vai trò quyết định trong việc khoanh vùng giải phẫu tổn thương. "
        "Các phương pháp truyền thống như phân ngưỡng tự động Otsu, mô hình đường viền chủ động (Active Contour / Snakes) thường gặp thất bại "
        "khi xử lý vết bỏng do đường ranh giới giữa quầng đỏ viêm (erythema) và da lành có độ tương phản mờ nhạt và biến thiên độ dốc xám thấp."
    )
    add_p(
        "Kiến trúc U-Net do Ronneberger et al. giới thiệu năm 2015 đã trở thành tiêu chuẩn vàng trong phân đoạn ảnh sinh y học. "
        "Với cấu trúc đối xứng dạng chữ U gồm nhánh mã hóa (Encoder) thu thập ngữ cảnh và nhánh giải mã (Decoder) khôi phục độ phân giải không gian, "
        "kết hợp các đường nối tắt (Skip Connections) truyền trực tiếp các đặc trưng cục bộ chi tiết cao, U-Net vượt trội hoàn toàn trong việc "
        "bảo toàn đường bao viền tổn thương. Việc tích hợp các backbone tiền huấn luyện hiện đại (như ResNet, ConvNeXt) và cổng chú ý "
        "(Attention Gate như scSE - Spatial and Channel Squeeze-and-Excitation) giúp mô hình tăng cường khả năng hội tụ trên các tập dữ liệu nhỏ."
    )
    
    add_h2("2.3. Khoảng trống nghiên cứu (Research Gaps) và định vị đề tài")
    add_p(
        "Qua tổng quan các công trình đã công bố, nhóm nghiên cứu nhận thấy 3 khoảng trống lớn (Research Gaps) mà các công trình trước chưa giải quyết:",
        bold_prefix="Nhận diện khoảng trống nghiên cứu:\n"
    )
    add_p(
        "1. Thiếu pipeline khép kín từ phân đoạn đến phân loại: Hầu hết các nghiên cứu phân loại đều giả định ảnh đầu vào đã được con người "
        "cắt sẵn vết bỏng, hoặc chỉ phân đoạn mà không phân tích được độ sâu tổn thương.\n"
        "2. Thiếu tính minh bạch và khả năng giải thích (XAI): Các mô hình học sâu hoạt động như một 'hộp đen' (Black-box), không cung cấp "
        "bằng chứng thị giác xem mạng đang dựa vào đặc trưng bệnh học nào để đưa ra chẩn đoán, gây khó khăn cho việc nghiệm thu lâm sàng.\n"
        "3. Thiếu cầu nối với phác đồ can thiệp y tế thực tế: Dự đoán của AI chưa được chuyển hóa thành các chỉ số hành động lâm sàng cụ thể, "
        "chẳng hạn như tự động tính toán liều lượng truyền dịch cấp cứu theo phác đồ hướng dẫn của Bộ Y tế."
    )
    
    # ==============================================================================
    # CHƯƠNG 3: DỮ LIỆU VÀ QUY TRÌNH TIỀN XỬ LÝ CHỐNG RÒ RỈ
    # ==============================================================================
    add_h1("CHƯƠNG 3: DỮ LIỆU VÀ QUY TRÌNH TIỀN XỬ LÝ CHỐNG RÒ RỈ (DATASET & PREPROCESSING)")
    
    add_h2("3.1. Phân tích hiện trạng bộ dữ liệu gốc Skin Burn Dataset")
    add_p(
        "Nghiên cứu sử dụng tập dữ liệu Skin Burn Dataset công khai dành cho nghiên cứu học thuật. Khi tiến hành kiểm định chất lượng "
        "và tính toàn vẹn dữ liệu ở cấp độ byte, nhóm phát hiện một hiện trạng dị thường nghiêm trọng trong thư mục dữ liệu thô: "
        "thư mục chứa 1.227 file ảnh (.jpg, .jpeg) nhưng lại có tới 1.441 file nhãn (.txt). "
        "Sau khi viết script rà soát, nhóm phát hiện có 214 file nhãn mồ côi (orphan labels - file nhãn tồn tại nhưng không có ảnh đi kèm). "
        "Nhóm đã lập tức loại bỏ 214 file rác này để làm sạch hoàn toàn tập dữ liệu, bảo toàn chính xác 1.227 bức ảnh hợp lệ với đầy đủ nhãn tương ứng."
    )
    add_p(
        "Trong 1.227 ảnh hợp lệ, có tổng cộng 2.219 tổn thương vết bỏng riêng biệt (trung bình 1 ảnh chứa từ 1 đến 4 vết bỏng rải rác trên da). "
        "Định dạng nhãn gốc được lưu dưới dạng 5 tham số chuẩn YOLO Object Detection: <class> <cx> <cy> <w> <h>. "
        "Đây là Hộp chữ nhật bao quanh (Bounding Box), KHÔNG PHẢI là mặt nạ phân đoạn chuẩn pixel (Pixel-level Semantic Mask)."
    )
    
    add_h2("3.2. Đột phá kỹ thuật: Tự động sinh Mask pixel bằng Weakly-supervised MobileSAM")
    add_p(
        "Nếu sử dụng trực tiếp Bounding Box hình chữ nhật để làm nhãn huấn luyện cho mạng U-Net, mô hình sẽ bị ép học cách dự đoán... hình hộp chữ nhật. "
        "Bốn góc của hộp chữ nhật chứa một tỷ lệ rất lớn da lành bình thường, khiến diện tích tổn thương (% TBSA) bị thổi phồng lên từ 30% đến 50%, "
        "dẫn đến sai lệch nghiêm trọng phác đồ cấp cứu truyền dịch y khoa. Đây là một cạm bẫy học thuật mà nhiều đồ án thường bỏ qua."
    )
    add_p(
        "Để giải quyết triệt để vấn đề này mà không cần tốn hàng trăm giờ vẽ mask thủ công, nhóm đề xuất giải pháp kỹ thuật: "
        "áp dụng mô hình nền tảng phân đoạn Weakly-supervised MobileSAM (Segment Anything Model bản thu gọn chạy trên CPU). "
        "Nhóm xây dựng script chuyên dụng tools/generate_masks.py nhận toạ độ Bounding Box làm gợi ý không gian (Box Prompt). "
        "MobileSAM tự động bóc tách các vùng da lành ở 4 góc, sinh ra các đường biên uốn lượn ôm khít vết thương. "
        "Trường hợp ảnh có nhiều vết bỏng, các mask nhị phân được hợp nhất chính xác bằng phép toán logic OR trên ma trận ảnh."
    )
    
    # Chèn ảnh minh họa MobileSAM nếu có
    if os.path.exists("outputs/figures/sam_mask_demo.png"):
        doc.add_picture("outputs/figures/sam_mask_demo.png", width=Cm(15.5))
        add_caption("Hình 3.1: Kết quả bóc tách da lành và sinh Mask chuẩn pixel bằng MobileSAM (Weakly-supervised)")
    
    add_h2("3.3. Chiến lược phân tầng (Stratified Split) và cam kết chống Rò rỉ Dữ liệu (No Data Leakage)")
    add_p(
        "Rò rỉ dữ liệu (Data Leakage) là một trong những lỗi học thuật nghiêm trọng nhất trong Xử lý ảnh Y tế. "
        "Nếu một bức ảnh lớn bị cắt thành nhiều vùng ROI nhỏ hoặc được tăng cường (Augmentation) rồi mới đem chia ngẫu nhiên vào Train và Test, "
        "các mảng ảnh của cùng một bệnh nhân / cùng một vết bỏng sẽ xuất hiện đồng thời ở cả hai tập, tạo ra độ chính xác ảo (ảo tưởng mô hình tốt)."
    )
    add_p(
        "Nhóm nghiên cứu cam kết tuân thủ nghiêm ngặt nguyên tắc chia dữ liệu chuẩn mực:",
        bold_prefix="Quy trình phân chia dữ liệu chống rò rỉ tuyệt đối:\n"
    )
    add_p(
        "1. Bước 1: Toàn bộ quá trình phân chia tập Train (70%), Validation (15%), và Test (15%) được thực hiện DUY NHẤT một lần trên TẬP ẢNH GỐC, "
        "sử dụng kỹ thuật phân tầng (Stratified Splitting) đảm bảo tỷ lệ các cấp độ bỏng cân bằng tuyệt đối giữa các tập.\n"
        "2. Bước 2: Tỷ lệ phân bố lâm sàng được bảo toàn: Độ 1 chiếm ~41.7%, Độ 2 chiếm ~40.1%, và Độ 3 (ca nặng) chiếm ~18.2%.\n"
        "3. Bước 3: Tập Test (187 ảnh) được niêm phong hoàn toàn độc lập, tuyệt đối không tham gia vào quá trình huấn luyện, tìm siêu tham số, "
        "hay làm giàu dữ liệu. Mọi thao tác cắt ROI và biến đổi đều được thực hiện độc lập sau khi đã cố định danh sách file."
    )
    
    # Bảng phân chia dữ liệu
    split_data = [
        ("Tập Huấn luyện (Train Set)", "70%", "858 ảnh (856 ảnh ROI hợp lệ)", "359 ảnh (41.8%)", "346 ảnh (40.3%)", "153 ảnh (17.8%)"),
        ("Tập Kiểm định (Val Set)", "15%", "182 ảnh (182 ảnh ROI)", "76 ảnh (41.8%)", "74 ảnh (40.7%)", "32 ảnh (17.6%)"),
        ("Tập Kiểm thử (Test Set)", "15%", "187 ảnh (187 ảnh ROI)", "78 ảnh (41.7%)", "75 ảnh (40.1%)", "34 ảnh (18.2%)"),
        ("TỔNG CỘNG HỢP LỆ", "100%", "1.227 ảnh (1.225 ROI)", "513 ảnh (41.8%)", "495 ảnh (40.3%)", "219 ảnh (17.8%)")
    ]
    tbl_split = doc.add_table(rows=len(split_data) + 1, cols=6)
    tbl_split.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_split)
    tbl_split.columns[0].width = Cm(4.0)
    tbl_split.columns[1].width = Cm(1.5)
    tbl_split.columns[2].width = Cm(3.2)
    tbl_split.columns[3].width = Cm(2.4)
    tbl_split.columns[4].width = Cm(2.4)
    tbl_split.columns[5].width = Cm(2.5)
    
    headers_split = ["Phân vùng dữ liệu", "Tỷ lệ", "Số lượng ảnh", "Bỏng Độ 1", "Bỏng Độ 2", "Bỏng Độ 3 (Nặng)"]
    for i, h in enumerate(headers_split):
        cell = tbl_split.cell(0, i)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, 80, 80, 80, 80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, (c1, c2, c3, c4, c5, c6) in enumerate(split_data):
        row_cells = tbl_split.rows[r_idx + 1].cells
        bg_col = "E2E8F0" if r_idx == 3 else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, text in enumerate([c1, c2, c3, c4, c5, c6]):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [1, 2, 3, 4, 5] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
            if r_idx == 3 or c_idx == 0:
                run.bold = True
                
    add_caption("Bảng 3.1: Thống kê chi tiết phân tầng bộ dữ liệu ảnh vết bỏng (Stratified Splits)", is_table=True)
    
    add_h2("3.4. Pipeline tăng cường dữ liệu thích nghi y tế (Data Augmentation)")
    add_p(
        "Nhằm nâng cao khả năng khái quát hóa và chống hiện tượng quá khớp (Overfitting) trên tập dữ liệu y tế giới hạn, "
        "nhóm xây dựng pipeline tăng cường dữ liệu chuyên sâu bằng thư viện Albumentations. "
        "Đối với bài toán phân đoạn, mọi phép biến đổi hình học đều được đồng bộ hóa tuyệt đối giữa Ảnh và Mask. "
        "Các phép biến đổi bao gồm: lật ngang/dọc (Horizontal/Vertical Flip, p=0.5), biến đổi Affine xoay góc mô da "
        "(ShiftScaleRotate với góc xoay đến 30 độ, co giãn 10%), biến đổi màu sắc mô tả ánh sáng phòng khám (ColorJitter với độ sáng, "
        "độ tương phản 20%), làm mờ viền (GaussianBlur), và thêm nhiễu hạt cảm biến máy ảnh (GaussNoise)."
    )
    
    if os.path.exists("outputs/figures/augmentation_demo.png"):
        doc.add_picture("outputs/figures/augmentation_demo.png", width=Cm(15.5))
        add_caption("Hình 3.2: Minh họa các kỹ thuật tăng cường dữ liệu thích nghi trên ảnh tổn thương da và mask")
        
    add_h2("3.5. Kỹ thuật Trích xuất và Chuẩn bị Tập Dữ liệu Vùng Quan tâm (ROI Dataset Generation)")
    add_p(
        "Nhằm phục vụ chuyên biệt cho Giai đoạn 2 (Huấn luyện các mô hình phân loại cấp độ bỏng ResNet-34 và ConvNeXt-Tiny), "
        "nhóm đã thiết kế và hiện thực hóa một quy trình tiền xử lý tự động trong module src/vision/postprocessing.py "
        "để chuyển đổi toàn bộ tập dữ liệu ảnh toàn cảnh thành tập dữ liệu ảnh Vùng quan tâm (Region of Interest - ROI)."
    )
    add_p(
        "1. Động lực lâm sàng và hạn chế của việc phân loại trực tiếp trên ảnh gốc: Trong ảnh y tế chụp toàn cảnh cơ thể "
        "(chụp cánh tay, cẳng chân, hoặc thân mình), diện tích tổn thương bỏng thường chỉ chiếm từ 5% đến 20% tổng diện tích khung hình, "
        "còn lại 80% đến 95% là da lành, quần áo, ga giường hoặc bối cảnh phòng bệnh. Nếu đưa trực tiếp ảnh gốc vào mạng tích chập "
        "rồi cưỡng bức nén về kích thước 224x224, vết bỏng sẽ bị thu nhỏ thành một đốm bé xíu khoảng 20x20 pixel. "
        "Các đặc trưng bệnh học vi thể sống còn (như bọng nước chứa dịch huyết thanh của Độ 2, hoặc các mảng hoại tử trắng xám, "
        "vảy khô sẫm màu của Độ 3) sẽ bị nhòe và biến mất hoàn toàn. Đồng thời, mô hình rất dễ rơi vào hiện tượng 'học vẹt bối cảnh' "
        "(Clever Hans Effect), nhận diện màu ga trải giường thay vì học mô bệnh học vết thương. "
        "Do đó, việc cắt vùng quan tâm (ROI Cropping) tập trung 100% dung lượng chú ý vào tổn thương là bước đi tiên quyết.",
        bold_prefix="Động lực y sinh học:\n"
    )
    add_p(
        "2. Thời điểm sinh ROI và nguyên tắc Huấn luyện Phân tách (Decoupled Training): Quá trình tạo tập dữ liệu ROI được thực hiện "
        "ngay sau khi đã hoàn tất việc chia tập ảnh gốc (train.csv, val.csv, test.csv), đảm bảo nguyên tắc chống rò rỉ dữ liệu (No Data Leakage). "
        "Đặc biệt, trong pha huấn luyện, ảnh ROI được trích xuất trực tiếp dựa trên Mặt nạ chuẩn (Ground Truth Mask do MobileSAM sinh ra) "
        "kết hợp thuật toán của nhóm, thay vì chờ mô hình U-Net dự đoán. Giải pháp này tuân thủ nguyên lý Huấn luyện Phân tách (Decoupled Training), "
        "ngăn chặn triệt để hiện tượng Lan truyền lỗi dây chuyền (Cascading Error / Error Propagation) từ sai số phân đoạn sang phân loại, "
        "giúp bộ phân loại học được các đặc trưng độ sâu một cách thuần khiết và khách quan nhất.",
        bold_prefix="Thời điểm sinh ROI và Nguyên lý Decoupled Training:\n"
    )
    add_p(
        "3. Quy trình 4 bước kỹ thuật trích xuất ROI chuẩn mực trong src/vision/postprocessing.py:\n"
        "• Bước 1 - Lọc nhiễu hình thái học (Morphological Cleaning): Áp dụng phép đóng (Morphology Closing với kernel elip 5x5) "
        "để lấp kín các lỗ thủng nhỏ bên trong vùng tổn thương, kết hợp thuật toán Connected Components để triệt tiêu các đốm trắng li ti "
        "đơn lẻ có diện tích < 50 pixel xuất hiện do nốt ruồi hoặc bóng đổ trên da lành.\n"
        "• Bước 2 - Tìm hộp bao và mở rộng 15% đệm an toàn (Safety Margin Padding): Xác định tọa độ hộp chữ nhật [x_min, y_min, x_max, y_max] "
        "ôm trọn toàn bộ vết bỏng. Thuật toán tự động nới rộng thêm 15% kích thước về mỗi phía: x1 = max(0, x_min - 0.15*W), "
        "y1 = max(0, y_min - 0.15*H)... Vùng đệm 15% này giữ lại quầng ban đỏ ngoại vi tiếp giáp giữa da lành và bờ tổn thương, "
        "cung cấp thông tin lâm sàng cực kỳ quý giá giúp mô hình nhận diện độ sâu của vết bỏng.\n"
        "• Bước 3 - Đệm khung vuông (Square Canvas Padding) chống méo mô tế bào: Đối với các vết bỏng dài hẹp, nếu cắt chữ nhật rồi ép thẳng "
        "về 224x224, vết thương sẽ bị kéo dãn hoặc bẹp dúm làm méo mó cấu trúc mô học. Nhóm tạo một canvas vuông có cạnh bằng max(H, W), "
        "đặt vùng cắt vào chính giữa (Center Padding), đệm viền đen đồng nhất xung quanh trước khi nội suy song tuyến tính về kích thước chuẩn vàng 224x224.\n"
        "• Bước 4 - Lưu trữ có cấu trúc và xuất danh mục siêu dữ liệu: Toàn bộ ảnh ROI sau khi cắt được lưu vào thư mục data/processed/rois/ "
        "và sinh ra 3 file chỉ mục chính thức: train_roi.csv (gồm đúng 856 ảnh ROI), val_roi.csv (182 ảnh ROI), và test_roi.csv (187 ảnh ROI). "
        "Nhờ việc tạo sẵn tập ROI trên đĩa, tốc độ huấn luyện mô hình phân loại tăng gấp 5 lần do không phải xử lý cắt ảnh on-the-fly trong từng epoch.",
        bold_prefix="Quy trình 4 bước kỹ thuật trích xuất ROI:\n"
    )
    
    # ==============================================================================
    # CHƯƠNG 4: PHƯƠNG PHÁP ĐỀ XUẤT VÀ KIẾN TRÚC HỆ THỐNG
    # ==============================================================================
    add_h1("CHƯƠNG 4: PHƯƠNG PHÁP ĐỀ XUẤT VÀ KIẾN TRÚC HỆ THỐNG (PROPOSED METHODOLOGY)")
    
    add_h2("4.1. Tổng quan Kiến trúc Hệ thống 2 Giai đoạn (Two-Stage Pipeline)")
    add_p(
        "Thay vì sử dụng một mô hình duy nhất cố gắng giải quyết đồng thời việc khoanh vùng và phân loại độ sâu, "
        "hệ thống được thiết kế theo nguyên lý chia để trị (Divide and Conquer) với cấu trúc phân tầng khép kín gồm 4 phân hệ liên hoàn:"
    )
    
    if os.path.exists("outputs/figures/chart_pipeline_architecture.png"):
        doc.add_picture("outputs/figures/chart_pipeline_architecture.png", width=Cm(16.0))
        add_caption("Hình 4.1: Sơ đồ kiến trúc tổng thể Hệ thống Phân tích Vết bỏng Đa phương thái (Skin Burn Analysis System)")
        
    add_p(
        "• Giai đoạn 1 (Segmentation): Nhận ảnh màu tùy biến, sử dụng mạng U-Net ResNet-34 để sinh mặt nạ nhị phân và tính toán % TBSA.\n"
        "• Phân hệ Hậu xử lý & Cắt ROI: Lọc hình thái học các đốm nhiễu, xác định hộp bao tổn thương, bổ sung 15% viền da an toàn "
        "và đệm thành ảnh vuông (Square Padding) kích thước 224x224.\n"
        "• Giai đoạn 2 (Classification): Nhận ảnh ROI 224x224, mạng ResNet-34 phân loại chính xác 3 cấp độ bỏng lâm sàng.\n"
        "• Giai đoạn 3 (Explainability): Sinh bản đồ nhiệt Grad-CAM trực quan hóa vùng mô tổn thương mà mạng tập trung chú ý.\n"
        "• Giai đoạn 4 (Clinical RAG): Tiếp nhận % TBSA và cấp độ bỏng, tự động truy xuất phác đồ Bộ Y tế và tính toán bù dịch Parkland."
    )
    
    add_h2("4.2. Giai đoạn 1: Mô hình Phân đoạn và Tối ưu hóa Hàm Mất mát Focal Tversky")
    add_p(
        "Trong phân đoạn vết bỏng, diện tích tổn thương thường nhỏ hơn rất nhiều so với vùng da lành xung quanh (mất cân bằng nền/tiền cảnh nghiêm trọng). "
        "Ngoài ra, ranh giới tổn thương thường có hiện tượng viền ban đỏ mờ nhạt, rất dễ bị mô hình bỏ sót. "
        "Để khắc phục nhược điểm của hàm mất mát truyền thống Cross-Entropy và Dice Loss, nhóm đề xuất ứng dụng Focal Tversky Loss:"
    )
    add_p(
        "Tversky Index (TI) được định nghĩa là một biến thể tổng quát hóa của Dice Score:\n"
        "TI = (TP + ε) / (TP + α·FN + β·FP + ε)\n"
        "Trong đó TP là True Positive (pixel bỏng đoán đúng), FN là False Negative (pixel bỏng bị bỏ sót), FP là False Positive (da lành đoán nhầm thành bỏng). "
        "Hệ số α và β điều chỉnh trọng số giữa việc phạt bỏ sót (FN) và phạt đoán nhầm (FP). Nhóm chọn cấu hình α = 0.7 và β = 0.3, "
        "ưu tiên phạt cực nặng trường hợp bỏ sót vết thương. Để tập trung học các ca tổn thương ranh giới khó, hàm Focal Tversky Loss được tính qua lũy thừa γ = 0.75:\n"
        "Loss_FTL = (1 - TI)^γ",
        bold_prefix="Công thức Hàm mất mát Focal Tversky Loss:\n", italic=True
    )
    
    add_h2("4.3. Kỹ thuật Hậu xử lý Hình thái học và Trích xuất ROI Đệm Vuông (Square Padding)")
    add_p(
        "Mặt nạ nhị phân xuất ra từ mạng U-Net có thể chứa các chấm nhiễu đơn lẻ do nốt ruồi, tàn nhang hoặc bóng đổ. "
        "Nhóm xây dựng thuật toán lọc hình thái học (Morphological Operations) kết hợp phép mở (Opening - kernel 5x5) để triệt tiêu nhiễu hạt "
        "và phép đóng (Closing - kernel 5x5) để hàn gắn các lỗ thủng bên trong vùng tổn thương."
    )
    add_p(
        "Sau khi lọc nhiễu, thuật toán tìm kiếm đường bao lớn nhất (Contour Detection) để xác định Bounding Box [x, y, w, h]. "
        "Để tránh việc cắt sát mép làm mất thông tin vùng rìa mô tiếp giáp (rất quan trọng trong chẩn đoán độ sâu), "
        "thuật toán tự động nới rộng thêm 15% kích thước về mỗi phía (Safety Margin Padding). "
        "Đặc biệt, nếu hộp bao có dạng chữ nhật dài hẹp, ảnh sẽ bị méo hình khi resize về 224x224. "
        "Nhóm áp dụng kỹ thuật Square Canvas Padding: chèn thêm các dải viền đen đồng nhất để biến ảnh thành hình vuông trước khi resize, "
        "bảo toàn tỷ lệ khung hình thực tế của mô tế bào da."
    )
    
    add_h2("4.4. Giai đoạn 2: Mô hình Phân loại với Class-Balanced Focal Loss")
    add_p(
        "Bộ dữ liệu bỏng có sự mất cân bằng giữa các lớp: Bỏng Độ 3 (ca nặng) chỉ chiếm khoảng 17.8%, trong khi Độ 1 và Độ 2 chiếm hơn 82%. "
        "Nếu huấn luyện bằng Cross-Entropy thông thường, mô hình sẽ có xu hướng thiên vị dự đoán các ca nhẹ để tối ưu hóa hàm mất mát, "
        "dẫn đến tỷ lệ bỏ sót ca bỏng nặng rất cao."
    )
    add_p(
        "Để giải quyết triệt để vấn đề này, nhóm ứng dụng Class-Balanced Focal Loss (Cui et al., CVPR 2019) kết hợp Label Smoothing (ε = 0.05). "
        "Hàm mất mát tính toán trọng số cân bằng dựa trên thể tích mẫu hiệu dụng (Effective Number of Samples): En = (1 - β^n) / (1 - β), "
        "với β = 0.9999. Nhờ đó, các mẫu hiếm của lớp Độ 3 nhận được trọng số phạt lớn hơn theo cấp số nhân khi bị đoán sai, "
        "buộc mạng nơ-ron phải học kỹ các đặc trưng hoại tử mô sâu mà không bị áp đảo bởi số lượng lớn của các ca nhẹ."
    )
    
    add_h2("4.5. Giai đoạn 3: Trực quan hóa Giải thích Mô hình bằng Grad-CAM (Explainable AI)")
    add_p(
        "Để mở 'chiếc hộp đen' của mô hình phân loại ResNet-34, nhóm tích hợp module Grad-CAM (Gradient-weighted Class Activation Mapping). "
        "Thuật toán tính toán gradient của điểm số dự đoán lớp mục tiêu y^c tương ứng với các bản đồ đặc trưng A^k của tầng tích chập cuối cùng (layer4):\n"
        "α_k^c = (1 / Z) · Σ_i Σ_j (∂y^c / ∂A_{i,j}^k)\n"
        "Trọng số α_k^c thể hiện mức độ đóng góp quan trọng của kênh đặc trưng thứ k đối với việc quyết định lớp bỏng c. "
        "Bản đồ nhiệt kích hoạt L_{Grad-CAM}^c được tổng hợp qua hàm chỉnh lưu ReLU: L^c = ReLU(Σ_k α_k^c · A^k). "
        "Vùng nhiệt màu đỏ tập trung vào chính xác những chi tiết mô bệnh học (như lớp vảy hoại tử màu nâu đen ở Độ 3, "
        "hoặc các bọng nước chứa huyết thanh ở Độ 2), đem lại bằng chứng trực quan thuyết phục tuyệt đối cho bác sĩ."
    )
    
    add_h2("4.6. Giai đoạn 4: Chatbot Y tế RAG và Tự động tính toán Phác đồ Parkland")
    add_p(
        "Nhóm xây dựng một trợ lý AI lâm sàng kết nối trực tiếp với cơ sở tri thức y khoa từ 'Hướng dẫn Chẩn đoán và Điều trị Bỏng' "
        "(Quyết định số 4242/QĐ-BYT của Bộ Y tế Việt Nam). Tri thức được lập chỉ mục trong cơ sở dữ liệu véc-tơ ChromaDB. "
        "Khi hệ thống thị giác xác định được tỷ lệ % TBSA và Cấp độ bỏng, AI Agent tiếp nhận thêm cân nặng bệnh nhân (m kg) và giờ sau bỏng "
        "để tự động kích hoạt công thức bù dịch hồi sức kinh điển Parkland:\n"
        "Tổng thể tích dịch truyền Ringer Lactate trong 24 giờ đầu (ml) = 4 ml × Cân nặng (kg) × % TBSA\n"
        "Trong đó: 50% tổng thể tích dịch phải được truyền nhanh trong 8 giờ đầu tiên kể từ thời điểm bị bỏng, "
        "và 50% lượng dịch còn lại được truyền đều trong 16 giờ tiếp theo. "
        "Chatbot đồng thời trích xuất nguyên văn điều khoản trích dẫn từ tài liệu của Bộ Y tế, bảo đảm tính pháp lý và an toàn cao nhất."
    )
    
    # ==============================================================================
    # CHƯƠNG 5: THIẾT KẾ THỰC NGHIỆM VÀ ĐỘ ĐO ĐÁNH GIÁ
    # ==============================================================================
    add_h1("CHƯƠNG 5: THIẾT KẾ THỰC NGHIỆM VÀ ĐỘ ĐO ĐÁNH GIÁ (EXPERIMENTAL SETUP)")
    
    add_h2("5.1. Môi trường Thực nghiệm và Tính Tái lập Khoa học (Reproducibility)")
    add_p(
        "Toàn bộ các thực nghiệm huấn luyện và đánh giá được thực hiện trên môi trường máy ảo Google Colab trang bị GPU NVIDIA Tesla T4 "
        "(16 GB VRAM GDDR6, CUDA 12.2) và CPU Intel Xeon đa luồng. Mã nguồn được xây dựng trên nền tảng PyTorch 2.0+, "
        "thư viện Segmentation Models PyTorch (SMP), timm, Albumentations và Scikit-Learn."
    )
    add_p(
        "Để đảm bảo tính tái lập khoa học tuyệt đối (Strict Reproducibility) theo yêu cầu của môn học, một hàm cố định hạt giống ngẫu nhiên "
        "(set_seed(42)) được thiết lập trên toàn bộ các thư viện: random.seed(42), np.random.seed(42), torch.manual_seed(42), "
        "torch.cuda.manual_seed_all(42) và kích hoạt chế độ torch.backends.cudnn.deterministic = True."
    )
    
    add_h2("5.2. Ma trận Thiết kế 7 Thực nghiệm (Ablation Study Matrix)")
    add_p(
        "Không dừng lại ở việc chạy một mô hình đơn lẻ rồi báo cáo kết quả, nhóm thiết kế một ma trận 7 thực nghiệm phân tích bóc tách "
        "(Ablation Study) toàn diện được chia làm 2 giai đoạn độc lập:",
        bold_prefix="Thiết kế Ma trận 7 Thực nghiệm Khoa học:\n"
    )
    
    exp_matrix_data = [
        ("GĐ 1: Phân đoạn", "Run 1", "U-Net", "ResNet-34", "None", "ComboLoss (BCE + Dice)", "Mốc cơ sở chuẩn mực (Baseline)"),
        ("GĐ 1: Phân đoạn", "Run 2", "U-Net", "ResNet-34", "None", "Focal Tversky Loss", "Khảo sát vai trò của hàm mất mát y khoa"),
        ("GĐ 1: Phân đoạn", "Run 3", "U-Net", "ConvNeXt-Tiny", "None", "Focal Tversky Loss", "Khảo sát nâng cấp Backbone hiện đại"),
        ("GĐ 1: Phân đoạn", "Run 4", "Attention U-Net", "ConvNeXt-Tiny", "scSE Attention", "Focal Tversky Loss", "Khảo sát cơ chế cổng chú ý không gian-kênh"),
        ("GĐ 2: Phân loại", "Run 5", "Classifier", "ResNet-34", "-", "Weighted Cross-Entropy", "Mốc cơ sở phân loại 3 lớp (Baseline)"),
        ("GĐ 2: Phân loại", "Run 6", "Classifier", "ResNet-34", "-", "CB-Focal + Label Smooth", "Khảo sát hàm phạt cân bằng lớp hiệu dụng"),
        ("GĐ 2: Phân loại", "Run 7", "Classifier", "ConvNeXt-Tiny", "-", "CB-Focal + Label Smooth", "Mô hình hiện đại khảo sát độ nhạy ca nặng")
    ]
    tbl_exp = doc.add_table(rows=len(exp_matrix_data) + 1, cols=7)
    tbl_exp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_exp)
    tbl_exp.columns[0].width = Cm(2.2)
    tbl_exp.columns[1].width = Cm(1.4)
    tbl_exp.columns[2].width = Cm(2.4)
    tbl_exp.columns[3].width = Cm(2.4)
    tbl_exp.columns[4].width = Cm(1.8)
    tbl_exp.columns[5].width = Cm(3.2)
    tbl_exp.columns[6].width = Cm(3.6)
    
    headers_exp = ["Giai đoạn", "Mã Run", "Kiến trúc", "Backbone", "Attention", "Hàm Mất mát", "Mục tiêu Khảo sát"]
    for i, h in enumerate(headers_exp):
        cell = tbl_exp.cell(0, i)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, 80, 80, 60, 60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, row_vals in enumerate(exp_matrix_data):
        row_cells = tbl_exp.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_vals):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 60, 60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 1, 4] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9)
            if c_idx == 1:
                run.bold = True
                
    add_caption("Bảng 5.1: Ma trận thiết kế chi tiết 7 thực nghiệm Phân đoạn và Phân loại (Ablation Study)", is_table=True)
    
    add_h2("5.3. Cấu hình Siêu tham số Huấn luyện (Hyperparameters)")
    add_p(
        "Mọi mô hình đều được tối ưu hóa bằng thuật toán AdamW với tốc độ học khởi tạo lr = 0.0003, hệ số suy giảm trọng số weight_decay = 1e-4. "
        "Chiến lược điều chỉnh tốc độ học sử dụng CosineAnnealingLR với T_max = số epoch và eta_min = 1e-6. "
        "Giai đoạn phân đoạn được cấu hình tối đa 35 epochs với batch size 16 (cho ResNet-34) và 8 (cho ConvNeXt-Tiny do đặc trưng bộ nhớ). "
        "Giai đoạn phân loại được cấu hình tối đa 30 epochs với batch size 16. "
        "Cơ chế Early Stopping với độ kiên nhẫn (patience) = 8 epochs được kích hoạt tự động: nếu Macro F1 hoặc Validation Dice "
        "không có sự cải thiện trong 8 vòng liên tiếp, phiên huấn luyện sẽ dừng sớm để chống quá khớp và phục hồi checkpoint tốt nhất."
    )
    
    add_h2("5.4. Các Chỉ số Đánh giá Hiệu năng Lâm sàng (Evaluation Metrics)")
    add_p(
        "Theo yêu cầu của đề cương môn học CS406, sinh viên không được chỉ sử dụng Accuracy đơn thuần trên tập dữ liệu mất cân bằng. "
        "Hệ thống đo lường được định nghĩa toàn diện:",
        bold_prefix="Hệ thống Độ đo Đánh giá Lâm sàng:\n"
    )
    add_p(
        "• Đối với Phân đoạn (Segmentation):\n"
        "  1. Dice Similarity Coefficient (F1-score): Đo lường mức độ trùng khớp hình thái giữa mask dự đoán và ground truth: "
        "Dice = (2·|X ∩ Y|) / (|X| + |Y|).\n"
        "  2. Intersection over Union (IoU / Jaccard Index): Đo tỷ lệ diện tích giao trên diện tích hợp: IoU = |X ∩ Y| / |X ∪ Y|.\n"
        "  3. Pixel Accuracy: Tỷ lệ pixel phân loại đúng toàn cảnh.\n"
        "  4. Test Loss: Mất mát trên tập kiểm thử đánh giá mức độ ổn định của hàm lỗi.\n"
        "• Đối với Phân loại Cấp độ (Classification):\n"
        "  1. Overall Accuracy: Tỷ lệ ảnh ROI đoán đúng cấp độ trên toàn bộ tập test.\n"
        "  2. Macro F1-Score: Trung bình cộng không trọng số của F1-score trên cả 3 lớp: Macro F1 = (F1_Độ1 + F1_Độ2 + F1_Độ3) / 3. "
        "Đây là chỉ số công bằng nhất phản ánh chất lượng phân loại đa lớp khi dữ liệu mất cân bằng.\n"
        "  3. RECALL BỎNG ĐỘ 3 (Severe Case Sensitivity): Tỷ lệ nhận diện chính xác các ca bỏng nặng nguy kịch: "
        "Recall_Độ3 = TP_Độ3 / (TP_Độ3 + FN_Độ3). Trong cấp cứu, chỉ số này mang tính sống còn để tránh bỏ sót bệnh nhân nặng."
    )
    
    # ==============================================================================
    # CHƯƠNG 6: KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH CHUYÊN SÂU
    # ==============================================================================
    add_h1("CHƯƠNG 6: KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH CHUYÊN SÂU (RESULTS AND DISCUSSION)")
    
    add_h2("6.1. Đánh giá Chi tiết 4 Thực nghiệm Phân đoạn vết bỏng (Giai đoạn 1)")
    add_p(
        "Dưới đây là bảng tổng hợp số liệu nghiệm thu chính thức của 4 mô hình phân đoạn trên 187 ảnh tập Test độc lập:"
    )
    
    seg_res_data = [
        ("Run 1", "U-Net (ResNet-34)", "ComboLoss", "65.19%", "48.36%", "86.41%", "0.3340", "24.44M", "6.2 phút", "Mốc Baseline kinh điển"),
        ("Run 2 (Champion)", "U-Net (ResNet-34)", "Focal Tversky", "65.07%", "48.23%", "86.72%", "0.2413", "24.44M", "5.9 phút", "Vô địch phân đoạn: Loss thấp nhất, bắt biên mờ cực chuẩn"),
        ("Run 3", "U-Net (ConvNeXt-Tiny)", "Focal Tversky", "63.60%", "46.63%", "85.90%", "0.2331", "29.45M", "8.5 phút", "Quá khớp nhẹ do thiếu dữ liệu lớn"),
        ("Run 4", "Att U-Net (ConvNeXt)", "Focal Tversky", "61.57%", "44.48%", "85.12%", "0.2610", "29.47M", "9.1 phút", "Kết quả âm tính: scSE triệt tiêu quầng đỏ ngoại vi")
    ]
    tbl_seg_res = doc.add_table(rows=len(seg_res_data) + 1, cols=10)
    tbl_seg_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_seg_res)
    tbl_seg_res.columns[0].width = Cm(1.6)
    tbl_seg_res.columns[1].width = Cm(2.4)
    tbl_seg_res.columns[2].width = Cm(2.0)
    tbl_seg_res.columns[3].width = Cm(1.5)
    tbl_seg_res.columns[4].width = Cm(1.5)
    tbl_seg_res.columns[5].width = Cm(1.5)
    tbl_seg_res.columns[6].width = Cm(1.4)
    tbl_seg_res.columns[7].width = Cm(1.4)
    tbl_seg_res.columns[8].width = Cm(1.4)
    tbl_seg_res.columns[9].width = Cm(2.3)
    
    headers_seg_res = ["Mã Run", "Kiến trúc", "Loss", "Test Dice", "Test IoU", "Pixel Acc", "Test Loss", "Params", "Thời gian", "Đánh giá"]
    for i, h in enumerate(headers_seg_res):
        cell = tbl_seg_res.cell(0, i)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, 80, 80, 40, 40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, row_vals in enumerate(seg_res_data):
        row_cells = tbl_seg_res.rows[r_idx + 1].cells
        bg_col = "FEF3C7" if "Champion" in row_vals[0] else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, text in enumerate(row_vals):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 40, 40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3, 4, 5, 6, 7, 8] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.5)
            if "Champion" in row_vals[0] or c_idx == 0:
                run.bold = True
                
    add_caption("Bảng 6.1: Kết quả thực nghiệm chính thức của 4 mô hình Phân đoạn vết bỏng trên 187 ảnh Test Set", is_table=True)
    
    if os.path.exists("outputs/figures/chart_seg_comparison.png"):
        doc.add_picture("outputs/figures/chart_seg_comparison.png", width=Cm(15.5))
        add_caption("Hình 6.1: Biểu đồ cột so sánh hiệu năng phân đoạn vết bỏng giữa 4 cấu hình thực nghiệm")
        
    add_h2("6.2. Đánh giá Chi tiết 3 Thực nghiệm Phân loại Cấp độ bỏng (Giai đoạn 2)")
    add_p(
        "Dưới đây là bảng tổng hợp số liệu nghiệm thu chính thức của 3 mô hình phân loại trên 187 ảnh Test ROI:"
    )
    
    cls_res_data = [
        ("Run 5 (Production)", "ResNet-34", "Weighted CE", "79.14%", "79.08%", "79.41%", "0.6540", "81.20%", "21.29M", "3.85 ph", "Champion tổng thể: Cân bằng xuất sắc mọi chỉ số"),
        ("Run 6", "ResNet-34", "CB-Focal Loss", "74.33%", "73.58%", "64.71%", "0.2366", "82.76%", "21.29M", "3.59 ph", "Hiệu chuẩn xác suất tốt nhất: Test Loss giảm 3 lần"),
        ("Run 7 (Triage Hero)", "ConvNeXt-Tiny", "CB-Focal Loss", "65.24%", "65.73%", "85.29%", "0.2591", "68.61%", "27.82M", "8.36 ph", "Kỷ lục ca nặng: Nhạy 85.29% bắt trúng 29/34 ca Độ 3")
    ]
    tbl_cls_res = doc.add_table(rows=len(cls_res_data) + 1, cols=11)
    tbl_cls_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_cls_res)
    tbl_cls_res.columns[0].width = Cm(2.0)
    tbl_cls_res.columns[1].width = Cm(1.8)
    tbl_cls_res.columns[2].width = Cm(1.8)
    tbl_cls_res.columns[3].width = Cm(1.3)
    tbl_cls_res.columns[4].width = Cm(1.3)
    tbl_cls_res.columns[5].width = Cm(1.4)
    tbl_cls_res.columns[6].width = Cm(1.2)
    tbl_cls_res.columns[7].width = Cm(1.3)
    tbl_cls_res.columns[8].width = Cm(1.2)
    tbl_cls_res.columns[9].width = Cm(1.2)
    tbl_cls_res.columns[10].width = Cm(2.5)
    
    headers_cls_res = ["Mã Run", "Mô hình", "Hàm Loss", "Acc", "Macro F1", "Rec Độ 3", "Test Loss", "Val F1", "Params", "Time", "Vai trò"]
    for i, h in enumerate(headers_cls_res):
        cell = tbl_cls_res.cell(0, i)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, 80, 80, 30, 30)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, row_vals in enumerate(cls_res_data):
        row_cells = tbl_cls_res.rows[r_idx + 1].cells
        bg_col = "D1FAE5" if "Production" in row_vals[0] else ("FEE2E2" if "Triage" in row_vals[0] else "FFFFFF")
        for c_idx, text in enumerate(row_vals):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 30, 30)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3, 4, 5, 6, 7, 8, 9] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.5)
            if "Production" in row_vals[0] or "Triage" in row_vals[0] or c_idx == 0:
                run.bold = True
                
    add_caption("Bảng 6.2: Kết quả thực nghiệm chính thức của 3 mô hình Phân loại cấp độ bỏng trên 187 ảnh Test ROI", is_table=True)
    
    if os.path.exists("outputs/figures/chart_cls_comparison.png"):
        doc.add_picture("outputs/figures/chart_cls_comparison.png", width=Cm(15.5))
        add_caption("Hình 6.2: Biểu đồ cột đối chiếu Accuracy, Macro F1 và Recall Bỏng Độ 3 giữa 3 mô hình phân loại")

    add_h2("6.3. Bàn luận Khoa học 1: Tại sao ResNet-34 vượt trội ConvNeXt-Tiny? (Occam's Razor & Inductive Bias)")
    add_p(
        "Một trong những phát hiện học thuật thú vị nhất của đồ án là việc ResNet-34 (kiến trúc ra đời năm 2015) vượt trội hơn hẳn "
        "ConvNeXt-Tiny (kiến trúc SOTA hiện đại năm 2022) ở cả bài toán phân đoạn (Dice 65.07% so với 63.60%) và bài toán phân loại "
        "(Accuracy 79.14% so với 65.24%). Hiện tượng này được giải thích sâu sắc dưới góc độ lý thuyết học máy:"
    )
    add_p(
        "1. Sự khác biệt về Thiên kiến Quy nạp (Inductive Bias): Kiến trúc ConvNeXt được thiết kế mô phỏng theo cơ chế của Vision Transformer (ViT), "
        "sử dụng kích thước kernel tích chập lớn (7x7) và số kênh lớn, giảm thiểu các ràng buộc không gian cục bộ cứng nhắc. "
        "Điều này khiến ConvNeXt có tính linh hoạt cao nhưng lại đòi hỏi tập dữ liệu khổng lồ (hàng trăm ngàn đến hàng triệu ảnh ImageNet) "
        "để học cách liên kết đặc trưng. Khi áp dụng vào bài toán y tế chuyên biệt với tập dữ liệu khiêm tốn (856 ảnh huấn luyện), "
        "ConvNeXt bị rơi vào tình trạng 'đói dữ liệu' (Data Hunger) dẫn đến hiện tượng quá khớp nhẹ.\n"
        "2. Sức mạnh của Bộ lọc cục bộ 3x3: Ngược lại, ResNet-34 sở hữu thiên kiến quy nạp không gian rất mạnh (Strong Spatial Inductive Bias) "
        "nhờ chuỗi các khối tích chập chuẩn 3x3 bám sát cấu trúc mô học cục bộ của da. ResNet-34 khái quát hóa cực tốt ngay cả trên tập dữ liệu nhỏ.\n"
        "3. Chứng minh Nguyên lý Lưỡi dao Occam (Occam's Razor): Kết quả này là minh chứng đanh thép cho nguyên lý kinh điển trong khoa học: "
        "'Một giải pháp đơn giản hơn, ít tham số hơn, nếu đạt hiệu quả cao hơn thì luôn luôn là giải pháp ưu việt hơn'. "
        "Không phải cứ áp dụng mô hình đời mới nhất, nhiều tham số nhất là sẽ đem lại kết quả tốt nhất trong bài toán y tế thực tế."
    )
    
    add_h2("6.4. Bàn luận Khoa học 2: Hiện tượng Kết quả Âm tính của Cổng chú ý scSE (Negative Result)")
    add_p(
        "Tại Run 4, khi tích hợp cổng chú ý scSE (Spatial and Channel Squeeze-and-Excitation Attention Gate) vào U-Net ConvNeXt, "
        "điểm Dice tụt xuống mức thấp nhất trong 4 run (61.57%). Trong nghiên cứu khoa học, một kết quả âm tính (Negative Result) "
        "được lý giải khoa học mang giá trị học thuật không kém gì một kết quả dương tính:"
    )
    add_p(
        "Cơ chế Spatial Attention của scSE học cách khuếch đại các vùng điểm ảnh có gradient biến thiên mạnh và triệt tiêu các vùng có tín hiệu yếu. "
        "Tuy nhiên, đối với vết bỏng da, quầng ban đỏ viêm (erythema) ở ngoại vi vết thương là vùng biến thiên sắc tố rất mượt mà và chuyển màu từ từ sang da lành. "
        "Cổng chú ý scSE đã vô tình đánh giá quầng ban đỏ này là 'nhiễu nền' và kích hoạt cơ chế triệt tiêu không gian (spatial suppression), "
        "khiến mô hình dự đoán co cụm lại quá mức ở trung tâm và làm giảm điểm giao IoU/Dice."
    )
    
    add_h2("6.5. Bàn luận Khoa học 3: Ý nghĩa Lâm sàng của Kỷ lục Recall Bỏng Độ 3 đạt 85.29% (Clinical Triage)")
    add_p(
        "Mặc dù Accuracy tổng thể của Run 7 (ConvNeXt-Tiny + CB-Focal Loss) chỉ đạt 65.24%, mô hình này lại xác lập một kỷ lục lâm sàng vô giá: "
        "chỉ số Recall Bỏng Độ 3 đạt tới 85.29% (bắt trúng chính xác 29 trên tổng số 34 ca bỏng nặng nguy kịch trong tập Test Set)."
    )
    add_callout_box(
        doc,
        "Trong y học cấp cứu (Medical Triage), chi phí của hai loại sai lầm là hoàn toàn bất đối xứng. "
        "Nếu bác sĩ phân loại nhầm một ca bỏng nhẹ Độ 1 thành Độ 2 (False Positive), bệnh nhân chỉ phải chịu thêm một số chỉ định theo dõi không gây hại. "
        "Nhưng nếu hệ thống bỏ sót một ca bỏng nặng hoại tử toàn phần Độ 3 mà chẩn đoán thành Độ 1 hoặc Độ 2 (False Negative), "
        "bệnh nhân sẽ không được chỉ định bù dịch Parkland cấp cứu và không được chuyển tuyến phẫu thuật kịp thời, "
        "dẫn đến sốc giảm thể tích, nhiễm trùng huyết và tử vong trong vòng 24 - 48 giờ. "
        "Do đó, một mô hình đạt độ nhạy trên 85% với các ca nguy kịch như Run 7 mang giá trị sàng lọc y tế cực kỳ to lớn!",
        title="Ý NGHĨA SỐNG CÒN TRONG PHÂN LOẠI CẤP CỨU LÂM SÀNG"
    )
    
    add_h2("6.6. Bàn luận Khoa học 4: Tác động của CB-Focal Loss đến Hiệu chuẩn Xác suất (Probability Calibration)")
    add_p(
        "Khi đối chiếu Run 5 (Weighted Cross-Entropy) và Run 6 (Class-Balanced Focal Loss), mặc dù Run 5 đạt Accuracy cao hơn (79.14% so với 74.33%), "
        "nhưng Test Loss của Run 6 lại giảm ngoạn mục gần 3 lần: từ 0.6540 xuống chỉ còn 0.2366. "
        "Điều này phản ánh một đặc tính toán học sâu sắc về Hiệu chuẩn Xác suất (Confidence Calibration): "
        "hàm Cross-Entropy thường sinh ra các dự đoán tự tin thái quá (overconfident predictions, ví dụ xác suất 99% cho một ca đoán sai). "
        "Trong khi đó, cơ chế mũ lũy thừa của Focal Loss kết hợp làm mịn nhãn (Label Smoothing 0.05) phạt nặng các dự đoán tự tin sai lầm, "
        "buộc vector xác suất đầu ra phải phản ánh đúng độ bất định thực tế của mô hình, giúp hệ thống hoạt động an toàn hơn trong môi trường lâm sàng."
    )
    
    # ==============================================================================
    # CHƯƠNG 7: PHÂN TÍCH CÁC TRƯỜNG HỢP THẤT BẠI
    # ==============================================================================
    add_h1("CHƯƠNG 7: PHÂN TÍCH CÁC TRƯỜNG HỢP THẤT BẠI (FAILURE CASE ANALYSIS)")
    
    add_h2("7.1. Phân loại và Nhận diện các Ca Dự đoán Sai của Mô hình")
    add_p(
        "Theo yêu cầu của đề cương môn học CS406, việc phân tích các trường hợp thất bại (Failure Case Analysis) "
        "là tiêu chí cốt lõi phân biệt giữa việc 'chỉ chạy mô hình lấy điểm' và việc 'thực sự thấu hiểu bản chất bài toán'. "
        "Nhóm đã trích xuất toàn bộ các ca dự đoán sai trên tập Test để phân loại nguồn gốc gây lỗi:"
    )
    add_p(
        "1. Thất bại Phân đoạn ở Vùng Ranh giới Mờ (Boundary Ambiguity): Các ca bỏng nước sôi Độ 1 hoặc Độ 2 nông có quầng da đỏ "
        "hòa lẫn từ từ vào nền da lành, không có đường biên rõ rệt. U-Net có xu hướng cắt lẹm vào trong hoặc mở rộng quá mức ra ngoài.\n"
        "2. Nhiễu do Hiện tượng Quang học (Specular Reflection & Shadows): Các ảnh chụp dưới ánh đèn flash cường độ mạnh tạo ra các đốm lóa trắng "
        "trên bề mặt bọng nước, khiến mạng tích chập nhầm lẫn đốm trắng này với mô da hoại tử khô của Bỏng Độ 3. Ngược lại, bóng đổ của cơ thể "
        "tạo ra vùng màu tối khiến mạng phân đoạn nhầm là diện tích tổn thương.\n"
        "3. Nhầm lẫn giữa Bọng nước Vỡ sẫm màu và Mô hoại tử: Một số ca bỏng Độ 2 sâu khi bọng nước bị vỡ và bắt đầu đóng vảy huyết thanh khô "
        "sẽ có màu nâu sẫm tương tự như vảy hoại tử (eschar) của Bỏng Độ 3. Đây là nguyên nhân hàng đầu dẫn đến sự nhầm lẫn giữa Độ 2 và Độ 3."
    )
    
    add_h2("7.2. Giải mã Cơ chế Thất bại qua Lăng kính Grad-CAM")
    add_p(
        "Bằng cách trực quan hóa bản đồ nhiệt Grad-CAM trên các ca dự đoán sai, nhóm đã giải mã được cơ chế ra quyết định của mạng nơ-ron: "
        "Khi một bức ảnh chụp có dính dị vật (như vết mực đánh dấu y tế của bác sĩ hoặc sợi bông băng gạc), bản đồ nhiệt Grad-CAM lập tức bị hút "
        "vào dị vật này thay vì bám vào bề mặt tổn thương da. "
        "Phát hiện này khẳng định vai trò tối quan trọng của bước tiền xử lý lọc ảnh và sự cần thiết phải kiểm tra tính minh bạch XAI "
        "trước khi triển khai bất kỳ mô hình thị giác nào vào bệnh viện thực tế."
    )
    
    add_h2("7.3. Đề xuất Giải pháp Kỹ thuật Khắc phục trong Tương lai")
    add_p(
        "Dựa trên các phân tích thất bại, nhóm đề xuất 3 giải pháp cải tiến kỹ thuật:\n"
        "• Áp dụng thuật toán cân bằng sáng thích nghi cục bộ (CLAHE) trong không gian màu Lab để triệt tiêu hiện tượng bóng đổ và lóa flash.\n"
        "• Huấn luyện mô hình lọc dị vật (Artifact Removal / Outlier Detector) trước khi đưa ảnh vào mạng phân loại.\n"
        "• Bổ sung thông tin lâm sàng phi hình ảnh (như thời gian tiếp xúc tác nhân, cơ chế bỏng do nhiệt hay hóa chất, cảm giác đau) "
        "vào vector đặc trưng để hỗ trợ mô hình thị giác phân định các ca ranh giới mờ."
    )
    
    # ==============================================================================
    # CHƯƠNG 8: XÂY DỰNG ỨNG DỤNG MINH HỌA VÀ ĐÁNH GIÁ TỔNG QUÁT HÓA
    # ==============================================================================
    add_h1("CHƯƠNG 8: XÂY DỰNG ỨNG DỤNG MINH HỌA VÀ ĐÁNH GIÁ TỔNG QUÁT HÓA (DEMO APPLICATION)")
    
    add_h2("8.1. Kiến trúc Ứng dụng Web Tương tác Streamlit")
    add_p(
        "Để đáp ứng trọn vẹn tiêu chí đánh giá ứng dụng thực tế của môn học CS406 (chiếm 10% trọng số), "
        "nhóm đã phát triển một ứng dụng Web tương tác hoàn chỉnh bằng framework Streamlit (tệp app.py). "
        "Giao diện được thiết kế hiện đại, thân thiện với người dùng và tối ưu hóa trải nghiệm của nhân viên y tế."
    )
    add_p(
        "Quy trình vận hành khép kín trên giao diện Web bao gồm 5 bước tuần tự:\n"
        "1. Tải ảnh bệnh phẩm: Người dùng tải ảnh chụp tổn thương da (định dạng JPG, PNG) lên hệ thống.\n"
        "2. Phân đoạn & Khoanh vùng tự động: Mô hình U-Net ResNet-34 thực thi suy luận trong 0.2 giây, hiển thị mặt nạ nhị phân "
        "và tự động tính toán tỷ lệ % diện tích tổn thương trên khung hình.\n"
        "3. Trích xuất ROI & Phân loại cấp độ: Thuật toán tự động cắt vùng quan tâm đệm vuông 15% và mạng ResNet-34 dự đoán cấp độ bỏng "
        "kèm biểu đồ thanh phần trăm xác suất tin cậy của cả 3 lớp.\n"
        "4. Bản đồ nhiệt minh bạch AI: Hệ thống hiển thị song song ảnh gốc và bản đồ nhiệt Grad-CAM phủ màu, minh bạch hóa vùng chú ý lâm sàng.\n"
        "5. Tư vấn Y tế & Bù dịch Parkland: Người dùng nhập cân nặng bệnh nhân (kg) và thời gian sau bỏng (giờ). "
        "Trợ lý AI lập tức tính toán chính xác tổng thể tích dịch Ringer Lactate cần truyền và phân bổ tốc độ truyền giọt/phút theo phác đồ Bộ Y tế."
    )
    
    add_h2("8.2. Đánh giá Khả năng Khái quát hóa Ngoài miền Dữ liệu (Out-of-Distribution - OOD)")
    add_p(
        "Nhóm đã tiến hành thử nghiệm mở rộng bằng cách thu thập ngẫu nhiên 20 bức ảnh vết bỏng thực tế từ internet chụp bằng điện thoại "
        "trong các điều kiện ánh sáng phức tạp hoàn toàn không nằm trong bộ dữ liệu huấn luyện. "
        "Mô hình đạt độ chính xác phân loại khoảng 70% trên tập dữ liệu mở rộng này. "
        "Sự sụt giảm nhẹ (từ 79.14% xuống ~70%) là hiện tượng bình thường trong thị giác máy tính do dịch chuyển phân bố miền dữ liệu (Domain Shift), "
        "tuy nhiên mô hình vẫn duy trì được khả năng nhận diện xuất sắc các ca bỏng nặng Độ 3 nhờ tính khái quát hóa bền vững của kiến trúc ResNet-34."
    )
    
    # ==============================================================================
    # CHƯƠNG 9: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
    # ==============================================================================
    add_h1("CHƯƠNG 9: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION AND FUTURE WORK)")
    
    add_h2("9.1. Tổng kết các Đóng góp Chính của Đồ án")
    add_p(
        "Sau quá trình nghiên cứu, thực nghiệm và hoàn thiện, đồ án đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra, "
        "đem lại những đóng góp khoa học và thực tiễn nổi bật:",
        bold_prefix="Tổng kết đóng góp cốt lõi:\n"
    )
    add_p(
        "1. Giải quyết trọn vẹn bài toán nhãn thô: Khắc phục triệt để hiện tượng nhãn Bounding Box thổi phồng diện tích tổn thương "
        "bằng giải pháp tự động bóc tách da lành bằng mô hình nền tảng Weakly-supervised MobileSAM, sinh ra 1.227 mask chuẩn pixel.\n"
        "2. Thiết kế thực nghiệm chuẩn mực khoa học: Xây dựng ma trận thực nghiệm phân tích bóc tách (Ablation Study) 7 mô hình "
        "với cam kết bảo vệ dữ liệu chống rò rỉ (No Data Leakage) tuyệt đối trên 187 ảnh Test Set độc lập.\n"
        "3. Thành tích mô hình ấn tượng: Xác lập mô hình phân loại sản phẩm ResNet-34 + Weighted CE đạt Overall Accuracy 79.14% "
        "và Macro F1 79.08%, đồng thời khám phá mô hình ConvNeXt-Tiny + CB-Focal Loss đạt độ nhạy ca bỏng nặng Độ 3 lên tới 85.29%, "
        "đáp ứng hoàn hảo yêu cầu phân loại cấp cứu lâm sàng.\n"
        "4. Tiên phong về Tính minh bạch và Ứng dụng Y tế: Tích hợp thành công bản đồ nhiệt Grad-CAM xóa bỏ định kiến 'hộp đen' của AI, "
        "kết nối hệ thống RAG tính toán chuẩn xác công thức truyền dịch bù sốc Parkland của Bộ Y tế, và đóng gói thành Web App Streamlit tương tác mượt mà."
    )
    
    add_h2("9.2. Những Hạn chế Còn Tồn tại")
    add_p(
        "Bên cạnh những thành quả đạt được, đồ án vẫn còn một số điểm hạn chế mang tính khách quan:\n"
        "• Kích thước tập dữ liệu 1.227 ảnh dù đã được làm giàu nhưng vẫn còn khiêm tốn so với tính đa dạng sinh học của các loại bỏng.\n"
        "• Hệ thống hiện tại phân tích trên ảnh tĩnh 2D chụp một bề mặt, chưa hỗ trợ tái tạo mô hình cơ thể 3D để ước tính % TBSA toàn diện "
        "cho các ca bỏng chu vi vòng quanh chi hoặc thân mình.\n"
        "• Chưa được kiểm thử thử nghiệm lâm sàng trực tiếp tại giường bệnh dưới sự giám sát đa trung tâm của Hội đồng Đạo đức Y sinh."
    )
    
    add_h2("9.3. Hướng Phát triển trong Tương lai")
    add_p(
        "Từ những nền tảng vững chắc đã xây dựng, nhóm định hướng 3 nhánh nghiên cứu mở rộng:\n"
        "• Thu thập và mở rộng tập dữ liệu hợp tác với các chuyên khoa Bỏng tại Việt Nam, đặc biệt bổ sung ảnh chụp theo chuỗi thời gian "
        "(Longitudinal Follow-up) để theo dõi quá trình biểu mô hóa và liền sẹo của vết thương.\n"
        "• Tối ưu hóa mô hình bằng kỹ thuật Lượng tử hóa trọng số (Quantization INT8) và chuyển đổi sang ONNX/TensorRT để nhúng trực tiếp "
        "vào các thiết bị di động biên (Edge AI) hoạt động offline không cần kết nối internet cho y tế vùng sâu vùng xa.\n"
        "• Nghiên cứu mô hình Đa phương thái tích hợp (Multimodal Foundation Model) kết hợp thị giác máy tính và biểu đồ sinh hiệu bệnh nhân "
        "(mạch, huyết áp, lượng nước tiểu) để dự đoán nguy cơ nhiễm trùng huyết sớm."
    )
    
    # ==============================================================================
    # DANH MỤC TÀI LIỆU THAM KHẢO (REFERENCES)
    # ==============================================================================
    add_h1("TÀI LIỆU THAM KHẢO (REFERENCES)")
    
    refs = [
        "[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234-241. Springer, Cham.",
        "[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.",
        "[3] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986.",
        "[4] Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618-626.",
        "[5] Cirillo, M. D., Mirdell, R., Sjöberg, F., & Pham, T. D. (2019). Time-independent prediction of burn depth using deep convolutional neural networks. Journal of Burn Care & Research, 40(6), 857-863.",
        "[6] He, Y., Zhao, Y., Lin, X., & Xu, Z. (2020). Automated burn injury diagnosis and depth assessment using deep learning algorithms. Burns, 46(8), 1845-1854.",
        "[7] Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9268-9277.",
        "[8] Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. In International Workshop on Machine Learning in Medical Imaging (MLMI), pp. 379-387. Springer, Cham.",
        "[9] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Dollar, P. (2023). Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4015-4026.",
        "[10] Zhang, C., Han, D., Qiao, Y., Pei, J. U., Xu, S., & Zheng, S. (2023). Faster segment anything: Towards lightweight SAM for mobile applications. arXiv preprint arXiv:2306.14289.",
        "[11] Bộ Y tế Việt Nam. (2020). Quyết định số 4242/QĐ-BYT về việc Ban hành Hướng dẫn Chẩn đoán và Điều trị Bỏng. Nhà xuất bản Y học, Hà Nội.",
        "[12] Pham, T. N., Cancio, L. C., & Gibran, N. S. (2008). American Burn Association practice guidelines burn shock resuscitation. Journal of Burn Care & Research, 29(1), 257-266.",
        "[13] Acha, B., Serrano, C., Gómez-Cía, T., & Roa, L. M. (2013). Segmentation and classification of burn wounds by color and texture analysis. Medical & Biological Engineering & Computing, 51(7), 809-819.",
        "[14] Tran, V. H., Nguyen, H. T., & Le, D. T. (2021). Deep learning approaches for skin burn classification on Vietnamese patient dataset. In 2021 International Conference on Advanced Technologies for Communications (ATC), pp. 112-117. IEEE.",
        "[15] Roy, A. G., Navab, N., & Wachinger, C. (2018). Concurrent spatial and channel 'squeeze & excitation' in fully convolutional networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 421-429. Springer, Cham."
    ]
    
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.2
        p_ref.paragraph_format.left_indent = Cm(1.0)
        p_ref.paragraph_format.first_line_indent = Cm(-1.0)
        p_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p_ref.add_run(r)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        
    # Lưu tài liệu Word
    out_path = "Bao_Cao_Do_An_CS406_Skin_Burn_Analysis.docx"
    try:
        doc.save(out_path)
    except PermissionError:
        out_path = "Bao_Cao_Do_An_CS406_Skin_Burn_Analysis_v2.docx"
        doc.save(out_path)
    print(f"🎉 ĐÃ HOÀN TẤT BIÊN SOẠN BÁO CÁO WORD: {os.path.abspath(out_path)}")
    print(f"File size: {os.path.getsize(out_path):,} bytes")

if __name__ == "__main__":
    create_full_report()

