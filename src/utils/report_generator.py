"""
Module: src/utils/report_generator.py
Mục đích: Xuất báo cáo y khoa bệnh án điện tử định dạng PDF chuyên nghiệp.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2

def generate_burn_report_pdf(output_pdf_path: str, data: dict, original_img_path: str, overlay_img_path: str, gradcam_img_path: str = None):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0058bc'),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#717786'),
        alignment=1
    )
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1a1c1f'),
        spaceBefore=12,
        spaceAfter=6
    )
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1a1c1f')
    )

    story.append(Paragraph("BURNSCAN CLINICAL ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Thoi gian lap: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | ID He thong: BS-{int(datetime.now().timestamp())}", subtitle_style))
    story.append(Spacer(1, 15))

    metrics = data.get("metrics", {})
    cls_data = data.get("classification", {})

    patient_data = [
        [Paragraph("<b>Thong so benh nhan</b>", normal_style), Paragraph("<b>Gia tri lam sang</b>", normal_style)],
        ["Vung ton thuong (Rule of Nines):", str(metrics.get("body_part", "Chua xac dinh"))],
        ["Dien tich ton thuong (% Khung anh):", f"{metrics.get('pct_frame_area', 0)} %"],
        ["Dien tich thuc te uoc luong:", f"{metrics.get('area_cm2', 'Chua hieu chuan')} cm2" if metrics.get('area_cm2') else "Chua hieu chuan thuoc"],
        ["Uoc tinh % TBSA co the:", f"{metrics.get('estimated_tbsa_pct', 0)} %"],
        ["Chan doan cap do bong (ResNet-34):", f"{cls_data.get('predicted_degree', 'N/A')} (Do tin cay: {cls_data.get('confidence', 0)}%)"]
    ]
    t1 = Table(patient_data, colWidths=[200, 320])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ededf2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0058bc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#c1c6d7')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    parkland = metrics.get("parkland_plan")
    if parkland:
        is_indicated = parkland.get("indicated", True)
        story.append(Paragraph("DANH GIA BU DICH CAP CUU (PARKLAND)", h2_style))
        if is_indicated:
            parkland_data = [
                ["Tong luong dich Ringer Lactate (24h):", f"{parkland.get('total_fluid_24h_ml', 0)} ml"],
                ["Luong truyen trong 8 gio dau (50%):", f"{parkland.get('first_8h_ml', 0)} ml (Toc do: {parkland.get('rate_first_8h_ml_h', 0)} ml/h)"],
                ["Luong truyen trong 16 gio tiep theo (50%):", f"{parkland.get('first_8h_ml', 0)} ml (Toc do: {parkland.get('rate_next_16h_ml_h', 0)} ml/h)"],
                ["Canh bao cap cuu (Triage):", "CHUYEN TUYEN VIEN BONG KHAN CAP!" if parkland.get('urgent_triage') else "Dieu tri noi tru / Tiep tuc theo doi"]
            ]
        else:
            is_deg3 = "3" in str(cls_data.get('predicted_degree', ''))
            parkland_data = [
                ["Chi dinh truyen dich tinh mach:", "KHONG CHI DINH (0 ml) - Bu dich duong uong"],
                ["Danh gia nguy co soc mat dich:", "Khong co nguy co soc mat dich (TBSA < 10%)"],
                ["Huong xu tri lam sang:", "Chuyen khoa Ngoai cat loc mo hoai tu (Debridement)" if is_deg3 else "Sat trung va cham soc vet thuong tai cho"],
                ["Canh bao cap cuu (Triage):", "KHAM CHUYEN KHOA NGOAI BONG (CAT LOC)" if is_deg3 else "Theo doi & cham soc tai cho"]
            ]
        t2 = Table(parkland_data, colWidths=[220, 300])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f9f9fe')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#c1c6d7')),
            ('TEXTCOLOR', (1,3), (1,3), colors.HexColor('#bb0011') if (parkland.get('urgent_triage') or (not is_indicated and is_deg3)) else colors.HexColor('#006e28')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t2)
        story.append(Spacer(1, 15))

    story.append(Paragraph("HINH ANH PHAN TICH THI GIAC & XAI", h2_style))
    img_row = []
    if os.path.exists(original_img_path):
        img_row.append(RLImage(original_img_path, width=160, height=120))
    if os.path.exists(overlay_img_path):
        img_row.append(RLImage(overlay_img_path, width=160, height=120))
    if gradcam_img_path and os.path.exists(gradcam_img_path):
        img_row.append(RLImage(gradcam_img_path, width=160, height=120))

    if img_row:
        t_imgs = Table([img_row])
        t_imgs.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_imgs)
        story.append(Spacer(1, 15))

    disclaimer = Paragraph(
        "<i>Khuyen cao mien tru trach nhiem: Ket qua phan tich do he thong tri tue nhan tao BurnScan ho tro mang tinh tham khao hoc thuat cho mon CS406. Bac si lam sang can tham kham truc tiep truoc khi ra quyet dinh dieu tri.</i>",
        subtitle_style
    )
    story.append(disclaimer)

    doc.build(story)
    return output_pdf_path

