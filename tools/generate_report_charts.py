"""
Script tạo các biểu đồ trực quan chất lượng cao phục vụ Báo cáo Đồ án CS406
"""
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("outputs/figures", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# 1. BIỂU ĐỒ SO SÁNH 4 MÔ HÌNH PHÂN ĐOẠN (SEGMENTATION)
# ------------------------------------------------------------------------------
def plot_seg_comparison():
    models = [
        "Run 1\nU-Net ResNet-34\n(ComboLoss)",
        "Run 2 (Champion)\nU-Net ResNet-34\n(Focal Tversky)",
        "Run 3\nU-Net ConvNeXt\n(Focal Tversky)",
        "Run 4\nAtt U-Net scSE\n(Focal Tversky)"
    ]
    dice = [65.19, 65.07, 63.60, 61.57]
    iou = [48.36, 48.23, 46.63, 44.48]
    pixel_acc = [86.41, 86.72, 85.90, 85.12]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width, dice, width, label='Test Dice (%)', color='#2563eb', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, iou, width, label='Test IoU (%)', color='#059669', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, pixel_acc, width, label='Pixel Accuracy (%)', color='#d97706', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hình 6.1: So sánh hiệu năng 4 cấu hình Phân đoạn vết bỏng (Giai đoạn 1)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.set_ylim(0, 100)

    # Hiển thị nhãn giá trị trên từng cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    save_path = "outputs/figures/chart_seg_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu: {save_path}")

# ------------------------------------------------------------------------------
# 2. BIỂU ĐỒ SO SÁNH 3 MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)
# ------------------------------------------------------------------------------
def plot_cls_comparison():
    models = [
        "Run 5 (Production)\nResNet-34\n(Weighted CE)",
        "Run 6\nResNet-34\n(CB-Focal Loss)",
        "Run 7 (Triage Hero)\nConvNeXt-Tiny\n(CB-Focal Loss)"
    ]
    acc = [79.14, 74.33, 65.24]
    macro_f1 = [79.08, 73.58, 65.73]
    recall_deg3 = [79.41, 64.71, 85.29]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rects1 = ax.bar(x - width, acc, width, label='Overall Accuracy (%)', color='#1d4ed8', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, macro_f1, width, label='Macro F1-Score (%)', color='#7c3aed', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, recall_deg3, width, label='Recall Bỏng Độ 3 (Ca nặng) (%)', color='#dc2626', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hình 6.2: So sánh hiệu năng 3 mô hình Phân loại cấp độ bỏng (Giai đoạn 2)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.set_ylim(0, 100)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    save_path = "outputs/figures/chart_cls_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu: {save_path}")

# ------------------------------------------------------------------------------
# 3. BIỂU ĐỒ SƠ ĐỒ HỆ THỐNG KIẾN TRÚC TỔNG THỂ (SYSTEM ARCHITECTURE PIPELINE)
# ------------------------------------------------------------------------------
def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')

    # Định nghĩa các khối hình chữ nhật trong luồng xử lý
    boxes = [
        {"text": "1. Ảnh Đầu Vào Thô\n(Ảnh da tổn thương)\n[Kích thước bất kỳ]", "pos": (0.08, 0.5), "color": "#e0f2fe", "border": "#0284c7"},
        {"text": "2. Giai đoạn 1:\nPhân đoạn (U-Net)\nResNet-34 Backbone\n[Mask 0/1 & % TBSA]", "pos": (0.32, 0.5), "color": "#dbeafe", "border": "#2563eb"},
        {"text": "3. Hậu xử lý & Cắt ROI\nLọc hình thái học\nThêm viền an toàn 15%\n[Ảnh ROI 224x224]", "pos": (0.55, 0.5), "color": "#fef3c7", "border": "#d97706"},
        {"text": "4. Giai đoạn 2:\nPhân loại (ResNet-34)\n3 Cấp độ lâm sàng\n[Độ 1 / Độ 2 / Độ 3]", "pos": (0.78, 0.72), "color": "#fce7f3", "border": "#db2777"},
        {"text": "5. Giai đoạn 3:\nGrad-CAM Heatmap\nMinh bạch AI y tế\n[Bản đồ nhiệt vùng chú ý]", "pos": (0.78, 0.28), "color": "#fee2e2", "border": "#dc2626"},
        {"text": "6. Giai đoạn 4: Chatbot RAG Y tế\nPhác đồ Bộ Y tế + Parkland Formula\n[Thể tích dịch bù V = 4*m*%TBSA]", "pos": (0.50, 0.08), "color": "#d1fae5", "border": "#059669"}
    ]

    for b in boxes:
        bbox_props = dict(boxstyle="round,pad=0.6", facecolor=b["color"], edgecolor=b["border"], linewidth=2)
        ax.text(b["pos"][0], b["pos"][1], b["text"], ha="center", va="center", fontsize=9.5, fontweight="bold", bbox=bbox_props)

    # Vẽ mũi tên kết nối
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2, color="#475569")
    # 1 -> 2
    ax.annotate("", xy=(0.23, 0.5), xytext=(0.17, 0.5), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(0.46, 0.5), xytext=(0.41, 0.5), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(0.69, 0.70), xytext=(0.64, 0.55), arrowprops=arrow_props)
    # 3 -> 5
    ax.annotate("", xy=(0.69, 0.30), xytext=(0.64, 0.45), arrowprops=arrow_props)
    # 2 & 4 -> 6
    ax.annotate("", xy=(0.42, 0.15), xytext=(0.35, 0.40), arrowprops=arrow_props)
    ax.annotate("", xy=(0.60, 0.15), xytext=(0.75, 0.20), arrowprops=arrow_props)

    plt.title("Hình 4.1: Sơ đồ kiến trúc tổng thể Hệ thống Phân tích Vết bỏng Đa phương thái (Skin Burn Analysis System)",
              fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    save_path = "outputs/figures/chart_pipeline_architecture.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu: {save_path}")

if __name__ == "__main__":
    plot_seg_comparison()
    plot_cls_comparison()
    plot_pipeline_architecture()
