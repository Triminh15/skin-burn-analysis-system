"""
Script tạo các biểu đồ trực quan chất lượng cao phục vụ Báo cáo Đồ án CS406
(Bao gồm Sơ đồ Kiến trúc Pipeline Toàn diện có Khối Tiền Xử Lý Đầu Vào)
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
# 3. BIỂU ĐỒ SƠ ĐỒ HỆ THỐNG KIẾN TRÚC TỔNG THỂ CÓ TIỀN XỬ LÝ (SYSTEM ARCHITECTURE)
# ------------------------------------------------------------------------------
def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    ax.axis('off')

    # Định nghĩa 7 khối trong luồng kiến trúc liên hoàn
    boxes = [
        {"text": "1. Ảnh Đầu Vào Thô\n(Ảnh da tổn thương)\n[Kích thước bất kỳ]", "pos": (0.08, 0.70), "color": "#f1f5f9", "border": "#475569"},
        {"text": "2. Tiền Xử Lý Chuẩn Hóa\n• Đổi màu BGR -> RGB\n• Resize chuẩn 256x256\n• ImageNet Normalize", "pos": (0.28, 0.70), "color": "#e0f2fe", "border": "#0284c7"},
        {"text": "3. Giai đoạn 1: Phân đoạn\n• U-Net (ResNet-34)\n• Xuất Mặt nạ nhị phân\n• Định lượng % TBSA", "pos": (0.50, 0.70), "color": "#dbeafe", "border": "#2563eb"},
        {"text": "4. Hậu xử lý & Cắt ROI\n• Lọc nhiễu Morphology\n• Mở rộng 15% đệm an toàn\n• Đệm khung vuông 224x224", "pos": (0.72, 0.70), "color": "#fef3c7", "border": "#d97706"},
        {"text": "5. Giai đoạn 2: Phân loại\n• ResNet-34 Classifier\n• Phân bố Softmax 3 lớp\n[Độ 1 / Độ 2 / Độ 3]", "pos": (0.92, 0.70), "color": "#fce7f3", "border": "#db2777"},
        {"text": "6. Giai đoạn 3: Minh Bạch AI (XAI)\n• Grad-CAM Heatmap tầng layer4\n• Trực quan hóa vùng chú ý lâm sàng\n[Bản đồ nhiệt minh bạch chẩn đoán]", "pos": (0.75, 0.22), "color": "#fee2e2", "border": "#dc2626"},
        {"text": "7. Giai đoạn 4: Chatbot RAG Y tế Lâm Sàng\n• Cơ sở tri thức QĐ 4242/QĐ-BYT Bộ Y tế\n• Tự động tính bù dịch Parkland: V = 4*m*%TBSA\n[Phác đồ truyền dịch 24h & Cảnh báo sốc]", "pos": (0.35, 0.22), "color": "#d1fae5", "border": "#059669"}
    ]

    for b in boxes:
        bbox_props = dict(boxstyle="round,pad=0.55", facecolor=b["color"], edgecolor=b["border"], linewidth=2)
        ax.text(b["pos"][0], b["pos"][1], b["text"], ha="center", va="center", fontsize=9.2, fontweight="bold", bbox=bbox_props)

    # Mũi tên kết nối luồng
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2, color="#334155")

    # 1 -> 2
    ax.annotate("", xy=(0.20, 0.70), xytext=(0.15, 0.70), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(0.42, 0.70), xytext=(0.36, 0.70), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(0.63, 0.70), xytext=(0.58, 0.70), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate("", xy=(0.84, 0.70), xytext=(0.80, 0.70), arrowprops=arrow_props)

    # 4 & 5 -> 6 (Cắt ROI và ResNet nạp vào Grad-CAM)
    ax.annotate("", xy=(0.75, 0.36), xytext=(0.75, 0.56), arrowprops=arrow_props)
    ax.annotate("", xy=(0.83, 0.36), xytext=(0.90, 0.56), arrowprops=arrow_props)

    # 3 & 5 -> 7 (% TBSA từ U-Net và Cấp độ bỏng từ ResNet nạp vào RAG Parkland)
    ax.annotate("", xy=(0.35, 0.36), xytext=(0.45, 0.56), arrowprops=arrow_props)
    ax.annotate("", xy=(0.45, 0.36), xytext=(0.85, 0.56), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", lw=1.5, color="#64748b", linestyle="--"))

    plt.title("Hình 4.1: Sơ đồ kiến trúc tổng thể Hệ thống Phân tích Vết bỏng Đa phương thái (Skin Burn Analysis System)",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save_path = "outputs/figures/chart_pipeline_architecture.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu: {save_path}")

if __name__ == "__main__":
    plot_seg_comparison()
    plot_cls_comparison()
    plot_pipeline_architecture()
