# 📋 BÁO CÁO TIẾN ĐỘ & TỔNG KẾT DỰ ÁN (PROJECT PROGRESS & HANDOVER)

* **Dự án:** Skin Burn Analysis System (Hệ thống Phân tích Vết bỏng Đa phương thái)
* **Nhánh Git hiện tại:** `clean_data`
* **Thời gian cập nhật:** 24/10/2026
* **Trạng thái:** ✅ **Giai đoạn Dữ liệu (Data Engineering & Augmentation) đã hoàn tất 100%**

---

## 📌 PHẦN 1: CÁC VẤN ĐỀ CỐT LÕI ĐÃ TRAO ĐỔI & THỐNG NHẤT

### 1. Đánh giá và phát hiện bất thường trên bộ dữ liệu gốc (`data/raw/`)
* **Hiện trạng ban đầu:** Thư mục `Skin Burn Dataset` chứa **1.227 file ảnh** (`.jpg`, `.jpeg`) nhưng lại có tới **1.441 file nhãn** (`.txt`).
* **Giải mã nguyên nhân:**
  * Có **214 file `.txt` bị mồ côi** (không có file ảnh đi kèm). Các file này được tự động loại bỏ để tránh lỗi nạp dữ liệu.
  * Toàn bộ **1.227 ảnh hợp lệ** đều có đầy đủ nhãn tương ứng.
  * Trong 1.227 ảnh có tổng cộng **2.219 vết bỏng** (trung bình 1 ảnh có thể chứa nhiều vết bỏng rải rác, ví dụ `img1.jpg` có 4 vết bỏng).
* **Bản chất nhãn thô:** File `.txt` có định dạng 5 con số `<class> <cx> <cy> <w> <h>` $\rightarrow$ Đây là **Bounding Box (Hộp chữ nhật - Object Detection)**, KHÔNG PHẢI là **Mask chuẩn Pixel (Semantic Segmentation)**.

### 2. Tại sao Bounding Box không dùng trực tiếp được cho U-Net?
* Bounding Box là khung chữ nhật, 4 góc chứa rất nhiều da lành. Nếu ép thành mask thô, U-Net sẽ học cách dự đoán... hình hộp chữ nhật thay vì bám theo đường cong vết thương.
* Diện tích tổn thương sẽ bị thổi phồng lên 30%–50%, làm sai lệch phác đồ cấp cứu truyền dịch y khoa (Parkland formula).

### 3. Thống nhất kiến trúc: Chọn "Hướng B"
* **Quyết định:** Giữ vững kiến trúc **U-Net + ResNet-50** như thiết kế ban đầu để đảm bảo tính chuẩn mực y khoa và giá trị học thuật của đồ án.
* **Giải pháp đột phá:** Áp dụng kỹ thuật *Weakly-supervised Segmentation* bằng cách sử dụng **MobileSAM (Segment Anything Model)** nhận Bounding Box làm gợi ý (Box Prompt) để **tự động bóc tách da lành, sinh ra Mask chuẩn từng pixel**.

---

## 🛠️ PHẦN 2: NHỮNG CÔNG VIỆC ĐÃ HOÀN THÀNH

### 1. Quản lý Thư viện (`requirements.txt`)
* Đã cấu trúc và ghi đầy đủ danh sách thư viện cho toàn bộ dự án vào [requirements.txt](file:///d:/burn_analysis/requirements.txt):
  * **Deep Learning:** `torch`, `torchvision`, `segmentation-models-pytorch`, `albumentations`, `ultralytics`.
  * **Xử lý số liệu:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`.
  * **Medical RAG & AI Agent:** `chromadb`, `sentence-transformers`, `rank-bm25`, `pypdf`, `google-generativeai`, `python-dotenv`.
  * **XAI & Bảo mật:** `grad-cam`, `piexif`, `streamlit`, `pytest`.
* Đã cài đặt thành công các gói nền tảng: `ultralytics`, `segmentation-models-pytorch`, `albumentations`, `scikit-learn`.

### 2. Xây dựng Script sinh Mask tự động (`tools/generate_masks.py`)
* Tạo script [tools/generate_masks.py](file:///d:/burn_analysis/tools/generate_masks.py) với các tính năng chuyên nghiệp:
  * Đọc toạ độ YOLO và chuyển đổi sang toạ độ pixel thực tế.
  * Điều phối MobileSAM nhận Box Prompt để sinh mask nhị phân (0: da lành, 255: vết bỏng).
  * Hợp nhất nhiều vết bỏng trên cùng một bức ảnh bằng phép toán logic OR.
  * Tối ưu hóa đa luồng CPU Intel Core 5, chuẩn hóa `imgsz=512` giúp tốc độ xử lý nhanh gấp 3 lần (~1s/ảnh).
  * Tích hợp cơ chế **Resume Mode** (tự động bỏ qua các ảnh đã xử lý nếu bị ngắt quãng).
* **Kết quả nghiệm thu:** Ảnh so sánh trực quan đã được lưu tại [outputs/figures/sam_mask_demo.png](file:///d:/burn_analysis/outputs/figures/sam_mask_demo.png). Mask ôm khít viền tổn thương, loại bỏ hoàn toàn da lành ở 4 góc.

### 3. Hoàn tất xử lý toàn bộ 1.227 ảnh và Mask
* **1.227 ảnh chuẩn hóa** đã được lưu vào [data/processed/images/](file:///d:/burn_analysis/data/processed/images/).
* **1.227 mask nhị phân chuẩn pixel** đã được lưu vào [data/processed/masks/](file:///d:/burn_analysis/data/processed/masks/).
* **Bảng tổng hợp siêu dữ liệu y khoa** đã được tạo tại [data/splits/metadata.csv](file:///d:/burn_analysis/data/splits/metadata.csv) (ghi nhận kích thước, số vết bỏng, độ nặng nhất, và **% diện tích tổn thương da**).

### 4. Phân tầng tập dữ liệu chuẩn mực (Stratified Splits)
Đã chia tập dữ liệu theo tỷ lệ **70% Train – 15% Validation – 15% Test**, phân tầng đồng đều tỷ lệ các cấp độ bỏng (Độ 1: ~42%, Độ 2: ~40%, Độ 3: ~18%):
* [data/splits/train.csv](file:///d:/burn_analysis/data/splits/train.csv): **858 ảnh** (359 Độ 1, 346 Độ 2, 153 Độ 3)
* [data/splits/val.csv](file:///d:/burn_analysis/data/splits/val.csv): **182 ảnh** (76 Độ 1, 74 Độ 2, 32 Độ 3)
* [data/splits/test.csv](file:///d:/burn_analysis/data/splits/test.csv): **187 ảnh** (78 Độ 1, 75 Độ 2, 34 Độ 3)

### 5. Xây dựng Pipeline Làm giàu dữ liệu (`src/vision/transforms.py`)
* File [src/vision/transforms.py](file:///d:/burn_analysis/src/vision/transforms.py) chuẩn Albumentations 2.0+:
  * **Train transforms:** Lật ngang/dọc, biến đổi Affine (xoay, co giãn mô da), chỉnh sáng tối `ColorJitter`, làm mờ `GaussianBlur`, nhiễu `GaussNoise`, chuẩn hóa ImageNet.
  * **Val/Test transforms:** Chỉ Resize (256x256) và Chuẩn hóa, tuyệt đối không làm méo hình.
  * Hàm `denormalize` hỗ trợ vẽ lại ảnh RGB từ tensor.

### 6. Xây dựng PyTorch Dataset & DataLoader (`src/vision/dataset.py`)
* File [src/vision/dataset.py](file:///d:/burn_analysis/src/vision/dataset.py):
  * Class `SkinBurnDataset` nạp ảnh và mask từ file CSV, đảm bảo biến đổi hình học diễn ra **đồng thời trên cả Ảnh và Mask**.
  * Chuyển Mask về Tensor nhị phân dạng float32 `(1, 256, 256)`.
  * Hàm `get_dataloaders()` gom batch tự động.
* **Kiểm thử thành công:** Đã nạp thử 1 batch từ `train_loader` (`torch.Size([4, 3, 256, 256])`) và xuất ảnh nghiệm thu tại [outputs/figures/augmentation_demo.png](file:///d:/burn_analysis/outputs/figures/augmentation_demo.png).

---

## 🚀 PHẦN 3: CÔNG VIỆC TIẾP THEO CẦN LÀM (KHI QUAY LẠI LÀM TIẾP)

Khi bắt đầu phiên làm việc tiếp theo, chúng ta sẽ bước ngay vào **Giai đoạn Huấn luyện Mô hình Thị giác (Bước 4)**:

1. **Xây dựng Hàm mất mát y khoa (`src/vision/losses.py`):**
   * Hiện thực hóa `ComboLoss = 0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss`.
   * Giúp mô hình vừa học xác suất từng điểm ảnh (BCE), vừa tối ưu độ trùng khớp diện tích (Dice) chống mất cân bằng giữa vùng bỏng và nền.
2. **Xây dựng Bộ chỉ số đo lường (`src/vision/metrics.py`):**
   * Lập trình tính toán **Dice Score (F1)** và **IoU (Intersection over Union / Jaccard Index)**.
3. **Khởi tạo Mô hình U-Net (`src/vision/models.py`):**
   * Định nghĩa kiến trúc `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)`.
   * Khởi tạo khung phân loại `ResNet-50 Classifier` cho bài toán xếp độ bỏng sau này.
4. **Viết Script huấn luyện U-Net (`src/vision/train_seg.py`):**
   * Xây dựng Training Loop chuẩn: Optimizer (AdamW), Learning Rate Scheduler (CosineAnnealingLR).
   * Cơ chế **Early Stopping** và **Model Checkpoint Manager** (tự động lưu trọng số tốt nhất vào `outputs/checkpoints/best_unet.pth`).
   * Xuất đồ thị Loss và Dice curves sau khi train vào `outputs/figures/`.
5. **Tiền xử lý ROI (`src/vision/postprocessing.py`):**
   * Dùng mask dự đoán của U-Net để crop chính xác vùng vết thương kèm padding, loại bỏ 100% da lành để đưa vào ResNet phân loại.
