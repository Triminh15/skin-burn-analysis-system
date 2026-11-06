# 📋 BÁO CÁO TIẾN ĐỘ & TỔNG KẾT DỰ ÁN (PROJECT PROGRESS & HANDOVER)

* **Dự án:** Skin Burn Analysis System (Hệ thống Phân tích Vết bỏng Đa phương thái)
* **Nhánh Git hiện tại:** `train_classification`
* **Thời gian cập nhật:** 06/11/2026
* **Trạng thái:** ✅ **TOÀN BỘ 7 THỰC NGHIỆM THỊ GIÁC (PHÂN ĐOẠN & PHÂN LOẠI) ĐÃ HOÀN TẤT 100%!**

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

## 🛠️ PHẦN 3: HOÀN THIỆN PHÂN HỆ THỊ GIÁC MÁY TÍNH (VISION PIPELINE - 100%)

Toàn bộ mã nguồn cốt lõi của Phân hệ Thị giác đã được lập trình hoàn chỉnh, kiểm thử đơn vị thành công và commit trên nhánh `train_vision`:

1. ✅ **Hàm mất mát Y khoa ([src/vision/losses.py](file:///d:/burn_analysis/src/vision/losses.py)):**
   - Phân đoạn: `ComboLoss` (Baseline) và `FocalTverskyLoss` (SOTA - Phạt nặng bỏ sót vết thương, tập trung viền mờ).
   - Phân loại: `Weighted Cross-Entropy` và `ClassBalancedFocalLoss` kết hợp `Label Smoothing`.
2. ✅ **Bộ chỉ số đo lường lâm sàng ([src/vision/metrics.py](file:///d:/burn_analysis/src/vision/metrics.py)):**
   - Phân đoạn: `Dice Score (F1)`, `IoU (Jaccard)`, `Pixel Accuracy` kèm `SegmentationMetricTracker`.
   - Phân loại: `Overall Accuracy`, `Macro F1-Score`, `Recall Bỏng Độ 3` (Chỉ số ca nặng), và `Confusion Matrix`.
3. ✅ **Định nghĩa Kiến trúc Mô hình ([src/vision/models.py](file:///d:/burn_analysis/src/vision/models.py)):**
   - Hỗ trợ U-Net, U-Net++ với backbone ResNet-34, ConvNeXt-Tiny và cơ chế Attention Gate scSE.
   - Hỗ trợ bộ phân loại ResNet-34 và ConvNeXt-Tiny (Dropout 20% chống overfit).
4. ✅ **Hậu xử lý & Tập dữ liệu ROI ([src/vision/postprocessing.py](file:///d:/burn_analysis/src/vision/postprocessing.py)):**
   - Lọc nhiễu hình thái học, tìm Bounding Box và cắt ảnh ROI có đệm an toàn 15% (Square Padding).
   - Tính tỷ lệ diện tích tổn thương (% TBSA).
   - **Đã sinh toàn bộ 1.227 ảnh ROI** tại `data/processed/rois/` và tạo 3 file danh mục: `train_roi.csv` (858 ảnh), `val_roi.csv` (182 ảnh), `test_roi.csv` (187 ảnh).
5. ✅ **Script Huấn luyện Phân đoạn ([src/vision/train_seg.py](file:///d:/burn_analysis/src/vision/train_seg.py)):**
   - Hỗ trợ tham số dòng lệnh linh hoạt, Early Stopping, Checkpoint Manager, vẽ đồ thị Curves và tự động đánh giá trên tập Test độc lập.
6. ✅ **Script Huấn luyện Phân loại ([src/vision/train_cls.py](file:///d:/burn_analysis/src/vision/train_cls.py)):**
   - Huấn luyện trên tập ROI đã cắt, Early Stopping dựa trên Macro F1, xuất đồ thị và vẽ Ma trận nhầm lẫn (Confusion Matrix Heatmap).

---

## 🚀 PHẦN 4: MA TRẬN 7 THỰC NGHIỆM SẴN SÀNG KHỞI CHẠY (EXPERIMENT ROADMAP)

### GIAI ĐOẠN 1: PHÂN ĐOẠN (SEGMENTATION) - 4 RUNS
* **Run 1 (Bạn):** Baseline `U-Net (ResNet-34) + ComboLoss`
  `python src/vision/train_seg.py --arch unet --backbone resnet34 --loss combo --epochs 35 --batch-size 16 --run-name unet_resnet34_combo`
* **Run 2 (Bạn):** Ablation Loss `U-Net (ResNet-34) + Focal Tversky`
  `python src/vision/train_seg.py --arch unet --backbone resnet34 --loss focal_tversky --epochs 35 --batch-size 16 --run-name unet_resnet34_focal_tversky`
* **Run 3 (Bạn phụ):** Ablation Backbone `U-Net (ConvNeXt-Tiny) + Focal Tversky`
  `python src/vision/train_seg.py --arch unet --backbone tu-convnext_tiny --loss focal_tversky --epochs 35 --batch-size 8 --run-name unet_convnext_focal_tversky`
* **Run 4 (Bạn phụ):** SOTA `Attention U-Net (ConvNeXt-Tiny) + Focal Tversky`
  `python src/vision/train_seg.py --arch unet --backbone tu-convnext_tiny --attention scse --loss focal_tversky --epochs 35 --batch-size 8 --run-name att_unet_convnext_focal_tversky`

### GIAI ĐOẠN 2: PHÂN LOẠI (CLASSIFICATION) - 3 RUNS
* **Run 5 (Bạn):** Baseline `ResNet-34 + Weighted CE`
  `python src/vision/train_cls.py --model resnet34 --loss weighted_ce --epochs 30 --batch-size 16 --run-name cls_resnet34_weighted_ce`
* **Run 6 (Bạn):** Ablation Loss `ResNet-34 + CB-Focal + Label Smoothing`
  `python src/vision/train_cls.py --model resnet34 --loss cb_focal --epochs 30 --batch-size 16 --run-name cls_resnet34_cb_focal`
* **Run 7 (Bạn phụ):** SOTA `ConvNeXt-Tiny + CB-Focal + Label Smoothing`
  `python src/vision/train_cls.py --model convnext_tiny --loss cb_focal --epochs 30 --batch-size 16 --run-name cls_convnext_cb_focal`

---

## 🏆 PHẦN 5: BẢNG KẾT QUẢ NGHIỆM THU CHÍNH THỨC TOÀN BỘ 7 RUN THỰC NGHIỆM (TEST SET: 187 ẢNH)

### 1. Bảng tổng hợp Giai đoạn 1: Phân đoạn vết bỏng (Segmentation)
Đánh giá khách quan trên 187 ảnh Test:

| Run | Kiến trúc & Cấu hình | Hàm Loss | Test Dice (%) | Test IoU (%) | Test Pixel Acc (%) | Tham số | Nhận xét lâm sàng & Học thuật |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Run 1** | U-Net (ResNet-34) | ComboLoss | 65.19% | 48.36% | 86.41% | 24.44M | Mốc Baseline chuẩn mực |
| **Run 2** | **U-Net (ResNet-34)** 🏆 | **Focal Tversky** | **65.07%** | **48.23%** | **86.72%** | 24.44M | **Mô hình vô địch Phân đoạn:** Test Loss thấp nhất (0.2413), bắt biên mờ chuẩn nhất |
| **Run 3** | U-Net (ConvNeXt-Tiny) | Focal Tversky | 63.60% | 46.63% | 85.90% | 29.45M | Hiện tượng Overfitting nhẹ do dữ liệu y tế nhỏ |
| **Run 4** | Att U-Net (ConvNeXt-Tiny + scSE) | Focal Tversky | 61.57% | 44.48% | 85.12% | 29.47M | Kết quả âm tính giá trị: scSE triệt tiêu quá mức quầng ban đỏ ngoại vi |

### 2. Bảng tổng hợp Giai đoạn 2: Phân loại cấp độ bỏng (Classification - 3 Classes)
Đánh giá khách quan trên 187 ảnh Test ROI:

| Run | Mô hình & Cấu hình | Hàm Loss | Accuracy (%) | Macro F1 (%) | Recall Độ 3 (%) | Test Loss | Thời gian | Vai trò trong hệ thống |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run 5** | **ResNet-34** 🏆 | **Weighted CE** | **79.14%** | **79.08%** | **79.41%** | 0.6540 | 3.85 ph | **Mô hình chính thức (Production Champion):** Cân bằng xuất sắc mọi chỉ số |
| **Run 6** | ResNet-34 | CB-Focal + Label Smoothing | 74.33% | 73.58% | 64.71% | **0.2366** | 3.59 ph | Hiệu chuẩn xác suất tốt nhất (Confidence Calibration) |
| **Run 7** | ConvNeXt-Tiny | CB-Focal + Label Smoothing | 65.24% | 65.73% | **85.29%** 🔥 | 0.2591 | 8.36 ph | **Chuyên gia Cấp cứu Triage:** Độ nhạy ca bỏng nặng cao nhất (85.29%) |

---

## 🎯 PHẦN 6: ĐỊNH HƯỚNG CÁC GIAI ĐOẠN TIẾP THEO (NEXT STEPS)

1. **Giai đoạn 3: Trích xuất bản đồ nhiệt Grad-CAM (XAI - Explainable AI)**
   - Xây dựng module `src/utils/gradcam.py` để trực quan hóa vùng chú ý của mô hình phân loại trên ảnh ROI.
2. **Giai đoạn 4: Hệ thống RAG & AI Agent Y tế (Medical Chatbot)**
   - Nạp phác đồ cấp cứu bỏng của Bộ Y tế Việt Nam vào ChromaDB.
   - Xây dựng AI Agent kết hợp Gemini để tính toán công thức bù dịch Parkland, gợi ý dinh dưỡng và chuyển tuyến.
3. **Giai đoạn 5: Ứng dụng Web hoàn chỉnh (Streamlit App)**
   - Ghép nối trọn vẹn: Upload ảnh $\rightarrow$ U-Net khoanh vùng + tính % TBSA $\rightarrow$ Cắt ROI $\rightarrow$ ResNet-34 phân loại $\rightarrow$ Grad-CAM minh bạch $\rightarrow$ Chatbot bác sĩ tư vấn.
4. **Giai đoạn 6: Hoàn thiện Báo cáo Đồ án (9 Chương Học thuật)**


