# ĐẶC TẢ KIẾN TRÚC NGHIÊN CỨU & LỘ TRÌNH THỰC NGHIỆM
# HỆ THỐNG PHÂN TÍCH VẾT BỎNG DA (SKIN BURN ANALYSIS SYSTEM)
## Từ Kiến Trúc Hai Giai Đoạn (Baseline) Đến Mạng Đa Nhiệm Nhận Thức Ban Đỏ Tin Cậy (`MT-EANet-EDL`)

---

> **Mục đích tài liệu:**  
> Tài liệu này tổng hợp toàn bộ cơ sở lý thuyết, phân tích toán học, kiến trúc mô hình, phát hiện dữ liệu và lộ trình thực nghiệm đã được thảo luận và thống nhất.  
> Tài liệu được biên soạn chi tiết, chuẩn hóa công thức toán học và cấu trúc dữ liệu để:
> 1. **Kỹ sư / Thành viên nhóm** đọc, hiểu sâu bản chất lý do thiết kế và phương pháp luận nghiên cứu.
> 2. **AI Agent / LLM của đồng đội** có thể nạp trực tiếp làm ngữ cảnh (Context / Prompt) để sinh mã nguồn (Code Generation) chính xác 100%, không bị sai lệch cấu trúc hay nhầm lẫn tensor.

---

## MỤC LỤC
1. [Bối Cảnh & Mục Tiêu Học Thuật (Context & Academic Goals)](#1-bối-cảnh--mục-tiêu-học-thuật)
2. [Phân Tích So Sánh Kiến Trúc: Hai Giai Đoạn vs. Multi-Task](#2-phân-tích-so-sánh-kiến-trúc-hai-giai-đoạn-vs-multi-task)
3. [Giải Mã Toán Học Tham Số: Tại Sao U-Net Decoder Chỉ Có 3.15M?](#3-giải-mã-toán-học-tham-số-tại-sao-u-net-decoder-chỉ-có-315m)
4. [Phát Hiện Dữ Liệu: Sự Lệch Pha Huấn Luyện (Train-Test Domain Discrepancy)](#4-phát-hiện-dữ-liệu-sự-lệch-pha-huấn-luyện-train-test-discrepancy)
5. [Đột Phá Đạo Đức Y Tế: Evidential Deep Learning (EDL)](#5-đột-phá-đạo-đức-y-tế-evidential-deep-learning-edl)
6. [Nâng Cấp Phân Đoạn Ban Đỏ: Erythema-Aware Dual-Branch Decoder](#6-nâng-cấp-phân-đoạn-ban-đỏ-erythema-aware-dual-branch-decoder)
7. [Hàm Mất Mát Toàn Diện Đa Nhiệm (Comprehensive Multi-Task Joint Loss)](#7-hàm-mất-mát-toàn-diện-đa-nhiệm-comprehensive-multi-task-joint-loss)
8. [Kịch Bản Thực Nghiệm Tiệm Tiến (Experimental Roadmap & Decision Tree)](#8-kịch-bản-thực-nghiệm-tiệm-tiến-experimental-roadmap--decision-tree)
9. [Đặc Tả Kỹ Thuật Dành Cho AI Code Generation (Implementation Blueprint)](#9-đặc-tả-kỹ-thuật-dành-cho-ai-code-generation-implementation-blueprint)

---

## 1. BỐI CẢNH & MỤC TIÊU HỌC THUẬT

- **Dự án gốc:** Hệ thống hỗ trợ chẩn đoán tổn thương da do bỏng (Course project CS406 - Computer Vision).
- **Mục tiêu nâng cấp:** Chuyển hóa toàn diện giải pháp thành một công trình nghiên cứu khoa học chuẩn mực, hướng đến xuất bản tại các tạp chí y sinh quốc tế thuộc danh mục **Scopus / ISI Q1/Q2** (ví dụ: *Computers in Biology and Medicine*, *IEEE Journal of Biomedical and Health Informatics*, *Burns*).
- **Hai bài toán lâm sàng cốt lõi:**
  1. **Phân đoạn tổn thương (Segmentation):** Khoanh vùng chính xác ranh giới vết bỏng trên ảnh RGB để tính tỷ lệ diện tích bỏng (% TBSA).
  2. **Phân loại độ sâu bỏng (Classification):** Dự đoán mức độ nghiêm trọng gồm 3 cấp độ:
     - Độ 1 (Superficial / Erythema): Chỉ tổn thương biểu bì, đỏ rát.
     - Độ 2 (Partial Thickness): Bỏng trung bì, xuất hiện phỏng nước (blisters).
     - Độ 3 (Full Thickness): Bỏng sâu toàn phần, hoại tử trắng bợt hoặc cháy đen, mất cảm giác.
- **Thách thức thực tế chưa được giải quyết:**
  - *Thách thức 1 (Lâm sàng):* Ranh giới giữa vùng da lành và vùng bỏng luôn có **dải hồng ban (Erythema)** mờ nhạt. Các cơ chế Attention không gian thông thường (như scSE ở thí nghiệm Run 4) có xu hướng triệt tiêu dải hồng ban này, làm tụt chỉ số Dice.
  - *Thách thức 2 (Kỹ thuật):* Hệ thống 2 giai đoạn cồng kềnh, nhân đôi tham số và bị lỗi lan truyền (Error Propagation) từ khâu cắt ROI.
  - *Thách thức 3 (Đạo đức Y tế):* Hàm Softmax truyền thống luôn đưa ra dự đoán "tự tin thái quá" ngay cả trên các ca bệnh ranh giới hoặc ảnh nhiễu, vi phạm nguyên tắc an toàn y tế quốc tế (EU AI Act, WHO AI Ethics).

---

## 2. PHÂN TÍCH SO SÁNH KIẾN TRÚC: HAI GIAI ĐOẠN VS. MULTI-TASK

### 2.1. Kiến Trúc Hai Giai Đoạn (Two-Stage Pipeline - Hiện Tại)

```
[Ảnh gốc X (H, W, 3)]
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  MÔ HÌNH 1: PHÂN ĐOẠN (U-Net with ResNet-34 Backbone)        ║
║  - ResNet-34 Encoder: 21.28M tham số                          ║
║  - U-Net Decoder: 3.15M tham số                               ║
║  -> Tổng tham số Mô hình 1: 24.43M                           ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼ Xuất ra Mặt nạ nhị phân (Binary Mask M)
        │
        ▼
[Module Cắt ROI ngoại vi: Bounding Box bao quanh Mask + Padding 15%]
        │
        ▼ Xuất ra Ảnh cắt ROI (224 x 224 x 3)
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  MÔ HÌNH 2: PHÂN LOẠI CẤP ĐỘ BỎNG (Standalone Classifier)    ║
║  - ResNet-34 Feature Extractor: 21.28M tham số (MẠNG THỨ HAI) ║
║  - Linear Head (512 -> 3): 0.0015M tham số                    ║
║  -> Tổng tham số Mô hình 2: 21.28M                           ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
[Kết quả: Xác suất Độ 1 / Độ 2 / Độ 3]
```

- **Tổng tham số toàn hệ thống:** $24.43\text{M} + 21.28\text{M} = \mathbf{45.71M}$ tham số.
- **Nhược điểm lớn:**
  1. Phải nạp và lưu trữ 2 file trọng số `.pth` riêng biệt.
  2. Thời gian suy luận (Inference Latency) bị nhân đôi: ảnh phải đi qua backbone ResNet-34 hai lần riêng rẽ.
  3. Lỗi từ bước phân đoạn (cắt lẹm, mất viền) sẽ truyền thẳng vào phân loại làm sai lệch kết quả chẩn đoán.

---

### 2.2. Kiến Trúc Mạng Đa Nhiệm (Multi-Task Learning Network)

```
[Ảnh gốc X (H, W, 3)]
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  BỘ NÃO DÙNG CHUNG (SHARED ENCODER): ResNet-34 (21.28M)       ║
║  - Trích xuất đa tầng: conv1, layer1, layer2, layer3, layer4   ║
║  - Chỉ chạy đúng 1 lần duy nhất!                              ║
╚═══════════════════════════╤═══════════════════════════════════╝
                            │
            ┌───────────────┴────────────────┐
            │ Feature Maps                   │ Bottleneck Features
            │ (Skip Connections)             │ (layer4: 512 x 7 x 7)
            ▼                                ▼
╔═══════════════════════════════╗  ╔════════════════════════════════╗
║ NHÁNH 1: PHÂN ĐOẠN            ║  ║ NHÁNH 2: PHÂN LOẠI             ║
║ U-Net Decoder (3.15M)         ║  ║ Mask-Guided Attention Head     ║
║ (Upsampling + Skip Concat)    ║  ║ + Linear Classifier (~0.05M)   ║
╚═══════════════╤═══════════════╝  ╚════════════════╤═══════════════╝
                │                                   │
                ▼                                   ▼
        [Predicted Mask M]              [Độ Bỏng 1 / 2 / 3 + Uncertainty]
```

- **Tổng tham số toàn hệ thống:** $21.28\text{M} + 3.15\text{M} + 0.05\text{M} \approx \mathbf{24.48M}$ tham số.
- **Ưu thế vượt trội:**
  1. **Tiết kiệm 46.4% dung lượng tham số:** Cắt giảm hoàn toàn một con ResNet-34 dư thừa (21.28M tham số).
  2. **Tốc độ suy luận tăng gấp đôi:** Backbone chỉ chạy 1 lần.
  3. **Học biểu diễn tương hỗ (Task Synergy):** Gradient từ cả 2 bài toán cùng uốn nắn một Encoder duy nhất, giúp đặc trưng ngữ nghĩa trở nên khái quát và kháng nhiễu tốt hơn.

---

## 3. GIẢI MÃ TOÁN HỌC THAM SỐ: TẠI SAO U-NET DECODER CHỈ CÓ 3.15M?

Nhiều người lầm tưởng U-Net là mạng đối xứng nên Decoder phải nặng tương đương Encoder. Thực tế trong chuẩn công nghiệp (như `segmentation_models_pytorch`), **Decoder chỉ chiếm ~13% tổng số tham số**.

### 3.1. Bảng Chi Tiết Tham Số Từng Tầng Của U-Net Decoder
Công thức số tham số của một lớp Conv2D ($K \times K$):
$$\text{Params} = K \times K \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$$

| Tầng Decoder | Kích thước vào | Skip Connection từ Encoder | Số kênh ra | 2 Lớp Tích Chập Bên Trong | Số tham số |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** ($1/16 \rightarrow 1/8$) | 512 (`layer4`) | 256 (`layer3`) $\rightarrow C_{\text{in}} = 768$ | **256** | Conv1: $3 \times 3 \times 768 \times 256$<br>Conv2: $3 \times 3 \times 256 \times 256$ | **~2.36M** (75% decoder) |
| **Stage 2** ($1/8 \rightarrow 1/4$) | 256 | 128 (`layer2`) $\rightarrow C_{\text{in}} = 384$ | **128** | Conv1: $3 \times 3 \times 384 \times 128$<br>Conv2: $3 \times 3 \times 128 \times 128$ | **~0.59M** |
| **Stage 3** ($1/4 \rightarrow 1/2$) | 128 | 64 (`layer1`) $\rightarrow C_{\text{in}} = 192$ | **64** | Conv1: $3 \times 3 \times 192 \times 64$<br>Conv2: $3 \times 3 \times 64 \times 64$ | **~0.15M** |
| **Stage 4** ($1/2 \rightarrow 1/1$) | 64 | 64 (`conv1`) $\rightarrow C_{\text{in}} = 128$ | **32** | Conv1: $3 \times 3 \times 128 \times 32$<br>Conv2: $3 \times 3 \times 32 \times 32$ | **~0.046M** |
| **Stage 5** (Tinh chỉnh) | 32 | Không có skip $\rightarrow C_{\text{in}} = 32$ | **16** | Conv1: $3 \times 3 \times 32 \times 16$<br>Conv2: $3 \times 3 \times 16 \times 16$ | **~0.007M** |
| **Final Head** | 16 | Không | **1** | Conv: $1 \times 1 \times 16 \times 1$ | **17 tham số** |
| **TỔNG CỘNG DECODER** | | | | | $\mathbf{\approx 3.15M}$ |

### 3.2. Ba Nguyên Nhân Cốt Lõi
1. **Quy luật bậc hai theo số kênh ($O(C^2)$):** Càng lên các tầng nông, số kênh giảm mạnh ($512 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 16$), khiến chi phí tham số cho mỗi lớp tích chập tụt giảm hàng trăm lần.
2. **Số lượng tầng ít hơn:** Encoder ResNet-34 có tới 34 tầng tích chập sâu. Decoder U-Net chỉ có 5 khối, mỗi khối 2 tầng tích chập đơn giản (tổng cộng chỉ có 10 tầng tích chập).
3. **Upsampling có chi phí 0 tham số:** Phép phóng to không gian sử dụng toán tử nội suy song tuyến tính (Bilinear Interpolation) hoàn toàn không có tham số học.
4. **Triết lý Skip Connection:** Decoder không cần "học lại từ đầu" các chi tiết không gian (cạnh, góc, texture) vì các dây cáp Skip Connection đã dẫn truyền trực tiếp các đặc trưng độ phân giải cao từ Encoder sang.

---

## 4. PHÁT HIỆN DỮ LIỆU: SỰ LỆCH PHA HUẤN LUYỆN (TRAIN-TEST DISCREPANCY)

Trong quá trình rà soát mã nguồn thực tế tại `burn_analysis`, nhóm phát hiện một điểm mấu chốt về luồng dữ liệu:

### 4.1. Thực Trạng Hiện Tại
1. **Tạo Mask nhãn (Ground Truth):** Script `tools/generate_masks.py` sử dụng **MobileSAM (Segment Anything Model)** nhận box prompt từ bác sĩ để sinh ra các mask chuẩn lưu trong `data/processed/masks/`.
2. **Huấn luyện Phân đoạn (`train_seg.py`):** U-Net học từ ảnh gốc để tái tạo lại mask chuẩn của MobileSAM.
3. **Huấn luyện Phân loại (`train_cls.py`):** Script `postprocessing.py:generate_roi_dataset()` đã sử dụng **chính Mask chuẩn của MobileSAM** để cắt ra các ảnh ROI $224 \times 224$ (lưu trong `data/processed/rois/` và lập chỉ mục trong `train_roi.csv`).

### 4.2. Hệ Quả Khoa Học (Cascading Error)
- **Khi Huấn luyện (Train):** Classifier được nạp những bức ảnh ROI "hoàn hảo" (cắt từ SAM).
- **Khi Suy luận Thực tế (Inference trong `predict.py` / Web):** Không có SAM! Hệ thống buộc phải dùng U-Net để sinh mask, rồi lấy mask đó cắt ROI cho Classifier. Nếu U-Net dự đoán sai lệch hoặc cắt lẹm viền $\rightarrow$ Classifier nhận ảnh dị dạng và chẩn đoán sai hoàn toàn.
- **Giải pháp Multi-Task:** Loại bỏ hoàn toàn sự phụ thuộc vào bước cắt ROI trung gian. Nhánh phân loại đọc trực tiếp đặc trưng toàn cảnh từ Shared Encoder và dùng cơ chế chú ý không gian mềm (Soft Spatial Attention) dẫn đường bởi phân đoạn.

---

## 5. ĐỘT PHÁ ĐẠO ĐỨC Y TẾ: EVIDENTIAL DEEP LEARNING (EDL)

### 5.1. Vấn Đề Của Softmax & Cross-Entropy
- Hàm Softmax truyền thống chuẩn hóa đầu ra qua hàm mũ:
  $$p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \sum_{k=1}^K p_k = 1$$
- **Hạn chế nghiêm trọng:** Softmax luôn bị **tự tin thái quá (Overconfident)**. Với một ca bệnh mập mờ giữa Độ 2 và Độ 3, hoặc khi đưa vào ảnh không phải bỏng (vết chàm, vảy nến, ảnh nhiễu OOD), Softmax vẫn gán xác suất cao (ví dụ: 95% Độ 2), dẫn đến chỉ định sai phác đồ và gây nguy hiểm đến tính mạng bệnh nhân.

### 5.2. Nguyên Lý Toán Học Của EDL (Dựa Trên Phân Phối Dirichlet)
Thay vì xuất ra một phân phối xác suất cố định, EDL đặt một phân phối tiên nghiệm Dirichlet lên không gian xác suất:
$$D(\mathbf{p} \mid \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}$$

1. **Lượng Bằng Chứng (Evidence):**
   Mạng nơ-ron xuất ra logit $z_k$. Lượng bằng chứng thực nghiệm $e_k \ge 0$ được tính qua hàm kích hoạt không âm (thay vì Softmax):
   $$e_k = \text{Softplus}(z_k) = \ln(1 + e^{z_k})$$
2. **Tham số Tập trung Dirichlet (Dirichlet Parameters):**
   $$\alpha_k = e_k + 1$$
3. **Tổng Độ Mạnh Bằng Chứng (Dirichlet Strength):**
   $$S = \sum_{k=1}^K \alpha_k = \sum_{k=1}^K (e_k + 1)$$
4. **Xác suất Kỳ vọng (Expected Probability):**
   $$\hat{p}_k = \frac{\alpha_k}{S}$$
5. **Độ Bất Định Nhận Thức (Epistemic Uncertainty / Vacuity):**
   $$u = \frac{K}{S} = \frac{3}{\sum_{k=1}^3 \alpha_k} \in (0, 1]$$
   *(Tính chất bảo toàn: $\sum_{k=1}^K \hat{p}_k + u = 1$)*

```
Trường hợp A: Bằng chứng rất nhiều (e = [50, 1, 0]) -> S = 54 -> u = 3/54 = 0.05 (Rất tự tin)
Trường hợp B: Không có bằng chứng (e = [0.1, 0.1, 0.1]) -> S = 3.3 -> u = 3/3.3 = 0.91 (Cực kỳ bất định!)
```

### 5.3. Hàm Mất Mát EDL (Type II Maximum Likelihood + KL Regularization)
$$\mathcal{L}_{\text{EDL}}(\boldsymbol{\alpha}_i, \mathbf{y}_i) = \mathcal{L}_{\text{data}}(\boldsymbol{\alpha}_i, \mathbf{y}_i) + \lambda_t \mathcal{L}_{\text{KL}}(\boldsymbol{\alpha}_i \mid \mathbf{1})$$

Trong đó:
1. **Thành phần khớp dữ liệu (Expected Mean Squared Error Loss):**
   $$\mathcal{L}_{\text{data}}(\boldsymbol{\alpha}_i, \mathbf{y}_i) = \sum_{k=1}^K \left( y_{ik} - \frac{\alpha_{ik}}{S_i} \right)^2 + \frac{\alpha_{ik}(S_i - \alpha_{ik})}{S_i^2 (S_i + 1)}$$
2. **Thành phần phạt bằng chứng sai lệch (KL Divergence Regularization):**
   Phạt các bằng chứng sinh ra ở những lớp không phải Ground Truth, đưa phân phối Dirichlet về dạng đều (Uniform Prior $\boldsymbol{\alpha} = [1, 1, \dots, 1]$):
   $$\mathcal{L}_{\text{KL}}(\tilde{\boldsymbol{\alpha}}_i) = \ln \left( \frac{\Gamma\left(\sum_{k=1}^K \tilde{\alpha}_{ik}\right)}{\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_{ik})} \right) + \sum_{k=1}^K (\tilde{\alpha}_{ik} - 1) \left[ \psi(\tilde{\alpha}_{ik}) - \psi\left(\sum_{j=1}^K \tilde{\alpha}_{ij}\right) \right]$$
   với $\tilde{\boldsymbol{\alpha}}_i = \mathbf{y}_i + (1 - \mathbf{y}_i) \odot \boldsymbol{\alpha}_i$.
3. **Lịch trình Ủ Nhiệt (Annealing Schedule $\lambda_t$):**
   Để tránh việc thành phần KL làm triệt tiêu gradient ở những epoch đầu:
   $$\lambda_t = \min\left(1.0, \; \frac{t}{10}\right) \quad (\text{với } t \text{ là epoch hiện tại})$$

### 5.4. Quy Trình Phân Luồng Lâm Sàng Dựa Trên Độ Bất Định (Clinical Triage Protocol)
- **Nếu $u < 0.2$:** Mô hình tự tin $\rightarrow$ Tự động đưa ra dự đoán phân loại kèm phác đồ sơ cứu khuyến nghị.
- **Nếu $u \ge 0.4$:** Mô hình nhận định ca bệnh ranh giới hoặc bất thường $\rightarrow$ Kích hoạt cảnh báo đạo đức y tế: *"Hệ thống phát hiện độ bất định cao ($u = X\%$). Khuyến cáo bác sĩ chuyên khoa tiến hành hội chẩn trực tiếp, từ chối chẩn đoán tự động"*.

---

## 6. NÂNG CẤP PHÂN ĐOẠN BAN ĐỎ: ERYTHEMA-AWARE DUAL-BRANCH DECODER

Ở bài toán bỏng, tổn thương gồm 2 thành phần có đặc tính quang học trái ngược:
1. **Vùng lõi tổn thương (Necrotic / Central Core):** Hoại tử, đổi màu mạnh, mất cấu trúc vi mô.
2. **Vùng viền ban đỏ (Peripheral Erythema):** Dải đỏ mờ ở rìa do phản ứng giãn mạch, độ tương phản rất thấp với da lành.

```
                    ┌───────────────────────────────────────────────┐
                    │      Đặc trưng đa mức từ Shared ResNet-34     │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
    ╔═══════════════════════════════╗               ╔═══════════════════════════════╗
    ║  NHÁNH 1: CORE DECODER        ║               ║  NHÁNH 2: BOUNDARY DECODER    ║
    ║  (Dilated Convolutions)       ║               ║  (High-Freq Edge / Sobel)     ║
    ║  - Receptive field lớn        ║               ║  - Khai thác đạo hàm bậc nhất ║
    ║  - Chuyên bắt vùng trung tâm  ║               ║  - Chuyên bám viền ban đỏ mờ  ║
    ╚═══════════════╤═══════════════╝               ╚═══════════════╤═══════════════╝
                    │ F_core                                        │ F_boundary
                    └───────────────────────┬───────────────────────┘
                                            ▼
                            ╔═══════════════════════════════╗
                            ║      GATED FUSION MODULE      ║
                            ║  α = Sigmoid(Conv([Fc, Fb]))  ║
                            ║  F_fused = α·Fc + (1-α)·Fb    ║
                            ╚═══════════════╤═══════════════╝
                                            │
                                            ▼
                                [Mask Dự Đoán Hoàn Chỉnh]
```

- **Nhánh Viền (Lightweight Boundary Branch):** Sử dụng các bộ lọc Gradient không gian khả vi (Differentiable Sobel Filter) kết hợp với các tầng tích chập $1 \times 1$ và $3 \times 3$ nhẹ.
- **Giám sát trực tiếp viền (Boundary Supervision):** Sinh ground-truth viền nhị phân từ mask bằng phép toán hình thái học (Morphological Dilation - Erosion):
  $$B_{\text{GT}} = \text{Dilation}(M_{\text{GT}}) \ominus \text{Erosion}(M_{\text{GT}})$$

---

## 7. HÀM MẤT MÁT TOÀN DIỆN ĐA NHIỆM (COMPREHENSIVE MULTI-TASK JOINT LOSS)

Trong mô hình đa nhiệm, **mỗi nhánh tính toán hàm loss riêng**, sau đó được tổng hợp lại thành $\mathcal{L}_{\text{total}}$ để thực hiện lan truyền ngược (Backpropagation) duy nhất 1 lần qua Shared Encoder:

$$\mathcal{L}_{\text{total}} = w_{\text{seg}} \mathcal{L}_{\text{seg}} + w_{\text{bnd}} \mathcal{L}_{\text{boundary}} + w_{\text{cls}} \mathcal{L}_{\text{cls}}$$

### Chi tiết từng thành phần:
1. **$\mathcal{L}_{\text{seg}}$ (Focal Tversky Loss cho Mặt nạ hoàn chỉnh):**
   $$\mathcal{L}_{\text{seg}} = \left( 1 - \frac{|P \cap G| + \epsilon}{|P \cap G| + \alpha |P \setminus G| + \beta |G \setminus P| + \epsilon} \right)^\gamma$$
   *(Thiết lập chuẩn: $\alpha = 0.3, \beta = 0.7, \gamma = 1.33$ để ưu tiên giảm False Negatives / chống bỏ sót viền).*
2. **$\mathcal{L}_{\text{boundary}}$ (Binary Cross-Entropy cho Nhánh Viền Ban Đỏ):**
   $$\mathcal{L}_{\text{boundary}} = -\frac{1}{N} \sum_{i=1}^N \left[ B_i \ln(\hat{B}_i) + (1 - B_i) \ln(1 - \hat{B}_i) \right]$$
3. **$\mathcal{L}_{\text{cls}}$ (Evidential Loss cho Phân Loại Độ Bỏng):**
   $$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{EDL}}(\boldsymbol{\alpha}, \mathbf{y}) \quad (\text{kèm Annealing Schedule } \lambda_t)$$

### Trọng số hàm Loss khuyến nghị (Loss Weights):
$$w_{\text{seg}} = 1.0, \quad w_{\text{bnd}} = 0.3, \quad w_{\text{cls}} = 0.5$$

---

## 8. KỊCH BẢN THỰC NGHIỆM TIỆM TIẾN (EXPERIMENTAL ROADMAP & DECISION TREE)

Để đảm bảo an toàn tuyệt đối và có đầy đủ số liệu cho bảng **Ablation Study** trong bài báo, quy trình thực nghiệm được triển khai theo 3 pha nghiêm ngặt:

```
                            [XUẤT PHÁT: BASELINE 2 GIAI ĐOẠN]
                         (U-Net ResNet-34 + Classifier ResNet-34)
                                            │
                                            ▼
                    ╔═══════════════════════════════════════════════╗
                    ║ PHA 1: THỬ NGHIỆM ĐỘC LẬP EDL TRÊN CLASSIFIER ║
                    ║ - Huấn luyện trên tập train_roi.csv (SAM ROIs)║
                    ║ - So sánh: Cross-Entropy vs. EDL              ║
                    ║ - Đo đạc: Macro F1, Recall Độ 3, ECE, u       ║
                    ╚═══════════════════════╤═══════════════════════╝
                                            │
                                            ▼
                    ╔═══════════════════════════════════════════════╗
                    ║ PHA 2: XÂY DỰNG MẠNG MULTI-TASK THUẦN         ║
                    ║ - Shared ResNet-34 Backbone                   ║
                    ║ - U-Net Decoder + EDL Classification Head     ║
                    ║ - Huấn luyện End-to-End trên ảnh gốc toàn cảnh║
                    ╚═══════════════════════╤═══════════════════════╝
                                            │
                            ┌───────────────┴───────────────┐
                            ▼                               ▼
                [Kết quả Multi-Task TỐT]        [Kết quả Multi-Task KHÔNG TỐT]
          (F1/Dice >= Baseline, ECE thấp)        (Bị xung đột Gradient nặng)
                            │                               │
                            ▼                               ▼
╔═════════════════════════════════════════════╗ ╔═════════════════════════════════════════════╗
║ PHA 3A: ĐỈNH CAO SOTA (MT-EANet-EDL)        ║ ║ PHA 3B: PHƯƠNG ÁN B DỰ PHÒNG AN TOÀN        ║
║ - Tích hợp Erythema Dual-Branch vào         ║ ║ - Giữ kiến trúc 2 giai đoạn                 ║
║   Multi-Task Decoder                        ║ ║ - Phân đoạn: Nâng cấp thành EA-UNet         ║
║ - Mô hình hoàn chỉnh duy nhất cho bài báo Q1║ ║ - Phân loại: Dùng riêng Classifier với EDL  ║
╚═════════════════════════════════════════════╝ ╚═════════════════════════════════════════════╝
```

### Bảng Mục Tiêu Ablation Study Cần Đạt Được Trong Bài Báo:

| Model ID | Cấu Hình Mô Hình | Phân Đoạn (Dice / IoU) | Phân Loại (F1 / Rec Độ 3) | ECE (%) | Độ Bất Định $u$ | Đóng Góp Khoa Học |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **M0** | 2-Stage Baseline (U-Net + ResNet-34 CE) | 0.865 / 0.771 | 83.2% / 88.0% | ~14.5% | N/A | Điểm mốc ban đầu (45.7M params) |
| **M1** | 2-Stage + **EDL Head** | 0.865 / 0.771 | **84.5%** / **91.5%** | **~3.8%** | Phân tách rõ | Minh chứng tính tin cậy & y đức |
| **M2** | **Multi-Task thuần** (Shared + EDL) | 0.870 / 0.778 | 85.0% / 91.0% | ~4.0% | Phân tách rõ | Giảm 46% params, triệt tiêu ROI crop |
| **M3** | **MT-EANet-EDL** (Multi-Task + Dual-Branch) | **0.892** / **0.815** | **87.6%** / **94.0%** | **~3.5%** | Tối ưu nhất | **Đỉnh cao toàn diện cho bài báo Q1** |

---

## 9. ĐẶC TẢ KỸ THUẬT DÀNH CHO AI CODE GENERATION (IMPLEMENTATION BLUEPRINT)

> *Phần này dành riêng cho AI sinh code: Cung cấp đầy đủ cấu trúc lớp, tensor dimensions và thuật toán toán học.*

### 9.1. Module EDL Loss & Utilities (`src/vision/edl_loss.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLLoss(nn.Module):
    """
    Evidential Deep Learning Loss (Dirichlet Distribution)
    Dựa trên: Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018.
    """
    def __init__(self, num_classes: int = 3, annealing_epochs: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs

    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        # Uniform prior Dirichlet: beta = [1, 1, ..., 1]
        beta = torch.ones((1, self.num_classes), dtype=alpha.dtype, device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def forward(self, evidence: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        evidence: Tensor (B, num_classes) >= 0 (đã qua F.softplus)
        target: Tensor (B,) chứa class labels [0, 1, 2]
        """
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # 1. Expected MSE Loss (Khớp dữ liệu)
        loss_data = torch.sum((y_one_hot - p) ** 2, dim=1, keepdim=True)
        variance = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0)), dim=1, keepdim=True)
        loss_fit = torch.mean(loss_data + variance)
        
        # 2. KL Regularization với Annealing Schedule
        annealing_coef = min(1.0, float(epoch) / float(self.annealing_epochs))
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        loss_kl = torch.mean(self.kl_divergence(alpha_tilde))
        
        return loss_fit + annealing_coef * loss_kl


def get_edl_metrics(evidence: torch.Tensor, num_classes: int = 3):
    """
    Tính Expected Probabilities và Epistemic Uncertainty từ evidence
    evidence: Tensor (B, num_classes) >= 0
    Returns:
        prob: Tensor (B, num_classes)
        uncertainty: Tensor (B,) in range (0, 1]
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    uncertainty = num_classes / S.squeeze(-1)
    return prob, uncertainty
```

---

### 9.2. Kiến Trúc Mạng Đa Nhiệm `MT-EANet` (`src/vision/models_multitask.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MaskGuidedSpatialAttention(nn.Module):
    """
    Sử dụng mặt nạ phân đoạn mềm để hội tụ đặc trưng vào vùng tổn thương
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Downsample mask về kích thước của feature map bottleneck
        down_mask = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=False)
        attn = self.conv(torch.cat([feat, down_mask], dim=1))
        attended_feat = feat * attn + feat # Residual attention
        return attended_feat

class MTEANet(nn.Module):
    """
    Mạng Đa Nhiệm Multi-Task Erythema-Aware Network (MT-EANet-EDL)
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        # 1. SHARED ENCODER (ResNet-34 trích xuất 5 mức phân giải)
        self.encoder = timm.create_model('resnet34', pretrained=pretrained, features_only=True)
        # Các kênh ra của ResNet-34: [64, 64, 128, 256, 512]
        
        # 2. NHÁNH DECODER PHÂN ĐOẠN (U-Net expansive path)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = ConvBlock(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(64 + 64, 32)
        
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = ConvBlock(32, 16)
        
        # Đầu ra Phân đoạn
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(16, 1, kernel_size=1) # Nhánh viền
        
        # 3. NHÁNH PHÂN LOẠI EVIDENTIAL (EDL CLASSIFIER HEAD)
        self.mask_guided_attn = MaskGuidedSpatialAttention(in_channels=512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x shape: (B, 3, H, W)
        feats = self.encoder(x)
        c0, c1, c2, c3, c4 = feats # c4: (B, 512, H/32, W/32)
        
        # --- DECODER PHÂN ĐOẠN ---
        d4 = self.dec4(torch.cat([self.up4(c4), c3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c0], dim=1))
        d0 = self.dec0(self.up0(d1))
        
        pred_mask_logits = self.mask_head(d0)         # (B, 1, H, W)
        pred_boundary_logits = self.boundary_head(d0) # (B, 1, H, W)
        
        pred_mask_prob = torch.sigmoid(pred_mask_logits)
        
        # --- NHÁNH PHÂN LOẠI MASK-GUIDED + EDL ---
        c4_attended = self.mask_guided_attn(c4, pred_mask_prob.detach())
        pooled = self.global_pool(c4_attended).flatten(1) # (B, 512)
        cls_logits = self.cls_head(pooled)                # (B, 3)
        cls_evidence = F.softplus(cls_logits)             # Biến đổi thành Evidence >= 0
        
        return {
            "mask_logits": pred_mask_logits,
            "boundary_logits": pred_boundary_logits,
            "evidence": cls_evidence
        }
```

---

### 9.3. Vòng Lặp Huấn Luyện Chuẩn Cho Multi-Task (`train_multitask_step`)

```python
def train_multitask_epoch(model, loader, optimizer, seg_loss_fn, edl_loss_fn, bce_loss_fn, epoch, device):
    model.train()
    total_loss_accum = 0.0
    
    for batch in loader:
        images = batch["image"].to(device)        # (B, 3, 256, 256)
        masks = batch["mask"].to(device)          # (B, 1, 256, 256)
        labels = batch["severity"].to(device)     # (B,)
        boundaries = batch["boundary"].to(device) # (B, 1, 256, 256)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 1. Tính toán từng Loss riêng
        l_seg = seg_loss_fn(outputs["mask_logits"], masks)
        l_bnd = bce_loss_fn(outputs["boundary_logits"], boundaries)
        l_cls = edl_loss_fn(outputs["evidence"], labels, epoch=epoch)
        
        # 2. Tổng hợp Loss Đa nhiệm
        loss = (1.0 * l_seg) + (0.3 * l_bnd) + (0.5 * l_cls)
        
        # 3. Lan truyền ngược 1 lần duy nhất cập nhật Shared Encoder
        loss.backward()
        optimizer.step()
        
        total_loss_accum += loss.item()
        
    return total_loss_accum / len(loader)
```

---

## 10. KẾT LUẬN & HÀNH ĐỘNG TIẾP THEO

Tài liệu này là kim chỉ nam kỹ thuật cho toàn bộ pha phát triển nâng cấp học thuật của dự án `burn_analysis`.
- **Bước thực hiện ngay:** Đồng đội hoặc AI tiếp nhận tài liệu này sẽ mở module `src/vision/losses.py` và `src/vision/train_cls.py` để triển khai **Pha 1 (Tích hợp EDL Loss vào bộ phân loại ROI hiện có)**.
- Khi hoàn thành Pha 1 và nghiệm thu chỉ số ECE / Macro F1, toàn đội sẽ tự tin bước tiếp sang **Pha 2 (Multi-Task)** mà không sợ bất kỳ rủi ro hay mơ hồ nào về mặt kiến trúc!

