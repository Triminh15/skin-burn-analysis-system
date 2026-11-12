# 📋 ĐẶC TẢ YÊU CẦU & THIẾT KẾ USE CASE CHO GIAO DIỆN HỆ THỐNG
# HỆ THỐNG PHÂN TÍCH VẾT BỎNG ĐA PHƯƠNG THÁI (SKIN BURN ANALYSIS SYSTEM)

* **Mã môn học:** CS406 - Xử lý ảnh và Ứng dụng
* **Phiên bản tài liệu:** v1.0 (Chuẩn hóa thiết kế Frontend / UI-UX)
* **Đối tượng người dùng:** 
  1. Người bệnh / Người sơ cứu ban đầu (Bệnh nhân ngoại viện).
  2. Bác sĩ / Nhân viên y tế tuyến cơ sở hoặc phòng cấp cứu (Triage).

---

## 🏗️ 1. TỔNG QUAN KIẾN TRÚC GIAO DIỆN (UI ARCHITECTURE)

Hệ thống Frontend được thiết kế theo hướng **Clinical Dashboard & Assistant** (Bảng điều khiển lâm sàng đa nhiệm), phân chia thành 4 khu vực giao diện chính:

```
+-----------------------------------------------------------------------------------------+
|                                    HEADER / TOPBAR                                      |
|  [Logo & Tên Hệ Thống] | [Chế độ: Bệnh nhân / Bác sĩ] | [Trạng thái Model: Sẵn sàng]   |
+------------------------------------+----------------------------------------------------+
| PANEL TRÁI: DỮ LIỆU ĐẦU VÀO        | PANEL PHẢI: KẾT QUẢ PHÂN TÍCH CHUYÊN SÂU           |
|                                    |                                                    |
| 1. Upload ảnh / Camera trực tiếp   | 1. Tab "Thị giác máy tính (Vision Analysis)":      |
|    - Kéo thả file (.jpg, .png)     |    - Mask phân đoạn U-Net (So sánh Overlay/Slider)  |
|    - Ảnh chụp vùng bỏng            |    - Chỉ số % TBSA & Diện tích (cm²)               |
|                                    |    - Ảnh ROI & Cấp độ bỏng (ResNet-34)             |
| 2. Thông tin bệnh nhân & Thước đo: |    - Bản đồ nhiệt Grad-CAM (Minh bạch XAI)         |
|    - Vị trí cơ thể (Vùng nines)    |                                                    |
|    - Cân nặng (kg), Giới tính, Tuổi| 2. Tab "Cấp cứu & Hỗ trợ Lâm sàng (Clinical RAG)": |
|    - Chiều dài thước đo chuẩn (cm) |    - Phác đồ bù dịch Parkland (Tính tự động theo h)|
|                                    |    - Đánh giá nguy cơ sốc bỏng & Chuyển tuyến     |
| [NÚT BẤM: BẮT ĐẦU PHÂN TÍCH]       |    - Chatbot Bác sĩ AI (Hỏi đáp phác đồ Bộ Y Tế)   |
+------------------------------------+----------------------------------------------------+
|                                    FOOTER                                               |
|           Cảnh báo Y khoa Miễn trừ trách nhiệm & Bản quyền Đồ án CS406                  |
+-----------------------------------------------------------------------------------------+
```

---

## 📑 2. DANH SÁCH USE CASE CHI TIẾT (USE CASE SPECIFICATIONS)

---

### 🔹 UC-01: Tải ảnh vết bỏng & Kiểm định tiền xử lý ảnh (Image Ingestion & Privacy Guard)
* **Mục tiêu:** Cho phép người dùng cung cấp ảnh chụp vết bỏng an toàn và hợp lệ.
* **Tác nhân:** Người bệnh hoặc Nhân viên y tế.
* **Tiền điều kiện:** Ứng dụng đã khởi chạy và kết nối sẵn sàng.
* **Luồng sự kiện chính (Main Flow):**
  1. Người dùng chọn tải ảnh từ máy tính hoặc bấm chụp ảnh trực tiếp qua Webcam/Camera điện thoại.
  2. Giao diện hiển thị ảnh xem trước (Thumbnail preview) với thông số kích thước (Width x Height).
  3. Hệ thống tự động kích hoạt **Bộ lọc Quyền riêng tư (Privacy Guard)**: Xóa toàn bộ thông tin siêu dữ liệu EXIF (vị trí GPS, thông số máy ảnh cá nhân) để đảm bảo chuẩn HIPAA.
  4. Người dùng có thể dùng công cụ xoay ảnh (Rotate) hoặc cắt khung nếu ảnh bị nghiêng.
* **Ngoại lệ (Alternative Flows):**
  - *Ảnh không đúng định dạng (không phải jpg/png):* Báo lỗi Toast đỏ "Định dạng không hỗ trợ. Vui lòng chọn JPG/PNG".
  - *Kích thước file quá lớn (>15MB):* Hệ thống tự động nén nhẹ trước khi chuyển vào pipeline.

---

### 🔹 UC-02: Thiết lập Tham chiếu Kích thước & Thông tin Lâm sàng (Calibration & Patient Metrics)
* **Mục tiêu:** Cung cấp thông số vật lý để tính diện tích thực tế ($cm^2$) và lượng dịch truyền Parkland.
* **Tác nhân:** Người dùng / Bác sĩ.
* **Thành phần giao diện (UI Components):**
  - **Tool đo thước (Calibration Tool):**
    - Cho phép vẽ một đoạn thẳng trên ảnh tương ứng với thước đo/vật chuẩn (ví dụ thẻ căn cước 8.56 cm, hoặc thước dây).
    - Ô nhập: `Chiều dài thực tế của đoạn vừa vẽ (cm)`.
    - Hệ thống tính ra hệ số: `Pixel-per-Centimeter (PPCM)`.
  - **Vùng giải phẫu tổn thương (Anatomical Site Selection):**
    - Dropdown / Radio button chọn vùng theo **Luật số 9 Wallace (Rule of Nines)**:
      - Đầu - Mặt - Cổ (9%)
      - Cánh tay trái (9%), Cánh tay phải (9%)
      - Ngực - Bụng trước (18%), Lưng - Mông sau (18%)
      - Đùi - Chân trái (18%), Đùi - Chân phải (18%)
      - Bộ phận sinh dục (1%)
  - **Thông số sinh hiệu:**
    - Cân nặng bệnh nhân (kg) — Bắt buộc nếu cần tính bù dịch.
    - Thời gian tính từ lúc bị bỏng (giờ) — Để chia mốc 8 giờ đầu / 16 giờ sau.

---

### 🔹 UC-03: Phân đoạn vết bỏng & Tính toán diện tích (Segmentation & Area Calculation)
* **Mục tiêu:** Khoanh vùng chính xác biên vết bỏng và tính diện tích tổn thương.
* **Tác nhân:** Hệ thống (khi người dùng bấm "Bắt đầu Phân tích").
* **Thành phần giao diện & Trực quan hóa:**
  1. **Hiển thị hình ảnh so sánh (Visual Comparison Modes):**
     - *Chế độ 1 - Side-by-side:* 2 khung hình song song (Ảnh gốc vs Mask nhị phân đen/trắng).
     - *Chế độ 2 - Alpha Overlay:* Đè vùng mask màu đỏ bán trong suốt (Alpha = 0.4) lên ảnh gốc có viền contour màu vàng ôm khít vết bỏng.
     - *Chế độ 3 - Interactive Image Slider:* Thanh trượt kéo qua lại giữa ảnh gốc và ảnh phân đoạn.
  2. **Thẻ đo lường diện tích (Area Metrics Card):**
     - **Số pixel tổn thương:** ví dụ `142,500 pixels`.
     - **Tỷ lệ diện tích trên ảnh (% Frame Area):** ví dụ `23.4%`.
     - **Diện tích thực tế ước lượng:** ví dụ `45.8 cm²` (nếu có thước chuẩn).
     - **Ước lượng % TBSA:** Tỷ lệ % diện tích tổn thương cơ thể tính theo vùng giải phẫu.

---

### 🔹 UC-04: Trích xuất ROI & Phân loại cấp độ bỏng (ROI Extraction & Severity Classification)
* **Mục tiêu:** Phân loại độ sâu vết bỏng: Độ 1 (Bỏng bề mặt), Độ 2 (Bỏng trung bì/Phỏng nước), Độ 3 (Bỏng sâu/Hoại tử).
* **Thành phần giao diện:**
  1. **Khung hiển thị ROI (Region of Interest):**
     - Hiển thị ảnh đã được crop ôm vết bỏng (với 15% padding an toàn, vuông hóa chống méo mô da).
  2. **Thẻ kết quả Phân loại (Severity Badge):**
     - Cấp độ nặng nhất: **Bỏng Độ 2 (Second Degree Burn)**.
     - Mức độ tin cậy (Confidence Score): ví dụ **91.4%**.
     - Thanh xác suất (Probability Bars) của cả 3 cấp độ:
       - Độ 1: `■■□□□ 5.2%`
       - Độ 2: `■■■■■ 91.4%` (Nổi bật)
       - Độ 3: `■□□□□ 3.4%`
     - Nhãn cảnh báo màu sắc y khoa:
       - Độ 1: Xanh lá (Nhẹ / Chăm sóc tại nhà).
       - Độ 2: Vàng cam (Trung bình / Nguy cơ nhiễm trùng).
       - Độ 3: Đỏ đậm (Nặng / Cấp cứu khẩn cấp).

---

### 🔹 UC-05: Giải thích quyết định mô hình bằng Grad-CAM (Explainable AI - XAI)
* **Mục tiêu:** Minh bạch hóa quyết định của mô hình AI, giúp bác sĩ kiểm chứng vị trí mô da mà ResNet-34 dựa vào để đưa ra chẩn đoán.
* **Thành phần giao diện:**
  1. Hiển thị bản đồ nhiệt (Heatmap) phản ánh vùng trọng số:
     - Màu đỏ/vàng: Vùng mô tổn thương có tính quyết định cao nhất (ví dụ bóng nước, mô hoại tử khô xám).
     - Màu xanh dương: Vùng da lành ít ảnh hưởng đến dự đoán.
  2. Nút chuyển đổi (Toggle): Bật/Tắt lớp nhiệt Grad-CAM đè lên ảnh ROI.
  3. Đoạn mô tả tự động: "Mô hình tập trung 88% chú ý vào vùng bọng nước vỡ và đáy vết thương xung huyết ở trung tâm".

---

### 🔹 UC-06: Tư vấn Phác đồ Cấp cứu & Bù dịch Parkland (Clinical Protocol & Parkland Calculator)
* **Mục tiêu:** Tính toán chính xác lượng dịch Ringer Lactate cần truyền cho bệnh nhân bỏng nặng theo chuẩn Bộ Y tế.
* **Công thức tích hợp:**
  $$\text{Tổng dịch (ml)} = 4 \times \text{Cân nặng (kg)} \times \% \text{TBSA}$$
* **Thành phần giao diện:**
  - Bảng chia mốc truyền dịch thời gian thực:
    - **8 giờ đầu tiên:** Truyền 50% tổng lượng dịch ($\text{Tốc độ: } X \text{ ml/h}$).
    - **16 giờ tiếp theo:** Truyền 50% lượng dịch còn lại ($\text{Tốc độ: } Y \text{ ml/h}$).
  - Cảnh báo Triage & Chuyển tuyến:
    - Tự động bật thông báo đỏ nếu: $\% \text{TBSA} > 15\%$ ở người lớn (hoặc $> 10\%$ ở trẻ em) hoặc có bỏng Độ 3 $\rightarrow$ "CHUYỂN TUYẾN CẤP CỨU VIỆN BỎNG NGAY LẬP TỨC".

---

### 🔹 UC-07: Chatbot Bác sĩ AI Hỗ trợ Hỏi đáp Y tế (Medical Assistant QA)
* **Mục tiêu:** Cho phép người dùng chat tự nhiên để hỏi về cách sơ cứu, phác đồ điều trị, chăm sóc vết thương và dinh dưỡng.
* **Thành phần giao diện:**
  - Khung Chatbot dạng hội thoại (Message Thread).
  - Có các câu hỏi gợi ý nhanh (Quick Prompt Pills):
    - *"Vết bỏng này có nên chọc vỡ bọng nước không?"*
    - *"Tôi nên bôi thuốc gì trong 24 giờ đầu?"*
    - *"Chế độ ăn cho người bỏng độ 2 kiêng gì?"*
  - Trích dẫn nguồn tài liệu chuẩn: "Theo Hướng dẫn Chẩn đoán và Điều trị Bỏng - Bộ Y tế Việt Nam".

---

### 🔹 UC-08: Xuất Báo cáo Bệnh án Điện tử (Export Clinical Report PDF)
* **Mục tiêu:** Xuất toàn bộ kết quả phân tích thành file PDF chuyên nghiệp để lưu trữ hoặc chuyển viện.
* **Thành phần tài liệu PDF:**
  - Tiêu đề: BÁO CÁO PHÂN TÍCH TỔN THƯƠNG BỎNG DA.
  - Ảnh gốc kèm vùng khoanh viền U-Net & Ảnh ROI Grad-CAM.
  - Bảng số liệu: % Diện tích, Diện tích $cm^2$, Cấp độ bỏng, Tỉ lệ nguy cơ.
  - Bảng hướng dẫn truyền dịch Parkland và khuyến nghị của Bác sĩ AI.
  - Chữ ký điện tử / Ngày giờ phân tích.

---

## 🎨 3. THIẾT KẾ TRẢI NGHIỆM NGƯỜI DÙNG (UI/UX DESIGN GUIDELINES)

1. **Bảng màu chủ đạo (Medical Color Palette):**
   - **Primary Blue (`#1E88E5` / `#0D47A1`):** Thể hiện sự chuyên nghiệp, tin cậy y khoa.
   - **Warning Amber (`#FFA000`):** Cảnh báo bỏng độ 2, vết thương cần chú ý.
   - **Danger Red (`#D32F2F`):** Bỏng độ 3, chỉ số sốc bỏng cao, cần cấp cứu gấp.
   - **Success Teal (`#00897B`):** Vết bỏng nhẹ độ 1, da lành.
   - **Dark Neutral (`#212121`):** Nền cho ảnh y tế giúp làm nổi bật tổn thương và bản đồ nhiệt.

2. **Khả năng phản hồi tức thì (Feedback & Latency):**
   - Khi bấm phân tích, hiển thị Stepper rõ ràng:
     1. *Lọc nhiễu & Chuẩn hóa ảnh...*
     2. *U-Net đang phân đoạn vết thương...*
     3. *ResNet-34 đang phân loại độ sâu...*
     4. *Tính toán Parkland & Hoàn tất!*
   - Toàn bộ quá trình chạy trên CPU chỉ mất $\sim 1.5 - 2.5$ giây.

