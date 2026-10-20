# 🏥 TÀI LIỆU Y KHOA TOÀN DIỆN: PHÁC ĐỒ CHẨN ĐOÁN, CẤP CỨU VÀ ĐIỀU TRỊ BỎNG
## Cẩm Nang Lâm Sàng Đa Nguồn Chuẩn Hóa Phục Vụ Hệ Thống AI Y Tế & Medical RAG
### Được đối chiếu và thẩm định từ: Bộ Y Tế Việt Nam, Viện Bỏng Quốc Gia Lê Hữu Trác, WHO, ABA, ISBI, NHS UK & Mayo Clinic

---

## 📌 MỤC LỤC CHI TIẾT
1. [Tổng Quan Khung Pháp Lý & Nguồn Dẫn Y Khoa Chính Thống](#1-tổng-quan-khung-pháp-lý--nguồn-dẫn-y-khoa-chính-thống)
2. [Cơ Chế Bệnh Sinh & Mô Hình Vết Thương Bỏng Jackson (Jackson's Burn Wound Model)](#2-cơ-chế-bệnh-sinh--mô-hình-vết-thương-bỏng-jackson)
3. [Quy Trình Sơ Cứu "Giờ Vàng" (15 - 20 Phút Đầu) Chuẩn Quốc Tế](#3-quy-trình-sơ-cứu-giờ-vàng-15---20-phút-đầu-chuẩn-quốc-tế)
4. [Phân Loại Độ Sâu Bỏng (Burn Depth Classification)](#4-phân-loại-độ-sâu-bỏng-burn-depth-classification)
5. [Đánh Giá Diện Tích Bỏng Toàn Phần (Total Body Surface Area - TBSA)](#5-đánh-giá-diện-tích-bỏng-toàn-phần-total-body-surface-area---tbsa)
6. [Các Thang Đo Tiên Lượng Độ Nặng & Nguy Cơ Tử Vong](#6-các-thang-đo-tiên-lượng-độ-nặng--nguy-cơ-tử-vong)
7. [Tiêu Chuẩn Cảnh Báo Đỏ Chuyển Tuyến Khẩn Cấp (Red Flags & Burn Center Referral)](#7-tiêu-chuẩn-cảnh-báo-đỏ-chuyển-tuyến-khẩn-cấp-red-flags--burn-center-referral)
8. [Phác Đồ Hồi Sức Chống Sốc Bỏng & Bù Dịch Toàn Thân (Fluid Resuscitation)](#8-phác-đồ-hồi-sức-chống-sốc-bỏng--bù-dịch-toàn-thân-fluid-resuscitation)
9. [Phác Đồ Chăm Sóc Vết Thương Bỏng Tại Chỗ (Local Wound Care & Topical Agents)](#9-phác-đồ-chăm-sóc-vết-thương-bỏng-tại-chỗ-local-wound-care--topical-agents)
10. [Xử Trí Các Thể Bỏng Đặc Biệt: Hóa Chất, Luồng Điện & Đường Hô Hấp](#10-xử-trí-các-thể-bỏng-đặc-biệt-hóa-chất-luồng-điện--đường-hô-hấp)
11. [Quản Lý Đau, Kháng Sinh, Dinh Dưỡng & Phòng Ngừa Uốn Ván](#11-quản-lý-đau-kháng-sinh-dinh-dưỡng--phòng-ngừa-uốn-ván)
12. [Hàng Rào Kiểm Duyệt An Toàn Cho Trợ Lý Ảo (Medical Guardrails & Anti-Myth System)](#12-hàng-rào-kiểm-duyệt-an-toàn-cho-trợ-lý-ảo-medical-guardrails--anti-myth-system)
13. [Bảng Tra Cứu Toàn Bộ Nguồn Dẫn Học Thuật & Văn Bản Chính Thức](#13-bảng-tra-cứu-toàn-bộ-nguồn-dẫn-học-thuật--văn-bản-chính-thức)

---

## 1. TỔNG QUAN KHUNG PHÁP LÝ & NGUỒN DẪN Y KHOA CHÍNH THỐNG

Tài liệu này đóng vai trò là **Cơ sở dữ liệu Tri thức Gốc (Ground-Truth Medical Knowledge Base)** cho hệ thống RAG và Trợ lý Bác sĩ Ảo chẩn đoán bỏng. Toàn bộ thông tin được tổng hợp trực tiếp từ các văn bản pháp lý y tế và tài liệu lâm sàng tiêu chuẩn:

* **Bộ Y Tế Việt Nam:**
  * *Quyết định số 635/QĐ-BYT* ngày 28/02/2014 về việc ban hành tài liệu *"Hướng dẫn quy trình kỹ thuật khám bệnh, chữa bệnh chuyên ngành Bỏng"*.
  * *Quyết định số 4501/QĐ-BYT* về sửa đổi, bổ sung quy trình kỹ thuật điều trị bỏng.
  * *Phác đồ điều trị bỏng lâm sàng* của **Viện Bỏng Quốc Gia Lê Hữu Trác - Học viện Quân y** (Bệnh viện tuyến cuối chuyên khoa Bỏng lớn nhất Việt Nam).
* **Hiệp Hội Bỏng Hoa Kỳ (American Burn Association - ABA):**
  * *ABA Practice Guidelines for Burn Care* & *Guidelines for the Operation of Burn Centers*.
  * *Advanced Burn Life Support (ABLS) Provider Manual*.
* **Tổ Chức Y Tế Thế Giới (World Health Organization - WHO):**
  * *WHO Guidelines for Burn Care Under Austere Conditions* & *WHO Technical Fact Sheet on Burns*.
* **Hiệp Hội Bỏng Quốc Tế (International Society for Burn Injuries - ISBI):**
  * *ISBI Practice Guidelines for Burn Care* (công bố trên tạp chí y khoa *Burns*).
* **Cơ Quan Y Tế Quốc Gia Anh (National Health Service - NHS & NICE):**
  * *NHS Clinical Knowledge Summaries (CKS): Burns and scalds*.
  * *British Burn Association (BBA) First Aid Guidelines*.
* **Mayo Clinic & Trung Tâm Thông Tin Công Nghệ Sinh Học Quốc Gia Hoa Kỳ (NCBI/NIH):**
  * *Mayo Clinic Clinical Reviews: Burns Management*.
  * *StatPearls Publishing / National Library of Medicine: Burn Resuscitation and Management*.

---

## 2. CƠ CHẾ BỆNH SINH & MÔ HÌNH VẾT THƯƠNG BỎNG JACKSON

Trong y học bỏng hiện đại, mô hình kinh điển của **Douglas Jackson (1953)** mô tả 3 vùng tổn thương đồng tâm đặc trưng của một vết bỏng nhiệt:

```text
       ┌────────────────────────────────────────────────────────┐
       │             VÙNG XUNG HUYẾT (Zone of Hyperemia)         │
       │    ┌──────────────────────────────────────────────┐    │
       │    │           VÙNG Ứ TRỆ (Zone of Stasis)        │    │
       │    │    ┌────────────────────────────────────┐    │    │
       │    │    │    VÙNG ĐÔNG VÓN (Coagulation)     │    │    │
       │    │    │   (Tổn thương hoại tử vĩnh viễn)   │    │    │
       │    │    └────────────────────────────────────┘    │    │
       │    │      (Mô thiếu máu cục bộ có thể cứu)        │    │
       │    └──────────────────────────────────────────────┘    │
       │        (Tăng tưới máu, tự phục hồi hoàn toàn)          │
       └────────────────────────────────────────────────────────┘
```

1. **Vùng Đông Vón (Zone of Coagulation):**
   * Nằm ở trung tâm vết thương, nơi nhiệt lượng cao nhất tác động trực tiếp.
   * Protein tế bào bị biến tính và đông vón hoàn toàn, mạng lưới mao mạch bị tắc nghẽn vô mạch.
   * **Mức độ:** Mô hoại tử không thể hồi phục (hoại tử khô hoặc ướt), bắt buộc phải cắt lọc hoặc chờ rụng vảy.
2. **Vùng Ứ Trệ (Zone of Stasis):**
   * Bao quanh vùng đông vón trung tâm.
   * Đặc trưng bởi hiện tượng giảm tưới máu cục bộ, phù nề nặng nề, bạch cầu kết tụ bám dính thành mạch gây vi huyết khối (microvascular thrombosis).
   * **Ý nghĩa sống còn:** Đây là **"vùng tranh chấp"** có tiềm năng cứu sống được nếu được sơ cứu hạ nhiệt kịp thời, bồi phụ dịch đủ và chống nhiễm khuẩn. Nếu xử trí sai (bôi đá lạnh, thiếu dịch, để nhiễm trùng), vùng ứ trệ sẽ hoại tử chuyển biến thành vùng đông vón (vết bỏng nông chuyển thành bỏng sâu).
3. **Vùng Xung Huyết (Zone of Hyperemia):**
   * Vòng rìa ngoài cùng của vết bỏng.
   * Tăng lưu lượng máu đến tưới do giãn mạch sinh lý phản ứng lại các chất trung gian gây viêm (histamine, bradykinin, prostaglandins).
   * **Tiên lượng:** Mô hồi phục hoàn toàn sau 7 - 10 ngày nếu không có biến chứng nhiễm khuẩn thứ phát.

---

## 3. QUY TRÌNH SƠ CỨU "GIỜ VÀNG" (15 - 20 PHÚT ĐẦU) CHUẨN QUỐC TẾ

Quy tắc cấp cứu ban đầu tuân thủ nguyên tắc đa chấn thương **ABCDE** (Airway, Breathing, Circulation, Disability, Exposure):

### Bước 1: Dập tắt nguồn nhiệt & Đảm bảo an toàn (Stop the Burning Process)
* **Bỏng lửa:** Áp dụng nguyên tắc quốc tế **"Dừng lại - Nằm xuống - Lăn tròn" (Stop, Drop and Roll)**. Che mặt nạn nhân bằng hai bàn tay. Dùng chăn bông dập lửa hoặc dùng nước xối dập tắt.
* **Bỏng điện:** Ngắt nguồn điện/cầu dao ngay lập tức trước khi chạm vào người nạn nhân. Dùng que gỗ khô gạt dây điện ra.
* **Bỏng hóa chất:** Cởi bỏ quần áo dính hóa chất, gạt hóa chất khô dạng bột trước khi xả nước.
* **Cắt bỏ trang phục:** Nhanh chóng dùng kéo cắt bỏ quần áo xung quanh vết thương. **Tuyệt đối KHÔNG cố giật hoặc bóc tách những mảng vải đã bị cháy dính chặt vào da**, vì sẽ làm rách toạc lớp biểu bì còn lại.

### Bước 2: Hạ nhiệt vết bỏng bằng nước sạch (Cool the Burn)
* **Quy chuẩn:** Ngâm hoặc xả nhẹ vùng bị tổn thương dưới vòi **nước máy sạch mát** (nhiệt độ lý tưởng từ **15°C đến 20°C**) liên tục trong **15 đến 20 phút**. Thời điểm can thiệp hiệu quả nhất là trong vòng 30 phút đầu, tối đa không quá 3 giờ sau tai nạn.
* **Lợi ích sinh lý học:**
  * Giảm nhiệt độ tích tụ dưới da, ngăn chặn hiện tượng truyền nhiệt gây sâu thêm vùng tổn thương (chuyển vùng ứ trệ thành đông vón).
  * Làm tê liệt tạm thời các thụ thể cảm giác đau (nociceptors), giảm đau rõ rệt mà không cần dùng thuốc ngay.
  * Giảm thoát huyết tương và phù nề mô kẽ.
* **Cảnh báo khẩn cấp (Contraindications):**
  * ❌ **TUYỆT ĐỐI KHÔNG DÙNG NƯỚC ĐÁ HOẶC ĐÁ VIÊN:** Tinh thể đá gây co mạch ngoại vi đột ngột, làm tắc nghẽn tưới máu nuôi dưỡng, gây hoại tử lạnh chồng chất lên hoại tử nhiệt (*frostbite on top of burn*).
  * ❌ **Không hạ nhiệt quá 20 phút trên diện rộng ở trẻ em và người già:** Nguy cơ cao gây **hạ thân nhiệt toàn thân (Hypothermia)**, làm rối loạn đông máu và loạn nhịp tim. Khi nhiệt độ cơ thể lõi tụt xuống dưới 35°C, tỷ lệ tử vong tăng gấp nhiều lần.

### Bước 3: Tháo bỏ trang sức & Vật co thắt (Remove Constrictive Items)
* Tháo ngay nhẫn, lắc tay, đồng hồ, thắt lưng, giày tất trước khi hiện tượng phù nề tiến triển.
* Nếu để muộn, ngón tay/ngón chân sưng to sẽ bị nhẫn thắt chặt như garô, gây thiếu máu hoại tử đầu chi phải cắt cụt.

### Bước 4: Che phủ bảo vệ vết thương (Cover the Burn)
* Theo khuyến cáo của **NHS Anh Quốc (NICE)**: Sử dụng **màng bọc thực phẩm sạch dạng cuộn (Cling Film)** đặt nhẹ nhàng dọc theo vết bỏng (đặt phẳng từng lớp, **không quấn tròn siết chặt** quanh chi thể để tránh tắc mạch khi mô sưng phù). Màng bọc thực phẩm trong suốt giúp bác sĩ tại viện quan sát được vết thương mà không cần bóc băng, chống dính mô và ngăn vi khuẩn xâm nhập hoàn hảo.
* Với bỏng bàn tay: Cho bàn tay vào túi nilon sạch vô trùng.
* Nếu không có màng bọc: Phủ nhẹ bằng gạc vô trùng khô hoặc khăn bông sạch khô. Không dùng khăn có lông sợi tưa dễ dính vào vết thương.

### Bước 5: Giữ ấm cơ thể (Keep Patient Warm)
* Đắp chăn ấm lên các vùng cơ thể lành lặn. Cho nạn nhân nằm nghỉ nơi thoáng mát, kín gió.

---

## 4. PHÂN LOẠI ĐỘ SÂU BỎNG (BURN DEPTH CLASSIFICATION)

Bộ Y Tế Việt Nam phân loại lâm sàng thành 4 độ, tương thích hoàn toàn với phân loại quốc tế của ABA (American Burn Association):

```text
Biểu bì (Epidermis)  ───► [Độ 1: Bỏng nông]
Màng đáy (Basement)  ──────────────────────────
Trung bì nhú (Papillary) ─► [Độ 2 Nông: Superficial Partial]
Trung bì lưới (Reticular) ─► [Độ 2 Sâu / Độ 3 Nông: Deep Partial]
Hạ bì (Subcutis/Fat) ───► [Độ 3 Sâu / Độ 4: Full-Thickness]
Cân, Cơ, Xương       ───► [Độ 4 Sâu: Subdermal / Fourth Degree]
```

### Bảng Đối Chiếu Lâm Sàng Chi Tiết:

| Tiêu Chí Đánh Giá | Bỏng Độ 1 (Nông) | Bỏng Độ 2 Nông (Trung bì nông) | Bỏng Độ 2 Sâu (Trung bì sâu) | Bỏng Độ 3 (Toàn phần) | Bỏng Độ 4 (Sâu vào cơ/xương) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Thuật ngữ quốc tế** | *Superficial burn* | *Superficial Partial-Thickness* | *Deep Partial-Thickness* | *Full-Thickness burn* | *Fourth-Degree burn* |
| **Lớp giải phẫu tổn thương** | Chỉ lớp biểu bì ngoài cùng | Biểu bì + Lớp nhú của trung bì (Papillary dermis) | Biểu bì + Lớp lưới trung bì (Reticular dermis) | Hủy hoại toàn bộ biểu bì, trung bì, chạm mỡ | Toàn bộ da + Cân cơ, gân, mạch máu, xương |
| **Đặc điểm bọng nước (Blisters)** | **Không có bọng nước** | **Bọng nước mỏng vách**, dịch trong hoặc vàng chanh | **Bọng nước vách dày**, dịch đục hoặc hồng lẫn máu | **Không có bọng nước**, da khô cứng | Không có bọng nước, mô cháy đen than hóa |
| **Màu sắc bề mặt mô** | Đỏ ửng (Erythema), khô ráo | Đỏ hồng tươi, ẩm ướt, phù nề | Đỏ đốm trắng hoặc trắng bệch lấm chấm | Trắng bệch như sáp, nâu xám, hoặc đen da thuộc | Đen than, xám xịt, lộ thớ cơ hoặc khung xương |
| **Thời gian đổ đầy mao mạch (Capillary Refill)** | Bình thường (< 2 giây) | Rất nhanh (< 1-2 giây), nhạy | Chậm trễ (> 2-3 giây) hoặc mất | **Hoàn toàn mất** (ấn không đổi màu) | Hoàn toàn mất |
| **Cảm giác đau (Sensation)** | Rát bỏng nhẹ khi chạm | **Đau đớn dữ dội**, cực kỳ nhạy cảm với gió/nước | Đau âm ỉ, giảm cảm giác châm kim | **Mất hoàn toàn cảm giác đau** ở trung tâm | Mất hoàn toàn cảm giác |
| **Thời gian phục hồi** | 3 - 6 ngày | 10 - 14 ngày | 3 - 4 tuần (thường chậm) | Không thể tự lành (chỉ co kéo mép) | Không thể tự lành |
| **Tiên lượng sẹo** | Không để lại sẹo | Thường không sẹo, đổi màu nhẹ | Nguy cơ sẹo phì đại cao (Hypertrophic) | Sẹo co kéo nặng, biến dạng vĩnh viễn | Mất chức năng chi, tàn phế, nguy cơ cắt cụt |
| **Phương thức điều trị** | Chăm sóc tại nhà, gel làm dịu | Băng vô khuẩn, mỡ kháng sinh | Cân nhắc phẫu thuật cắt lọc sớm | Phẫu thuật cắt lọc mô hoại tử & Ghép da | Phẫu thuật phức tạp, chuyển vạt cơ, vi phẫu |

---

## 5. ĐÁNH GIÁ DIỆN TÍCH BỎNG TOÀN PHẦN (TOTAL BODY SURFACE AREA - TBSA)

Chỉ số % TBSA đóng vai trò quyết định trong việc tính lượng dịch truyền chống sốc và quyết định chuyển tuyến. **Quy tắc quốc tế: KHÔNG TÍNH diện tích bỏng Độ 1 vào % TBSA** vì bỏng Độ 1 không gây thoát dịch qua mao mạch.

### 1. Quy tắc số 9 của Wallace (Wallace Rule of Nines - Áp dụng cho Người lớn)
Cơ thể người lớn được chia thành các phân vùng là bội số của 9%:

* **Đầu, Mặt, Cổ:** **9%** (Mặt trước 4.5%, Mặt sau 4.5%)
* **Chi trên phải:** **9%** (Mặt trước 4.5%, Mặt sau 4.5%)
* **Chi trên trái:** **9%** (Mặt trước 4.5%, Mặt sau 4.5%)
* **Thân trước:** **18%** (Ngực 9%, Bụng 9%)
* **Thân sau:** **18%** (Lưng trên 9%, Lưng dưới/Mông 9%)
* **Chi dưới phải:** **18%** (Mặt trước 9%, Mặt sau 9%)
* **Chi dưới trái:** **18%** (Mặt trước 9%, Mặt sau 9%)
* **Bộ phận sinh dục ngoài & Tầng sinh môn:** **1%**
* **Tổng cộng:** $9 + 9 + 9 + 18 + 18 + 18 + 18 + 1 = 100\%$

### 2. Biểu đồ Lund & Browder (Lund and Browder Chart - Chuẩn Vàng Cho Trẻ Em)
Do trẻ em có tỷ lệ đầu to hơn và chi dưới ngắn hơn người lớn, tỷ lệ phân chia thay đổi theo từng mốc tuổi:

| Phân Vùng Giải Phẫu | Sơ sinh (0 tuổi) | 1 Tuổi | 5 Tuổi | 10 Tuổi | 15 Tuổi | Người lớn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đầu & Cổ** | **19%** | **17%** | **13%** | **11%** | **9%** | **7%** |
| **Một Đùi** | **5.5%** | **6.5%** | **8%** | **8.5%** | **9%** | **9.5%** |
| **Một Cẳng chân** | **5%** | **5%** | **5.5%** | **6%** | **6.5%** | **7%** |

### 3. Phương pháp Lòng bàn tay (Palm Method - Palmer Rule)
* Diện tích **toàn bộ lòng bàn tay tính cả các ngón tay khép kín** của chính bệnh nhân tương đương xấp xỉ **1% TBSA**.
* Nếu chỉ tính phần gan bàn tay (không tính ngón tay) thì tương đương **0.5% TBSA**.
* **Ứng dụng:** Đo các mảng bỏng nhỏ, rải rác phân tán khắp người, hoặc dùng để trừ đi những vùng da lành lặn còn sót lại trong một mảng bỏng lớn.

### 4. Phương pháp 1.3.6.9.18 của GS. Lê Thế Trung (Viện Bỏng Quốc Gia)
Được chuẩn hóa phù hợp với thể trạng người Việt Nam:
* Bàn tay, cổ, mặt: 1 - 3%
* Cẳng tay: 3%
* Cánh tay: 3%
* Cẳng chân: 6%
* Đùi: 9%
* Lưng, ngực, bụng: 9 - 18%

---

## 6. CÁC THANG ĐO TIÊN LƯỢNG ĐỘ NẶNG & NGUY CƠ TỬ VONG

### 1. Chỉ số Frank (Frank Index - Áp dụng tại Viện Bỏng Lê Hữu Trác)
Được sử dụng rộng rãi tại Việt Nam để tiên lượng mức độ sốc bỏng:

$$\text{Chỉ số Frank (FI)} = \text{\% Bỏng nông} + (3 \times \text{\% Bỏng sâu})$$

* $FI < 30$: Tiên lượng nhẹ, hiếm khi sốc.
* $FI = 30 - 50$: Tiên lượng vừa, có thể có sốc nhẹ.
* $FI = 51 - 90$: Tiên lượng nặng, sốc bỏng nặng, cần hồi sức tích cực.
* $FI > 90$: Tiên lượng rất nặng, nguy cơ tử vong cao.
* $FI > 120$: Tiên lượng cực kỳ nguy kịch.

### 2. Chỉ số Baux Điều Chỉnh (Revised Baux Score - Chuẩn Quốc Tế)
Dùng để dự đoán tỷ lệ tử vong toàn cầu do bỏng:

$$\text{Revised Baux Score} = \text{Tuổi (năm)} + \text{\% TBSA} + [17 \text{ (nếu có Bỏng đường thở)}]$$

* Điểm $< 75$: Tỷ lệ tử vong $< 5\%$.
* Điểm $75 - 100$: Tỷ lệ tử vong khoảng $15 - 35\%$.
* Điểm $100 - 120$: Tỷ lệ tử vong $> 50\%$.
* Điểm $> 140$: Tỷ lệ tử vong vượt quá $90\%$.

---

## 7. TIÊU CHUẨN CẢNH BÁO ĐỎ CHUYỂN TUYẾN KHẨN CẤP (RED FLAGS & BURN CENTER REFERRAL)

Theo quy định nghiêm ngặt của **Hiệp hội Bỏng Hoa Kỳ (ABA)** và **Bộ Y Tế Việt Nam**, bệnh nhân **BẮT BUỘC PHẢI CHUYỂN ĐẾN TRUNG TÂM BỎNG CHUYÊN KHOA** nếu có bất kỳ dấu hiệu nào sau đây:

1. **Về Diện Tích & Độ Sâu:**
   * Bỏng Độ 2 (Partial-thickness) có diện tích **$\ge 10\%$ TBSA** ở người lớn.
   * Bỏng Độ 2 có diện tích **$\ge 5\%$ TBSA** ở trẻ em dưới 10 tuổi hoặc người già trên 50 tuổi.
   * **Bỏng Độ 3 (Full-thickness) ở bất kỳ lứa tuổi nào và bất kỳ diện tích nào** (kể cả chỉ 1% diện tích).
2. **Về Vị Trí Giải Phẫu Trọng Yếu (Critical Functional/Cosmetic Zones):**
   * Bỏng liên quan đến: **Mặt, Cổ, Mắt, Tai, Bàn tay, Bàn chân, Vùng sinh dục ngoài, Tầng sinh môn**, hoặc **bỏng bắc qua các khớp vận động lớn** (khớp khuỷu, khớp gối, cổ tay, cổ chân) do nguy cơ co rút gân cơ gây tàn tật.
3. **Bỏng Đường Hô Hấp (Inhalation Injury):**
   * Nạn nhân bị kẹt trong đám cháy phòng kín có các dấu hiệu: lông mũi cháy xém, muội than bám quanh mũi miệng, khạc đờm màu đen, giọng khàn đặc, thở rít (*Stridor*), khó thở, SpO2 tụt. Cần đặt nội khí quản sớm trước khi phù nề thanh môn bít tắc khí đạo.
4. **Bỏng Do Tác Nhân Đặc Biệt Nguy Hiểm:**
   * **Bỏng điện cao thế ($\ge 1000\text{V}$)** hoặc sét đánh: Nguy cơ cao ngừng tim do rung thất, tổn thương nội tạng âm thầm, hoại tử cơ sâu gây suy thận cấp do Myoglobin niệu.
   * **Bỏng hóa chất mạnh:** Axit đậm đặc (H2SO4, HNO3, HF) hoặc bazơ đậm đặc (NaOH, KOH) gây ăn mòn kéo dài.
5. **Bỏng Vòng Tròn Quanh Chi Thể / Quanh Thân Mình (Circumferential Burns):**
   * Vết bỏng sâu tạo thành vòng siết cứng quanh chi thể hoặc quanh lồng ngực $\rightarrow$ Nguy cơ cao gây **Hội chứng chèn ép khoang (Compartment Syndrome)** làm hoại tử tắc mạch ngọn chi hoặc gây suy hô hấp do lồng ngực không thể nở ra khi thở. Cần phẫu thuật rạch giải áp khẩn cấp (*Escharotomy*).
6. **Bỏng Trên Cơ Địa Đặc Biệt:**
   * Bệnh nhân có tiền sử đái tháo đường, suy tim, suy thận mãn, bệnh phổi tắc nghẽn mãn tính (COPD), suy giảm miễn dịch (HIV/AIDS, ung thư), hoặc đa chấn thương phối hợp (chấn thương sọ não, gãy xương lớn).

---

## 8. PHÁC ĐỒ HỒI SỨC CHỐNG SỐC BỎNG & BÙ DỊCH TOÀN THÂN (FLUID RESUSCITATION)

Bỏng nặng gây thoát huyết tương ồ ạt qua thành mao mạch bị giãn nở trên toàn cơ thể, dẫn đến **sốc giảm thể tích (Burn Shock)** và suy đa tạng.

### 1. Chỉ định bù dịch đường tĩnh mạch:
* Người lớn có diện tích bỏng $\ge 15 - 20\%$ TBSA.
* Trẻ em có diện tích bỏng $\ge 10\%$ TBSA.

### 2. Công thức Parkland Kinh Điển (Parkland Formula - Tiêu chuẩn quốc tế):
Dung dịch sử dụng: **Ringer Lactate** (tốt nhất, tránh dùng NaCl 0.9% khối lượng lớn vì gây toan chuyển hóa tăng clo máu):

$$\text{Tổng thể tích dịch trong 24 giờ đầu (ml)} = 4\text{ ml} \times \text{Cân nặng (kg)} \times \text{\% TBSA (Độ 2 + Độ 3)}$$

*Ví dụ: Bệnh nhân 60 kg, bỏng 30% TBSA:*
$$\text{Thể tích} = 4 \times 60 \times 30 = 7,200\text{ ml (Ringer Lactate)}$$

**Quy tắc phân chia tốc độ truyền:**
* **8 giờ đầu:** Truyền **50%** tổng lượng dịch tính toán (tính từ THỜI ĐIỂM XẢY RA TAI NẠN, không phải tính từ lúc nhập viện).
* **16 giờ tiếp theo:** Truyền **50%** lượng dịch còn lại.

### 3. Công thức Modified Brooke (Khuyến nghị mới của ABA):
ABA hiện khuyến nghị giảm liều dịch khởi đầu xuống **$2\text{ ml} \times \text{kg} \times \% \text{TBSA}$** ở người lớn để tránh hiện tượng **"Fluid Creep"** (quá tải dịch gây phù phổi cấp và hội chứng khoang bụng).

### 4. Tiêu chuẩn vàng để giám sát hiệu quả hồi sức: Lượng nước tiểu (Urine Output):
Không dựa vào huyết áp vì cơ chế co mạch bù trừ có thể đánh lừa. Bắt buộc đặt ống thông tiểu (Foley catheter) và đo lượng nước tiểu từng giờ:
* **Người lớn:** Duy trì **0.5 – 1.0 ml/kg/giờ** (khoảng 30 – 50 ml/giờ).
* **Trẻ em (< 30 kg):** Duy trì **1.0 – 1.5 ml/kg/giờ**.
* **Bỏng điện có myoglobin niệu (nước tiểu màu xá xị):** Phải duy trì **1.5 – 2.0 ml/kg/giờ** (khoảng 100 ml/giờ ở người lớn) kết hợp kiềm hóa nước tiểu bằng Natri Bicarbonat để ngăn myoglobin kết tủa làm tắc ống thận.

---

## 9. PHÁC ĐỒ CHĂM SÓC VẾT THƯƠNG BỎNG TẠI CHỖ (LOCAL WOUND CARE & TOPICAL AGENTS)

### 1. Xử lý Bọng Nước (Blister Management):
* **Bọng nước còn nguyên vẹn, kích thước nhỏ/vừa ($< 2 - 3\text{ cm}$):** **BẢO TỒN NGUYÊN VẸN, TUYỆT ĐỐI KHÔNG CHỌC THỦNG.** Lớp da biểu bì của bọng nước là màng che chắn sinh học vô trùng tự nhiên hoàn hảo nhất giúp ngăn vi khuẩn xâm nhập và giảm đau.
* **Bọng nước lớn, căng tức nguy cơ tự vỡ, hoặc nằm ở khớp vận động:** Dùng kim tiêm vô trùng chọc hút dịch ở đáy nốt phỏng, giữ lại lớp da áp vào nền tổn thương.
* **Bọng nước đã vỡ, dịch đục nhiễm khuẩn, màng da bong tróc nham nhở:** Dùng kéo và kẹp vô trùng cắt lọc toàn bộ lớp da hoại tử chết, rửa sạch nền mô bằng nước muối sinh lý NaCl 0.9%.

### 2. Danh Mục Thuốc Kháng Khuẩn Bôi Tại Chỗ (Topical Antimicrobials):
1. **Silver Sulfadiazine 1% (Bạc Sulfadiazine / Silvadene):**
   * *Ưu điểm:* Tiêu chuẩn vàng cho bỏng Độ 2 và Độ 3. Kháng khuẩn phổ rộng chống lại cả vi khuẩn Gram âm (*Pseudomonas aeruginosa* - trực khuẩn mủ xanh), Gram dương (*Staphylococcus aureus*) và nấm. Không gây đau rát khi bôi.
   * *Chống chỉ định:* Phụ nữ có thai 3 tháng cuối, trẻ sơ sinh dưới 2 tháng tuổi (nguy cơ vàng da nhân não Kernicterus), người dị ứng nhóm Sulfa.
2. **Mafenide Acetate (Sulfamylon 5% - 8.5%):**
   * *Ưu điểm:* Thấm sâu qua mô hoại tử khô dày (Eschar) cực tốt, rất hiệu quả với bỏng sụn vành tai và mô hoại tử sâu.
   * *Nhược điểm:* Ức chế men carbonic anhydrase gây toan chuyển hóa tăng thông khí; gây đau rát dữ dội khi bôi lên mô còn sống.
3. **Mỡ Kháng Sinh Phối Hợp (Bacitracin / Neomycin / Polymyxin B):**
   * Thích hợp cho bỏng nông vùng mặt và trẻ em, không làm đen vùng da, dễ theo dõi tiến triển mô.
4. **Băng Gạc Tiên Tiến Phủ Vết Bỏng:**
   * *Gạc mỡ chống dính (Paraffin gauze / Jelonet / UrgoTul):* Lớp lót trực tiếp lên vết thương ngăn gạc bám dính vào mô hạt, không gây đau hay chảy máu khi thay băng.
   * *Gạc Nano Bạc phóng thích chậm (Acticoat, Aquacel Ag):* Phóng thích ion bạc liên tục trong 3 - 7 ngày, giảm tần suất thay băng, giảm đau đớn cho bệnh nhân.
   * *Hydrogel / Hydrocolloid:* Dùng cho bỏng nông ít dịch tiết, tạo môi trường vi khí hậu ẩm lý tưởng kích thích tế bào biểu mô di chuyển đóng miệng vết thương.

### 3. Phẫu Thuật Cắt Lọc Hoại Tử Sớm & Ghép Da (Early Excision and Grafting):
* Với bỏng sâu Độ 3, Độ 4: Không chờ mô hoại tử tự rụng vì sẽ dẫn đến nhiễm trùng huyết và suy mòn.
* Tiêu chuẩn hiện đại: Phẫu thuật cắt bỏ mô hoại tử theo từng lớp (Tangential Excision) trong vòng 48 - 72 giờ đầu sau khi ổn định huyết động, sau đó phủ kín vết thương bằng **Ghép da tự thân (Autograft)** xẻ mỏng dạng mắt lưới (Mesh graft).

---

## 10. XỬ TRÍ CÁC THỂ BỎNG ĐẶC BIỆT: HÓA CHẤT, LUỒNG ĐIỆN & ĐƯỜNG HÔ HẤP

### 1. Bỏng Hóa Chất (Chemical Burns):
* **Nguyên tắc cốt tử:** Rửa trôi cơ học bằng nước sạch là yếu tố sống còn nhất. Phải rửa nước **ngay lập tức trong tối thiểu 30 - 60 phút**.
* **Bỏng Axit:** Gây hoại tử đông protein (Coagulative necrosis) tạo lớp vảy hạn chế thấm sâu.
* **Bỏng Bazơ (Kiềm):** Gây hoại tử hóa lỏng (Liquefactive necrosis), xà phòng hóa chất béo giúp kiềm ăn mòn liên tục vào mô rất sâu $\rightarrow$ Cần rửa nước liên tục từ 1 đến 2 giờ.
* **Các trường hợp đặc biệt:**
  * *Vôi sống (CaO):* Phải **quét và chải sạch toàn bộ bột vôi khô** trước khi xả nước. Nếu xả nước ngay khi còn bột vôi, phản ứng tôi vôi tỏa nhiệt lượng cực lớn ($CaO + H_2O \rightarrow Ca(OH)_2 + Nhiệt$) sẽ gây bỏng nhiệt kép nặng nề.
  * *Axit Flohydric (HF - dùng trong khắc kính, bán dẫn):* Ion $F^-$ thấm xuyên da hút sạch Canxi máu gây hoại tử xương và ngừng tim đột ngột do hạ Canxi. **Xử trí đặc hiệu:** Thoa liên tục gel **Calci Gluconate 2.5%** hoặc tiêm truyền dung dịch Calci Gluconate.
  * *Bỏng hóa chất vào mắt:* Xả nước rửa mắt liên tục ít nhất 30 phút bằng bộ rửa mắt khẩn cấp hoặc chai dịch truyền, giữ mở mi mắt.

### 2. Bỏng Điện (Electrical Burns):
* Tổn thương bên ngoài da tại điểm vào và điểm ra (Entrance & Exit wounds) thường rất nhỏ, nhưng **tổn thương mô cơ bên trong là khổng lồ** do nhiệt lượng sinh ra theo định luật Joule ($Q = I^2 \times R \times t$).
* **Biến chứng nguy hiểm:**
  * Loạn nhịp tim cấp tính (Rung thất, vô tâm thu) $\rightarrow$ Bắt buộc đo điện tim (ECG) 12 chuyển đạo và theo dõi monitor tim liên tục 24 giờ.
  * Hội chứng chèn ép khoang các chi $\rightarrow$ Phải rạch mở cân giải áp (Fasciotomy).
  * Tiêu cơ vân giải phóng Myoglobin $\rightarrow$ Suy thận cấp vô niệu.

### 3. Bỏng Đường Hô Hấp (Inhalation Injury):
* Xảy ra khi hít phải khói nóng, khí độc (CO, Cyanide, Khí Clo) trong không gian kín.
* **Triệu chứng gợi ý:** Bỏng vùng mặt quanh miệng mũi, lông mũi cháy xém, tiếng thở rít, khàn giọng, đờm có muội than đen.
* **Xử trí:**
  * Thở oxy 100% qua mặt nạ có túi dự trữ (giúp hạ thời gian bán hủy của Carboxyhemoglobin từ 320 phút xuống còn 80 phút).
  * **Đặt ống nội khí quản sớm (Prophylactic Intubation):** Khi nghi ngờ phù nề thanh quản, phải đặt ống nội khí quản ngay trước khi dây thanh âm phù nề bít kín hoàn toàn đường thở.

---

## 11. QUẢN LÝ ĐAU, KHÁNG SINH, DINH DƯỠNG & PHÒNG NGỪA UỐN VÁN

### 1. Quản Lý Cơn Đau Đa Mô Thức:
* Bỏng Độ 2 là một trong những chấn thương gây đau đớn khủng khiếp nhất trong y khoa.
* Bỏng nhẹ/vừa: Paracetamol (10 - 15 mg/kg mỗi 4 - 6h) kết hợp NSAIDs (Ibuprofen 400mg) nếu không có suy thận hoặc xuất huyết tiêu hóa.
* Bỏng nặng nhập viện: Sử dụng Opioids đường tĩnh mạch (Morphine hoặc Fentanyl), chuẩn độ liều theo thang điểm đau VAS (Visual Analog Scale). Không tiêm bắp do hấp thu thất thường trong giai đoạn sốc bỏng.

### 2. Chỉ Định Kháng Sinh Chuẩn Mực:
* **TUYỆT ĐỐI KHÔNG DÙNG KHÁNG SINH TOÀN THÂN ĐỂ PHÒNG NGỪA (Prophylactic Antibiotics) ở bệnh nhân bỏng mới:** Y văn thế giới đã chứng minh việc dùng kháng sinh dự phòng toàn thân không ngăn được nhiễm trùng mà chỉ làm chọn lọc các chủng vi khuẩn đa kháng thuốc (*MRSA, Acinetobacter baumannii, Pseudomonas aeruginosa kháng Carbapenem*).
* Kháng sinh toàn thân chỉ được dùng khi có bằng chứng lâm sàng rõ rệt của nhiễm trùng toàn thân: Sốt cao rét run, bạch cầu tăng vọt, vết bỏng chuyển màu đen thối, cấy máu hoặc cấy dịch mủ dương tính.

### 3. Dinh Dưỡng Hồi Phục Bỏng (Hypermetabolic State):
* Cơ thể bệnh nhân bỏng rơi vào trạng thái tăng chuyển hóa dị hóa cực độ (tiêu hao năng lượng có thể tăng tới 100 - 200%).
* Cần nuôi dưỡng sớm qua đường tiêu hóa (trong vòng 6 - 12 giờ đầu sau bỏng) bằng ống thông dạ dày nếu không tự ăn được.
* Chế độ ăn: Giàu Protein ($1.5 - 2.0\text{ g/kg/ngày}$ ở người lớn, $2.0 - 3.0\text{ g/kg/ngày}$ ở trẻ em) để tái tạo mô hạt, bổ sung Kẽm, Vitamin C, Vitamin A và Glutamine.

### 4. Phòng Ngừa Uốn Ván (Tetanus Prophylaxis):
* Vết bỏng là vết thương hở giàu protein hoại tử và thiếu oxy, môi trường hoàn hảo cho bào tử trực khuẩn uốn ván (*Clostridium tetani*) phát triển.
* Bệnh nhân chưa tiêm phòng uốn ván đầy đủ trong 5 năm gần nhất: Bắt buộc tiêm vắc-xin giải độc tố uốn ván (Tetanus Toxoid - TT) kết hợp Huyết thanh kháng độc tố uốn ván (Tetanus Immunoglobulin - TIG hoặc SAT 1,500 IU).

---

## 12. HÀNG RÀO KIỂM DUYỆT AN TOÀN CHO TRỢ LÝ ẢO (MEDICAL GUARDRAILS & ANTI-MYTH SYSTEM)

Hệ thống AI/RAG bắt buộc phải tích hợp bộ lọc phản xạ từ chối (Negative Rejection Filters) và cảnh báo đỏ ngay lập tức khi phát hiện người dùng hỏi hoặc có ý định áp dụng các phương pháp dân gian nguy hại:

```text
┌────────────────────────────────────────────────────────────────────────┐
│             BẢNG ĐỐI CHIẾU CÁC QUAN NIỆM DÂN GIAN SAI LẦM NGUY HIỂM    │
├─────────────────────────┬──────────────────────────────────────────────┤
│ PHƯƠNG PHÁP SAI LẦM     │ CƠ CHẾ NGUY HIỂM GÂY HẠI CHO BỆNH NHÂN       │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 1. Bôi kem đánh răng    │ Chứa kiềm và tinh dầu bạc hà làm rách mô     │
│                         │ tế bào, gây bỏng kiềm hóa học thứ phát.     │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 2. Đắp nước mắm, dấm    │ Dung dịch ưu trương đậm đặc hút nước mô,     │
│                         │ gây xót bỏng rát kinh hoàng và hoại tử tế bào│
├─────────────────────────┼──────────────────────────────────────────────┤
│ 3. Bôi mỡ trăn, dầu ăn  │ Giữ nhiệt lượng lại trong da không thoát ra  │
│                         │ được, tạo màng kỵ khí cho vi khuẩn uốn ván.  │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 4. Đắp lòng đỏ trứng gà,│ Đưa hàng triệu trực khuẩn mủ xanh và tụ cầu   │
│    tro bếp, bùn non     │ vào thẳng dòng máu, gây sốc nhiễm trùng chết │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 5. Chườm đá lạnh cục    │ Co mạch ngoại vi đột ngột, làm chết mô nuôi  │
│                         │ dưỡng, gây hoại tử bỏng lạnh (Frostbite).    │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 6. Tự ý chọc nốt phỏng  │ Phá hủy lớp màng vô trùng tự nhiên, biến vết │
│                         │ bỏng sạch thành ổ mủ áp-xe nhiễm khuẩn.      │
└─────────────────────────┴──────────────────────────────────────────────┘
```

### Nguyên Tắc Pháp Lý Miễn Trừ Trách Nhiệm (Medical AI Disclaimer):
Hệ thống AI chỉ mang tính chất **hỗ trợ sàng lọc, tính toán thông số kỹ thuật (TBSA, Parkland, Triage) và cung cấp thông tin sơ cứu chuẩn mực**. Hệ thống không thay thế quyết định chẩn đoán lâm sàng trực tiếp của Bác sĩ Chuyên khoa Bỏng được cấp chứng chỉ hành nghề. Trong mọi tình huống cấp cứu, tính mạng của bệnh nhân luôn là ưu tiên cao nhất thông qua việc liên hệ Tổng đài Cấp cứu 115 hoặc đưa tới cơ sở y tế gần nhất.

---

## 13. BẢNG TRA CỨU TOÀN BỘ NGUỒN DẪN HỌC THUẬT & VĂN BẢN CHÍNH THỨC

Dưới đây là các đường dẫn chính thức đến các cơ quan y tế đầu ngành phục vụ trích dẫn trong Báo cáo Đồ án tốt nghiệp và Cơ sở tri thức RAG:

| Cơ Quan Ban Hành | Tên Văn Bản / Tài Liệu | Loại Tài Liệu | Đường Dẫn Chính Thức |
| :--- | :--- | :--- | :--- |
| **Bộ Y Tế Việt Nam** | *Quy trình kỹ thuật khám chữa bệnh chuyên ngành Bỏng (QĐ 635/QĐ-BYT)* | Văn bản pháp quy | [Thư Viện Pháp Luật](https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-635-QD-BYT-Quy-trinh-ky-thuat-kham-benh-chua-benh-chuyen-nganh-Bong-276707.aspx) |
| **Viện Bỏng Quốc Gia** | *Phác đồ chẩn đoán, phân loại & xử trí vết thương bỏng lâm sàng* | Phác đồ chuyên khoa | [Cổng TT Viện Bỏng Lê Hữu Trác](https://vienbongquocgia.vn) |
| **WHO (Geneva)** | *Burn Prevention and Clinical Management Guidelines* | Hướng dẫn toàn cầu | [WHO Burns Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/burns) |
| **American Burn Association** | *Burn Center Referral Criteria & Guidelines for Care* | Tiêu chuẩn lâm sàng | [ABA Official Referral Criteria](https://ameriburn.org/education-resources/quality-burn-care/burn-center-referral-criteria/) |
| **American Burn Association** | *Advanced Burn Life Support (ABLS) Provider Manual* | Giáo trình cấp cứu | [ABA ABLS Program](https://ameriburn.org/education-resources/abls-program/) |
| **NHS (Vương Quốc Anh)** | *Burns and Scalds: Overview, First Aid and Management* | Hướng dẫn quốc gia | [NHS UK Burns & Scalds](https://www.nhs.uk/conditions/burns-and-scalds/) |
| **Mayo Clinic (Hoa Kỳ)** | *Burns: First aid, Clinical Diagnosis and Long-term Care* | Hướng dẫn y khoa | [Mayo Clinic Burns Guide](https://www.mayoclinic.org/first-aid/first-aid-burns/basics/art-20056649) |
| **ISBI (Quốc Tế)** | *ISBI Practice Guidelines for Burn Care (Burns Journal)* | Bài báo khoa học | [ISBI Guidelines via PubMed](https://pubmed.ncbi.nlm.nih.gov/27542295/) |
| **NCBI / StatPearls** | *Burn Resuscitation and Fluid Management (Parkland)* | Tổng quan chuyên sâu | [NCBI Bookshelf NBK482329](https://www.ncbi.nlm.nih.gov/books/NBK482329/) |
| **British Burn Association** | *First Aid Clinical Practice Guidelines for Burn Injuries* | Hướng dẫn lâm sàng | [British Burn Association](https://www.britishburnassociation.org/) |
