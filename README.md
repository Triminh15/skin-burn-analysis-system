# Skin Burn Analysis System (Hệ thống Phân tích Vết bỏng Đa phương thái)

Hệ thống ứng dụng Trí tuệ Nhân tạo hỗ trợ phân vùng, đánh giá mức độ tổn thương do bỏng dựa trên Thị giác Máy tính (Deep Learning), kết hợp Hệ thống RAG (Retrieval-Augmented Generation) và AI Agent hỗ trợ tham vấn sơ cứu y khoa an toàn.

---

## 📂 Kiến trúc Cấu trúc Dự án

```text
burn_analysis/
│
├── 🚀 app.py                         # Điểm khởi chạy Web Demo (Streamlit / Gradio)
├── ⚙️ config.py                      # Quản lý cấu hình tập trung (Hyperparameters, Paths, Model weights, API Keys)
├── 🔐 .env.example                   # Mẫu khai báo biến môi trường (GEMINI_API_KEY)
├── 📦 requirements.txt               # Danh sách phiên bản thư viện cố định (PyTorch, SMP, Albumentations, ChromaDB...)
├── 🐳 Dockerfile                     # Container hóa hệ thống
├── 📖 README.md                      # Tài liệu giới thiệu, hướng dẫn cài đặt và kết quả thực nghiệm
├── 🚫 .gitignore                     # Chặn commit thư mục dữ liệu nặng, file checkpoints và cache
│
├── 📁 data/                          # ================= QUẢN LÝ DỮ LIỆU THỊ GIÁC =================
│   ├── raw/                          # Dữ liệu ảnh gốc & mask tải về (BIP-US / Kaggle)
│   ├── processed/                    # Dữ liệu sau resize và chuẩn hóa (gồm 2 thư mục images/ và masks/)
│   │   ├── images/
│   │   └── masks/
│   ├── splits/                       # File CSV phân chia tập dữ liệu (không nhân bản file ảnh)
│   │   ├── train.csv                 # image_path, mask_path, severity_degree, body_part
│   │   ├── val.csv
│   │   └── test.csv
│   └── reference/                    # Ảnh các vật đo chuẩn (nếu có dùng vật quy chiếu cm)
│
├── 📓 notebooks/                     # ================= THỬ NGHIỆM & TRỰC QUAN HÓA =================
│   ├── 01_eda_dataset.ipynb          # Khám phá dữ liệu, kiểm tra phân phối nhãn và kiểm tra mask
│   ├── 02_check_augmentation.ipynb   # Kiểm tra trực quan hóa các biến thể ảnh và mask sau Albumentations
│   └── 03_pipeline_evaluation.ipynb  # Đánh giá trực quan kết quả U-Net + ResNet trên ảnh mẫu
│
├── 📚 knowledge_base/                # ================= KHO TRI THỨC Y KHOA RAG =================
│   ├── guidelines/                   # Tài liệu y văn chính thống (.pdf, .md từ Bộ Y Tế, ABA)
│   │   └── phac_do_bo_y_te.pdf
│   └── chroma_db/                    # Thư mục vector database lưu trữ cục bộ (ChromaDB Persistent)
│
├── 🧠 src/                           # ================= MÃ NGUỒN CỐT LÕI =================
│   ├── __init__.py
│   │
│   ├── 👁️ vision/                    # --- PHÂN TẦNG THỊ GIÁC MÁY TÍNH (TRỌNG TÂM ĐỒ ÁN) ---
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset nạp ảnh, mask và nhãn từ CSV
│   │   ├── transforms.py             # Pipeline tăng cường dữ liệu Albumentations (Train vs Val/Test)
│   │   ├── models.py                 # Định nghĩa smp.Unet (ResNet-34) & ResNet-50 Classifier Head
│   │   ├── losses.py                 # ComboLoss: 0.5 BCEWithLogits + 0.5 Soft Dice Loss
│   │   ├── metrics.py                # Tính toán IoU, Dice, Accuracy, F1-score, Confusion Matrix
│   │   ├── postprocessing.py         # Cắt ROI vết bỏng kèm padding, tính tỷ lệ % tổn thương
│   │   ├── train_seg.py              # Script huấn luyện U-Net (kèm Early Stopping & Checkpoint Manager)
│   │   └── train_cls.py              # Script huấn luyện mô hình phân loại ResNet-50
│   │
│   ├── 🔍 rag/                       # --- PHÂN TẦNG TRUY XUẤT Y KHOA (RAG) ---
│   │   ├── __init__.py
│   │   ├── indexer.py                # Bóc tách văn bản PDF, chia chunk và nạp vào ChromaDB
│   │   ├── retriever.py              # Truy xuất kết hợp Hybrid Search (Dense + BM25) có Metadata Filter
│   │   └── guardrails.py             # Negative Rejection Filter: Chặn đứng bài thuốc dân gian độc hại
│   │
│   ├── 🤖 agent/                     # --- PHÂN TẦNG ĐIỀU PHỐI ĐA PHƯƠNG THÁI ---
│   │   ├── __init__.py
│   │   ├── prompt_templates.py       # System Prompt Bác sĩ Bỏng & Cấu trúc chỉ dẫn 4 bước sơ cứu
│   │   ├── llm_client.py             # Client gọi Gemini API (hỗ trợ streaming và xử lý lỗi)
│   │   └── clinical_chain.py         # Luồng khép kín: Nhận ảnh -> Vision -> Triage -> RAG -> Lời khuyên
│   │
│   └── 🛠️ utils/                      # --- TIỆN ÍCH HỆ THỐNG & GIẢI THÍCH MÔ HÌNH ---
│       ├── __init__.py
│       ├── gradcam.py                # Bản đồ nhiệt Grad-CAM giải thích vùng kích hoạt của ResNet
│       ├── visualization.py          # Vẽ ảnh phủ Mask (Overlay Alpha Blending), vẽ ma trận nhầm lẫn
│       └── privacy.py                # Xóa siêu dữ liệu EXIF máy ảnh, tự động làm mờ mặt (HIPAA)
│
├── 💾 outputs/                       # ================= KẾT QUẢ HUẤN LUYỆN =================
│   ├── checkpoints/                  # Lưu trọng số mô hình tốt nhất (.pth)
│   └── figures/                      # Lưu đồ thị Loss curves, ma trận nhầm lẫn và ảnh Grad-CAM
│
└── 🧪 tests/                         # ================= BỘ KIỂM THỬ TỰ ĐỘNG =================
    ├── test_pipeline.py              # Kiểm tra tính toàn vẹn của tensor từ Vision sang Classifier
    └── test_safety.py                # Kiểm tra bộ lọc chặn các câu hỏi về mẹo vặt phản khoa học
```

---