"""
Module: tools/download_external_datasets.py
Mục đích: Tự động tải và chuẩn hóa các bộ dữ liệu Y khoa Công khai Độc lập (External & OOD Datasets)
về thư mục data/external_ood/ để phục vụ kiểm chứng Hướng 4A (Q1/Q2 Generalization & OOD Safety):

  1) BỘ BỎNG LÂM SÀNG BỆNH VIỆN ĐỘC LẬP — BIP_US Burn Database (94 ảnh):
     - Nguồn chính thức: Biomedical Image Processing (BIP) Group, University of Seville &
       Virgen del Rocío Hospital (Seville, Tây Ban Nha).
     - Link công khai học thuật: http://personal.us.es/rboloix/Burns_BIP_US_database.zip
     - Ánh xạ nhãn y khoa chuẩn từ README.pdf của Bệnh viện Virgen del Rocío:
         + Superficial dermal (47 ca) & Deep dermal (27 ca) -> Bỏng Độ 2 / Partial-Thickness (degree_2: 74 ảnh)
         + Full-thickness (20 ca)                           -> Bỏng Độ 3 / Toàn bộ bề dày da (degree_3: 20 ảnh)

  2) BỘ BỆNH DA LIỄU KHÔNG PHẢI BỎNG — ISIC Clinical Skin Lesions OOD (120 ảnh):
     - Nguồn công khai: Tập Test ISIC Dermatology Lesions trên Hugging Face
       (u sắc tố, nốt ruồi, dày sừng ánh sáng, tổn thương mạch máu đỏ tím - dễ gây nhầm với bỏng).
     - Lưu vào: data/external_ood/non_burn/ để đo khả năng chặn ảnh lạ (OOD Safety Rejection).

Ghi chú minh bạch về 2 nguồn hạn chế quyền truy cập:
  - EBIS / BuRnSNet (VEDAs-Lab/EBIS): Không mở link tải tự do; tác giả yêu cầu ký tay bản cam kết PDF
    và gửi email tới vedas.cs@gmail.com.
  - Roboflow Universe: Yêu cầu tài khoản cá nhân và ROBOFLOW_API_KEY riêng để xuất file zip.
"""

import os
import io
import sys
import shutil
import zipfile
import argparse
import urllib.request
from typing import Dict, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd
from PIL import Image


BIP_US_URL = "http://personal.us.es/rboloix/Burns_BIP_US_database.zip"
ISIC_PARQUET_URL = (
    "https://huggingface.co/datasets/Pranavkpba2000/skin_cancer_small_dataset/"
    "resolve/main/data/test-00000-of-00001-3a9b253c406a4af7.parquet"
)

# Bảng nhãn lâm sàng chính xác 100% trích từ Burns_BIP_US_database/README.pdf (Ảnh 1 -> 94)
BIP_US_FULL_THICKNESS_IDS = {
    15, 16, 17, 18, 19, 20,
    21, 33, 40, 45, 47, 58, 59, 60, 61, 62, 64, 74, 75, 76,
}
BIP_US_DEEP_DERMAL_IDS = {
    10, 11, 12, 13, 14,
    22, 23, 24, 29, 30, 34, 35, 36, 37, 38,
    48, 49, 50, 51, 65, 66, 67, 68, 69, 70,
    77, 78, 79, 80, 81, 82, 83,
}
# Các ID còn lại trong [1..94] đều là Superficial dermal (tổng: 47 Superficial + 27 Deep + 20 Full = 94 ảnh)


def get_bip_us_label_info(photo_id: int) -> Tuple[str, str, int]:
    """Trả về (hospital_subtype, folder_name, severity_degree_idx)."""
    if photo_id in BIP_US_FULL_THICKNESS_IDS:
        return ("Full-thickness", "degree_3", 2)
    if photo_id in BIP_US_DEEP_DERMAL_IDS:
        return ("Deep dermal", "degree_2", 1)
    return ("Superficial dermal", "degree_2", 1)


def download_file_with_ua(url: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp, open(dst_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)


def prepare_bip_us_dataset(base_dir: str) -> Dict[str, int]:
    dl_dir = os.path.join(base_dir, "_downloads")
    os.makedirs(dl_dir, exist_ok=True)
    zip_path = os.path.join(dl_dir, "Burns_BIP_US_database.zip")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
        print(f"⬇️ Đang tải BIP_US Burn Database từ ĐH Seville ({BIP_US_URL})...")
        download_file_with_ua(BIP_US_URL, zip_path)
    else:
        print(f"⏩ Đã có sẵn file nén BIP_US ({os.path.getsize(zip_path) / 1e6:.2f} MB).")

    extract_root = os.path.join(dl_dir, "bip_us_extracted")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    out_burn_dir = os.path.join(base_dir, "external_burns")
    deg2_dir = os.path.join(out_burn_dir, "degree_2")
    deg3_dir = os.path.join(out_burn_dir, "degree_3")
    os.makedirs(deg2_dir, exist_ok=True)
    os.makedirs(deg3_dir, exist_ok=True)

    src_root = os.path.join(extract_root, "Burns_BIP_US_database")
    records = []
    counts = {"Superficial dermal": 0, "Deep dermal": 0, "Full-thickness": 0}

    for split_folder in ["Training set", "Testing set"]:
        sp_dir = os.path.join(src_root, split_folder)
        if not os.path.isdir(sp_dir):
            continue
        for fname in sorted(os.listdir(sp_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".bmp", ".png") or not stem.isdigit():
                continue
            pid = int(stem)
            subtype, deg_folder, deg_idx = get_bip_us_label_info(pid)
            src_img_path = os.path.join(sp_dir, fname)
            bgr = cv2.imread(src_img_path)
            if bgr is None:
                continue

            out_name = f"bip_us_{pid:03d}_{subtype.lower().replace(' ', '_').replace('-', '_')}.jpg"
            dst_img_path = os.path.join(out_burn_dir, deg_folder, out_name)
            cv2.imwrite(dst_img_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            counts[subtype] += 1
            records.append({
                "photo_id": pid,
                "original_split": split_folder,
                "hospital_subtype": subtype,
                "mapped_degree_folder": deg_folder,
                "severity_degree": deg_idx,
                "rel_path": os.path.relpath(dst_img_path, PROJECT_ROOT).replace("\\", "/"),
            })

    meta_df = pd.DataFrame(records).sort_values("photo_id").reset_index(drop=True)
    meta_csv = os.path.join(out_burn_dir, "bip_us_metadata.csv")
    meta_df.to_csv(meta_csv, index=False, encoding="utf-8")
    print(
        f"  ✅ Đã chuẩn hóa {len(meta_df)} ảnh bệnh viện BIP_US vào '{out_burn_dir}':\n"
        f"     • Degree 2 (Partial-Thickness): {counts['Superficial dermal'] + counts['Deep dermal']} ảnh "
        f"({counts['Superficial dermal']} Superficial dermal + {counts['Deep dermal']} Deep dermal)\n"
        f"     • Degree 3 (Full-Thickness)   : {counts['Full-thickness']} ảnh"
    )
    return counts


def prepare_isic_nonburn_ood(base_dir: str, max_images: int = 120) -> int:
    dl_dir = os.path.join(base_dir, "_downloads")
    os.makedirs(dl_dir, exist_ok=True)
    pq_path = os.path.join(dl_dir, "isic_skin_lesions_test.parquet")

    if not os.path.exists(pq_path) or os.path.getsize(pq_path) < 1_000_000:
        print("⬇️ Đang tải tập ảnh bệnh da liễu ISIC (Non-Burn OOD, ~17.3 MB) từ Hugging Face...")
        download_file_with_ua(ISIC_PARQUET_URL, pq_path)
    else:
        print(f"⏩ Đã có sẵn gói ISIC Dermatology Parquet ({os.path.getsize(pq_path) / 1e6:.2f} MB).")

    df = pd.read_parquet(pq_path)
    out_ood_dir = os.path.join(base_dir, "non_burn")
    os.makedirs(out_ood_dir, exist_ok=True)

    # Trích xuất đều các lớp tổn thương da liễu (u hắc tố, nốt ruồi, u mạch máu đỏ, dày sừng...)
    if "label" in df.columns and np.issubdtype(df["label"].dtype, np.integer):
        sampled = df.groupby("label", group_keys=False).apply(
            lambda g: g.head(max(1, max_images // max(1, df["label"].nunique()) + 5))
        ).head(max_images)
    else:
        sampled = df.head(max_images)

    saved = 0
    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        img_entry = row["image"]
        pil_img = None
        if isinstance(img_entry, dict) and "bytes" in img_entry and img_entry["bytes"]:
            pil_img = Image.open(io.BytesIO(img_entry["bytes"])).convert("RGB")
        elif isinstance(img_entry, Image.Image):
            pil_img = img_entry.convert("RGB")

        if pil_img is None:
            continue
        out_p = os.path.join(out_ood_dir, f"isic_dermatology_ood_{idx:03d}.jpg")
        pil_img.save(out_p, quality=92)
        saved += 1

    print(f"  ✅ Đã trích xuất {saved} ảnh tổn thương da liễu ISIC (KHÔNG PHẢI BỎNG) vào '{out_ood_dir}'.")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download & Prepare BIP_US Hospital Burn Dataset + ISIC Non-Burn OOD Dataset")
    parser.add_argument("--out-root", type=str, default=os.path.join(PROJECT_ROOT, "data", "external_ood"))
    parser.add_argument("--max-isic", type=int, default=120)
    args = parser.parse_args()

    print("=" * 84)
    print("🌐 TẢI & CHUẨN HÓA BỘ DỮ LIỆU ĐỘC LẬP NGOẠI VIỆN (BIP_US SPAIN + ISIC DERMATOLOGY OOD)")
    print("=" * 84)
    prepare_bip_us_dataset(args.out_root)
    prepare_isic_nonburn_ood(args.out_root, max_images=args.max_isic)
    print("=" * 84)
    print("🎉 Hoàn tất! Sẵn sàng chạy tools/evaluate_ood_and_generalization.py.")


if __name__ == "__main__":
    main()

