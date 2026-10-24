"""
Module: src/vision/dataset.py
Mục đích: Định nghĩa PyTorch Dataset (SkinBurnDataset) và DataLoader
để nạp và chuẩn hóa ảnh, mask cùng nhãn cấp độ bỏng từ các file CSV.
"""

import os
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from src.vision.transforms import get_train_transforms, get_val_transforms


class SkinBurnDataset(Dataset):
    """
    Dataset nạp ảnh vết bỏng và mask nhị phân cho PyTorch:
    - Đọc đường dẫn từ train.csv / val.csv / test.csv
    - Áp dụng các phép tăng cường dữ liệu đồng thời trên Ảnh và Mask
    - Đưa Mask về dạng nhị phân [0.0, 1.0] kích thước (1, H, W)
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str = "d:/burn_analysis",
        transform: Optional[A.Compose] = None,
    ):
        self.root_dir = root_dir
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(root_dir, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file danh sách dữ liệu: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Đường dẫn file ảnh và mask
        img_rel_path = row["image_path"]
        mask_rel_path = row["mask_path"]

        img_full_path = os.path.join(self.root_dir, img_rel_path)
        mask_full_path = os.path.join(self.root_dir, mask_rel_path)

        # 1. Đọc ảnh RGB
        img = cv2.imread(img_full_path)
        if img is None:
            raise FileNotFoundError(f"Lỗi khi đọc file ảnh: {img_full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Đọc mask nhị phân (ảnh Grayscale)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Lỗi khi đọc file mask: {mask_full_path}")

        # Chuẩn hóa mask về giá trị nhị phân 0 hoặc 1 (dạng float32)
        mask = (mask > 127).astype(np.float32)

        # 3. Áp dụng Data Augmentation (đồng bộ cả ảnh và mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_transformed = augmented["mask"]
        else:
            # Dự phòng nếu không truyền transform
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_transformed = torch.from_numpy(mask).float()

        # Định hình mask về shape (1, H, W) chuẩn cho hàm loss của U-Net
        if isinstance(mask_transformed, torch.Tensor):
            if mask_transformed.ndim == 2:
                mask_tensor = mask_transformed.unsqueeze(0).float()
            else:
                mask_tensor = mask_transformed.float()
        else:
            mask_tensor = torch.from_numpy(mask_transformed).unsqueeze(0).float()

        # Thông tin nhãn y khoa đi kèm
        severity = torch.tensor(int(row["severity_degree"]), dtype=torch.long)
        burn_pct = torch.tensor(float(row.get("burn_area_percentage", 0.0)), dtype=torch.float32)

        return {
            "image": img_tensor,            # Tensor (3, H, W)
            "mask": mask_tensor,            # Tensor (1, H, W)
            "severity": severity,          # Tensor scalar (0: Độ 1, 1: Độ 2, 2: Độ 3)
            "burn_percentage": burn_pct,    # Tensor scalar (% diện tích)
            "image_id": str(row["image_id"]),
        }


def get_dataloaders(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    root_dir: str = "d:/burn_analysis",
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Hàm tiện ích tạo nhanh cặp DataLoader cho Train và Validation.
    Lưu ý: Trên Windows, num_workers=0 là an toàn và tránh lỗi xung đột multiprocessing.
    """
    train_dataset = SkinBurnDataset(
        csv_path=train_csv,
        root_dir=root_dir,
        transform=get_train_transforms(img_size=img_size),
    )

    val_dataset = SkinBurnDataset(
        csv_path=val_csv,
        root_dir=root_dir,
        transform=get_val_transforms(img_size=img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
