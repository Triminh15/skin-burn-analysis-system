"""
Module: src/vision/dataset_multitask.py
Mục đích: Định nghĩa PyTorch Dataset (SkinBurnMultiTaskDataset) và DataLoader
cho mô hình đa nhiệm MT-EANet-EDL:
- Nạp ảnh gốc toàn cảnh (Full Image)
- Nạp mặt nạ tổn thương chuẩn (Ground-Truth Lesion Mask)
- Trích xuất tự động mặt nạ đường viền ban đỏ ngoại vi (Boundary Mask)
  bằng toán tử hình thái học: B_GT = Dilation(M_GT) - Erosion(M_GT)
- Nạp nhãn cấp độ bỏng (Severity Degree: 0, 1, 2) và % diện tích tổn thương
- Áp dụng Data Augmentation đồng bộ qua Albumentations
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.transforms import get_train_transforms, get_val_transforms


def derive_boundary_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """
    Tạo Ground-Truth ranh giới ban đỏ tổn thương (Peripheral Erythema Boundary Mask)
    từ mặt nạ nhị phân gốc bằng phép toán hình thái học (Morphological Gradient):
        B_GT = Dilation(M_GT) - Erosion(M_GT)

    Args:
        mask: Ma trận numpy 2D (H, W), giá trị nhị phân {0, 1} hoặc {0, 255}
        kernel_size: Kích thước phần tử cấu trúc (Structuring Element), mặc định 5x5
        iterations: Số lần lặp giãn nở và co ngót (mặc định 1)

    Returns:
        boundary: Ma trận numpy 2D (H, W) kiểu float32, giá trị {0.0, 1.0}
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
        was_tensor = True
    else:
        mask_np = mask
        was_tensor = False

    # Chuyển về ma trận nhị phân uint8 {0, 1}
    if mask_np.max() > 1.0:
        mask_u8 = (mask_np > 127).astype(np.uint8)
    else:
        mask_u8 = (mask_np > 0.5).astype(np.uint8)

    # Sử dụng kernel hình elip/tròn để đường viền mềm mại, tự nhiên
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_u8, kernel, iterations=iterations)
    eroded = cv2.erode(mask_u8, kernel, iterations=iterations)

    # Phép trừ số học tạo viền ban đỏ
    boundary = (dilated - eroded).astype(np.float32)

    if was_tensor:
        return torch.from_numpy(boundary).to(mask.device, dtype=mask.dtype)
    return boundary


class SkinBurnMultiTaskDataset(Dataset):
    """
    Dataset đa nhiệm cho bài toán phân tích vết bỏng:
    Cung cấp đồng thời:
    - Ảnh RGB toàn cảnh (3, H, W)
    - Mặt nạ phân đoạn tổn thương (1, H, W)
    - Mặt nạ viền ban đỏ ngoại vi (1, H, W)
    - Nhãn cấp độ bỏng (scalar long: 0, 1, 2)
    - Phần trăm diện tích vết bỏng (% TBSA)
    - Image ID
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        boundary_kernel_size: int = 5,
    ):
        self.root_dir = root_dir if root_dir is not None else PROJECT_ROOT
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.root_dir, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file danh sách dữ liệu: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.boundary_kernel_size = boundary_kernel_size
        self.transform = transform

        # Lọc bỏ các mẫu thiếu file ảnh hoặc mask
        if "image_path" in self.df.columns and "mask_path" in self.df.columns:
            valid = self.df.apply(
                lambda r: os.path.exists(os.path.join(self.root_dir, str(r["image_path"])))
                and os.path.exists(os.path.join(self.root_dir, str(r["mask_path"]))),
                axis=1,
            )
            self.df = self.df[valid].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_rel_path = str(row["image_path"])
        mask_rel_path = str(row["mask_path"])

        img_full_path = os.path.join(self.root_dir, img_rel_path)
        mask_full_path = os.path.join(self.root_dir, mask_rel_path)

        # 1. Đọc ảnh RGB
        img = cv2.imread(img_full_path)
        if img is None:
            raise FileNotFoundError(f"Lỗi khi đọc file ảnh: {img_full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Đọc mask nhị phân (Grayscale)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Lỗi khi đọc file mask: {mask_full_path}")

        mask = (mask > 127).astype(np.float32)

        # 3. Áp dụng Biến đổi Tăng cường Dữ liệu (Đồng bộ ảnh và mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_transformed = augmented["mask"]
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_transformed = torch.from_numpy(mask).float()

        # Đảm bảo mask_transformed là dạng numpy 2D để trích xuất boundary chính xác
        if isinstance(mask_transformed, torch.Tensor):
            mask_np_2d = mask_transformed.squeeze().cpu().numpy()
            mask_tensor = mask_transformed if mask_transformed.ndim == 3 else mask_transformed.unsqueeze(0).float()
        else:
            mask_np_2d = mask_transformed
            mask_tensor = torch.from_numpy(mask_transformed).unsqueeze(0).float()

        # Đảm bảo nhị phân hóa rõ ràng
        mask_np_2d = (mask_np_2d > 0.5).astype(np.float32)

        # 4. Trích xuất Mặt nạ Viền Ban Đỏ (Boundary Mask) từ mask đã biến đổi
        boundary_np = derive_boundary_mask(
            mask_np_2d,
            kernel_size=self.boundary_kernel_size,
            iterations=1
        )
        boundary_tensor = torch.from_numpy(boundary_np).unsqueeze(0).float()  # (1, H, W)

        # 5. Nhãn độ sâu bỏng (Severity Degree: 0, 1, 2) và % diện tích
        severity = torch.tensor(int(row["severity_degree"]), dtype=torch.long)
        burn_pct = torch.tensor(float(row.get("burn_area_percentage", 0.0)), dtype=torch.float32)

        return {
            "image": img_tensor,            # Tensor (3, H, W)
            "mask": mask_tensor,            # Tensor (1, H, W) - Ground Truth Mask
            "boundary": boundary_tensor,    # Tensor (1, H, W) - Peripheral Boundary Mask
            "severity": severity,          # Tensor scalar (0: Độ 1, 1: Độ 2, 2: Độ 3)
            "burn_percentage": burn_pct,    # Tensor scalar (% diện tích)
            "image_id": str(row["image_id"]),
        }


def get_multitask_dataloaders(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: Optional[str] = "data/splits/test.csv",
    root_dir: Optional[str] = None,
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    boundary_kernel_size: int = 5,
) -> Tuple[DataLoader, ...]:
    """
    Hàm tiện ích tạo DataLoader cho mô hình đa nhiệm:
    - Nếu test_csv được cung cấp và tồn tại: Trả về (train_loader, val_loader, test_loader)
    - Nếu không: Trả về (train_loader, val_loader)
    """
    root = root_dir if root_dir is not None else PROJECT_ROOT

    train_dataset = SkinBurnMultiTaskDataset(
        csv_path=train_csv,
        root_dir=root,
        transform=get_train_transforms(img_size=img_size),
        boundary_kernel_size=boundary_kernel_size,
    )

    val_dataset = SkinBurnMultiTaskDataset(
        csv_path=val_csv,
        root_dir=root,
        transform=get_val_transforms(img_size=img_size),
        boundary_kernel_size=boundary_kernel_size,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if test_csv is not None:
        test_path = os.path.join(root, test_csv) if not os.path.isabs(test_csv) else test_csv
        if os.path.exists(test_path):
            test_dataset = SkinBurnMultiTaskDataset(
                csv_path=test_csv,
                root_dir=root,
                transform=get_val_transforms(img_size=img_size),
                boundary_kernel_size=boundary_kernel_size,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader, test_loader

    return train_loader, val_loader


if __name__ == "__main__":
    print("--- Testing Unit Tests for src/vision/dataset_multitask.py ---")

    # 1. Kiểm tra derive_boundary_mask với mask giả lập
    synthetic_mask = np.zeros((100, 100), dtype=np.uint8)
    synthetic_mask[20:80, 20:80] = 1

    bnd = derive_boundary_mask(synthetic_mask, kernel_size=5)
    assert bnd.shape == (100, 100), f"Sai kích thước boundary: {bnd.shape}"
    assert bnd.max() == 1.0 and bnd.min() == 0.0, "Boundary mask không đúng dải [0, 1]"
    # Tâm của mask (50, 50) nằm sâu bên trong nên boundary phải bằng 0
    assert bnd[50, 50] == 0.0, "Tâm vùng tổn thương không được có viền!"
    # Vùng viền quanh (20, 50) phải bằng 1
    assert bnd[20, 50] == 1.0, "Ranh giới ngoài phải có giá trị 1.0"
    print(f"[OK] Morphological Boundary generation test passed: active_pixels={int(bnd.sum())}")

    # 2. Kiểm tra nạp dữ liệu thật nếu có train.csv
    sample_csv = os.path.join(PROJECT_ROOT, "data/splits/train.csv")
    if os.path.exists(sample_csv):
        ds = SkinBurnMultiTaskDataset(csv_path="data/splits/train.csv", boundary_kernel_size=5)
        print(f"[OK] MultiTask Dataset loaded successfully with {len(ds)} samples.")
        sample = ds[0]
        print(f"[OK] Sample image shape: {sample['image'].shape}")
        print(f"[OK] Sample mask shape: {sample['mask'].shape}")
        print(f"[OK] Sample boundary shape: {sample['boundary'].shape}")
        print(f"[OK] Sample severity degree: {sample['severity'].item()} (type: {sample['severity'].dtype})")
        print(f"[OK] Sample burn area %: {sample['burn_percentage'].item():.2f}%")
        print(f"[OK] Sample image_id: {sample['image_id']}")
    else:
        print("[WARN] train.csv not found for live dataset test, skipped live test.")

    print("--- All Sanity Checks in dataset_multitask.py Passed! ---")
