"""
Module: src/vision/transforms.py
Mục đích: Xây dựng pipeline tăng cường dữ liệu (Data Augmentation) bằng Albumentations
đảm bảo đồng bộ 100% giữa Ảnh và Mask vết bỏng (chuẩn Albumentations 2.0+).
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Giá trị chuẩn hóa tiêu chuẩn của ImageNet (phù hợp với các pre-trained backbone như ResNet-34)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 256) -> A.Compose:
    """
    Pipeline tăng cường dữ liệu cho tập Train:
    - Resize về kích thước chuẩn (256x256)
    - Lật ngang/dọc, xoay nhẹ góc tự do
    - Biến đổi hình học Affine (co giãn, tịnh tiến, xoay)
    - Chỉnh sắc thái ánh sáng, độ tương phản (mô phỏng camera điện thoại và ánh sáng bệnh viện)
    - Thêm nhiễu nhẹ Gaussian mô phỏng ảnh chụp mờ nhòe
    - Chuẩn hóa ImageNet & chuyển sang PyTorch Tensor (3, H, W)
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # Biến đổi hình học đồng bộ (Spatial Transformations)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.06, 0.06),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            # Biến đổi sắc thái màu & quang học (Pixel-level Transformations)
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=0.3,
            ),
            # Chuẩn hóa ImageNet & ToTensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """
    Pipeline cho tập Validation và Test:
    - CHỈ Resize và Chuẩn hóa (Normalize)
    - Tuyệt đối KHÔNG lật hay biến dạng ảnh để đảm bảo tính khách quan khi chấm điểm
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Hàm tiện ích giải chuẩn hóa Tensor (C, H, W) về ảnh RGB (H, W, C) dải [0, 255]
    dùng để vẽ và trực quan hóa kết quả cho người xem.
    """
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)
