import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_multimodal_transforms(target_size=224, is_train=True):
    """构建支持多图像同步增强的 Albumentations pipeline"""
    if is_train:
        # 主变换：包含几何 + 光度增强
        base_transforms = [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=3,
                border_mode=0,  # 黑边填充
                p=0.5
            ),
            A.Resize(height=target_size, width=target_size),
        ]

        # 光度增强（每个模态独立应用）
        photometric = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=1.0)

        return base_transforms, photometric
    else:
        # 验证阶段：仅 Resize
        base_transforms = [A.Resize(height=target_size, width=target_size)]
        return base_transforms, None


def DicomProcess(ax_data, fs_data, t1_data, t2_data, is_train=True, target_size=224):
    """
    对四模态三通道数据进行同步增强和格式转换

    Args:
        ax_data, fs_data, t1_data, t2_data:  np.float32, value in [0,1]
        is_train: bool, 是否使用训练增强
        target_size: int, 目标尺寸

    Returns:
        四个 torch.Tensor, shape (C, target_size, target_size)
    """
    # Step 1: 获取变换
    base_transforms, photometric = get_multimodal_transforms(target_size, is_train)

    if is_train:
        # 构建支持多图像的 Compose（同步几何变换）
        base_compose = A.Compose(
            base_transforms,
            additional_targets={
                'image1': 'image',
                'image2': 'image',
                'image3': 'image'
            },
            p=1.0,
            is_check_shapes=False  # 关闭形状一致性检查
        )

        # 应用同步几何变换（包括 Resize）
        transformed = base_compose(
            image=ax_data,
            image1=fs_data,
            image2=t1_data,
            image3=t2_data
        )

        ax_geo = transformed['image']
        fs_geo = transformed['image1']
        t1_geo = transformed['image2']
        t2_geo = transformed['image3']

        # 各自独立应用光度增强（避免模态间干扰）
        ax_final = photometric(image=ax_geo)['image']
        fs_final = photometric(image=fs_geo)['image']
        t1_final = photometric(image=t1_geo)['image']
        t2_final = photometric(image=t2_geo)['image']

    else:
        # 验证阶段：只做 Resize（无随机性）
        resize_only = A.Compose(base_transforms, p=1.0)
        ax_final = resize_only(image=ax_data)['image']
        fs_final = resize_only(image=fs_data)['image']
        t1_final = resize_only(image=t1_data)['image']
        t2_final = resize_only(image=t2_data)['image']

    # Step 2: 转为 Tensor (HWC -> CHW)
    to_tensor = ToTensorV2()
    ax_tensor = to_tensor(image=ax_final)['image']
    fs_tensor = to_tensor(image=fs_final)['image']
    t1_tensor = to_tensor(image=t1_final)['image']
    t2_tensor = to_tensor(image=t2_final)['image']

    return ax_tensor, fs_tensor, t1_tensor, t2_tensor