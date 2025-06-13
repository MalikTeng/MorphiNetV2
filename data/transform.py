import torch
import numpy as np
from monai.transforms import (
    AdjustContrastd,
    Compose,
    LoadImaged,
    CropForegroundd,
    CopyItemsd,
    Orientationd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandZoomd,
    RandFlipd,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    EnsureTyped
)

from data.components import *

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple, modal: str, section: str, rotation: bool,
        crop_window_size: list, pixdim: list, spacing: float = 2.0,
        phase: str = "validation",  # "unet", "resnet", "gsn", "ndf", "validation"
        **kwargs
):
    """
    Conducting pre-transformation that comprises multichannel conversion,
    resampling in regard of space distance, reorientation, foreground cropping,
    normalization and data augmentation.
    
    :params
        keys: designated items for pre-transformation (image and label).
        modal: modality of data the pre-transformation applied to.
        section: identifier of either train, valid or test set.
        rotation: whether to apply rotation augmentation. @deprecated
        crop_window_size: image and label will be cropped to match the size of network input.
        pixdim: the spatial distance of the downsampled images and labels.
        spacing: target spacing for isotropic resampling.
        phase: current processing phase, determining which keys are generated.
    """
    target = kwargs.get("target")
    target = target.lower() if target is not None else None
    
    # data loading
    transforms = [
        LoadImaged(keys, ensure_channel_first=False if modal == "mr" and target != 'acdc' else True, image_only=True, allow_missing_keys=True),
    ]

    # pre-transformation
    if target == "acdc":
        # ACDC data is with different orientation
        transforms.extend([
            # isotropic resampling
            Adjustd(keys, allow_missing_keys=True, target="acdc"),
            Spacingd(keys, [-1, spacing, spacing],
                     mode=("bilinear", "nearest"), 
                     allow_missing_keys=True),
        ])
    else:
        transforms.extend([
            Adjustd(keys, allow_missing_keys=True),
            Spacingd(keys, 
                    [spacing] * 3 if modal == "ct" else [spacing, spacing, -1], 
                    mode=("bilinear", "nearest"), 
                    allow_missing_keys=True),
            Orientationd(keys, axcodes="RAS", allow_missing_keys=True),     # (D, W, H)
        ])

    # Determine if full data (including _ds and _df keys) is needed
    load_full_data = phase not in ["unet", "gsn"]

    if load_full_data:
        transforms.extend([
            CopyItemsd(keys[1], names=f"{keys[1]}_ds"),
            Spacingd(f"{keys[1]}_ds", [spacing] * 3,
                    mode="nearest", padding_mode="zeros"),
            CropForegroundd(f"{keys[1]}_ds", source_key=f"{keys[1]}_ds"),
            # create distance field from down-sampled label
            Maskd([f"{keys[1]}_ds", f"{keys[1][:2]}"], allow_missing_keys=True),
            FlexResized(
                f"{keys[1]}_ds", 
                (-1, crop_window_size[0], -1)
                ),
            Resized(
                f"{keys[1]}_ds", 
                int(crop_window_size[0] // pixdim[0]), 
                size_mode="longest", mode="nearest-exact"
                ),
            ResizeWithPadOrCropd(
                f"{keys[1]}_ds", 
                int(crop_window_size[0] // pixdim[0]), 
                mode="constant", value=0
                ),
            DFConvertd(f"{keys[1]}_ds"),
        ])

    # keys_to_ensure = list(keys)
    # if load_full_data:
    #     keys_to_ensure.extend([f"{keys[0][:2]}_df", f"{keys[1]}_ds"])

    if section == "train":
        # if rotation:
        #     transforms.extend([
        #         # spatial augmentation
        #         RandRotate90d(keys, prob=0.5, spatial_axes=(1, 2), lazy=True),
        #         RandFlipd(keys, prob=0.5, spatial_axis=[1], lazy=True),
        #         RandFlipd(keys, prob=0.5, spatial_axis=[2], lazy=True),
        #         RandZoomd(
        #             keys,
        #             min_zoom=0.7 if modal == "ct" else [1.0, 0.7, 0.7], 
        #             max_zoom=1.4 if modal == "ct" else [1.0, 1.4, 1.4],
        #             mode=("trilinear", "nearest-exact"),
        #             align_corners=(True, None), prob=0.15, lazy=True,
        #         ),
        #         RandGaussianNoised(keys[0], std=0.01, prob=0.15),
        #         RandGaussianSmoothd(
        #             keys[0], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
        #             sigma_z=(0.5, 1.15), prob=0.15,
        #         ),
        #         RandAdjustContrastd(keys[0], gamma=(0.65, 1.5), prob=0.15),
        #         RandScaleIntensityd(keys[0], factors=0.3, prob=0.15),
        #         # normalize the image intensity to 0-1
        #         ScaleIntensityRangePercentilesd(keys[0], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True),
        #     ])
        #     float_keys_train_rot = [keys[0]]
        #     int_keys_train_rot = [keys[1]]
        #     if load_full_data:
        #         float_keys_train_rot.append(f"{keys[0][:2]}_df")
        #         int_keys_train_rot.append(f"{keys[1]}_ds")
        #     transforms.extend([
        #         EnsureTyped(float_keys_train_rot, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
        #         EnsureTyped(int_keys_train_rot, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        #     ])
        # else:
        transforms.extend([
            # spatial augmentation
            RandZoomd(
                keys,
                min_zoom=0.7 if modal == "ct" else [1.0, 0.7, 0.7], 
                max_zoom=1.4 if modal == "ct" else [1.0, 1.4, 1.4],
                mode=("trilinear", "nearest-exact"),
                align_corners=(True, None), prob=0.15,
            ),
            RandGaussianNoised(keys[0], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys[0], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15), prob=0.15,
            ),
            RandAdjustContrastd(keys[0], gamma=(0.65, 1.5), prob=0.15),
            RandScaleIntensityd(keys[0], factors=0.3, prob=0.15),
            # normalize the image intensity to 0-1
            ScaleIntensityRangePercentilesd(keys[0], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True),
        ])
        float_keys_train_no_rot = [keys[0]]
        int_keys_train_no_rot = [keys[1]]
        if load_full_data:
            float_keys_train_no_rot.append(f"{keys[0][:2]}_df")
            int_keys_train_no_rot.append(f"{keys[1]}_ds")
        transforms.extend([
            EnsureTyped(float_keys_train_no_rot, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(int_keys_train_no_rot, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        ])
    else: # "valid" or "test" section
        transforms.extend([
            ScaleIntensityRangePercentilesd(keys[0], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True),
        ])
        float_keys_valid = [keys[0]]
        int_keys_valid = [keys[1]]
        if load_full_data:
            float_keys_valid.append(f"{keys[0][:2]}_df")
            int_keys_valid.append(f"{keys[1]}_ds")
        transforms.extend([
            EnsureTyped(float_keys_valid, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(int_keys_valid, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        ])

    return Compose(transforms)

