import torch
import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    CopyItemsd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd,
    RandFlipd,
    Resized,
    ResizeWithPadOrCropd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    Spacingd,
    EnsureTyped
)

from data.components import *

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple, modal: str, section: str,
        crop_window_size: list, pixdim: list, spacing: float = 2.0,
        phase: str = None,  # "unet", "resnet", "gsn"
        upscale_ratio: int = 2,  # Add upscale_ratio parameter for decoder-sized distance field
        dataset: str = None,
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
        crop_window_size: image and label will be cropped to match the size of network input.
        pixdim: the spatial distance of the downsampled images and labels.
        spacing: target spacing for isotropic resampling.
        phase: current processing phase, determining which keys are generated.
        dataset: dataset name for specific handling (e.g., 'acdc', 'cap', 'scotheart', 'mmwhs').
    """
    # Handle backward compatibility for target parameter
    if dataset is None:
        dataset = kwargs.get("target")
    if dataset is not None:
        dataset = dataset.lower()
    
    # Get stride configuration for DynUNet padding
    strides = kwargs.get("strides", (1, 2, 2, 2, 2))  # Default stride configuration
    
    # Removed UNet transform logging as per cleanup requirements
    
    # Unified loading, canonicalization, and resampling
    transforms = [
        UniversalCanonicalResampled(
            keys, 
            dataset=dataset, 
            modal=modal, 
            target_spacing=(spacing, spacing, spacing)
        )
    ]

    # Histogram matching removed - functionality archived

    # Add DynUNet-compatible padding based on modality
    if modal == "ct":
        # Apply 3D padding for all CT datasets
        transforms.append(
            DynUNetPaddingd([keys[0], keys[1]], strides=strides, spatial_dims=3, allow_missing_keys=True)
        )
    elif modal == "mr":
        # Apply 2D padding for all MR datasets
        transforms.append(
            DynUNetPaddingd([keys[0], keys[1]], strides=strides, spatial_dims=2, allow_missing_keys=True)
        )

    # Add dynamic intensity rescaling with dataset-specific background detection
    transforms.append(
        DynamicIntensityRangeScalesd(
            keys=[keys[0]], 
            dataset=dataset,
        )
    )
    
    # Load distance fields for full pipeline validation and testing phases
    load_distance_fields = (section in ["valid", "test"] and phase == "gsn")

    if load_distance_fields:
        # Calculate target size for distance field (decoder-sized for validation)
        df_target_size = int(crop_window_size[0] // pixdim[0] * upscale_ratio)
        
        df_transforms = [
            CopyItemsd(keys[1], names=f"{keys[1]}_ds"),
            SequentialTransformd(f"{keys[1]}_ds", sequence="s:xy f:x f:z"),
            Spacingd(f"{keys[1]}_ds", [spacing] * 3, mode="nearest"),
            CropForegroundd(f"{keys[1]}_ds", source_key=f"{keys[1]}_ds", margin=3),
            # create distance field from down-sampled label at decoder size
            Maskd(
                [f"{keys[1]}_ds", f"{keys[1][:2]}"], 
                allow_missing_keys=True
                ),
            FlexResized(
                f"{keys[1]}_ds", 
                (-1, crop_window_size[0], -1),
                force_nearest=True
            ),
            Resized(
                f"{keys[1]}_ds", 
                df_target_size,  # Use decoder-sized target for validation
                size_mode="longest", mode="nearest-exact"
                ),
            ResizeWithPadOrCropd(
                f"{keys[1]}_ds", 
                df_target_size,  # Use decoder-sized target for validation
                mode="constant", value=0
                ),
            DFConvertd(f"{keys[1]}_ds"),
        ]
        
        transforms.extend(df_transforms)

    if section == "train":
        if phase == "unet":
            # Only apply random crop for UNet phase
            transforms.extend([
                RandCropByPosNegLabeld(
                    keys, 
                    label_key=keys[1], 
                    spatial_size=crop_window_size, 
                    pos=2, 
                    neg=1,
                    num_samples=4,
                    allow_smaller=True,
                    allow_missing_keys=True
                ),
                RandFlipd(keys, prob=0.5, spatial_axis=[0]),  # H
                RandFlipd(keys, prob=0.5, spatial_axis=[1]),  # W
            ])

        transforms.extend([
            # spatial augmentation
            RandZoomd(
                keys,
                min_zoom=0.8 if modal == "ct" else [0.6, 0.6, 1.0], 
                max_zoom=1.2 if modal == "ct" else [1.4, 1.4, 1.0],
                mode=("trilinear", "nearest-exact"),
                padding_mode="constant",
                align_corners=(True, None), prob=0.15,
            ),
            RandGaussianNoised(keys[0], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys[0], 
                sigma_x=(0.5, 1.15), 
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15), 
                prob=0.15,
            ),
            RandAdjustContrastd(keys[0], gamma=(0.65, 1.5), prob=0.5),
        ])
        
        float_keys_train = [keys[0]]
        int_keys_train = [keys[1]]
        if load_distance_fields:
            float_keys_train.append(f"{keys[0][:2]}_df")
        transforms.extend([
            EnsureTyped(float_keys_train, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(int_keys_train, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        ])
    else: # "valid" or "test" section
        float_keys_valid = [keys[0]]
        int_keys_valid = [keys[1]]
        if load_distance_fields:
            float_keys_valid.append(f"{keys[0][:2]}_df")
        transforms.extend([
            EnsureTyped(float_keys_valid, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(int_keys_valid, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        ])

    return Compose(transforms)

