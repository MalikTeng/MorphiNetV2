import torch
import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    CopyItemsd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd,
    Resized,
    ResizeWithPadOrCropd,
    Spacingd,
    EnsureTyped
)

from data.components import *

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple, modal: str, section: str,
        crop_window_size: list, pixdim: list, spacing: float = 2.0,
        phase: str = "validation",  # "unet", "resnet", "gsn", "ndf", "validation"
        upscale_ratio: int = 2,  # Add upscale_ratio parameter for decoder-sized distance field
        dataset: str = None,
        custom_sequence: str = None,  # Optional custom transformation sequence (e.g., "s:xy f:x f:z")
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
        custom_sequence: optional transformation sequence string (e.g., "s:xy f:x f:z") applied before distance field generation.
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

    # Add histogram matching transform for automatic intensity normalization
    transforms.append(
        HistogramMatchd([keys[0]], modal=modal, dataset=dataset, cdf_dir="./cdf_cache", allow_missing_keys=True)
    )

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

    # Only load distance fields for GSN/full network validation (not needed for UNet or ResNet phases)
    load_distance_fields = (section == "valid" and phase in ["gsn", "validation"])

    if load_distance_fields:
        # Calculate target size for distance field (decoder-sized for validation)
        df_target_size = int(crop_window_size[0] // pixdim[0] * upscale_ratio)
        
        # Add custom sequential transformation if specified
        df_transforms = []
        if custom_sequence:
            df_transforms.append(SequentialTransformd(keys[1], sequence=custom_sequence))
        
        df_transforms.extend([
            CopyItemsd(keys[1], names=f"{keys[1]}_ds"),
            Spacingd(f"{keys[1]}_ds", [spacing] * 3,
                    mode="nearest", padding_mode="zeros"),
            CropForegroundd(f"{keys[1]}_ds", source_key=f"{keys[1]}_ds"),
            # create distance field from down-sampled label at decoder size
            Maskd([f"{keys[1]}_ds", f"{keys[1][:2]}"], allow_missing_keys=True),
            FlexResized(
                f"{keys[1]}_ds", 
                (-1, crop_window_size[0], -1)
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
        ])
        
        transforms.extend(df_transforms)

    # keys_to_ensure = list(keys)
    # if load_full_data:
    #     keys_to_ensure.extend([f"{keys[0][:2]}_df", f"{keys[1]}_ds"])

    if section == "train":
        transforms.extend([
            # spatial augmentation
            RandZoomd(
                keys,
                min_zoom=0.3 if modal == "ct" else [1.0, 0.3, 0.3], 
                max_zoom=1.2 if modal == "ct" else [1.0, 1.2, 1.2],
                mode=("trilinear", "nearest-exact"),
                align_corners=(True, None), prob=0.5,
            ),
            RandGaussianNoised(keys[0], std=0.01, prob=0.5),
            RandGaussianSmoothd(
                keys[0], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15), prob=0.5,
            ),
            # RandAdjustContrastd(keys[0], gamma=(0.65, 1.5), prob=0.5),
            # RandScaleIntensityd(keys[0], factors=0.3, prob=0.5),
            # Note: ThresholdIntensityd, HistogramNormalized, and ScaleIntensityd are now handled by HistogramMatchd
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
        transforms.extend([
        # Note: ThresholdIntensityd, HistogramNormalized, and ScaleIntensityd are now handled by HistogramMatchd
        ])
        float_keys_valid = [keys[0]]
        int_keys_valid = [keys[1]]
        if load_distance_fields:
            float_keys_valid.append(f"{keys[0][:2]}_df")
        transforms.extend([
            EnsureTyped(float_keys_valid, data_type="tensor", dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(int_keys_valid, data_type="tensor", dtype=torch.int8, allow_missing_keys=True),
        ])

    return Compose(transforms)

