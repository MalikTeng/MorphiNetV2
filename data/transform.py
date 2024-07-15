import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    CropForegroundd,
    CopyItemsd,
    Orientationd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandZoomd,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    EnsureTyped
)

from data.components import *

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple, modal: str, section: str, rotation: bool,
        crop_window_size: list, pixdim: list, spacing: float = 2.0
):
    """
    Conducting pre-transformation that comprises multichannel conversion,
    resampling in regard of space distance, reorientation, foreground cropping,
    normalization and data augmentation.
    
    :params
        keys: designated items for pre-transformation (image and label).
        modal: modality of data the pre-transformation applied to.
        section: identifier of either train, valid or test set.
        rotation: whether to apply rotation augmentation.
        crop_window_size: image and label will be cropped to match the size of network input.
        pixdim: the spatial distance of the downsampled images and labels.
        spacing: target spacing for isotropic resampling.
    """
    # data loading
    transforms = [
        LoadImaged(keys, ensure_channel_first=False, image_only=True, allow_missing_keys=True),
    ]

    # # mask out the CTAs segmentation labels
    # if modal.lower() == "ct":
    #     transforms.append(MaskCTd(keys))

    transforms.extend([
        # isotropic resampling
        Adjustd(keys, allow_missing_keys=True),
        Spacingd(keys, 
                [spacing] * 3 if modal == "ct" else [spacing, spacing, -1], 
                mode=("bilinear", "nearest"), 
                allow_missing_keys=True),
        Orientationd(keys, axcodes="RAS", allow_missing_keys=True),     # (D, W, H)
        CopyItemsd(keys[1], names=f"{keys[1]}_ds"),         # keys: {"image", "label", "label_ds"}

        # distance field transformation
        # resampling and cropping                           keys: {"image", "label"}
        Spacingd(f"{keys[1]}_ds", 
                [-1] * 3 if modal == "ct" else [spacing, -1, -1],
                mode="nearest", padding_mode="zeros"),
        CropForegroundd(f"{keys[1]}_ds", source_key=keys[1], margin=1),
        # create distance field from down-sampled label
        Resized(
            f"{keys[1]}_ds", int(crop_window_size[0] // pixdim[0]), 
            size_mode="longest", mode="nearest"
            ),
        SpatialPadd(
            f"{keys[1]}_ds", int(crop_window_size[0] // pixdim[0]),
            method="symmetric", mode="minimum"
            ),
        DFConvertd(f"{keys[1]}_ds"),                        # keys: {"image", "label", "df"}
    ])

    # ensure images are with normalised intensity
    transforms.append(ScaleIntensityd(keys[0]))

    # random data augmentation
    if section == "train":
        if rotation:
            transforms.extend([
                # intensity argmentation (image only)
                RandGaussianNoised(keys[0], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys[0],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys[0], factors=0.3, prob=0.15),
                # spatial augmentation
                RandZoomd(
                    keys,
                    min_zoom=0.9 if modal == "ct" else [1.0, 0.9, 0.9], 
                    max_zoom=1.2 if modal == "ct" else [1.0, 1.2, 1.2],
                    mode=("trilinear", "nearest-exact"),
                    align_corners=(True, None),
                    prob=0.15,
                ),
                RandRotate90d(keys, prob=0.5, spatial_axes=(1, 2)),
                RandFlipd(keys, prob=0.5, spatial_axis=[1]),
                RandFlipd(keys, prob=0.5, spatial_axis=[2]),
                # ensure the data type
                EnsureTyped([*keys, f"{keys[0][:2]}_df"], data_type="tensor", dtype=torch.float32),
            ])
        else:
            transforms.extend([
                # intensity argmentation (image only)
                RandGaussianNoised(keys[0], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys[0],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys[0], factors=0.3, prob=0.15),
                # spatial augmentation
                RandZoomd(
                    keys,
                    min_zoom=0.9 if modal == "ct" else [1.0, 0.9, 0.9], 
                    max_zoom=1.2 if modal == "ct" else [1.0, 1.2, 1.2],
                    mode=("trilinear", "nearest-exact"),
                    align_corners=(True, None),
                    prob=0.15,
                ),
                # ensure the data type
                EnsureTyped([*keys, f"{keys[0][:2]}_df"], data_type="tensor", dtype=torch.float32),
            ])
    else:
        transforms.append(
            EnsureTyped([*keys, f"{keys[0][:2]}_df"], 
                        data_type="tensor", dtype=torch.float32, allow_missing_keys=True)
            )

    return Compose(transforms)

