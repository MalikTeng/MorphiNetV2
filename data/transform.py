import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    CropForegroundd,
    CopyItemsd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandFlipd,
    Spacingd,
    Resized, 
    SpatialPadd,
    EnsureTyped
)

from data.components import *

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple, modal: str, section: str,
        crop_window_size: list, pixdim: list,
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
        one_or_many_frames: identifier of either task for stationary or animated meshing.
    """
    # data loading
    transforms = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
    ]

    # mask out the CTAs segmentation labels
    if modal.lower() == "ct":
        transforms.append(MaskCTAd(keys))

    # downsampling and cropping                         keys: {"image", "label"}
    transforms.extend([
        CropForegroundd(keys, source_key=keys[1], margin=1),
        Resized(keys, crop_window_size[0], size_mode="longest", 
                mode=("bilinear", "nearest-exact")),
        SpatialPadd(keys, crop_window_size[0], method="symmetric", mode="minimum"),
        CopyItemsd(keys[1], names=f"{keys[1]}_ds"),     # keys: {"image", "label", "label_ds"}

        # create distance field from down-sampled label
        Resized(f"{keys[1]}_ds", [w // p for w, p in zip(crop_window_size, pixdim)], 
                size_mode="all", mode="nearest-exact"),
        DFConvertd(f"{keys[1]}_ds"),                    # keys: {"image", "label", "df"}

        # ensure images are with normalised intensity
        # NormalizeIntensityd(keys[0], nonzero=False, channel_wise=False),
        ScaleIntensityd(keys[0], minv=0, maxv=1),
    ])

    # spatial transforms
    if section == "train":
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
            
            # ensure the data type
            EnsureTyped([*keys, f"{keys[0][:2]}_df"], data_type="tensor", dtype=torch.float32),
        ])
    else:
        transforms.append(
            EnsureTyped([*keys, f"{keys[0][:2]}_df"], data_type="tensor", dtype=torch.float32)
            )

    return Compose(transforms)

# def pre_transform(
#         keys: tuple, modal: str, section: str,
#         crop_window_size: list, pixdim: list,
# ):
#     """
#     Conducting pre-transformation that comprises multichannel conversion,
#     resampling in regard of space distance, reorientation, foreground cropping,
#     normalization and data augmentation.
    
#     :params
#         keys: designated items for pre-transformation (image and label).
#         modal: modality of data the pre-transformation applied to.
#         section: identifier of either train, valid or test set.
#         crop_window_size: image and label will be cropped to match the size of network input.
#         pixdim: the spatial distance of the downsampled images and labels.
#         one_or_many_frames: identifier of either task for stationary or animated meshing.
#     """
#     # data loading
#     transforms = [
#         LoadImaged(keys, ensure_channel_first=True, image_only=True),
#     ]

#     # mask out the CTAs segmentation labels
#     if modal.lower() == "ct":
#         transforms.append(MaskCTAd(keys))

#     # downsampling and cropping -- keys: {"image", "label"}
#     transforms.extend([
#         CopyItemsd(keys[1], names=f"{keys[1]}_origin"),
#         CropForegroundd([*keys, f"{keys[1]}_origin"], 
#                         source_key=keys[1], margin=1),
        
#         # downsampling the image and label to the desired spatial resolution
#         Spacingd(keys, pixdim, mode=("bilinear", "nearest")),
#         Resized(keys,  crop_window_size[0] // pixdim[0], 
#                 size_mode="longest", mode=("bilinear", "nearest-exact")),
#         SpatialPadd(keys, [i // j for i, j in zip(crop_window_size, pixdim)], 
#                     method="symmetric", mode="minimum"),
#         DFConvertd(keys[1]),

#         # process images and labels with original resolution
#         Resized(f"{keys[1]}_origin", crop_window_size[0], size_mode="longest",
#                 mode="nearest-exact"),
#         SpatialPadd(f"{keys[1]}_origin", crop_window_size[0], method="symmetric", 
#                     mode="minimum"),

#         # ensure all images with regularised intensity
#         NormalizeIntensityd(keys[0], nonzero=False, channel_wise=False),
#     ])

#     # spatial transforms
#     if section == "train":
#         transforms.extend([
#             # intensity argmentation (image only)
#             RandGaussianNoised(keys[0], std=0.01, prob=0.15),
#             RandGaussianSmoothd(
#                 keys[0],
#                 sigma_x=(0.5, 1.15),
#                 sigma_y=(0.5, 1.15),
#                 sigma_z=(0.5, 1.15),
#                 prob=0.15,
#             ),
#             RandScaleIntensityd(keys[0], factors=0.3, prob=0.15),
            
#             # ensure the data type
#             EnsureTyped([*keys, f"{keys[1]}_origin", f"{keys[0][:2]}_df"], 
#                         data_type="tensor", dtype=torch.float32),
#         ])
#     else:
#         transforms.append(EnsureTyped([*keys, f"{keys[1]}_origin", f"{keys[0][:2]}_df"],
#                                       data_type="tensor", dtype=torch.float32))

#     return Compose(transforms)
