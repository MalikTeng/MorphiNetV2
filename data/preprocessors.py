import os
import torch
import numpy as np
from typing import Union, Optional, List, Dict, Sequence, Hashable
import gc
from monai.transforms import (
    Compose, 
    AsDiscrete,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    CropForegroundd,
    Resized,
    Spacingd,
    ResizeWithPadOrCropd,
    EnsureTyped, 
)
from monai.config.type_definitions import KeysCollection
from monai.data import MetaTensor
from data.components import Maskd, FlexResized, SequentialTransformd
from einops import rearrange


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataPreprocessor:
    """Handles data preprocessing and post-processing for MorphiNet."""
    
    def __init__(self, super_params, dataset=None):
        """
        Initialize the data preprocessor.
        
        Args:
            super_params: Configuration parameters containing crop_window_size, pixdim, etc.
            dataset: Dataset identifier for dataset-specific handling
        """
        self.super_params = super_params
        self.dataset = dataset
    
    def _create_post_transform(self, keys=["pred", "label"], modal="ct", to_gpu=True, decoder_size=False, nearest_interpolate=False):
        """
        Create a unified post-transform pipeline that can handle both CPU/GPU and regular/decoder-sized outputs.
        
        Args:
            keys: Keys to transform (default: ["pred", "label"])
            modal: Modal type ("ct" or "mr")
            to_gpu: Whether to move final output to GPU
            decoder_size: Whether to use decoder-sized transform (upscaled) or regular transform
            nearest_interpolate: Whether use nearest interpolate to segmentation prediction created by UNET
        
        Returns:
            Composed transform pipeline
        """
        # Calculate target size based on decoder_size flag
        if isinstance(decoder_size, bool):
            if decoder_size:
                target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0] * self.super_params.upscale_ratio)
            else:
                target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0])
        elif isinstance(decoder_size, int):
            target_size = decoder_size
        else:
            raise ValueError(f"Invalid decoder_size value: {decoder_size}")
        
        # Choose target device
        target_device = DEVICE if to_gpu else "cpu"
        
        # Define spacing vector for this modal - ISOTROPIC for all modalities
        spacing_vector = [2, 2, 2]
                
        # Dynamic mode configuration based on key types
        mode_list = []
        for key in keys:
            if ('pred' in key.lower() or 'image' in key.lower()) and not nearest_interpolate:
                mode_list.append('bilinear')
            elif 'label' in key.lower() or nearest_interpolate:
                mode_list.append('nearest')
            else:
                mode_list.append('bilinear')

        # Convert to appropriate format for MONAI
        if len(mode_list) == 1:
            spacing_mode = mode_list[0]
        else:
            spacing_mode = tuple(mode_list)

        # Build transform list - Apply ReliableSpacingd LAST to ensure pixdim preservation
        transforms = [
            SequentialTransformd(keys, sequence="s:xy f:x f:z", allow_missing_keys=True),
            Spacingd(keys, spacing_vector, mode=spacing_mode, allow_missing_keys=True),
            CropForegroundd(
                keys, 
                source_key=keys[1] if len(keys) > 1 else keys[0], 
                margin=3, 
                allow_missing_keys=True,
            ),
            Maskd(keys, allow_missing_keys=True),
            FlexResized(
                keys, 
                (-1, target_size, -1), 
                force_nearest=True if nearest_interpolate else False,
                allow_missing_keys=True
            ),
            Resized(
                keys, 
                target_size, 
                size_mode="longest", 
                mode=spacing_mode, 
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys,
                target_size, 
                mode="constant", value=0,
                allow_missing_keys=True
            ),
            EnsureTyped(keys, device=target_device, allow_missing_keys=True),
        ]
                
        return Compose(transforms)
    
    def _create_label_post_transform(self, keys=["label"], modal="ct", to_gpu=True, decoder_size=False):
        """
        Create a label-specific post-transform pipeline that uses nearest interpolation throughout
        to preserve discrete label values and prevent corruption during resizing operations.
        
        Args:
            keys: Keys to transform (should be label keys only)
            modal: Modal type ("ct" or "mr")
            to_gpu: Whether to move final output to GPU
            decoder_size: Whether to use decoder-sized transform (upscaled) or regular transform
        
        Returns:
            Composed transform pipeline with nearest interpolation for labels
        """
        raise NotImplementedError("Use _create_post_transform instead")
        # Calculate target size based on decoder_size flag
        if decoder_size:
            target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0] * self.super_params.upscale_ratio)
        else:
            target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0])
        
        # Choose target device
        target_device = DEVICE if to_gpu else "cpu"
        
        # Define spacing vector for this modal - ISOTROPIC for all modalities
        spacing_vector = [2.0, 2.0, 2.0]

        # Dynamic mode configuration for label-specific transform
        mode_list = []
        for key in keys:
            if 'label' in key.lower():
                mode_list.append('nearest')
            else:
                mode_list.append('nearest')
        
        # Convert to appropriate format for MONAI
        if len(mode_list) == 1:
            spacing_mode = mode_list[0]
        else:
            spacing_mode = tuple(mode_list)

        # Use identical sequential transformation logic as _create_post_transform
        # Both predictions and labels must use same SequentialTransformd for consistency
        label_sequence = "s:xy f:x f:z"  # Same as _create_post_transform
                
        # Build transform list
        transforms = [
            SequentialTransformd(keys, sequence=label_sequence, allow_missing_keys=True),
            Spacingd(keys, spacing_vector, mode=spacing_mode, allow_missing_keys=True),
            CropForegroundd(keys, source_key=keys[0], allow_missing_keys=True),
            # Skip Maskd transform for labels - it's not needed and causes key issues
            FlexResized(
                keys, 
                (-1, self.super_params.crop_window_size[0], -1), 
                allow_missing_keys=True,
                force_nearest=True  # Force nearest interpolation for labels
            ),
            Resized(
                keys, 
                target_size, 
                size_mode="longest", mode=spacing_mode,  # Use dynamic mode (nearest for labels)
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys,
                target_size, 
                mode="constant", value=0,
                allow_missing_keys=True
            ),
            EnsureTyped(keys, device=target_device, allow_missing_keys=True),
        ]        
        
        return Compose(transforms)
    
    def _apply_mr_unflatten(self, tensor, modal):
        """
        Apply unflatten operation for MR data to handle [D*B,C,H,W] → [B,C,H,W,D] transformation.
        
        Args:
            tensor: Input tensor to unflatten
            modal: Modality type ("ct" or "mr")
        
        Returns:
            Unflattened tensor for MR, unchanged tensor for CT
        """
        if modal != 'mr':
            return tensor
            
        # Apply unflatten operation: [D*B,C,H,W] → [B,C,H,W,D]
        tensor_unflattened = rearrange(tensor, '(d b) c h w -> b c h w d', b=1)
            
        return tensor_unflattened
    
    def _generate_downsampled_gt(self, seg_true, modal, decoder_size=False):
        """
        Generates a downsampled ground truth tensor. This method now handles a single
        tensor directly, unifying MR and CT shapes to 5D before processing.
        
        Args:
            seg_true: Ground truth tensor (4D for MR, 5D for CT)
            modal: Modality type ("ct" or "mr")
            decoder_size: Whether to use decoder-sized transform or regular transform
        
        Returns:
            Downsampled ground truth tensor.
        """
        # Unify tensor shape to 5D
        if modal == 'mr':
            seg_true = self._apply_mr_unflatten(seg_true, modal)
        
        # Create the appropriate transform
        transform = self._create_post_transform(
            keys=["label"], 
            modal=modal, 
            to_gpu=False,
            decoder_size=decoder_size,
        )
        
        # Move to CPU for memory-efficient processing
        seg_true = seg_true.cpu()

        # Process each item in the batch
        batch_size = seg_true.shape[0]
        batch_processed = []
        for b in range(batch_size):
            true_4d = seg_true[b]
            result = transform({"label": true_4d, "modal": modal})
            batch_processed.append(result["label"])
        
        # Stack results and move to the correct device
        processed_true = torch.stack(batch_processed, dim=0)
        if torch.cuda.is_available():
            processed_true = processed_true.to(DEVICE)
        
        return processed_true
    
    def _memory_efficient_post_transform(self, seg_pred, seg_true, modal, to_gpu=True, decoder_size=False, **kwargs):
        """
        Memory-efficient post-transform processing that handles tensors directly.
        Unifies tensor shapes to 5D before processing each item in the batch.
        
        Args:
            seg_pred: Prediction tensor (4D for MR, 5D for CT)
            seg_true: Ground truth tensor (4D for MR, 5D for CT)
            modal: Modal type ("ct" or "mr")
            to_gpu: Whether to move final output to GPU
            decoder_size: Whether to use decoder-sized transform or regular transform
        """
        # Unify tensor shapes to 5D: [B,C,H,W,D] by unflattening MR data
        if modal == 'mr':
            seg_pred = self._apply_mr_unflatten(seg_pred, modal)
            seg_true = self._apply_mr_unflatten(seg_true, modal)

        # Create appropriate transform for post-processing
        transform = self._create_post_transform(
            keys=["pred", "label"], 
            modal=modal, 
            to_gpu=False,  # Always process on CPU first to save memory
            decoder_size=decoder_size,
            nearest_interpolate=kwargs.get('nearest_interpolate', False)  # Allow skipping masking if specified
        )

        # Move to CPU for memory efficiency before processing
        if hasattr(seg_pred, 'is_cuda') and seg_pred.is_cuda:
            seg_pred = seg_pred.cpu()
        if hasattr(seg_true, 'is_cuda') and seg_true.is_cuda:
            seg_true = seg_true.cpu()

        # Process each item in the batch individually
        assert seg_pred.dim() == 5, f"Tensor must be 5D after potential unflattening, but got {seg_pred.dim()}D"
        
        batch_size = seg_pred.shape[0]
        batch_processed = []

        for b in range(batch_size):
            # Extract 4D tensors for each batch item (C, H, W, D)
            pred_4d = seg_pred[b]
            true_4d = seg_true[b]
            
            # Apply post-transform to 4D tensors
            result = transform({"pred": pred_4d, "label": true_4d, "modal": modal})
            processed_pred_4d = result["pred"]
            
            batch_processed.append(processed_pred_4d)
            
            # Clear intermediate results
            del pred_4d, true_4d, result

        # Stack batch results back to 5D
        processed_pred = torch.stack(batch_processed, dim=0)

        # Move to GPU only when needed
        if to_gpu and hasattr(processed_pred, 'to'):
            processed_pred = processed_pred.to(DEVICE)
        
        # Clear intermediate results to free memory
        del seg_pred, seg_true
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        return processed_pred
    
    def _filter_unlabeled_slices(self, img, seg):
        """
        Filter out slices without labels between the first and last labeled slice.
        Maintains corresponding slices in both halves of the data.
        
        Args:
            img: Input image tensor
            seg: Input segmentation tensor
            
        Returns:
            Filtered image and segmentation tensors
        """
        half_size = seg.shape[0]//2
        mask = seg[:half_size] > 0
        has_label = mask.any(dim=1).any(dim=1).any(dim=1)
        
        if has_label.sum() > 0:  # Only process if at least one slice has a label
            start_idx = torch.where(has_label)[0].min()
            end_idx = torch.where(has_label)[0].max()
            
            # Create a full mask: keep slices outside [start_idx, end_idx] and labeled slices within range
            full_mask = torch.ones_like(has_label, device=img.device, dtype=torch.bool)
            full_mask[start_idx:end_idx+1] = has_label[start_idx:end_idx+1]  # Only filter unlabeled slices within range
            
            # Get valid indices from first half
            first_half_indices = torch.where(full_mask)[0]
            
            # Create corresponding indices for the second half
            second_half_indices = first_half_indices + half_size
            
            # Combine indices from both halves
            valid_indices = torch.cat([first_half_indices, second_half_indices])
            
            # Apply the mask
            img = img[valid_indices]
            seg = seg[valid_indices]
            
        return img, seg
    
    def _convert_to_onehot(self, seg_tensor, num_classes):
        """
        Convert segmentation tensor to one-hot encoding.
        
        Deprecated: using model/inference._convert_to_onehot instead
        """
        raise NotImplementedError("Use model/inference._convert_to_onehot instead")
    
    def _prepare_slice_for_wandb(self, slice_tensor, is_segmentation, num_classes=None):
        """
        Prepares a 2D tensor slice for logging to Weights & Biases as an image.
        
        Args:
            slice_tensor: 2D tensor slice
            is_segmentation: Whether this is a segmentation mask
            num_classes: Number of classes (required for segmentation)
        
        Returns:
            Prepared slice as numpy array
        """
        slice_np = slice_tensor.cpu().numpy().astype(np.float32)
        
        if is_segmentation:
            if num_classes is None:
                raise ValueError("num_classes must be provided for segmentation masks.")
            scale_factor = 255.0 / (num_classes - 1) if num_classes > 1 else 255.0
            slice_viz = (slice_np * scale_factor).astype(np.uint8)
        else:  # Input image
            min_val = slice_np.min()
            max_val = slice_np.max()
            if max_val - min_val > 1e-6:
                slice_norm = (slice_np - min_val) / (max_val - min_val)
            else:
                slice_norm = slice_np
            slice_viz = (slice_norm * 255).astype(np.uint8)
        
        return slice_viz
    
