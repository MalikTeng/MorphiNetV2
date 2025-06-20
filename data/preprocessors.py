import torch
import numpy as np
import gc
from monai.transforms import (
    Compose, 
    AsDiscrete,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    CropForegroundd,
    Resized,
    Spacingd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    EnsureTyped, 
)
from data.components import Maskd, FlexResized


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataPreprocessor:
    """Handles data preprocessing and post-processing for MorphiNet."""
    
    def __init__(self, super_params):
        """
        Initialize the data preprocessor.
        
        Args:
            super_params: Configuration parameters containing crop_window_size, pixdim, etc.
        """
        self.super_params = super_params
    
    def _create_post_transform(self, keys=["pred", "label"], modal="ct", to_gpu=True, decoder_size=False):
        """
        Create a unified post-transform pipeline that can handle both CPU/GPU and regular/decoder-sized outputs.
        
        Args:
            keys: Keys to transform (default: ["pred", "label"])
            modal: Modal type ("ct" or "mr")
            to_gpu: Whether to move final output to GPU
            decoder_size: Whether to use decoder-sized transform (upscaled) or regular transform
        
        Returns:
            Composed transform pipeline
        """
        # Calculate target size based on decoder_size flag
        if decoder_size:
            target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0] * self.super_params.upscale_ratio)
        else:
            target_size = int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0])
        
        # Choose target device
        target_device = DEVICE if to_gpu else "cpu"
        
        return Compose([
            Spacingd(keys, [2.0, 2.0, 2.0], mode=("bilinear", "nearest"), allow_missing_keys=True),
            CropForegroundd(keys, source_key=keys[1] if len(keys) > 1 else keys[0], allow_missing_keys=True),
            Maskd(keys + [modal], allow_missing_keys=True),
            FlexResized(
                keys, 
                (-1, self.super_params.crop_window_size[0], -1), 
                allow_missing_keys=True
            ),
            Resized(
                keys, 
                target_size, 
                size_mode="longest", mode=("bilinear", "nearest-exact"), 
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys,
                target_size, 
                mode="constant", value=0,
                allow_missing_keys=True
            ),
            EnsureTyped(keys, device=target_device, allow_missing_keys=True),
        ])
    
    def _generate_downsampled_gt(self, seg_true, modal, decoder_size=False):
        """
        Generate downsampled ground truth on-the-fly from full resolution ground truth.
        
        Args:
            seg_true: Full resolution ground truth tensor (4D or 5D)
            modal: Modal type ("ct" or "mr")
            decoder_size: Whether to generate decoder-sized output
        
        Returns:
            Downsampled ground truth tensor (preserving original dimensionality)
        """
        result = self._memory_efficient_post_transform(
            seg_pred_list=[seg_true], 
            seg_true_list=[seg_true], 
            modal=modal, 
            to_gpu=True, 
            decoder_size=decoder_size
        )
        return result
    
    def _memory_efficient_post_transform(self, seg_pred_list, seg_true_list, modal, to_gpu=True, decoder_size=False):
        """
        Memory-efficient post-transform processing that handles tensors individually.
        
        Args:
            seg_pred_list: List of prediction tensors or single tensor (4D or 5D)
            seg_true_list: List of ground truth tensors or single tensor (4D or 5D)
            modal: Modal type ("ct" or "mr")
            to_gpu: Whether to move final output to GPU
            decoder_size: Whether to use decoder-sized transform (upscaled) or regular transform
        """
        # Handle single tensor inputs by converting to list
        if not isinstance(seg_pred_list, (list, tuple)):
            seg_pred_list = [seg_pred_list]
        if not isinstance(seg_true_list, (list, tuple)):
            seg_true_list = [seg_true_list]
        
        processed_preds = []
        
        # Create appropriate transform based on flags
        transform = self._create_post_transform(
            keys=["pred", "label"], 
            modal=modal, 
            to_gpu=False,  # Always process on CPU first to save memory
            decoder_size=decoder_size
        )
        
        # Process each tensor individually to avoid large batch processing
        for i, (pred, true) in enumerate(zip(seg_pred_list, seg_true_list)):
            # Move to CPU if not already there
            if hasattr(pred, 'is_cuda') and pred.is_cuda:
                pred = pred.cpu()
            if hasattr(true, 'is_cuda') and true.is_cuda:
                true = true.cpu()
            
            # Handle batch dimension: process each batch item if 5D
            if pred.dim() == 5:  # 5D: (B, C, D, H, W)
                batch_size = pred.shape[0]
                batch_processed = []
                
                for b in range(batch_size):
                    # Extract 4D tensors for each batch item
                    pred_4d = pred[b]  # (C, D, H, W)
                    true_4d = true[b]  # (C, D, H, W)
                    
                    # Apply post-transform to 4D tensors
                    result = transform({"pred": pred_4d, "label": true_4d, "modal": modal})
                    processed_pred_4d = result["pred"]
                    
                    batch_processed.append(processed_pred_4d)
                    
                    # Clear intermediate results
                    del pred_4d, true_4d, result
                
                # Stack batch results back to 5D
                processed_pred = torch.stack(batch_processed, dim=0)
                
            elif pred.dim() == 4:  # 4D: (C, D, H, W)
                # Apply post-transform directly to 4D tensors
                result = transform({"pred": pred, "label": true, "modal": modal})
                processed_pred = result["pred"]
                del result
                
            else:
                raise ValueError(f"Unsupported tensor dimensions: {pred.dim()}D. Expected 4D or 5D tensors.")
            
            # Move to GPU only when needed and one at a time
            if to_gpu and hasattr(processed_pred, 'to'):
                processed_pred = processed_pred.to(DEVICE)
            
            processed_preds.append(processed_pred)
            
            # Clear intermediate results to free memory
            del pred, true
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        # Return results appropriately
        if len(processed_preds) == 1:
            return processed_preds[0]
        elif processed_preds and hasattr(processed_preds[0], 'dim'):
            return torch.stack(processed_preds, dim=0)
        else:
            return processed_preds
    
    def _filter_unlabeled_slices(self, seg_pred_list, seg_true_list, modal, threshold=0.1):
        """
        Filter out slices with minimal foreground content to focus training on informative slices.
        
        Args:
            seg_pred_list: List of prediction tensors
            seg_true_list: List of ground truth tensors
            modal: Modal type ("ct" or "mr")
            threshold: Minimum foreground ratio to keep a slice
        
        Returns:
            Filtered lists of predictions and ground truth tensors
        """
        filtered_pred_list = []
        filtered_true_list = []
        
        for seg_pred, seg_true in zip(seg_pred_list, seg_true_list):
            # Calculate foreground ratio
            if seg_true.dim() == 4:  # (C, D, H, W)
                foreground_ratio = (seg_true > 0).float().mean().item()
            else:
                foreground_ratio = (seg_true > 0).float().mean().item()
            
            # Keep slice if it has sufficient foreground content
            if foreground_ratio >= threshold:
                filtered_pred_list.append(seg_pred)
                filtered_true_list.append(seg_true)
        
        return filtered_pred_list, filtered_true_list
    
    def _convert_to_onehot(self, seg_tensor, num_classes):
        """
        Convert segmentation tensor to one-hot encoding.
        
        Args:
            seg_tensor: Segmentation tensor
            num_classes: Number of classes
        
        Returns:
            One-hot encoded tensor
        """
        # Use PyTorch's built-in one-hot function
        if seg_tensor.dim() == 4:  # (C, D, H, W)
            seg_tensor = seg_tensor.squeeze(0)  # Remove channel dimension
        
        # Convert to long tensor for one-hot encoding
        seg_long = seg_tensor.long()
        
        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(seg_long, num_classes=num_classes)
        
        # Rearrange dimensions: (D, H, W, C) -> (C, D, H, W)
        one_hot = one_hot.permute(-1, 0, 1, 2)
        
        return one_hot.float()
    
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