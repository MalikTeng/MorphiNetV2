from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized
from monai.transforms.utils import distance_transform_edt
import os
from pathlib import Path
from typing import Optional, Tuple

# import nibabel as nib  # Unused import


__all__ = ["Maskd", "DFConvertd", "Adjustd", "FlexResized", "Probd", "DynUNetPaddingd", "SequentialTransformd", "ACDCSequentialTransform", "DatasetCanonicalizer", "UniversalCanonicalResampled", "HistogramMatchd"]


class Maskd(MapTransform):
    """
    this transform mask the CT pred near the basal and apex plane, i.e., the first and last slices.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        for key in ["pred", "label"]:
            try:
                array = data[key]
            except KeyError:
                continue
            else:
                # Ensure we work with CPU arrays to avoid GPU memory issues
                if hasattr(array, 'is_cuda') and array.is_cuda:
                    array = array.cpu()
                
                array = array.get_array()

                if data["modal"] == "ct" and "pred" in key:
                    # mask the CTA images near the basal and apex plane
                    mask = np.zeros_like(array).astype(bool)
                    mask[:, 6:-6] = True
                    array[~mask] = array.min()

                    data[key] = MetaTensor(array, affine=data[key].affine, 
                                           applied_operations=data[key].applied_operations)
                
                elif data["modal"] == "mr":
                    # pad slices on the top and bottom of the image
                    array = np.pad(array, ((0, 0), (6, 6), (0, 0), (0, 0)), mode="constant", constant_values=array.min())
                    # update the affine
                    affine = data[key].affine.clone()
                    affine[:3, -1] -= 6 * data[key].pixdim[0]
                    data[key] = MetaTensor(array, affine=affine,
                                           applied_operations=data[key].applied_operations)


        return data


class Adjustd(MapTransform):
    """
    process the input data to be compatible with the rest transforms.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, target: str = None) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target = target

    def __call__(self, data):
        for key in self.keys:
            try:
                pixel_array = data[key].get_array().copy()

                if 'mr' in key and len(pixel_array.shape) == 4 and self.target != 'acdc':
                    affine = data[key].affine.clone()
                    # update the affine matrix
                    m = torch.eye(4)
                    m[:3, 0] = affine[1, :3]
                    m[:3, 1] = affine[2, :3]
                    m[:3, 2] = affine[3, :3]
                    m[:3, -1] = affine[:3, -1]
                    data[key] = MetaTensor(pixel_array, affine=m)

                if "label" in key:
                    # Combine RV-MYO (label 4) with LV-MYO (label 2) into a single MYO label
                    pixel_array[pixel_array == 4] = 2
                    # Keep all other labels as they are: background(0), LV(1), combined MYO(2), RV(3)
                    # Update the data with the modified pixel array
                    data[key] = MetaTensor(pixel_array, affine=data[key].affine, 
                                           applied_operations=data[key].applied_operations)

            except KeyError:
                pass  # Key not found in data dictionary

        return data


class FlexResized(MapTransform):
    """
    Flexible resize transform that resizes image/label to a fixed scale at the second dimension.
    Supports -1 as a wildcard to preserve original dimensions.
    
    Args:
        keys: Keys to apply the transform to (e.g., ["pred", "label"])
        size: Target size tuple, -1 preserves original dimension (e.g., (-1, 128, -1))
        allow_missing_keys: Whether to allow missing keys
        force_nearest: Whether to force nearest interpolation for all keys (useful for labels)
    """
    def __init__(self, keys: KeysCollection, size: tuple, allow_missing_keys: bool = False, force_nearest: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_size = np.array([int(s) for s in size])
        self.allow_missing_keys = allow_missing_keys
        self.force_nearest = force_nearest

    def __call__(self, data):
        # Determine available keys and their roles
        available_keys = []
        pred_key = None
        label_key = None
        
        for key in self.keys:
            if key in data:
                available_keys.append(key)
                if "pred" in key:
                    pred_key = key
                elif "label" in key:
                    label_key = key
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in data")
        
        if not available_keys:
            return data
        
        # Use label key as reference for shape, fallback to first available key
        reference_key = label_key if label_key else available_keys[0]
        
        # Get current data shape (excluding channel dimension)
        if hasattr(data[reference_key], 'get_array'):
            current_shape = np.array(data[reference_key].get_array().shape[1:])  # Skip channel dim
        else:
            current_shape = np.array(data[reference_key].shape[1:])  # Skip channel dim
        
        # Handle dimension mismatch: ensure target_size and current_shape have same length
        if len(self.target_size) != len(current_shape):
            # Pad target_size with -1 if it's shorter, or truncate if longer
            if len(self.target_size) < len(current_shape):
                padded_size = np.full(len(current_shape), -1, dtype=int)
                padded_size[:len(self.target_size)] = self.target_size
                target_size = padded_size
            else:
                target_size = self.target_size[:len(current_shape)]
        else:
            target_size = self.target_size
        
        # Replace -1 with current dimensions
        final_size = np.where(target_size == -1, current_shape, target_size)
        
        # Calculate rescale ratio based on the second dimension (index 1)
        if len(final_size) > 1 and final_size[1] != current_shape[1]:
            rescale_ratio = final_size[1] / current_shape[1]
            new_shape = [int(np.ceil(d * rescale_ratio)) for d in current_shape]
            # Ensure the target dimension matches exactly
            new_shape[1] = int(final_size[1])
        else:
            new_shape = [int(s) for s in final_size]
        
        # Apply resize transformation
        if pred_key and label_key:
            # Both prediction and label available
            if self.force_nearest:
                # Force nearest interpolation for all keys (label-safe)
                data = Resized([pred_key, label_key], new_shape, size_mode="all", 
                              mode="nearest")(data)
            else:
                # Use appropriate interpolation for each key type
                data = Resized([pred_key, label_key], new_shape, size_mode="all", 
                              mode=("bilinear", "nearest"))(data)
        elif available_keys:
            # Only one key available - determine appropriate mode
            if self.force_nearest:
                mode = "nearest"  # Force nearest when requested
            else:
                mode = "nearest" if "label" in available_keys[0] else "bilinear"
            data = Resized(available_keys, new_shape, size_mode="all", mode=mode)(data)
        
        return data


class Probd(MapTransform):
    """
    read the input data to see its shape and dimension.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            _ = data[key].get_array().copy()  # Debug: examine array
            # Debug information for key shape and pixdim

        return data


class DFConvertd(MapTransform):
    """
    this transform convert the ground truth segmentation to signed distance fields.
    """
    def __init__(self, key: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(key, allow_missing_keys)
        self.key = key
        self.modal = key[:2]

    def __call__(self, data):
        label = data[self.key]
        label = label.as_tensor().clone()

        # Four channels for GSN phase: (foreground, left ventricle, right ventricle, myocardium)
        # Labels are preprocessed to combine LV-MYO and RV-MYO into label 2
        foreground = label > 0
        lv = label == 1
        rv = label == 3  # RV label
        myo = label == 2  # Combined LV-MYO and RV-MYO (preprocessed)

        df = []
        for mask in [foreground, lv, rv, myo]:  # Compute DF for foreground, lv, rv, myo
            df_class = distance_transform_edt(mask) + distance_transform_edt(~mask)
            df.append(df_class[:, None])

        df = MetaTensor(torch.cat(df, dim=1), affine=data[self.key].affine)

        data[f"{self.modal}_df"] = df

        # Remove the downsampled label after generating distance field (as per user requirement)
        data.pop(self.key)

        return data


class DynUNetPaddingd(MapTransform):
    """
    Pad spatial dimensions to ensure compatibility with DynUNet skip connections.
    
    This transform pads the spatial dimensions (height, width, depth) to be divisible 
    by the stride factor, preventing odd shapes in DynUNet encoder/decoder layers.
    
    For 2D DynUNet (MR): pads H, W dimensions
    For 3D DynUNet (CT): pads H, W, D dimensions
    
    Args:
        keys: Keys to apply the padding to (typically image and label)
        strides: Stride configuration (e.g., (1, 2, 2, 2, 2))
        spatial_dims: Either 2 for 2D DynUNet or 3 for 3D DynUNet
        mode: Padding mode ('constant', 'reflect', 'replicate', 'circular')
        value: Padding value when mode='constant'
    """
    def __init__(
        self, 
        keys: KeysCollection, 
        strides: tuple = (1, 2, 2, 2, 2),
        spatial_dims: int = 3,
        mode: str = "constant",
        value: float = 0.0,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.strides = strides
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.value = value
        
        # Calculate stride factor for spatial dimensions
        # For DynUNet with default 5-level architecture: strides = (1, 2, 2, 2, 2)
        # The stride factor should be the product of the downsampling strides
        if self.spatial_dims == 2:
            # For 2D DynUNet, use all 4 downsampling levels: factor = 2*2*2*2 = 16
            effective_strides = self.strides[1:5]
        elif self.spatial_dims == 3:
            # For 3D DynUNet, use all 4 downsampling levels: factor = 2*2*2*2 = 16
            effective_strides = self.strides[1:5]
        else:
            raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}. Must be 2 or 3.")
        
        self.stride_factor = 1
        for s in effective_strides:
            self.stride_factor *= s
        
        # Silent initialization - no logging
        # Store configuration for debugging if needed
        self._config_info = {
            'spatial_dims': self.spatial_dims,
            'strides': self.strides,
            'effective_strides': effective_strides,
            'stride_factor': self.stride_factor
        }

    def __call__(self, data):
        data_dict = dict(data)
        
        # Iterate through the keys explicitly
        for key in self.keys:
            if key not in data_dict:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in data")
                    
            try:
                array = data_dict[key]
                
                # Ensure we work with CPU arrays to avoid GPU memory issues
                if hasattr(array, 'is_cuda') and array.is_cuda:
                    array = array.cpu()
                
                if hasattr(array, 'get_array'):
                    pixel_array = array.get_array()
                else:
                    pixel_array = array
                
                # Get original shape
                original_shape = pixel_array.shape
                
                if len(original_shape) == 4:  # (C, H, W, D) format
                    c, h, w, d = original_shape
                    
                    if self.spatial_dims == 2:
                        # 2D DynUNet: pad only H, W dimensions
                        pad_h = (self.stride_factor - h % self.stride_factor) % self.stride_factor
                        pad_w = (self.stride_factor - w % self.stride_factor) % self.stride_factor
                        
                        if pad_h > 0 or pad_w > 0:
                            # PyTorch pad format: (D_left, D_right, W_left, W_right, H_left, H_right)
                            padding = (0, 0, 0, pad_w, 0, pad_h)
                            
                            # Apply padding
                            if isinstance(pixel_array, torch.Tensor):
                                import torch.nn.functional as F
                                padded_array = F.pad(pixel_array, padding, mode=self.mode, value=self.value)
                            else:
                                # Convert to tensor, pad, then convert back
                                tensor_array = torch.from_numpy(pixel_array) if isinstance(pixel_array, np.ndarray) else pixel_array
                                import torch.nn.functional as F
                                padded_tensor = F.pad(tensor_array, padding, mode=self.mode, value=self.value)
                                padded_array = padded_tensor.numpy() if isinstance(pixel_array, np.ndarray) else padded_tensor
                            
                            # Silent padding - no logging
                            pass
                        else:
                            padded_array = pixel_array
                    
                    elif self.spatial_dims == 3:
                        # 3D DynUNet: pad H, W, D dimensions
                        pad_h = (self.stride_factor - h % self.stride_factor) % self.stride_factor
                        pad_w = (self.stride_factor - w % self.stride_factor) % self.stride_factor
                        pad_d = (self.stride_factor - d % self.stride_factor) % self.stride_factor
                        
                        if pad_h > 0 or pad_w > 0 or pad_d > 0:
                            # PyTorch pad format: (D_left, D_right, W_left, W_right, H_left, H_right)  
                            padding = (0, pad_d, 0, pad_w, 0, pad_h)
                            
                            # Apply padding
                            if isinstance(pixel_array, torch.Tensor):
                                import torch.nn.functional as F
                                padded_array = F.pad(pixel_array, padding, mode=self.mode, value=self.value)
                            else:
                                # Convert to tensor, pad, then convert back
                                tensor_array = torch.from_numpy(pixel_array) if isinstance(pixel_array, np.ndarray) else pixel_array
                                import torch.nn.functional as F
                                padded_tensor = F.pad(tensor_array, padding, mode=self.mode, value=self.value)
                                padded_array = padded_tensor.numpy() if isinstance(pixel_array, np.ndarray) else padded_tensor
                            
                            # Silent padding - no logging
                            pass
                        else:
                            padded_array = pixel_array
                    
                    # Update the data with padded array, preserving metadata
                    if hasattr(array, 'affine'):
                        # Create new MetaTensor with preserved metadata
                        if hasattr(array, 'applied_operations'):
                            data_dict[key] = MetaTensor(padded_array, affine=array.affine, 
                                                       applied_operations=array.applied_operations)
                        else:
                            data_dict[key] = MetaTensor(padded_array, affine=array.affine)
                    else:
                        data_dict[key] = padded_array
                
                else:
                    pass  # Skipping unsupported shape - expected 4D (C, H, W, D)
                    
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in DynUNetPaddingd: {str(e)}")
                # Warning: Skipping key due to error
        
        return data_dict




class SequentialTransformd(MapTransform):
    """
    Generic sequential flip/swap transformation with affine compensation.
    
    Applies user-specified sequences like "s:xy f:x f:z" to MetaTensor data
    while maintaining coordinate system integrity through affine compensation.
    
    Mathematical Framework:
    - Applies transformation P@X to data volume using tensor operations
    - Calculates complement matrix P' = P^(-1) for affine compensation
    - Updates affine: A @ P' (where A is original affine, P' is complement)
    
    Supported Operations:
    - f:x, f:y, f:z - Flip along X, Y, or Z axis
    - s:xy, s:xz, s:yz - Swap coordinate pairs
    
    Examples:
    - ACDC sequence: "s:yz s:xz f:z f:x s:xy"
    - Custom sequence: "s:xy f:x f:z"
    
    Args:
        keys: Keys to apply the transformation to
        sequence: Transformation sequence string (e.g., "s:xy f:x f:z")
                 If None, defaults to ACDC sequence for backward compatibility
        allow_missing_keys: Whether to allow missing keys
    """
    
    def __init__(self, keys: KeysCollection, sequence: Optional[str] = None, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        from data.utils.geometry import parse_transform_sequence
        
        # Default to ACDC sequence for backward compatibility
        if sequence is None:
            sequence = "s:yz s:xz f:z f:x s:xy"
        
        # Parse and cache the transformation sequence
        self.sequence_str = sequence
        self.sequence_steps = parse_transform_sequence(sequence)
        self.complement_matrix_cache = {}
        
    def __call__(self, data):
        data_dict = dict(data)
        
        for key in self.keys:
            if key not in data_dict:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in data")
                
            try:
                from data.utils.geometry import apply_tensor_sequence, compose_sequence_matrix
                
                pixel_array = data_dict[key].get_array().copy()
                original_affine = data_dict[key].affine.clone()
                
                # Convert to tensor for sequential transformation
                if isinstance(pixel_array, np.ndarray):
                    pixel_tensor = torch.from_numpy(pixel_array)
                else:
                    pixel_tensor = pixel_array
                
                # Apply sequential transformation P@X using the configured sequence
                transformed_tensor = apply_tensor_sequence(pixel_tensor, self.sequence_steps)
                
                # Calculate complement matrix P' = P^(-1) for affine compensation
                # Cache by shape for performance with multiple volumes
                original_shape = pixel_array.shape[-3:] if pixel_array.ndim == 4 else pixel_array.shape
                shape_key = tuple(original_shape)
                
                if shape_key not in self.complement_matrix_cache:
                    P_matrix = compose_sequence_matrix(self.sequence_steps, original_shape)
                    complement_matrix = torch.from_numpy(np.linalg.inv(P_matrix)).float()
                    self.complement_matrix_cache[shape_key] = complement_matrix
                
                # Update affine: A @ P' (where A is original affine, P' is complement)
                # Ensure dtype compatibility for matrix multiplication
                complement_matrix = self.complement_matrix_cache[shape_key].to(original_affine.dtype)
                compensated_affine = original_affine @ complement_matrix
                
                # Update pixel array and affine
                pixel_array = transformed_tensor.numpy() if isinstance(transformed_tensor, torch.Tensor) else transformed_tensor
                data_dict[key] = MetaTensor(pixel_array, affine=compensated_affine)
                
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in SequentialTransformd: {str(e)}")
        
        return data_dict


# Backward compatibility alias
ACDCSequentialTransform = SequentialTransformd


class DatasetCanonicalizer(MapTransform):
    """
    Applies dataset-specific canonicalization transformations.
    
    Handles affine matrix adjustments for MR datasets and label cleanup.
    """
    
    def __init__(self, keys: KeysCollection, dataset: str, modal: str, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dataset = dataset.lower() if dataset else None
        self.modal = modal.lower() if modal else None
        
    def __call__(self, data):
        data_dict = dict(data)
        
        for key in self.keys:
            if key not in data_dict:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in data")
                
            try:
                pixel_array = data_dict[key].get_array().copy()
                original_affine = data_dict[key].affine.clone()
                
                # Handle MR affine swap for non-ACDC datasets
                if 'mr' in key and len(pixel_array.shape) == 4 and self.dataset != 'acdc':
                    affine = data_dict[key].affine.clone()
                    # Update the affine matrix (same logic as Adjustd)
                    m = torch.eye(4)
                    m[:3, 0] = affine[1, :3]
                    m[:3, 1] = affine[2, :3]
                    m[:3, 2] = affine[3, :3]
                    m[:3, -1] = affine[:3, -1]
                    data_dict[key] = MetaTensor(pixel_array, affine=m)
                
                # Label cleanup: merge RV-MYO (label 4) with LV-MYO (label 2)
                if "label" in key:
                    # Get current pixel array (might have been transformed above)
                    current_array = data_dict[key].get_array() if hasattr(data_dict[key], 'get_array') else data_dict[key]
                    if isinstance(current_array, torch.Tensor):
                        current_array = current_array.numpy()
                    current_array = current_array.copy()
                    current_array[current_array == 4] = 2
                    
                    # Preserve current affine and applied operations
                    current_affine = data_dict[key].affine if hasattr(data_dict[key], 'affine') else original_affine
                    current_ops = data_dict[key].applied_operations if hasattr(data_dict[key], 'applied_operations') else None
                    
                    data_dict[key] = MetaTensor(
                        current_array, 
                        affine=current_affine, 
                        applied_operations=current_ops
                    )
                    
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in DatasetCanonicalizer: {str(e)}")
        
        return data_dict


class UniversalCanonicalResampled(MapTransform):
    """
    Unified loader-orient-resample transform.

    Composes multiple transforms to:
    • Load image/label from file paths (wraps MONAI LoadImaged)
    • Apply dataset-specific canonicalization (axis swaps/flips)
    • Handle ACDC sequential transformation with affine compensation
    • Resample to target spacing and ensure 4D output shape [C,D,H,W]
    """
    
    def __init__(self,
                 keys: KeysCollection,
                 dataset: str,
                 modal: str,
                 target_spacing: tuple,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dataset = dataset.lower() if dataset else None
        self.modal = modal.lower() if modal else None
        self.target_spacing = target_spacing
        
        # Import required transforms
        from monai.transforms import LoadImaged, Spacingd
        
        # Initialize component transforms
        # CAP uses NRRD files which require ensure_channel_first=True for proper loading
        # Other MR datasets might need different handling
        ensure_channel_first = True  # Default to True for better compatibility
        if modal == "mr" and dataset not in ['acdc', 'cap']:
            ensure_channel_first = False  # Only set False for other MR datasets if needed
            
        self.loader = LoadImaged(
            keys, 
            ensure_channel_first=ensure_channel_first, 
            image_only=True, 
            allow_missing_keys=allow_missing_keys
        )
        
        # Setup ACDC sequential transformation if needed
        self.acdc_transform = None
        if dataset == "acdc":
            self.acdc_transform = ACDCSequentialTransform(keys, allow_missing_keys)
        
        # Setup dataset canonicalizer
        self.canonicalizer = DatasetCanonicalizer(keys, dataset, modal, allow_missing_keys)
        
        # Setup spacing parameters based on dataset and modality
        if dataset == "acdc":
            # After sequential transformation, ACDC will match CAP layout
            # So use the same spacing vector as CAP (MR non-ACDC)
            spacing_vector = [target_spacing[0], target_spacing[1], -1]
        elif modal == "ct":
            spacing_vector = list(target_spacing)  # isotropic
        else:  # MR non-ACDC (including CAP)
            spacing_vector = [target_spacing[0], target_spacing[1], -1]
            
        self.spacer = Spacingd(
            keys,
            spacing_vector,
            mode=("bilinear", "nearest"),
            allow_missing_keys=allow_missing_keys
        )
        
    def __call__(self, data):
        data_dict = dict(data)
        
        # Step 1: Load data
        data_dict = self.loader(data_dict)
        
        # Step 2: Apply ACDC sequential transformation if needed
        if self.acdc_transform:
            data_dict = self.acdc_transform(data_dict)
        
        # Step 3: Apply dataset canonicalization
        data_dict = self.canonicalizer(data_dict)
        
        # Step 4: Resample to target spacing
        data_dict = self.spacer(data_dict)
        
        # Step 5: Ensure 4D shape [C,D,H,W]
        for key in self.keys:
            if key in data_dict:
                tensor_data = data_dict[key]
                if hasattr(tensor_data, 'get_array'):
                    array = tensor_data.get_array()
                    if array.ndim == 3:  # Add channel dimension if missing
                        array = array.unsqueeze(0) if hasattr(array, 'unsqueeze') else array[None, ...]
                        data_dict[key] = MetaTensor(
                            array,
                            affine=tensor_data.affine,
                            applied_operations=tensor_data.applied_operations if hasattr(tensor_data, 'applied_operations') else None
                        )
        
        return data_dict


class HistogramMatchd(MapTransform):
    """
    Lightweight histogram matching transform for cross-dataset intensity normalization.
    
    This transform applies pre-computed intensity mappings using cached CDFs and LUTs
    generated by the offline histogram preprocessing pipeline.
    
    The transform handles:
    • Percentile-based intensity scaling (5th-95th percentile to [0,1] range)
    • Intensity mapping using pre-computed LUTs for source datasets
    • No thresholding applied (removed for both CT and MR)
    
    The mapping operates silently without CLI flags:
    • CT source "scotheart" → CT target "mmwhs"
    • MR source "acdc" → MR target "cap"
    
    Args:
        keys: Keys to apply histogram matching to (typically image key only)
        modal: Modality ("ct" or "mr")
        dataset: Current dataset name
        cdf_dir: Directory containing cached CDFs and LUTs (default: "./cdf_cache")
        allow_missing_keys: Whether to allow missing keys
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        modal: str,
        dataset: str,
        cdf_dir: str = "./cdf_cache",
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.modal = modal.lower() if modal else None
        self.dataset = dataset.lower() if dataset else None
        self.cdf_dir = Path(cdf_dir)
        
        # Import shared utilities
        from utils.histogram_preprocessor import HistogramUtils
        self.utils = HistogramUtils
        
        # Define target and source datasets
        if self.modal == "ct":
            self.target_dataset = "mmwhs"
            self.source_dataset = "scotheart"
        elif self.modal == "mr":
            self.target_dataset = "cap"
            self.source_dataset = "acdc"
        else:
            raise ValueError(f"Unsupported modality: {modal}. Must be 'ct' or 'mr'.")
        
        # Determine if current dataset needs histogram matching
        self.needs_matching = (self.dataset == self.source_dataset)
        
        # Cache paths
        self.lut_path = self.cdf_dir / f"{self.source_dataset}_to_{self.target_dataset}_lut.npy"
        
        # Load LUT if needed for source dataset
        self._intensity_lut = None
        if self.needs_matching:
            if self.lut_path.exists():
                self._intensity_lut = np.load(self.lut_path)
            else:
                raise FileNotFoundError(
                    f"Required LUT not found: {self.lut_path}\n"
                    f"Please run histogram preprocessing first:\n"
                    f"python -m utils.histogram_preprocessor <data_root> --stage b"
                )
    
    def _apply_intensity_map(self, img: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
        """Apply intensity mapping to image using lookup table. Input should already be scaled to [0,1]."""
        original_dtype = img.dtype
        original_device = img.device
        
        # Convert to numpy for processing
        if torch.is_tensor(img):
            img_np = img.cpu().numpy()
        else:
            img_np = np.asarray(img)
        
        # Input should already be in [0,1] range from caller's scaling
        # Convert to LUT index range (0-1023 for new 1024-bin float32 LUTs)
        lut_bins = len(lut)
        img_indices = np.clip(img_np * (lut_bins - 1), 0, lut_bins - 1).astype(np.int32)
        
        # Apply lookup table (LUT values are already normalized to [0,1])
        img_result = lut[img_indices].astype(np.float32)
        
        # Convert back to tensor with original dtype and device
        result_tensor = torch.from_numpy(img_result).to(dtype=original_dtype, device=original_device)
        
        return result_tensor
    
    def __call__(self, data):
        data_dict = dict(data)
        
        for key in self.keys:
            if key not in data_dict:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in data")
            
            try:
                # Get array from MetaTensor
                meta_tensor = data_dict[key]
                if hasattr(meta_tensor, 'get_array'):
                    img_array = meta_tensor.get_array()
                else:
                    img_array = meta_tensor
                
                # Ensure we work with CPU arrays
                if hasattr(img_array, 'is_cuda') and img_array.is_cuda:
                    img_array = img_array.cpu()
                
                # Convert to numpy if tensor
                if isinstance(img_array, torch.Tensor):
                    img_numpy = img_array.numpy()
                else:
                    img_numpy = np.asarray(img_array)
                
                # Step 1: Apply modality-aware scaling to [0, 1] range (single scaling with CT windowing)
                img_scaled = self.utils.scale_to_percentile(img_numpy, modal=self.modal)
                
                # Step 2: Apply histogram matching (for source datasets only)
                if self.needs_matching and self._intensity_lut is not None:
                    # Apply intensity mapping using pre-computed LUT (input already scaled)
                    img_mapped = self._apply_intensity_map(torch.from_numpy(img_scaled), self._intensity_lut)
                    img_final = img_mapped.numpy()
                else:
                    img_final = img_scaled
                
                # Step 4: Update data with transformed image
                # Preserve metadata from original MetaTensor
                if hasattr(meta_tensor, 'affine'):
                    affine = meta_tensor.affine
                    applied_ops = meta_tensor.applied_operations if hasattr(meta_tensor, 'applied_operations') else None
                    data_dict[key] = MetaTensor(img_final, affine=affine, applied_operations=applied_ops)
                else:
                    data_dict[key] = img_final
                
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in HistogramMatchd: {str(e)}")
        
        return data_dict

