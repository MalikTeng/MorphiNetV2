from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized, ScaleIntensityRangePercentilesd
from monai.transforms.utils import distance_transform_edt
import os
from typing import Optional, Tuple
from scipy import stats

# import nibabel as nib  # Unused import


__all__ = ["Maskd", "DFConvertd", "FlexResized", "DynUNetPaddingd", "SequentialTransformd", "DatasetCanonicalizer", "UniversalCanonicalResampled", "DynamicIntensityRangeScalesd"]


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


class FlexResized(MapTransform):
    """
    Flexible resize transform that resizes image/label to a fixed scale at the second dimension.
    Supports -1 as a wildcard to preserve original dimensions.
    
    Args:
        keys: Keys to apply the transform to (e.g., ["pred", "label"])
        size: Target size tuple, -1 preserves original dimension (e.g., (-1, 128, -1))
        allow_missing_keys: Whether to allow missing keys
        force_nearest: Whether to force nearest interpolation for all keys (useful for labels)
        min_dimension_size: Minimum allowed size for any dimension to prevent compression to zero
    """
    def __init__(self, keys: KeysCollection, size: tuple, allow_missing_keys: bool = False, 
                 force_nearest: bool = False, min_dimension_size: int = 4) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_size = np.array([int(s) for s in size])
        self.allow_missing_keys = allow_missing_keys
        self.force_nearest = force_nearest
        self.min_dimension_size = min_dimension_size

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
        
        # IMPROVED CALCULATION: Prevent dimension compression
        if len(final_size) > 1 and final_size[1] != current_shape[1]:
            rescale_ratio = final_size[1] / current_shape[1]
            
            # Calculate new shape with dimension safeguards
            new_shape = []
            for i, d in enumerate(current_shape):
                if i == 1:
                    # Second dimension: set to exact target
                    new_dim = int(final_size[1])
                else:
                    # Other dimensions: apply rescale ratio but enforce minimum size
                    scaled_dim = int(np.ceil(d * rescale_ratio))
                    new_dim = max(scaled_dim, self.min_dimension_size)
                new_shape.append(new_dim)
            
        else:
            new_shape = [int(s) for s in final_size]
            # Apply minimum dimension enforcement even when no rescaling
            new_shape = [max(dim, self.min_dimension_size) for dim in new_shape]
        
        # Final validation: ensure no zero dimensions
        if any(dim <= 0 for dim in new_shape):
            new_shape = [max(dim, self.min_dimension_size) for dim in new_shape]
        
        
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
            df.append(df_class)

        df = MetaTensor(torch.cat(df, dim=0), affine=data[self.key].affine)

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
                
                if len(original_shape) == 4:  
                    # Handle different 4D formats: (C, H, W, D) for most datasets, (time_frame, H, W, D) for CAP
                    dim0, h, w, d = original_shape
                    
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


class DatasetCanonicalizer(MapTransform):
    """
    Applies dataset-specific canonicalization transformations.
    
    Handles affine matrix adjustments for CAP datasets and label cleanup.
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
                
                # Handle MR affine swap for CAP dataset (4D format without channel dimension)
                if 'mr' in key and self.dataset == 'cap' and len(pixel_array.shape) == 4:
                    affine = data_dict[key].affine.clone()
                    # Update the affine matrix
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
    • Resample to target spacing and ensure 4D output shape [C,H,W,D]
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
        if dataset == "cap" and modal == "mr":
            ensure_channel_first = False  # CAP MR: Load as (C=time_frame, H, W, D)
        else:
            ensure_channel_first = True   # CT and other datasets: Add channel dimension (C, H, W, D)
            
        self.loader = LoadImaged(
            keys, 
            ensure_channel_first=ensure_channel_first, 
            image_only=True, 
            allow_missing_keys=allow_missing_keys
        )
        
        # Setup ACDC sequential transformation if needed
        self.acdc_transform = None
        if dataset == "acdc":
            self.acdc_transform = SequentialTransformd(keys, sequence=None, allow_missing_keys=allow_missing_keys)
        
        # Setup dataset canonicalizer
        self.canonicalizer = DatasetCanonicalizer(keys, dataset, modal, allow_missing_keys)
        
        # Setup spacing parameters based on dataset and modality
        if modal == "ct":
            spacing_vector = list(target_spacing)   #isotropic
        else:
            spacing_vector = [target_spacing[0], target_spacing[1], -1] # use the same spacing vector as CAP (MR non-ACDC)
            
        # Dynamic mode configuration based on actual keys
        mode_list = []
        for key in keys:
            if 'image' in key.lower():
                mode_list.append('bilinear')
            elif 'label' in key.lower():
                mode_list.append('nearest')
            else:
                # Default to bilinear for unknown key types
                mode_list.append('bilinear')
        
        # Convert to tuple for MONAI compatibility
        mode_tuple = tuple(mode_list) if len(mode_list) > 1 else mode_list[0]
        
        self.spacer = Spacingd(
            keys,
            spacing_vector,
            mode=mode_tuple,
            allow_missing_keys=allow_missing_keys
        )
        
        # Store spacing vector and mode configuration for debugging
        self.spacing_vector = spacing_vector
        self.mode_tuple = mode_tuple
        
    def __call__(self, data):
        data_dict = dict(data)
        
        # Step 1: Load data
        data_dict = self.loader(data_dict)
        
        # Step 2: Apply ACDC sequential transformation if needed
        if self.acdc_transform:
            data_dict = self.acdc_transform(data_dict)
        
        # Step 3: Apply dataset canonicalization
        data_dict = self.canonicalizer(data_dict)
        
        # Step 4: Resample to target spacing (CRITICAL STEP)
        data_dict = self.spacer(data_dict)
        
        # Validate output after Spacingd
        for key in self.keys:
            if key in data_dict:
                tensor_data = data_dict[key]
                if hasattr(tensor_data, 'affine') and tensor_data.affine is not None:
                    import numpy as np
                    pixdim = [np.linalg.norm(tensor_data.affine[:3, i]) for i in range(3)]
                    if any(np.isnan(pixdim)) or any(np.isinf(pixdim)):
                        raise ValueError(f"Invalid pixdim after Spacingd for {key}: {pixdim}")
                    
                    # Check if spacing was applied correctly (allow some tolerance)
                    expected_spacing = self.spacing_vector
                    for i, (actual, expected) in enumerate(zip(pixdim, expected_spacing)):
                        if expected != -1:  # -1 means preserve original spacing
                            if abs(actual - expected) > 0.1:
                                import warnings
                                warnings.warn(f"Warning: {key} dimension {i} spacing mismatch. "
                                            f"Expected: {expected}, Got: {actual:.3f}")
        
        # Ensure proper shape format
        assert data_dict[self.keys[0]].get_array().ndim == 4, "Output should be 4D"
        
        return data_dict


class DynamicIntensityRangeScalesd(MapTransform):
    """
    Dynamic intensity range scaling based on background detection.
    
    This transform automatically detects the background (air) intensity in medical images
    and uses it to dynamically determine the lower percentile for intensity rescaling.
    The background is identified as the most common intensity value in the image.
    
    Workflow:
    1. Extract pixel array from MetaTensor
    2. Find background intensity (mode) using histogram analysis
    3. Calculate what percentile the background intensity represents
    4. Apply ScaleIntensityRangePercentilesd with dynamic percentile range
    
    Args:
        keys: Keys to apply the transform to (typically image keys only)
        upper_percentile: Upper percentile for scaling (default: 99.0)
        b_min: Target minimum intensity value (default: 0.0)
        b_max: Target maximum intensity value (default: 1.0)
        clip: Whether to clip values outside the range (default: True)
        num_bins: Number of histogram bins for background detection (default: 256)
        min_background_percentile: Minimum allowed background percentile (default: 1.0)
        max_background_percentile: Maximum allowed background percentile (default: 80.0)
        allow_missing_keys: Whether to allow missing keys (default: False)
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        dataset: str = None,
        dual_background_threshold: float = 200.0,
        upper_percentile: float = 99.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        num_bins: int = 256,
        min_background_percentile: float = 1.0,
        max_background_percentile: float = 80.0,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dataset = dataset.lower() if dataset else None
        self.dual_background_threshold = dual_background_threshold
        self.upper_percentile = upper_percentile
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.num_bins = num_bins
        self.min_background_percentile = min_background_percentile
        self.max_background_percentile = max_background_percentile
        
    def _detect_background_intensity(self, pixel_array: np.ndarray) -> tuple:
        """
        Detect background intensity using histogram analysis.
        
        For SCOTHEART dataset: Handles dual-background scenario where air pixels
        around -900 HU and padded pixels around -1100 HU both represent background.
        
        Algorithm:
        1. Create histogram and find lowest intensity bins with significant counts
        2. Check if the two lowest intensities differ by ≤ dual_background_threshold  
        3. If SCOTHEART + dual background detected: use maximum of the two (more conservative)
        4. Otherwise: return single lowest intensity (value, value)
        
        Args:
            pixel_array: Flattened pixel intensity array
            
        Returns:
            Tuple of (intensity, intensity) for background intensity (consistent API)
        """
        # Remove any NaN or infinite values
        valid_pixels = pixel_array[np.isfinite(pixel_array)]
        
        if len(valid_pixels) == 0:
            raise ValueError("No valid pixel values found in the image")
        
        # Create histogram to find intensity distribution
        hist, bin_edges = np.histogram(valid_pixels, bins=self.num_bins)
        
        # Standard single-peak detection for non-SCOTHEART datasets
        if self.dataset != 'scotheart':
            max_bin_idx = np.argmax(hist)
            background_intensity = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            return (background_intensity, background_intensity)
        
        # SCOTHEART-specific dual-background detection
        # For CT images, background (air) has lowest intensities around -900 to -1100 HU
        # Find the two lowest intensity peaks (not highest!)
        
        # Only consider bins with significant counts to avoid noise
        min_count_threshold = max(10, len(valid_pixels) * 0.001)  # At least 0.1% of pixels
        significant_bins = hist >= min_count_threshold
        
        if not np.any(significant_bins):
            # Fallback to highest peak if no significant bins found
            max_bin_idx = np.argmax(hist)
            background_intensity = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            return (background_intensity, background_intensity)
        
        # Get indices of significant bins, sorted by intensity (ascending for lowest values)
        significant_indices = np.where(significant_bins)[0]
        
        # Find the two lowest intensity bins with significant counts
        if len(significant_indices) >= 2:
            # Get the two lowest intensity bins
            first_low_idx = significant_indices[0]   # Lowest intensity bin
            second_low_idx = significant_indices[1]  # Second lowest intensity bin
            
            first_intensity = (bin_edges[first_low_idx] + bin_edges[first_low_idx + 1]) / 2
            second_intensity = (bin_edges[second_low_idx] + bin_edges[second_low_idx + 1]) / 2
            
            # Check if intensities are within threshold (dual-background condition)
            intensity_diff = abs(first_intensity - second_intensity)
            
            if intensity_diff <= self.dual_background_threshold:
                # Dual-background detected: use maximum of the two background peaks
                # (higher value to be more conservative about excluding tissue)
                combined_intensity = max(first_intensity, second_intensity)
                return (combined_intensity, combined_intensity)
            else:
                # Peaks too far apart: use the lowest intensity
                background_intensity = first_intensity
                return (background_intensity, background_intensity)
        else:
            # Only one significant bin found, use it
            first_low_idx = significant_indices[0]
            background_intensity = (bin_edges[first_low_idx] + bin_edges[first_low_idx + 1]) / 2
            return (background_intensity, background_intensity)
    
    def _calculate_background_percentile(self, pixel_array: np.ndarray, background_range: tuple) -> float:
        """
        Calculate what percentile the background intensity represents.
        
        For SCOTHEART dual-background: Uses the selected background intensity (conservative approach)
        For other datasets: Uses the single detected intensity value
        
        Args:
            pixel_array: Flattened pixel intensity array
            background_range: Tuple of (intensity, intensity) for background intensity
            
        Returns:
            Percentile rank of the background intensity
        """
        # Remove any NaN or infinite values
        valid_pixels = pixel_array[np.isfinite(pixel_array)]
        
        background_intensity, _ = background_range
        
        # Since we now use minimum approach, both values in the tuple are the same
        # Extract the single background intensity value
        
        # Calculate percentile rank of background intensity
        percentile_rank = stats.percentileofscore(valid_pixels, background_intensity, kind='mean')
        
        # Clamp to reasonable bounds
        percentile_rank = np.clip(percentile_rank, self.min_background_percentile, self.max_background_percentile)
        
        return percentile_rank
    
    def __call__(self, data):
        data_dict = dict(data)
        
        for key in self.keys:
            if key not in data_dict:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in data")
            
            try:
                # Extract pixel array from MetaTensor
                if hasattr(data_dict[key], 'get_array'):
                    pixel_array = data_dict[key].get_array()
                else:
                    pixel_array = data_dict[key]
                
                # Ensure we work with CPU arrays
                if hasattr(pixel_array, 'is_cuda') and pixel_array.is_cuda:
                    pixel_array = pixel_array.cpu()
                
                # Convert to numpy if needed and ensure float32
                if isinstance(pixel_array, torch.Tensor):
                    pixel_array = pixel_array.float().numpy()
                else:
                    pixel_array = pixel_array.astype(np.float32)
                
                # Flatten for analysis (exclude channel dimension if present)
                if pixel_array.ndim == 4:  # (C, H, W, D)
                    flat_pixels = pixel_array.flatten()
                elif pixel_array.ndim == 3:  # (H, W, D)
                    flat_pixels = pixel_array.flatten()
                else:
                    flat_pixels = pixel_array.flatten()
                
                # Detect background intensity range
                background_range = self._detect_background_intensity(flat_pixels)
                
                # Calculate background percentile
                background_percentile = self._calculate_background_percentile(flat_pixels, background_range)
                
                # Apply ScaleIntensityRangePercentilesd with dynamic parameters
                scale_transform = ScaleIntensityRangePercentilesd(
                    keys=[key],
                    lower=background_percentile,
                    upper=self.upper_percentile,
                    b_min=self.b_min,
                    b_max=self.b_max,
                    clip=self.clip,
                    allow_missing_keys=self.allow_missing_keys
                )
                
                # Apply the transform
                data_dict = scale_transform(data_dict)
                
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in DynamicIntensityRangeScalesd: {str(e)}")
        
        return data_dict

