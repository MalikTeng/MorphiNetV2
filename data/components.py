from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized, ScaleIntensityRangePercentilesd
from monai.transforms.utils import distance_transform_edt
import os
from typing import Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks

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
                array = array.get_array() # (C, H, D, W)

                if data["modal"] == "ct" and "pred" in key:
                    # mask the CT images near the basal and apex plane
                    mask = np.zeros_like(array).astype(bool)
                    mask[:, :, 16:-16] = True
                    array[~mask] = array.min()

                    data[key] = MetaTensor(array, affine=data[key].affine, 
                                           applied_operations=data[key].applied_operations)
                
                elif data["modal"] == "mr":
                    # pad slices on the top and bottom along the SAX direction
                    array = np.pad(array, ((0, 0), (0, 0), (16, 16), (0, 0)), mode="constant", constant_values=array.min())
                    # update the affine
                    affine = data[key].affine.clone()   # affine is 4x4
                    affine[:3, -2] -= 16 * data[key].pixdim[0]
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
        assert len(size) == 3, "Size must be a 3-tuple"
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
        current_shape = np.array(data[reference_key].get_array().shape[1:])  # Skip channel dim
        
        assert len(self.target_size) == len(current_shape), "Target size and current shape must have the same length"
        
        # Replace -1 with current dimensions
        final_size = np.where(self.target_size == -1, current_shape, self.target_size)
        
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
                
                # CAP data now harmonized - no special affine handling needed
                # if 'mr' in key and self.dataset == 'cap' and len(pixel_array.shape) == 4:
                #     affine = data_dict[key].affine.clone()
                #     # Update the affine matrix
                #     m = torch.eye(4)
                #     m[:3, 0] = affine[1, :3]
                #     m[:3, 1] = affine[2, :3]
                #     m[:3, 2] = affine[3, :3]
                #     m[:3, -1] = affine[:3, -1]
                #     data_dict[key] = MetaTensor(pixel_array, affine=m)
                
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
        # CAP data now harmonized as 3D NIFTI - treat same as other datasets
        ensure_channel_first = True   # All datasets: Add channel dimension (C, H, W, D)
            
        self.loader = LoadImaged(
            keys, 
            ensure_channel_first=ensure_channel_first, 
            image_only=True, 
            allow_missing_keys=allow_missing_keys
        )
        
        # ACDC sequential transformation (handling orientation error)
        if dataset == "acdc":
            self.acdc_transform = SequentialTransformd(keys, sequence="f:x f:z")
        else:
            self.acdc_transform = None
        
        # Setup dataset canonicalizer
        self.canonicalizer = DatasetCanonicalizer(keys, dataset, modal, allow_missing_keys)
        
        # Setup spacing parameters based on dataset and modality
        if modal == "ct":
            spacing_vector = list(target_spacing)   #isotropic
        else:
            spacing_vector = [target_spacing[0], target_spacing[1], -1]
            
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
        
        # Step 2: Skip ACDC transformation
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
    Dynamic intensity range scaling with optional dual-background clipping.
    
    Detects background intensity in medical images and applies dynamic scaling.
    For SCOTHEART CT data with dual-background (air and padding artifacts), 
    clips the lower intensity peak before scaling.
    
    Workflow:
    1. Detect background intensity using histogram peak analysis
    2. For SCOTHEART: Clip pixels below higher background peak if dual-background detected
    3. Calculate background percentile for intensity scaling  
    4. Apply percentile-based intensity scaling to [0, 1] range
    
    Args:
        keys: Keys to apply transform to (typically image keys only)
        dataset: Dataset name for dataset-specific processing
        upper_percentile: Upper percentile for scaling (default: 99.0)
        b_min: Target minimum intensity (default: 0.0)
        b_max: Target maximum intensity (default: 1.0)
        clip: Whether to clip values outside range (default: True)
        num_bins: Histogram bins for background detection (default: 256)
        min_background_percentile: Minimum background percentile (default: 1.0)
        max_background_percentile: Maximum background percentile (default: 80.0)
        allow_missing_keys: Whether to allow missing keys (default: False)
        debug: Enable debug mode for detailed logging (default: False)
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        dataset: str = None,
        upper_percentile: float = 99.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        num_bins: int = 256,
        min_background_percentile: float = 1.0,
        max_background_percentile: float = 80.0,
        allow_missing_keys: bool = False,
        debug: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dataset = dataset.lower() if dataset else None
        self.upper_percentile = upper_percentile
        self.debug = debug
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.num_bins = num_bins
        self.min_background_percentile = min_background_percentile
        self.max_background_percentile = max_background_percentile
        # Convert boolean debug to level (0=off, 1=basic, 2=full)
        self.debug_level = 2 if debug else 0
        
    def _create_debug_info(self, detection_type: str, **kwargs) -> dict:
        """Create standardized debug info structure.
        
        Args:
            detection_type: Type of detection ('mr_dual_peaks', 'scotheart_dual', 'mmwhs_plateau', 'single')
            **kwargs: Additional debug information
        
        Returns:
            Standardized debug dictionary
        """
        base_info = {
            'detection_type': detection_type,
            'method': kwargs.get('method', detection_type),
            'threshold': kwargs.get('threshold'),
            'detected': kwargs.get('detected', False)
        }
        
        if self.debug_level >= 2:  # Full debug
            base_info.update(kwargs)
        elif self.debug_level == 1:  # Basic debug
            # Only keep essential fields
            for key in ['threshold', 'clipping_applied', 'selection_method']:
                if key in kwargs:
                    base_info[key] = kwargs[key]
        
        return base_info
    
    def _validate_peaks(self, peaks: list, criteria: dict, required: int = 4) -> tuple:
        """Generic peak validation with configurable criteria.
        
        Args:
            peaks: List of detected peaks
            criteria: Dictionary of validation criteria (name -> bool)
            required: Minimum criteria to pass
        
        Returns:
            Tuple of (is_valid, validation_info)
        """
        if len(peaks) < 2:
            return False, {'validation_passed': False, 'reason': 'insufficient_peaks'}
        
        passed = sum(criteria.values())
        is_valid = passed >= required
        
        validation_info = {
            'validation_passed': is_valid,
            'criteria_met': f"{passed}/{len(criteria)}"
        }
        
        if self.debug_level >= 2:
            validation_info['validation_checks'] = criteria
            validation_info['passed_validations'] = passed
            validation_info['required_validations'] = required
        
        return is_valid, validation_info
    
    def _find_ct_background_peaks(self, hist: np.ndarray, bin_centers: np.ndarray, 
                                 all_pixels: np.ndarray) -> tuple:
        """
        Unified CT peak detection for both SCOTHEART and MMWHS datasets.
        
        Returns:
            Tuple of (peaks_list, detection_type)
            - peaks_list: List of (bin_idx, count, intensity) tuples
            - detection_type: 'dual_background', 'plateau', or 'single'
        """
        # First try SCOTHEART dual-background detection
        scotheart_peaks = self._find_scotheart_background_peaks(hist, bin_centers, all_pixels)
        
        # Then try MMWHS plateau detection  
        mmwhs_peaks = self._find_mmwhs_plateau_peaks(hist, bin_centers, all_pixels)
        
        # Determine best approach based on peak characteristics
        if len(scotheart_peaks) >= 2:
            # Check if peaks are in typical SCOTHEART range (-1400 to -700)
            in_range = all(p[2] >= -1400 and p[2] <= -700 for p in scotheart_peaks[:2])
            if in_range:
                return scotheart_peaks, 'dual_background'
        
        if len(mmwhs_peaks) >= 2:
            # Check for U-shape plateau pattern (>150 HU separation)
            intensity_sep = abs(mmwhs_peaks[-1][2] - mmwhs_peaks[0][2])
            if intensity_sep >= 150.0:
                return mmwhs_peaks, 'plateau'
        
        # Fallback to single peak
        all_peaks = scotheart_peaks + mmwhs_peaks
        if all_peaks:
            return all_peaks[:1], 'single'
        else:
            return [], 'single'
    
    def _find_scotheart_background_peaks(self, hist: np.ndarray, bin_centers: np.ndarray, 
                                        all_pixels: np.ndarray) -> list:
        """Find dual-background peaks specific to SCOTHEART CT data."""
        # Target background region where air (-1100) and padding (-900) peaks appear
        background_mask = (bin_centers >= -1400) & (bin_centers <= -700)
        background_hist = hist[background_mask]
        background_bins = bin_centers[background_mask]
        
        if len(background_hist) == 0:
            return []
        
        # Find local maxima with sufficient pixel count
        peaks = []
        min_count = max(100, len(all_pixels) * 0.001)
        
        for i in range(1, len(background_hist) - 1):
            if (background_hist[i] > background_hist[i-1] and 
                background_hist[i] > background_hist[i+1] and
                background_hist[i] >= min_count):
                bin_idx = np.where(background_mask)[0][i]
                peaks.append((bin_idx, background_hist[i], background_bins[i]))
        
        # Return top 2 peaks sorted by intensity
        peaks.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        prominent_peaks = peaks[:2] if len(peaks) >= 2 else peaks
        prominent_peaks.sort(key=lambda x: x[2])  # Sort by intensity
        
        return prominent_peaks
    
    def _find_mmwhs_plateau_peaks(self, hist: np.ndarray, bin_centers: np.ndarray, 
                                 all_pixels: np.ndarray) -> list:
        """
        Find U-shape plateau boundary peaks specific to MMWHS CT data.
        Identifies leftmost and rightmost significant peaks that define plateau boundaries
        for subsequent clipping to remove flat distribution artifacts.
        
        Args:
            hist: Histogram counts array
            bin_centers: Bin center intensity values  
            all_pixels: All pixel values for threshold calculation
            
        Returns:
            List of plateau boundary peaks (bin_idx, count, intensity) sorted by intensity
        """
        # Algorithm: Scan full histogram for significant peaks, identify U-shape boundaries
        # 1. Find all peaks with sufficient pixel count (>0.1% threshold)
        # 2. Filter peaks by minimum prominence to avoid noise
        # 3. Identify leftmost peak (plateau start) and rightmost peak (plateau end)
        # 4. Validate minimum intensity separation (>150 HU) between boundaries
        # 5. Return boundary peaks for plateau removal (clip below left peak)
        
        if len(hist) == 0:
            return []
        
        # Use scipy.signal.find_peaks for robust peak detection
        min_count = max(100, len(all_pixels) * 0.001)  # 0.1% of pixels minimum
        peak_indices = find_peaks(hist, height=min_count, prominence=min_count * 0.5)[0]
        
        if len(peak_indices) == 0:
            return []
        
        # Convert peak indices to (bin_idx, count, intensity) format
        peaks = []
        for peak_idx in peak_indices:
            if peak_idx < len(bin_centers):
                peaks.append((peak_idx, hist[peak_idx], bin_centers[peak_idx]))
        
        # Sort by intensity to identify plateau boundaries
        peaks.sort(key=lambda x: x[2])  # Sort by intensity
        
        # For U-shape plateau: need at least 2 significant peaks with proper separation
        if len(peaks) >= 2:
            leftmost_peak = peaks[0]   # Plateau start (background side)
            rightmost_peak = peaks[-1]  # Plateau end (tissue side)
            
            # Validate minimum intensity separation for true plateau
            intensity_separation = rightmost_peak[2] - leftmost_peak[2]
            if intensity_separation >= 150.0:  # Minimum 150 HU separation
                return [leftmost_peak, rightmost_peak]
        
        # Return all peaks if insufficient separation (fallback case)
        return peaks[:2] if len(peaks) >= 2 else peaks
    
    def _find_mr_background_noise_peaks(self, hist: np.ndarray, bin_centers: np.ndarray, 
                                       all_pixels: np.ndarray) -> list:
        """Detect background/noise peaks in MR low-intensity region.
        
        Args:
            hist: Histogram counts
            bin_centers: Bin center intensities
            all_pixels: All pixel values
        
        Returns:
            List of (bin_idx, count, intensity) tuples for up to 2 peaks
        """
        # Focus on 0-20% intensity range
        max_intensity = np.percentile(all_pixels, 99)
        low_mask = bin_centers <= (0.2 * max_intensity)
        
        if not low_mask.any():
            return []
        
        # Find peaks with sufficient pixels
        min_count = max(100, len(all_pixels) * 0.001)
        low_hist = hist[low_mask]
        low_bins = bin_centers[low_mask]
        
        peak_indices = find_peaks(low_hist, height=min_count, prominence=min_count * 0.5)[0]
        
        peaks = []
        for idx in peak_indices:
            full_idx = np.where(low_mask)[0][idx]
            peaks.append((full_idx, low_hist[idx], low_bins[idx]))
        
        return sorted(peaks, key=lambda x: x[2])[:2]
    
    def _validate_mr_dual_peaks(self, peaks: list, all_pixels: np.ndarray) -> tuple:
        """Validate MR dual peaks (background/noise)."""
        if len(peaks) < 2:
            return self._handle_mr_fallback(all_pixels)
        
        p1, p2 = peaks[0], peaks[1]
        max_int = np.percentile(all_pixels, 99)
        int_range = max_int - np.min(all_pixels)
        
        # Validate using generic validator
        valid, val_info = self._validate_peaks(peaks, {
            'low_intensity': p2[2] <= 0.2 * max_int,
            'near_zero': p1[2] <= 0.05 * max_int,
            'separated': (p2[2] - p1[2]) >= 0.02 * int_range,
            'sufficient_pixels': p2[1] >= len(all_pixels) * 0.005,
            'below_tissue': p2[2] < 0.15 * max_int,
            'balanced': p2[1] / max(p1[1], 1) <= 10.0
        })
        
        if valid:
            threshold = p2[2]
            return (threshold, threshold), self._create_debug_info(
                'mr_dual_peaks',
                detected=True,
                threshold=threshold,
                mr_dual_peaks_detected=True,  # Keep for compatibility
                noise_threshold=threshold,
                peaks=[p1[2], p2[2]] if self.debug_level >= 2 else None,
                background_peak={'intensity': p1[2]} if self.debug_level >= 2 else None,
                noise_peak={'intensity': p2[2]} if self.debug_level >= 2 else None,
                selection_method='validated_mr_dual_peaks',
                **val_info
            )
        
        return self._handle_mr_fallback(all_pixels, attempted=True, validation_info=val_info)
    
    def _handle_mr_fallback(self, all_pixels: np.ndarray, attempted: bool = False, 
                           validation_info: dict = None) -> tuple:
        """Fallback to percentile method for MR."""
        threshold = float(np.percentile(all_pixels, 5))
        
        debug_kwargs = {
            'threshold': threshold,
            'mr_dual_peaks_detected': False,
            'selected_intensity': threshold,
            'selection_method': 'percentile_fallback',
            'percentile_used': 5
        }
        
        if attempted and validation_info:
            debug_kwargs.update(validation_info)
        
        return (threshold, threshold), self._create_debug_info('mr_fallback', **debug_kwargs)
    
    def _validate_plateau_detection(self, peaks: list, all_pixels: np.ndarray) -> tuple:
        """
        Validate MMWHS plateau detection and return clipping threshold.
        Verifies plateau characteristics and selects appropriate clipping intensity
        to remove U-shape distribution artifacts while preserving tissue data.
        
        Args:
            peaks: List of detected plateau boundary peaks
            all_pixels: All pixel values for validation
            
        Returns:
            Tuple of (clipping_threshold, debug_info)
        """
        if len(peaks) < 2:
            return self._handle_single_background(peaks, all_pixels)
        
        # Take leftmost and rightmost peaks as plateau boundaries
        left_peak, right_peak = peaks[0], peaks[-1]
        
        intensity_separation = abs(right_peak[2] - left_peak[2])
        count_ratio = max(left_peak[1], right_peak[1]) / max(min(left_peak[1], right_peak[1]), 1)
        
        # MMWHS plateau validation criteria
        validation_checks = {
            'sufficient_separation': intensity_separation >= 150.0,  # Minimum 150 HU for plateau
            'reasonable_separation': intensity_separation <= 2000.0,  # Max reasonable range
            'sufficient_left_count': left_peak[1] >= len(all_pixels) * 0.001,  # 0.1% minimum
            'sufficient_right_count': right_peak[1] >= len(all_pixels) * 0.001,  # 0.1% minimum
            'balanced_prominence': count_ratio <= 50.0,  # Allow significant imbalance
            'left_is_background': left_peak[2] <= 0,  # Left peak should be background/air
        }
        
        passed_validations = sum(validation_checks.values())
        required_validations = 4  # Need at least 4/6 criteria
        
        plateau_valid = passed_validations >= required_validations
        
        if plateau_valid:
            # Use left peak intensity as clipping threshold (remove plateau start)
            clipping_threshold = left_peak[2]
            
            debug_info = self._create_debug_info(
                'mmwhs_plateau',
                plateau_detected=True,
                detected=True,
                threshold=clipping_threshold,
                validation_checks=validation_checks,
                passed_validations=passed_validations,
                required_validations=required_validations,
                left_peak={'bin_idx': left_peak[0], 'count': left_peak[1], 'intensity': left_peak[2]},
                right_peak={'bin_idx': right_peak[0], 'count': right_peak[1], 'intensity': right_peak[2]},
                intensity_separation=intensity_separation,
                clipping_threshold=clipping_threshold,
                selection_method='validated_plateau_detection'
            )
            
            return (clipping_threshold, clipping_threshold), debug_info
        else:
            # Fall back to single background approach
            single_result, single_debug = self._handle_single_background([left_peak], all_pixels)
            
            # Add plateau validation failure info
            single_debug.update({
                'plateau_attempted': True,
                'validation_checks': validation_checks,
                'passed_validations': passed_validations,
                'required_validations': required_validations,
                'fallback_reason': 'plateau_validation_failed'
            })
            
            return single_result, single_debug
    
    def _validate_ct_peaks(self, peaks: list, detection_type: str, 
                          bin_centers: np.ndarray, all_pixels: np.ndarray) -> tuple:
        """
        Route to appropriate validation based on detection type.
        
        Args:
            peaks: List of detected peaks
            detection_type: Type of detection ('dual_background', 'plateau', 'single')
            bin_centers: Bin center intensities
            all_pixels: All pixel values
            
        Returns:
            Tuple of (result, debug_info)
        """
        if detection_type == 'dual_background':
            return self._validate_dual_background(peaks, bin_centers, all_pixels)
        elif detection_type == 'plateau':
            return self._validate_plateau_detection(peaks, all_pixels)
        else:
            return self._handle_single_background(peaks, all_pixels)
    
    def _validate_dual_background(self, peaks: list, bin_centers: np.ndarray, 
                                 all_pixels: np.ndarray) -> tuple:
        """
        Validate dual-background detection with robust criteria.
        
        Args:
            peaks: List of detected peaks (bin_idx, count, intensity)
            bin_centers: All bin center intensities
            all_pixels: All pixel values for validation
            
        Returns:
            Tuple of (selected_intensity, debug_info)
        """
        if len(peaks) < 2:
            return self._handle_single_background(peaks, all_pixels)
        
        # Take the two lowest intensity peaks
        peak1, peak2 = peaks[0], peaks[1]
        
        intensity_diff = abs(peak1[2] - peak2[2])
        spatial_distance = abs(peak1[0] - peak2[0])
        count_ratio = max(peak1[1], peak2[1]) / max(min(peak1[1], peak2[1]), 1)
        
        # Enhanced validation criteria for SCOTHEART dual-background
        # Relaxed to better detect actual dual-background scenarios
        validation_checks = {
            'min_intensity_separation': intensity_diff >= 100.0,  # At least 100 HU apart (air vs padding)
            'max_intensity_separation': intensity_diff <= 400.0,  # Not too far apart
            'min_spatial_separation': spatial_distance >= 5,      # At least 5 bins apart (~50 HU)
            'both_in_background': peak1[2] <= -800 and peak2[2] <= -800,  # Background range
            'sufficient_count_peak1': peak1[1] >= len(all_pixels) * 0.005,  # 0.5% of pixels
            'sufficient_count_peak2': peak2[1] >= len(all_pixels) * 0.001,  # 0.1% of pixels
            'balanced_prominence': count_ratio <= 20.0,  # Allow more imbalance
            'reasonable_range': peak1[2] >= -1400 and peak2[2] <= -800  # Typical SCOTHEART range
        }
        
        # Count passed validations
        passed_validations = sum(validation_checks.values())
        required_validations = 5  # Need at least 5/8 criteria (more lenient)
        
        dual_background_valid = passed_validations >= required_validations
        
        if dual_background_valid:
            # Use the higher intensity peak (more conservative)
            selected_intensity = max(peak1[2], peak2[2])
            
            debug_info = self._create_debug_info(
                'scotheart_dual',
                dual_background_detected=True,
                detected=True,
                threshold=selected_intensity,
                validation_checks=validation_checks,
                passed_validations=passed_validations,
                required_validations=required_validations,
                peak1={'bin_idx': peak1[0], 'count': peak1[1], 'intensity': peak1[2]},
                peak2={'bin_idx': peak2[0], 'count': peak2[1], 'intensity': peak2[2]},
                intensity_difference=intensity_diff,
                spatial_distance=spatial_distance,
                selected_intensity=selected_intensity,
                selection_method='validated_dual_background'
            )
            
            return (selected_intensity, selected_intensity), debug_info
        else:
            # Fall back to single background
            single_result, single_debug = self._handle_single_background([peak1], all_pixels)
            
            # Add validation failure info
            single_debug.update({
                'dual_background_attempted': True,
                'validation_checks': validation_checks,
                'passed_validations': passed_validations,
                'required_validations': required_validations,
                'fallback_reason': 'validation_failed'
            })
            
            return single_result, single_debug
    
    def _handle_single_background(self, peaks: list, all_pixels: np.ndarray) -> tuple:
        """Handle single background detection."""
        if len(peaks) == 0:
            threshold = float(np.min(all_pixels))
            method = 'fallback_minimum'
        else:
            threshold = peaks[0][2]
            method = 'single_peak'
        
        return (threshold, threshold), self._create_debug_info(
            'single',
            threshold=threshold,
            selected_intensity=threshold,
            selection_method=method,
            dual_background_detected=False,
            num_peaks_found=len(peaks)
        )
    
    def _apply_clipping_if_needed(self, pixel_array: np.ndarray, key: str, data_dict: dict) -> np.ndarray:
        """Apply dataset-specific clipping if conditions met."""
        if not hasattr(self, '_last_debug_info') or not self.debug_level:
            return pixel_array
        
        clip_configs = {
            'scotheart': ('dual_background_detected', 'selected_intensity'),
            'acdc': ('mr_dual_peaks_detected', 'noise_threshold'),
            'cap': ('mr_dual_peaks_detected', 'noise_threshold'),
            'mmwhs': ('plateau_detected', 'clipping_threshold')
        }
        
        if self.dataset in clip_configs:
            detect_key, thresh_key = clip_configs[self.dataset]
            if self._last_debug_info.get(detect_key, False):
                threshold = self._last_debug_info.get(thresh_key)
                if threshold is not None:
                    # Apply clipping
                    pixel_array[pixel_array <= threshold] = threshold
                    
                    # Update MetaTensor
                    if hasattr(data_dict[key], 'affine'):
                        data_dict[key] = MetaTensor(
                            pixel_array, 
                            affine=data_dict[key].affine,
                            applied_operations=data_dict[key].applied_operations
                        )
                    else:
                        data_dict[key] = pixel_array
                    
                    # Mark clipping as applied
                    self._last_debug_info['clipping_applied'] = True
                    self._last_debug_info['clip_threshold'] = threshold
        
        return pixel_array
    
    def _detect_background_intensity(self, pixel_array: np.ndarray) -> tuple:
        """
        Detect background intensity using histogram peak analysis.
        
        For non-SCOTHEART datasets: Uses histogram mode (highest peak)
        For SCOTHEART: Detects dual-background peaks and validates for clipping
        
        Args:
            pixel_array: Flattened pixel intensity array
            
        Returns:
            Tuple of (background_intensity, background_intensity) for API compatibility
        """
        # Remove any NaN or infinite values
        valid_pixels = pixel_array[np.isfinite(pixel_array)]
        
        if len(valid_pixels) == 0:
            raise ValueError("No valid pixel values found in the image")
        
        # Create histogram
        hist, bin_edges = np.histogram(valid_pixels, bins=self.num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # MR dual-peak detection for background/noise removal
        if self.dataset in ['acdc', 'cap']:
            try:
                mr_peaks = self._find_mr_background_noise_peaks(hist, bin_centers, valid_pixels)
                background_result, debug_info = self._validate_mr_dual_peaks(mr_peaks, valid_pixels)
                
                if self.debug_level:
                    self._last_debug_info = debug_info
                return background_result
                
            except Exception as e:
                # Fallback to percentile method for MR
                return self._handle_mr_fallback(valid_pixels)
        
        # Unified CT processing for both SCOTHEART and MMWHS
        elif self.dataset in ['scotheart', 'mmwhs']:
            try:
                ct_peaks, detection_type = self._find_ct_background_peaks(hist, bin_centers, valid_pixels)
                background_result, debug_info = self._validate_ct_peaks(
                    ct_peaks, detection_type, bin_centers, valid_pixels)
                
                if self.debug_level:
                    self._last_debug_info = debug_info
                return background_result
                
            except Exception as e:
                # Fallback to simple mode detection
                max_bin_idx = np.argmax(hist)
                background_intensity = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
                return (background_intensity, background_intensity)
        
        # Standard single-peak detection for other non-CT datasets
        else:
            max_bin_idx = np.argmax(hist)
            background_intensity = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            
            debug_info = self._create_debug_info(
                'single',
                threshold=background_intensity,
                selected_intensity=background_intensity,
                selection_method='histogram_mode',
                dual_background_detected=False
            )
            
            if self.debug_level:
                self._last_debug_info = debug_info
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
                
                # Copy for modification
                pixel_array = pixel_array.copy()
                
                # Flatten for analysis (exclude channel dimension if present)
                if pixel_array.ndim == 4:  # (C, H, W, D)
                    flat_pixels = pixel_array.flatten()
                elif pixel_array.ndim == 3:  # (H, W, D)
                    flat_pixels = pixel_array.flatten()
                else:
                    flat_pixels = pixel_array.flatten()
                
                # Detect background intensity range
                background_range = self._detect_background_intensity(flat_pixels)
                
                # Apply dataset-specific clipping if needed
                pixel_array = self._apply_clipping_if_needed(pixel_array, key, data_dict)
                
                # Recalculate flattened pixels if clipping was applied
                if hasattr(self, '_last_debug_info') and self._last_debug_info.get('clipping_applied', False):
                    flat_pixels = pixel_array.flatten()
                
                # Calculate background percentile (now on potentially clipped data)
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
                
                # Add debug information to metadata if debug mode is enabled
                if self.debug_level >= 2 and hasattr(self, '_last_debug_info'):
                    if hasattr(data_dict[key], 'meta'):
                        if data_dict[key].meta is not None:
                            data_dict[key].meta['intensity_scaling_debug'] = self._last_debug_info
                        else:
                            data_dict[key].meta = {'intensity_scaling_debug': self._last_debug_info}
                
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in DynamicIntensityRangeScalesd: {str(e)}")
        
        return data_dict
