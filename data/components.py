from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized
from monai.transforms.utils import distance_transform_edt

import nibabel as nib


__all__ = ["Maskd", "DFConvertd", "Adjustd", "FlexResized", "Probd", "DynUNetPaddingd"]


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
                print(f"Error: {key} is not in the data dictionary.")

        return data


class FlexResized(MapTransform):
    """
    resize 3D image and label regarding designated axis.
    """
    def __init__(self, keys: KeysCollection, size: tuple, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = np.array([i for i in map(int, size)])
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        if len(self.keys) == 2:
            keys = []
            for key, tag in zip(self.keys, ["pred", "label"]):
                if (tag in key) and (key in data):
                    keys.append(key)
                elif (key not in data) and self.allow_missing_keys:
                    keys.append(None)
                else:
                    raise KeyError(f"key: {key} should contain {tag}.")
            tag_pred = keys[0]
            tag_label = keys[1]
        elif len(self.keys) == 1:
            assert "label" in self.keys[0], f"key: {self.keys[0]} should contain label."
            tag_pred = None
            tag_label = self.keys[0]

        data_shape = data[tag_label].get_array().shape[1:]
        self.size = np.where(self.size == -1, data_shape, self.size)
        # Calculate rescale ratio based on the second dimension (index 1)
        rescale_ratio = self.size[1] / data_shape[1]
        new_shape = [np.ceil(d * rescale_ratio).astype(np.uint8) for d in data_shape]

        assert new_shape[1] == self.size[1], f"new shape: {new_shape}, crop window size: {self.size}"

        if tag_pred is not None:
            data = Resized([tag_pred, tag_label], new_shape, size_mode="all", mode=("bilinear", "nearest"))(data)
        else:
            data = Resized([tag_label], new_shape, size_mode="all", mode="nearest")(data)

        return data


class Probd(MapTransform):
    """
    read the input data to see its shape and dimension.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            pixel_array = data[key].get_array().copy()
            print(f"shape of {key}: {pixel_array.shape}")
            print(f"pixdim of {key}: {data[key].pixdim}")

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
        for c in [foreground, lv, rv, myo]:  # Compute DF for foreground, lv, rv, myo
            df_class = distance_transform_edt(c) + distance_transform_edt(~c)
            df.append(df_class[:, None])

        df = MetaTensor(torch.cat(df, dim=1), affine=data[self.key].affine)

        data[f"{self.modal}_df"] = df

        # # remove the original label
        # data.pop(self.key)

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
            # For 2D DynUNet, typically use 4 levels: stride factor = 2*2*2 = 8
            effective_strides = self.strides[1:4]  # Skip first stride, take next 3
        elif self.spatial_dims == 3:
            # For 3D DynUNet, typically use 4 levels: stride factor = 2*2*2 = 8  
            effective_strides = self.strides[1:4]  # Skip first stride, take next 3
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
                    print(f"Skipping {key}: unsupported shape {original_shape}. Expected 4D (C, H, W, D).")
                    
            except Exception as e:
                if not self.allow_missing_keys:
                    raise KeyError(f"Error processing key '{key}' in DynUNetPaddingd: {str(e)}")
                print(f"Warning: Skipping key '{key}' due to error: {str(e)}")
        
        return data_dict


