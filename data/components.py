from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized
from monai.transforms.utils import distance_transform_edt

import nibabel as nib


__all__ = ["MaskCTd", "DFConvertd", "Adjustd", "FlexResized", "Probd"]


class MaskCTd(MapTransform):
    """
    this transform mask the CTA images near the basal and apex plane, i.e., the first and last slices.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        image = data["ct_image"]
        image = image.get_array()

        # mask the CTA images near the basal and apex plane
        mask = np.zeros_like(image).astype(bool)
        mask[..., 20:-30] = True
        image[~mask] = image.min()

        data["ct_image"] = MetaTensor(image, affine=data["ct_image"].affine)

        return data


# class RandomShiftd(MapTransform):
#     """
#     create slice shift along the last axis mimic the respiratory motion.
#     """
#     def __init__(self, keys: KeysCollection, pixdim: list, allow_missing_keys: bool = False) -> None:
#         super().__init__(keys, allow_missing_keys)
#         self.pixdim = list(map(int, pixdim))

#     def __call__(self, data):
#         image = data["ct_image"]
#         label = data["ct_label"]
#         image = image.get_array()
#         label = label.get_array()

#         # randomly selet three batch from range(image.shape[-1] // self.pixdim[-1])
#         batches = np.random.choice(range(int(image.shape[-1] // self.pixdim[-1]) - 1), 3, replace=False)
#         for batch in batches:
#             idx_start, idx_end = batch * self.pixdim[-1], (batch + 1) * self.pixdim[-1]
#             # randomly shifting the image and label by a number of slices a batch (defined by pixdim) along the last axis
#             shift = np.random.randint(-3, 3)
#             image[..., idx_start: idx_end] = np.roll(image[..., idx_start: idx_end], shift, axis=0)
#             image[..., idx_start: idx_end] = np.roll(image[..., idx_start: idx_end], shift, axis=1)
#             label[..., idx_start: idx_end] = np.roll(label[..., idx_start: idx_end], shift, axis=0)
#             label[..., idx_start: idx_end] = np.roll(label[..., idx_start: idx_end], shift, axis=1)

#         data["ct_image"] = MetaTensor(image, meta=data["ct_image"].meta, applied_operations=["RandomShiftd"])
#         data["ct_label"] = MetaTensor(label, meta=data["ct_label"].meta, applied_operations=["RandomShiftd"])

#         return data


class Adjustd(MapTransform):
    """
    process the input data to be compatible with the rest transforms.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            try:
                pixel_array = data[key].get_array().copy()

                if "label" in key:
                    # combine label index 2 and 4 as ventricular myocardium
                    pixel_array[pixel_array == 4] = 2

                if len(pixel_array.shape) == 4:
                    affine = data[key].affine.clone()
                    # update the affine matrix
                    m = torch.eye(4)
                    m[:3, 0] = affine[1, :3]
                    m[:3, 1] = affine[2, :3]
                    m[:3, 2] = affine[3, :3]
                    m[:3, -1] = affine[:3, -1]
                    data[key] = MetaTensor(pixel_array, affine=m)

                elif len(pixel_array.shape) == 3:
                    # insert a new axis for the channel (first axis)
                    pixel_array = pixel_array[None]
                    data[key] = MetaTensor(pixel_array, affine=data[key].affine)

            except KeyError:
                print(f"Error: {key} is not in the data dictionary.")

        return data

class FlexResized(MapTransform):
    """
    resize 3D image and label by slices rather than volume.
    """
    def __init__(self, keys: KeysCollection, size: int, modal: str, down_sampled: bool, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = int(size)
        self.modal = modal
        self.down_sampled = down_sampled

    def __call__(self, data):
        if self.modal == "ct" or self.down_sampled:
            if self.down_sampled:
                keys = self.keys[0]
                mode = "nearest"
            else:
                keys = self.keys
                mode = ("bilinear", "nearest")

            data = Resized(keys, self.size, size_mode="longest", mode=mode)(data)

            return data
        
        else:
            data_shape = data[self.keys[0]].get_array().shape[1:-1]
            rescale_ratio = self.size / max(data_shape)
            new_shape = [np.ceil(s * rescale_ratio).astype(np.uint8) for s in data_shape] + [-1]

            data = Resized(self.keys, new_shape, size_mode="all", mode=("bilinear", "nearest"))(data)

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

        # combine left and right myocardium (index 2 and 4) to have four classes (background: 0, left ventricle: 1, myocardium: 2, right ventricle: 3)
        lv = label == 1
        myo = label == 2
        rv = label == 3

        df = []
        for c in [lv, myo, rv]:
            df_class = distance_transform_edt(c.to(torch.float32)) +\
                distance_transform_edt(1 - c.to(torch.float32))
            df.append(df_class[:, None])

        df = MetaTensor(torch.cat(df, dim=1), 
                        meta=data[self.key].meta, 
                        applied_operations=["DFConvertd"])
        data[f"{self.modal}_df"] = df

        # remove the original label
        data.pop(self.key)

        return data


