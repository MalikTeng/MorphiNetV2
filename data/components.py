from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized
from monai.transforms.utils import distance_transform_edt


__all__ = ["MaskCTAd", "DFConvertd", "Adjustd", "Probd"]


class MaskCTAd(MapTransform):
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

        data["ct_image"] = MetaTensor(image, meta=data["ct_image"].meta, applied_operations=["MaskCTAd"])

        return data


class Adjustd(MapTransform):
    """
    process the input data to be compatible with the rest transforms.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            pixel_array = data[key].get_array().copy()

            if len(pixel_array.shape) == 4:
                # update the affine matrix
                affine = data[key].affine.clone()
                m = torch.eye(4)
                m[:3, 0] = affine[1, :3]
                m[:3, 1] = affine[2, :3]
                m[:3, 2] = affine[3, :3]
                m[:3, -1] = affine[:3, -1]
                data[key] = MetaTensor(pixel_array, affine=m)

            elif len(pixel_array.shape) == 3:
                # insert a new axis for the channel (first axis)
                pixel_array = pixel_array[None]
                data[key] = MetaTensor(pixel_array, meta=data[key].meta)

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
        label = label.get_array().copy()

        # combine left and right myocardium (index 2 and 4) to have four classes (background: 0, left ventricle: 1, myocardium: 2, right ventricle: 3)
        label[label == 4] = 2
        lv = label == 1
        # myo = label == 2
        myo = label > 0
        rv = label == 3

        df = []
        for c in [lv, myo, rv]:
            df_class = distance_transform_edt(c) +\
                distance_transform_edt(1 - c)
            df.append(df_class[:, None])

        df = MetaTensor(np.concatenate(df, axis=1), 
                        meta=data[self.key].meta, applied_operations=["DFConvertd"])
        data[f"{self.modal}_df"] = df

        # remove the original label
        data.pop(self.key)

        return data
