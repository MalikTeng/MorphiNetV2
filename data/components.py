from monai.config import KeysCollection
import numpy as np

from monai.data import MetaTensor
from monai.transforms import MapTransform
from monai.transforms.utils import distance_transform_edt


__all__ = ["MaskCTAd", "DFConvertd"]


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
        label = label.squeeze()

        # combine left and right myocardium (index 2 and 4) to have four classes (background: 0, left ventricle: 1, right ventricle: 2, myocardium: 3)
        label[label == 4] = 2
        lv = label == 1
        myo = label > 0
        rv = label == 3

        df = []
        for c in [lv, myo, rv]:
            df_class = distance_transform_edt(c[None])[0] +\
                distance_transform_edt(1 - c[None])[0]
            df.append(df_class)

        df = MetaTensor(np.stack(df), meta=data[self.key].meta, applied_operations=["DFConvertd"])
        data[f"{self.modal}_df"] = df

        # remove the original label
        data.pop(self.key)

        return data
