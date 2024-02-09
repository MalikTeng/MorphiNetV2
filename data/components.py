from monai.config import KeysCollection
import numpy as np

from monai.data import MetaTensor
from monai.transforms import MapTransform

import edt

__all__ = ["MaskCTAd", "SDFConvertd"]


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


class SDFConvertd(MapTransform):
    """
    this transform convert the ground truth segmentation to signed distance fields.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            if "label" in key:
                label = data[key]
                label = label.get_array().copy()
                label = label.squeeze()

                # combine left and right myocardium (index 2 and 4) to have four classes (background, left ventricle, right ventricle, myocardium)
                label[label == 4] = 2

                sdf = []
                unq_class = np.unique(label)[1:]    # exclude background
                for c in unq_class:
                    sdf_class = edt.sdf(label == c, 
                                anisotropy=(1, 1, 1), black_border=False, order='C', parallel=1)
                    sdf.append(sdf_class.astype(np.float32))

                sdf = MetaTensor(np.stack(sdf), meta=data[key].meta, applied_operations=["SDFConvertd"])
                data[f"{key[:2]}_sdf"] = sdf

        return data
