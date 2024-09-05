from monai.config import KeysCollection
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import MapTransform, Resized
from monai.transforms.utils import distance_transform_edt

import nibabel as nib


__all__ = ["Maskd", "DFConvertd", "Adjustd", "FlexResized", "Probd"]


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
        rescale_ratio = max([s / d for s, d in zip(self.size, data_shape)])
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

        # four classes (background: 0, left ventricle: 1, myocardium: 2, right ventricle: 3)
        foreground = label > 0
        lv = label == 1
        myo = label == 2
        rv = label == 3

        df = []
        for c in [foreground, lv, rv, myo]:
            df_class = distance_transform_edt(c) + distance_transform_edt(~c)
            df.append(df_class[:, None])

        df = MetaTensor(torch.cat(df, dim=1), affine=data[self.key].affine)

        data[f"{self.modal}_df"] = df

        # remove the original label
        data.pop(self.key)

        return data


