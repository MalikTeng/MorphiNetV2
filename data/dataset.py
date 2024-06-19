import os
import sys
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Union, SupportsIndex

import numpy as np

from monai.config.type_definitions import PathLike
from monai.data import (
    CacheDataset,
    partition_dataset,
    select_cross_validation_folds,
)
from monai.data.utils import list_data_collate
from monai.transforms import LoadImaged, Randomizable, MapTransform, Transform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

from torch.utils.data import Dataset
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

from pytorch3d.structures import Meshes, Pointclouds

import torch

__all__ = ["Dataset", "collate_4D_batch"]


def collate_4D_batch(data: List[Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for 4D data.
    """
    batch = {}
    for key in data[0].keys():
        if isinstance(data[0][key], torch.Tensor):
            batch[key] = torch.concat([d[key] for d in data], dim=0)
            if batch[key].dim() == 4:
                batch[key] = batch[key].unsqueeze(1)
        else:
            batch[key] = np.stack([d[key] for d in data], axis=0)
    return batch


def load_multimodal_datalist(
        image_paths: list,
        label_paths: list,
        keys: tuple,
        slice_info_paths: list,
) -> list:
    """
    :params image_paths: sorted list of image file paths
    :params label_paths: sorted list of label file paths
    :params keys: IDs of image and label in a sequence, e.g., ('image', 'label')

    :returns sorted list of dictionary -- {'label': label_path, 'image': image_path}
    """
    assert len(image_paths) == len(label_paths)
    if slice_info_paths is not None:
        expected_data = [
            {
                keys[0]: image_path,
                keys[1]: label_path,
                keys[2]: slice_info_path,
            } for image_path, label_path, slice_info_path in zip(image_paths, label_paths, slice_info_paths)
        ]
    else:
        expected_data = [
            {
                keys[0]: image_path,
                keys[1]: label_path,
            } for image_path, label_path in zip(image_paths, label_paths)
        ]

    return expected_data


class Dataset(Randomizable, CacheDataset):
    """
        :params 
            data: list of dictionary -- {'label': label_path, 'image': image_path}
            transform: composed MONAI transforms to execute operations on input data.
            seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0. note to set same seed for `training` and `validation` sections.
            cache_num: number of items to be cached. Default is `sys.maxsize`. will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all). will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads to use. if 0 a single thread will be used. Default is 0.
    """
    def __init__(
            self,
            data: list,
            transform: Union[Sequence[Callable], Callable] = (),
            seed: int = 0,
            cache_num: int = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = 0,
            ):
        self.set_random_state(seed=seed)
        self.indices: np.ndarray = np.array([])

        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers,
            )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.
        """
        return self.indices

