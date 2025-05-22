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

__all__ = ["collate_4D_batch"]


def collate_4D_batch(data: List[Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for 4D data.
    """
    batch = {}
    for key in data[0].keys():
        if isinstance(data[0][key], torch.Tensor):
            if "mr" not in key or "df" in key:
                batch[key] = torch.concat([d[key] for d in data], dim=0)
                if batch[key].dim() == 4:
                    batch[key] = batch[key].unsqueeze(1)
            else:
                batch[key] = torch.concat([d[key] for d in data], dim=1)
                batch[key] = batch[key].flatten(0, 1).unsqueeze(1)
        else:
            batch[key] = np.stack([d[key] for d in data], axis=0)
    return batch

