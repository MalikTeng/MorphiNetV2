"""
Dataset utilities for MorphiNet modular architecture.
"""

import torch
import numpy as np
from typing import List, Dict, Union


def collate_4D_batch(data: List[Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for 4D data.
    
    Args:
        data: List of data dictionaries
        
    Returns:
        Batched data dictionary
    """
    batch = {}
    for key in data[0].keys():
        if isinstance(data[0][key], torch.Tensor):
            if "mr" not in key or "df" in key:
                batch[key] = torch.concat([d[key] for d in data], dim=0)
                if batch[key].dim() == 4:
                    batch[key] = batch[key].unsqueeze(1)
            else:
                # For MR data (not distance fields), concatenate along slice dimension
                # and flatten to create 2D slices for 2D UNet processing
                batch[key] = torch.concat([d[key] for d in data], dim=1)
                batch[key] = batch[key].flatten(0, 1).unsqueeze(1)
        else:
            batch[key] = [d[key] for d in data]
    
    return batch