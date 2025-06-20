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
                batch[key] = torch.stack([d[key] for d in data], dim=0)
        else:
            batch[key] = [d[key] for d in data]
    
    return batch