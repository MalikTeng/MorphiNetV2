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
                # Handle CT data and distance fields normally
                batch[key] = torch.concat([d[key] for d in data], dim=0)
                # Ensure CT data has proper dimensions: add batch dim if needed, then channel dim
                if batch[key].dim() == 3:  # [H*B, W, D] -> [B, 1, H, W, D]
                    # Reshape to separate batch and spatial dimensions
                    original_shape = batch[key].shape
                    batch_size = len(data)
                    spatial_dims = (original_shape[0] // batch_size, original_shape[1], original_shape[2])
                    batch[key] = batch[key].view(batch_size, *spatial_dims).unsqueeze(1)
                elif batch[key].dim() == 4:  # [B, H, W, D] -> [B, 1, H, W, D]
                    batch[key] = batch[key].unsqueeze(1)
            else:
                # For MR data, the input shape for each sample is [N, H, W, D].
                # This code converts it to [N * D, C, H, W] for 2D UNet processing.
                all_slices = [
                    d[key].permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1) for d in data
                ]
                batch[key] = torch.concat(all_slices, dim=0)
        else:
            batch[key] = [d[key] for d in data]
    
    return batch