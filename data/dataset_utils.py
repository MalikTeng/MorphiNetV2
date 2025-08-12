"""
Dataset utilities for MorphiNet modular architecture.
"""

import torch
import numpy as np
from typing import List, Dict, Union
from einops import rearrange


def collate_4D_batch(data: List[Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for 4D data.
    
    Args:
        data: List of data dictionaries
        
    Returns:
        Batched data dictionary
    """
    # Collate function processes different data shapes:
    # CT: image/label [B, H, W, D] -> [B, C, H, W, D], df [B, C, H, W, D] (unchanged)
    # MR: image/label [B, H, W, D] -> [B*D, C, H, W], df [B, C, H, W, D] (unchanged)
    
    batch = {}
    if not isinstance(data[0], list):   # handling the returned list of tensors by RandCropbyPosNegLabeld
        for key in data[0].keys():
            if isinstance(data[0][key], torch.Tensor):
                if "mr" not in key or "df" in key:
                    # Handle CT data and MR/CT distance fields normally
                    batch[key] = torch.concat([d[key] for d in data], dim=0)
                    batch[key] = batch[key].unsqueeze(0 if "df" in key else 1)
                else:
                    # For MR data, handle both ACDC and CAP formats:
                    # [B, H, W, D] -> [B*D, C, H, W]  
                    batch[key] = torch.concat([d[key] for d in data], dim=0)
                    batch[key] = rearrange(batch[key], '(b c) h w d -> (b d) c h w', c=1)
            else:
                batch[key] = [d[key] for d in data]
    
    else:
        for key in data[0][0].keys():
            if isinstance(data[0][0][key], torch.Tensor):
                if "mr" not in key or "df" in key:
                    # Handle CT data and MR/CT distance fields normally
                    batch[key] = torch.concat([d[key] for b in data for d in b], dim=0)
                    batch[key] = batch[key].unsqueeze(0 if "df" in key else 1)
                else:
                    # For MR data, handle both ACDC and CAP formats:
                    # [B, H, W, D] -> [B*D, C, H, W]
                    batch[key] = torch.concat([d[key] for b in data for d in b], dim=0)
                    batch[key] = rearrange(batch[key], '(b c) h w d -> (b d) c h w', c=1)
            else:
                batch[key] = [d[0][key] for d in data]
    
    return batch