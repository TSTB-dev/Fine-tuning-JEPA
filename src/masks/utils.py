import torch
from typing import List 

def apply_masks(x, masks: List[torch.Tensor]):
    """Apply masks to input tensor x.
    Args:
        x (torch.Tensor): Input tensor of shape [B, N, D].
        masks (List[torch.Tensor]): List of tensors containing indices of patches in [N] to keep.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, N, D)
        all_x += [torch.gather(x, dim=1, index=mask_keep)]  # (B, V, D)
    return torch.cat(all_x, dim=0) # (len(masks) * B, N, D)
        