import math
import torch
from typing import List
from logging import getLogger

logger = getLogger()

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0, std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def apply_masks(x, masks: List[torch.Tensor]):
    """Apply masks to input tensor x.
    Args:
        x (torch.Tensor): Input tensor of shape [B, N, D].
        masks (List[torch.Tensor]): List of tensors containing indices of patches in [N] to keep.
    Returns:
        torch.Tensor: Tensor of shape [len(masks) * B, N, D].  i.e. Context Blocks
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, N, D)
        all_x += [torch.gather(x, dim=1, index=mask_keep)]  # (B, V, D)
    return torch.cat(all_x, dim=0)  # (len(masks) * B, N, D)

def repeat_interleave_batch(x, B, repeat):
    """Repeat each batch B times and then repeat the entire tensor repeat times.
    Args:
        x (torch.Tensor): Input tensor of shape (N * B, min_keep, D), note: n means n_preds or n_encs
        B (int): Number of times to repeat each batch.
        repeat (int): Number of times to repeat the entire tensor. we denote repeat as R.
    Returns:
        torch.Tensor: Tensor of shape [N * B * repeat, D].
    """
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)  # (R * B, min_keep, D)
        for i in range(N)  
    ], dim=0)  # (N * R * B, min_keep, D)
    return x