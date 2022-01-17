import torch


def get_mask(seq):
    length = seq.shape[-1]
    mask = torch.triu(torch.ones((1, length, length), device=seq.device), diagonal=1)
    mask = mask.bool()
    return mask
