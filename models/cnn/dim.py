import torch.nn as nn

def _get(dim: int, layers: dict):
    if dim not in layers:
        raise ValueError(f"Only 2 or 3 Dimension is possible: dim: {dim}")
    return layers[dim]

def get_conv(dim: int):
    return _get(dim, {2: nn.Conv2d, 3: nn.Conv3d})

def get_norm(dim: int):
    return _get(dim, {2: nn.BatchNorm2d, 3: nn.BatchNorm3d})
