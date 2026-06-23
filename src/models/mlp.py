"""PyTorch RayMLP baseline for ray-wise STEC regression."""

from __future__ import annotations


def build_ray_mlp(input_dim: int, dropout: float = 0.1):
    """Create the recommended RayMLP architecture.

    PyTorch is imported lazily so dataset construction can run without torch.
    """

    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

