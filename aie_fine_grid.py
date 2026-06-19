"""Differentiable transformations between projector and fine simulation grids."""

import torch
import torch.nn.functional as F


def _validate_2d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")


def initialize_projector_mask(
    target: torch.Tensor,
    projector_shape: tuple[int, int] = (300, 300),
    refinement: int = 2,
) -> torch.Tensor:
    """Initialize each projector pixel from its fine-grid block average."""
    _validate_2d(target, "target")
    if refinement < 1:
        raise ValueError("refinement must be at least 1")
    expected = tuple(size * refinement for size in projector_shape)
    if tuple(target.shape) != expected:
        raise ValueError(f"expected target shape {expected}, got {tuple(target.shape)}")
    return F.avg_pool2d(target[None, None], refinement, refinement)[0, 0]


def expand_projector_mask(mask: torch.Tensor, refinement: int = 2) -> torch.Tensor:
    """Expand each projector pixel to a constant fine-grid block."""
    _validate_2d(mask, "projector mask")
    if refinement < 1:
        raise ValueError("refinement must be at least 1")
    return mask.repeat_interleave(refinement, 0).repeat_interleave(refinement, 1)
