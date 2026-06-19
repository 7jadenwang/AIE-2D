from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aie_fine_grid import expand_projector_mask, initialize_projector_mask


def test_fine_grid_script_preserves_projector_grid_and_expands_before_physics():
    source = Path("AIE_re2.1.py").read_text(encoding="utf-8")
    assert "PROJECTOR_SHAPE = (300, 300)" in source
    assert "REFINEMENT = 2" in source
    assert "PROJECTOR_PITCH = 7.395e-6" in source
    assert "dx = dy = PROJECTOR_PITCH / REFINEMENT" in source
    assert "initialize_projector_mask(target_mask, PROJECTOR_SHAPE, REFINEMENT)" in source
    assert "expand_projector_mask(opt_mask, REFINEMENT)" in source
    assert 'TARGET_PATH = os.environ.get("AIE_TARGET_PATH", "./Lshape600.png")' in source
    assert "img = Image.open(TARGET_PATH)" in source


def test_fine_grid_optimization_smoke_preserves_shapes_and_gradients():
    target = torch.zeros(600, 600)
    target[150:450, 200:400] = 1.0
    projector = torch.nn.Parameter(initialize_projector_mask(target))

    expanded = expand_projector_mask(projector)
    diagnostic = F.avg_pool2d(
        expanded.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    )[0, 0]
    F.mse_loss(diagnostic, target).backward()
    projector_export = projector.detach()

    assert projector_export.shape == (300, 300)
    assert expanded.shape == (600, 600)
    assert diagnostic.shape == (600, 600)
    assert projector.grad is not None
    assert projector.grad.shape == (300, 300)
    assert torch.isfinite(projector.grad).all()


def test_initialize_projector_mask_averages_refinement_block():
    target = torch.tensor([[0.0, 2.0], [4.0, 6.0]])

    mask = initialize_projector_mask(target, projector_shape=(1, 1), refinement=2)

    torch.testing.assert_close(mask, torch.tensor([[3.0]]))


def test_expand_projector_mask_repeats_blocks_and_preserves_gradient():
    mask = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    fine = expand_projector_mask(mask, refinement=2)
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ]
    )
    torch.testing.assert_close(fine, expected)

    fine.sum().backward()

    torch.testing.assert_close(mask.grad, torch.full_like(mask, 4.0))


def test_production_shape_round_trip():
    target = torch.zeros(600, 600)

    mask = initialize_projector_mask(target)
    fine = expand_projector_mask(mask)

    assert mask.shape == (300, 300)
    assert fine.shape == target.shape


def test_initialize_projector_mask_rejects_wrong_target_shape():
    target = torch.zeros(599, 600)

    with pytest.raises(ValueError, match="expected target shape"):
        initialize_projector_mask(target, projector_shape=(300, 300), refinement=2)


def test_initialize_projector_mask_rejects_non_2d_target():
    with pytest.raises(ValueError, match="must be 2D"):
        initialize_projector_mask(torch.zeros(1, 2, 2), projector_shape=(1, 1))


def test_expand_projector_mask_rejects_non_2d_mask():
    with pytest.raises(ValueError, match="must be 2D"):
        expand_projector_mask(torch.zeros(1, 2, 2))


def test_initialize_projector_mask_rejects_non_positive_refinement():
    with pytest.raises(ValueError, match="refinement must be at least 1"):
        initialize_projector_mask(
            torch.zeros(2, 2), projector_shape=(1, 1), refinement=0
        )


def test_expand_projector_mask_rejects_non_positive_refinement():
    with pytest.raises(ValueError, match="refinement must be at least 1"):
        expand_projector_mask(torch.zeros(1, 1), refinement=0)
