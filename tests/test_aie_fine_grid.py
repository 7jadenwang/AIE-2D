import pytest
import torch

from aie_fine_grid import expand_projector_mask, initialize_projector_mask


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
