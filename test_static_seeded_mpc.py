"""Lightweight benchmark checks, without static/MPC production optimization."""
from types import SimpleNamespace

import torch

from aie_model import AIEModel
from static_seeded_mpc import accept_candidate, objective, optimize, verify_static_seed


def toy_forward(controls):
    doc = controls.mean(dim=0)
    return SimpleNamespace(doc=doc, tensors=lambda: (doc,))


def test_objective_is_full_image_squared_frobenius():
    state = SimpleNamespace(doc=torch.tensor([[0.8, 0.4]]))
    torch.testing.assert_close(objective(state, torch.tensor([[0.9, 0.0]])), torch.tensor(0.17))


def test_incumbent_protected_against_worse_equal_and_invalid_candidates():
    desired = torch.full((2, 2), 0.9)
    seed = torch.full((2, 2, 2), 0.8)
    for candidate in (seed.clone(), torch.zeros_like(seed), torch.full_like(seed, float("nan"))):
        accepted, old, new, changed = accept_candidate(toy_forward, seed, candidate, desired)
        assert not changed
        assert new >= old
        torch.testing.assert_close(accepted, seed, rtol=0, atol=0)
    accepted, old, new, changed = accept_candidate(toy_forward, seed, torch.full_like(seed, 0.85), desired)
    assert changed and new < old


def test_best_iterate_retention_and_bounds_on_tiny_algebraic_problem():
    seed = torch.full((2, 2, 2), 0.8)
    desired = torch.full((2, 2), 0.9)
    best, history, _ = optimize(toy_forward, seed, desired, iterations=4, learning_rate=0.3)
    assert bool(((best >= 0) & (best <= 1)).all())
    assert history[-1]["best_J"] <= history[0]["best_J"]
    assert all(b["best_J"] <= a["best_J"] for a, b in zip(history, history[1:]))
    assert float(objective(toy_forward(best), desired)) == history[-1]["best_J"]


def test_real_physics_static_seed_and_shrinking_replay():
    model = AIEModel(dtype=torch.float64)
    size = max(model.scattering_kernel_1d.numel(), model.o2_kernel_1d.numel(),
               model.tempo_kernel_1d.numel()) // 2 + 3
    mask = torch.full((size, size), 0.8, dtype=torch.float64)
    initial = model.initialize_state(mask.shape)
    # Two tiny intervals test the same segmentation invariant as eight intervals.
    sequence = verify_static_seed(model, initial, mask, 2, 1, mask * 0.9)
    with torch.no_grad():
        full = model.rollout(initial, sequence, 1)
        actual = model.advance(initial, sequence[0], 1).detach()
        remaining = model.rollout(actual, sequence[1:], 1)
    for a, b in zip(full.tensors(), remaining.tensors()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
