"""Lightweight unit tests for optical-only MPC cold-start initialization."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from aie_mpc_initialization import (
    PhysicsAwareInitializationError,
    build_physics_aware_initial_mask,
    solve_checkpoint_nominal_intensity,
)


def _o2_only_parameters() -> SimpleNamespace:
    return SimpleNamespace(
        intensity_mw_cm2=10.0,
        o2_inhibition_mj_cm2=2.0,
        tempo_inhibition_mj_cm2=0.0,
        chain_growth_noise_std=0.0,
        b_slope=0.0,
        b_intercept=1.0,
        projector_refinement=1,
    )


class _IdentityOpticalModel:
    """Minimal differentiable optics fixture; it has no chemical transition."""

    def __init__(self) -> None:
        self.params = _o2_only_parameters()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    @staticmethod
    def control_shape_for(state_shape: torch.Size) -> tuple[int, int]:
        return int(state_shape[0]), int(state_shape[1])

    def prepare_control(
        self, projector_mask: torch.Tensor, state_shape: torch.Size
    ) -> SimpleNamespace:
        if tuple(projector_mask.shape) != tuple(state_shape):
            raise ValueError("identity fixture shape mismatch")
        return SimpleNamespace(
            local_intensity=projector_mask * self.params.intensity_mw_cm2
        )


class NominalIntensityTests(unittest.TestCase):
    def test_bisection_recovers_known_intensity(self) -> None:
        params = _o2_only_parameters()
        checkpoint_doc = 1.0 - math.exp(-1.0)
        solution = solve_checkpoint_nominal_intensity(
            params, checkpoint_time_s=2.0, checkpoint_doc=checkpoint_doc
        )
        self.assertAlmostEqual(solution.intensity_mw_cm2, 2.0, places=7)
        self.assertAlmostEqual(solution.normalized_level, 0.2, places=8)
        self.assertAlmostEqual(solution.predicted_doc, checkpoint_doc, places=8)

    def test_infeasible_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            PhysicsAwareInitializationError, "checkpoint is infeasible"
        ):
            solve_checkpoint_nominal_intensity(
                _o2_only_parameters(), checkpoint_time_s=2.0, checkpoint_doc=0.99
            )


class OpticalInitializationTests(unittest.TestCase):
    def test_optical_inverse_is_bounded_and_repeated_mask_ready(self) -> None:
        target = torch.zeros((5, 5), dtype=torch.float32)
        target[2, 2] = 1.0
        checkpoint_doc = 1.0 - math.exp(-1.0)
        result = build_physics_aware_initial_mask(
            model=_IdentityOpticalModel(),
            target=target,
            target_threshold=0.5,
            checkpoint_time_s=2.0,
            checkpoint_doc=checkpoint_doc,
            background_level=0.02,
            num_iterations=3,
            learning_rate=0.1,
            outside_weight=0.5,
        )
        self.assertEqual(tuple(result.projector_mask.shape), (5, 5))
        self.assertEqual(tuple(result.local_normalized_intensity.shape), (5, 5))
        self.assertTrue(bool((result.projector_mask >= 0.0).all()))
        self.assertTrue(bool((result.projector_mask <= 1.0).all()))
        self.assertTrue(bool(torch.isfinite(result.local_normalized_intensity).all()))
        self.assertAlmostEqual(float(result.projector_mask[2, 2]), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
