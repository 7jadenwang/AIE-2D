"""Differentiable nonlinear MPC for the AIE physical model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from aie_fine_grid import initialize_projector_mask
from aie_model import AIEModel, AIEState
from doc_reference import DoCReferenceCurve
from tracking_config import TrackingSpecification


class DifferentiableMPC:
    """Optimize a receding sequence of DLP masks through ``AIEModel``.

    The optimizer acts on unconstrained logits and maps them to valid
    normalized grayscale masks with a sigmoid. Control duration is always
    ``physics_steps_per_control * model.params.dt`` and is reported by the
    runner from the effective reference configuration.
    """

    def __init__(
        self,
        *,
        model: AIEModel,
        target: torch.Tensor,
        reference_curve: DoCReferenceCurve | None = None,
        tracking_mode: str = "curve",
        checkpoints: Sequence[tuple[float, float]] | None = None,
        tracking_specification: TrackingSpecification | None = None,
        horizon: int = 4,
        physics_steps_per_control: int = 10,
        target_threshold: float = 0.5,
        target_weight: float = 1.0,
        outside_weight: float = 0.5,
        energy_weight: float = 1e-4,
        smoothness_weight: float = 1e-2,
        num_iterations: int = 30,
        learning_rate: float = 0.1,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if physics_steps_per_control < 1:
            raise ValueError(
                "physics_steps_per_control must be at least 1, got "
                f"{physics_steps_per_control}"
            )
        if num_iterations < 1:
            raise ValueError(
                f"num_iterations must be at least 1, got {num_iterations}"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if not 0 <= target_threshold <= 1:
            raise ValueError(
                f"target_threshold must be in [0, 1], got {target_threshold}"
            )
        if tracking_specification is None:
            if tracking_mode == "curve":
                if reference_curve is None:
                    raise ValueError("curve tracking requires reference_curve")
                if checkpoints:
                    raise ValueError("curve tracking cannot also define checkpoints")
                tracking_specification = TrackingSpecification.curve(reference_curve)
            elif tracking_mode == "checkpoints":
                if reference_curve is not None:
                    raise ValueError("checkpoint tracking must not mix a fitted curve")
                tracking_specification = TrackingSpecification.checkpoints(checkpoints or ())
            else:
                raise ValueError(
                    "sampled-curve tracking requires an explicit TrackingSpecification"
                )
        elif reference_curve is not None or checkpoints:
            raise ValueError(
                "tracking_specification cannot be combined with legacy reference_curve/checkpoints arguments"
            )
        weights = {
            "target_weight": target_weight,
            "outside_weight": outside_weight,
            "energy_weight": energy_weight,
            "smoothness_weight": smoothness_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")

        if target.ndim != 2:
            raise ValueError(f"target must be 2D, got shape {tuple(target.shape)}")
        target = target.to(device=model.device, dtype=model.dtype)
        with torch.no_grad():
            if not bool(torch.isfinite(target).all()):
                raise ValueError("target contains NaN or Inf")
            if float(target.min()) < 0 or float(target.max()) > 1:
                raise ValueError("target must be normalized to [0, 1]")
        target_region = (target > target_threshold).to(dtype=model.dtype)
        if not bool(target_region.any()):
            raise ValueError(
                "target contains no pixels above target_threshold; choose a lower "
                "threshold or a nonempty target"
            )

        self.model = model
        self.target = target
        self.tracking_specification = tracking_specification
        self.reference_curve = tracking_specification.reference_curve
        self.tracking_mode = tracking_specification.tracking_mode
        self.checkpoints = tuple(
            zip(tracking_specification.point_times_s, tracking_specification.point_values)
        )
        self.target_region = target_region
        self.outside_region = 1.0 - target_region
        self.horizon = horizon
        self.physics_steps_per_control = physics_steps_per_control
        self.target_weight = target_weight
        self.outside_weight = outside_weight
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    @property
    def control_shape(self) -> tuple[int, int]:
        """DLP mask shape implied by the target and model refinement."""

        return self.model.control_shape_for(self.target.shape)

    def initial_control_sequence(
        self,
        *,
        target_level: float = 0.9,
        background_level: float = 0.02,
    ) -> torch.Tensor:
        """Build a target-shaped, valid initial guess for all horizon slots."""

        if not 0 <= target_level <= 1 or not 0 <= background_level <= 1:
            raise ValueError("initial control levels must lie in [0, 1]")
        if self.model.params.projector_refinement > 1:
            projector_target = initialize_projector_mask(
                self.target,
                self.control_shape,
                self.model.params.projector_refinement,
            )
        else:
            projector_target = self.target
        mask = background_level + (target_level - background_level) * projector_target
        return mask.clamp(0.0, 1.0).unsqueeze(0).repeat(self.horizon, 1, 1)

    def optimize(
        self,
        current_state: AIEState,
        initial_guess: torch.Tensor | None = None,
        *,
        current_process_time_s: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Optimize the nonlinear prediction horizon with PyTorch Adam.

        The supplied state is detached once because it is the fixed state
        estimate at the MPC boundary.  No tensor is detached inside the AIE
        physical forward model, so gradients still flow through every future
        physical step to every candidate mask.
        """

        if tuple(current_state.shape) != tuple(self.target.shape):
            raise ValueError(
                f"current state shape {tuple(current_state.shape)} does not match "
                f"target shape {tuple(self.target.shape)}"
            )
        guess = self._validate_and_prepare_guess(initial_guess)
        epsilon = 1e-5
        bounded_guess = guess.clamp(epsilon, 1.0 - epsilon)
        logits = torch.nn.Parameter(torch.logit(bounded_guess))
        optimizer = torch.optim.Adam([logits], lr=self.learning_rate)
        fixed_state = current_state.detach()

        loss_history: list[float] = []
        gradient_norm_history: list[float] = []
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            controls = torch.sigmoid(logits)
            predicted_states = self.predict_stages(
                fixed_state,
                controls,
            )
            loss, _ = self.cost(
                predicted_states,
                controls,
                current_process_time_s=current_process_time_s,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("MPC loss became NaN or Inf")
            loss_history.append(float(loss.detach()))
            loss.backward()
            if logits.grad is None:
                raise RuntimeError("MPC logits did not receive a gradient")
            if not bool(torch.isfinite(logits.grad).all()):
                raise FloatingPointError("MPC gradient became NaN or Inf")
            gradient_norm_history.append(float(logits.grad.norm().detach()))
            optimizer.step()

        with torch.no_grad():
            optimized_controls = torch.sigmoid(logits)
            final_states = self.predict_stages(
                fixed_state,
                optimized_controls,
            )
            final_loss, final_components = self.cost(
                final_states,
                optimized_controls,
                current_process_time_s=current_process_time_s,
            )
            final_loss_value = float(final_loss)
            loss_history.append(final_loss_value)
            result = optimized_controls.detach().clone()

        info: dict[str, Any] = {
            "initial_loss": loss_history[0],
            "final_loss": final_loss_value,
            "loss_history": loss_history,
            "gradient_norm_history": gradient_norm_history,
            "final_cost_components": {
                name: float(value) for name, value in final_components.items()
            },
            "optimized_control_sequence": result,
            "stage_times_s": self.stage_times(current_process_time_s).tolist(),
        }
        stage_values, stage_active = self.stage_tracking_values(current_process_time_s)
        stage_weights = self.stage_tracking_weights(current_process_time_s)
        info["stage_reference_doc"] = [
            float(value) if bool(active) else None
            for value, active in zip(stage_values.cpu(), stage_active.cpu())
        ]
        info["stage_tracking_active"] = stage_active.cpu().tolist()
        info["stage_tracking_weights"] = stage_weights.cpu().tolist()
        return result, info

    @property
    def control_dt_s(self) -> float:
        """Physical duration of one held MPC control."""

        return self.physics_steps_per_control * self.model.params.dt

    def predict_stages(
        self, current_state: AIEState, controls: torch.Tensor
    ) -> tuple[AIEState, ...]:
        """Return the physical state at every future control boundary."""

        expected_control_shape = (self.horizon, *self.control_shape)
        if tuple(controls.shape) != expected_control_shape:
            raise ValueError(
                f"controls must have shape {expected_control_shape}, got "
                f"{tuple(controls.shape)}"
            )
        stages: list[AIEState] = []
        state = current_state
        for control in controls:
            state = self.model.advance(
                state, control, physics_steps=self.physics_steps_per_control
            )
            stages.append(state)
        return tuple(stages)

    def stage_reference_values(self, current_process_time_s: float) -> torch.Tensor:
        """Return scalar target values; inactive checkpoint stages contain zero."""

        values, _ = self.stage_tracking_values(current_process_time_s)
        return values

    def stage_times(self, current_process_time_s: float) -> np.ndarray:
        """Return absolute future control-boundary times for the horizon."""

        if not math.isfinite(current_process_time_s) or current_process_time_s < 0:
            raise ValueError("current process time must be finite and nonnegative")
        return current_process_time_s + self.control_dt_s * np.arange(
            1, self.horizon + 1, dtype=float
        )

    def stage_tracking_values(
        self, current_process_time_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage values and a mask selecting target-tracking stages."""
        schedule = self.tracking_specification.stage_schedule(
            current_process_time_s, self.control_dt_s, self.horizon
        )
        return (
            torch.as_tensor(schedule.required_doc, device=self.model.device, dtype=self.model.dtype),
            torch.as_tensor(schedule.active, device=self.model.device, dtype=torch.bool),
        )

    def stage_tracking_weights(self, current_process_time_s: float) -> torch.Tensor:
        schedule = self.tracking_specification.stage_schedule(
            current_process_time_s, self.control_dt_s, self.horizon
        )
        return torch.as_tensor(
            schedule.point_weights, device=self.model.device, dtype=self.model.dtype
        )

    def desired_doc_stages(self, current_process_time_s: float) -> torch.Tensor:
        """Construct ``M(x,y) * r(t_absolute)`` for every prediction stage."""

        reference = self.stage_reference_values(current_process_time_s)
        return reference[:, None, None] * self.target[None, :, :]

    def cost(
        self,
        predicted_states: Sequence[AIEState],
        controls: torch.Tensor,
        *,
        current_process_time_s: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate equal-weight absolute-time tracking and existing penalties."""

        if len(predicted_states) != self.horizon:
            raise ValueError(
                f"predicted_states must contain {self.horizon} stages, got "
                f"{len(predicted_states)}"
            )
        for index, predicted_state in enumerate(predicted_states):
            if tuple(predicted_state.doc.shape) != tuple(self.target.shape):
                raise ValueError(
                    f"predicted stage {index} DoC shape "
                    f"{tuple(predicted_state.doc.shape)} does not match target "
                    f"shape {tuple(self.target.shape)}"
                )
        expected_control_shape = (self.horizon, *self.control_shape)
        if tuple(controls.shape) != expected_control_shape:
            raise ValueError(
                f"controls must have shape {expected_control_shape}, got "
                f"{tuple(controls.shape)}"
            )

        desired_doc_stages = self.desired_doc_stages(current_process_time_s)
        stage_reference, target_tracking_active = self.stage_tracking_values(
            current_process_time_s
        )
        stage_tracking_weights = self.stage_tracking_weights(current_process_time_s)
        target_stage_costs: list[torch.Tensor] = []
        active_weights: list[torch.Tensor] = []
        outside_stage_costs: list[torch.Tensor] = []
        for stage_index, (predicted_state, desired_doc) in enumerate(
            zip(predicted_states, desired_doc_stages)
        ):
            if bool(target_tracking_active[stage_index]):
                if self.tracking_specification.spatial_definition == "pixelwise":
                    error = predicted_state.doc - desired_doc
                    stage_cost = self._masked_mean(
                        self._tracking_loss(error), self.target_region
                    )
                else:
                    target_mean = self._masked_mean(
                        predicted_state.doc, self.target_region
                    )
                    error = target_mean - stage_reference[stage_index]
                    stage_cost = self._tracking_loss(error)
                target_stage_costs.append(
                    stage_tracking_weights[stage_index] * stage_cost
                )
                active_weights.append(stage_tracking_weights[stage_index])
            outside_stage_costs.append(
                self._masked_mean(
                    predicted_state.doc.square(), self.outside_region
                )
            )
        target_cost = (
            torch.stack(target_stage_costs).sum()
            / torch.stack(active_weights).sum().clamp_min(1e-12)
            if target_stage_costs
            else predicted_states[0].doc.sum() * 0.0
        )
        outside_cost = torch.stack(outside_stage_costs).mean()
        energy_cost = controls.square().mean()
        if self.horizon > 1:
            smoothness_cost = (controls[1:] - controls[:-1]).square().mean()
        else:
            smoothness_cost = controls.sum() * 0.0
        total = (
            self.target_weight * target_cost
            + self.outside_weight * outside_cost
            + self.energy_weight * energy_cost
            + self.smoothness_weight * smoothness_cost
        )
        return total, {
            "target": target_cost,
            "outside": outside_cost,
            "energy": energy_cost,
            "smoothness": smoothness_cost,
        }

    def _tracking_loss(self, error: torch.Tensor) -> torch.Tensor:
        """Apply the configured pointwise target loss without changing penalties."""

        if self.tracking_specification.tracking_loss == "mse":
            return error.square()
        if self.tracking_specification.tracking_loss == "mae":
            return error.abs()
        delta = self.tracking_specification.huber_delta
        absolute = error.abs()
        return torch.where(
            absolute <= delta,
            0.5 * error.square(),
            delta * (absolute - 0.5 * delta),
        )

    def shift_warm_start(self, optimized_controls: torch.Tensor) -> torch.Tensor:
        """Shift ``[u0, u1, ..., uN]`` to ``[u1, ..., uN, uN]``."""

        expected_shape = (self.horizon, *self.control_shape)
        if tuple(optimized_controls.shape) != expected_shape:
            raise ValueError(
                f"optimized_controls must have shape {expected_shape}, got "
                f"{tuple(optimized_controls.shape)}"
            )
        shifted = torch.cat(
            (optimized_controls[1:], optimized_controls[-1:].clone()), dim=0
        )
        return shifted.detach().clone()

    def _validate_and_prepare_guess(
        self, initial_guess: torch.Tensor | None
    ) -> torch.Tensor:
        if initial_guess is None:
            guess = self.initial_control_sequence()
        else:
            guess = initial_guess.to(device=self.model.device, dtype=self.model.dtype)
            if guess.ndim == 2:
                guess = guess.unsqueeze(0).repeat(self.horizon, 1, 1)
        expected_shape = (self.horizon, *self.control_shape)
        if tuple(guess.shape) != expected_shape:
            raise ValueError(
                f"initial_guess must have shape {expected_shape} (or one mask with "
                f"shape {self.control_shape}), got {tuple(guess.shape)}"
            )
        if not bool(torch.isfinite(guess).all()):
            raise ValueError("initial_guess contains NaN or Inf")
        if float(guess.min()) < 0 or float(guess.max()) > 1:
            raise ValueError("initial_guess must be normalized to [0, 1]")
        return guess.detach().clone()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
