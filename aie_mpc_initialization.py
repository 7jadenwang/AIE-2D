"""Physics-aware cold-start initialization for differentiable AIE MPC.

Physics-aware initialization is an optimization heuristic only.  It does not
alter AIE forward physics or the MPC objective.  The optical inverse problem
reuses :meth:`AIEModel.prepare_control` and never advances chemical state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from aie_fine_grid import initialize_projector_mask
from aie_model import AIEModel, AIEParameters


DEFAULT_PHYSICS_INIT_ITERATIONS = 100
DEFAULT_PHYSICS_INIT_LEARNING_RATE = 0.1
DEFAULT_PHYSICS_INIT_OUTSIDE_WEIGHT = 0.5
NOMINAL_INTENSITY_BISECTION_ITERATIONS = 100


class PhysicsAwareInitializationError(ValueError):
    """Raised when the requested cold-start approximation is unsupported."""


@dataclass(frozen=True)
class NominalIntensitySolution:
    """Bounded constant-intensity solution for one checkpoint requirement."""

    checkpoint_time_s: float
    checkpoint_doc: float
    intensity_mw_cm2: float
    normalized_level: float
    predicted_doc: float
    maximum_achievable_doc: float
    bisection_iterations: int


@dataclass(frozen=True)
class PhysicsAwareInitializationResult:
    """Optimized projector mask and its forward-model optical field."""

    projector_mask: torch.Tensor
    local_normalized_intensity: torch.Tensor
    nominal: NominalIntensitySolution
    initial_optical_loss: float
    final_optical_loss: float
    optical_iterations: int
    optical_learning_rate: float
    optical_outside_weight: float


def constant_intensity_checkpoint_doc(
    params: AIEParameters,
    intensity_mw_cm2: float,
    checkpoint_time_s: float,
) -> float:
    """Evaluate the documented O2-only constant-local-intensity heuristic.

    This scalar approximation is deliberately separate from the authoritative
    spatial forward model.  The active affine B/intensity coefficients and O2
    inhibition are read from ``params``; no collaborator values are duplicated.
    """

    intensity = float(intensity_mw_cm2)
    checkpoint_time = float(checkpoint_time_s)
    if not math.isfinite(intensity) or intensity < 0.0:
        raise PhysicsAwareInitializationError(
            f"nominal intensity must be finite and nonnegative, got {intensity}"
        )
    if not math.isfinite(checkpoint_time) or checkpoint_time <= 0.0:
        raise PhysicsAwareInitializationError(
            "initialization checkpoint time must be finite and positive"
        )
    if intensity == 0.0:
        return 0.0

    curing_duration_s = (
        checkpoint_time - params.o2_inhibition_mj_cm2 / intensity
    )
    if curing_duration_s <= 0.0:
        return 0.0
    b_value = params.b_slope * intensity + params.b_intercept
    exponent = b_value * curing_duration_s
    if not math.isfinite(exponent) or exponent < 0.0:
        raise PhysicsAwareInitializationError(
            "active B/intensity parameters do not produce a finite nonnegative "
            "constant-intensity cure exponent"
        )
    return float(-math.expm1(-exponent))


def solve_checkpoint_nominal_intensity(
    params: AIEParameters,
    checkpoint_time_s: float,
    checkpoint_doc: float,
    *,
    max_iterations: int = NOMINAL_INTENSITY_BISECTION_ITERATIONS,
) -> NominalIntensitySolution:
    """Solve the O2-only checkpoint heuristic by monotone bounded bisection."""

    checkpoint_time = float(checkpoint_time_s)
    requested_doc = float(checkpoint_doc)
    maximum_intensity = float(params.intensity_mw_cm2)
    if not math.isfinite(checkpoint_time) or checkpoint_time <= 0.0:
        raise PhysicsAwareInitializationError(
            "initialization checkpoint time must be finite and positive"
        )
    if not math.isfinite(requested_doc) or not 0.0 < requested_doc < 1.0:
        raise PhysicsAwareInitializationError(
            "physics-aware initialization requires checkpoint DoC strictly in (0, 1)"
        )
    if not math.isfinite(maximum_intensity) or maximum_intensity <= 0.0:
        raise PhysicsAwareInitializationError(
            "active maximum intensity must be finite and positive"
        )
    if params.o2_inhibition_mj_cm2 < 0.0:
        raise PhysicsAwareInitializationError(
            "active O2 inhibition must be nonnegative"
        )
    if params.tempo_inhibition_mj_cm2 > 1e-12:
        raise PhysicsAwareInitializationError(
            "physics-aware checkpoint initialization currently supports only "
            "the O2-only/zero-TEMPO-inhibition approximation"
        )
    if params.chain_growth_noise_std > 0.0:
        raise PhysicsAwareInitializationError(
            "physics-aware checkpoint initialization requires zero active "
            "chain-growth noise"
        )
    if (
        not math.isfinite(params.b_slope)
        or not math.isfinite(params.b_intercept)
        or params.b_slope < 0.0
        or params.b_intercept < 0.0
        or (params.b_slope == 0.0 and params.b_intercept == 0.0)
    ):
        raise PhysicsAwareInitializationError(
            "active affine B/intensity law must be finite, nonnegative, and nonzero "
            "for monotone bisection"
        )
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise PhysicsAwareInitializationError(
            "nominal-intensity bisection iterations must be a positive integer"
        )

    maximum_doc = constant_intensity_checkpoint_doc(
        params, maximum_intensity, checkpoint_time
    )
    feasibility_tolerance = 1e-10
    if requested_doc > maximum_doc + feasibility_tolerance:
        raise PhysicsAwareInitializationError(
            "requested checkpoint is infeasible under the bounded O2-only "
            "constant-intensity heuristic: "
            f"requested DoC={requested_doc:.8g} at {checkpoint_time:.8g} s, "
            f"maximum DoC={maximum_doc:.8g} at I_max={maximum_intensity:.8g} mW/cm^2"
        )
    if math.isclose(
        requested_doc, maximum_doc, rel_tol=0.0, abs_tol=feasibility_tolerance
    ):
        solved_intensity = maximum_intensity
        completed_iterations = 0
    else:
        lower = 0.0
        upper = maximum_intensity
        intensity_tolerance = max(1e-10, 1e-10 * maximum_intensity)
        completed_iterations = 0
        for completed_iterations in range(1, max_iterations + 1):
            midpoint = 0.5 * (lower + upper)
            midpoint_doc = constant_intensity_checkpoint_doc(
                params, midpoint, checkpoint_time
            )
            if midpoint_doc < requested_doc:
                lower = midpoint
            else:
                upper = midpoint
            if upper - lower <= intensity_tolerance:
                break
        solved_intensity = 0.5 * (lower + upper)

    if not 0.0 < solved_intensity <= maximum_intensity:
        raise PhysicsAwareInitializationError(
            f"bounded solver returned invalid intensity {solved_intensity}"
        )
    normalized_level = solved_intensity / maximum_intensity
    if not math.isfinite(normalized_level):
        raise PhysicsAwareInitializationError(
            "bounded solver returned a nonfinite normalized intensity"
        )
    normalized_level = min(1.0, max(0.0, normalized_level))
    predicted_doc = constant_intensity_checkpoint_doc(
        params, solved_intensity, checkpoint_time
    )
    if abs(predicted_doc - requested_doc) > 1e-7:
        raise PhysicsAwareInitializationError(
            "nominal-intensity bisection did not converge to the requested checkpoint: "
            f"requested={requested_doc:.8g}, predicted={predicted_doc:.8g}"
        )
    return NominalIntensitySolution(
        checkpoint_time_s=checkpoint_time,
        checkpoint_doc=requested_doc,
        intensity_mw_cm2=solved_intensity,
        normalized_level=normalized_level,
        predicted_doc=predicted_doc,
        maximum_achievable_doc=maximum_doc,
        bisection_iterations=completed_iterations,
    )


def build_physics_aware_initial_mask(
    *,
    model: AIEModel,
    target: torch.Tensor,
    target_threshold: float,
    checkpoint_time_s: float,
    checkpoint_doc: float,
    background_level: float = 0.02,
    num_iterations: int = DEFAULT_PHYSICS_INIT_ITERATIONS,
    learning_rate: float = DEFAULT_PHYSICS_INIT_LEARNING_RATE,
    outside_weight: float = DEFAULT_PHYSICS_INIT_OUTSIDE_WEIGHT,
) -> PhysicsAwareInitializationResult:
    """Build one geometry-aware first-solve mask using optical inversion only.

    Physics-aware initialization is an optimization heuristic only.  It does
    not alter AIE forward physics or the MPC objective.  In particular, this
    function calls ``prepare_control`` but never ``step`` or ``advance``.
    """

    if target.ndim != 2:
        raise PhysicsAwareInitializationError(
            f"physics-aware target must be 2D, got shape {tuple(target.shape)}"
        )
    target = target.to(device=model.device, dtype=model.dtype)
    if not 0.0 <= target_threshold <= 1.0:
        raise PhysicsAwareInitializationError("target_threshold must lie in [0, 1]")
    if not 0.0 <= background_level <= 1.0:
        raise PhysicsAwareInitializationError(
            "physics-aware background seed must lie in [0, 1]"
        )
    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise PhysicsAwareInitializationError(
            "physics-aware optical iterations must be a positive integer"
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise PhysicsAwareInitializationError(
            "physics-aware optical learning rate must be finite and positive"
        )
    if not math.isfinite(outside_weight) or outside_weight < 0.0:
        raise PhysicsAwareInitializationError(
            "physics-aware optical outside weight must be finite and nonnegative"
        )
    with torch.no_grad():
        if not bool(torch.isfinite(target).all()):
            raise PhysicsAwareInitializationError("physics-aware target is nonfinite")
        if float(target.min()) < 0.0 or float(target.max()) > 1.0:
            raise PhysicsAwareInitializationError(
                "physics-aware target must be normalized to [0, 1]"
            )

    target_region = (target > target_threshold).to(dtype=model.dtype)
    if not bool(target_region.any()):
        raise PhysicsAwareInitializationError(
            "physics-aware target has no pixels above target_threshold"
        )
    outside_region = 1.0 - target_region
    nominal = solve_checkpoint_nominal_intensity(
        model.params, checkpoint_time_s, checkpoint_doc
    )

    control_shape = model.control_shape_for(target.shape)
    projector_target = (
        initialize_projector_mask(
            target,
            control_shape,
            model.params.projector_refinement,
        )
        if model.params.projector_refinement > 1
        else target
    )
    base_mask = background_level + (
        nominal.normalized_level - background_level
    ) * projector_target
    logit_epsilon = 1e-5
    logits = torch.nn.Parameter(
        torch.logit(base_mask.clamp(logit_epsilon, 1.0 - logit_epsilon))
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    desired_local_field = nominal.normalized_level * target

    def optical_cost(projector_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = model.prepare_control(projector_mask, target.shape)
        local_normalized = (
            prepared.local_intensity / model.params.intensity_mw_cm2
        )
        target_error = (local_normalized - desired_local_field).square()
        target_cost = (target_error * target_region).sum() / target_region.sum()
        outside_cost = (
            (local_normalized.square() * outside_region).sum()
            / outside_region.sum().clamp_min(1.0)
        )
        return target_cost + outside_weight * outside_cost, local_normalized

    initial_loss: float | None = None
    for _ in range(num_iterations):
        optimizer.zero_grad()
        projector_mask = torch.sigmoid(logits)
        loss, _ = optical_cost(projector_mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "physics-aware optical initialization loss became NaN or Inf"
            )
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError(
                "physics-aware optical initialization gradient is missing or nonfinite"
            )
        optimizer.step()

    with torch.no_grad():
        optimized_mask = torch.sigmoid(logits)
        final_loss, local_normalized = optical_cost(optimized_mask)
        if not bool(torch.isfinite(final_loss)) or not bool(
            torch.isfinite(local_normalized).all()
        ):
            raise FloatingPointError(
                "physics-aware optical initialization result is nonfinite"
            )
        if float(optimized_mask.min()) < 0.0 or float(optimized_mask.max()) > 1.0:
            raise AssertionError("physics-aware projector mask escaped [0, 1]")
        tolerance = 1e-6
        if (
            float(local_normalized.min()) < -tolerance
            or float(local_normalized.max()) > 1.0 + tolerance
        ):
            raise AssertionError(
                "forward-model normalized local intensity escaped [0, 1]"
            )

    assert initial_loss is not None
    return PhysicsAwareInitializationResult(
        projector_mask=optimized_mask.detach().clone(),
        local_normalized_intensity=local_normalized.detach().clone(),
        nominal=nominal,
        initial_optical_loss=initial_loss,
        final_optical_loss=float(final_loss),
        optical_iterations=num_iterations,
        optical_learning_rate=float(learning_rate),
        optical_outside_weight=float(outside_weight),
    )
