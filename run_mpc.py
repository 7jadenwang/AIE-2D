"""Run native or physically consistent coarse-grid differentiable AIE MPC.

Each experiment's native grid is the loaded target's exact 2-D pixel shape.
Coarse mode optimizes a block-averaged target on a lower-resolution grid with
the same physical field of view, expands applied masks by constant blocks, and
replays those masks through a fresh native-resolution AIE model.

Run ``python run_mpc.py --smoke-test`` for lightweight physics, gradient, MPC,
resolution-conversion, field-of-view, and native-replay validation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, fields, replace
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import GifImagePlugin, Image

from aie_fine_grid import expand_projector_mask, initialize_projector_mask
from aie_model import (
    DOC_HISTORY_DESCRIPTION,
    DOC_HISTORY_MODE,
    AIEModel,
    AIEParameters,
    AIEState,
)
from aie_mpc import DifferentiableMPC
from aie_mpc_initialization import (
    DEFAULT_PHYSICS_INIT_ITERATIONS,
    DEFAULT_PHYSICS_INIT_LEARNING_RATE,
    DEFAULT_PHYSICS_INIT_OUTSIDE_WEIGHT,
    PhysicsAwareInitializationResult,
    build_physics_aware_initial_mask,
)
import aie_reference
from aie_reference import (
    SUPPORTED_FORWARD_CONDITIONS,
    load_reference_config,
    load_reference_config_for_condition,
    reference_step_torch,
)
from doc_reference import (
    CURVE_MODEL_IDS,
    DEFAULT_REFERENCE_PATH,
    DEFAULT_REFERENCE_ID,
    LEGACY_REFERENCE_PATH,
    DoCReferenceCurve,
    available_curve_models,
    available_doc_reference_ids,
    load_doc_reference,
)
from tracking_config import (
    SPATIAL_DEFINITIONS,
    TRACKING_LOSSES,
    TRACKING_MODES,
    TRACKING_VARIABLES,
    TrackingConfigurationError,
    TrackingSpecification,
    doc_to_reaction_progress,
    parse_checkpoints as parse_tracking_checkpoints,
    parse_float_list,
    resolve_sampled_tracking_times,
)
from mpc_metrics import (
    GEOMETRY_THRESHOLD_NOTE,
    calculate_final_metrics,
    initializer_component_metrics,
    temporal_tracking_metrics,
)


REPOSITORY_DIR = Path(__file__).resolve().parent
GEO_DIR = REPOSITORY_DIR / "GEO"


def assess_reference_physics_match(
    reference_curve: DoCReferenceCurve,
    params: AIEParameters,
    physics_condition_id: str | None = None,
) -> dict[str, object]:
    """Audit whether authoritative forward physics matches a tracking condition."""

    condition = reference_curve.metadata["condition"]
    intensity = float(condition["intensity_mw_cm2"])
    tempo_mM = float(condition["tempo_concentration_mM"])
    missing: list[str] = []
    try:
        selected_condition = physics_condition_id or reference_curve.reference_id
        expected = AIEParameters.from_reference(
            load_reference_config_for_condition(selected_condition)
        )
    except (ValueError, aie_reference.ReferenceResolutionError) as error:
        expected = None
        missing.append(f"named authoritative condition: {error}")
    if expected is not None:
        compared_fields = (
            "intensity_mw_cm2",
            "tempo_concentration_mM",
            "o2_inhibition_mj_cm2",
            "total_inhibition_mj_cm2",
            "o2_diffusivity_m2_s",
            "tempo_diffusivity_m2_s",
            "b_slope",
            "b_intercept",
            "scattering_blur_size_m",
            "dt",
        )
        for field_name in compared_fields:
            actual_value = getattr(params, field_name)
            expected_value = getattr(expected, field_name)
            if actual_value is None or expected_value is None:
                equal = actual_value is expected_value
            else:
                equal = math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            if not equal:
                missing.append(
                    f"{field_name}={expected_value!r} (actual {actual_value!r})"
                )
    if not math.isclose(
        params.intensity_mw_cm2, intensity, rel_tol=0.0, abs_tol=1e-12
    ):
        missing.append(f"tracking-reference intensity={intensity:g} mW/cm^2")
    if params.tempo_concentration_mM is None or not math.isclose(
        params.tempo_concentration_mM, tempo_mM, rel_tol=0.0, abs_tol=1e-12
    ):
        missing.append(f"tracking-reference TEMPO={tempo_mM:g} mM")

    matched = not missing
    return {
        "matched": matched,
        "tracking_reference_id": reference_curve.reference_id,
        "tracking_condition": condition,
        "selected_forward_condition": physics_condition_id or reference_curve.reference_id,
        "forward_intensity_mw_cm2": params.intensity_mw_cm2,
        "forward_tempo_concentration_mM": params.tempo_concentration_mM,
        "forward_o2_inhibition_mj_cm2": params.o2_inhibition_mj_cm2,
        "forward_total_inhibition_mj_cm2": params.total_inhibition_mj_cm2,
        "forward_tempo_inhibition_mj_cm2": params.tempo_inhibition_mj_cm2,
        "forward_tempo_diffusivity_m2_s": params.tempo_diffusivity_m2_s,
        "forward_b_condition_label": params.b_condition_label,
        "missing_for_physical_match": missing,
    }


@dataclass(frozen=True)
class ResolutionConfig:
    """Grid geometry for one native or coarse MPC run."""

    resolution_mode: str
    native_shape: tuple[int, int]
    optimization_shape: tuple[int, int]
    coarsen_factor: int
    native_pixel_pitch_m: float
    optimization_pixel_pitch_m: float

    @property
    def native_fov_m(self) -> tuple[float, float]:
        """Return physical field of view as ``(FOV_y, FOV_x)``."""

        return tuple(
            dimension * self.native_pixel_pitch_m for dimension in self.native_shape
        )

    @property
    def optimization_fov_m(self) -> tuple[float, float]:
        """Return optimization field of view as ``(FOV_y, FOV_x)``."""

        return tuple(
            dimension * self.optimization_pixel_pitch_m
            for dimension in self.optimization_shape
        )


@dataclass(frozen=True)
class TargetMaskBaselineResult:
    """Outputs from the unoptimized repeated target-mask forward rollout."""

    final_state: AIEState
    timeline: dict[str, np.ndarray]
    initial_target_doc_summary: dict[str, float]
    initial_target_o2_summary: dict[str, float]
    tracking: dict[str, float]
    tracking_points: dict[str, object]
    final_metrics: dict[str, object]
    metrics_document: dict[str, object]
    output_dir: Path


def build_resolution_config(
    resolution_mode: str,
    coarsen_factor: int,
    native_shape: tuple[int, int],
    reference: object | None = None,
) -> ResolutionConfig:
    """Validate a resolution mode and preserve the native physical field of view."""

    resolved_reference = reference or load_reference_config()
    native_shape = tuple(native_shape)
    if len(native_shape) != 2 or any(
        not isinstance(dimension, int) or dimension <= 0
        for dimension in native_shape
    ):
        raise ValueError(
            f"native_shape must contain two positive integer dimensions, got "
            f"{native_shape}"
        )
    native_pitch = resolved_reference.native_pixel_pitch_m
    if resolution_mode not in {"native", "coarse"}:
        raise ValueError(
            f"resolution_mode must be 'native' or 'coarse', got {resolution_mode!r}"
        )
    if resolution_mode == "native":
        factor = 1
    else:
        if coarsen_factor < 2:
            raise ValueError(
                f"coarsen_factor must be at least 2 in coarse mode, got {coarsen_factor}"
            )
        if any(dimension % coarsen_factor for dimension in native_shape):
            raise ValueError(
                f"coarsen_factor {coarsen_factor} must divide both dimensions of "
                f"the native grid {native_shape} exactly; no resizing, cropping, "
                "padding, or rounding is allowed"
            )
        factor = coarsen_factor

    optimization_shape = tuple(dimension // factor for dimension in native_shape)
    config = ResolutionConfig(
        resolution_mode=resolution_mode,
        native_shape=native_shape,
        optimization_shape=optimization_shape,
        coarsen_factor=factor,
        native_pixel_pitch_m=native_pitch,
        optimization_pixel_pitch_m=factor * native_pitch,
    )
    for native_fov, optimization_fov in zip(
        config.native_fov_m, config.optimization_fov_m
    ):
        if not math.isclose(
            native_fov, optimization_fov, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise AssertionError(
                "physical field-of-view mismatch: "
                f"native={config.native_fov_m}, optimization={config.optimization_fov_m}"
            )
    return config


def resolve_target_path(target: Path) -> Path:
    """Resolve absolute, ``GEO/...``, or GEO-relative target paths."""

    if target.is_absolute():
        resolved = target
    elif target.parts and target.parts[0].lower() == "geo":
        resolved = REPOSITORY_DIR / target
    else:
        resolved = GEO_DIR / target
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"target image not found: {resolved}. Unqualified target names are "
            f"resolved relative to {GEO_DIR}"
        )
    return resolved


def load_normalized_target(path: Path) -> torch.Tensor:
    """Load an 8/16-bit or color target using the legacy normalization rule."""

    with Image.open(path) as image:
        if image.mode == "I;16":
            array = np.asarray(image)
            maximum = 2**16 - 1
        elif image.mode == "L":
            array = np.asarray(image)
            maximum = 255
        else:
            array = np.asarray(image.convert("L"))
            maximum = 255
    return torch.from_numpy((array / maximum).astype(np.float32))


def require_native_target(target: torch.Tensor, path: Path | None = None) -> None:
    """Validate a native target without changing its spatial dimensions."""

    location = f" {path}" if path is not None else ""
    if target.ndim != 2:
        raise ValueError(
            f"target{location} must be a 2D grayscale image, got shape "
            f"{tuple(target.shape)}"
        )
    if any(dimension <= 0 for dimension in target.shape):
        raise ValueError(
            f"target{location} dimensions must be positive, got {tuple(target.shape)}"
        )
    with torch.no_grad():
        if not bool(torch.isfinite(target).all()):
            raise ValueError(f"target{location} contains NaN or Inf")
        minimum = float(target.min())
        maximum = float(target.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"target{location} values must be normalized to [0, 1], got "
            f"[{minimum:.6g}, {maximum:.6g}]"
        )


def construct_optimization_target(
    target_native: torch.Tensor, config: ResolutionConfig
) -> torch.Tensor:
    """Return the unchanged native target or an exact block-average target."""

    require_native_target(target_native)
    if tuple(target_native.shape) != config.native_shape:
        raise ValueError(
            f"target shape {tuple(target_native.shape)} does not match configured "
            f"native shape {config.native_shape}; targets are never resized, "
            "cropped, or padded"
        )
    if config.resolution_mode == "native":
        return target_native
    target_coarse = initialize_projector_mask(
        target_native,
        projector_shape=config.optimization_shape,
        refinement=config.coarsen_factor,
    )
    if tuple(target_coarse.shape) != config.optimization_shape:
        raise AssertionError(
            f"coarse target has shape {tuple(target_coarse.shape)}, expected "
            f"{config.optimization_shape}"
        )
    return target_coarse


def recover_control_to_native(
    optimization_control: torch.Tensor, config: ResolutionConfig
) -> torch.Tensor:
    """Recover one physical mask using constant nearest/block expansion."""

    if tuple(optimization_control.shape) != config.optimization_shape:
        raise ValueError(
            f"optimization control must have shape {config.optimization_shape}, got "
            f"{tuple(optimization_control.shape)}"
        )
    if config.resolution_mode == "native":
        native_control = optimization_control
    else:
        native_control = expand_projector_mask(
            optimization_control, refinement=config.coarsen_factor
        )
    if tuple(native_control.shape) != config.native_shape:
        raise AssertionError(
            f"recovered physical mask must have shape {config.native_shape}, got "
            f"{tuple(native_control.shape)}"
        )
    return native_control


def save_grayscale(tensor: torch.Tensor, path: Path) -> None:
    """Save a normalized tensor as a 16-bit grayscale PNG."""

    values = tensor.detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.round(values * 65535).astype(np.uint16)).save(path)


def validate_doc_field(doc: torch.Tensor, expected_shape: tuple[int, int]) -> None:
    """Require a finite DoC field in the numerical range represented by images."""

    if tuple(doc.shape) != expected_shape:
        raise ValueError(
            f"DoC frame must have shape {expected_shape}, got {tuple(doc.shape)}"
        )
    with torch.no_grad():
        if not bool(torch.isfinite(doc).all()):
            raise ValueError("DoC frame contains NaN or Inf")
        minimum = float(doc.min())
        maximum = float(doc.max())
    tolerance = 1e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"DoC frame values must remain in [0, 1], got "
            f"[{minimum:.6g}, {maximum:.6g}]"
        )


def save_doc_frame(
    doc: torch.Tensor, path: Path, expected_shape: tuple[int, int]
) -> None:
    """Validate and save one DoC frame with the final-DoC grayscale convention."""

    validate_doc_field(doc, expected_shape)
    save_grayscale(doc, path)


def _gif_frame_from_doc_png(path: Path) -> Image.Image:
    """Convert a 16-bit normalized DoC PNG to a fixed grayscale GIF palette."""

    with Image.open(path) as image:
        values = np.asarray(image)
    if np.issubdtype(values.dtype, np.integer) and values.dtype.itemsize > 1:
        values_8bit = np.round(values.astype(np.float64) / 257.0).astype(np.uint8)
    else:
        values_8bit = np.asarray(values, dtype=np.uint8)
    frame = Image.fromarray(values_8bit, mode="P")
    grayscale_palette = [component for level in range(256) for component in (level,) * 3]
    frame.putpalette(grayscale_palette)
    return frame


def validate_gif(
    gif_path: Path,
    expected_frame_count: int,
    expected_shape: tuple[int, int],
) -> None:
    """Confirm GIF existence, exact timeline length, resolution, and looping."""

    if not gif_path.is_file():
        raise AssertionError(f"GIF was not created: {gif_path}")
    with Image.open(gif_path) as gif:
        actual_shape = (gif.height, gif.width)
        actual_frame_count = getattr(gif, "n_frames", 1)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"GIF {gif_path} has shape {actual_shape}, expected {expected_shape}"
            )
        if actual_frame_count != expected_frame_count:
            raise AssertionError(
                f"GIF {gif_path} has {actual_frame_count} frames, expected "
                f"{expected_frame_count}"
            )
        if gif.info.get("loop") != 0:
            raise AssertionError(f"GIF {gif_path} is not configured to loop continuously")


def save_gif_from_frames(
    frame_paths: list[Path], gif_path: Path, duration_ms: int = 300
) -> None:
    """Create a looping GIF while retaining even identical chronological frames.

    Pillow's high-level multi-frame saver merges consecutive identical images.
    Early DoC frames are often identically zero because of inhibition, so this
    PIL writer emits each full frame explicitly to preserve one frame per saved
    control state.
    """

    if not frame_paths:
        raise ValueError("at least one frame is required to create a GIF")
    if duration_ms < 1:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")
    frames = [_gif_frame_from_doc_png(path) for path in frame_paths]
    expected_size = frames[0].size
    if any(frame.size != expected_size for frame in frames):
        raise ValueError("all GIF source frames must have the same dimensions")
    try:
        with gif_path.open("wb") as gif_file:
            header, _ = GifImagePlugin.getheader(frames[0], info={"loop": 0})
            for block in header:
                gif_file.write(block)
            for frame in frames:
                for block in GifImagePlugin.getdata(frame, duration=duration_ms):
                    gif_file.write(block)
            gif_file.write(b";")
    finally:
        for frame in frames:
            frame.close()
    validate_gif(
        gif_path,
        expected_frame_count=len(frame_paths),
        expected_shape=(expected_size[1], expected_size[0]),
    )


def native_aie_parameters(reference: object | None = None) -> AIEParameters:
    """Create native parameters only from the authoritative reference."""

    return AIEParameters.from_reference(reference or load_reference_config())


def replay_native_controls(
    native_params: AIEParameters,
    native_shape: tuple[int, int],
    applied_controls_native: list[torch.Tensor],
    physics_steps_per_control: int,
    device: torch.device,
    doc_frame_callback: Callable[[int, torch.Tensor], None] | None = None,
    authoritative_reference: object | None = None,
    state_callback: Callable[[int, AIEState], None] | None = None,
    native_model: AIEModel | None = None,
    initial_state: AIEState | None = None,
) -> AIEState:
    """Replay recovered masks through the resolved native forward model."""

    current_reference_params = AIEParameters.from_reference(
        authoritative_reference or load_reference_config()
    )
    if native_params != current_reference_params:
        raise ValueError(
            "native replay parameters must exactly match the current "
            "AIE_TEMPOv1.1.py reference configuration"
        )
    native_shape = tuple(native_shape)
    if len(native_shape) != 2 or any(dimension <= 0 for dimension in native_shape):
        raise ValueError(
            f"native replay shape must contain two positive dimensions, got "
            f"{native_shape}"
        )
    if native_model is None:
        native_model = AIEModel(native_params, device=device)
    elif native_model.params != native_params or native_model.device != device:
        raise ValueError(
            "reused native model must have the requested authoritative parameters "
            "and device"
        )
    native_state = (
        native_model.initialize_state(native_shape)
        if initial_state is None
        else initial_state.detach()
    )
    if tuple(native_state.shape) != native_shape:
        raise ValueError(
            f"native replay initial state has shape {tuple(native_state.shape)}, "
            f"expected {native_shape}"
        )
    if doc_frame_callback is not None:
        doc_frame_callback(0, native_state.doc)
    if state_callback is not None:
        state_callback(0, native_state)
    with torch.no_grad():
        for replay_step, control in enumerate(applied_controls_native, start=1):
            if tuple(control.shape) != native_shape:
                raise ValueError(
                    f"native replay control must have shape {native_shape}, got "
                    f"{tuple(control.shape)}"
                )
            native_state = native_model.advance(
                native_state,
                control,
                physics_steps=physics_steps_per_control,
            )
            if doc_frame_callback is not None:
                doc_frame_callback(replay_step, native_state.doc)
            if state_callback is not None:
                state_callback(replay_step, native_state)
    return native_state


def doc_region_metrics(
    doc: torch.Tensor, target: torch.Tensor, target_threshold: float
) -> tuple[float, float]:
    """Compute target and outside mean DoC with the MPC threshold convention."""

    summary = doc_region_summary(doc, target, target_threshold)
    return summary["mean_target_doc"], summary["mean_outside_doc"]


def doc_region_summary(
    doc: torch.Tensor, target: torch.Tensor, target_threshold: float
) -> dict[str, float]:
    """Compute target-region spread and outside mean using the MPC convention."""

    if tuple(doc.shape) != tuple(target.shape):
        raise ValueError(
            f"DoC shape {tuple(doc.shape)} does not match target shape "
            f"{tuple(target.shape)}"
        )
    target_region = target > target_threshold
    if not bool(target_region.any()):
        raise ValueError("target has no pixels above target_threshold")
    outside_region = ~target_region
    target_values = doc[target_region]
    outside_mean = (
        float(doc[outside_region].mean()) if bool(outside_region.any()) else 0.0
    )
    return {
        "mean_target_doc": float(target_values.mean()),
        "min_target_doc": float(target_values.min()),
        "max_target_doc": float(target_values.max()),
        "std_target_doc": float(target_values.std(unbiased=False)),
        "mean_outside_doc": outside_mean,
    }


def target_field_summary(
    field: torch.Tensor,
    target: torch.Tensor,
    target_threshold: float,
    *,
    label: str,
) -> dict[str, float]:
    """Compute target-region statistics for a physical model-state field."""

    if tuple(field.shape) != tuple(target.shape):
        raise ValueError(
            f"{label} shape {tuple(field.shape)} does not match target shape "
            f"{tuple(target.shape)}"
        )
    target_region = target > target_threshold
    if not bool(target_region.any()):
        raise ValueError("target has no pixels above target_threshold")
    values = field[target_region]
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
    }


def doc_array_to_reaction_progress_for_reporting(
    required_doc: np.ndarray,
) -> np.ndarray:
    """Map DoC requirements to R for output; exact DoC=1 is reported as infinity."""

    values = np.asarray(required_doc, dtype=float)
    result = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    below_one = finite & (values < 1.0)
    result[below_one] = -np.log1p(-values[below_one])
    result[finite & (values == 1.0)] = np.inf
    return result


def json_safe_float_list(values: np.ndarray) -> list[float | None]:
    """Represent non-finite diagnostic values explicitly as JSON null."""

    return [float(value) if math.isfinite(float(value)) else None for value in values]


def normalized_field_region_summary(
    field: torch.Tensor,
    target: torch.Tensor,
    target_threshold: float,
    *,
    label: str,
) -> dict[str, float]:
    """Summarize a normalized initialization field on target and background."""

    if tuple(field.shape) != tuple(target.shape):
        raise ValueError(
            f"{label} shape {tuple(field.shape)} does not match target shape "
            f"{tuple(target.shape)}"
        )
    target_region = target > target_threshold
    if not bool(target_region.any()):
        raise ValueError("target has no pixels above target_threshold")
    outside_region = ~target_region
    target_values = field[target_region]
    outside_mean = (
        float(field[outside_region].mean()) if bool(outside_region.any()) else 0.0
    )
    return {
        "target_mean": float(target_values.mean()),
        "target_min": float(target_values.min()),
        "target_max": float(target_values.max()),
        "outside_mean": outside_mean,
    }


def resolve_control_timing(
    *, total_time_s: float | None, control_steps: int | None, control_dt_s: float
) -> tuple[int, float]:
    """Resolve a physical duration without independently hard-coding step count."""

    if not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("control_dt_s must be finite and positive")
    if total_time_s is not None and control_steps is not None:
        raise ValueError("provide either total_time_s or control_steps, not both")
    if total_time_s is None and control_steps is None:
        total_time_s = 20.0
    if total_time_s is not None:
        if not math.isfinite(total_time_s) or total_time_s <= 0:
            raise ValueError("total_time_s must be finite and positive")
        raw_steps = total_time_s / control_dt_s
        derived_steps = int(round(raw_steps))
        if derived_steps < 1 or not math.isclose(
            derived_steps * control_dt_s,
            total_time_s,
            rel_tol=0.0,
            abs_tol=max(1e-10, 1e-9 * total_time_s),
        ):
            raise ValueError(
                f"total time {total_time_s:g} s is not compatible with control "
                f"interval {control_dt_s:g} s"
            )
        return derived_steps, float(total_time_s)
    if not isinstance(control_steps, int) or control_steps < 1:
        raise ValueError(f"control_steps must be at least 1, got {control_steps}")
    return control_steps, control_steps * control_dt_s


def parse_checkpoints(specification: str | None) -> tuple[tuple[float, float], ...]:
    """Parse ``time:DoC,time:DoC`` without inventing intermediate targets."""

    if specification is None or not specification.strip():
        raise ValueError("checkpoint mode requires --checkpoints time:DoC,...")
    checkpoints: list[tuple[float, float]] = []
    for item in specification.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                f"invalid checkpoint {item!r}; expected time:DoC,time:DoC"
            )
        try:
            time_s, required_doc = (float(field.strip()) for field in fields)
        except ValueError as error:
            raise ValueError(f"checkpoint {item!r} must contain numbers") from error
        if not math.isfinite(time_s) or time_s <= 0:
            raise ValueError("checkpoint times must be finite and positive")
        if not math.isfinite(required_doc) or not 0 <= required_doc <= 1:
            raise ValueError("checkpoint DoC values must be finite and in [0,1]")
        checkpoints.append((time_s, required_doc))
    times = [time_s for time_s, _ in checkpoints]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError("checkpoint times must be strictly increasing and unique")
    return tuple(checkpoints)


def validate_checkpoint_schedule(
    checkpoints: tuple[tuple[float, float], ...],
    *,
    control_dt_s: float,
    total_time_s: float,
) -> None:
    """Require exact absolute control-grid alignment and in-run checkpoints."""

    for time_s, _ in checkpoints:
        control_index = round(time_s / control_dt_s)
        aligned_time = control_index * control_dt_s
        if not math.isclose(time_s, aligned_time, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"checkpoint time {time_s:g} s is not aligned to control_dt="
                f"{control_dt_s:g} s; checkpoints are never rounded"
            )
        if time_s > total_time_s + 1e-9:
            raise ValueError(
                f"checkpoint time {time_s:g} s exceeds total time {total_time_s:g} s"
            )


def checkpoint_horizon_warnings(
    checkpoints: tuple[tuple[float, float], ...],
    *,
    horizon: int,
    control_dt_s: float,
) -> list[str]:
    """Return visibility warnings without overriding the requested horizon."""

    lookahead_s = horizon * control_dt_s
    messages = []
    for (first_time, _), (second_time, _) in zip(checkpoints, checkpoints[1:]):
        spacing = second_time - first_time
        if spacing > lookahead_s + 1e-9:
            messages.append(
                f"checkpoint spacing {spacing:g} s ({first_time:g}->{second_time:g} s) "
                f"exceeds the {lookahead_s:g} s prediction horizon"
            )
    return messages


def resolve_tracking_configuration(
    args: argparse.Namespace,
    *,
    control_dt_s: float,
    total_time_s: float,
) -> tuple[TrackingSpecification, DoCReferenceCurve | None, tuple[str, ...]]:
    """Resolve and validate all target-side tracking choices before optimization."""

    mode = args.tracking_mode
    point_weights = parse_float_list(args.point_weights, label="--point-weights")
    common = {
        "point_weights": point_weights,
        "spatial_definition": args.tracking_spatial_definition,
        "tracking_loss": args.tracking_loss,
        "tracking_variable": args.tracking_variable,
        "huber_delta": args.huber_delta,
    }
    warnings: list[str] = []
    if mode == "checkpoints":
        if args.doc_reference is not None:
            raise TrackingConfigurationError(
                "checkpoint mode must not specify --doc-reference"
            )
        if args.tracking_times or args.num_tracking_points is not None:
            raise TrackingConfigurationError(
                "checkpoint mode uses --checkpoints, not sampled-curve time options"
            )
        specification = TrackingSpecification.checkpoints(
            parse_tracking_checkpoints(args.checkpoints), **common
        )
        reference_curve = None
    else:
        if args.checkpoints is not None:
            raise TrackingConfigurationError(
                "--checkpoints is valid only with --tracking-mode checkpoints"
            )
        reference_curve = load_doc_reference(
            args.doc_reference or DEFAULT_REFERENCE_ID,
            path=args.reference_artifact,
            curve_model=args.curve_model,
        )
        if mode == "curve":
            if (
                args.tracking_times
                or args.num_tracking_points is not None
                or args.tracking_start is not None
                or args.tracking_end is not None
                or point_weights
            ):
                raise TrackingConfigurationError(
                    "dense curve mode must not define sparse times or point weights"
                )
            specification = TrackingSpecification.curve(
                reference_curve,
                spatial_definition=args.tracking_spatial_definition,
                tracking_loss=args.tracking_loss,
                tracking_variable=args.tracking_variable,
                huber_delta=args.huber_delta,
            )
        else:
            explicit_times = parse_float_list(
                args.tracking_times, label="--tracking-times"
            )
            times, sampled_warnings = resolve_sampled_tracking_times(
                explicit_times_s=explicit_times,
                count=args.num_tracking_points,
                start_s=args.tracking_start,
                end_s=args.tracking_end,
            )
            warnings.extend(sampled_warnings)
            specification = TrackingSpecification.sampled_curve(
                reference_curve, times, **common
            )
    warnings.extend(
        specification.validate_runtime(control_dt_s, total_time_s, args.horizon)
    )
    return specification, reference_curve, tuple(warnings)


def calculate_checkpoint_tracking(
    control_times_s: np.ndarray,
    mean_target_doc: np.ndarray,
    min_target_doc: np.ndarray,
    max_target_doc: np.ndarray,
    std_target_doc: np.ndarray,
    mean_target_reaction_progress: np.ndarray,
    min_target_reaction_progress: np.ndarray,
    max_target_reaction_progress: np.ndarray,
    std_target_reaction_progress: np.ndarray,
    target_o2_mean: np.ndarray,
    target_o2_min: np.ndarray,
    target_o2_max: np.ndarray,
    target_o2_std: np.ndarray,
    checkpoints: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    """Extract target-region DoC, reaction-progress, and O2 point statistics."""

    requested = np.asarray([value for _, value in checkpoints], dtype=float)
    checkpoint_times = np.asarray([time_s for time_s, _ in checkpoints], dtype=float)
    achieved = np.empty_like(requested)
    minimum = np.empty_like(requested)
    maximum = np.empty_like(requested)
    standard_deviation = np.empty_like(requested)
    achieved_reaction_progress = np.empty_like(requested)
    minimum_reaction_progress = np.empty_like(requested)
    maximum_reaction_progress = np.empty_like(requested)
    reaction_progress_standard_deviation = np.empty_like(requested)
    o2_mean = np.empty_like(requested)
    o2_minimum = np.empty_like(requested)
    o2_maximum = np.empty_like(requested)
    o2_standard_deviation = np.empty_like(requested)
    for index, checkpoint_time in enumerate(checkpoint_times):
        matches = np.flatnonzero(
            np.isclose(control_times_s, checkpoint_time, rtol=0.0, atol=1e-9)
        )
        if matches.size != 1:
            raise AssertionError(
                f"checkpoint {checkpoint_time:g} s matched {matches.size} applied times"
            )
        matched_index = int(matches[0])
        achieved[index] = mean_target_doc[matched_index]
        minimum[index] = min_target_doc[matched_index]
        maximum[index] = max_target_doc[matched_index]
        standard_deviation[index] = std_target_doc[matched_index]
        achieved_reaction_progress[index] = mean_target_reaction_progress[matched_index]
        minimum_reaction_progress[index] = min_target_reaction_progress[matched_index]
        maximum_reaction_progress[index] = max_target_reaction_progress[matched_index]
        reaction_progress_standard_deviation[index] = (
            std_target_reaction_progress[matched_index]
        )
        o2_mean[index] = target_o2_mean[matched_index]
        o2_minimum[index] = target_o2_min[matched_index]
        o2_maximum[index] = target_o2_max[matched_index]
        o2_standard_deviation[index] = target_o2_std[matched_index]
    errors = achieved - requested
    requested_reaction_progress = doc_array_to_reaction_progress_for_reporting(
        requested
    )
    reaction_progress_errors = (
        achieved_reaction_progress - requested_reaction_progress
    )
    finite_reaction_errors = reaction_progress_errors[
        np.isfinite(reaction_progress_errors)
    ]
    return {
        "times_s": checkpoint_times,
        "requested_doc": requested,
        "achieved_doc": achieved,
        "mean_target_doc": achieved,
        "min_target_doc": minimum,
        "max_target_doc": maximum,
        "std_target_doc": standard_deviation,
        "lower_error": achieved - minimum,
        "upper_error": maximum - achieved,
        "errors": errors,
        "requested_reaction_progress": requested_reaction_progress,
        "mean_target_reaction_progress": achieved_reaction_progress,
        "min_target_reaction_progress": minimum_reaction_progress,
        "max_target_reaction_progress": maximum_reaction_progress,
        "std_target_reaction_progress": reaction_progress_standard_deviation,
        "reaction_progress_error": reaction_progress_errors,
        "target_o2_mean": o2_mean,
        "target_o2_min": o2_minimum,
        "target_o2_max": o2_maximum,
        "target_o2_std": o2_standard_deviation,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "reaction_progress_rmse": (
            float(np.sqrt(np.mean(np.square(finite_reaction_errors))))
            if finite_reaction_errors.size
            else None
        ),
        "reaction_progress_mae": (
            float(np.mean(np.abs(finite_reaction_errors)))
            if finite_reaction_errors.size
            else None
        ),
        "reaction_progress_max_absolute_error": (
            float(np.max(np.abs(finite_reaction_errors)))
            if finite_reaction_errors.size
            else None
        ),
    }


def save_tracking_points_csv(
    tracking: dict[str, object],
    path: Path,
    *,
    tracking_variable: str = "doc",
) -> None:
    """Save sparse/checkpoint target-region statistics in a tidy table."""

    columns = (
        "tracking_variable",
        "time_s",
        "requested_target_doc",
        "mean_target_doc",
        "min_target_doc",
        "max_target_doc",
        "std_target_doc",
        "lower_error",
        "upper_error",
        "mean_error",
        "requested_reaction_progress",
        "mean_target_reaction_progress",
        "min_target_reaction_progress",
        "max_target_reaction_progress",
        "std_target_reaction_progress",
        "reaction_progress_error",
        "target_o2_mean_mj_cm2",
        "target_o2_min_mj_cm2",
        "target_o2_max_mj_cm2",
        "target_o2_std_mj_cm2",
    )
    arrays = (
        tracking["times_s"],
        tracking["requested_doc"],
        tracking["mean_target_doc"],
        tracking["min_target_doc"],
        tracking["max_target_doc"],
        tracking["std_target_doc"],
        tracking["lower_error"],
        tracking["upper_error"],
        tracking["errors"],
        tracking["requested_reaction_progress"],
        tracking["mean_target_reaction_progress"],
        tracking["min_target_reaction_progress"],
        tracking["max_target_reaction_progress"],
        tracking["std_target_reaction_progress"],
        tracking["reaction_progress_error"],
        tracking["target_o2_mean"],
        tracking["target_o2_min"],
        tracking["target_o2_max"],
        tracking["target_o2_std"],
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for values in zip(*arrays):
            row = dict(
                zip(columns[1:], (float(value) for value in values))
            )
            row["tracking_variable"] = tracking_variable
            writer.writerow(row)


def save_component_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Save the complete component table without flooding terminal output."""

    columns = (
        "component_id",
        "area_pixels",
        "area_um2",
        "mean_final_doc",
        "p05_final_doc",
        "cured_fraction",
        "undercure_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tracking_points_metadata(
    tracking: dict[str, object],
    *,
    tracking_variable: str,
    table_path: str | None,
) -> dict[str, object]:
    """Convert shared tracking-point statistics to JSON-safe metadata."""

    return {
        "tracking_variable": tracking_variable,
        "times_s": tracking["times_s"].tolist(),
        "requested_doc": tracking["requested_doc"].tolist(),
        "achieved_doc": tracking["achieved_doc"].tolist(),
        "mean_target_doc": tracking["mean_target_doc"].tolist(),
        "min_target_doc": tracking["min_target_doc"].tolist(),
        "max_target_doc": tracking["max_target_doc"].tolist(),
        "std_target_doc": tracking["std_target_doc"].tolist(),
        "lower_error": tracking["lower_error"].tolist(),
        "upper_error": tracking["upper_error"].tolist(),
        "errors": tracking["errors"].tolist(),
        "requested_reaction_progress": json_safe_float_list(
            tracking["requested_reaction_progress"]
        ),
        "mean_target_reaction_progress": tracking[
            "mean_target_reaction_progress"
        ].tolist(),
        "min_target_reaction_progress": tracking[
            "min_target_reaction_progress"
        ].tolist(),
        "max_target_reaction_progress": tracking[
            "max_target_reaction_progress"
        ].tolist(),
        "std_target_reaction_progress": tracking[
            "std_target_reaction_progress"
        ].tolist(),
        "reaction_progress_error": json_safe_float_list(
            tracking["reaction_progress_error"]
        ),
        "reaction_progress_rmse": tracking["reaction_progress_rmse"],
        "reaction_progress_mae": tracking["reaction_progress_mae"],
        "reaction_progress_max_absolute_error": tracking[
            "reaction_progress_max_absolute_error"
        ],
        "target_o2_mean": tracking["target_o2_mean"].tolist(),
        "target_o2_min": tracking["target_o2_min"].tolist(),
        "target_o2_max": tracking["target_o2_max"].tolist(),
        "target_o2_std": tracking["target_o2_std"].tolist(),
        "target_o2_units": "mJ/cm^2",
        "rmse": tracking["rmse"],
        "mae": tracking["mae"],
        "max_absolute_error": tracking["max_absolute_error"],
        "table_path": table_path,
    }


def run_unoptimized_target_mask_baseline(
    *,
    model: AIEModel,
    initial_state: AIEState,
    target_native: torch.Tensor,
    control_times_s: np.ndarray,
    reference_doc: np.ndarray,
    checkpoints: tuple[tuple[float, float], ...],
    tracking_mode: str,
    tracking_variable: str,
    reference_final_doc: float,
    target_threshold: float,
    geometry_threshold: float,
    pixel_pitch_um: float,
    physics_steps_per_control: int,
    output_dir: Path,
    resolution_mode: str,
    physics_condition: str,
    authoritative_reference: aie_reference.ReferenceConfig,
    tracking_specification_metadata: dict[str, object],
) -> TargetMaskBaselineResult:
    """Run and report ``u_k(x,y) = M(x,y)`` with no optimization."""

    baseline_label = "unoptimized repeated target-mask baseline"
    require_native_target(target_native)
    control_times_s = np.asarray(control_times_s, dtype=float)
    reference_doc = np.asarray(reference_doc, dtype=float)
    if (
        control_times_s.ndim != 1
        or control_times_s.size < 1
        or reference_doc.shape != control_times_s.shape
    ):
        raise ValueError(
            "baseline control times and requested DoC must be equal nonempty 1D arrays"
        )
    expected_control_dt_s = model.params.dt * physics_steps_per_control
    expected_times_s = expected_control_dt_s * np.arange(
        1, control_times_s.size + 1, dtype=float
    )
    if not np.allclose(control_times_s, expected_times_s, rtol=0.0, atol=1e-9):
        raise ValueError("baseline must use the MPC physical/control time grid")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_control = target_native.detach().clone()
    save_grayscale(fixed_control, output_dir / "target_mask_baseline.png")

    timeline_lists: dict[str, list[float]] = {
        "mean_target_doc": [],
        "min_target_doc": [],
        "max_target_doc": [],
        "std_target_doc": [],
        "mean_outside_doc": [],
        "mean_target_reaction_progress": [],
        "min_target_reaction_progress": [],
        "max_target_reaction_progress": [],
        "std_target_reaction_progress": [],
        "target_o2_mean": [],
        "target_o2_min": [],
        "target_o2_max": [],
        "target_o2_std": [],
    }
    initial_summaries: dict[str, dict[str, float]] = {}
    doc_frame_paths: list[Path] = []

    def record_baseline_state(frame_index: int, state: AIEState) -> None:
        frame_path = output_dir / f"doc_frame_native_{frame_index:03d}.png"
        save_doc_frame(state.doc, frame_path, tuple(target_native.shape))
        doc_frame_paths.append(frame_path)
        doc_summary = doc_region_summary(
            state.doc, target_native, target_threshold
        )
        o2_summary = target_field_summary(
            state.o2,
            target_native,
            target_threshold,
            label="baseline O2",
        )
        if frame_index == 0:
            initial_summaries["doc"] = doc_summary
            initial_summaries["o2"] = o2_summary
            return
        reaction_summary = target_field_summary(
            state.reaction_progress,
            target_native,
            target_threshold,
            label="baseline reaction_progress",
        )
        timeline_lists["mean_target_doc"].append(doc_summary["mean_target_doc"])
        timeline_lists["min_target_doc"].append(doc_summary["min_target_doc"])
        timeline_lists["max_target_doc"].append(doc_summary["max_target_doc"])
        timeline_lists["std_target_doc"].append(doc_summary["std_target_doc"])
        timeline_lists["mean_outside_doc"].append(doc_summary["mean_outside_doc"])
        timeline_lists["mean_target_reaction_progress"].append(
            reaction_summary["mean"]
        )
        timeline_lists["min_target_reaction_progress"].append(
            reaction_summary["min"]
        )
        timeline_lists["max_target_reaction_progress"].append(
            reaction_summary["max"]
        )
        timeline_lists["std_target_reaction_progress"].append(
            reaction_summary["std"]
        )
        timeline_lists["target_o2_mean"].append(o2_summary["mean"])
        timeline_lists["target_o2_min"].append(o2_summary["min"])
        timeline_lists["target_o2_max"].append(o2_summary["max"])
        timeline_lists["target_o2_std"].append(o2_summary["std"])

    final_state = replay_native_controls(
        model.params,
        tuple(target_native.shape),
        [fixed_control] * int(control_times_s.size),
        physics_steps_per_control,
        model.device,
        authoritative_reference=authoritative_reference,
        state_callback=record_baseline_state,
        native_model=model,
        initial_state=initial_state,
    )
    if set(initial_summaries) != {"doc", "o2"}:
        raise AssertionError("baseline rollout did not report its initialized state")
    if len(doc_frame_paths) != control_times_s.size + 1:
        raise AssertionError("baseline DoC timeline has an unexpected frame count")
    save_gif_from_frames(
        doc_frame_paths, output_dir / "doc_evolution_native.gif"
    )
    save_grayscale(final_state.doc, output_dir / "final_doc_native.png")
    save_grayscale(
        final_state.o2 / max(model.params.o2_inhibition_mj_cm2, 1e-12),
        output_dir / "final_o2_native.png",
    )

    timeline = {
        name: np.asarray(values, dtype=float)
        for name, values in timeline_lists.items()
    }
    if any(values.size != control_times_s.size for values in timeline.values()):
        raise AssertionError("baseline physical timeline has an unexpected length")
    requested_reaction_progress = doc_array_to_reaction_progress_for_reporting(
        reference_doc
    )
    reaction_progress_error = (
        timeline["mean_target_reaction_progress"] - requested_reaction_progress
    )
    if tracking_mode == "curve":
        tracking = temporal_tracking_metrics(
            timeline["mean_target_doc"], reference_doc
        )
        finite_reaction_reference = np.isfinite(requested_reaction_progress)
        reaction_progress_tracking = (
            temporal_tracking_metrics(
                timeline["mean_target_reaction_progress"][finite_reaction_reference],
                requested_reaction_progress[finite_reaction_reference],
            )
            if np.any(finite_reaction_reference)
            else None
        )
        evaluation_points = tuple(
            zip(control_times_s.tolist(), reference_doc.tolist())
        )
    else:
        evaluation_points = checkpoints

    tracking_points = calculate_checkpoint_tracking(
        control_times_s,
        timeline["mean_target_doc"],
        timeline["min_target_doc"],
        timeline["max_target_doc"],
        timeline["std_target_doc"],
        timeline["mean_target_reaction_progress"],
        timeline["min_target_reaction_progress"],
        timeline["max_target_reaction_progress"],
        timeline["std_target_reaction_progress"],
        timeline["target_o2_mean"],
        timeline["target_o2_min"],
        timeline["target_o2_max"],
        timeline["target_o2_std"],
        evaluation_points,
    )
    if tracking_mode != "curve":
        tracking = {
            "rmse": tracking_points["rmse"],
            "mae": tracking_points["mae"],
            "max_absolute_error": tracking_points["max_absolute_error"],
        }
        reaction_progress_tracking = (
            None
            if tracking_points["reaction_progress_rmse"] is None
            else {
                "rmse": tracking_points["reaction_progress_rmse"],
                "mae": tracking_points["reaction_progress_mae"],
                "max_absolute_error": tracking_points[
                    "reaction_progress_max_absolute_error"
                ],
            }
        )

    final_metrics = calculate_final_metrics(
        final_state.doc.detach().cpu().numpy(),
        target_native.detach().cpu().numpy(),
        reference_final_doc,
        target_threshold,
        geometry_threshold,
        pixel_pitch_um,
    )
    component_metrics_path = output_dir / "component_metrics.csv"
    tracking_points_path = output_dir / "doc_tracking_points.csv"
    save_component_metrics_csv(
        final_metrics["component_rows"], component_metrics_path
    )
    save_tracking_points_csv(
        tracking_points,
        tracking_points_path,
        tracking_variable=tracking_variable,
    )
    components_summary = {
        **final_metrics["components"],
        "table_path": component_metrics_path.name,
    }
    holes_summary = {
        **final_metrics["holes"],
        "details": final_metrics["hole_rows"],
    }
    metrics_document = {
        "schema_version": 1,
        "label": baseline_label,
        "optimization_performed": False,
        "control_policy": {
            "definition": "u_k(x,y) = M(x,y)",
            "mask_source": "original normalized native target image",
            "fixed_for_entire_process": True,
            "control_intervals": int(control_times_s.size),
            "mask_path": "target_mask_baseline.png",
        },
        "initial_state_reused_from_mpc_primary_rollout": True,
        "tracking_variable": tracking_variable,
        "tracking_variable_role": "evaluation only; baseline control is independent",
        "tracking_mode": tracking_mode,
        "tracking_specification": tracking_specification_metadata,
        "resolution_mode": resolution_mode,
        "primary_result_grid": "native",
        "forward_model_provenance": {
            "physics_condition": physics_condition,
            "parameters": model.params.provenance_metadata(),
            "reference_config": authoritative_reference.to_metadata(),
            "history_mode": DOC_HISTORY_MODE,
            "history_description": DOC_HISTORY_DESCRIPTION,
        },
        "target": {
            "shape": list(target_native.shape),
            "target_threshold": target_threshold,
            "normalized_min": float(target_native.min()),
            "normalized_max": float(target_native.max()),
        },
        "control_settings": {
            "dt_s": model.params.dt,
            "physics_steps_per_control": physics_steps_per_control,
            "control_dt_s": expected_control_dt_s,
            "control_steps": int(control_times_s.size),
            "total_process_time_s": float(control_times_s[-1]),
        },
        "final_geometry_reference": {"doc": reference_final_doc},
        "target_region_doc_timeline": {
            "times_s": control_times_s.tolist(),
            "requested_target_doc": json_safe_float_list(reference_doc),
            "mean_target_doc": timeline["mean_target_doc"].tolist(),
            "min_target_doc": timeline["min_target_doc"].tolist(),
            "max_target_doc": timeline["max_target_doc"].tolist(),
            "std_target_doc": timeline["std_target_doc"].tolist(),
            "lower_error": (
                timeline["mean_target_doc"] - timeline["min_target_doc"]
            ).tolist(),
            "upper_error": (
                timeline["max_target_doc"] - timeline["mean_target_doc"]
            ).tolist(),
            "mean_outside_doc": timeline["mean_outside_doc"].tolist(),
        },
        "target_region_reaction_progress_timeline": {
            "times_s": control_times_s.tolist(),
            "requested_reaction_progress": json_safe_float_list(
                requested_reaction_progress
            ),
            "mean_target_reaction_progress": timeline[
                "mean_target_reaction_progress"
            ].tolist(),
            "min_target_reaction_progress": timeline[
                "min_target_reaction_progress"
            ].tolist(),
            "max_target_reaction_progress": timeline[
                "max_target_reaction_progress"
            ].tolist(),
            "std_target_reaction_progress": timeline[
                "std_target_reaction_progress"
            ].tolist(),
            "reaction_progress_error": json_safe_float_list(
                reaction_progress_error
            ),
        },
        "target_region_o2_timeline_mj_cm2": {
            "units": "mJ/cm^2",
            "times_s": control_times_s.tolist(),
            "target_o2_mean": timeline["target_o2_mean"].tolist(),
            "target_o2_min": timeline["target_o2_min"].tolist(),
            "target_o2_max": timeline["target_o2_max"].tolist(),
            "target_o2_std": timeline["target_o2_std"].tolist(),
        },
        "tracking_evaluation": tracking,
        "temporal_tracking": tracking if tracking_mode == "curve" else None,
        "temporal_reaction_progress_tracking": (
            reaction_progress_tracking if tracking_mode == "curve" else None
        ),
        "tracking_points": tracking_points_metadata(
            tracking_points,
            tracking_variable=tracking_variable,
            table_path=tracking_points_path.name,
        ),
        "sparse_tracking": (
            None
            if tracking_mode == "curve"
            else tracking_points_metadata(
                tracking_points,
                tracking_variable=tracking_variable,
                table_path=tracking_points_path.name,
            )
        ),
        "soft_doc": final_metrics["soft_doc"],
        "geometry": final_metrics["geometry"],
        "boundary": final_metrics["boundary"],
        "components": components_summary,
        "holes": holes_summary,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TargetMaskBaselineResult(
        final_state=final_state,
        timeline=timeline,
        initial_target_doc_summary=initial_summaries["doc"],
        initial_target_o2_summary=initial_summaries["o2"],
        tracking=tracking,
        tracking_points=tracking_points,
        final_metrics=final_metrics,
        metrics_document=metrics_document,
        output_dir=output_dir,
    )


def build_baseline_comparison(
    *,
    mpc_control_times_s: np.ndarray,
    mpc_reference_doc: np.ndarray,
    mpc_target_doc: np.ndarray,
    mpc_final_target_summary: dict[str, float],
    mpc_tracking: dict[str, float],
    mpc_tracking_points: dict[str, object] | None,
    mpc_final_metrics: dict[str, object],
    baseline: TargetMaskBaselineResult,
) -> dict[str, object]:
    """Build like-for-like MPC/baseline values from shared metric definitions."""

    if mpc_tracking_points is None:
        comparison_time_s = float(mpc_control_times_s[-1])
        requested_doc = float(mpc_reference_doc[-1])
        mpc_checkpoint_mean_doc = float(mpc_target_doc[-1])
    else:
        comparison_time_s = float(mpc_tracking_points["times_s"][-1])
        requested_doc = float(mpc_tracking_points["requested_doc"][-1])
        mpc_checkpoint_mean_doc = float(
            mpc_tracking_points["mean_target_doc"][-1]
        )
    baseline_points = baseline.tracking_points
    baseline_checkpoint_mean_doc = float(
        baseline_points["mean_target_doc"][-1]
    )

    mpc_components = mpc_final_metrics["components"]
    baseline_components = baseline.final_metrics["components"]
    mpc_geometry = mpc_final_metrics["geometry"]
    baseline_geometry = baseline.final_metrics["geometry"]
    baseline_doc_timeline = baseline.timeline
    comparison_metrics = {
        "checkpoint_mean_doc": {
            "mpc": mpc_checkpoint_mean_doc,
            "baseline": baseline_checkpoint_mean_doc,
        },
        "checkpoint_absolute_error": {
            "mpc": abs(mpc_checkpoint_mean_doc - requested_doc),
            "baseline": abs(baseline_checkpoint_mean_doc - requested_doc),
        },
        "tracking_doc_rmse": {
            "mpc": mpc_tracking["rmse"],
            "baseline": baseline.tracking["rmse"],
        },
        "target_doc_rmse": {
            "mpc": mpc_final_metrics["soft_doc"]["target_region_rmse"],
            "baseline": baseline.final_metrics["soft_doc"]["target_region_rmse"],
        },
        "target_doc_std": {
            "mpc": mpc_final_target_summary["std_target_doc"],
            "baseline": float(baseline_doc_timeline["std_target_doc"][-1]),
        },
        "component_mean_doc_min": {
            "mpc": mpc_components["component_mean_doc_min"],
            "baseline": baseline_components["component_mean_doc_min"],
        },
        "component_mean_doc_max": {
            "mpc": mpc_components["component_mean_doc_max"],
            "baseline": baseline_components["component_mean_doc_max"],
        },
        "mean_outside_doc": {
            "mpc": mpc_final_target_summary["mean_outside_doc"],
            "baseline": float(baseline_doc_timeline["mean_outside_doc"][-1]),
        },
        "iou": {
            "mpc": mpc_geometry["iou"],
            "baseline": baseline_geometry["iou"],
        },
        "dice": {
            "mpc": mpc_geometry["dice"],
            "baseline": baseline_geometry["dice"],
        },
    }
    return {
        "enabled": True,
        "label": "unoptimized repeated target-mask baseline",
        "baseline_metrics_path": "baseline_target_mask/metrics.json",
        "comparison_point": {
            "time_s": comparison_time_s,
            "requested_doc": requested_doc,
            "selection": (
                "final dense-curve point"
                if mpc_tracking_points is None
                else "last sparse tracking point"
            ),
        },
        "metrics": comparison_metrics,
    }


def print_baseline_comparison(comparison: dict[str, object]) -> None:
    """Print a compact MPC versus fixed-target benchmark table."""

    labels = {
        "checkpoint_mean_doc": "checkpoint mean DoC",
        "checkpoint_absolute_error": "checkpoint abs error",
        "tracking_doc_rmse": "tracking DoC RMSE",
        "target_doc_rmse": "target DoC RMSE",
        "target_doc_std": "target DoC std",
        "component_mean_doc_min": "component mean DoC min",
        "component_mean_doc_max": "component mean DoC max",
        "mean_outside_doc": "mean outside DoC",
        "iou": "IoU",
        "dice": "Dice",
    }

    def format_value(value: object) -> str:
        return "undefined" if value is None else f"{float(value):.6f}"

    print("\n=== MPC vs unoptimized target-mask baseline ===")
    print(f"{'metric':30s} {'MPC':>12s} {'baseline':>12s}")
    metrics = comparison["metrics"]
    for key, label in labels.items():
        values = metrics[key]
        print(
            f"{label:30s} {format_value(values['mpc']):>12s} "
            f"{format_value(values['baseline']):>12s}"
        )


def save_initializer_component_metrics_csv(
    rows: list[dict[str, object]], path: Path
) -> None:
    """Save component-wise cold-start optics for diagnostics only."""

    columns = (
        "component_id",
        "pixel_count",
        "projector_mask_mean",
        "local_normalized_intensity_mean",
        "local_normalized_intensity_min",
        "local_normalized_intensity_max",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _add_target_o2_axis(
    axis: object,
    times_s: np.ndarray,
    target_o2_mean: np.ndarray,
    target_o2_min: np.ndarray,
    target_o2_max: np.ndarray,
) -> object:
    """Add consistently styled physical O2 diagnostics to a DoC axis."""

    o2_color = "tab:green"
    o2_axis = axis.twinx()
    o2_axis.fill_between(
        times_s,
        target_o2_min,
        target_o2_max,
        color=o2_color,
        alpha=0.10,
        label="target O2 min-max range",
    )
    o2_axis.plot(
        times_s,
        target_o2_mean,
        color=o2_color,
        linestyle="--",
        linewidth=1.7,
        label="mean target O2",
    )
    o2_axis.set_ylabel("Target O2 state (mJ/cm$^2$)", color=o2_color)
    o2_axis.tick_params(axis="y", colors=o2_color)
    o2_axis.spines["right"].set_color(o2_color)
    o2_upper = max(float(np.max(target_o2_max)), 0.0)
    o2_axis.set_ylim(0.0, max(1.05 * o2_upper, 1e-6))
    return o2_axis


def _place_combined_tracking_legend(figure: object, axis: object, o2_axis: object) -> None:
    """Place one compact legend above the axes so it does not obscure data."""

    handles, labels = axis.get_legend_handles_labels()
    o2_handles, o2_labels = o2_axis.get_legend_handles_labels()
    figure.legend(
        handles + o2_handles,
        labels + o2_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))


def save_tracking_plot(
    reference_curve: DoCReferenceCurve,
    total_time_s: float,
    control_times_s: np.ndarray,
    actual_target_doc: np.ndarray,
    min_target_doc: np.ndarray,
    max_target_doc: np.ndarray,
    target_o2_mean: np.ndarray,
    target_o2_min: np.ndarray,
    target_o2_max: np.ndarray,
    path: Path,
    *,
    initial_target_doc_summary: dict[str, float],
    initial_target_o2_summary: dict[str, float],
    baseline_target_doc: np.ndarray | None = None,
    baseline_initial_target_doc_summary: dict[str, float] | None = None,
) -> None:
    """Save experimental-reference versus closed-loop mean target DoC."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_time = np.linspace(0.0, total_time_s, max(401, int(total_time_s / 0.01) + 1))
    reference_doc = np.asarray(reference_curve.at(reference_time))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        reference_time,
        reference_doc,
        color="tab:gray",
        linestyle=":",
        label="experimental reference",
        linewidth=2,
    )
    achieved_times = np.r_[0.0, control_times_s]
    achieved_mean = np.r_[
        initial_target_doc_summary["mean_target_doc"], actual_target_doc
    ]
    achieved_min = np.r_[
        initial_target_doc_summary["min_target_doc"], min_target_doc
    ]
    achieved_max = np.r_[
        initial_target_doc_summary["max_target_doc"], max_target_doc
    ]
    plotted_o2_mean = np.r_[initial_target_o2_summary["mean"], target_o2_mean]
    plotted_o2_min = np.r_[initial_target_o2_summary["min"], target_o2_min]
    plotted_o2_max = np.r_[initial_target_o2_summary["max"], target_o2_max]
    axis.fill_between(
        achieved_times,
        achieved_min,
        achieved_max,
        color="tab:orange",
        alpha=0.22,
        label="MPC target-region min-max range",
    )
    axis.plot(
        achieved_times,
        achieved_mean,
        "o-",
        color="tab:blue",
        markersize=3,
        label="optimized MPC mean target DoC",
    )
    if baseline_target_doc is not None:
        if baseline_initial_target_doc_summary is None:
            raise ValueError("baseline plot requires its initialized DoC summary")
        baseline_target_doc = np.asarray(baseline_target_doc, dtype=float)
        if baseline_target_doc.shape != np.asarray(actual_target_doc).shape:
            raise ValueError("baseline and MPC DoC timelines must have equal length")
        axis.plot(
            achieved_times,
            np.r_[
                baseline_initial_target_doc_summary["mean_target_doc"],
                baseline_target_doc,
            ],
            color="tab:purple",
            linestyle="--",
            linewidth=2.0,
            label="unoptimized repeated target-mask baseline",
        )
    axis.set_xlim(0.0, total_time_s)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Absolute process time (s)")
    axis.set_ylabel("Degree of conversion")
    o2_axis = _add_target_o2_axis(
        axis,
        achieved_times,
        plotted_o2_mean,
        plotted_o2_min,
        plotted_o2_max,
    )
    title = reference_curve.metadata["condition_label"].replace("cm^2", r"cm$^2$")
    axis.set_title(f"DoC trajectory tracking: {title}")
    axis.grid(alpha=0.25)
    _place_combined_tracking_legend(figure, axis, o2_axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_checkpoint_tracking_plot(
    total_time_s: float,
    control_times_s: np.ndarray,
    actual_target_doc: np.ndarray,
    min_target_doc: np.ndarray,
    max_target_doc: np.ndarray,
    target_o2_mean: np.ndarray,
    target_o2_min: np.ndarray,
    target_o2_max: np.ndarray,
    checkpoint_tracking: dict[str, object],
    path: Path,
    *,
    initial_target_doc_summary: dict[str, float],
    initial_target_o2_summary: dict[str, float],
    marker_label: str = "explicit target checkpoints",
    title: str = "Checkpoint DoC tracking",
    baseline_target_doc: np.ndarray | None = None,
    baseline_initial_target_doc_summary: dict[str, float] | None = None,
) -> None:
    """Plot simulated DoC continuously and sparse requirements only as markers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    plotted_times = np.r_[0.0, control_times_s]
    plotted_doc_mean = np.r_[
        initial_target_doc_summary["mean_target_doc"], actual_target_doc
    ]
    plotted_doc_min = np.r_[
        initial_target_doc_summary["min_target_doc"], min_target_doc
    ]
    plotted_doc_max = np.r_[
        initial_target_doc_summary["max_target_doc"], max_target_doc
    ]
    plotted_o2_mean = np.r_[initial_target_o2_summary["mean"], target_o2_mean]
    plotted_o2_min = np.r_[initial_target_o2_summary["min"], target_o2_min]
    plotted_o2_max = np.r_[initial_target_o2_summary["max"], target_o2_max]
    axis.plot(
        plotted_times,
        plotted_doc_mean,
        "o-",
        color="tab:blue",
        markersize=3,
        label="optimized MPC mean target DoC",
    )
    axis.fill_between(
        plotted_times,
        plotted_doc_min,
        plotted_doc_max,
        color="tab:orange",
        alpha=0.12,
        label="MPC target-region min-max range",
    )
    if baseline_target_doc is not None:
        if baseline_initial_target_doc_summary is None:
            raise ValueError("baseline plot requires its initialized DoC summary")
        baseline_target_doc = np.asarray(baseline_target_doc, dtype=float)
        if baseline_target_doc.shape != np.asarray(actual_target_doc).shape:
            raise ValueError("baseline and MPC DoC timelines must have equal length")
        axis.plot(
            plotted_times,
            np.r_[
                baseline_initial_target_doc_summary["mean_target_doc"],
                baseline_target_doc,
            ],
            color="tab:purple",
            linestyle="--",
            linewidth=2.0,
            label="unoptimized repeated target-mask baseline",
        )
    axis.scatter(
        checkpoint_tracking["times_s"],
        checkpoint_tracking["requested_doc"],
        s=90,
        marker="X",
        color="tab:red",
        zorder=5,
        label=marker_label,
    )
    axis.errorbar(
        checkpoint_tracking["times_s"],
        checkpoint_tracking["mean_target_doc"],
        yerr=np.vstack(
            (
                checkpoint_tracking["lower_error"],
                checkpoint_tracking["upper_error"],
            )
        ),
        fmt="o",
        markersize=5,
        capsize=4,
        color="tab:orange",
        ecolor="tab:orange",
        zorder=4,
        label="MPC achieved mean with target min-max range",
    )
    axis.set_xlim(0.0, total_time_s)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Absolute process time (s)")
    axis.set_ylabel("Degree of conversion")
    o2_axis = _add_target_o2_axis(
        axis,
        plotted_times,
        plotted_o2_mean,
        plotted_o2_min,
        plotted_o2_max,
    )
    axis.set_title(title)
    axis.grid(alpha=0.25)
    _place_combined_tracking_legend(figure, axis, o2_axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "undefined" if value is None else f"{value:.6f}{suffix}"


def print_metrics_summary(
    *,
    reference_id: str,
    tracking_mode: str,
    tracking_variable: str,
    total_time_s: float,
    tracking: dict[str, float],
    checkpoint_tracking: dict[str, object] | None,
    reference_final_doc: float,
    target_mean_doc: float,
    target_min_doc: float,
    target_max_doc: float,
    target_std_doc: float,
    outside_mean_doc: float,
    final_metrics: dict[str, object],
) -> None:
    """Print the compact terminal summary for the primary native result."""

    soft = final_metrics["soft_doc"]
    geometry = final_metrics["geometry"]
    boundary = final_metrics["boundary"]
    components = final_metrics["components"]
    holes = final_metrics["holes"]
    if tracking_mode != "curve":
        assert checkpoint_tracking is not None
        heading = (
            "Checkpoint tracking summary"
            if tracking_mode == "checkpoints"
            else "Sampled-curve tracking summary"
        )
        reference_type = (
            "explicit target checkpoints"
            if tracking_mode == "checkpoints"
            else "selected curve sampled at explicit absolute times"
        )
        print(f"\n=== {heading} ===")
        print(f"reference type:            {reference_type}")
        print(f"tracking variable:         {tracking_variable}")
        print(f"physics condition:         {reference_id}")
        print(f"total time:                {total_time_s:.2f} s")
        print("time    requested      mean       min       max       std     mean error")
        for time_s, requested, achieved, minimum, maximum, std, error in zip(
            checkpoint_tracking["times_s"],
            checkpoint_tracking["requested_doc"],
            checkpoint_tracking["mean_target_doc"],
            checkpoint_tracking["min_target_doc"],
            checkpoint_tracking["max_target_doc"],
            checkpoint_tracking["std_target_doc"],
            checkpoint_tracking["errors"],
        ):
            print(
                f"{time_s:5.2f}   {requested:9.4f}  {achieved:8.4f}  "
                f"{minimum:8.4f}  {maximum:8.4f}  {std:8.4f}  {error:+10.4f}"
            )
        print(f"sparse-point RMSE:         {tracking['rmse']:.6f}")
        print(f"sparse-point MAE:          {tracking['mae']:.6f}")
        print(f"sparse max abs error:      {tracking['max_absolute_error']:.6f}")
        print("time     requested R    mean R     min R     max R     std R    R error    O2 mean/min/max")
        for values in zip(
            checkpoint_tracking["times_s"],
            checkpoint_tracking["requested_reaction_progress"],
            checkpoint_tracking["mean_target_reaction_progress"],
            checkpoint_tracking["min_target_reaction_progress"],
            checkpoint_tracking["max_target_reaction_progress"],
            checkpoint_tracking["std_target_reaction_progress"],
            checkpoint_tracking["reaction_progress_error"],
            checkpoint_tracking["target_o2_mean"],
            checkpoint_tracking["target_o2_min"],
            checkpoint_tracking["target_o2_max"],
        ):
            time_s, requested_r, mean_r, min_r, max_r, std_r, error_r, o2_mean, o2_min, o2_max = values
            print(
                f"{time_s:5.2f}  {requested_r:12.6f}  {mean_r:8.4f}  "
                f"{min_r:8.4f}  {max_r:8.4f}  {std_r:8.4f}  {error_r:+9.4f}  "
                f"{o2_mean:.4f}/{o2_min:.4f}/{o2_max:.4f}"
            )
    else:
        print("\n=== Closed-loop tracking ===")
        print(f"tracking variable:         {tracking_variable}")
        print(f"reference ID:              {reference_id}")
        print(f"total time:                {total_time_s:.2f} s")
        print(f"tracking RMSE:             {tracking['rmse']:.6f}")
        print(f"tracking MAE:              {tracking['mae']:.6f}")
        print(f"maximum absolute error:    {tracking['max_absolute_error']:.6f}")
    print("\n=== Final DoC ===")
    print(f"reference final DoC:       {reference_final_doc:.6f}")
    print(f"mean target DoC:           {target_mean_doc:.6f}")
    print(f"min target DoC:            {target_min_doc:.6f}")
    print(f"max target DoC:            {target_max_doc:.6f}")
    print(f"std target DoC:            {target_std_doc:.6f}")
    print(f"mean outside DoC:          {outside_mean_doc:.6f}")
    print(f"full-image DoC RMSE:       {soft['full_image_rmse']:.6f}")
    print(f"target DoC RMSE:           {soft['target_region_rmse']:.6f}")
    print(f"outside DoC RMSE:          {soft['outside_region_rmse']:.6f}")
    print(f"full-image DoC MAE:        {soft['full_image_mae']:.6f}")
    print(f"\n=== Geometry @ threshold {geometry['threshold']:.2f} ===")
    print(f"IoU:                       {geometry['iou']:.6f}")
    print(f"Dice:                      {geometry['dice']:.6f}")
    print(f"precision:                 {geometry['precision']:.6f}")
    print(f"recall:                    {geometry['recall']:.6f}")
    print(f"undercure fraction:        {geometry['undercure_fraction']:.6f}")
    print(
        "overcure / target area:    "
        f"{geometry['overcure_fraction_target_normalized']:.6f}"
    )
    print(f"overcure / image area:     {geometry['overcure_fraction_image']:.6f}")
    print(f"area error:                {geometry['area_error_fraction']:.6f}")
    counts = geometry["pixel_counts"]
    print(
        "pixels target/cured/TP/FP/FN: "
        f"{counts['target_area']}/{counts['cured_area']}/{counts['true_positive']}/"
        f"{counts['false_positive']}/{counts['false_negative']}"
    )
    print(f"threshold note:            {GEOMETRY_THRESHOLD_NOTE}")
    print("\n=== Boundary ===")
    print(
        "mean symmetric error:      "
        f"{_format_optional(boundary['mean_symmetric_distance_um'], ' um')}"
    )
    print(
        "mean symmetric error:      "
        f"{_format_optional(boundary['mean_symmetric_distance_px'], ' px')}"
    )
    print(
        "p95 symmetric error:       "
        f"{_format_optional(boundary['p95_symmetric_distance_um'], ' um')}"
    )
    print("\n=== Features ===")
    print(f"target components:         {components['count']}")
    print(
        "component mean DoC mean/min/max: "
        f"{_format_optional(components['component_mean_doc_mean'])}/"
        f"{_format_optional(components['component_mean_doc_min'])}/"
        f"{_format_optional(components['component_mean_doc_max'])}"
    )
    print(
        "worst component mean DoC:  "
        f"{_format_optional(components['worst_mean_doc'])}"
    )
    print(
        "worst component cured:     "
        f"{_format_optional(components['worst_cured_fraction'])}"
    )
    print(
        "worst component undercure: "
        f"{_format_optional(components['worst_undercure_fraction'])}"
    )
    print(f"holes:                     {holes['count']}")
    print(f"hole overcure fraction:    {holes['cured_fraction']:.6f}")
    print(f"hole mean/p95/max DoC:     {holes['mean_doc']:.6f}/{holes['p95_doc']:.6f}/{holes['max_doc']:.6f}")


def _print_resolution_header(
    config: ResolutionConfig,
    device: torch.device,
    args: argparse.Namespace,
    params: AIEParameters,
) -> None:
    native_fov_y_um, native_fov_x_um = (
        value * 1e6 for value in config.native_fov_m
    )
    optimization_fov_y_um, optimization_fov_x_um = (
        value * 1e6 for value in config.optimization_fov_m
    )
    print(f"resolution_mode={config.resolution_mode}")
    print(f"target_shape={config.native_shape}")
    print(f"native_grid={config.native_shape}")
    print(f"optimization_grid={config.optimization_shape}")
    if config.resolution_mode == "coarse":
        print(f"coarsen_factor={config.coarsen_factor}")
    print(f"pixel_pitch={config.optimization_pixel_pitch_m * 1e6:.6f} um")
    if config.resolution_mode == "coarse":
        print(f"native_pixel_pitch={config.native_pixel_pitch_m * 1e6:.6f} um")
    print(
        f"native FOV: x={native_fov_x_um:.6f} um, "
        f"y={native_fov_y_um:.6f} um"
    )
    if config.resolution_mode == "coarse":
        print(
            f"coarse FOV: x={optimization_fov_x_um:.6f} um, "
            f"y={optimization_fov_y_um:.6f} um"
        )
    print(
        f"device={device} dt={params.dt:.3f}s "
        f"control_dt={params.dt * args.physics_steps_per_control:.3f}s "
        f"horizon={args.horizon} "
        f"prediction_horizon_seconds="
        f"{args.horizon * args.physics_steps_per_control * params.dt:.3f}s "
        f"iterations={args.iterations}"
    )
    print(f"total_process_time={args.total_time:.3f}s")
    print(f"derived_control_steps={args.control_steps}")


def _print_parameter_provenance(
    params: AIEParameters, reference: aie_reference.ReferenceConfig
) -> None:
    """Print the effective physical/model source at simulation startup."""

    print(f"Reference model: {params.reference_model_source}")
    print(f"Reference SHA256: {params.reference_model_sha256}")
    print(f"Reference structure SHA256: {params.reference_structure_sha256}")
    print(f"resolved source mode: {reference.physics_resolution_mode}")
    print(f"DoC calibration: {params.doc_calibration_source}")
    print(f"DoC calibration SHA256: {params.doc_calibration_sha256}")
    print(
        "reference calibration grid: "
        f"{params.native_shape[0]} x {params.native_shape[1]}"
    )
    print(f"native pixel pitch: {params.native_pixel_pitch_m * 1e6:.6f} um")
    print(f"dt: {params.dt:.6g} s")
    print(f"intensity: {params.intensity_mw_cm2:.6g} mW/cm^2")
    print(f"O2 diffusivity: {params.o2_diffusivity_m2_s:.6g} m^2/s")
    print(f"TEMPO diffusivity: {params.tempo_diffusivity_m2_s:.6g} m^2/s")
    print(f"O2 inhibition: {params.o2_inhibition_mj_cm2:.6g} mJ/cm^2")
    print(f"total inhibition: {params.total_inhibition_mj_cm2:.6g} mJ/cm^2")
    print(f"TEMPO inhibition: {params.tempo_inhibition_mj_cm2:.6g} mJ/cm^2")
    print(f"scattering physical size: {params.scattering_blur_size_m * 1e6:.6g} um")
    print(
        f"O2 diffusion: {'active' if params.o2_diffusion_enabled else 'bypassed'}"
    )
    print(
        f"TEMPO diffusion: {'active' if params.tempo_diffusion_enabled else 'bypassed'}; "
        f"Gaussian sigma scale={params.tempo_gaussian_sigma_scale:.6g}"
    )
    print(f"DoC model: {params.doc_model_id}")
    print(f"DoC history mode: {DOC_HISTORY_MODE}")
    print(f"DoC history status: {DOC_HISTORY_DESCRIPTION}")
    print(
        f"active B: {params.b_slope:.12g} * local_intensity + "
        f"{params.b_intercept:.12g} ({params.b_condition_label})"
    )
    value_sources = dict(reference.physics_value_sources)
    print(
        "physics value sources: "
        f"intensity={value_sources.get('intensity', 'unknown')}; "
        f"O2 inhibition={value_sources.get('o2_inhibition', 'unknown')}; "
        f"total inhibition={value_sources.get('total_inhibition', 'unknown')}; "
        f"TEMPO inhibition={value_sources.get('tempo_inhibition', 'unknown')}; "
        f"B law={value_sources.get('b_law', 'unknown')}"
    )
    print(f"chain-growth B noise std: {params.chain_growth_noise_std:.6g}")
    print(f"DoC fit selection: {params.doc_fit_selection_status}")
    if params.doc_fit_condition_id is not None:
        print(f"DoC fit condition: {params.doc_fit_condition_id}")
        print(f"DoC fit model: {params.doc_fit_formula_id}")
        print(
            f"a/b/c: {params.doc_fit_a:.12g}, "
            f"{params.doc_fit_b:.12g}, {params.doc_fit_c:.12g}"
        )
    else:
        print(
            "DoC fit condition: none selected from legacy doc_fit_parameters.json"
        )
    if not params.doc_fit_applied_to_governing_law:
        print("DoC fit usage: provenance only; governing B/intensity law preserved")


def run_demo(args: argparse.Namespace) -> None:
    """Run native MPC or coarse MPC followed by required native replay."""

    physics_condition = args.physics_condition
    reference = load_reference_config_for_condition(physics_condition)
    native_params = native_aie_parameters(reference)
    control_dt_s = native_params.dt * args.physics_steps_per_control
    control_steps, total_time_s = resolve_control_timing(
        total_time_s=args.total_time,
        control_steps=args.control_steps,
        control_dt_s=control_dt_s,
    )
    args.control_steps = control_steps
    args.total_time = total_time_s
    tracking_specification, doc_reference, tracking_warnings = (
        resolve_tracking_configuration(
            args, control_dt_s=control_dt_s, total_time_s=total_time_s
        )
    )
    checkpoints = tuple(
        zip(tracking_specification.point_times_s, tracking_specification.point_values)
    )
    if doc_reference is not None:
        physics_match = assess_reference_physics_match(
            doc_reference, native_params, physics_condition
        )
        physics_match["override_used"] = bool(args.allow_reference_physics_mismatch)
        if not physics_match["matched"] and not args.allow_reference_physics_mismatch:
            missing = "; ".join(physics_match["missing_for_physical_match"])
            raise ValueError(
                f"tracking reference {doc_reference.reference_id} does not have a "
                f"validated matching authoritative forward-physics configuration: "
                f"{missing}. Use --allow-reference-physics-mismatch only for an "
                "explicit research/debug comparison."
            )
    else:
        physics_match = {
            "matched": True,
            "tracking_reference_type": "explicit_checkpoints",
            "tracking_reference_source": "collaborator_specification",
            "forward_condition_id": physics_condition,
            "forward_intensity_mw_cm2": native_params.intensity_mw_cm2,
            "forward_tempo_concentration_mM": native_params.tempo_concentration_mM,
            "override_used": False,
        }
    target_path = resolve_target_path(args.target)
    target_native = load_normalized_target(target_path)
    require_native_target(target_native, target_path)
    native_shape = tuple(target_native.shape)
    config = build_resolution_config(
        args.resolution_mode,
        args.coarsen_factor,
        native_shape,
        reference=reference,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimization_params = (
        native_params
        if config.resolution_mode == "native"
        else replace(
            native_params,
            pixel_pitch_m=config.optimization_pixel_pitch_m,
            projector_refinement=1,
        )
    )
    print(f"forward physics selector: {physics_condition}")
    _print_parameter_provenance(native_params, reference)
    print(f"tracking spatial definition: {args.tracking_spatial_definition}")
    print(f"tracking loss: {args.tracking_loss}")
    print(f"tracking_variable={args.tracking_variable}")
    if args.tracking_loss == "huber":
        print(f"Huber delta: {args.huber_delta:g}")
    print(
        "controller weights target/outside/energy/smoothness: "
        f"{args.target_weight:g}/{args.outside_weight:g}/"
        f"{args.energy_weight:g}/{args.smoothness_weight:g}"
    )
    if doc_reference is not None:
        condition = doc_reference.metadata["condition"]
        thresholds = doc_reference.metadata.get("threshold_times_s", {})
        print(f"tracking_mode={args.tracking_mode}")
        print(f"DoC tracking reference ID: {doc_reference.reference_id}")
        print(
            "DoC tracking condition: "
            f"{condition['intensity_mw_cm2']:g} mW/cm^2, "
            f"{condition['tempo_concentration_mM']:g} mM TEMPO"
        )
        print(
            "DoC curve model: "
            f"{doc_reference.curve_model} "
            f"({doc_reference.metadata.get('curve_model_role', 'unspecified role')})"
        )
        if all(name in thresholds for name in ("t10_s", "t50_s", "t90_s")):
            print(
                "artifact t10/t50/t90: "
                f"{thresholds['t10_s']:.3f}/{thresholds['t50_s']:.3f}/"
                f"{thresholds['t90_s']:.3f} s"
            )
        print(f"DoC reference artifact SHA256: {doc_reference.source_sha256}")
        if args.tracking_mode == "sampled-curve":
            print("sampled curve requirements (absolute process time):")
            for point_time_s, required_doc in checkpoints:
                print(f"  t={point_time_s:.2f} s -> DoC={required_doc:.4f}")
    else:
        print("tracking_mode=checkpoints")
        print("tracking_reference_type=explicit_checkpoints")
        print("tracking_reference_source=collaborator_specification")
        print(f"forward_physics_condition={physics_condition}")
        print("checkpoint requirements (absolute process time):")
        for checkpoint_time_s, required_doc in checkpoints:
            print(f"  t={checkpoint_time_s:.2f} s -> DoC={required_doc:.4f}")
        print(
            "target-side tracking is evaluated only at listed checkpoints; "
            "outside/energy/smoothness penalties remain dense"
        )
    if args.tracking_variable == "reaction-progress":
        if checkpoints:
            print("reaction-progress tracking requirements:")
            for point_time_s, required_doc in checkpoints:
                print(
                    f"  t={point_time_s:.2f} s -> DoC*={required_doc:.4f} "
                    f"-> R*={doc_to_reaction_progress(required_doc):.6f}"
                )
        else:
            print("dense curve DoC requirements are mapped stagewise with R*=-log1p(-DoC*)")
    for message in tracking_warnings:
        print(f"WARNING: {message}")
    if doc_reference is not None and not physics_match["matched"]:
        print("\n!!!!!!!!!!!!!!!! PHYSICS/REFERENCE MISMATCH !!!!!!!!!!!!!!!!")
        print(f"tracking reference = {doc_reference.reference_id}")
        print("forward physics != validated matching experimental condition")
        for item in physics_match["missing_for_physical_match"]:
            print(f"missing: {item}")
        print("override: --allow-reference-physics-mismatch (research/debug only)")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    print(
        "DoC reference use: control target only; AIE/reaction_progress forward "
        "physics unchanged"
    )
    _print_resolution_header(config, device, args, optimization_params)

    target_native = target_native.to(device=device)
    optimization_target = construct_optimization_target(target_native, config)
    model = AIEModel(optimization_params, device=device)
    native_forward_model: AIEModel | None = (
        model if config.resolution_mode == "native" else None
    )
    state = model.initialize_state(config.optimization_shape)
    native_initial_state: AIEState | None = (
        state.detach() if config.resolution_mode == "native" else None
    )
    optimization_initial_target_doc_summary = doc_region_summary(
        state.doc, optimization_target, args.target_threshold
    )
    optimization_initial_target_o2_summary = target_field_summary(
        state.o2,
        optimization_target,
        args.target_threshold,
        label="initial O2",
    )
    controller = DifferentiableMPC(
        model=model,
        target=optimization_target,
        tracking_specification=tracking_specification,
        horizon=args.horizon,
        physics_steps_per_control=args.physics_steps_per_control,
        target_threshold=args.target_threshold,
        target_weight=args.target_weight,
        outside_weight=args.outside_weight,
        energy_weight=args.energy_weight,
        smoothness_weight=args.smoothness_weight,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    physics_initialization: PhysicsAwareInitializationResult | None = None
    initial_guess_projector_native: torch.Tensor | None = None
    initial_guess_local_intensity_native: torch.Tensor | None = None
    initializer_component_rows: list[dict[str, object]] = []
    if args.initialization_mode == "uniform":
        # Preserve the historical cold start exactly.
        guess = controller.initial_control_sequence(
            target_level=args.initial_target_level,
            background_level=args.initial_background_level,
        )
        initialization_metadata: dict[str, object] = {
            "mode": "uniform",
            "heuristic_only": False,
            "initial_target_level": args.initial_target_level,
            "initial_background_level": args.initial_background_level,
        }
        print("initialization_mode=uniform")
    else:
        if not tracking_specification.point_times_s:
            raise TrackingConfigurationError(
                "physics-aware initialization requires an explicit checkpoint or "
                "sampled-curve target time; dense curve mode has no single cold-start "
                "checkpoint. Use --initialization-mode uniform or provide sparse "
                "tracking requirements."
            )
        initialization_checkpoint_time = tracking_specification.point_times_s[0]
        initialization_checkpoint_doc = tracking_specification.point_values[0]
        physics_initialization = build_physics_aware_initial_mask(
            model=model,
            target=optimization_target,
            target_threshold=args.target_threshold,
            checkpoint_time_s=initialization_checkpoint_time,
            checkpoint_doc=initialization_checkpoint_doc,
            background_level=args.initial_background_level,
            num_iterations=args.physics_init_iterations,
            learning_rate=args.physics_init_lr,
            outside_weight=args.physics_init_outside_weight,
        )
        # The precompensated mask is repeated only for the first MPC solve.  All
        # later solves retain the normal shifted optimized-horizon warm start.
        guess = physics_initialization.projector_mask.unsqueeze(0).repeat(
            args.horizon, 1, 1
        )
        initial_guess_projector_native = recover_control_to_native(
            physics_initialization.projector_mask, config
        ).detach().clone()
        if config.resolution_mode == "native":
            initial_guess_local_intensity_native = (
                physics_initialization.local_normalized_intensity.detach().clone()
            )
        else:
            native_optical_model = AIEModel(native_params, device=device)
            with torch.no_grad():
                native_prepared = native_optical_model.prepare_control(
                    initial_guess_projector_native, config.native_shape
                )
                initial_guess_local_intensity_native = (
                    native_prepared.local_intensity
                    / native_optical_model.params.intensity_mw_cm2
                ).detach().clone()

        projector_summary = normalized_field_region_summary(
            initial_guess_projector_native,
            target_native,
            args.target_threshold,
            label="physics-aware native projector mask",
        )
        local_summary = normalized_field_region_summary(
            initial_guess_local_intensity_native,
            target_native,
            args.target_threshold,
            label="physics-aware native local normalized intensity",
        )
        initializer_component_rows = initializer_component_metrics(
            (target_native > args.target_threshold).detach().cpu().numpy(),
            initial_guess_projector_native.detach().cpu().numpy(),
            initial_guess_local_intensity_native.detach().cpu().numpy(),
        )
        initialization_metadata = {
            "mode": "physics-aware",
            "heuristic_only": True,
            "heuristic_scope": (
                "first MPC solve cold start only; AIE forward physics and MPC "
                "objective are unchanged"
            ),
            "checkpoint_selection": "earliest explicit target requirement",
            "checkpoint_time_s": initialization_checkpoint_time,
            "checkpoint_doc": initialization_checkpoint_doc,
            "nominal_intensity_mw_cm2": (
                physics_initialization.nominal.intensity_mw_cm2
            ),
            "nominal_normalized_level": (
                physics_initialization.nominal.normalized_level
            ),
            "nominal_predicted_doc": physics_initialization.nominal.predicted_doc,
            "nominal_maximum_achievable_doc": (
                physics_initialization.nominal.maximum_achievable_doc
            ),
            "nominal_solver": {
                "method": "bounded_monotone_bisection",
                "iterations": physics_initialization.nominal.bisection_iterations,
                "model": "O2-only constant-local-intensity approximation",
                "b_law_source": "active AIEModel.params affine B/intensity law",
            },
            "optical_initializer": {
                "method": "Adam on sigmoid-bounded projector logits",
                "iterations": physics_initialization.optical_iterations,
                "learning_rate": physics_initialization.optical_learning_rate,
                "outside_weight": physics_initialization.optical_outside_weight,
                "initial_loss": physics_initialization.initial_optical_loss,
                "final_loss": physics_initialization.final_optical_loss,
                "operator": "AIEModel.prepare_control",
                "initial_target_level_argument": "unused",
                "initial_background_level_seed": args.initial_background_level,
            },
            "projector_mask_native": projector_summary,
            "local_normalized_intensity_native": local_summary,
            "component_count": len(initializer_component_rows),
            "component_metrics": initializer_component_rows,
            "component_metrics_path": "initial_guess_component_metrics.csv",
        }
        print("initialization_mode=physics-aware")
        print(
            f"initialization_checkpoint_time={initialization_checkpoint_time:g} s"
        )
        print(f"initialization_checkpoint_doc={initialization_checkpoint_doc:g}")
        print(
            "physics_aware_nominal_intensity="
            f"{physics_initialization.nominal.intensity_mw_cm2:.8g} mW/cm^2"
        )
        print(
            "physics_aware_nominal_normalized_level="
            f"{physics_initialization.nominal.normalized_level:.8g}"
        )
        print(
            "physics_aware_mask target mean/min/max="
            f"{projector_summary['target_mean']:.6f}/"
            f"{projector_summary['target_min']:.6f}/"
            f"{projector_summary['target_max']:.6f}"
        )
        print(
            "physics_aware_optical_field target mean/min/max="
            f"{local_summary['target_mean']:.6f}/"
            f"{local_summary['target_min']:.6f}/"
            f"{local_summary['target_max']:.6f}"
        )
        print(
            "physics_aware_optical_field outside mean="
            f"{local_summary['outside_mean']:.6f}"
        )
        print(
            "physics_aware_initializer_components="
            f"{len(initializer_component_rows)} "
            "(details: initial_guess_component_metrics.csv)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_grayscale(target_native, args.output_dir / "target_native.png")
    if config.resolution_mode == "coarse":
        save_grayscale(optimization_target, args.output_dir / "target_coarse.png")
    if physics_initialization is not None:
        assert initial_guess_projector_native is not None
        assert initial_guess_local_intensity_native is not None
        save_grayscale(
            initial_guess_projector_native,
            args.output_dir / "initial_guess_projector_native.png",
        )
        save_grayscale(
            initial_guess_local_intensity_native,
            args.output_dir / "initial_guess_local_intensity_native.png",
        )
        if config.resolution_mode == "coarse":
            save_grayscale(
                physics_initialization.projector_mask,
                args.output_dir / "initial_guess_projector_coarse.png",
            )
            save_grayscale(
                physics_initialization.local_normalized_intensity,
                args.output_dir / "initial_guess_local_intensity_coarse.png",
            )
        save_initializer_component_metrics_csv(
            initializer_component_rows,
            args.output_dir / "initial_guess_component_metrics.csv",
        )

    applied_controls_optimization: list[torch.Tensor] = []
    applied_controls_native: list[torch.Tensor] = []
    optimization_histories: list[list[float]] = []
    optimization_control_times_s: list[float] = []
    optimization_reference_doc: list[float] = []
    optimization_target_doc: list[float] = []
    optimization_target_doc_min: list[float] = []
    optimization_target_doc_max: list[float] = []
    optimization_target_doc_std: list[float] = []
    optimization_outside_doc: list[float] = []
    optimization_target_reaction_progress: list[float] = []
    optimization_target_reaction_progress_min: list[float] = []
    optimization_target_reaction_progress_max: list[float] = []
    optimization_target_reaction_progress_std: list[float] = []
    optimization_target_o2: list[float] = []
    optimization_target_o2_min: list[float] = []
    optimization_target_o2_max: list[float] = []
    optimization_target_o2_std: list[float] = []
    coarse_doc_frame_paths: list[Path] = []
    native_doc_frame_paths: list[Path] = []
    if config.resolution_mode == "coarse":
        initial_doc_path = args.output_dir / "doc_frame_coarse_000.png"
        save_doc_frame(state.doc, initial_doc_path, config.optimization_shape)
        coarse_doc_frame_paths.append(initial_doc_path)
    else:
        initial_doc_path = args.output_dir / "doc_frame_native_000.png"
        save_doc_frame(state.doc, initial_doc_path, config.native_shape)
        native_doc_frame_paths.append(initial_doc_path)
    horizon_energy_at_full_power = (
        optimization_params.intensity_mw_cm2
        * optimization_params.dt
        * args.physics_steps_per_control
        * args.horizon
    )
    if optimization_params.o2_inhibition_mj_cm2 > horizon_energy_at_full_power:
        print(
            "note: the initial horizon is shorter than the legacy hard O2 "
            "inhibition time; DoC tracking gradients appear after receding "
            "exposure has depleted enough O2"
        )

    for control_step in range(args.control_steps):
        started = time.perf_counter()
        current_process_time_s = control_step * control_dt_s
        optimized_controls, info = controller.optimize(
            state,
            initial_guess=guess,
            current_process_time_s=current_process_time_s,
        )
        applied_control = optimized_controls[0].detach().clone()
        state = model.advance(
            state,
            applied_control,
            physics_steps=args.physics_steps_per_control,
        )
        doc_frame_index = control_step + 1
        if config.resolution_mode == "coarse":
            doc_frame_path = (
                args.output_dir / f"doc_frame_coarse_{doc_frame_index:03d}.png"
            )
            save_doc_frame(state.doc, doc_frame_path, config.optimization_shape)
            coarse_doc_frame_paths.append(doc_frame_path)
        else:
            doc_frame_path = (
                args.output_dir / f"doc_frame_native_{doc_frame_index:03d}.png"
            )
            save_doc_frame(state.doc, doc_frame_path, config.native_shape)
            native_doc_frame_paths.append(doc_frame_path)
        guess = controller.shift_warm_start(optimized_controls)
        applied_native = recover_control_to_native(applied_control, config).detach().clone()

        applied_controls_optimization.append(applied_control)
        applied_controls_native.append(applied_native)
        optimization_histories.append(info["loss_history"])
        if config.resolution_mode == "coarse":
            save_grayscale(
                applied_control,
                args.output_dir / f"applied_mask_coarse_{control_step:03d}.png",
            )
        save_grayscale(
            applied_native,
            args.output_dir / f"applied_mask_native_{control_step:03d}.png",
        )

        target_doc_summary = doc_region_summary(
            state.doc, optimization_target, args.target_threshold
        )
        target_doc_mean = target_doc_summary["mean_target_doc"]
        target_doc_min = target_doc_summary["min_target_doc"]
        target_doc_max = target_doc_summary["max_target_doc"]
        target_doc_std = target_doc_summary["std_target_doc"]
        outside_doc_mean = target_doc_summary["mean_outside_doc"]
        target_reaction_summary = target_field_summary(
            state.reaction_progress,
            optimization_target,
            args.target_threshold,
            label="reaction_progress",
        )
        target_o2_summary = target_field_summary(
            state.o2,
            optimization_target,
            args.target_threshold,
            label="O2",
        )
        applied_time_s = (control_step + 1) * control_dt_s
        if args.tracking_mode == "curve":
            assert doc_reference is not None
            reference_doc = float(doc_reference.at(applied_time_s))
        else:
            reference_doc = float("nan")
            for point_time_s, point_doc in checkpoints:
                if math.isclose(
                    applied_time_s, point_time_s, rel_tol=0.0, abs_tol=1e-9
                ):
                    reference_doc = float(point_doc)
                    break
        optimization_control_times_s.append(applied_time_s)
        optimization_reference_doc.append(reference_doc)
        optimization_target_doc.append(target_doc_mean)
        optimization_target_doc_min.append(target_doc_min)
        optimization_target_doc_max.append(target_doc_max)
        optimization_target_doc_std.append(target_doc_std)
        optimization_outside_doc.append(outside_doc_mean)
        optimization_target_reaction_progress.append(target_reaction_summary["mean"])
        optimization_target_reaction_progress_min.append(target_reaction_summary["min"])
        optimization_target_reaction_progress_max.append(target_reaction_summary["max"])
        optimization_target_reaction_progress_std.append(target_reaction_summary["std"])
        optimization_target_o2.append(target_o2_summary["mean"])
        optimization_target_o2_min.append(target_o2_summary["min"])
        optimization_target_o2_max.append(target_o2_summary["max"])
        optimization_target_o2_std.append(target_o2_summary["std"])
        running_reference = np.asarray(optimization_reference_doc, dtype=float)
        running_actual = np.asarray(optimization_target_doc, dtype=float)
        active_reporting = np.isfinite(running_reference)
        running_rmse = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            running_actual[active_reporting]
                            - running_reference[active_reporting]
                        )
                    )
                )
            )
            if np.any(active_reporting)
            else float("nan")
        )
        elapsed = time.perf_counter() - started
        reference_text = f"{reference_doc:.4f}" if math.isfinite(reference_doc) else "--"
        running_text = f"{running_rmse:.5f}" if math.isfinite(running_rmse) else "--"
        print(
            f"step {control_step + 1:02d}/{args.control_steps:02d} "
            f"t={applied_time_s:.2f}s ref={reference_text} "
            f"loss {info['initial_loss']:.5f}->{info['final_loss']:.5f} "
            "DoC(target mean/min/max/std)="
            f"{target_doc_mean:.4f}/{target_doc_min:.4f}/"
            f"{target_doc_max:.4f}/{target_doc_std:.4f} "
            f"out={outside_doc_mean:.4f} "
            f"track_rmse={running_text} "
            f"mask_mean={float(applied_control.mean()):.3f} solve={elapsed:.2f}s"
        )
        if args.tracking_mode != "curve" and math.isfinite(reference_doc):
            point_heading = (
                "CHECKPOINT"
                if args.tracking_mode == "checkpoints"
                else "SAMPLED TRACKING POINT"
            )
            print(f"=== {point_heading} ===")
            print(f"time:                  {applied_time_s:.2f} s")
            print(f"requested target DoC:  {reference_doc:.4f}")
            print(f"achieved mean DoC:     {target_doc_mean:.4f}")
            print(f"minimum target DoC:    {target_doc_min:.4f}")
            print(f"maximum target DoC:    {target_doc_max:.4f}")
            print(f"target DoC std:        {target_doc_std:.4f}")
            print(f"mean error:            {target_doc_mean - reference_doc:+.4f}")
            requested_reaction_progress = float(
                doc_array_to_reaction_progress_for_reporting(
                    np.asarray([reference_doc], dtype=float)
                )[0]
            )
            print(f"requested target R:    {requested_reaction_progress:.6f}")
            print(
                "achieved R mean/min/max/std: "
                f"{target_reaction_summary['mean']:.6f}/"
                f"{target_reaction_summary['min']:.6f}/"
                f"{target_reaction_summary['max']:.6f}/"
                f"{target_reaction_summary['std']:.6f}"
            )
            print(
                "R tracking error:       "
                f"{target_reaction_summary['mean'] - requested_reaction_progress:+.6f}"
            )
            print(
                "target O2 mean/min/max:  "
                f"{target_o2_summary['mean']:.6f}/"
                f"{target_o2_summary['min']:.6f}/"
                f"{target_o2_summary['max']:.6f} mJ/cm^2"
            )

    optimization_state = state
    coarse_gif_path: Path | None = None
    if config.resolution_mode == "coarse":
        save_grayscale(optimization_state.doc, args.output_dir / "final_doc_coarse.png")
        save_grayscale(
            optimization_state.o2
            / max(optimization_params.o2_inhibition_mj_cm2, 1e-12),
            args.output_dir / "final_o2_coarse.png",
        )
        coarse_gif_path = args.output_dir / "doc_evolution_coarse.gif"
        if len(coarse_doc_frame_paths) != args.control_steps + 1:
            raise AssertionError("coarse DoC timeline has an unexpected frame count")
        save_gif_from_frames(coarse_doc_frame_paths, coarse_gif_path)

        native_replay_target_doc: list[float] = []
        native_replay_target_doc_min: list[float] = []
        native_replay_target_doc_max: list[float] = []
        native_replay_target_doc_std: list[float] = []
        native_replay_outside_doc: list[float] = []
        native_replay_target_reaction_progress: list[float] = []
        native_replay_target_reaction_progress_min: list[float] = []
        native_replay_target_reaction_progress_max: list[float] = []
        native_replay_target_reaction_progress_std: list[float] = []
        native_replay_target_o2: list[float] = []
        native_replay_target_o2_min: list[float] = []
        native_replay_target_o2_max: list[float] = []
        native_replay_target_o2_std: list[float] = []
        native_replay_initial_plot_summaries: dict[
            str, dict[str, float]
        ] = {}

        def save_native_replay_frame(frame_index: int, doc: torch.Tensor) -> None:
            frame_path = args.output_dir / f"doc_frame_native_{frame_index:03d}.png"
            save_doc_frame(doc, frame_path, config.native_shape)
            native_doc_frame_paths.append(frame_path)
            if frame_index > 0:
                target_summary = doc_region_summary(
                    doc, target_native, args.target_threshold
                )
                native_replay_target_doc.append(target_summary["mean_target_doc"])
                native_replay_target_doc_min.append(target_summary["min_target_doc"])
                native_replay_target_doc_max.append(target_summary["max_target_doc"])
                native_replay_target_doc_std.append(target_summary["std_target_doc"])
                native_replay_outside_doc.append(target_summary["mean_outside_doc"])

        def record_native_replay_state(frame_index: int, replay_state: AIEState) -> None:
            if frame_index == 0:
                native_replay_initial_plot_summaries["doc"] = doc_region_summary(
                    replay_state.doc, target_native, args.target_threshold
                )
                native_replay_initial_plot_summaries["o2"] = target_field_summary(
                    replay_state.o2,
                    target_native,
                    args.target_threshold,
                    label="initial native replay O2",
                )
                return
            reaction_summary = target_field_summary(
                replay_state.reaction_progress,
                target_native,
                args.target_threshold,
                label="native replay reaction_progress",
            )
            o2_summary = target_field_summary(
                replay_state.o2,
                target_native,
                args.target_threshold,
                label="native replay O2",
            )
            native_replay_target_reaction_progress.append(reaction_summary["mean"])
            native_replay_target_reaction_progress_min.append(reaction_summary["min"])
            native_replay_target_reaction_progress_max.append(reaction_summary["max"])
            native_replay_target_reaction_progress_std.append(reaction_summary["std"])
            native_replay_target_o2.append(o2_summary["mean"])
            native_replay_target_o2_min.append(o2_summary["min"])
            native_replay_target_o2_max.append(o2_summary["max"])
            native_replay_target_o2_std.append(o2_summary["std"])

        native_forward_model = AIEModel(native_params, device=device)
        native_initial_state = native_forward_model.initialize_state(
            config.native_shape
        )
        native_state = replay_native_controls(
            native_params,
            config.native_shape,
            applied_controls_native,
            args.physics_steps_per_control,
            device,
            doc_frame_callback=save_native_replay_frame,
            authoritative_reference=reference,
            state_callback=record_native_replay_state,
            native_model=native_forward_model,
            initial_state=native_initial_state,
        )
        if set(native_replay_initial_plot_summaries) != {"doc", "o2"}:
            raise AssertionError("native replay did not report its initialized state")
        initial_target_doc_summary_for_plot = (
            native_replay_initial_plot_summaries["doc"]
        )
        initial_target_o2_summary_for_plot = (
            native_replay_initial_plot_summaries["o2"]
        )
    else:
        native_state = optimization_state
        native_replay_target_doc = optimization_target_doc
        native_replay_target_doc_min = optimization_target_doc_min
        native_replay_target_doc_max = optimization_target_doc_max
        native_replay_target_doc_std = optimization_target_doc_std
        native_replay_outside_doc = optimization_outside_doc
        native_replay_target_reaction_progress = optimization_target_reaction_progress
        native_replay_target_reaction_progress_min = (
            optimization_target_reaction_progress_min
        )
        native_replay_target_reaction_progress_max = (
            optimization_target_reaction_progress_max
        )
        native_replay_target_reaction_progress_std = (
            optimization_target_reaction_progress_std
        )
        native_replay_target_o2 = optimization_target_o2
        native_replay_target_o2_min = optimization_target_o2_min
        native_replay_target_o2_max = optimization_target_o2_max
        native_replay_target_o2_std = optimization_target_o2_std
        initial_target_doc_summary_for_plot = (
            optimization_initial_target_doc_summary
        )
        initial_target_o2_summary_for_plot = optimization_initial_target_o2_summary

    if len(native_doc_frame_paths) != args.control_steps + 1:
        raise AssertionError("native DoC timeline has an unexpected frame count")
    native_gif_path = args.output_dir / "doc_evolution_native.gif"
    save_gif_from_frames(native_doc_frame_paths, native_gif_path)

    for name, field in zip(
        ("o2", "tempo", "dose", "reaction_progress", "doc"),
        native_state.tensors(),
    ):
        if tuple(field.shape) != config.native_shape:
            raise AssertionError(
                f"final native {name} must have shape {config.native_shape}, got "
                f"{tuple(field.shape)}"
            )
    save_grayscale(native_state.doc, args.output_dir / "final_doc_native.png")
    save_grayscale(
        native_state.o2 / max(native_params.o2_inhibition_mj_cm2, 1e-12),
        args.output_dir / "final_o2_native.png",
    )

    native_target_summary = doc_region_summary(
        native_state.doc, target_native, args.target_threshold
    )
    native_target_doc = native_target_summary["mean_target_doc"]
    native_outside_doc = native_target_summary["mean_outside_doc"]
    primary_control_times_s = np.asarray(optimization_control_times_s, dtype=float)
    primary_reference_doc = np.asarray(optimization_reference_doc, dtype=float)
    primary_target_doc = np.asarray(native_replay_target_doc, dtype=float)
    primary_target_doc_min = np.asarray(native_replay_target_doc_min, dtype=float)
    primary_target_doc_max = np.asarray(native_replay_target_doc_max, dtype=float)
    primary_target_doc_std = np.asarray(native_replay_target_doc_std, dtype=float)
    primary_target_doc_lower_error = primary_target_doc - primary_target_doc_min
    primary_target_doc_upper_error = primary_target_doc_max - primary_target_doc
    primary_outside_doc = np.asarray(native_replay_outside_doc, dtype=float)
    primary_target_reaction_progress = np.asarray(
        native_replay_target_reaction_progress, dtype=float
    )
    primary_target_reaction_progress_min = np.asarray(
        native_replay_target_reaction_progress_min, dtype=float
    )
    primary_target_reaction_progress_max = np.asarray(
        native_replay_target_reaction_progress_max, dtype=float
    )
    primary_target_reaction_progress_std = np.asarray(
        native_replay_target_reaction_progress_std, dtype=float
    )
    primary_target_o2 = np.asarray(native_replay_target_o2, dtype=float)
    primary_target_o2_min = np.asarray(native_replay_target_o2_min, dtype=float)
    primary_target_o2_max = np.asarray(native_replay_target_o2_max, dtype=float)
    primary_target_o2_std = np.asarray(native_replay_target_o2_std, dtype=float)
    if not (
        primary_control_times_s.size
        == primary_reference_doc.size
        == primary_target_doc.size
        == primary_target_doc_min.size
        == primary_target_doc_max.size
        == primary_target_doc_std.size
        == primary_outside_doc.size
        == primary_target_reaction_progress.size
        == primary_target_reaction_progress_min.size
        == primary_target_reaction_progress_max.size
        == primary_target_reaction_progress_std.size
        == primary_target_o2.size
        == primary_target_o2_min.size
        == primary_target_o2_max.size
        == primary_target_o2_std.size
        == args.control_steps
    ):
        raise AssertionError("primary native tracking timeline has an unexpected length")
    primary_tracking_error = primary_target_doc - primary_reference_doc
    primary_reference_reaction_progress = (
        doc_array_to_reaction_progress_for_reporting(primary_reference_doc)
    )
    primary_reaction_progress_tracking_error = (
        primary_target_reaction_progress - primary_reference_reaction_progress
    )
    checkpoint_tracking: dict[str, object] | None = None
    if args.tracking_mode == "curve":
        tracking = temporal_tracking_metrics(primary_target_doc, primary_reference_doc)
        finite_reaction_reference = np.isfinite(primary_reference_reaction_progress)
        reaction_progress_tracking = (
            temporal_tracking_metrics(
                primary_target_reaction_progress[finite_reaction_reference],
                primary_reference_reaction_progress[finite_reaction_reference],
            )
            if np.any(finite_reaction_reference)
            else None
        )
        assert doc_reference is not None
        reference_final_doc = float(doc_reference.at(total_time_s))
    else:
        checkpoint_tracking = calculate_checkpoint_tracking(
            primary_control_times_s,
            primary_target_doc,
            primary_target_doc_min,
            primary_target_doc_max,
            primary_target_doc_std,
            primary_target_reaction_progress,
            primary_target_reaction_progress_min,
            primary_target_reaction_progress_max,
            primary_target_reaction_progress_std,
            primary_target_o2,
            primary_target_o2_min,
            primary_target_o2_max,
            primary_target_o2_std,
            checkpoints,
        )
        tracking = {
            "rmse": checkpoint_tracking["rmse"],
            "mae": checkpoint_tracking["mae"],
            "max_absolute_error": checkpoint_tracking["max_absolute_error"],
        }
        reaction_progress_tracking = (
            None
            if checkpoint_tracking["reaction_progress_rmse"] is None
            else {
                "rmse": checkpoint_tracking["reaction_progress_rmse"],
                "mae": checkpoint_tracking["reaction_progress_mae"],
                "max_absolute_error": checkpoint_tracking[
                    "reaction_progress_max_absolute_error"
                ],
            }
        )
        reference_final_doc = float(checkpoints[-1][1])
    final_metrics = calculate_final_metrics(
        native_state.doc.detach().cpu().numpy(),
        target_native.detach().cpu().numpy(),
        reference_final_doc,
        args.target_threshold,
        args.geometry_threshold,
        config.native_pixel_pitch_m * 1e6,
    )
    baseline_result: TargetMaskBaselineResult | None = None
    if getattr(args, "baseline_target_mask", True):
        if native_forward_model is None or native_initial_state is None:
            raise AssertionError("native forward model/state was not initialized")
        print("\nRunning unoptimized repeated target-mask baseline...")
        baseline_result = run_unoptimized_target_mask_baseline(
            model=native_forward_model,
            initial_state=native_initial_state,
            target_native=target_native,
            control_times_s=primary_control_times_s,
            reference_doc=primary_reference_doc,
            checkpoints=checkpoints,
            tracking_mode=args.tracking_mode,
            tracking_variable=args.tracking_variable,
            reference_final_doc=reference_final_doc,
            target_threshold=args.target_threshold,
            geometry_threshold=args.geometry_threshold,
            pixel_pitch_um=config.native_pixel_pitch_m * 1e6,
            physics_steps_per_control=args.physics_steps_per_control,
            output_dir=args.output_dir / "baseline_target_mask",
            resolution_mode=config.resolution_mode,
            physics_condition=physics_condition,
            authoritative_reference=reference,
            tracking_specification_metadata=(
                tracking_specification.provenance_metadata()
            ),
        )
        baseline_comparison = build_baseline_comparison(
            mpc_control_times_s=primary_control_times_s,
            mpc_reference_doc=primary_reference_doc,
            mpc_target_doc=primary_target_doc,
            mpc_final_target_summary=native_target_summary,
            mpc_tracking=tracking,
            mpc_tracking_points=checkpoint_tracking,
            mpc_final_metrics=final_metrics,
            baseline=baseline_result,
        )
    else:
        baseline_comparison = {
            "enabled": False,
            "label": "unoptimized repeated target-mask baseline",
            "reason": "disabled by --no-baseline-target-mask",
        }
    component_metrics_path = args.output_dir / "component_metrics.csv"
    save_component_metrics_csv(final_metrics["component_rows"], component_metrics_path)
    if args.tracking_mode == "curve":
        tracking_plot_path = args.output_dir / "doc_tracking_curve.png"
        assert doc_reference is not None
        save_tracking_plot(
            doc_reference,
            total_time_s,
            primary_control_times_s,
            primary_target_doc,
            primary_target_doc_min,
            primary_target_doc_max,
            primary_target_o2,
            primary_target_o2_min,
            primary_target_o2_max,
            tracking_plot_path,
            initial_target_doc_summary=initial_target_doc_summary_for_plot,
            initial_target_o2_summary=initial_target_o2_summary_for_plot,
            baseline_target_doc=(
                None
                if baseline_result is None
                else baseline_result.timeline["mean_target_doc"]
            ),
            baseline_initial_target_doc_summary=(
                None
                if baseline_result is None
                else baseline_result.initial_target_doc_summary
            ),
        )
    else:
        tracking_plot_path = args.output_dir / "doc_checkpoint_tracking.png"
        assert checkpoint_tracking is not None
        save_checkpoint_tracking_plot(
            total_time_s,
            primary_control_times_s,
            primary_target_doc,
            primary_target_doc_min,
            primary_target_doc_max,
            primary_target_o2,
            primary_target_o2_min,
            primary_target_o2_max,
            checkpoint_tracking,
            tracking_plot_path,
            initial_target_doc_summary=initial_target_doc_summary_for_plot,
            initial_target_o2_summary=initial_target_o2_summary_for_plot,
            marker_label=(
                "sampled fitted-curve requirements"
                if args.tracking_mode == "sampled-curve"
                else "explicit target checkpoints"
            ),
            title=(
                "Sparse sampled-curve DoC tracking"
                if args.tracking_mode == "sampled-curve"
                else "Checkpoint DoC tracking"
            ),
            baseline_target_doc=(
                None
                if baseline_result is None
                else baseline_result.timeline["mean_target_doc"]
            ),
            baseline_initial_target_doc_summary=(
                None
                if baseline_result is None
                else baseline_result.initial_target_doc_summary
            ),
        )
    tracking_points_csv_path: Path | None = None
    if checkpoint_tracking is not None:
        tracking_points_csv_path = args.output_dir / "doc_tracking_points.csv"
        save_tracking_points_csv(
            checkpoint_tracking,
            tracking_points_csv_path,
            tracking_variable=args.tracking_variable,
        )
    components_summary = {
        **final_metrics["components"],
        "table_path": component_metrics_path.name,
    }
    holes_summary = {
        **final_metrics["holes"],
        "details": final_metrics["hole_rows"],
    }
    metrics_document = {
        "schema_version": 1,
        "tracking_variable": args.tracking_variable,
        "primary_result_grid": (
            "native_replay" if config.resolution_mode == "coarse" else "native"
        ),
        "tracking_specification": tracking_specification.provenance_metadata(),
        "reference_provenance": (
            None if doc_reference is None else doc_reference.provenance_metadata()
        ),
        "forward_model_provenance": {
            "source": native_params.reference_model_source,
            "sha256": native_params.reference_model_sha256,
            "selector": reference.physics_selector_id,
            "resolved_source_mode": reference.physics_resolution_mode,
            "value_sources": dict(reference.physics_value_sources),
            "history_mode": DOC_HISTORY_MODE,
            "history_description": DOC_HISTORY_DESCRIPTION,
            "tracking_reference_physics_match": physics_match,
        },
        "target": {
            "name": target_path.name,
            "path": str(target_path),
            "shape": list(config.native_shape),
            "target_threshold": args.target_threshold,
        },
        "initialization": initialization_metadata,
        "total_process_time_s": total_time_s,
        "final_geometry_reference": {
            "doc": reference_final_doc,
            "source": (
                "selected_curve_at_total_time"
                if args.tracking_mode == "curve"
                else "last_sparse_requirement_at_or_before_total_time"
            ),
        },
        "control_settings": {
            "dt_s": native_params.dt,
            "physics_steps_per_control": args.physics_steps_per_control,
            "control_dt_s": control_dt_s,
            "control_steps": args.control_steps,
            "horizon": args.horizon,
            "prediction_horizon_seconds": args.horizon * control_dt_s,
            "target_weight": args.target_weight,
            "outside_weight": args.outside_weight,
            "energy_weight": args.energy_weight,
            "smoothness_weight": args.smoothness_weight,
        },
        "target_region_doc_timeline": {
            "times_s": primary_control_times_s.tolist(),
            "mean_target_doc": primary_target_doc.tolist(),
            "min_target_doc": primary_target_doc_min.tolist(),
            "max_target_doc": primary_target_doc_max.tolist(),
            "std_target_doc": primary_target_doc_std.tolist(),
            "lower_error": primary_target_doc_lower_error.tolist(),
            "upper_error": primary_target_doc_upper_error.tolist(),
            "mean_outside_doc": primary_outside_doc.tolist(),
        },
        "target_region_reaction_progress_timeline": {
            "times_s": primary_control_times_s.tolist(),
            "requested_reaction_progress": json_safe_float_list(
                primary_reference_reaction_progress
            ),
            "mean_target_reaction_progress": (
                primary_target_reaction_progress.tolist()
            ),
            "min_target_reaction_progress": (
                primary_target_reaction_progress_min.tolist()
            ),
            "max_target_reaction_progress": (
                primary_target_reaction_progress_max.tolist()
            ),
            "std_target_reaction_progress": (
                primary_target_reaction_progress_std.tolist()
            ),
            "reaction_progress_error": json_safe_float_list(
                primary_reaction_progress_tracking_error
            ),
        },
        "target_region_o2_timeline_mj_cm2": {
            "units": "mJ/cm^2",
            "times_s": primary_control_times_s.tolist(),
            "target_o2_mean": primary_target_o2.tolist(),
            "target_o2_min": primary_target_o2_min.tolist(),
            "target_o2_max": primary_target_o2_max.tolist(),
            "target_o2_std": primary_target_o2_std.tolist(),
        },
        "temporal_tracking": tracking if args.tracking_mode == "curve" else None,
        "temporal_reaction_progress_tracking": (
            reaction_progress_tracking if args.tracking_mode == "curve" else None
        ),
        "sparse_tracking": (
            None
            if checkpoint_tracking is None
            else tracking_points_metadata(
                checkpoint_tracking,
                tracking_variable=args.tracking_variable,
                table_path=(
                    None
                    if tracking_points_csv_path is None
                    else tracking_points_csv_path.name
                ),
            )
        ),
        "soft_doc": final_metrics["soft_doc"],
        "geometry": final_metrics["geometry"],
        "boundary": final_metrics["boundary"],
        "components": components_summary,
        "holes": holes_summary,
        "baseline_comparison": baseline_comparison,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    common_results: dict[str, object] = {
        "resolution_mode": np.asarray(config.resolution_mode),
        "native_shape": np.asarray(config.native_shape, dtype=np.int64),
        "optimization_shape": np.asarray(config.optimization_shape, dtype=np.int64),
        "coarsen_factor": np.int64(config.coarsen_factor),
        "native_pixel_pitch_m": np.float64(config.native_pixel_pitch_m),
        "optimization_pixel_pitch_m": np.float64(
            config.optimization_pixel_pitch_m
        ),
        "native_fov_y_m": np.float64(config.native_fov_m[0]),
        "native_fov_x_m": np.float64(config.native_fov_m[1]),
        "optimization_fov_y_m": np.float64(config.optimization_fov_m[0]),
        "optimization_fov_x_m": np.float64(config.optimization_fov_m[1]),
        "dt": np.float64(native_params.dt),
        "physics_steps_per_control": np.int64(args.physics_steps_per_control),
        "horizon": np.int64(args.horizon),
        "control_dt_s": np.float64(
            native_params.dt * args.physics_steps_per_control
        ),
        "prediction_horizon_seconds": np.float64(
            args.horizon * args.physics_steps_per_control * native_params.dt
        ),
        "total_process_time_s": np.float64(total_time_s),
        "control_steps": np.int64(args.control_steps),
        "geometry_threshold": np.float64(args.geometry_threshold),
        "geometry_threshold_note": np.asarray(GEOMETRY_THRESHOLD_NOTE),
        "control_times_s": primary_control_times_s,
        "reference_doc_values": primary_reference_doc,
        "reference_reaction_progress_values": primary_reference_reaction_progress,
        "actual_mean_target_doc": primary_target_doc,
        "actual_min_target_doc": primary_target_doc_min,
        "actual_max_target_doc": primary_target_doc_max,
        "actual_std_target_doc": primary_target_doc_std,
        "actual_target_doc_lower_error": primary_target_doc_lower_error,
        "actual_target_doc_upper_error": primary_target_doc_upper_error,
        "actual_mean_outside_doc": primary_outside_doc,
        "actual_mean_target_reaction_progress": primary_target_reaction_progress,
        "actual_min_target_reaction_progress": primary_target_reaction_progress_min,
        "actual_max_target_reaction_progress": primary_target_reaction_progress_max,
        "actual_std_target_reaction_progress": primary_target_reaction_progress_std,
        "reaction_progress_tracking_errors": primary_reaction_progress_tracking_error,
        "target_o2_mean": primary_target_o2,
        "target_o2_min": primary_target_o2_min,
        "target_o2_max": primary_target_o2_max,
        "target_o2_std": primary_target_o2_std,
        "target_o2_units": np.asarray("mJ/cm^2"),
        "temporal_tracking_errors": primary_tracking_error,
        "temporal_tracking_rmse": np.float64(tracking["rmse"]),
        "temporal_tracking_mae": np.float64(tracking["mae"]),
        "temporal_tracking_max_absolute_error": np.float64(
            tracking["max_absolute_error"]
        ),
        "temporal_reaction_progress_tracking_rmse": np.float64(
            np.nan
            if reaction_progress_tracking is None
            else reaction_progress_tracking["rmse"]
        ),
        "temporal_reaction_progress_tracking_mae": np.float64(
            np.nan
            if reaction_progress_tracking is None
            else reaction_progress_tracking["mae"]
        ),
        "temporal_reaction_progress_tracking_max_absolute_error": np.float64(
            np.nan
            if reaction_progress_tracking is None
            else reaction_progress_tracking["max_absolute_error"]
        ),
        "tracking_mode": np.asarray(args.tracking_mode),
        "tracking_variable": np.asarray(args.tracking_variable),
        "tracking_spatial_definition": np.asarray(
            args.tracking_spatial_definition
        ),
        "tracking_loss": np.asarray(args.tracking_loss),
        "tracking_specification_json": np.asarray(
            json.dumps(tracking_specification.provenance_metadata(), sort_keys=True)
        ),
        "tracking_point_times_s": np.asarray(
            tracking_specification.point_times_s, dtype=float
        ),
        "tracking_point_requested_doc": np.asarray(
            tracking_specification.point_values, dtype=float
        ),
        "tracking_point_requested_reaction_progress": (
            doc_array_to_reaction_progress_for_reporting(
                np.asarray(tracking_specification.point_values, dtype=float)
            )
        ),
        "tracking_point_weights": np.asarray(
            tracking_specification.point_weights
            or tuple(1.0 for _ in tracking_specification.point_times_s),
            dtype=float,
        ),
        "tracking_point_achieved_doc": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["achieved_doc"],
            dtype=float,
        ),
        "tracking_point_mean_target_doc": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["mean_target_doc"],
            dtype=float,
        ),
        "tracking_point_min_target_doc": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["min_target_doc"],
            dtype=float,
        ),
        "tracking_point_max_target_doc": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["max_target_doc"],
            dtype=float,
        ),
        "tracking_point_std_target_doc": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["std_target_doc"],
            dtype=float,
        ),
        "tracking_point_lower_error": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["lower_error"],
            dtype=float,
        ),
        "tracking_point_upper_error": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["upper_error"],
            dtype=float,
        ),
        "tracking_point_errors": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["errors"],
            dtype=float,
        ),
        "tracking_point_mean_target_reaction_progress": np.asarray(
            []
            if checkpoint_tracking is None
            else checkpoint_tracking["mean_target_reaction_progress"],
            dtype=float,
        ),
        "tracking_point_min_target_reaction_progress": np.asarray(
            []
            if checkpoint_tracking is None
            else checkpoint_tracking["min_target_reaction_progress"],
            dtype=float,
        ),
        "tracking_point_max_target_reaction_progress": np.asarray(
            []
            if checkpoint_tracking is None
            else checkpoint_tracking["max_target_reaction_progress"],
            dtype=float,
        ),
        "tracking_point_std_target_reaction_progress": np.asarray(
            []
            if checkpoint_tracking is None
            else checkpoint_tracking["std_target_reaction_progress"],
            dtype=float,
        ),
        "tracking_point_reaction_progress_error": np.asarray(
            []
            if checkpoint_tracking is None
            else checkpoint_tracking["reaction_progress_error"],
            dtype=float,
        ),
        "tracking_point_target_o2_mean": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["target_o2_mean"],
            dtype=float,
        ),
        "tracking_point_target_o2_min": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["target_o2_min"],
            dtype=float,
        ),
        "tracking_point_target_o2_max": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["target_o2_max"],
            dtype=float,
        ),
        "tracking_point_target_o2_std": np.asarray(
            [] if checkpoint_tracking is None else checkpoint_tracking["target_o2_std"],
            dtype=float,
        ),
        "doc_reference_metadata_json": np.asarray(
            json.dumps(
                None
                if doc_reference is None
                else doc_reference.provenance_metadata(),
                sort_keys=True,
            )
        ),
        "doc_reference_source_sha256": np.asarray(
            "" if doc_reference is None else doc_reference.source_sha256
        ),
        "doc_reference_id": np.asarray(
            "" if doc_reference is None else doc_reference.reference_id
        ),
        "doc_reference_curve_model": np.asarray(
            "" if doc_reference is None else doc_reference.curve_model
        ),
        "doc_reference_condition_json": np.asarray(
            json.dumps(
                None if doc_reference is None else doc_reference.metadata["condition"],
                sort_keys=True,
            )
        ),
        "reference_physics_match_json": np.asarray(
            json.dumps(physics_match, sort_keys=True)
        ),
        "metrics_json": np.asarray(json.dumps(metrics_document, sort_keys=True)),
        "initialization_json": np.asarray(
            json.dumps(initialization_metadata, sort_keys=True)
        ),
        "component_metrics_json": np.asarray(
            json.dumps(final_metrics["component_rows"], sort_keys=True)
        ),
        "hole_metrics_json": np.asarray(
            json.dumps(final_metrics["hole_rows"], sort_keys=True)
        ),
        "hole_mask_native": final_metrics["hole_mask"],
        "reference_model_source": np.asarray(native_params.reference_model_source),
        "reference_model_sha256": np.asarray(
            native_params.reference_model_sha256
        ),
        "reference_structure_sha256": np.asarray(
            native_params.reference_structure_sha256
        ),
        "model_structure_version": np.int64(
            native_params.model_structure_version
        ),
        "doc_calibration_source": np.asarray(
            native_params.doc_calibration_source
        ),
        "doc_calibration_sha256": np.asarray(
            native_params.doc_calibration_sha256
        ),
        "doc_fit_selection_status": np.asarray(
            native_params.doc_fit_selection_status
        ),
        "doc_fit_condition_id": np.asarray(
            native_params.doc_fit_condition_id or ""
        ),
        "doc_fit_formula_id": np.asarray(native_params.doc_fit_formula_id or ""),
        "doc_model_id": np.asarray(native_params.doc_model_id),
        "doc_history_mode": np.asarray(DOC_HISTORY_MODE),
        "doc_history_description": np.asarray(DOC_HISTORY_DESCRIPTION),
        "reference_config_json": np.asarray(
            json.dumps(reference.to_metadata(), sort_keys=True)
        ),
        "reference_parameters_json": np.asarray(
            json.dumps(native_params.provenance_metadata(), sort_keys=True)
        ),
        "optimization_parameters_json": np.asarray(
            json.dumps(optimization_params.provenance_metadata(), sort_keys=True)
        ),
        "target_native": target_native.detach().cpu().numpy(),
        "applied_controls_native": torch.stack(applied_controls_native).cpu().numpy(),
        "final_o2_native": native_state.o2.detach().cpu().numpy(),
        "final_tempo_native": native_state.tempo.detach().cpu().numpy(),
        "final_dose_native": native_state.dose.detach().cpu().numpy(),
        "final_reaction_progress_native": native_state.reaction_progress.detach()
        .cpu()
        .numpy(),
        "final_doc_native": native_state.doc.detach().cpu().numpy(),
        "loss_histories": np.asarray(optimization_histories),
    }
    if physics_initialization is not None:
        assert initial_guess_projector_native is not None
        assert initial_guess_local_intensity_native is not None
        common_results.update(
            {
                "initial_guess_projector_native": (
                    initial_guess_projector_native.detach().cpu().numpy()
                ),
                "initial_guess_local_normalized_intensity_native": (
                    initial_guess_local_intensity_native.detach().cpu().numpy()
                ),
                "initial_guess_component_metrics_json": np.asarray(
                    json.dumps(initializer_component_rows, sort_keys=True)
                ),
            }
        )
    if checkpoint_tracking is not None:
        prefix = "checkpoint" if args.tracking_mode == "checkpoints" else "sampled_curve"
        common_results.update(
            {
                f"{prefix}_times_s": checkpoint_tracking["times_s"],
                f"{prefix}_requested_doc": checkpoint_tracking["requested_doc"],
                f"{prefix}_achieved_doc": checkpoint_tracking["achieved_doc"],
                f"{prefix}_mean_target_doc": checkpoint_tracking["mean_target_doc"],
                f"{prefix}_min_target_doc": checkpoint_tracking["min_target_doc"],
                f"{prefix}_max_target_doc": checkpoint_tracking["max_target_doc"],
                f"{prefix}_std_target_doc": checkpoint_tracking["std_target_doc"],
                f"{prefix}_lower_error": checkpoint_tracking["lower_error"],
                f"{prefix}_upper_error": checkpoint_tracking["upper_error"],
                f"{prefix}_errors": checkpoint_tracking["errors"],
                f"{prefix}_requested_reaction_progress": checkpoint_tracking[
                    "requested_reaction_progress"
                ],
                f"{prefix}_mean_target_reaction_progress": checkpoint_tracking[
                    "mean_target_reaction_progress"
                ],
                f"{prefix}_min_target_reaction_progress": checkpoint_tracking[
                    "min_target_reaction_progress"
                ],
                f"{prefix}_max_target_reaction_progress": checkpoint_tracking[
                    "max_target_reaction_progress"
                ],
                f"{prefix}_std_target_reaction_progress": checkpoint_tracking[
                    "std_target_reaction_progress"
                ],
                f"{prefix}_reaction_progress_error": checkpoint_tracking[
                    "reaction_progress_error"
                ],
                f"{prefix}_target_o2_mean": checkpoint_tracking["target_o2_mean"],
                f"{prefix}_target_o2_min": checkpoint_tracking["target_o2_min"],
                f"{prefix}_target_o2_max": checkpoint_tracking["target_o2_max"],
                f"{prefix}_target_o2_std": checkpoint_tracking["target_o2_std"],
                f"{prefix}_rmse": np.float64(checkpoint_tracking["rmse"]),
                f"{prefix}_mae": np.float64(checkpoint_tracking["mae"]),
                f"{prefix}_max_abs_error": np.float64(
                    checkpoint_tracking["max_absolute_error"]
                ),
                f"{prefix}_reaction_progress_rmse": np.float64(
                    np.nan
                    if checkpoint_tracking["reaction_progress_rmse"] is None
                    else checkpoint_tracking["reaction_progress_rmse"]
                ),
                f"{prefix}_reaction_progress_mae": np.float64(
                    np.nan
                    if checkpoint_tracking["reaction_progress_mae"] is None
                    else checkpoint_tracking["reaction_progress_mae"]
                ),
                f"{prefix}_reaction_progress_max_abs_error": np.float64(
                    np.nan
                    if checkpoint_tracking[
                        "reaction_progress_max_absolute_error"
                    ]
                    is None
                    else checkpoint_tracking[
                        "reaction_progress_max_absolute_error"
                    ]
                ),
            }
        )
    if config.resolution_mode == "coarse":
        common_results.update(
            {
                "target_coarse": optimization_target.detach().cpu().numpy(),
                "applied_controls_coarse": torch.stack(
                    applied_controls_optimization
                )
                .cpu()
                .numpy(),
                "final_o2_coarse": optimization_state.o2.detach().cpu().numpy(),
                "final_tempo_coarse": optimization_state.tempo.detach()
                .cpu()
                .numpy(),
                "final_dose_coarse": optimization_state.dose.detach().cpu().numpy(),
                "final_reaction_progress_coarse": (
                    optimization_state.reaction_progress.detach().cpu().numpy()
                ),
                "final_doc_coarse": optimization_state.doc.detach().cpu().numpy(),
            }
        )
    results_name = f"mpc_results_{config.resolution_mode}.npz"
    np.savez_compressed(args.output_dir / results_name, **common_results)

    print_metrics_summary(
        reference_id=(
            physics_condition if doc_reference is None else doc_reference.reference_id
        ),
        tracking_mode=args.tracking_mode,
        tracking_variable=args.tracking_variable,
        total_time_s=total_time_s,
        tracking=tracking,
        checkpoint_tracking=checkpoint_tracking,
        reference_final_doc=reference_final_doc,
        target_mean_doc=native_target_doc,
        target_min_doc=native_target_summary["min_target_doc"],
        target_max_doc=native_target_summary["max_target_doc"],
        target_std_doc=native_target_summary["std_target_doc"],
        outside_mean_doc=native_outside_doc,
        final_metrics=final_metrics,
    )
    if baseline_result is not None:
        print_baseline_comparison(baseline_comparison)

    print(f"resolution_mode={config.resolution_mode}")
    print(f"native_grid={config.native_shape}")
    print(f"optimization_grid={config.optimization_shape}")
    print(f"coarsen_factor={config.coarsen_factor}")
    if config.resolution_mode == "coarse":
        print(
            "native replay "
            "DoC(target mean/min/max/std/out)="
            f"{native_target_doc:.4f}/{native_target_summary['min_target_doc']:.4f}/"
            f"{native_target_summary['max_target_doc']:.4f}/"
            f"{native_target_summary['std_target_doc']:.4f}/{native_outside_doc:.4f}"
        )
    else:
        print(
            "native DoC(target mean/min/max/std/out)="
            f"{native_target_doc:.4f}/{native_target_summary['min_target_doc']:.4f}/"
            f"{native_target_summary['max_target_doc']:.4f}/"
            f"{native_target_summary['std_target_doc']:.4f}/{native_outside_doc:.4f}"
        )
    if coarse_gif_path is not None:
        print(f"saved coarse DoC GIF: {coarse_gif_path.resolve()}")
        print(f"saved native replay DoC GIF: {native_gif_path.resolve()}")
    else:
        print(f"saved native DoC GIF: {native_gif_path.resolve()}")
    print(f"saved tracking plot: {tracking_plot_path.resolve()}")
    print(f"saved component metrics: {component_metrics_path.resolve()}")
    if tracking_points_csv_path is not None:
        print(f"saved tracking-point statistics: {tracking_points_csv_path.resolve()}")
    if baseline_result is not None:
        print(f"saved baseline outputs: {baseline_result.output_dir.resolve()}")
    print(f"saved metrics summary: {metrics_path.resolve()}")
    print(f"saved outputs to {args.output_dir.resolve()}")


def _legacy_full_gaussian(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Full 2-D convolution form used verbatim by the legacy script."""

    padding = kernel.shape[-1] // 2
    padded = F.pad(
        field[None, None],
        (padding, padding, padding, padding),
        mode="reflect",
    )
    return F.conv2d(padded, kernel)[0, 0]


def _reference_architecture_test(device: torch.device) -> dict[str, object]:
    """Validate AST extraction, fingerprints, calibration, and reference-only defaults."""

    with tempfile.TemporaryDirectory(prefix="aie_reference_parse_test_") as directory:
        original_directory = Path.cwd()
        capture = StringIO()
        try:
            os.chdir(directory)
            with redirect_stdout(capture):
                reference = load_reference_config()
        finally:
            os.chdir(original_directory)
        if capture.getvalue():
            raise AssertionError(
                "read-only reference parsing printed output: "
                f"{capture.getvalue()!r}"
            )
        if list(Path(directory).iterdir()):
            raise AssertionError("read-only reference parsing created files")

    expected_reference_hash = hashlib.sha256(
        aie_reference.REFERENCE_MODEL_PATH.read_bytes()
    ).hexdigest()
    expected_calibration_hash = hashlib.sha256(
        aie_reference.DOC_FIT_PATH.read_bytes()
    ).hexdigest()
    if reference.reference_model_sha256 != expected_reference_hash:
        raise AssertionError("reference SHA-256 does not match source bytes")
    if reference.doc_calibration_sha256 != expected_calibration_hash:
        raise AssertionError("calibration SHA-256 does not match JSON bytes")

    params = AIEParameters.from_reference(reference)
    mappings = {
        "native_pixel_pitch_m": "native_pixel_pitch_m",
        "projector_refinement": "projector_refinement",
        "dt": "dt",
        "total_simulation_time_s": "total_simulation_time_s",
        "loss_sample_times_s": "loss_sample_times_s",
        "intensity_mw_cm2": "intensity_mw_cm2",
        "tempo_concentration_mM": "tempo_concentration_mM",
        "o2_diffusivity_m2_s": "o2_diffusivity_m2_s",
        "tempo_diffusivity_m2_s": "tempo_diffusivity_m2_s",
        "o2_inhibition_mj_cm2": "o2_inhibition_mj_cm2",
        "total_inhibition_mj_cm2": "total_inhibition_mj_cm2",
        "scattering_blur_size_m": "scattering_blur_size_m",
        "tempo_gaussian_sigma_scale": "tempo_gaussian_sigma_scale",
        "o2_diffusion_enabled": "o2_diffusion_enabled",
        "tempo_diffusion_enabled": "tempo_diffusion_enabled",
        "chain_growth_noise_std": "chain_growth_noise_std",
        "chain_growth_noise_enabled": "chain_growth_noise_enabled",
        "b_slope": "b_slope",
        "b_intercept": "b_intercept",
        "b_condition_label": "b_condition_label",
        "minimum_normalized_intensity": "minimum_normalized_intensity",
        "division_epsilon": "division_epsilon",
        "mask_grayscale_max": "mask_grayscale_max",
        "reference_model_path": "reference_model_path",
        "reference_model_sha256": "reference_model_sha256",
        "reference_structure_sha256": "reference_structure_sha256",
        "doc_calibration_path": "doc_calibration_path",
        "doc_calibration_sha256": "doc_calibration_sha256",
        "doc_model_id": "doc_model_id",
        "doc_model_formula": "doc_model_formula",
    }
    for parameter_name, reference_name in mappings.items():
        if getattr(params, parameter_name) != getattr(reference, reference_name):
            raise AssertionError(
                f"AIEParameters.{parameter_name} does not match "
                f"ReferenceConfig.{reference_name}"
            )
    if params.pixel_pitch_m != reference.native_pixel_pitch_m:
        raise AssertionError("native model pitch does not match the reference")
    if params.native_shape != tuple(reference.native_shape):
        raise AssertionError("native model shape does not match the reference")

    try:
        AIEParameters()
    except TypeError:
        pass
    else:
        raise AssertionError("AIEParameters silently constructed stale defaults")
    default_model = AIEModel(device=device)
    if default_model.params != params:
        raise AssertionError("AIEModel() did not resolve current reference parameters")

    calibration_document = json.loads(
        aie_reference.DOC_FIT_PATH.read_text(encoding="utf-8")
    )
    _, calibration_catalog = aie_reference._load_doc_fit_catalog()
    exported_by_id = {
        condition["condition_id"]: condition
        for condition in calibration_document["conditions"]
    }
    if tuple(exported_by_id) != reference.available_doc_fit_condition_ids:
        raise AssertionError("reference calibration catalog is incomplete")
    for condition in calibration_catalog:
        exported = exported_by_id[condition.condition_id]
        for coefficient in ("a", "b", "c"):
            if getattr(condition, coefficient) != exported["average"][coefficient]:
                raise AssertionError(
                    f"{condition.condition_id}.{coefficient} differs from export"
                )
    if reference.tempo_concentration_mM is None:
        if reference.doc_fit is not None:
            raise AssertionError(
                "a/b/c was selected despite no resolved active TEMPO condition"
            )
    else:
        if reference.doc_fit is None:
            raise AssertionError(
                "matching collaborator a/b/c provenance was not selected"
            )
        if not math.isclose(
            reference.doc_fit.intensity_mw_cm2,
            reference.intensity_mw_cm2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            reference.doc_fit.tempo_concentration_mM,
            reference.tempo_concentration_mM,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("selected a/b/c provenance does not match active physics")
    if reference.doc_fit_applied_to_governing_law:
        raise AssertionError("B-based reference physics unexpectedly applies a/b/c")

    def expect_reference_error(
        error_type: type[Exception], message_fragment: str
    ) -> None:
        try:
            load_reference_config()
        except error_type as error:
            if message_fragment not in str(error):
                raise AssertionError(f"unexpected reference error: {error}") from error
        else:
            raise AssertionError(
                f"reference validation did not raise {error_type.__name__}"
            )

    original_reference_path = aie_reference.REFERENCE_MODEL_PATH
    original_calibration_path = aie_reference.DOC_FIT_PATH
    safety_checks = 0
    with tempfile.TemporaryDirectory(prefix="aie_reference_safety_test_") as directory:
        safety_directory = Path(directory)
        try:
            aie_reference.REFERENCE_MODEL_PATH = safety_directory / "missing.py"
            expect_reference_error(FileNotFoundError, "reference model is missing")
            safety_checks += 1

            source_tree = ast.parse(
                original_reference_path.read_text(encoding="utf-8")
            )
            changed = False
            for statement in source_tree.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "intensity"
                    for target in statement.targets
                ):
                    statement.value = ast.Constant(reference.intensity_mw_cm2 + 1.25)
                    changed = True
                    break
            if not changed:
                raise AssertionError("test could not locate top-level intensity")
            parameter_path = safety_directory / "parameter_change.py"
            parameter_path.write_text(
                ast.unparse(ast.fix_missing_locations(source_tree)) + "\n",
                encoding="utf-8",
            )
            aie_reference.REFERENCE_MODEL_PATH = parameter_path
            changed_reference = load_reference_config()
            if changed_reference.intensity_mw_cm2 != reference.intensity_mw_cm2 + 1.25:
                raise AssertionError("numeric reference change did not propagate")
            safety_checks += 1

            structural_tree = ast.parse(
                original_reference_path.read_text(encoding="utf-8")
            )
            changed = False
            for node in ast.walk(structural_tree):
                if not isinstance(node, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "Dosenext"
                    for target in node.targets
                ):
                    node.value = ast.Name(id="Dose", ctx=ast.Load())
                    changed = True
                    break
            if not changed:
                raise AssertionError("test could not locate Dosenext")
            structural_path = safety_directory / "structural_change.py"
            structural_path.write_text(
                ast.unparse(ast.fix_missing_locations(structural_tree)) + "\n",
                encoding="utf-8",
            )
            aie_reference.REFERENCE_MODEL_PATH = structural_path
            expect_reference_error(
                aie_reference.UnsupportedReferencePhysicsError,
                "unsupported active Dosenext",
            )
            safety_checks += 1
            aie_reference.REFERENCE_MODEL_PATH = original_reference_path

            aie_reference.DOC_FIT_PATH = safety_directory / "missing.json"
            expect_reference_error(FileNotFoundError, "calibration export is missing")
            safety_checks += 1

            invalid_documents: list[tuple[str, dict[str, object], str]] = []
            invalid_schema = json.loads(json.dumps(calibration_document))
            invalid_schema["schema_version"] = 999
            invalid_documents.append(("schema.json", invalid_schema, "schema_version"))
            missing_coefficient = json.loads(json.dumps(calibration_document))
            del missing_coefficient["conditions"][0]["average"]["a"]
            invalid_documents.append(
                ("missing_a.json", missing_coefficient, "must contain a, b, and c")
            )
            nonfinite = json.loads(json.dumps(calibration_document))
            nonfinite["conditions"][0]["average"]["a"] = float("nan")
            invalid_documents.append(("nan.json", nonfinite, "must be finite"))
            for filename, invalid_document, expected_message in invalid_documents:
                invalid_path = safety_directory / filename
                invalid_path.write_text(
                    json.dumps(invalid_document), encoding="utf-8"
                )
                aie_reference.DOC_FIT_PATH = invalid_path
                expect_reference_error(ValueError, expected_message)
                safety_checks += 1

            valid_path = safety_directory / "valid.json"
            valid_path.write_text(
                json.dumps(calibration_document), encoding="utf-8"
            )
            aie_reference.DOC_FIT_PATH = valid_path
            _, valid_catalog = aie_reference._load_doc_fit_catalog()
            unresolved, status = aie_reference._select_doc_fit(
                valid_catalog, 999.0, 0.0
            )
            if unresolved is not None or not status.startswith("no_exact_fit"):
                raise AssertionError("unresolved calibration was silently selected")
            safety_checks += 1
        finally:
            aie_reference.REFERENCE_MODEL_PATH = original_reference_path
            aie_reference.DOC_FIT_PATH = original_calibration_path

    coarse_config = build_resolution_config(
        "coarse", 2, tuple(reference.native_shape), reference=reference
    )
    coarse_params = replace(
        params, pixel_pitch_m=coarse_config.optimization_pixel_pitch_m
    )
    if coarse_params.pixel_pitch_m != 2 * params.native_pixel_pitch_m:
        raise AssertionError("q=2 coarse pitch is not twice the reference pitch")
    for parameter_field in fields(AIEParameters):
        if parameter_field.name == "pixel_pitch_m":
            continue
        if getattr(coarse_params, parameter_field.name) != getattr(
            params, parameter_field.name
        ):
            raise AssertionError(
                f"coarsening changed reference field {parameter_field.name}"
            )
    return {
        "condition_id": params.doc_fit_condition_id or "none_selected",
        "fit_selection_status": params.doc_fit_selection_status,
        "reference_sha256": params.reference_model_sha256,
        "calibration_sha256": params.doc_calibration_sha256,
        "native_pitch_um": params.native_pixel_pitch_m * 1e6,
        "coarse_pitch_um": coarse_params.pixel_pitch_m * 1e6,
        "field_count": len(fields(AIEParameters)),
        "safety_checks": safety_checks,
    }


def _legacy_reference_step(
    model: AIEModel, state: AIEState, normalized_mask: torch.Tensor
) -> AIEState:
    """Run the independent fixture for the AST-validated reference equations."""

    reference_config = load_reference_config()
    result = reference_step_torch(
        o2=state.o2,
        tempo=state.tempo,
        dose=state.dose,
        doc=state.doc,
        normalized_mask=normalized_mask,
        scattering_kernel_2d=model.scattering_kernel_2d,
        o2_kernel_2d=model.o2_kernel_2d,
        tempo_kernel_2d=model.tempo_kernel_2d,
        chain_growth_multiplier=state.chain_growth_multiplier,
        config=reference_config,
    )
    return AIEState(
        o2=result.o2,
        tempo=result.tempo,
        dose=result.dose,
        reaction_progress=state.reaction_progress,
        doc=result.doc,
        chain_growth_multiplier=state.chain_growth_multiplier,
    )


def _static_mask_equivalence_test(device: torch.device) -> float:
    model = AIEModel(device=device)
    largest_kernel = max(
        model.scattering_kernel_1d.numel(),
        model.o2_kernel_1d.numel(),
        model.tempo_kernel_1d.numel(),
    )
    height = width = int(largest_kernel // 2 + 3)
    coordinates = torch.linspace(0, 1, height, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    mask = 0.25 + 0.65 * torch.exp(
        -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.08
    )
    initialized = model.initialize_state((height, width))
    prepared = model.prepare_control(mask, initialized.shape)
    initial_dose = 0.03 + 0.05 * xx * yy
    effective_b = prepared.b
    if initialized.chain_growth_multiplier is not None:
        effective_b = effective_b * initialized.chain_growth_multiplier
    initial_progress = (
        effective_b
        * initial_dose
        / prepared.local_intensity.clamp_min(model.params.division_epsilon)
    )
    initial = AIEState(
        o2=0.05 + 0.20 * xx,
        tempo=0.02 + 0.18 * yy,
        dose=initial_dose,
        reaction_progress=initial_progress,
        doc=1.0 - torch.exp(-torch.clamp(initial_progress, min=0.0)),
        chain_growth_multiplier=initialized.chain_growth_multiplier,
    )
    refactored, reference = initial, initial
    for _ in range(2):
        refactored = model.step(refactored, mask)
        reference = _legacy_reference_step(model, reference, mask)
    maximum_error = max(
        float((actual - expected).abs().max())
        for actual, expected in zip(
            refactored.reference_tensors(), reference.reference_tensors()
        )
    )
    for actual, expected in zip(
        refactored.reference_tensors(), reference.reference_tensors()
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    effective_b = prepared.b
    if refactored.chain_growth_multiplier is not None:
        effective_b = effective_b * refactored.chain_growth_multiplier
    expected_progress = (
        effective_b
        * refactored.dose
        / prepared.local_intensity.clamp_min(model.params.division_epsilon)
    )
    torch.testing.assert_close(
        refactored.reaction_progress,
        expected_progress,
        rtol=2e-5,
        atol=2e-6,
    )
    return maximum_error


def _physics_synchronization_test(device: torch.device) -> dict[str, object]:
    """Check timing, diffusion, scattering, and active B against the adapter."""

    reference = load_reference_config()
    model = AIEModel(AIEParameters.from_reference(reference), device=device)
    largest_kernel = max(
        model.o2_kernel_1d.numel(),
        model.tempo_kernel_1d.numel(),
        model.scattering_kernel_1d.numel(),
    )
    size = int(largest_kernel // 2 + 4)
    coordinates = torch.linspace(0.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def expected_gaussian(kernel_size: int, sigma: float) -> torch.Tensor:
        locations = torch.arange(kernel_size, dtype=torch.float64, device=device)
        locations = locations - (kernel_size - 1) / 2
        kernel = torch.exp(-locations.square() / (2 * sigma**2))
        return (kernel / kernel.sum()).to(dtype=model.dtype)

    o2_sigma_pixels = math.sqrt(
        2 * reference.o2_diffusivity_m2_s * reference.dt
    ) / reference.native_pixel_pitch_m
    tempo_sigma_pixels = math.sqrt(
        2 * reference.tempo_diffusivity_m2_s * reference.dt
    ) / reference.native_pixel_pitch_m
    torch.testing.assert_close(
        model.o2_kernel_1d,
        expected_gaussian(model.o2_kernel_1d.numel(), o2_sigma_pixels),
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        model.tempo_kernel_1d,
        expected_gaussian(
            model.tempo_kernel_1d.numel(),
            tempo_sigma_pixels * reference.tempo_gaussian_sigma_scale,
        ),
        rtol=2e-6,
        atol=2e-7,
    )

    nonuniform_o2 = 0.1 + xx.square() + 0.25 * yy
    o2_diffused = model._gaussian_blur(
        nonuniform_o2, model.o2_kernel_1d, "O2 synchronization test"
    )
    if reference.o2_diffusion_enabled and model.o2_kernel_1d.numel() > 1:
        if bool(torch.equal(o2_diffused, nonuniform_o2)):
            raise AssertionError("active reference O2 diffusion did not alter a field")

    nonuniform_tempo = 0.2 + yy.square() + 0.15 * xx
    tempo_diffused = model._gaussian_blur(
        nonuniform_tempo,
        model.tempo_kernel_1d,
        "TEMPO synchronization test",
    )
    if reference.tempo_diffusion_enabled and model.tempo_kernel_1d.numel() > 1:
        if bool(torch.equal(tempo_diffused, nonuniform_tempo)):
            raise AssertionError("active reference TEMPO diffusion did not alter a field")

    mask = 0.1 + 0.8 * xx * yy
    prepared = model.prepare_control(mask, mask.shape)
    scattering_reference = _legacy_full_gaussian(
        mask, model.scattering_kernel_2d
    )
    torch.testing.assert_close(
        prepared.scattered_mask, scattering_reference, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        prepared.b,
        reference.b_slope * prepared.local_intensity + reference.b_intercept,
        rtol=1e-6,
        atol=1e-7,
    )
    zero_prepared = model.prepare_control(torch.zeros_like(mask), mask.shape)
    expected_zero_intensity = (
        reference.minimum_normalized_intensity * reference.intensity_mw_cm2
    )
    torch.testing.assert_close(
        zero_prepared.local_intensity,
        torch.full_like(zero_prepared.local_intensity, expected_zero_intensity),
        rtol=1e-6,
        atol=0.0,
    )

    default_physics_steps = 10
    default_horizon = 8
    control_dt = default_physics_steps * reference.dt
    prediction_horizon = default_horizon * control_dt
    if control_dt != model.params.dt * default_physics_steps:
        raise AssertionError("reference dt did not propagate to control timing")
    return {
        "dt": reference.dt,
        "control_dt": control_dt,
        "prediction_horizon": prediction_horizon,
        "o2_diffusion_enabled": reference.o2_diffusion_enabled,
        "tempo_diffusion_enabled": reference.tempo_diffusion_enabled,
        "o2_kernel_size": model.o2_kernel_1d.numel(),
        "tempo_kernel_size": model.tempo_kernel_1d.numel(),
        "tempo_sigma_scale": reference.tempo_gaussian_sigma_scale,
        "scattering_kernel_size": model.scattering_kernel_size,
        "b_slope": reference.b_slope,
        "b_intercept": reference.b_intercept,
    }


def _gradient_test(device: torch.device) -> tuple[float, int]:
    # The reference O2 threshold creates a pre-cure zero-gradient interval.
    # This TEST-ONLY override exercises ten-step differentiability without
    # changing any transition equation or production construction path.
    params = replace(
        AIEParameters.from_reference(), o2_inhibition_mj_cm2=0.0
    )
    model = AIEModel(params, device=device)
    shape = (model.scattering_kernel_size // 2 + 3,) * 2
    mask = torch.nn.Parameter(torch.full(shape, 0.7, device=device))
    state = model.initialize_state(shape)
    for _ in range(10):
        state = model.step(state, mask)
    state.doc.mean().backward()
    if mask.grad is None:
        raise AssertionError("mask.grad is None")
    if not bool(torch.isfinite(mask.grad).all()):
        raise AssertionError("mask gradient contains NaN or Inf")
    nonzero = int(torch.count_nonzero(mask.grad))
    if nonzero == 0:
        raise AssertionError("mask gradient is zero in the exposed region")
    return float(mask.grad.norm()), nonzero


def _shape_validation_test(device: torch.device) -> str:
    model = AIEModel(device=device)
    size = model.scattering_kernel_size // 2 + 3
    state = model.initialize_state((size, size))
    try:
        model.step(state, torch.zeros((size - 1, size), device=device))
    except ValueError as error:
        if "projector_mask must have shape" not in str(error):
            raise AssertionError(f"unexpected shape error: {error}") from error
        return str(error)
    raise AssertionError("incorrect mask shape was not rejected")


def _mpc_optimization_test(device: torch.device) -> tuple[float, float]:
    params = replace(
        AIEParameters.from_reference(), o2_inhibition_mj_cm2=0.0
    )
    model = AIEModel(params, device=device)
    size = model.scattering_kernel_size // 2 + 3
    target = torch.ones((size, size), device=device)
    state = model.initialize_state(target.shape)
    controller = DifferentiableMPC(
        model=model,
        target=target,
        reference_curve=load_doc_reference(),
        horizon=2,
        physics_steps_per_control=5,
        outside_weight=0.0,
        energy_weight=0.0,
        smoothness_weight=0.0,
        num_iterations=10,
        learning_rate=0.3,
    )
    guess = torch.full((2, size, size), 0.2, device=device)
    _, info = controller.optimize(
        state, initial_guess=guess, current_process_time_s=1.5
    )
    initial_loss = info["initial_loss"]
    final_loss = info["final_loss"]
    if not final_loss < initial_loss:
        raise AssertionError(
            f"MPC loss did not decrease: {initial_loss} -> {final_loss}"
        )
    return initial_loss, final_loss


def _legacy_isotonic_reference_regression_test() -> dict[str, object]:
    """Validate both conditions, raw-tail handling, interpolation, and selection."""

    ids = available_doc_reference_ids()
    if ids != ("30mW_0mM", "30mW_5mM"):
        raise AssertionError(f"unexpected DoC reference IDs: {ids}")
    references = {reference_id: load_doc_reference(reference_id) for reference_id in ids}
    block_counts: dict[str, int] = {}
    for reference_id, reference in references.items():
        if reference.metadata["schema_version"] != 2:
            raise AssertionError("unexpected DoC reference schema")
        if reference.time_s[0] != 0.0 or reference.time_s[-1] != 20.0:
            raise AssertionError("DoC reference does not span exactly 0 to 20 s")
        if not np.isfinite(reference.doc_reference).all():
            raise AssertionError("DoC reference contains NaN or Inf")
        if np.any(np.diff(reference.doc_reference) < -1e-12):
            raise AssertionError("DoC reference is not monotonic nondecreasing")
        if np.min(reference.doc_reference) < 0 or np.max(reference.doc_reference) > 1:
            raise AssertionError("DoC reference left [0,1]")
        if (
            reference.metadata["production_reference_method"]
            != PRODUCTION_REFERENCE_METHOD
            or reference.metadata["selected_fit_model"]
            != "isotonic_monotonic_benchmark"
        ):
            raise AssertionError("production reference is not equal-replicate isotonic")
        if reference.metadata["equal_replicate_construction"][
            "replicate_weights"
        ] != {"T1": 0.5, "T2": 0.5}:
            raise AssertionError("T1 and T2 do not have equal production influence")
        if reference.metadata["compact_diagnostic_fit"]["model_id"] != (
            "delayed_avrami_fixed_plateau"
        ):
            raise AssertionError("compact Avrami diagnostic was not preserved")
        block_counts[reference_id] = int(
            reference.metadata["isotonic_regression"]["block_count"]
        )
        plateau = reference.time_s >= reference.saturation_time_s - 1e-12
        if not np.all(reference.doc_reference[plateau] == 1.0):
            raise AssertionError("DoC reference does not stay exactly one after saturation")
        for replicate in reference.metadata["replicate_fits"].values():
            if replicate["excluded_post_saturation_sample_count"] <= 0:
                raise AssertionError("post-saturation raw tail was not excluded")
            if not replicate["raw_normalized_final"] < 0.1:
                raise AssertionError("test data do not contain the audited declining tail")
            errors = [
                replicate["fit"]["threshold_time_errors_s"][key]
                for key in ("t10_s", "t30_s", "t50_s", "t90_s", "t95_s")
            ]
            if max(abs(value) for value in errors if value is not None) > 0.20:
                raise AssertionError("raw-versus-fit threshold timing exceeds 0.20 s")
        for comparison in reference.metadata[
            "production_threshold_comparison_s"
        ].values():
            errors = comparison["production_minus_raw_errors_s"]
            if max(
                abs(errors[key])
                for key in ("t10_s", "t30_s", "t50_s", "t90_s", "t95_s")
                if errors[key] is not None
            ) > 0.20:
                raise AssertionError(
                    "raw-versus-isotonic production timing exceeds 0.20 s"
                )
        left_index = int(np.searchsorted(reference.time_s, 2.025) - 1)
        expected_midpoint = 0.5 * (
            reference.doc_reference[left_index]
            + reference.doc_reference[left_index + 1]
        )
        if not math.isclose(
            float(reference.at(2.025)),
            float(expected_midpoint),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("DoC reference linear interpolation is incorrect")
        if reference.at(-1.0) != reference.doc_reference[0]:
            raise AssertionError("DoC reference does not hold its first endpoint")
        if reference.at(25.0) != reference.doc_reference[-1]:
            raise AssertionError("DoC reference does not hold its final plateau")
    if math.isclose(
        float(references["30mW_0mM"].at(5.0)),
        float(references["30mW_5mM"].at(5.0)),
        rel_tol=0.0,
        abs_tol=1e-3,
    ):
        raise AssertionError("switching reference condition did not change r(t)")
    try:
        load_doc_reference("not_a_condition")
    except ValueError as error:
        if "available IDs" not in str(error):
            raise AssertionError(f"unclear wrong-ID error: {error}") from error
    else:
        raise AssertionError("unknown DoC reference ID was accepted")
    with tempfile.TemporaryDirectory() as directory:
        invalid_path = Path(directory) / "invalid_reference.json"
        invalid = json.loads(Path(references["30mW_0mM"].source_path).read_text())
        invalid["references"]["30mW_0mM"]["condition"]["intensity_mw_cm2"] = 20.0
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        try:
            load_doc_reference("30mW_0mM", invalid_path)
        except ValueError as error:
            if "30" not in str(error):
                raise AssertionError(f"unexpected condition validation error: {error}")
        else:
            raise AssertionError("incorrect DoC reference condition was accepted")
    return {
        "ids": ids,
        "samples": int(references["30mW_0mM"].time_s.size),
        "final_doc": references["30mW_0mM"].final_doc,
        "condition_delta_at_5s": float(
            references["30mW_0mM"].at(5.0)
            - references["30mW_5mM"].at(5.0)
        ),
        "block_counts": block_counts,
    }


def _doc_reference_curve_test() -> dict[str, object]:
    """Validate schema-v3 model selection and non-destructive schema-v2 support."""

    ids = available_doc_reference_ids(DEFAULT_REFERENCE_PATH)
    if ids != ("30mW_0mM", "30mW_5mM"):
        raise AssertionError(f"unexpected DoC reference IDs: {ids}")
    values_at_5s: dict[str, dict[str, float]] = {}
    for reference_id in ids:
        models = available_curve_models(reference_id, DEFAULT_REFERENCE_PATH)
        if models != CURVE_MODEL_IDS:
            raise AssertionError(f"incomplete curve-model catalog for {reference_id}: {models}")
        values_at_5s[reference_id] = {}
        for model_id in models:
            reference = load_doc_reference(
                reference_id, DEFAULT_REFERENCE_PATH, curve_model=model_id
            )
            if reference.time_s[0] != 0.0 or reference.time_s[-1] != 20.0:
                raise AssertionError("DoC reference does not span 0 to 20 s")
            if not np.isfinite(reference.doc_reference).all():
                raise AssertionError("DoC reference contains NaN or Inf")
            if np.any(np.diff(reference.doc_reference) < -1e-10):
                raise AssertionError(f"{model_id} reference decreases")
            if float(reference.doc_reference.min()) < 0 or float(reference.doc_reference.max()) > 1:
                raise AssertionError(f"{model_id} reference left [0,1]")
            values_at_5s[reference_id][model_id] = float(reference.at(5.0))
        default_reference = load_doc_reference(reference_id)
        if default_reference.curve_model != "collaborator_original":
            raise AssertionError("schema-v3 default is not collaborator_original")
        p = default_reference.model_parameters
        assert p is not None
        expected = 1.0 - p["a"] * np.exp(
            np.minimum(p["b"] * (p["c"] - default_reference.time_s), 0.0)
        )
        np.testing.assert_allclose(
            default_reference.doc_reference, expected, rtol=0.0, atol=1e-12
        )

    legacy_hash_before = hashlib.sha256(LEGACY_REFERENCE_PATH.read_bytes()).hexdigest()
    legacy = load_doc_reference("30mW_0mM", LEGACY_REFERENCE_PATH)
    if legacy.curve_model != "isotonic" or legacy.metadata["source_schema_version"] != 2:
        raise AssertionError("schema-v2 runtime migration did not preserve isotonic default")
    if "collaborator_original" in available_curve_models(
        "30mW_0mM", LEGACY_REFERENCE_PATH
    ):
        raise AssertionError("schema-v2 migration synthesized collaborator coefficients")
    legacy_hash_after = hashlib.sha256(LEGACY_REFERENCE_PATH.read_bytes()).hexdigest()
    if legacy_hash_before != legacy_hash_after:
        raise AssertionError("schema-v2 compatibility rewrote the source artifact")
    try:
        load_doc_reference("not_a_condition")
    except ValueError as error:
        if "available IDs" not in str(error):
            raise AssertionError(f"unclear wrong-ID error: {error}") from error
    else:
        raise AssertionError("unknown DoC reference ID was accepted")
    return {
        "ids": ids,
        "models": CURVE_MODEL_IDS,
        "default_model": "collaborator_original",
        "legacy_default_model": legacy.curve_model,
        "legacy_sha256": legacy_hash_after,
        "condition_delta_at_5s": (
            values_at_5s["30mW_0mM"]["collaborator_original"]
            - values_at_5s["30mW_5mM"]["collaborator_original"]
        ),
    }


def _trajectory_tracking_construction_test(device: torch.device) -> dict[str, object]:
    """Validate absolute-time stage references, spatial construction, and gradients."""

    reference = load_doc_reference()
    params = replace(AIEParameters.from_reference(), o2_inhibition_mj_cm2=0.0)
    model = AIEModel(params, device=device)
    size = model.scattering_kernel_size // 2 + 3
    target = torch.zeros((size, size), device=device)
    target[1:-1, 1:-1] = 1.0
    controller = DifferentiableMPC(
        model=model,
        target=target,
        reference_curve=reference,
        horizon=8,
        physics_steps_per_control=10,
        outside_weight=0.5,
        energy_weight=1e-4,
        smoothness_weight=1e-2,
        num_iterations=1,
    )
    stage_times = reference.stage_times(6.0, controller.control_dt_s, 8)
    expected_times = np.arange(6.5, 10.0 + 0.25, 0.5)
    np.testing.assert_allclose(stage_times, expected_times, rtol=0.0, atol=1e-12)
    stage_reference = reference.stage_values(6.0, controller.control_dt_s, 8)
    restarted_reference = reference.stage_values(0.0, controller.control_dt_s, 8)
    if np.allclose(stage_reference, restarted_reference):
        raise AssertionError("stage reference incorrectly restarts at each MPC solve")
    desired = controller.desired_doc_stages(6.0)
    torch.testing.assert_close(
        desired[:, 1, 1],
        torch.as_tensor(stage_reference, device=device, dtype=model.dtype),
    )
    torch.testing.assert_close(
        desired[:, 0, 0], torch.zeros(8, device=device, dtype=model.dtype)
    )

    controls = torch.full(
        (controller.horizon, *controller.control_shape),
        0.4,
        device=device,
        requires_grad=True,
    )
    states = controller.predict_stages(model.initialize_state(target.shape), controls)
    loss, _ = controller.cost(
        states, controls, current_process_time_s=1.0
    )
    loss.backward()
    if controls.grad is None or not bool(torch.isfinite(controls.grad).all()):
        raise AssertionError("trajectory-tracking loss gradient is missing/non-finite")
    if int(torch.count_nonzero(controls.grad)) == 0:
        raise AssertionError("trajectory-tracking loss gradient is identically zero")
    return {
        "stage_times": stage_times.tolist(),
        "stage_reference": stage_reference.tolist(),
        "gradient_norm": float(controls.grad.norm()),
    }


def _tracking_configuration_matrix_test(device: torch.device) -> dict[str, object]:
    """Cover all temporal modes, spatial definitions, and target losses."""

    reference = load_doc_reference("30mW_0mM")
    resolved_times, precedence_warnings = resolve_sampled_tracking_times(
        explicit_times_s=(1.0, 2.0), count=99, start_s=0.5, end_s=5.0
    )
    if resolved_times != (1.0, 2.0) or not precedence_warnings:
        raise AssertionError("explicit sampled times did not take precedence")
    try:
        resolve_sampled_tracking_times(count=3, start_s=None, end_s=2.0)
    except TrackingConfigurationError:
        pass
    else:
        raise AssertionError("sample count without explicit start/end was accepted")
    try:
        TrackingSpecification.sampled_curve(reference, (0.75,)).validate_runtime(
            0.5, 2.0, 4
        )
    except TrackingConfigurationError:
        pass
    else:
        raise AssertionError("non-grid-aligned sampled time was accepted")
    line = TrackingSpecification.checkpoints(((3.5, 0.4), (6.0, 0.9)))
    rect = TrackingSpecification.checkpoints(((3.5, 0.4), (9.0, 0.9)))
    if line.validate_runtime(0.5, 6.0, 8):
        raise AssertionError("H8 should cover the Sync_line checkpoint gap")
    if not rect.validate_runtime(0.5, 9.0, 8):
        raise AssertionError("H8 did not warn for the Sync_rect checkpoint gap")
    if rect.validate_runtime(0.5, 9.0, 12):
        raise AssertionError("H12 should cover the Sync_rect checkpoint gap")
    schedule = line.stage_schedule(3.0, 0.5, 8)
    active_times = schedule.times_s[schedule.active]
    np.testing.assert_allclose(active_times, [3.5, 6.0], rtol=0.0, atol=1e-12)
    if int(schedule.active.sum()) != 2:
        raise AssertionError("checkpoint mode generated intermediate target references")
    weighted = TrackingSpecification.checkpoints(
        ((3.5, 0.4), (6.0, 0.9)), point_weights=(2.0, 0.5)
    )
    weighted_schedule = weighted.stage_schedule(3.0, 0.5, 8)
    np.testing.assert_allclose(
        weighted_schedule.point_weights[weighted_schedule.active],
        [2.0, 0.5],
        rtol=0.0,
        atol=0.0,
    )
    try:
        TrackingSpecification(
            tracking_mode="checkpoints",
            reference_curve=reference,
            point_times_s=(0.5,),
            point_values=(0.4,),
        )
    except TrackingConfigurationError:
        pass
    else:
        raise AssertionError("checkpoint plus curve reference was accepted")

    params = replace(AIEParameters.from_reference(), o2_inhibition_mj_cm2=0.0)
    model = AIEModel(params, device=device)
    size = model.scattering_kernel_size // 2 + 3
    target = torch.ones((size, size), device=device)
    configurations = 0
    gradient_norms: list[float] = []
    for temporal_mode in TRACKING_MODES:
        for spatial_definition in SPATIAL_DEFINITIONS:
            for loss_name in TRACKING_LOSSES:
                common = {
                    "spatial_definition": spatial_definition,
                    "tracking_loss": loss_name,
                    "huber_delta": 0.1,
                }
                if temporal_mode == "curve":
                    specification = TrackingSpecification.curve(reference, **common)
                elif temporal_mode == "sampled-curve":
                    specification = TrackingSpecification.sampled_curve(
                        reference, (0.5, 1.0), **common
                    )
                else:
                    specification = TrackingSpecification.checkpoints(
                        ((0.5, 0.3), (1.0, 0.6)), **common
                    )
                controller = DifferentiableMPC(
                    model=model,
                    target=target,
                    tracking_specification=specification,
                    horizon=2,
                    physics_steps_per_control=10,
                    outside_weight=0.0,
                    energy_weight=0.0,
                    smoothness_weight=0.0,
                    num_iterations=1,
                )
                controls = torch.full(
                    (2, *controller.control_shape),
                    0.5,
                    device=device,
                    requires_grad=True,
                )
                states = controller.predict_stages(
                    model.initialize_state(target.shape), controls
                )
                loss, components = controller.cost(
                    states, controls, current_process_time_s=0.0
                )
                if not bool(torch.isfinite(loss)) or not bool(
                    torch.isfinite(components["target"])
                ):
                    raise AssertionError("tracking matrix produced a non-finite loss")
                loss.backward()
                if controls.grad is None or not bool(torch.isfinite(controls.grad).all()):
                    raise AssertionError("tracking matrix produced a non-finite gradient")
                gradient_norms.append(float(controls.grad.norm()))
                configurations += 1
    if min(gradient_norms) <= 0:
        raise AssertionError("a tracking configuration produced a zero gradient")

    checkpoint_controller = DifferentiableMPC(
        model=model,
        target=target,
        tracking_specification=TrackingSpecification.checkpoints(
            ((0.5, 0.4), (1.0, 0.8))
        ),
        horizon=2,
        physics_steps_per_control=10,
        outside_weight=0.0,
        energy_weight=0.0,
        smoothness_weight=0.0,
        num_iterations=6,
        learning_rate=0.3,
    )
    _, checkpoint_info = checkpoint_controller.optimize(
        model.initialize_state(target.shape),
        initial_guess=torch.full((2, size, size), 0.2, device=device),
        current_process_time_s=0.0,
    )
    if not checkpoint_info["final_loss"] < checkpoint_info["initial_loss"]:
        raise AssertionError("tiny checkpoint MPC optimization did not reduce loss")
    return {
        "configurations": configurations,
        "minimum_gradient_norm": min(gradient_norms),
        "checkpoint_initial_loss": checkpoint_info["initial_loss"],
        "checkpoint_final_loss": checkpoint_info["final_loss"],
        "h8_rect_warning": rect.validate_runtime(0.5, 9.0, 8)[0],
    }


def _reference_physics_condition_test() -> dict[str, object]:
    """Verify both condition selectors and rejection of an unselected mixture."""

    zero_reference = load_doc_reference("30mW_0mM")
    five_reference = load_doc_reference("30mW_5mM")
    zero_params = AIEParameters.from_reference(
        load_reference_config_for_condition("30mW_0mM")
    )
    five_params = AIEParameters.from_reference(
        load_reference_config_for_condition("30mW_5mM")
    )
    zero_match = assess_reference_physics_match(zero_reference, zero_params)
    five_match = assess_reference_physics_match(five_reference, five_params)
    if not zero_match["matched"] or not five_match["matched"]:
        raise AssertionError(
            f"validated condition selector was rejected: zero={zero_match}, "
            f"five={five_match}"
        )
    if not math.isclose(
        five_params.total_inhibition_mj_cm2, 119.7295, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("5 mM total inhibition was not resolved from source")
    if not math.isclose(five_params.b_slope, 0.0069, abs_tol=1e-12) or not math.isclose(
        five_params.b_intercept, 0.3815, abs_tol=1e-12
    ):
        raise AssertionError("5 mM B/intensity relation was not resolved from source")
    unsafe_mix = assess_reference_physics_match(
        five_reference, AIEParameters.from_reference()
    )
    if unsafe_mix["matched"]:
        raise AssertionError("unselected 0 mM physics was accepted for the 5 mM reference")
    return {
        "five_total_inhibition": five_params.total_inhibition_mj_cm2,
        "five_tempo_inhibition": five_params.tempo_inhibition_mj_cm2,
        "five_b_slope": five_params.b_slope,
        "five_b_intercept": five_params.b_intercept,
        "unsafe_mix_missing": unsafe_mix["missing_for_physical_match"],
    }


def _control_timing_test() -> tuple[int, float]:
    steps, total = resolve_control_timing(
        total_time_s=20.0, control_steps=None, control_dt_s=0.5
    )
    if steps != 40 or total != 20.0:
        raise AssertionError("20 s at 0.5 s/control did not resolve to 40 controls")
    try:
        resolve_control_timing(
            total_time_s=20.1, control_steps=None, control_dt_s=0.5
        )
    except ValueError as error:
        if "not compatible" not in str(error):
            raise AssertionError(f"unexpected timing validation error: {error}")
    else:
        raise AssertionError("incompatible total time was accepted")
    return steps, total


def _quantitative_metrics_test() -> dict[str, float]:
    """Validate perfect geometry and a known one-pixel boundary shift."""

    target = np.zeros((10, 12), dtype=float)
    target[2:8, 3:9] = 1.0
    perfect_doc = 0.9 * target
    perfect = calculate_final_metrics(
        perfect_doc,
        target,
        reference_final_doc=0.9,
        target_threshold=0.5,
        geometry_threshold=0.5,
        pixel_pitch_um=2.5,
    )
    geometry = perfect["geometry"]
    boundary = perfect["boundary"]
    for name in ("iou", "dice", "precision", "recall"):
        if geometry[name] != 1.0:
            raise AssertionError(f"perfect {name} != 1")
    for name in (
        "undercure_fraction",
        "overcure_fraction_target_normalized",
        "overcure_fraction_image",
        "area_error_fraction",
    ):
        if geometry[name] != 0.0:
            raise AssertionError(f"perfect {name} != 0")
    if boundary["mean_symmetric_distance_px"] != 0.0:
        raise AssertionError("perfect boundary distance is not zero")

    line_target = np.zeros((8, 8), dtype=float)
    line_target[2:6, 2] = 1.0
    shifted_doc = np.zeros_like(line_target)
    shifted_doc[2:6, 3] = 0.9
    shifted = calculate_final_metrics(
        shifted_doc,
        line_target,
        reference_final_doc=0.9,
        target_threshold=0.5,
        geometry_threshold=0.5,
        pixel_pitch_um=2.5,
    )["boundary"]
    if not math.isclose(
        shifted["mean_symmetric_distance_px"], 1.0, abs_tol=1e-12
    ) or not math.isclose(
        shifted["p95_symmetric_distance_um"], 2.5, abs_tol=1e-12
    ):
        raise AssertionError(f"known one-pixel boundary shift is incorrect: {shifted}")
    return {
        "perfect_iou": geometry["iou"],
        "perfect_boundary_px": boundary["mean_symmetric_distance_px"],
        "shifted_boundary_px": shifted["mean_symmetric_distance_px"],
        "shifted_boundary_p95_um": shifted["p95_symmetric_distance_um"],
    }


def _constant_block_assertion(
    coarse: torch.Tensor, native: torch.Tensor, factor: int
) -> None:
    height, width = coarse.shape
    blocks = native.reshape(height, factor, width, factor)
    expected = coarse[:, None, :, None].expand_as(blocks)
    torch.testing.assert_close(blocks, expected, rtol=0, atol=0)


def _resolution_modes_test(device: torch.device) -> dict[str, object]:
    """Validate dynamic native grids, coarsening, FOV, and native replay."""

    reference_params = native_aie_parameters()
    native_model = AIEModel(reference_params, device=device)
    native_shapes = ((300, 300), (320, 512), (256, 384))
    native_results: dict[tuple[int, int], dict[str, object]] = {}
    for native_shape in native_shapes:
        target_native = torch.zeros(native_shape, device=device)
        target_native[0, 0] = 1.0
        native_config = build_resolution_config("native", 2, native_shape)
        native_target = construct_optimization_target(target_native, native_config)
        native_control = torch.full(native_shape, 0.4, device=device)
        native_applied = recover_control_to_native(native_control, native_config)
        native_state = native_model.initialize_state(native_shape)
        native_state = native_model.advance(native_state, native_applied, physics_steps=1)
        if native_target.data_ptr() != target_native.data_ptr():
            raise AssertionError(
                "native mode must use the original target without resampling"
            )
        for field in native_state.tensors():
            if tuple(field.shape) != native_shape:
                raise AssertionError(
                    f"native state shape {tuple(field.shape)} != input {native_shape}"
                )
            if not bool(torch.isfinite(field).all()):
                raise AssertionError("native state contains NaN or Inf")
        if tuple(native_target.shape) != native_shape:
            raise AssertionError("native optimization target changed input shape")
        if tuple(native_applied.shape) != native_shape:
            raise AssertionError("native applied control changed input shape")
        native_results[native_shape] = {
            "optimization_shape": native_config.optimization_shape,
            "state_shape": tuple(native_state.shape),
            "mask_shape": tuple(native_applied.shape),
            "doc_shape": tuple(native_state.doc.shape),
            "fov_yx_um": tuple(value * 1e6 for value in native_config.native_fov_m),
        }

    kernel_sizes: dict[int, int] = {}
    coarse_reaction_ranges: dict[int, tuple[float, float]] = {}
    coarse_cases = (
        (2, (320, 512), (160, 256), 0.25),
        (3, (300, 300), (100, 100), 1.0 / 9.0),
    )
    for factor, native_shape, expected_shape, expected_fraction in coarse_cases:
        target_native = torch.zeros(native_shape, device=device)
        target_native[0, 0] = 1.0
        config = build_resolution_config("coarse", factor, native_shape)
        target_coarse = construct_optimization_target(target_native, config)
        if tuple(target_coarse.shape) != expected_shape:
            raise AssertionError(
                f"q={factor} target shape {tuple(target_coarse.shape)} != "
                f"{expected_shape}"
            )
        torch.testing.assert_close(
            target_coarse[0, 0],
            torch.tensor(expected_fraction, device=device),
            rtol=1e-6,
            atol=1e-7,
        )
        coarse = torch.arange(
            expected_shape[0] * expected_shape[1],
            device=device,
            dtype=torch.float32,
        ).reshape(expected_shape)
        coarse = coarse / max(float(coarse.max()), 1.0)
        recovered = recover_control_to_native(coarse, config)
        if tuple(recovered.shape) != native_shape:
            raise AssertionError(
                f"q={factor} recovery shape {tuple(recovered.shape)} != "
                f"native shape {native_shape}"
            )
        _constant_block_assertion(coarse, recovered, factor)
        coarse_model = AIEModel(
            replace(
                reference_params,
                pixel_pitch_m=config.optimization_pixel_pitch_m,
            ),
            device=device,
        )
        kernel_sizes[factor] = coarse_model.scattering_kernel_size
        coarse_state = coarse_model.initialize_state(expected_shape)
        full_coarse_control = torch.ones(expected_shape, device=device)
        full_coarse_prepared = coarse_model.prepare_control(
            full_coarse_control, coarse_state.shape
        )
        inhibitor_energy = (
            coarse_model.params.o2_inhibition_mj_cm2
            + coarse_model.params.tempo_inhibition_mj_cm2
        )
        steps_through_inhibition = (
            math.ceil(inhibitor_energy / float(full_coarse_prepared.energy.min()))
            + 1
        )
        coarse_state = coarse_model.advance(
            coarse_state,
            full_coarse_control,
            physics_steps=steps_through_inhibition,
        )
        coarse_reaction_ranges[factor] = (
            float(coarse_state.reaction_progress.min()),
            float(coarse_state.reaction_progress.max()),
        )
        if coarse_reaction_ranges[factor][1] <= 0:
            raise AssertionError(
                f"q={factor} reaction progress did not survive coarse advancement"
            )

    kernel_sizes[1] = native_model.scattering_kernel_size
    for factor, kernel_size in kernel_sizes.items():
        pitch = reference_params.native_pixel_pitch_m * factor
        raw_size = int(reference_params.scattering_blur_size_m / pitch)
        expected_kernel_size = raw_size if raw_size % 2 else raw_size + 1
        if kernel_size != expected_kernel_size:
            raise AssertionError(
                f"q={factor} scattering kernel {kernel_size} does not match "
                f"reference formula result {expected_kernel_size}"
            )

    rectangular_shape = (320, 512)
    replay_config = build_resolution_config("coarse", 2, rectangular_shape)
    full_native_control = recover_control_to_native(
        torch.ones(replay_config.optimization_shape, device=device),
        replay_config,
    )
    native_full_step_energy = (
        reference_params.intensity_mw_cm2 * reference_params.dt
    )
    replay_step_count = (
        math.ceil(
            (
                reference_params.o2_inhibition_mj_cm2
                + reference_params.tempo_inhibition_mj_cm2
            )
            / native_full_step_energy
        )
        + 1
    )
    replay_masks = [full_native_control] * replay_step_count
    replay_state = replay_native_controls(
        reference_params,
        rectangular_shape,
        replay_masks,
        physics_steps_per_control=1,
        device=device,
    )
    for field in replay_state.tensors():
        if tuple(field.shape) != rectangular_shape:
            raise AssertionError(
                f"native replay state shape {tuple(field.shape)} != "
                f"{rectangular_shape}"
            )
        if not bool(torch.isfinite(field).all()):
            raise AssertionError("native replay state contains NaN or Inf")
    if float(replay_state.doc.min()) < 0 or float(replay_state.doc.max()) > 1:
        raise AssertionError("native replay DoC left the valid [0, 1] range")
    if float(replay_state.reaction_progress.max()) <= 0:
        raise AssertionError("native replay reset or lost reaction progress")

    try:
        build_resolution_config("coarse", 3, rectangular_shape)
    except ValueError as error:
        if "must divide both dimensions" not in str(error):
            raise AssertionError(f"unexpected non-divisible-grid error: {error}") from error
        nondivisible_error = str(error)
    else:
        raise AssertionError("320 x 512 with q=3 was not rejected")

    for invalid_target, expected_message in (
        (torch.zeros(12, device=device), "2D grayscale"),
        (torch.empty((0, 12), device=device), "dimensions must be positive"),
        (torch.full((8, 9), torch.nan, device=device), "NaN or Inf"),
        (torch.full((8, 9), 1.1, device=device), "normalized to [0, 1]"),
    ):
        try:
            require_native_target(invalid_target)
        except ValueError as error:
            if expected_message not in str(error):
                raise AssertionError(f"unexpected target validation error: {error}") from error
        else:
            raise AssertionError("invalid native target was not rejected")

    return {
        "native_results": native_results,
        "q2_shape": (160, 256),
        "q3_shape": (100, 100),
        "rectangular_fov_yx_um": native_results[rectangular_shape]["fov_yx_um"],
        "kernel_sizes": kernel_sizes,
        "coarse_reaction_ranges": coarse_reaction_ranges,
        "native_replay_steps": replay_step_count,
        "native_replay_shape": tuple(replay_state.shape),
        "nondivisible_error": nondivisible_error,
        "replay_reaction_range": (
            float(replay_state.reaction_progress.min()),
            float(replay_state.reaction_progress.max()),
        ),
        "replay_doc_range": (
            float(replay_state.doc.min()),
            float(replay_state.doc.max()),
        ),
    }


def _gif_visualization_test(device: torch.device) -> dict[str, object]:
    """Validate DoC PNG/GIF counts, dimensions, looping, and value checks."""

    with tempfile.TemporaryDirectory(prefix="aie_doc_gif_test_") as temporary_dir:
        output_dir = Path(temporary_dir)
        results: dict[str, object] = {}
        gif_shapes = {
            "native_300x300": (300, 300),
            "native_320x512": (320, 512),
            "native_256x384": (256, 384),
            "coarse_160x256": (160, 256),
        }
        for label, shape in gif_shapes.items():
            target_path = output_dir / f"target_{label}.png"
            Image.fromarray(np.zeros(shape, dtype=np.uint8), mode="L").save(target_path)
            loaded_target = load_normalized_target(target_path)
            require_native_target(loaded_target, target_path)
            if tuple(loaded_target.shape) != shape:
                raise AssertionError(
                    f"target loader changed {shape} to {tuple(loaded_target.shape)}"
                )
            # Identical frames confirm that inhibited zero-DoC time steps are
            # retained instead of being merged by the GIF encoder.
            levels = (0.0, 0.25, 0.5)
            frame_paths: list[Path] = []
            for frame_index, level in enumerate(levels):
                doc = torch.full(shape, level, device=device)
                frame_path = output_dir / f"doc_frame_{label}_{frame_index:03d}.png"
                save_doc_frame(doc, frame_path, shape)
                frame_paths.append(frame_path)
            with Image.open(frame_paths[-1]) as saved_frame:
                saved_shape = (saved_frame.height, saved_frame.width)
            if saved_shape != shape:
                raise AssertionError(
                    f"DoC PNG changed {shape} to {saved_shape}"
                )
            gif_path = output_dir / f"doc_evolution_{label}.gif"
            save_gif_from_frames(frame_paths, gif_path, duration_ms=300)
            validate_gif(gif_path, len(levels), shape)
            results[f"{label}_frames"] = len(levels)
            results[f"{label}_shape"] = shape

        invalid_shape = (17, 23)
        invalid_doc = torch.zeros(invalid_shape, device=device)
        invalid_doc[0, 0] = torch.nan
        try:
            validate_doc_field(invalid_doc, invalid_shape)
        except ValueError as error:
            if "NaN or Inf" not in str(error):
                raise AssertionError(f"unexpected DoC validation error: {error}") from error
        else:
            raise AssertionError("non-finite DoC frame was not rejected")
        results["shapes"] = gif_shapes
        results["frames_per_gif"] = 3
        return results


def run_smoke_tests() -> None:
    """Run reference, physics, optimization, resolution, and GIF validation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"running smoke tests on {device}")
    reference_results = _reference_architecture_test(device)
    print(
        "[pass] read-only AST reference adapter; "
        f"condition={reference_results['condition_id']} "
        f"native_pitch={reference_results['native_pitch_um']:.6f}um "
        f"q2_pitch={reference_results['coarse_pitch_um']:.6f}um; "
        "no stale AIEParameters defaults; "
        f"reference/calibration_safety_checks={reference_results['safety_checks']}"
    )
    print(
        f"[pass] fingerprints reference={reference_results['reference_sha256']} "
        f"calibration={reference_results['calibration_sha256']}"
    )
    doc_reference_results = _doc_reference_curve_test()
    print(
        "[pass] DoC schema-v3 model catalog and non-destructive schema-v2 migration; "
        f"ids={doc_reference_results['ids']} "
        f"models={doc_reference_results['models']} "
        f"default={doc_reference_results['default_model']} "
        f"legacy_default={doc_reference_results['legacy_default_model']} "
        f"delta_at_5s={doc_reference_results['condition_delta_at_5s']:.6f} "
        f"legacy_sha256={doc_reference_results['legacy_sha256']}"
    )
    condition_physics = _reference_physics_condition_test()
    print(
        "[pass] 30mW_0mM and 30mW_5mM forward-physics selectors; "
        f"5mM total_inhibition={condition_physics['five_total_inhibition']:.6g}, "
        f"B={condition_physics['five_b_slope']:.6g}*I+"
        f"{condition_physics['five_b_intercept']:.6g}; unsafe mix guarded"
    )
    timing_steps, timing_total = _control_timing_test()
    print(
        f"[pass] total-time conversion {timing_total:.1f}s -> "
        f"{timing_steps} controls at 0.5s/control"
    )
    physics = _physics_synchronization_test(device)
    print(
        f"[pass] latest reference dt={physics['dt']:.6g}s "
        f"control_dt={physics['control_dt']:.6g}s "
        f"default_horizon={physics['prediction_horizon']:.6g}s; "
        f"O2_diffusion={physics['o2_diffusion_enabled']} "
        f"TEMPO_diffusion={physics['tempo_diffusion_enabled']} "
        f"TEMPO_sigma_scale={physics['tempo_sigma_scale']:.6g}"
    )
    print(
        f"[pass] kernels O2={physics['o2_kernel_size']} "
        f"TEMPO={physics['tempo_kernel_size']} "
        f"scattering={physics['scattering_kernel_size']}; "
        f"B={physics['b_slope']:.12g}*I+{physics['b_intercept']:.12g}"
    )
    equivalence_error = _static_mask_equivalence_test(device)
    print(f"[pass] static-mask equivalence max_abs_error={equivalence_error:.3e}")
    gradient_norm, nonzero = _gradient_test(device)
    print(f"[pass] gradient norm={gradient_norm:.3e}, nonzero_elements={nonzero}")
    shape_message = _shape_validation_test(device)
    print(f"[pass] shape validation: {shape_message}")
    initial_loss, final_loss = _mpc_optimization_test(device)
    print(f"[pass] MPC loss {initial_loss:.6e}->{final_loss:.6e}")
    tracking_construction = _trajectory_tracking_construction_test(device)
    print(
        "[pass] absolute-time stage references "
        f"{tracking_construction['stage_times'][0]:.1f}-"
        f"{tracking_construction['stage_times'][-1]:.1f}s; "
        f"tracking gradient norm={tracking_construction['gradient_norm']:.3e}"
    )
    tracking_matrix = _tracking_configuration_matrix_test(device)
    print(
        "[pass] tracking matrix "
        f"{tracking_matrix['configurations']} mode/spatial/loss combinations; "
        f"minimum gradient={tracking_matrix['minimum_gradient_norm']:.3e}; "
        "tiny checkpoint loss "
        f"{tracking_matrix['checkpoint_initial_loss']:.3e}->"
        f"{tracking_matrix['checkpoint_final_loss']:.3e}; "
        f"H8 warning={tracking_matrix['h8_rect_warning']}"
    )
    quantitative = _quantitative_metrics_test()
    print(
        "[pass] perfect geometry IoU=1 Dice=1 under/overcure=0 "
        f"boundary={quantitative['perfect_boundary_px']:.1f}px; "
        f"known shift={quantitative['shifted_boundary_px']:.1f}px/"
        f"p95={quantitative['shifted_boundary_p95_um']:.1f}um"
    )
    resolution = _resolution_modes_test(device)
    print(
        "[pass] dynamic native grids "
        f"{tuple(resolution['native_results'])}; "
        f"320x512 q=2->{resolution['q2_shape']} "
        f"300x300 q=3->{resolution['q3_shape']}"
    )
    print(
        "[pass] 320x512 physical FOV (y, x)="
        f"{resolution['rectangular_fov_yx_um']} um; "
        f"scattering kernels={resolution['kernel_sizes']}"
    )
    print(
        "[pass] non-divisible coarse grid rejected: "
        f"{resolution['nondivisible_error']}"
    )
    print(
        "[pass] coarse reaction-progress state finite/nonzero "
        f"ranges={resolution['coarse_reaction_ranges']}"
    )
    print(
        f"[pass] native replay shape={resolution['native_replay_shape']} finite=True "
        f"steps={resolution['native_replay_steps']} "
        f"reaction_progress_range={resolution['replay_reaction_range']} "
        f"DoC_range={resolution['replay_doc_range']}"
    )
    gif_results = _gif_visualization_test(device)
    print(
        f"[pass] DoC GIFs {gif_results['frames_per_gif']} frames each at "
        f"{gif_results['shapes']}, duration=300ms loop=continuous"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("circle.png"),
        help=(
            "2D grayscale target whose exact dimensions define the native grid; "
            "unqualified names resolve inside GEO/"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_DIR / "mpc_output"
    )
    parser.add_argument(
        "--baseline-target-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "also run the unoptimized repeated raw target-mask forward rollout "
            "(default: enabled; disable with --no-baseline-target-mask)"
        ),
    )
    parser.add_argument(
        "--resolution-mode", choices=("native", "coarse"), default="native"
    )
    parser.add_argument("--coarsen-factor", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--physics-steps-per-control", type=int, default=10)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "--total-time",
        type=float,
        default=None,
        help=(
            "total physical process time in seconds (default behavior: 20 s); "
            "must be an integer multiple of the derived control interval"
        ),
    )
    timing.add_argument(
        "--control-steps",
        type=int,
        default=None,
        help="backward-compatible explicit control-step count",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument(
        "--geometry-threshold",
        type=float,
        default=0.5,
        help=(
            "computational cured/uncured segmentation threshold; not an "
            "experimentally calibrated gel point (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--tracking-mode",
        choices=TRACKING_MODES,
        default="curve",
        help=(
            "target-side time specification: dense fitted curve (default), "
            "sparse samples from a selected curve, or direct checkpoints"
        ),
    )
    parser.add_argument(
        "--physics-condition",
        choices=SUPPORTED_FORWARD_CONDITIONS,
        default="active",
        help=(
            "authoritative forward-physics selector, independent of the DoC "
            "tracking reference (default: exact active collaborator source state)"
        ),
    )
    parser.add_argument(
        "--doc-reference",
        choices=("30mW_0mM", "30mW_5mM"),
        default=None,
        help=(
            "experimental curve condition ID; curve modes default to 30mW_0mM, "
            "checkpoint mode must not use this option"
        ),
    )
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="schema-v2 or schema-v3 DoC reference artifact",
    )
    parser.add_argument(
        "--curve-model",
        choices=CURVE_MODEL_IDS,
        default="collaborator_original",
        help="curve model used by curve/sample-curve modes (default: collaborator_original)",
    )
    parser.add_argument(
        "--tracking-times",
        default=None,
        help="sampled-curve absolute times in seconds, comma separated",
    )
    parser.add_argument("--num-tracking-points", type=int, default=None)
    parser.add_argument("--tracking-start", type=float, default=None)
    parser.add_argument("--tracking-end", type=float, default=None)
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="direct collaborator requirements as time:DoC,time:DoC,...",
    )
    parser.add_argument(
        "--tracking-spatial-definition",
        choices=SPATIAL_DEFINITIONS,
        default="pixelwise",
        help="pixelwise spatial tracking (current default) or target-region mean",
    )
    parser.add_argument(
        "--tracking-loss",
        choices=TRACKING_LOSSES,
        default="mse",
        help="target tracking loss only; other controller penalties are unchanged",
    )
    parser.add_argument(
        "--tracking-variable",
        choices=TRACKING_VARIABLES,
        default="doc",
        help=(
            "target-side tracking coordinate: historical DoC (default) or "
            "reaction progress; non-target penalties are unchanged"
        ),
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.1,
        help="Huber transition delta when --tracking-loss huber",
    )
    parser.add_argument(
        "--point-weights",
        default=None,
        help="optional positive comma-separated sparse-point weights (default: equal)",
    )
    parser.add_argument(
        "--allow-reference-physics-mismatch",
        action="store_true",
        help=(
            "research/debug override allowing a tracking reference without a "
            "validated matching authoritative forward-physics condition"
        ),
    )
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--outside-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=1e-2)
    parser.add_argument(
        "--initialization-mode",
        choices=("uniform", "physics-aware"),
        default="uniform",
        help=(
            "first-solve cold start: historical target-shaped levels (uniform, "
            "default) or checkpoint/optics-derived precompensation (physics-aware)"
        ),
    )
    parser.add_argument(
        "--initial-target-level",
        type=float,
        default=0.95,
        help=(
            "target level for uniform initialization only; unused in "
            "physics-aware mode"
        ),
    )
    parser.add_argument(
        "--initial-background-level",
        type=float,
        default=0.02,
        help=(
            "background level for uniform initialization and starting seed for "
            "physics-aware optical inversion"
        ),
    )
    parser.add_argument(
        "--physics-init-iterations",
        type=int,
        default=DEFAULT_PHYSICS_INIT_ITERATIONS,
        help="lightweight optical-only precompensation iterations",
    )
    parser.add_argument(
        "--physics-init-lr",
        type=float,
        default=DEFAULT_PHYSICS_INIT_LEARNING_RATE,
        help="Adam learning rate for optical-only precompensation",
    )
    parser.add_argument(
        "--physics-init-outside-weight",
        type=float,
        default=DEFAULT_PHYSICS_INIT_OUTSIDE_WEIGHT,
        help="outside-field penalty weight for optical-only precompensation",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run lightweight validation and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        run_smoke_tests()
    else:
        run_demo(arguments)
