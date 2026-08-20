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
import aie_reference
from aie_reference import load_reference_config, reference_step_torch
from doc_reference import DoCReferenceCurve, load_doc_reference
from mpc_metrics import (
    GEOMETRY_THRESHOLD_NOTE,
    calculate_final_metrics,
    temporal_tracking_metrics,
)


REPOSITORY_DIR = Path(__file__).resolve().parent
GEO_DIR = REPOSITORY_DIR / "GEO"


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
) -> AIEState:
    """Replay recovered masks through a fresh native model without optimization."""

    current_reference_params = AIEParameters.from_reference()
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
    native_model = AIEModel(native_params, device=device)
    native_state = native_model.initialize_state(native_shape)
    if doc_frame_callback is not None:
        doc_frame_callback(0, native_state.doc)
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
    return native_state


def doc_region_metrics(
    doc: torch.Tensor, target: torch.Tensor, target_threshold: float
) -> tuple[float, float]:
    """Compute target and outside mean DoC with the MPC threshold convention."""

    if tuple(doc.shape) != tuple(target.shape):
        raise ValueError(
            f"DoC shape {tuple(doc.shape)} does not match target shape "
            f"{tuple(target.shape)}"
        )
    target_region = target > target_threshold
    if not bool(target_region.any()):
        raise ValueError("target has no pixels above target_threshold")
    outside_region = ~target_region
    target_mean = float(doc[target_region].mean())
    outside_mean = (
        float(doc[outside_region].mean()) if bool(outside_region.any()) else 0.0
    )
    return target_mean, outside_mean


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


def save_tracking_plot(
    reference_curve: DoCReferenceCurve,
    total_time_s: float,
    control_times_s: np.ndarray,
    actual_target_doc: np.ndarray,
    path: Path,
) -> None:
    """Save experimental-reference versus closed-loop mean target DoC."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_time = np.linspace(0.0, total_time_s, max(401, int(total_time_s / 0.01) + 1))
    reference_doc = np.asarray(reference_curve.at(reference_time))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(reference_time, reference_doc, label="experimental reference", linewidth=2)
    axis.plot(
        np.r_[0.0, control_times_s],
        np.r_[0.0, actual_target_doc],
        "o-",
        markersize=3,
        label="closed-loop mean target DoC",
    )
    axis.set_xlim(0.0, total_time_s)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Absolute process time (s)")
    axis.set_ylabel("Degree of conversion")
    axis.set_title("DoC trajectory tracking: 0 mM TEMPO, 30 mW/cm²")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "undefined" if value is None else f"{value:.6f}{suffix}"


def print_metrics_summary(
    *,
    total_time_s: float,
    tracking: dict[str, float],
    reference_final_doc: float,
    target_mean_doc: float,
    outside_mean_doc: float,
    final_metrics: dict[str, object],
) -> None:
    """Print the compact terminal summary for the primary native result."""

    soft = final_metrics["soft_doc"]
    geometry = final_metrics["geometry"]
    boundary = final_metrics["boundary"]
    components = final_metrics["components"]
    holes = final_metrics["holes"]
    print("\n=== Closed-loop tracking ===")
    print(f"total time:                {total_time_s:.2f} s")
    print(f"tracking RMSE:             {tracking['rmse']:.6f}")
    print(f"tracking MAE:              {tracking['mae']:.6f}")
    print(f"maximum absolute error:    {tracking['max_absolute_error']:.6f}")
    print("\n=== Final DoC ===")
    print(f"reference final DoC:       {reference_final_doc:.6f}")
    print(f"mean target DoC:           {target_mean_doc:.6f}")
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


def _print_parameter_provenance(params: AIEParameters) -> None:
    """Print the effective physical/model source at simulation startup."""

    print(f"Reference model: {params.reference_model_source}")
    print(f"Reference SHA256: {params.reference_model_sha256}")
    print(f"Reference structure SHA256: {params.reference_structure_sha256}")
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
        print("DoC fit condition: none selected; no numeric TEMPO condition is active")
    if not params.doc_fit_applied_to_governing_law:
        print("DoC fit usage: provenance only; governing B/intensity law preserved")


def run_demo(args: argparse.Namespace) -> None:
    """Run native MPC or coarse MPC followed by required native replay."""

    reference = load_reference_config()
    native_params = native_aie_parameters(reference)
    control_dt_s = native_params.dt * args.physics_steps_per_control
    control_steps, total_time_s = resolve_control_timing(
        total_time_s=args.total_time,
        control_steps=args.control_steps,
        control_dt_s=control_dt_s,
    )
    args.control_steps = control_steps
    args.total_time = total_time_s
    doc_reference = load_doc_reference(args.doc_reference)
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
    _print_parameter_provenance(native_params)
    print(
        "DoC trajectory reference: "
        f"{doc_reference.metadata['reference_id']} "
        f"({doc_reference.metadata.get('condition_label', '30 mW/cm^2, 0 mM TEMPO')})"
    )
    print(f"DoC reference SHA256: {doc_reference.source_sha256}")
    print(
        "DoC reference use: control target only; AIE/reaction_progress forward "
        "physics unchanged"
    )
    _print_resolution_header(config, device, args, optimization_params)

    target_native = target_native.to(device=device)
    optimization_target = construct_optimization_target(target_native, config)
    model = AIEModel(optimization_params, device=device)
    state = model.initialize_state(config.optimization_shape)
    controller = DifferentiableMPC(
        model=model,
        target=optimization_target,
        reference_curve=doc_reference,
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
    guess = controller.initial_control_sequence(
        target_level=args.initial_target_level,
        background_level=args.initial_background_level,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_grayscale(target_native, args.output_dir / "target_native.png")
    if config.resolution_mode == "coarse":
        save_grayscale(optimization_target, args.output_dir / "target_coarse.png")

    applied_controls_optimization: list[torch.Tensor] = []
    applied_controls_native: list[torch.Tensor] = []
    optimization_histories: list[list[float]] = []
    optimization_control_times_s: list[float] = []
    optimization_reference_doc: list[float] = []
    optimization_target_doc: list[float] = []
    optimization_outside_doc: list[float] = []
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

        target_doc_mean, outside_doc_mean = doc_region_metrics(
            state.doc, optimization_target, args.target_threshold
        )
        applied_time_s = (control_step + 1) * control_dt_s
        reference_doc = float(doc_reference.at(applied_time_s))
        optimization_control_times_s.append(applied_time_s)
        optimization_reference_doc.append(reference_doc)
        optimization_target_doc.append(target_doc_mean)
        optimization_outside_doc.append(outside_doc_mean)
        running_error = np.asarray(optimization_target_doc) - np.asarray(
            optimization_reference_doc
        )
        running_rmse = float(np.sqrt(np.mean(np.square(running_error))))
        elapsed = time.perf_counter() - started
        print(
            f"step {control_step + 1:02d}/{args.control_steps:02d} "
            f"t={applied_time_s:.2f}s ref={reference_doc:.4f} "
            f"loss {info['initial_loss']:.5f}->{info['final_loss']:.5f} "
            f"DoC(target/out)={target_doc_mean:.4f}/{outside_doc_mean:.4f} "
            f"track_rmse={running_rmse:.5f} "
            f"mask_mean={float(applied_control.mean()):.3f} solve={elapsed:.2f}s"
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
        native_replay_outside_doc: list[float] = []

        def save_native_replay_frame(frame_index: int, doc: torch.Tensor) -> None:
            frame_path = args.output_dir / f"doc_frame_native_{frame_index:03d}.png"
            save_doc_frame(doc, frame_path, config.native_shape)
            native_doc_frame_paths.append(frame_path)
            if frame_index > 0:
                target_mean, outside_mean = doc_region_metrics(
                    doc, target_native, args.target_threshold
                )
                native_replay_target_doc.append(target_mean)
                native_replay_outside_doc.append(outside_mean)

        native_state = replay_native_controls(
            native_params,
            config.native_shape,
            applied_controls_native,
            args.physics_steps_per_control,
            device,
            doc_frame_callback=save_native_replay_frame,
        )
    else:
        native_state = optimization_state
        native_replay_target_doc = optimization_target_doc
        native_replay_outside_doc = optimization_outside_doc

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

    native_target_doc, native_outside_doc = doc_region_metrics(
        native_state.doc, target_native, args.target_threshold
    )
    primary_control_times_s = np.asarray(optimization_control_times_s, dtype=float)
    primary_reference_doc = np.asarray(
        doc_reference.at(primary_control_times_s), dtype=float
    )
    primary_target_doc = np.asarray(native_replay_target_doc, dtype=float)
    primary_outside_doc = np.asarray(native_replay_outside_doc, dtype=float)
    if not (
        primary_control_times_s.size
        == primary_reference_doc.size
        == primary_target_doc.size
        == primary_outside_doc.size
        == args.control_steps
    ):
        raise AssertionError("primary native tracking timeline has an unexpected length")
    primary_tracking_error = primary_target_doc - primary_reference_doc
    tracking = temporal_tracking_metrics(primary_target_doc, primary_reference_doc)
    reference_final_doc = float(doc_reference.at(total_time_s))
    final_metrics = calculate_final_metrics(
        native_state.doc.detach().cpu().numpy(),
        target_native.detach().cpu().numpy(),
        reference_final_doc,
        args.target_threshold,
        args.geometry_threshold,
        config.native_pixel_pitch_m * 1e6,
    )
    component_metrics_path = args.output_dir / "component_metrics.csv"
    save_component_metrics_csv(final_metrics["component_rows"], component_metrics_path)
    tracking_plot_path = args.output_dir / "doc_tracking_curve.png"
    save_tracking_plot(
        doc_reference,
        total_time_s,
        primary_control_times_s,
        primary_target_doc,
        tracking_plot_path,
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
        "primary_result_grid": (
            "native_replay" if config.resolution_mode == "coarse" else "native"
        ),
        "reference_provenance": doc_reference.provenance_metadata(),
        "forward_model_provenance": {
            "source": native_params.reference_model_source,
            "sha256": native_params.reference_model_sha256,
            "history_mode": DOC_HISTORY_MODE,
            "history_description": DOC_HISTORY_DESCRIPTION,
        },
        "target": {
            "name": target_path.name,
            "path": str(target_path),
            "shape": list(config.native_shape),
            "target_threshold": args.target_threshold,
        },
        "total_process_time_s": total_time_s,
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
        "temporal_tracking": tracking,
        "soft_doc": final_metrics["soft_doc"],
        "geometry": final_metrics["geometry"],
        "boundary": final_metrics["boundary"],
        "components": components_summary,
        "holes": holes_summary,
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
        "actual_mean_target_doc": primary_target_doc,
        "actual_mean_outside_doc": primary_outside_doc,
        "temporal_tracking_errors": primary_tracking_error,
        "temporal_tracking_rmse": np.float64(tracking["rmse"]),
        "temporal_tracking_mae": np.float64(tracking["mae"]),
        "temporal_tracking_max_absolute_error": np.float64(
            tracking["max_absolute_error"]
        ),
        "doc_reference_metadata_json": np.asarray(
            json.dumps(doc_reference.provenance_metadata(), sort_keys=True)
        ),
        "doc_reference_source_sha256": np.asarray(doc_reference.source_sha256),
        "metrics_json": np.asarray(json.dumps(metrics_document, sort_keys=True)),
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
        total_time_s=total_time_s,
        tracking=tracking,
        reference_final_doc=reference_final_doc,
        target_mean_doc=native_target_doc,
        outside_mean_doc=native_outside_doc,
        final_metrics=final_metrics,
    )

    print(f"resolution_mode={config.resolution_mode}")
    print(f"native_grid={config.native_shape}")
    print(f"optimization_grid={config.optimization_shape}")
    print(f"coarsen_factor={config.coarsen_factor}")
    if config.resolution_mode == "coarse":
        print(
            "native replay "
            f"DoC(target/out)={native_target_doc:.4f}/{native_outside_doc:.4f}"
        )
    else:
        print(f"native DoC(target/out)={native_target_doc:.4f}/{native_outside_doc:.4f}")
    if coarse_gif_path is not None:
        print(f"saved coarse DoC GIF: {coarse_gif_path.resolve()}")
        print(f"saved native replay DoC GIF: {native_gif_path.resolve()}")
    else:
        print(f"saved native DoC GIF: {native_gif_path.resolve()}")
    print(f"saved tracking plot: {tracking_plot_path.resolve()}")
    print(f"saved component metrics: {component_metrics_path.resolve()}")
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
    if reference.doc_fit is not None:
        raise AssertionError(
            "a/b/c was selected despite no numeric active TEMPO condition"
        )
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
    default_horizon = 4
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


def _doc_reference_curve_test() -> dict[str, object]:
    """Validate loader schema, monotonicity, interpolation, and endpoint hold."""

    reference = load_doc_reference()
    if reference.metadata["schema_version"] != 1:
        raise AssertionError("unexpected DoC reference schema")
    if reference.time_s[0] != 0.0 or reference.time_s[-1] != 20.0:
        raise AssertionError("DoC reference does not span exactly 0 to 20 s")
    if not np.isfinite(reference.doc_reference).all():
        raise AssertionError("DoC reference contains NaN or Inf")
    if np.any(np.diff(reference.doc_reference) < -1e-10):
        raise AssertionError("DoC reference is not monotonic nondecreasing")
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
    with tempfile.TemporaryDirectory() as directory:
        invalid_path = Path(directory) / "invalid_reference.json"
        invalid = json.loads(json.dumps(reference.metadata))
        invalid["condition"]["intensity_mw_cm2"] = 20.0
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        try:
            load_doc_reference(invalid_path)
        except ValueError as error:
            if "30" not in str(error):
                raise AssertionError(f"unexpected condition validation error: {error}")
        else:
            raise AssertionError("incorrect DoC reference condition was accepted")
    return {
        "samples": int(reference.time_s.size),
        "final_doc": reference.final_doc,
        "interpolated_2p025": float(reference.at(2.025)),
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
        "[pass] DoC reference loader/schema/condition/monotonic/interpolation; "
        f"samples={doc_reference_results['samples']} "
        f"final={doc_reference_results['final_doc']:.6f}"
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
        "--resolution-mode", choices=("native", "coarse"), default="native"
    )
    parser.add_argument("--coarsen-factor", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=4)
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
        "--doc-reference",
        type=Path,
        default=REPOSITORY_DIR / "doc_reference_curve.json",
        help="validated experimental time-domain DoC reference JSON",
    )
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--outside-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=1e-2)
    parser.add_argument("--initial-target-level", type=float, default=0.95)
    parser.add_argument("--initial-background-level", type=float, default=0.02)
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
