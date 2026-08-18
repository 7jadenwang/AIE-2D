"""Run native or physically consistent coarse-grid differentiable AIE MPC.

All experiment targets and final physical masks use the native 300 x 300 DLP
grid. Coarse mode optimizes a block-averaged target on a lower-resolution
grid with the same physical field of view, expands applied masks by constant
blocks, and replays those masks through a fresh native-resolution AIE model.

Run ``python run_mpc.py --smoke-test`` for lightweight physics, gradient, MPC,
resolution-conversion, field-of-view, and native-replay validation.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import GifImagePlugin, Image

from aie_fine_grid import expand_projector_mask, initialize_projector_mask
from aie_model import AIEModel, AIEParameters, AIEState
from aie_mpc import DifferentiableMPC


REPOSITORY_DIR = Path(__file__).resolve().parent
GEO_DIR = REPOSITORY_DIR / "GEO"
NATIVE_GRID = (300, 300)
# Established by AIE_TEMPOv1.2.py, AIE_re2.1.py, and the fine-grid design docs.
NATIVE_PROJECTOR_PIXEL_PITCH_M = 7.395e-6


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
        return tuple(
            dimension * self.native_pixel_pitch_m for dimension in self.native_shape
        )

    @property
    def optimization_fov_m(self) -> tuple[float, float]:
        return tuple(
            dimension * self.optimization_pixel_pitch_m
            for dimension in self.optimization_shape
        )


def build_resolution_config(
    resolution_mode: str, coarsen_factor: int
) -> ResolutionConfig:
    """Validate a resolution mode and preserve the native physical field of view."""

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
        if any(dimension % coarsen_factor for dimension in NATIVE_GRID):
            raise ValueError(
                f"coarsen_factor {coarsen_factor} must divide the native grid "
                f"{NATIVE_GRID} exactly; no rounding is allowed"
            )
        factor = coarsen_factor

    optimization_shape = tuple(dimension // factor for dimension in NATIVE_GRID)
    config = ResolutionConfig(
        resolution_mode=resolution_mode,
        native_shape=NATIVE_GRID,
        optimization_shape=optimization_shape,
        coarsen_factor=factor,
        native_pixel_pitch_m=NATIVE_PROJECTOR_PIXEL_PITCH_M,
        optimization_pixel_pitch_m=factor * NATIVE_PROJECTOR_PIXEL_PITCH_M,
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
    """Reject any target that is not already the physical 300 x 300 grid."""

    if tuple(target.shape) != NATIVE_GRID:
        location = f" {path}" if path is not None else ""
        raise ValueError(
            f"target{location} must be exactly {NATIVE_GRID}, got "
            f"{tuple(target.shape)}; targets are never resized, cropped, or padded"
        )


def construct_optimization_target(
    target_native: torch.Tensor, config: ResolutionConfig
) -> torch.Tensor:
    """Return the unchanged native target or an exact block-average target."""

    require_native_target(target_native)
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
    if tuple(native_control.shape) != NATIVE_GRID:
        raise AssertionError(
            f"recovered physical mask must have shape {NATIVE_GRID}, got "
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


def native_aie_parameters() -> AIEParameters:
    """Create the established 300 x 300 projector calibration cleanly."""

    return replace(
        AIEParameters(),
        pixel_pitch_m=NATIVE_PROJECTOR_PIXEL_PITCH_M,
        projector_refinement=1,
    )


def replay_native_controls(
    native_params: AIEParameters,
    applied_controls_native: list[torch.Tensor],
    physics_steps_per_control: int,
    device: torch.device,
    doc_frame_callback: Callable[[int, torch.Tensor], None] | None = None,
) -> AIEState:
    """Replay recovered masks through a fresh native model without optimization."""

    native_model = AIEModel(native_params, device=device)
    native_state = native_model.initialize_state(NATIVE_GRID)
    if doc_frame_callback is not None:
        doc_frame_callback(0, native_state.doc)
    with torch.no_grad():
        for replay_step, control in enumerate(applied_controls_native, start=1):
            if tuple(control.shape) != NATIVE_GRID:
                raise ValueError(
                    f"native replay control must have shape {NATIVE_GRID}, got "
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


def _print_resolution_header(
    config: ResolutionConfig, device: torch.device, args: argparse.Namespace
) -> None:
    native_fov_um = tuple(value * 1e6 for value in config.native_fov_m)
    optimization_fov_um = tuple(value * 1e6 for value in config.optimization_fov_m)
    print(f"resolution_mode={config.resolution_mode}")
    print(f"native_grid={config.native_shape}")
    print(f"optimization_grid={config.optimization_shape}")
    if config.resolution_mode == "coarse":
        print(f"coarsen_factor={config.coarsen_factor}")
    print(f"pixel_pitch={config.optimization_pixel_pitch_m * 1e6:.6f} um")
    if config.resolution_mode == "coarse":
        print(f"native_pixel_pitch={config.native_pixel_pitch_m * 1e6:.6f} um")
    print(f"native FOV: {native_fov_um[0]:.6f} x {native_fov_um[1]:.6f} um")
    if config.resolution_mode == "coarse":
        print(
            f"coarse FOV: {optimization_fov_um[0]:.6f} x "
            f"{optimization_fov_um[1]:.6f} um"
        )
    print(
        f"device={device} dt={AIEParameters().dt:.3f}s "
        f"control_dt={AIEParameters().dt * args.physics_steps_per_control:.3f}s "
        f"horizon={args.horizon} iterations={args.iterations}"
    )


def run_demo(args: argparse.Namespace) -> None:
    """Run native MPC or coarse MPC followed by required native replay."""

    if args.control_steps < 1:
        raise ValueError(f"control_steps must be at least 1, got {args.control_steps}")
    config = build_resolution_config(args.resolution_mode, args.coarsen_factor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_path = resolve_target_path(args.target)
    target_native = load_normalized_target(target_path)
    require_native_target(target_native, target_path)
    target_native = target_native.to(device=device)
    optimization_target = construct_optimization_target(target_native, config)

    native_params = native_aie_parameters()
    optimization_params = (
        native_params
        if config.resolution_mode == "native"
        else replace(
            native_params,
            pixel_pitch_m=config.optimization_pixel_pitch_m,
            projector_refinement=1,
        )
    )
    model = AIEModel(optimization_params, device=device)
    state = model.initialize_state(config.optimization_shape)
    controller = DifferentiableMPC(
        model=model,
        target=optimization_target,
        horizon=args.horizon,
        physics_steps_per_control=args.physics_steps_per_control,
        target_doc=args.target_doc,
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
    coarse_doc_frame_paths: list[Path] = []
    native_doc_frame_paths: list[Path] = []
    if config.resolution_mode == "coarse":
        initial_doc_path = args.output_dir / "doc_frame_coarse_000.png"
        save_doc_frame(state.doc, initial_doc_path, config.optimization_shape)
        coarse_doc_frame_paths.append(initial_doc_path)
    else:
        initial_doc_path = args.output_dir / "doc_frame_native_000.png"
        save_doc_frame(state.doc, initial_doc_path, NATIVE_GRID)
        native_doc_frame_paths.append(initial_doc_path)
    _print_resolution_header(config, device, args)

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
        optimized_controls, info = controller.optimize(state, initial_guess=guess)
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
            save_doc_frame(state.doc, doc_frame_path, NATIVE_GRID)
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
        elapsed = time.perf_counter() - started
        print(
            f"step {control_step + 1:02d}/{args.control_steps:02d} "
            f"loss {info['initial_loss']:.5f}->{info['final_loss']:.5f} "
            f"DoC(target/out)={target_doc_mean:.4f}/{outside_doc_mean:.4f} "
            f"mask_mean={float(applied_control.mean()):.3f} time={elapsed:.2f}s"
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

        def save_native_replay_frame(frame_index: int, doc: torch.Tensor) -> None:
            frame_path = args.output_dir / f"doc_frame_native_{frame_index:03d}.png"
            save_doc_frame(doc, frame_path, NATIVE_GRID)
            native_doc_frame_paths.append(frame_path)

        native_state = replay_native_controls(
            native_params,
            applied_controls_native,
            args.physics_steps_per_control,
            device,
            doc_frame_callback=save_native_replay_frame,
        )
    else:
        native_state = optimization_state

    if len(native_doc_frame_paths) != args.control_steps + 1:
        raise AssertionError("native DoC timeline has an unexpected frame count")
    native_gif_path = args.output_dir / "doc_evolution_native.gif"
    save_gif_from_frames(native_doc_frame_paths, native_gif_path)

    for name, field in zip(
        ("o2", "tempo", "dose", "doc"), native_state.tensors()
    ):
        if tuple(field.shape) != NATIVE_GRID:
            raise AssertionError(
                f"final native {name} must have shape {NATIVE_GRID}, got "
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
    common_results: dict[str, object] = {
        "resolution_mode": np.asarray(config.resolution_mode),
        "native_shape": np.asarray(config.native_shape, dtype=np.int64),
        "optimization_shape": np.asarray(config.optimization_shape, dtype=np.int64),
        "coarsen_factor": np.int64(config.coarsen_factor),
        "native_pixel_pitch_m": np.float64(config.native_pixel_pitch_m),
        "optimization_pixel_pitch_m": np.float64(
            config.optimization_pixel_pitch_m
        ),
        "dt": np.float64(native_params.dt),
        "physics_steps_per_control": np.int64(args.physics_steps_per_control),
        "horizon": np.int64(args.horizon),
        "target_native": target_native.detach().cpu().numpy(),
        "applied_controls_native": torch.stack(applied_controls_native).cpu().numpy(),
        "final_o2_native": native_state.o2.detach().cpu().numpy(),
        "final_tempo_native": native_state.tempo.detach().cpu().numpy(),
        "final_dose_native": native_state.dose.detach().cpu().numpy(),
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
                "final_doc_coarse": optimization_state.doc.detach().cpu().numpy(),
            }
        )
    results_name = f"mpc_results_{config.resolution_mode}.npz"
    np.savez_compressed(args.output_dir / results_name, **common_results)

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


def _legacy_reference_step(
    model: AIEModel, state: AIEState, normalized_mask: torch.Tensor
) -> AIEState:
    """Independent one-step transcription of the v1.1 physical loop."""

    params = model.params
    blur_mask = _legacy_full_gaussian(normalized_mask, model.scattering_kernel_2d)
    local_intensity = (
        blur_mask.clamp_min(params.minimum_normalized_intensity)
        * params.intensity_mw_cm2
    )
    energy = local_intensity * params.dt
    b = params.b_slope * local_intensity + params.b_intercept

    # v1.1 overwrites its convolved O2 field with the previous local field.
    o2_diffused = state.o2
    tempo_diffused = _legacy_full_gaussian(state.tempo, model.tempo_kernel_2d)
    o2_next = torch.clamp(o2_diffused - energy, min=0)
    tempo_next = torch.where(
        o2_next <= 0,
        torch.clamp(tempo_diffused - energy, min=0),
        tempo_diffused,
    )
    curing = (o2_next <= 0) & (tempo_next <= 0)
    dose_next = torch.where(
        curing,
        state.dose + energy - o2_diffused - tempo_diffused,
        state.dose,
    )
    exposure_time = dose_next / local_intensity.clamp_min(params.division_epsilon)
    doc_candidate = 1 - torch.exp(-torch.clamp(b * exposure_time, min=0))
    doc_next = torch.where(curing, doc_candidate, state.doc)
    return AIEState(o2_next, tempo_next, dose_next, doc_next)


def _static_mask_equivalence_test(device: torch.device) -> float:
    model = AIEModel(device=device)
    height = width = model.scattering_kernel_size // 2 + 3
    coordinates = torch.linspace(0, 1, height, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    mask = 0.25 + 0.65 * torch.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.08)
    initial = AIEState(
        o2=0.05 + 0.20 * xx,
        tempo=0.02 + 0.18 * yy,
        dose=0.03 + 0.05 * xx * yy,
        doc=0.01 + 0.02 * xx,
    )
    refactored, reference = initial, initial
    for _ in range(2):
        refactored = model.step(refactored, mask)
        reference = _legacy_reference_step(model, reference, mask)
    maximum_error = max(
        float((actual - expected).abs().max())
        for actual, expected in zip(refactored.tensors(), reference.tensors())
    )
    for actual, expected in zip(refactored.tensors(), reference.tensors()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    return maximum_error


def _gradient_test(device: torch.device) -> tuple[float, int]:
    # The default 27.51 mJ/cm^2 O2 threshold creates the legacy zero-gradient
    # pre-cure interval. Setting only that calibrated initial condition to
    # zero exercises the ten-step differentiability test without changing the
    # transition equations.
    params = replace(AIEParameters(), o2_inhibition_mj_cm2=0.0)
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
    params = replace(AIEParameters(), o2_inhibition_mj_cm2=0.0)
    model = AIEModel(params, device=device)
    size = model.scattering_kernel_size // 2 + 3
    target = torch.ones((size, size), device=device)
    state = model.initialize_state(target.shape)
    controller = DifferentiableMPC(
        model=model,
        target=target,
        horizon=2,
        physics_steps_per_control=5,
        target_doc=0.16,
        outside_weight=0.0,
        energy_weight=0.0,
        smoothness_weight=0.0,
        num_iterations=10,
        learning_rate=0.3,
    )
    guess = torch.full((2, size, size), 0.2, device=device)
    _, info = controller.optimize(state, initial_guess=guess)
    initial_loss = info["initial_loss"]
    final_loss = info["final_loss"]
    if not final_loss < initial_loss:
        raise AssertionError(
            f"MPC loss did not decrease: {initial_loss} -> {final_loss}"
        )
    return initial_loss, final_loss


def _constant_block_assertion(
    coarse: torch.Tensor, native: torch.Tensor, factor: int
) -> None:
    height, width = coarse.shape
    blocks = native.reshape(height, factor, width, factor)
    expected = coarse[:, None, :, None].expand_as(blocks)
    torch.testing.assert_close(blocks, expected, rtol=0, atol=0)


def _resolution_modes_test(device: torch.device) -> dict[str, object]:
    """Validate native, q=2, q=3, FOV, block recovery, and native replay."""

    target_native = torch.zeros(NATIVE_GRID, device=device)
    target_native[0, 0] = 1.0

    native_config = build_resolution_config("native", 2)
    native_target = construct_optimization_target(target_native, native_config)
    native_control = torch.full(NATIVE_GRID, 0.4, device=device)
    native_applied = recover_control_to_native(native_control, native_config)
    if native_target.data_ptr() != target_native.data_ptr():
        raise AssertionError("native mode must use the original target without resampling")
    if tuple(native_target.shape) != NATIVE_GRID or tuple(native_applied.shape) != NATIVE_GRID:
        raise AssertionError("native target and applied mask must remain 300 x 300")

    coarse_controls: dict[int, torch.Tensor] = {}
    kernel_sizes: dict[int, int] = {}
    for factor, expected_shape, expected_fraction in (
        (2, (150, 150), 0.25),
        (3, (100, 100), 1.0 / 9.0),
    ):
        config = build_resolution_config("coarse", factor)
        target_coarse = construct_optimization_target(target_native, config)
        if tuple(target_coarse.shape) != expected_shape:
            raise AssertionError(
                f"q={factor} target shape {tuple(target_coarse.shape)} != {expected_shape}"
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
        if tuple(recovered.shape) != NATIVE_GRID:
            raise AssertionError(f"q={factor} recovery did not produce 300 x 300")
        _constant_block_assertion(coarse, recovered, factor)
        coarse_controls[factor] = coarse
        coarse_model = AIEModel(
            replace(
                native_aie_parameters(),
                pixel_pitch_m=config.optimization_pixel_pitch_m,
            ),
            device=device,
        )
        kernel_sizes[factor] = coarse_model.scattering_kernel_size

    native_model = AIEModel(native_aie_parameters(), device=device)
    kernel_sizes[1] = native_model.scattering_kernel_size
    if not kernel_sizes[1] > kernel_sizes[2] > kernel_sizes[3]:
        raise AssertionError(
            f"coarse physical kernels were not recomputed as expected: {kernel_sizes}"
        )

    replay_config = build_resolution_config("coarse", 3)
    replay_masks = [
        recover_control_to_native(coarse_controls[3] * level, replay_config)
        for level in (0.5, 0.75)
    ]
    replay_state = replay_native_controls(
        native_aie_parameters(), replay_masks, physics_steps_per_control=1, device=device
    )
    for field in replay_state.tensors():
        if tuple(field.shape) != NATIVE_GRID:
            raise AssertionError("native replay state field is not 300 x 300")
        if not bool(torch.isfinite(field).all()):
            raise AssertionError("native replay state contains NaN or Inf")
    if float(replay_state.doc.min()) < 0 or float(replay_state.doc.max()) > 1:
        raise AssertionError("native replay DoC left the valid [0, 1] range")

    try:
        require_native_target(torch.zeros((299, 300), device=device))
    except ValueError as error:
        if "never resized, cropped, or padded" not in str(error):
            raise AssertionError(f"unexpected target shape error: {error}") from error
    else:
        raise AssertionError("non-300 x 300 target was not rejected")

    return {
        "q2_shape": (150, 150),
        "q3_shape": (100, 100),
        "native_fov_um": native_config.native_fov_m[0] * 1e6,
        "kernel_sizes": kernel_sizes,
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
        for label, shape, levels in (
            # Identical frames confirm that inhibited zero-DoC time steps are
            # retained instead of being merged by the GIF encoder.
            ("native", NATIVE_GRID, (0.0, 0.0, 0.0)),
            ("coarse", (150, 150), (0.0, 0.25, 0.5)),
        ):
            frame_paths: list[Path] = []
            for frame_index, level in enumerate(levels):
                doc = torch.full(shape, level, device=device)
                frame_path = output_dir / f"doc_frame_{label}_{frame_index:03d}.png"
                save_doc_frame(doc, frame_path, shape)
                frame_paths.append(frame_path)
            gif_path = output_dir / f"doc_evolution_{label}.gif"
            save_gif_from_frames(frame_paths, gif_path, duration_ms=300)
            validate_gif(gif_path, len(levels), shape)
            results[f"{label}_frames"] = len(levels)
            results[f"{label}_shape"] = shape

        invalid_doc = torch.zeros(NATIVE_GRID, device=device)
        invalid_doc[0, 0] = torch.nan
        try:
            validate_doc_field(invalid_doc, NATIVE_GRID)
        except ValueError as error:
            if "NaN or Inf" not in str(error):
                raise AssertionError(f"unexpected DoC validation error: {error}") from error
        else:
            raise AssertionError("non-finite DoC frame was not rejected")
        return results


def run_smoke_tests() -> None:
    """Run the existing and resolution-mode lightweight validation suite."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"running smoke tests on {device}")
    equivalence_error = _static_mask_equivalence_test(device)
    print(f"[pass] static-mask equivalence max_abs_error={equivalence_error:.3e}")
    gradient_norm, nonzero = _gradient_test(device)
    print(f"[pass] gradient norm={gradient_norm:.3e}, nonzero_elements={nonzero}")
    shape_message = _shape_validation_test(device)
    print(f"[pass] shape validation: {shape_message}")
    initial_loss, final_loss = _mpc_optimization_test(device)
    print(f"[pass] MPC loss {initial_loss:.6e}->{final_loss:.6e}")
    resolution = _resolution_modes_test(device)
    print(
        f"[pass] resolution grids native={NATIVE_GRID} "
        f"q=2->{resolution['q2_shape']} q=3->{resolution['q3_shape']}"
    )
    print(
        f"[pass] physical FOV={resolution['native_fov_um']:.6f} um per axis; "
        f"scattering kernels={resolution['kernel_sizes']}"
    )
    print(
        f"[pass] native replay shape={NATIVE_GRID} finite=True "
        f"DoC_range={resolution['replay_doc_range']}"
    )
    gif_results = _gif_visualization_test(device)
    print(
        f"[pass] DoC GIFs native={gif_results['native_frames']} frames at "
        f"{gif_results['native_shape']}, coarse={gif_results['coarse_frames']} "
        f"frames at {gif_results['coarse_shape']}, duration=300ms loop=continuous"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("circle.png"),
        help="300 x 300 target; unqualified names resolve inside GEO/",
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
    parser.add_argument("--control-steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--target-doc", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
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
