"""Conservative receding-horizon demonstration for the differentiable AIE MPC.

The defaults are intended as a small demonstration, not final research
settings.  Run ``python run_mpc.py --smoke-test`` for the lightweight model,
gradient, shape, and optimizer validations.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aie_model import AIEModel, AIEParameters, AIEState
from aie_mpc import DifferentiableMPC


REPOSITORY_DIR = Path(__file__).resolve().parent


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


def resize_for_demo(target: torch.Tensor, maximum_dimension: int) -> torch.Tensor:
    """Make an aspect-preserving reduced-domain target for the smoke-scale demo."""

    if maximum_dimension <= 0 or max(target.shape) <= maximum_dimension:
        return target
    scale = maximum_dimension / max(target.shape)
    output_shape = tuple(max(1, round(dimension * scale)) for dimension in target.shape)
    return F.interpolate(
        target[None, None],
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def save_grayscale(tensor: torch.Tensor, path: Path) -> None:
    """Save a normalized tensor as a 16-bit grayscale PNG."""

    values = tensor.detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.round(values * 65535).astype(np.uint16)).save(path)


def run_demo(args: argparse.Namespace) -> None:
    """Run a complete nonlinear receding-horizon simulation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = resize_for_demo(
        load_normalized_target(args.target), args.demo_size
    ).to(device)
    params = AIEParameters()
    model = AIEModel(params, device=device)
    state = model.initialize_state(target.shape)
    controller = DifferentiableMPC(
        model=model,
        target=target,
        horizon=args.horizon,
        physics_steps_per_control=args.physics_steps_per_control,
        target_doc=args.target_doc,
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
    save_grayscale(target, args.output_dir / "target.png")
    applied_controls: list[np.ndarray] = []
    optimization_histories: list[list[float]] = []
    target_region = controller.target_region.bool()
    outside_region = controller.outside_region.bool()

    print(
        f"device={device} grid={tuple(target.shape)} dt={params.dt:.3f}s "
        f"control_dt={params.dt * args.physics_steps_per_control:.3f}s "
        f"horizon={args.horizon} iterations={args.iterations}"
    )
    horizon_energy_at_full_power = (
        params.intensity_mw_cm2
        * params.dt
        * args.physics_steps_per_control
        * args.horizon
    )
    if params.o2_inhibition_mj_cm2 > horizon_energy_at_full_power:
        print(
            "note: the initial horizon is shorter than the legacy hard O2 "
            "inhibition time; DoC tracking gradients appear after receding "
            "exposure has depleted enough O2"
        )
    for control_step in range(args.control_steps):
        started = time.perf_counter()
        optimized_controls, info = controller.optimize(state, initial_guess=guess)
        applied_control = optimized_controls[0]
        state = model.advance(
            state,
            applied_control,
            physics_steps=args.physics_steps_per_control,
        )
        guess = controller.shift_warm_start(optimized_controls)
        applied_controls.append(applied_control.cpu().numpy())
        optimization_histories.append(info["loss_history"])
        save_grayscale(
            applied_control,
            args.output_dir / f"applied_mask_{control_step:03d}.png",
        )

        target_doc_mean = float(state.doc[target_region].mean())
        outside_doc_mean = (
            float(state.doc[outside_region].mean())
            if bool(outside_region.any())
            else 0.0
        )
        elapsed = time.perf_counter() - started
        print(
            f"step {control_step + 1:02d}/{args.control_steps:02d} "
            f"loss {info['initial_loss']:.5f}->{info['final_loss']:.5f} "
            f"DoC(target/out)={target_doc_mean:.4f}/{outside_doc_mean:.4f} "
            f"mask_mean={float(applied_control.mean()):.3f} time={elapsed:.2f}s"
        )

    save_grayscale(state.doc, args.output_dir / "final_doc.png")
    save_grayscale(state.o2 / max(params.o2_inhibition_mj_cm2, 1e-12), args.output_dir / "final_o2.png")
    np.savez_compressed(
        args.output_dir / "mpc_results.npz",
        target=target.detach().cpu().numpy(),
        applied_controls=np.stack(applied_controls),
        final_o2=state.o2.detach().cpu().numpy(),
        final_tempo=state.tempo.detach().cpu().numpy(),
        final_dose=state.dose.detach().cpu().numpy(),
        final_doc=state.doc.detach().cpu().numpy(),
        loss_histories=np.asarray(optimization_histories),
        dt=params.dt,
        physics_steps_per_control=args.physics_steps_per_control,
    )
    print(f"saved demonstration outputs to {args.output_dir.resolve()}")


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
    # pre-cure interval.  Setting only that calibrated initial condition to
    # zero exercises the requested ten-step differentiability test without
    # altering the transition equations.
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


def run_smoke_tests() -> None:
    """Run the required lightweight validation suite."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=REPOSITORY_DIR / "circle_DLP.png",
        help="target image (default: repository circle_DLP.png)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_DIR / "mpc_output"
    )
    parser.add_argument(
        "--demo-size",
        type=int,
        default=96,
        help="maximum target dimension; <=0 keeps native resolution",
    )
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--physics-steps-per-control", type=int, default=10)
    parser.add_argument("--control-steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--target-doc", type=float, default=0.9)
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
