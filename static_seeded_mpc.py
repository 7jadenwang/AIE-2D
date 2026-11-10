"""Opt-in matched static versus shrinking-horizon MPC benchmark.

Normal run_mpc.py objectives and policies are not used or changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from aie_model import AIEModel, AIEParameters
from aie_reference import load_reference_config_for_condition
from mpc_metrics import calculate_final_metrics
from run_mpc import load_normalized_target, save_grayscale, _add_pixel_scale_bar


def objective(state, desired):
    return (state.doc - desired).square().sum()


def valid_state(state):
    return all(bool(torch.isfinite(field).all()) for field in state.tensors())


def score(forward, controls, desired):
    with torch.no_grad():
        if not bool(torch.isfinite(controls).all()) or bool(((controls < 0) | (controls > 1)).any()):
            return math.inf
        state = forward(controls)
        loss = objective(state, desired)
        if not valid_state(state) or not bool(torch.isfinite(loss)):
            return math.inf
        return float(loss)


def optimize(forward, seed, desired, iterations, learning_rate=0.05, patience=30):
    """Projected Adam; retain the exact incumbent and every valid best iterate.

    Projection uses the existing [0,1] physical bounds, avoiding sigmoid
    saturation at the initially dark background. No regularizers are added.
    """
    best = seed.detach().clone()
    best_loss = score(forward, best, desired)
    if not math.isfinite(best_loss):
        raise ValueError("Nonfinite incumbent replay")
    controls = torch.nn.Parameter(best.clone())
    adam = torch.optim.Adam([controls], lr=learning_rate)
    history = [{"iteration": 0, "candidate_J": best_loss, "best_J": best_loss}]
    stale = 0
    stop = "iteration_limit"
    for iteration in range(1, iterations + 1):
        adam.zero_grad(set_to_none=True)
        state = forward(controls)
        loss = objective(state, desired)
        if not valid_state(state) or not bool(torch.isfinite(loss)):
            stop = "nonfinite_forward_retained_incumbent"
            break
        loss.backward()
        if controls.grad is None or not bool(torch.isfinite(controls.grad).all()):
            stop = "nonfinite_gradient_retained_incumbent"
            break
        torch.nn.utils.clip_grad_norm_([controls], 10.0)
        adam.step()
        with torch.no_grad():
            controls.clamp_(0, 1)
        candidate = score(forward, controls, desired)
        previous = best_loss
        if candidate < best_loss:
            best_loss, best = candidate, controls.detach().clone()
        history.append({"iteration": iteration,
                        "candidate_J": candidate if math.isfinite(candidate) else None,
                        "best_J": best_loss})
        stale = 0 if previous - best_loss > 1e-6 * max(1.0, previous) else stale + 1
        if not math.isfinite(candidate):
            stop = "nonfinite_candidate_retained_incumbent"
            break
        if stale >= patience:
            stop = "early_stopping"
            break
    return best, history, stop


def accept_candidate(forward, incumbent, candidate, desired):
    """Fresh replays from the same actual state; only strict improvement wins."""
    old = score(forward, incumbent, desired)
    new = score(forward, candidate, desired)
    if not math.isfinite(old):
        raise ValueError("Invalid remaining incumbent")
    accepted = new < old
    return (candidate if accepted else incumbent).detach().clone(), old, new, accepted


def verify_static_seed(model, initial, static, count, steps, desired):
    sequence = static.unsqueeze(0).repeat(count, 1, 1)
    with torch.no_grad():
        single = model.advance(initial, static, count * steps)
        repeated = model.rollout(initial, sequence, steps)
        for a, b in zip(single.tensors(), repeated.tensors()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(objective(single, desired), objective(repeated, desired),
                                   rtol=1e-5, atol=1e-7)
    return sequence


def synchronize(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def summarize(state, target, params):
    doc = state.doc.detach().cpu().numpy()
    mask = target.detach().cpu().numpy().astype(bool)
    metrics = calculate_final_metrics(doc, mask, 0.90, 0.5, 0.5,
                                      params.native_pixel_pitch_m * 1e6)
    return {
        "final_DoC_frobenius": metrics["final_DoC_frobenius"],
        "J": metrics["final_DoC_frobenius"] ** 2,
        "target_DoC_mean": float(doc[mask].mean()),
        "target_DoC_min": float(doc[mask].min()),
        "target_DoC_max": float(doc[mask].max()),
        "outside_DoC_mean": float(doc[~mask].mean()) if (~mask).any() else 0.0,
        "outside_DoC_max": float(doc[~mask].max()) if (~mask).any() else 0.0,
        "iou": metrics["geometry"]["iou"], "dice": metrics["geometry"]["dice"],
    }


def save_plots(out, desired, static_doc, dynamic_doc, static_history, solves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def draw_fields(fig, axes):
        for ax, title, field in zip(axes, ("Desired DoC", "Optimized static", "MPC"),
                                    (desired, static_doc, dynamic_doc)):
            array = field.detach().cpu().numpy()
            rendered = _add_pixel_scale_bar(np.round(np.clip(array, 0, 1) * 65535).astype(np.uint16))
            im = ax.imshow(rendered.astype(float) / 65535, cmap="gray", vmin=0, vmax=1,
                           aspect="equal", interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.colorbar(im, ax=list(axes), label="DoC", shrink=0.65)

    def draw_convergence(ax):
        ax.plot([h["iteration"] for h in static_history],
                [math.sqrt(h["best_J"]) for h in static_history], label="Static")
        # Concatenate successive solves; preserve their actual saved losses.
        iterations, losses, offset = [], [], 0
        for solve in solves:
            history = solve["history"]
            iterations.extend(offset + h["iteration"] for h in history)
            losses.extend(math.sqrt(h["best_J"]) for h in history)
            offset += history[-1]["iteration"]
        ax.plot(iterations, losses, label="MPC")
        ax.set_xlim(0, 300)
        ax.set_title("Convergence comparison")
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("Full-image DoC Frobenius")
        ax.legend()
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    draw_fields(fig, axes)
    fig.savefig(out / "static_vs_mpc_final_doc.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    draw_convergence(ax)
    fig.savefig(out / "static_vs_mpc_loss.png", dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1))
    draw_fields(fig, [fig.add_subplot(grid[0, i]) for i in range(3)])
    draw_convergence(fig.add_subplot(grid[1, :]))
    fig.savefig(out / "figure4.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("GEO/seg/Picture4.png"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--static-iterations", type=int, default=1000)
    parser.add_argument("--dynamic-iterations", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.static_iterations <= 1000 or not 1 <= args.dynamic_iterations <= 200:
        parser.error("static iterations must be 1..1000; dynamic iterations 1..200")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    reference = load_reference_config_for_condition("30mW_0mM")
    params = AIEParameters.from_reference(reference)
    print(f"Matched current physics: B={params.b_coefficient}*I**{params.b_exponent}; "
          f"pre_slope={params.pre_slope}; source SHA256={params.reference_model_sha256}", flush=True)
    model = AIEModel(params, device=args.device)
    target = (load_normalized_target(args.target) > 0.5).to(device=model.device, dtype=model.dtype)
    if not bool(target.any()):
        parser.error("target mask is empty")
    count, duration = 8, 4.0
    steps = round(duration / (count * params.dt))
    if steps < 1 or not math.isclose(count * steps * params.dt, duration, abs_tol=1e-10):
        parser.error("source dt cannot represent eight equal intervals over four seconds")
    if params.chain_growth_noise_enabled:
        parser.error("benchmark requires deterministic current physics")
    desired = 0.90 * target
    initial = model.initialize_state(target.shape)
    args.output_dir.mkdir(parents=True)
    synchronize(model)
    start = time.perf_counter()
    static, static_history, static_stop = optimize(
        lambda u: model.advance(initial, u, count * steps), 0.9 * target,
        desired, args.static_iterations,
    )
    synchronize(model)
    static_seconds = time.perf_counter() - start
    # Save protected static artifacts before attempting any dynamic optimization.
    save_grayscale(static, args.output_dir / "optimized_static_mask.png")
    with torch.no_grad():
        static_state = model.advance(initial, static, count * steps)
    static_metrics = summarize(static_state, target, params)
    np.savez_compressed(args.output_dir / "optimized_static.npz",
                        mask=static.cpu().numpy(), doc=static_state.doc.cpu().numpy())
    (args.output_dir / "static_optimization.json").write_text(json.dumps({
        "metrics": static_metrics, "history": static_history, "stop": static_stop,
        "runtime_seconds": static_seconds}, indent=2, allow_nan=False))
    remaining = verify_static_seed(model, initial, static, count, steps, desired)
    state = initial.detach()
    applied, solves = [], []
    synchronize(model)
    start = time.perf_counter()
    for index in range(count):
        actual = state.detach()
        forward = lambda u: model.rollout(actual, u, steps)
        candidate, history, stop = optimize(forward, remaining, desired, args.dynamic_iterations)
        remaining, old, new, accepted = accept_candidate(forward, remaining, candidate, desired)
        solves.append({"time_s": index * steps * params.dt,
                       "remaining_intervals": count - index, "final_time_s": duration,
                       "incumbent_J": old, "candidate_J": new if math.isfinite(new) else None,
                       "accepted": accepted, "stop": stop, "history": history})
        applied.append(remaining[0].clone())
        with torch.no_grad():
            state = model.advance(actual, applied[-1], steps).detach()
        remaining = remaining[1:].clone()
        print(f"solve {index+1}/{count}: final Frobenius={math.sqrt(min(old,new)):.8g}; accepted={accepted}", flush=True)
    synchronize(model)
    dynamic_seconds = time.perf_counter() - start
    applied = torch.stack(applied)
    # Independent fresh final replays, never metrics from Adam's cached states.
    with torch.no_grad():
        static_state = model.advance(initial, static, count * steps)
        dynamic_state = model.rollout(initial, applied, steps)
    if not valid_state(static_state) or not valid_state(dynamic_state):
        raise ValueError("Nonfinite final replay; protected static artifacts retained")
    static_metrics = summarize(static_state, target, params)
    dynamic_metrics = summarize(dynamic_state, target, params)
    sf, df = static_metrics["final_DoC_frobenius"], dynamic_metrics["final_DoC_frobenius"]
    improved = df < sf - max(1e-7, 1e-6 * sf)
    status = "Dynamic improvement demonstrated." if improved else "No dynamic improvement demonstrated."
    mask_dir = args.output_dir / "optimized_mpc_masks"
    mask_dir.mkdir()
    for index, mask in enumerate(applied):
        save_grayscale(mask, mask_dir / f"mask_{index:03d}.png")
    np.savez_compressed(args.output_dir / "static_vs_mpc_fields.npz",
                        target=target.cpu().numpy(), desired_doc=desired.cpu().numpy(),
                        static_mask=static.cpu().numpy(), mpc_masks=applied.cpu().numpy(),
                        static_doc=static_state.doc.cpu().numpy(), mpc_doc=dynamic_state.doc.cpu().numpy())
    report = {"status": status, "dynamic_improvement_demonstrated": improved,
              "static": static_metrics, "mpc": dynamic_metrics,
              "percentage_improvement": 100 * (sf - df) / sf if sf else None,
              "objective": "sum_pixels((DoC_final - 0.90 * (target > 0.5))**2)",
              "target_path": str(args.target.resolve()), "target_threshold": 0.5,
              "geometry_threshold": 0.5, "control_bounds": [0.0, 1.0],
              "final_time_s": duration, "control_intervals": count,
              "physics_steps_per_control": steps, "static_seed_equivalence": "passed",
              "parameters": params.provenance_metadata(), "reference": reference.to_metadata(),
              "optimizer": {"name": "projected Adam", "learning_rate": 0.05,
                            "gradient_clip_norm": 10.0, "patience": 30,
                            "static_max_iterations": args.static_iterations,
                            "dynamic_max_iterations_per_solve": args.dynamic_iterations},
              "runtime_seconds": {"static_optimization": static_seconds,
                                  "dynamic_optimization_and_execution": dynamic_seconds},
              "static_history": static_history, "static_stop": static_stop, "dynamic_solves": solves,
              "historical_comparison": "4.166 used a different pre-cure slope; not a matched benchmark",
              "png_note": "PNG controls are quantized previews; NPZ stores exact replay controls"}
    (args.output_dir / "static_vs_mpc_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save_plots(args.output_dir, desired, static_state.doc, dynamic_state.doc, static_history, solves)
    print(f"Static Frobenius: {sf:.8g}\nMPC Frobenius: {df:.8g}\n{status}")


if __name__ == "__main__":
    main()
