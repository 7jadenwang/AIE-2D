"""Diagnose open-loop reachability of experimental DoC trajectories.

This standalone diagnostic runs the production differentiable AIE forward
physics with two fixed, full-power controls.  It never constructs an MPC
optimizer and never changes the experimental reference or physical model.

Experiments for each matched condition:

* uniform: projector control ``u = 1`` everywhere for the entire run;
* target: ``u = 1`` on the binary L-shape target and ``u = 0`` elsewhere.

The experimental reference is always queried in absolute process time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aie_model import DOC_HISTORY_DESCRIPTION, DOC_HISTORY_MODE, AIEModel, AIEParameters
from aie_reference import load_reference_config_for_condition
from doc_reference import LEGACY_REFERENCE_PATH, DoCReferenceCurve, load_doc_reference
from run_mpc import (
    assess_reference_physics_match,
    load_normalized_target,
    require_native_target,
    resolve_target_path,
)


CONDITION_IDS = ("30mW_0mM", "30mW_5mM")
CONDITION_TITLES = {
    "30mW_0mM": "30 mW/cm^2 / 0 mM TEMPO",
    "30mW_5mM": "30 mW/cm^2 / 5 mM TEMPO",
}
THRESHOLDS = (0.10, 0.30, 0.50, 0.90, 0.95, 0.99)
DEFAULT_MPC_OUTPUTS = {
    "30mW_0mM": REPOSITORY_DIR
    / "results"
    / "mpc"
    / "tracking_Lshape_30mW_0mM_iso_20s",
    "30mW_5mM": REPOSITORY_DIR
    / "results"
    / "mpc"
    / "tracking_Lshape_30mW_5mM_iso_20s",
}
DEFAULT_RESULTS_DIR = (
    REPOSITORY_DIR / "results" / "diagnostics" / "open_loop_feasibility"
)
CLASSIFICATION_RULE = {
    "material_forward_limit": (
        "maximum target-full-power feasibility deficit > 0.05, or duration "
        "with deficit > 0.05 is at least 0.5 s"
    ),
    "material_controller_gap": (
        "rise-region RMS positive gap between target-full-power and MPC DoC "
        "> 0.02 while rise-region mean target-region MPC control is < 0.95"
    ),
    "scope_note": (
        "These fixed diagnostic cutoffs support an approximate attribution; "
        "they are not fitted or calibrated decision thresholds."
    ),
}


@dataclass(frozen=True)
class OpenLoopTrace:
    """One fixed-control state trajectory sampled at every physics step."""

    time_s: np.ndarray
    doc_mean: np.ndarray
    outside_doc_mean: np.ndarray
    o2_mean: np.ndarray
    tempo_mean: np.ndarray
    dose_mean: np.ndarray
    reaction_progress_mean: np.ndarray
    control_definition: str
    control_min: float
    control_max: float
    control_target_mean: float
    control_outside_mean: float
    validation: dict[str, Any]


@dataclass(frozen=True)
class MPCComparison:
    """Closed-loop values loaded from an existing production result archive."""

    source_path: Path
    control_times_s: np.ndarray
    target_doc: np.ndarray
    reference_doc: np.ndarray
    target_control_mean: np.ndarray
    summary: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Recursively map NumPy/Path values into strict JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _masked_mean(field: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> float:
    return float(((field * mask).sum() / count).detach().cpu())


def _record_state(
    state: Any,
    target_mask: torch.Tensor | None,
    outside_mask: torch.Tensor | None,
    target_count: torch.Tensor | None,
    outside_count: torch.Tensor | None,
) -> tuple[float, float, float, float, float, float]:
    """Return DoC/outside DoC and four state means for one state."""

    if target_mask is None:
        doc = float(state.doc.mean().detach().cpu())
        outside_doc = doc
        return (
            doc,
            outside_doc,
            float(state.o2.mean().detach().cpu()),
            float(state.tempo.mean().detach().cpu()),
            float(state.dose.mean().detach().cpu()),
            float(state.reaction_progress.mean().detach().cpu()),
        )
    assert outside_mask is not None
    assert target_count is not None and outside_count is not None
    return (
        _masked_mean(state.doc, target_mask, target_count),
        _masked_mean(state.doc, outside_mask, outside_count),
        _masked_mean(state.o2, target_mask, target_count),
        _masked_mean(state.tempo, target_mask, target_count),
        _masked_mean(state.dose, target_mask, target_count),
        _masked_mean(state.reaction_progress, target_mask, target_count),
    )


def _state_is_finite(state: Any) -> bool:
    checks = torch.stack([torch.isfinite(field).all() for field in state.tensors()])
    return bool(checks.all().detach().cpu())


def _simulate_fixed_control(
    *,
    params: AIEParameters,
    state_shape: tuple[int, int],
    projector_control: torch.Tensor,
    total_time_s: float,
    device: torch.device,
    seed: int,
    control_definition: str,
    target_region: torch.Tensor | None = None,
) -> OpenLoopTrace:
    """Propagate one exactly constant control without an optimizer."""

    ratio = total_time_s / params.dt
    physics_steps = int(round(ratio))
    if not math.isclose(ratio, physics_steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"total_time={total_time_s:g} s is not compatible with dt={params.dt:g} s"
        )
    if tuple(projector_control.shape) != tuple(
        dimension // params.projector_refinement for dimension in state_shape
    ):
        raise ValueError("fixed projector control shape does not match model geometry")
    if float(projector_control.min()) < 0.0 or float(projector_control.max()) > 1.0:
        raise ValueError("fixed projector control must remain in [0,1]")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = AIEModel(params=params, device=device)
    control = projector_control.to(device=device, dtype=model.dtype)
    state = model.initialize_state(state_shape)
    prepared = model.prepare_control(control, state_shape)

    target_mask: torch.Tensor | None = None
    outside_mask: torch.Tensor | None = None
    target_count: torch.Tensor | None = None
    outside_count: torch.Tensor | None = None
    if target_region is not None:
        target_mask = target_region.to(device=device, dtype=model.dtype)
        outside_mask = 1.0 - target_mask
        target_count = target_mask.sum()
        outside_count = outside_mask.sum()
        if float(target_count.detach().cpu()) <= 0 or float(outside_count.detach().cpu()) <= 0:
            raise ValueError("target-shaped diagnostic requires nonempty target and outside")

    time_s = np.arange(physics_steps + 1, dtype=float) * params.dt
    rows: list[tuple[float, float, float, float, float, float]] = []
    finite_all_steps = True
    doc_min = math.inf
    doc_max = -math.inf
    reaction_progress_pointwise_min_step = math.inf
    reaction_progress_mean_min_step = math.inf
    reaction_progress_pointwise_decrease_steps = 0
    previous_reaction_progress = state.reaction_progress.detach().clone()

    with torch.no_grad():
        initial_row = _record_state(
            state, target_mask, outside_mask, target_count, outside_count
        )
        rows.append(initial_row)
        previous_reaction_progress_mean = initial_row[5]
        for step in range(1, physics_steps + 1):
            state = model.step_prepared(state, prepared)
            finite_all_steps = finite_all_steps and _state_is_finite(state)
            if not finite_all_steps:
                raise RuntimeError(
                    f"{control_definition} produced a nonfinite state at step {step}"
                )
            current_doc_min = float(state.doc.min().detach().cpu())
            current_doc_max = float(state.doc.max().detach().cpu())
            doc_min = min(doc_min, current_doc_min)
            doc_max = max(doc_max, current_doc_max)
            min_increment = float(
                (state.reaction_progress - previous_reaction_progress)
                .min()
                .detach()
                .cpu()
            )
            reaction_progress_pointwise_min_step = min(
                reaction_progress_pointwise_min_step, min_increment
            )
            if min_increment < -1e-6:
                reaction_progress_pointwise_decrease_steps += 1
            current_row = _record_state(
                state, target_mask, outside_mask, target_count, outside_count
            )
            mean_increment = current_row[5] - previous_reaction_progress_mean
            reaction_progress_mean_min_step = min(
                reaction_progress_mean_min_step, mean_increment
            )
            if mean_increment < -1e-6:
                raise RuntimeError(
                    f"{control_definition} mean reaction_progress decreased at step "
                    f"{step}: increment={mean_increment:.6g}"
                )
            previous_reaction_progress = state.reaction_progress.detach().clone()
            previous_reaction_progress_mean = current_row[5]
            rows.append(current_row)

    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{control_definition} recorded nonfinite state means")
    if doc_min < -1e-6 or doc_max > 1.0 + 1e-6:
        raise RuntimeError(
            f"{control_definition} DoC outside [0,1]: [{doc_min:.6g}, {doc_max:.6g}]"
        )
    control_cpu = control.detach().cpu()
    if target_region is None:
        control_target_mean = float(control_cpu.mean())
        control_outside_mean = float(control_cpu.mean())
    else:
        control_target_mean = float(control_cpu[target_region].mean())
        control_outside_mean = float(control_cpu[~target_region].mean())
    return OpenLoopTrace(
        time_s=time_s,
        doc_mean=values[:, 0],
        outside_doc_mean=values[:, 1],
        o2_mean=values[:, 2],
        tempo_mean=values[:, 3],
        dose_mean=values[:, 4],
        reaction_progress_mean=values[:, 5],
        control_definition=control_definition,
        control_min=float(control_cpu.min()),
        control_max=float(control_cpu.max()),
        control_target_mean=control_target_mean,
        control_outside_mean=control_outside_mean,
        validation={
            "all_states_finite": finite_all_steps,
            "doc_min": doc_min,
            "doc_max": doc_max,
            "reaction_progress_mean_minimum_step_increment": (
                reaction_progress_mean_min_step
            ),
            "reaction_progress_mean_monotonic": (
                reaction_progress_mean_min_step >= -1e-6
            ),
            "reaction_progress_pointwise_minimum_step_increment": (
                reaction_progress_pointwise_min_step
            ),
            "reaction_progress_pointwise_decrease_step_count": (
                reaction_progress_pointwise_decrease_steps
            ),
            "reaction_progress_pointwise_note": (
                "The unchanged governing equations can yield local negative "
                "increments at diffusion/cure-transition boundary pixels; the "
                "reported full-field or target-region mean is required monotonic."
            ),
            "sample_count": int(time_s.size),
            "time_start_s": float(time_s[0]),
            "time_end_s": float(time_s[-1]),
        },
    )


def _threshold_times(time_s: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    """Find first threshold crossings with linear interpolation."""

    result: dict[str, float | None] = {}
    for threshold in THRESHOLDS:
        label = f"t{int(round(threshold * 100)):02d}_s"
        indices = np.flatnonzero(values >= threshold)
        if not indices.size:
            result[label] = None
            continue
        index = int(indices[0])
        if index == 0:
            result[label] = float(time_s[0])
            continue
        t0, t1 = float(time_s[index - 1]), float(time_s[index])
        y0, y1 = float(values[index - 1]), float(values[index])
        if math.isclose(y1, y0, rel_tol=0.0, abs_tol=1e-15):
            result[label] = t1
        else:
            result[label] = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
    return result


def _threshold_deltas(
    simulated: dict[str, float | None], reference: dict[str, float | None]
) -> dict[str, float | None]:
    return {
        f"delta_{label}": (
            None
            if simulated[label] is None or reference[label] is None
            else float(simulated[label] - reference[label])
        )
        for label in reference
    }


def _error_metrics(
    time_s: np.ndarray,
    simulated: np.ndarray,
    reference: np.ndarray,
    reference_saturation_s: float,
    reference_thresholds: dict[str, float | None],
) -> dict[str, float | int | None]:
    error = simulated - reference

    def region(mask: np.ndarray) -> tuple[float | None, float | None, int]:
        count = int(mask.sum())
        if not count:
            return None, None, 0
        selected = error[mask]
        return (
            float(np.sqrt(np.mean(selected**2))),
            float(np.mean(np.abs(selected))),
            count,
        )

    full_rmse, full_mae, full_count = region(np.ones(time_s.shape, dtype=bool))
    pre_rmse, pre_mae, pre_count = region(
        time_s <= reference_saturation_s + 1e-12
    )
    t10 = reference_thresholds["t10_s"]
    t99 = reference_thresholds["t99_s"]
    rise_mask = (
        np.zeros(time_s.shape, dtype=bool)
        if t10 is None or t99 is None
        else (time_s >= t10 - 1e-12) & (time_s <= t99 + 1e-12)
    )
    rise_rmse, rise_mae, rise_count = region(rise_mask)
    return {
        "full_duration_rmse": full_rmse,
        "full_duration_mae": full_mae,
        "full_duration_sample_count": full_count,
        "pre_saturation_rmse": pre_rmse,
        "pre_saturation_mae": pre_mae,
        "pre_saturation_sample_count": pre_count,
        "rise_region_rmse": rise_rmse,
        "rise_region_mae": rise_mae,
        "rise_region_sample_count": rise_count,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "maximum_positive_feasibility_deficit": float(
            np.max(np.maximum(reference - simulated, 0.0))
        ),
    }


def _above_intervals(
    time_s: np.ndarray, values: np.ndarray, threshold: float
) -> list[tuple[float, float]]:
    """Return piecewise-linear intervals on which ``values > threshold``."""

    pieces: list[tuple[float, float]] = []
    for index in range(time_s.size - 1):
        t0, t1 = float(time_s[index]), float(time_s[index + 1])
        y0, y1 = float(values[index]), float(values[index + 1])
        above0, above1 = y0 > threshold, y1 > threshold
        if above0 and above1:
            pieces.append((t0, t1))
        elif above0 != above1:
            if math.isclose(y0, y1, rel_tol=0.0, abs_tol=1e-15):
                crossing = 0.5 * (t0 + t1)
            else:
                crossing = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
            pieces.append((t0, crossing) if above0 else (crossing, t1))

    merged: list[tuple[float, float]] = []
    for start, end in pieces:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _feasibility_summary(
    time_s: np.ndarray, reference: np.ndarray, target_doc: np.ndarray
) -> dict[str, Any]:
    deficit = np.maximum(reference - target_doc, 0.0)
    maximum_index = int(np.argmax(deficit))
    intervals_001 = _above_intervals(time_s, deficit, 0.01)
    intervals_005 = _above_intervals(time_s, deficit, 0.05)
    positive_intervals = _above_intervals(time_s, deficit, 0.0)
    largest = max(positive_intervals, key=lambda pair: pair[1] - pair[0], default=None)
    return {
        "maximum_deficit": float(deficit[maximum_index]),
        "time_of_maximum_deficit_s": float(time_s[maximum_index]),
        "duration_deficit_gt_0_01_s": float(
            sum(end - start for start, end in intervals_001)
        ),
        "duration_deficit_gt_0_05_s": float(
            sum(end - start for start, end in intervals_005)
        ),
        "largest_contiguous_infeasible_interval": (
            None
            if largest is None
            else {
                "start_s": float(largest[0]),
                "end_s": float(largest[1]),
                "duration_s": float(largest[1] - largest[0]),
            }
        ),
        "intervals_deficit_gt_0_01_s": [list(pair) for pair in intervals_001],
        "intervals_deficit_gt_0_05_s": [list(pair) for pair in intervals_005],
    }


def _resolve_npz(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".npz":
            raise ValueError(f"MPC comparison file must be NPZ: {resolved}")
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"MPC output path does not exist: {resolved}")
    preferred = sorted(resolved.glob("mpc_results_native.npz"))
    candidates = preferred or sorted(resolved.glob("*.npz"))
    if len(candidates) != 1:
        raise ValueError(
            f"MPC directory must resolve to one NPZ file, found {len(candidates)}: "
            f"{resolved}"
        )
    return candidates[0]


def _load_mpc_comparison(
    path: Path,
    reference_id: str,
    target_region: np.ndarray,
) -> MPCComparison:
    source = _resolve_npz(path)
    required = {
        "control_times_s",
        "reference_doc_values",
        "actual_mean_target_doc",
        "applied_controls_native",
        "doc_reference_id",
        "reference_model_sha256",
        "doc_reference_source_sha256",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"MPC archive {source} is missing keys: {missing}")
        stored_id = str(archive["doc_reference_id"].item())
        if stored_id != reference_id:
            raise ValueError(
                f"MPC archive reference ID {stored_id!r} does not match {reference_id!r}"
            )
        times = np.asarray(archive["control_times_s"], dtype=float)
        reference = np.asarray(archive["reference_doc_values"], dtype=float)
        target_doc = np.asarray(archive["actual_mean_target_doc"], dtype=float)
        controls = np.asarray(archive["applied_controls_native"], dtype=float)
        stored_model_sha = str(archive["reference_model_sha256"].item())
        stored_doc_sha = str(archive["doc_reference_source_sha256"].item())

    if times.ndim != 1 or not np.all(np.diff(times) > 0):
        raise ValueError(f"MPC control times must be strictly increasing: {source}")
    if reference.shape != times.shape or target_doc.shape != times.shape:
        raise ValueError(f"MPC trajectory arrays have inconsistent shapes: {source}")
    if controls.ndim != 3 or controls.shape[0] != times.size:
        raise ValueError(f"MPC applied controls do not match control times: {source}")
    if tuple(controls.shape[1:]) != tuple(target_region.shape):
        raise ValueError(f"MPC control grid does not match diagnostic target: {source}")
    if not all(np.isfinite(item).all() for item in (times, reference, target_doc, controls)):
        raise ValueError(f"MPC archive contains NaN or Inf: {source}")

    target_count = int(target_region.sum())
    target_control_mean = (
        controls[:, target_region].sum(axis=1) / target_count
    )
    summary = {
        "loaded": True,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "stored_reference_model_sha256": stored_model_sha,
        "stored_doc_reference_sha256": stored_doc_sha,
        "mean_target_region_control": float(np.mean(target_control_mean)),
        "maximum_target_region_control": float(np.max(target_control_mean)),
        "minimum_target_region_control": float(np.min(target_control_mean)),
        "fraction_control_times_target_mean_ge_0_95": float(
            np.mean(target_control_mean >= 0.95)
        ),
        "control_time_count": int(times.size),
    }
    return MPCComparison(
        source_path=source,
        control_times_s=times,
        target_doc=target_doc,
        reference_doc=reference,
        target_control_mean=target_control_mean,
        summary=summary,
    )


def _align_mpc(
    comparison: MPCComparison, physics_times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    doc_times = comparison.control_times_s
    doc_values = comparison.target_doc
    if doc_times[0] > 0.0:
        doc_times = np.r_[0.0, doc_times]
        doc_values = np.r_[0.0, doc_values]
    aligned_doc = np.interp(
        physics_times_s,
        doc_times,
        doc_values,
        left=doc_values[0],
        right=doc_values[-1],
    )
    held_indices = np.searchsorted(
        comparison.control_times_s, physics_times_s, side="left"
    )
    held_indices = np.clip(held_indices, 0, comparison.target_control_mean.size - 1)
    aligned_control = comparison.target_control_mean[held_indices]
    return aligned_doc, aligned_control


def _mpc_gap_summary(
    comparison: MPCComparison,
    time_s: np.ndarray,
    full_power_target_doc: np.ndarray,
    reference_thresholds: dict[str, float | None],
) -> dict[str, Any]:
    full_at_controls = np.interp(
        comparison.control_times_s, time_s, full_power_target_doc
    )
    positive_gap = np.maximum(full_at_controls - comparison.target_doc, 0.0)
    t10, t99 = reference_thresholds["t10_s"], reference_thresholds["t99_s"]
    rise = (
        np.ones(comparison.control_times_s.shape, dtype=bool)
        if t10 is None or t99 is None
        else (comparison.control_times_s >= t10 - 1e-12)
        & (comparison.control_times_s <= t99 + 1e-12)
    )
    rise_gap = positive_gap[rise]
    rise_control = comparison.target_control_mean[rise]
    return {
        **comparison.summary,
        "control_times_s": comparison.control_times_s.tolist(),
        "target_region_control_mean_at_control_times": (
            comparison.target_control_mean.tolist()
        ),
        "full_power_target_doc_at_control_times": full_at_controls.tolist(),
        "positive_full_power_minus_mpc_doc": positive_gap.tolist(),
        "rise_region_rms_positive_full_power_minus_mpc_doc": (
            None
            if not rise_gap.size
            else float(np.sqrt(np.mean(rise_gap**2)))
        ),
        "maximum_positive_full_power_minus_mpc_doc": float(np.max(positive_gap)),
        "mean_positive_full_power_minus_mpc_doc": float(np.mean(positive_gap)),
        "rise_region_control_time_count": int(rise_control.size),
        "rise_region_mean_target_control": (
            None if not rise_control.size else float(np.mean(rise_control))
        ),
        "rise_region_maximum_target_control": (
            None if not rise_control.size else float(np.max(rise_control))
        ),
        "rise_region_minimum_target_control": (
            None if not rise_control.size else float(np.min(rise_control))
        ),
        "rise_region_fraction_target_mean_ge_0_95": (
            None
            if not rise_control.size
            else float(np.mean(rise_control >= 0.95))
        ),
    }


def _classify(
    feasibility: dict[str, Any], mpc_summary: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    forward_limited = bool(
        feasibility["maximum_deficit"] > 0.05
        or feasibility["duration_deficit_gt_0_05_s"] >= 0.5
    )
    controller_gap = False
    if mpc_summary is not None:
        gap_rms = mpc_summary[
            "rise_region_rms_positive_full_power_minus_mpc_doc"
        ]
        controller_gap = bool(
            gap_rms is not None
            and gap_rms > 0.02
            and mpc_summary["rise_region_mean_target_control"] is not None
            and mpc_summary["rise_region_mean_target_control"] < 0.95
        )
    if forward_limited and controller_gap:
        classification = "MIXED"
    elif forward_limited:
        classification = "FORWARD-PHYSICS-LIMITED"
    elif controller_gap:
        classification = "CONTROLLER-LIMITED"
    elif mpc_summary is not None:
        classification = "GOOD MATCH"
    else:
        classification = "UNRESOLVED WITHOUT MPC COMPARISON"
    return classification, {
        "material_forward_limit": forward_limited,
        "material_controller_gap": controller_gap,
        "rule": CLASSIFICATION_RULE,
    }


def _save_csv(
    path: Path,
    reference: np.ndarray,
    uniform: OpenLoopTrace,
    target: OpenLoopTrace,
    control_dt_s: float,
    mpc: MPCComparison | None,
) -> None:
    deficit = np.maximum(reference - target.doc_mean, 0.0)
    aligned_mpc: np.ndarray | None = None
    aligned_control: np.ndarray | None = None
    if mpc is not None:
        aligned_mpc, aligned_control = _align_mpc(mpc, target.time_s)
    fieldnames = [
        "time_s",
        "is_control_reporting_time",
        "reference_doc",
        "uniform_full_power_doc",
        "target_full_power_doc",
        "target_full_power_outside_doc",
        "feasibility_deficit",
        "reference_minus_target_full_power",
        "target_full_power_minus_reference",
        "uniform_full_power_minus_reference",
        "uniform_o2_mean",
        "uniform_tempo_mean",
        "uniform_dose_mean",
        "uniform_reaction_progress_mean",
        "o2_target_mean",
        "tempo_target_mean",
        "dose_target_mean",
        "reaction_progress_target_mean",
    ]
    if mpc is not None:
        fieldnames.extend(
            ["mpc_target_doc", "mpc_tracking_error", "target_region_control_mean"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_value in enumerate(target.time_s):
            report_ratio = time_value / control_dt_s
            row: dict[str, Any] = {
                "time_s": f"{time_value:.10g}",
                "is_control_reporting_time": int(
                    math.isclose(
                        report_ratio, round(report_ratio), rel_tol=0.0, abs_tol=1e-9
                    )
                ),
                "reference_doc": f"{reference[index]:.12g}",
                "uniform_full_power_doc": f"{uniform.doc_mean[index]:.12g}",
                "target_full_power_doc": f"{target.doc_mean[index]:.12g}",
                "target_full_power_outside_doc": f"{target.outside_doc_mean[index]:.12g}",
                "feasibility_deficit": f"{deficit[index]:.12g}",
                "reference_minus_target_full_power": f"{reference[index] - target.doc_mean[index]:.12g}",
                "target_full_power_minus_reference": f"{target.doc_mean[index] - reference[index]:.12g}",
                "uniform_full_power_minus_reference": f"{uniform.doc_mean[index] - reference[index]:.12g}",
                "uniform_o2_mean": f"{uniform.o2_mean[index]:.12g}",
                "uniform_tempo_mean": f"{uniform.tempo_mean[index]:.12g}",
                "uniform_dose_mean": f"{uniform.dose_mean[index]:.12g}",
                "uniform_reaction_progress_mean": f"{uniform.reaction_progress_mean[index]:.12g}",
                "o2_target_mean": f"{target.o2_mean[index]:.12g}",
                "tempo_target_mean": f"{target.tempo_mean[index]:.12g}",
                "dose_target_mean": f"{target.dose_mean[index]:.12g}",
                "reaction_progress_target_mean": f"{target.reaction_progress_mean[index]:.12g}",
            }
            if aligned_mpc is not None and aligned_control is not None:
                row.update(
                    {
                        "mpc_target_doc": f"{aligned_mpc[index]:.12g}",
                        "mpc_tracking_error": f"{aligned_mpc[index] - reference[index]:.12g}",
                        "target_region_control_mean": f"{aligned_control[index]:.12g}",
                    }
                )
            writer.writerow(row)


def _save_condition_plot(
    path: Path,
    reference_id: str,
    reference: np.ndarray,
    uniform: OpenLoopTrace,
    target: OpenLoopTrace,
    mpc: MPCComparison | None,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    ax, deficit_ax = axes
    ax.plot(target.time_s, reference, color="black", linewidth=2.8, label="legacy isotonic experimental reference")
    ax.plot(uniform.time_s, uniform.doc_mean, color="#1f77b4", linewidth=2.0, label="uniform full-field, u=1")
    ax.plot(target.time_s, target.doc_mean, color="#d62728", linewidth=2.0, label="L-shape full-power")
    if mpc is not None:
        ax.plot(
            np.r_[0.0, mpc.control_times_s],
            np.r_[0.0, mpc.target_doc],
            color="#2ca02c",
            marker="o",
            markersize=3,
            linewidth=1.6,
            label="closed-loop MPC",
        )
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-0.025, 1.04)
    ax.set_ylabel("Mean DoC")
    ax.set_title(f"Open-loop feasibility - {CONDITION_TITLES[reference_id]}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    signed_deficit = reference - target.doc_mean
    deficit_ax.axhline(0.0, color="0.2", linewidth=1.0)
    deficit_ax.fill_between(
        target.time_s,
        0.0,
        signed_deficit,
        where=signed_deficit > 0.0,
        interpolate=True,
        color="#d62728",
        alpha=0.25,
        label="positive: reference infeasible at full target power",
    )
    deficit_ax.plot(target.time_s, signed_deficit, color="#8c1d18", linewidth=1.8)
    deficit_ax.set_xlabel("Absolute process time (s)")
    deficit_ax.set_ylabel("Reference - full-power DoC")
    deficit_ax.grid(alpha=0.25)
    deficit_ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_plot(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.4), sharex=True)
    colors = {"30mW_0mM": "#1f77b4", "30mW_5mM": "#d62728"}
    for reference_id, item in results.items():
        time_s = item["target_trace"].time_s
        reference = item["reference_values"]
        target_doc = item["target_trace"].doc_mean
        color = colors[reference_id]
        label = CONDITION_TITLES[reference_id]
        axes[0].plot(time_s, reference, color=color, linewidth=2.7, label=f"reference: {label}")
        axes[0].plot(time_s, target_doc, color=color, linewidth=1.8, linestyle="--", label=f"L-shape full-power: {label}")
        axes[1].plot(time_s, reference - target_doc, color=color, linewidth=2.0, label=label)
    axes[0].axhline(1.0, color="0.6", linestyle=":", linewidth=1.0)
    axes[0].set_xlim(0.0, 20.0)
    axes[0].set_ylim(-0.025, 1.04)
    axes[0].set_ylabel("Mean DoC")
    axes[0].set_title("Open-loop temporal feasibility comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="0.2", linewidth=1.0)
    axes[1].set_xlabel("Absolute process time (s)")
    axes[1].set_ylabel("Reference - full-power DoC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_time(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.4f} s"


def _print_threshold_table(
    reference: dict[str, float | None],
    uniform: dict[str, float | None],
    target: dict[str, float | None],
) -> None:
    print(f"{'threshold':<10} {'reference':>14} {'uniform u=1':>14} {'L-shape u=1':>14} {'delta target':>14}")
    for label in reference:
        delta = (
            None
            if target[label] is None or reference[label] is None
            else target[label] - reference[label]
        )
        print(
            f"{label.removesuffix('_s'):<10} "
            f"{_format_time(reference[label]):>14} "
            f"{_format_time(uniform[label]):>14} "
            f"{_format_time(target[label]):>14} "
            f"{_format_time(delta):>14}"
        )


def _print_condition_summary(reference_id: str, item: dict[str, Any]) -> None:
    print(f"\n=== {CONDITION_TITLES[reference_id]} ===")
    _print_threshold_table(
        item["reference_threshold_times_s"],
        item["uniform_full_power"]["threshold_times_s"],
        item["target_full_power"]["threshold_times_s"],
    )
    target_errors = item["target_full_power"]["tracking_error_metrics"]
    feasibility = item["target_full_power"]["feasibility"]
    print("Target-shaped full-power error")
    print(f"  full-duration RMSE: {target_errors['full_duration_rmse']:.6f}")
    print(f"  pre-saturation RMSE: {target_errors['pre_saturation_rmse']:.6f}")
    print(f"  rise-region RMSE/MAE: {target_errors['rise_region_rmse']:.6f} / {target_errors['rise_region_mae']:.6f}")
    print(f"  maximum feasibility deficit: {feasibility['maximum_deficit']:.6f} at t={feasibility['time_of_maximum_deficit_s']:.3f} s")
    print(f"  deficit >0.01 / >0.05 duration: {feasibility['duration_deficit_gt_0_01_s']:.3f} / {feasibility['duration_deficit_gt_0_05_s']:.3f} s")
    mpc = item["mpc_comparison"]
    if mpc is None:
        print("  closed-loop MPC comparison: not loaded")
    else:
        print(f"  MPC mean/max target control: {mpc['mean_target_region_control']:.4f} / {mpc['maximum_target_region_control']:.4f}")
        print(f"  MPC fraction target mean u>=0.95: {mpc['fraction_control_times_target_mean_ge_0_95']:.4f}")
        print(f"  MPC rise mean/min target control: {mpc['rise_region_mean_target_control']:.4f} / {mpc['rise_region_minimum_target_control']:.4f}")
        print(f"  MPC rise fraction target mean u>=0.95: {mpc['rise_region_fraction_target_mean_ge_0_95']:.4f}")
        print(f"  rise RMS positive full-power-MPC gap: {mpc['rise_region_rms_positive_full_power_minus_mpc_doc']:.6f}")
    print(f"Interpretation: {item['classification']}")
    evidence = item["classification_evidence"]
    print(
        "  evidence flags: "
        f"forward_limit={evidence['material_forward_limit']} "
        f"controller_gap={evidence['material_controller_gap']}"
    )


def _parse_mpc_overrides(entries: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--mpc-output must use REFERENCE_ID=PATH")
        reference_id, raw_path = entry.split("=", 1)
        if reference_id not in CONDITION_IDS:
            raise ValueError(
                f"unknown --mpc-output reference ID {reference_id!r}; expected {CONDITION_IDS}"
            )
        if not raw_path:
            raise ValueError("--mpc-output path may not be empty")
        result[reference_id] = Path(raw_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed full-power open-loop feasibility diagnostics for both "
            "matched 30 mW experimental conditions. No optimization is run."
        )
    )
    parser.add_argument(
        "--target", type=Path, default=REPOSITORY_DIR / "GEO" / "Lshape.png"
    )
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--total-time", type=float, default=20.0)
    parser.add_argument("--control-dt", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mpc-output",
        action="append",
        default=[],
        metavar="REFERENCE_ID=PATH",
        help=(
            "Optional existing MPC directory or NPZ. May be repeated. By default "
            "the two standard tracking_Lshape_* directories are auto-detected."
        ),
    )
    parser.add_argument(
        "--no-auto-mpc",
        action="store_true",
        help="Do not auto-load the two standard existing MPC output directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.target_threshold < 1.0:
        raise ValueError("--target-threshold must lie strictly between 0 and 1")
    if not math.isfinite(args.total_time) or args.total_time <= 0:
        raise ValueError("--total-time must be finite and positive")
    if not math.isfinite(args.control_dt) or args.control_dt <= 0:
        raise ValueError("--control-dt must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )

    target_path = resolve_target_path(args.target)
    target_native = load_normalized_target(target_path)
    require_native_target(target_native, target_path)
    target_region = target_native >= args.target_threshold
    if not bool(target_region.any()) or bool(target_region.all()):
        raise ValueError("binary target must contain both target and outside pixels")
    target_sha = _sha256(target_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_mpc = _parse_mpc_overrides(args.mpc_output)
    mpc_paths: dict[str, Path] = {}
    if not args.no_auto_mpc:
        mpc_paths.update(
            {
                reference_id: path
                for reference_id, path in DEFAULT_MPC_OUTPUTS.items()
                if path.exists()
            }
        )
    mpc_paths.update(explicit_mpc)

    print("Open-loop feasibility diagnostic (NO optimization)")
    print(f"branch/repository must be verified externally: {REPOSITORY_DIR}")
    print(f"device={device} target={target_path} shape={tuple(target_native.shape)}")
    print(f"total_time={args.total_time:g} s reporting_interval={args.control_dt:g} s")
    print(f"DoC history mode={DOC_HISTORY_MODE}")
    print("fixed controls: uniform u=1; L-shape u=1 inside binary target/u=0 outside")

    runtime_results: dict[str, dict[str, Any]] = {}
    summary_conditions: dict[str, Any] = {}
    reference_model_hashes: set[str] = set()
    reference_structure_hashes: set[str] = set()
    doc_artifact_hashes: set[str] = set()

    for reference_id in CONDITION_IDS:
        print(f"\nLoading matched condition {reference_id}")
        curve: DoCReferenceCurve = load_doc_reference(
            reference_id, LEGACY_REFERENCE_PATH, curve_model="isotonic"
        )
        config = load_reference_config_for_condition(reference_id)
        params = AIEParameters.from_reference(config)
        physics_match = assess_reference_physics_match(curve, params)
        if not bool(physics_match["matched"]):
            raise RuntimeError(
                f"tracking/physics mismatch for {reference_id}: "
                f"{physics_match['missing_for_physical_match']}"
            )
        if not math.isclose(config.dt, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"authoritative dt changed from expected diagnostic sampling: {config.dt:g} s"
            )
        control_ratio = args.control_dt / config.dt
        if not math.isclose(
            control_ratio, round(control_ratio), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("--control-dt must be an integer number of physics steps")
        if tuple(target_native.shape) != tuple(config.native_shape):
            raise ValueError(
                f"target shape {tuple(target_native.shape)} does not match authoritative "
                f"native shape {config.native_shape}"
            )

        model_for_shape = AIEModel(params=params, device=device)
        control_shape = model_for_shape.control_shape_for(config.native_shape)
        if control_shape != tuple(target_native.shape):
            raise ValueError(
                "diagnostic currently requires native projector and state grids to match"
            )
        uniform_control = torch.ones(control_shape, dtype=torch.float32)
        target_control = target_region.to(dtype=torch.float32)
        del model_for_shape

        print(
            f"  matched physics: I={params.intensity_mw_cm2:g} mW/cm^2, "
            f"TEMPO={params.tempo_concentration_mM:g} mM, "
            f"B source={params.b_condition_label}"
        )
        print(
            f"  hashes: physics={params.reference_model_sha256[:12]}... "
            f"structure={params.reference_structure_sha256[:12]}... "
            f"DoC reference={curve.source_sha256[:12]}..."
        )
        print("  simulating uniform full-field fixed u=1...")
        uniform = _simulate_fixed_control(
            params=params,
            state_shape=config.native_shape,
            projector_control=uniform_control,
            total_time_s=args.total_time,
            device=device,
            seed=args.seed,
            control_definition="uniform_full_field_u_equals_1_for_entire_run",
        )
        print("  simulating L-shape target fixed u=1 inside/u=0 outside...")
        target = _simulate_fixed_control(
            params=params,
            state_shape=config.native_shape,
            projector_control=target_control,
            total_time_s=args.total_time,
            device=device,
            seed=args.seed,
            control_definition=(
                "binary_Lshape_u_equals_1_inside_target_u_equals_0_outside_for_entire_run"
            ),
            target_region=target_region,
        )
        if not np.array_equal(uniform.time_s, target.time_s):
            raise AssertionError("open-loop experiments produced different time grids")
        reference_values = np.asarray(curve.at(target.time_s), dtype=float)
        if not np.isfinite(reference_values).all():
            raise RuntimeError("absolute-time reference interpolation is nonfinite")
        if float(np.min(np.diff(reference_values))) < -1e-12:
            raise RuntimeError("experimental reference decreased during diagnostic")

        reference_thresholds = _threshold_times(target.time_s, reference_values)
        uniform_thresholds = _threshold_times(uniform.time_s, uniform.doc_mean)
        target_thresholds = _threshold_times(target.time_s, target.doc_mean)
        uniform_errors = _error_metrics(
            uniform.time_s,
            uniform.doc_mean,
            reference_values,
            curve.saturation_time_s,
            reference_thresholds,
        )
        target_errors = _error_metrics(
            target.time_s,
            target.doc_mean,
            reference_values,
            curve.saturation_time_s,
            reference_thresholds,
        )
        feasibility = _feasibility_summary(
            target.time_s, reference_values, target.doc_mean
        )

        mpc_comparison: MPCComparison | None = None
        mpc_summary: dict[str, Any] | None = None
        if reference_id in mpc_paths:
            mpc_comparison = _load_mpc_comparison(
                mpc_paths[reference_id], reference_id, target_region.numpy()
            )
            if mpc_comparison.summary["stored_reference_model_sha256"] != params.reference_model_sha256:
                raise RuntimeError(
                    f"MPC archive physics SHA does not match current source for {reference_id}"
                )
            if mpc_comparison.summary["stored_doc_reference_sha256"] != curve.source_sha256:
                raise RuntimeError(
                    f"MPC archive DoC reference SHA does not match current artifact for {reference_id}"
                )
            mpc_summary = _mpc_gap_summary(
                mpc_comparison,
                target.time_s,
                target.doc_mean,
                reference_thresholds,
            )

        classification, classification_evidence = _classify(
            feasibility, mpc_summary
        )
        condition_summary = {
            "reference_id": reference_id,
            "condition": curve.metadata["condition"],
            "reference_provenance": curve.provenance_metadata(),
            "forward_physics_match": physics_match,
            "forward_physics_parameters": params.provenance_metadata(),
            "reference_production_saturation_time_s": curve.saturation_time_s,
            "reference_threshold_times_s": reference_thresholds,
            "uniform_full_power": {
                "control_definition": uniform.control_definition,
                "control_min": uniform.control_min,
                "control_max": uniform.control_max,
                "threshold_times_s": uniform_thresholds,
                "threshold_deltas_vs_reference_s": _threshold_deltas(
                    uniform_thresholds, reference_thresholds
                ),
                "tracking_error_metrics": uniform_errors,
                "validation": uniform.validation,
            },
            "target_full_power": {
                "control_definition": target.control_definition,
                "control_min": target.control_min,
                "control_max": target.control_max,
                "control_target_mean": target.control_target_mean,
                "control_outside_mean": target.control_outside_mean,
                "threshold_times_s": target_thresholds,
                "threshold_deltas_vs_reference_s": _threshold_deltas(
                    target_thresholds, reference_thresholds
                ),
                "tracking_error_metrics": target_errors,
                "feasibility": feasibility,
                "validation": target.validation,
            },
            "mpc_comparison": mpc_summary,
            "classification": classification,
            "classification_evidence": classification_evidence,
        }

        csv_path = output_dir / f"open_loop_feasibility_{reference_id}.csv"
        plot_path = output_dir / f"open_loop_feasibility_{reference_id}.png"
        _save_csv(
            csv_path,
            reference_values,
            uniform,
            target,
            args.control_dt,
            mpc_comparison,
        )
        _save_condition_plot(
            plot_path,
            reference_id,
            reference_values,
            uniform,
            target,
            mpc_comparison,
        )
        condition_summary["output_csv"] = str(csv_path)
        condition_summary["output_plot"] = str(plot_path)
        summary_conditions[reference_id] = condition_summary
        runtime_results[reference_id] = {
            "reference_values": reference_values,
            "uniform_trace": uniform,
            "target_trace": target,
        }
        reference_model_hashes.add(params.reference_model_sha256)
        reference_structure_hashes.add(params.reference_structure_sha256)
        doc_artifact_hashes.add(curve.source_sha256)

    combined_plot = output_dir / "open_loop_feasibility_comparison.png"
    _save_comparison_plot(combined_plot, runtime_results)
    if len(reference_model_hashes) != 1 or len(reference_structure_hashes) != 1:
        raise RuntimeError("matched conditions did not resolve the same authoritative source")
    if len(doc_artifact_hashes) != 1:
        raise RuntimeError("matched conditions did not load the same reference artifact")
    summary_path = output_dir / "open_loop_feasibility_summary.json"
    summary_document = {
        "schema_version": 1,
        "diagnostic_id": "fixed_full_power_open_loop_feasibility_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "optimization": {
            "executed": False,
            "optimizer_instances_created": 0,
            "description": "Both forward runs use one prepared control held fixed for every physics step.",
        },
        "repository_dir": str(REPOSITORY_DIR),
        "diagnostic_script": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
        "protected_reference_model_sha256": next(iter(reference_model_hashes)),
        "reference_structure_sha256": next(iter(reference_structure_hashes)),
        "doc_reference_artifact_sha256": next(iter(doc_artifact_hashes)),
        "doc_history_mode": DOC_HISTORY_MODE,
        "doc_history_description": DOC_HISTORY_DESCRIPTION,
        "absolute_reference_time": True,
        "reference_time_restart_count": 0,
        "target": {
            "path": str(target_path),
            "sha256": target_sha,
            "shape": list(target_native.shape),
            "binary_threshold": args.target_threshold,
            "target_pixels": int(target_region.sum()),
            "outside_pixels": int((~target_region).sum()),
        },
        "timing": {
            "total_time_s": args.total_time,
            "physics_dt_s": summary_conditions[CONDITION_IDS[0]][
                "forward_physics_parameters"
            ]["dt"],
            "control_reporting_interval_s": args.control_dt,
            "physics_sample_count_including_t0": int(
                runtime_results[CONDITION_IDS[0]]["target_trace"].time_s.size
            ),
            "time_range_s": [0.0, args.total_time],
        },
        "classification_rule": CLASSIFICATION_RULE,
        "conditions": summary_conditions,
        "combined_plot": str(combined_plot),
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary_document), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for reference_id in CONDITION_IDS:
        _print_condition_summary(reference_id, summary_conditions[reference_id])
    print("\n=== Validation ===")
    print("optimization executed: False")
    print("uniform control exact min/max: 1.0/1.0")
    print("target control exact inside/outside mean: 1.0/0.0")
    print("reference queried once on absolute 0-20 s physics grid; restart count: 0")
    print(
        "all open-loop finiteness, DoC-bound, and region-mean "
        "reaction-progress checks: PASS"
    )
    print(
        "pointwise reaction-progress decreases, if any, are recorded separately "
        "in validation metadata"
    )
    print(f"summary JSON: {summary_path}")
    print(f"combined plot: {combined_plot}")


if __name__ == "__main__":
    main()
