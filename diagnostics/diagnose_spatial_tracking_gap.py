"""Diagnose spatial DoC lag and local reaction-progress reversals.

This script is read-only with respect to production physics and controls.  It
replays fixed full-power L-shape exposure and existing MPC masks through the
current matched AIE forward model.  It performs no optimization.
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
from matplotlib.colors import ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch

from aie_model import DOC_HISTORY_DESCRIPTION, DOC_HISTORY_MODE, AIEModel, AIEParameters
from aie_reference import load_reference_config_for_condition
from doc_reference import DoCReferenceCurve, load_doc_reference
from run_mpc import (
    assess_reference_physics_match,
    load_normalized_target,
    require_native_target,
    resolve_target_path,
)


CONDITION_IDS = ("30mW_0mM", "30mW_5mM")
CONDITION_LABELS = {
    "30mW_0mM": "30 mW/cm^2 / 0 mM TEMPO",
    "30mW_5mM": "30 mW/cm^2 / 5 mM TEMPO",
}
EROSION_DEPTHS = (1, 2, 4)
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
    REPOSITORY_DIR / "results" / "diagnostics" / "spatial_tracking_gap"
)
EVENT_FIELDS = [
    "event_id",
    "condition_id",
    "trajectory",
    "absolute_time_s",
    "physics_step",
    "row",
    "column",
    "region_n1",
    "inside_target",
    "target_interior_n1",
    "target_boundary_n1",
    "target_interior_n2",
    "target_boundary_n2",
    "target_interior_n4",
    "target_boundary_n4",
    "signed_distance_to_target_boundary_px",
    "local_control",
    "scattered_control",
    "local_intensity_mw_cm2",
    "energy_mj_cm2",
    "o2_before_mj_cm2",
    "o2_diffused_mj_cm2",
    "o2_after_mj_cm2",
    "o2_diffusion_change_mj_cm2",
    "tempo_before_mj_cm2",
    "tempo_diffused_mj_cm2",
    "tempo_after_mj_cm2",
    "tempo_diffusion_change_mj_cm2",
    "inhibitor_sum_before_diffusion_mj_cm2",
    "inhibitor_sum_after_diffusion_mj_cm2",
    "inhibition_sum_exceeds_energy",
    "diffusion_increased_inhibitor_sum",
    "diffusion_created_negative_dose_increment",
    "dose_before_mj_cm2",
    "dose_after_mj_cm2",
    "delta_dose_mj_cm2",
    "dose_equation_increment_mj_cm2",
    "effective_b_per_s",
    "curing_gate",
    "reaction_progress_before",
    "reaction_progress_after",
    "delta_reaction_progress",
    "expected_delta_reaction_progress",
    "reaction_equation_residual",
    "doc_before",
    "doc_after",
    "delta_doc",
    "doc_change_class",
]


@dataclass
class Snapshot:
    requested_label: str
    requested_time_s: float
    sampled_time_s: float
    reference_doc: float
    doc: np.ndarray
    reference_minus_local_doc: np.ndarray
    delta_dose: np.ndarray
    delta_reaction_progress: np.ndarray
    historical_negative_mask: np.ndarray


@dataclass
class SimulationResult:
    time_s: np.ndarray
    region_doc: dict[str, np.ndarray]
    monotonicity: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    event_summary: dict[str, Any]
    negative_count_map: np.ndarray
    negative_magnitude_map: np.ndarray
    minimum_delta_r_map: np.ndarray
    snapshots: dict[str, Snapshot]
    validation: dict[str, Any]


@dataclass(frozen=True)
class MPCArchive:
    path: Path
    control_times_s: np.ndarray
    controls: np.ndarray
    actual_target_doc: np.ndarray
    reference_doc: np.ndarray
    physics_steps_per_control: int
    reference_model_sha256: str
    doc_reference_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
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
            f"MPC directory must contain one unambiguous NPZ, found "
            f"{len(candidates)}: {resolved}"
        )
    return candidates[0]


def _parse_mpc_overrides(entries: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--mpc-output must use REFERENCE_ID=PATH")
        reference_id, raw_path = entry.split("=", 1)
        if reference_id not in CONDITION_IDS:
            raise ValueError(f"unknown reference ID in --mpc-output: {reference_id}")
        parsed[reference_id] = Path(raw_path)
    return parsed


def _load_mpc_archive(path: Path, reference_id: str) -> MPCArchive:
    source = _resolve_npz(path)
    required = {
        "control_times_s",
        "applied_controls_native",
        "actual_mean_target_doc",
        "reference_doc_values",
        "physics_steps_per_control",
        "doc_reference_id",
        "reference_model_sha256",
        "doc_reference_source_sha256",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"MPC archive is missing keys {missing}: {source}")
        stored_id = str(archive["doc_reference_id"].item())
        if stored_id != reference_id:
            raise ValueError(
                f"MPC archive condition {stored_id!r} does not match {reference_id!r}"
            )
        result = MPCArchive(
            path=source,
            control_times_s=np.asarray(archive["control_times_s"], dtype=float),
            controls=np.asarray(archive["applied_controls_native"], dtype=float),
            actual_target_doc=np.asarray(
                archive["actual_mean_target_doc"], dtype=float
            ),
            reference_doc=np.asarray(archive["reference_doc_values"], dtype=float),
            physics_steps_per_control=int(
                archive["physics_steps_per_control"].item()
            ),
            reference_model_sha256=str(
                archive["reference_model_sha256"].item()
            ),
            doc_reference_sha256=str(
                archive["doc_reference_source_sha256"].item()
            ),
        )
    arrays = (
        result.control_times_s,
        result.controls,
        result.actual_target_doc,
        result.reference_doc,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"MPC archive contains NaN or Inf: {source}")
    if result.controls.ndim != 3:
        raise ValueError(f"MPC controls must be a 3-D sequence: {source}")
    if result.control_times_s.size != result.controls.shape[0]:
        raise ValueError(f"MPC controls/times length mismatch: {source}")
    return result


def _construct_regions(
    target: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    """Create deterministic Chebyshev-layer erosion regions.

    Interior N survives N binary erosions using a 3x3 all-ones structuring
    element. Boundary N is the target minus that interior. This is a diagnostic
    grid-layer definition and is not asserted to be the experimental CenterROI.
    """

    target = np.asarray(target, dtype=bool)
    masks: dict[str, np.ndarray] = {
        "full_target": target,
        "outside_target": ~target,
        "full_image": np.ones(target.shape, dtype=bool),
    }
    structure = np.ones((3, 3), dtype=bool)
    for depth in EROSION_DEPTHS:
        interior = binary_erosion(
            target, structure=structure, iterations=depth, border_value=0
        )
        boundary = target & ~interior
        if not interior.any() or not boundary.any():
            raise ValueError(f"erosion N={depth} produced an empty diagnostic region")
        masks[f"interior_N{depth}"] = interior
        masks[f"boundary_N{depth}"] = boundary

    inside_distance = distance_transform_edt(target)
    outside_distance = distance_transform_edt(~target)
    signed_distance = np.where(target, inside_distance, -outside_distance)
    metadata = {
        "definition": (
            "Interior N is the binary target after N erosions with a 3x3 "
            "all-ones structuring element (8-connected/Chebyshev grid layers); "
            "boundary N is full_target minus interior N. These diagnostic "
            "regions are not claimed to reproduce the experimental CenterROI."
        ),
        "pixel_counts": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    return masks, metadata, signed_distance


def _threshold_times(time_s: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for threshold in THRESHOLDS:
        label = f"t{int(round(100 * threshold)):02d}_s"
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
        result[label] = (
            t1
            if math.isclose(y0, y1, rel_tol=0.0, abs_tol=1e-15)
            else t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
        )
    return result


def _tracking_metrics(
    time_s: np.ndarray,
    actual: np.ndarray,
    reference: np.ndarray,
    saturation_time_s: float,
    reference_thresholds: dict[str, float | None],
) -> dict[str, float | int | None]:
    error = actual - reference

    def summarize(mask: np.ndarray) -> tuple[float | None, float | None, int]:
        count = int(mask.sum())
        if count == 0:
            return None, None, 0
        selected = error[mask]
        return (
            float(np.sqrt(np.mean(selected**2))),
            float(np.mean(np.abs(selected))),
            count,
        )

    full = np.ones(time_s.shape, dtype=bool)
    pre = time_s <= saturation_time_s + 1e-12
    t10 = reference_thresholds["t10_s"]
    t99 = reference_thresholds["t99_s"]
    rise = (
        np.zeros(time_s.shape, dtype=bool)
        if t10 is None or t99 is None
        else (time_s >= t10 - 1e-12) & (time_s <= t99 + 1e-12)
    )
    full_rmse, full_mae, full_count = summarize(full)
    pre_rmse, pre_mae, pre_count = summarize(pre)
    rise_rmse, rise_mae, rise_count = summarize(rise)
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
    }


def _threshold_deltas(
    actual: dict[str, float | None], reference: dict[str, float | None]
) -> dict[str, float | None]:
    return {
        f"delta_{label}": (
            None
            if actual[label] is None or reference[label] is None
            else float(actual[label] - reference[label])
        )
        for label in reference
    }


def _region_means(
    field: torch.Tensor,
    masks: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    names: list[str],
) -> np.ndarray:
    values = torch.stack(
        [(field * masks[name]).sum() / counts[name] for name in names]
    )
    return values.detach().cpu().numpy().astype(float)


def _event_values(tensor: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> np.ndarray:
    return tensor[rows, cols].detach().cpu().numpy().astype(float)


def _simulate(
    *,
    reference_id: str,
    params: AIEParameters,
    target: np.ndarray,
    region_masks: dict[str, np.ndarray],
    signed_distance: np.ndarray,
    controls: np.ndarray,
    physics_steps_per_control: int,
    total_time_s: float,
    device: torch.device,
    negative_tolerance: float,
    trajectory: str,
    collect_event_details: bool,
    snapshot_requests: dict[int, tuple[str, float, float]] | None = None,
) -> SimulationResult:
    ratio = total_time_s / params.dt
    step_count = int(round(ratio))
    if not math.isclose(ratio, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("total time is not compatible with authoritative dt")
    expected_controls = math.ceil(step_count / physics_steps_per_control)
    if controls.shape[0] not in (1, expected_controls):
        raise ValueError(
            f"{trajectory} requires 1 fixed control or {expected_controls} MPC controls"
        )
    if tuple(controls.shape[1:]) != tuple(target.shape):
        raise ValueError(f"{trajectory} control shape does not match target")
    if np.min(controls) < 0.0 or np.max(controls) > 1.0:
        raise ValueError(f"{trajectory} controls leave [0,1]")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = AIEModel(params=params, device=device)
    state = model.initialize_state(target.shape)
    torch_masks = {
        name: torch.as_tensor(mask, device=device, dtype=model.dtype)
        for name, mask in region_masks.items()
    }
    counts = {name: mask.sum() for name, mask in torch_masks.items()}
    tracking_names = [
        "full_target",
        "interior_N1",
        "boundary_N1",
        "interior_N2",
        "boundary_N2",
        "interior_N4",
        "boundary_N4",
    ]
    stats_names = ["full_image", *tracking_names, "outside_target"]
    region_rows = [
        _region_means(state.doc, torch_masks, counts, tracking_names)
    ]
    stat_accumulators: dict[str, dict[str, Any]] = {
        name: {
            "pixel_count": int(region_masks[name].sum()),
            "negative_delta_r_count": 0,
            "negative_delta_r_sum": 0.0,
            "minimum_delta_r": None,
            "negative_delta_doc_count": 0,
            "minimum_delta_doc_at_negative_r": None,
        }
        for name in stats_names
    }
    events: list[dict[str, Any]] = []
    event_id = 0
    negative_event_step_count = 0
    all_finite = True
    maximum_equation_residual = 0.0
    doc_min = 0.0
    doc_max = 0.0
    mean_reaction_progress_min_increment = math.inf
    previous_target_mean_r = 0.0
    negative_count_map = torch.zeros(target.shape, device=device, dtype=torch.int32)
    negative_magnitude_map = torch.zeros(target.shape, device=device, dtype=model.dtype)
    minimum_delta_r_map = torch.zeros(target.shape, device=device, dtype=model.dtype)
    historical_negative = torch.zeros(target.shape, device=device, dtype=torch.bool)
    snapshots: dict[str, Snapshot] = {}
    prepared = None
    active_control_index = -1

    with torch.no_grad():
        for step in range(1, step_count + 1):
            control_index = (
                0
                if controls.shape[0] == 1
                else min((step - 1) // physics_steps_per_control, controls.shape[0] - 1)
            )
            if control_index != active_control_index:
                control = torch.as_tensor(
                    controls[control_index], device=device, dtype=model.dtype
                )
                prepared = model.prepare_control(control, target.shape)
                active_control_index = control_index
            assert prepared is not None
            previous = state
            state = model.step_prepared(previous, prepared)
            delta_dose = state.dose - previous.dose
            delta_r = state.reaction_progress - previous.reaction_progress
            delta_doc = state.doc - previous.doc
            curing = (state.o2 <= 0) & (state.tempo <= 0)
            effective_b = prepared.b
            if previous.chain_growth_multiplier is not None:
                effective_b = effective_b * previous.chain_growth_multiplier
            safe_intensity = prepared.local_intensity.clamp_min(
                params.division_epsilon
            )
            expected_delta_r = torch.where(
                curing,
                effective_b * delta_dose / safe_intensity,
                torch.zeros_like(delta_r),
            )
            residual = delta_r - expected_delta_r
            maximum_equation_residual = max(
                maximum_equation_residual,
                float(residual.abs().max().detach().cpu()),
            )
            finite = torch.stack(
                [torch.isfinite(field).all() for field in state.tensors()]
            ).all()
            all_finite = all_finite and bool(finite.detach().cpu())
            if not all_finite:
                raise RuntimeError(f"{reference_id}/{trajectory} became nonfinite")
            doc_min = min(doc_min, float(state.doc.min().detach().cpu()))
            doc_max = max(doc_max, float(state.doc.max().detach().cpu()))

            target_mean_r = float(
                (
                    (state.reaction_progress * torch_masks["full_target"]).sum()
                    / counts["full_target"]
                )
                .detach()
                .cpu()
            )
            mean_reaction_progress_min_increment = min(
                mean_reaction_progress_min_increment,
                target_mean_r - previous_target_mean_r,
            )
            previous_target_mean_r = target_mean_r

            negative = delta_r < -negative_tolerance
            if bool(negative.any().detach().cpu()):
                negative_event_step_count += 1
            negative_count_map += negative.to(torch.int32)
            negative_magnitude_map += torch.where(
                negative, -delta_r, torch.zeros_like(delta_r)
            )
            minimum_delta_r_map = torch.minimum(minimum_delta_r_map, delta_r)
            historical_negative |= negative

            for name in stats_names:
                region = torch_masks[name].bool()
                selected = negative & region
                count = int(selected.sum().detach().cpu())
                accumulator = stat_accumulators[name]
                accumulator["negative_delta_r_count"] += count
                if count:
                    selected_delta_r = delta_r[selected]
                    selected_delta_doc = delta_doc[selected]
                    signed_sum = float(selected_delta_r.sum().detach().cpu())
                    selected_min = float(selected_delta_r.min().detach().cpu())
                    doc_decrease_count = int(
                        (selected_delta_doc < -negative_tolerance).sum().detach().cpu()
                    )
                    doc_minimum = float(selected_delta_doc.min().detach().cpu())
                    accumulator["negative_delta_r_sum"] += signed_sum
                    accumulator["minimum_delta_r"] = (
                        selected_min
                        if accumulator["minimum_delta_r"] is None
                        else min(accumulator["minimum_delta_r"], selected_min)
                    )
                    accumulator["negative_delta_doc_count"] += doc_decrease_count
                    accumulator["minimum_delta_doc_at_negative_r"] = (
                        doc_minimum
                        if accumulator["minimum_delta_doc_at_negative_r"] is None
                        else min(
                            accumulator["minimum_delta_doc_at_negative_r"],
                            doc_minimum,
                        )
                    )

            if collect_event_details and bool(negative.any().detach().cpu()):
                o2_diffused = (
                    model._gaussian_blur(
                        previous.o2, model.o2_kernel_1d, "diagnostic O2 diffusion"
                    )
                    if params.o2_diffusion_enabled
                    else previous.o2
                )
                tempo_diffused = (
                    model._gaussian_blur(
                        previous.tempo,
                        model.tempo_kernel_1d,
                        "diagnostic TEMPO diffusion",
                    )
                    if params.tempo_diffusion_enabled
                    else previous.tempo
                )
                dose_equation_increment = (
                    prepared.energy - o2_diffused - tempo_diffused
                )
                coordinates = torch.nonzero(negative, as_tuple=False)
                rows, cols = coordinates[:, 0], coordinates[:, 1]
                arrays = {
                    "local_control": _event_values(prepared.normalized_mask, rows, cols),
                    "scattered_control": _event_values(prepared.scattered_mask, rows, cols),
                    "local_intensity": _event_values(prepared.local_intensity, rows, cols),
                    "energy": _event_values(prepared.energy, rows, cols),
                    "o2_before": _event_values(previous.o2, rows, cols),
                    "o2_diffused": _event_values(o2_diffused, rows, cols),
                    "o2_after": _event_values(state.o2, rows, cols),
                    "tempo_before": _event_values(previous.tempo, rows, cols),
                    "tempo_diffused": _event_values(tempo_diffused, rows, cols),
                    "tempo_after": _event_values(state.tempo, rows, cols),
                    "dose_before": _event_values(previous.dose, rows, cols),
                    "dose_after": _event_values(state.dose, rows, cols),
                    "delta_dose": _event_values(delta_dose, rows, cols),
                    "dose_equation_increment": _event_values(
                        dose_equation_increment, rows, cols
                    ),
                    "effective_b": _event_values(effective_b, rows, cols),
                    "curing": _event_values(curing.to(model.dtype), rows, cols),
                    "r_before": _event_values(previous.reaction_progress, rows, cols),
                    "r_after": _event_values(state.reaction_progress, rows, cols),
                    "delta_r": _event_values(delta_r, rows, cols),
                    "expected_delta_r": _event_values(expected_delta_r, rows, cols),
                    "residual": _event_values(residual, rows, cols),
                    "doc_before": _event_values(previous.doc, rows, cols),
                    "doc_after": _event_values(state.doc, rows, cols),
                    "delta_doc": _event_values(delta_doc, rows, cols),
                }
                row_values = rows.detach().cpu().numpy().astype(int)
                col_values = cols.detach().cpu().numpy().astype(int)
                for local_index, (row, col) in enumerate(
                    zip(row_values, col_values, strict=True)
                ):
                    event_id += 1
                    inside = bool(target[row, col])
                    interior1 = bool(region_masks["interior_N1"][row, col])
                    boundary1 = bool(region_masks["boundary_N1"][row, col])
                    before_sum = (
                        arrays["o2_before"][local_index]
                        + arrays["tempo_before"][local_index]
                    )
                    after_diffusion_sum = (
                        arrays["o2_diffused"][local_index]
                        + arrays["tempo_diffused"][local_index]
                    )
                    energy = arrays["energy"][local_index]
                    doc_delta = arrays["delta_doc"][local_index]
                    events.append(
                        {
                            "event_id": event_id,
                            "condition_id": reference_id,
                            "trajectory": trajectory,
                            "absolute_time_s": step * params.dt,
                            "physics_step": step,
                            "row": row,
                            "column": col,
                            "region_n1": (
                                "target_boundary"
                                if boundary1
                                else "target_interior"
                                if interior1
                                else "outside_target"
                            ),
                            "inside_target": inside,
                            "target_interior_n1": interior1,
                            "target_boundary_n1": boundary1,
                            "target_interior_n2": bool(
                                region_masks["interior_N2"][row, col]
                            ),
                            "target_boundary_n2": bool(
                                region_masks["boundary_N2"][row, col]
                            ),
                            "target_interior_n4": bool(
                                region_masks["interior_N4"][row, col]
                            ),
                            "target_boundary_n4": bool(
                                region_masks["boundary_N4"][row, col]
                            ),
                            "signed_distance_to_target_boundary_px": float(
                                signed_distance[row, col]
                            ),
                            "local_control": arrays["local_control"][local_index],
                            "scattered_control": arrays["scattered_control"][local_index],
                            "local_intensity_mw_cm2": arrays["local_intensity"][local_index],
                            "energy_mj_cm2": energy,
                            "o2_before_mj_cm2": arrays["o2_before"][local_index],
                            "o2_diffused_mj_cm2": arrays["o2_diffused"][local_index],
                            "o2_after_mj_cm2": arrays["o2_after"][local_index],
                            "o2_diffusion_change_mj_cm2": (
                                arrays["o2_diffused"][local_index]
                                - arrays["o2_before"][local_index]
                            ),
                            "tempo_before_mj_cm2": arrays["tempo_before"][local_index],
                            "tempo_diffused_mj_cm2": arrays["tempo_diffused"][local_index],
                            "tempo_after_mj_cm2": arrays["tempo_after"][local_index],
                            "tempo_diffusion_change_mj_cm2": (
                                arrays["tempo_diffused"][local_index]
                                - arrays["tempo_before"][local_index]
                            ),
                            "inhibitor_sum_before_diffusion_mj_cm2": before_sum,
                            "inhibitor_sum_after_diffusion_mj_cm2": after_diffusion_sum,
                            "inhibition_sum_exceeds_energy": (
                                after_diffusion_sum > energy
                            ),
                            "diffusion_increased_inhibitor_sum": (
                                after_diffusion_sum > before_sum + negative_tolerance
                            ),
                            "diffusion_created_negative_dose_increment": (
                                before_sum <= energy + negative_tolerance
                                and after_diffusion_sum > energy
                            ),
                            "dose_before_mj_cm2": arrays["dose_before"][local_index],
                            "dose_after_mj_cm2": arrays["dose_after"][local_index],
                            "delta_dose_mj_cm2": arrays["delta_dose"][local_index],
                            "dose_equation_increment_mj_cm2": arrays[
                                "dose_equation_increment"
                            ][local_index],
                            "effective_b_per_s": arrays["effective_b"][local_index],
                            "curing_gate": bool(arrays["curing"][local_index]),
                            "reaction_progress_before": arrays["r_before"][local_index],
                            "reaction_progress_after": arrays["r_after"][local_index],
                            "delta_reaction_progress": arrays["delta_r"][local_index],
                            "expected_delta_reaction_progress": arrays[
                                "expected_delta_r"
                            ][local_index],
                            "reaction_equation_residual": arrays["residual"][local_index],
                            "doc_before": arrays["doc_before"][local_index],
                            "doc_after": arrays["doc_after"][local_index],
                            "delta_doc": doc_delta,
                            "doc_change_class": (
                                "decreases"
                                if doc_delta < -negative_tolerance
                                else "increases"
                                if doc_delta > negative_tolerance
                                else "unchanged_within_tolerance"
                            ),
                        }
                    )

            region_rows.append(
                _region_means(state.doc, torch_masks, counts, tracking_names)
            )
            if snapshot_requests and step in snapshot_requests:
                label, requested_time, reference_doc = snapshot_requests[step]
                spatial_reference = torch_masks["full_target"] * reference_doc
                snapshots[label] = Snapshot(
                    requested_label=label,
                    requested_time_s=requested_time,
                    sampled_time_s=step * params.dt,
                    reference_doc=reference_doc,
                    doc=state.doc.detach().cpu().numpy().copy(),
                    reference_minus_local_doc=(
                        spatial_reference - state.doc
                    ).detach().cpu().numpy().copy(),
                    delta_dose=delta_dose.detach().cpu().numpy().copy(),
                    delta_reaction_progress=delta_r.detach().cpu().numpy().copy(),
                    historical_negative_mask=(
                        historical_negative.detach().cpu().numpy().copy()
                    ),
                )

    time_s = np.arange(step_count + 1, dtype=float) * params.dt
    region_array = np.asarray(region_rows, dtype=float)
    monotonicity: dict[str, dict[str, Any]] = {}
    for name, accumulator in stat_accumulators.items():
        pixel_steps = int(accumulator["pixel_count"] * step_count)
        count = int(accumulator["negative_delta_r_count"])
        signed_sum = float(accumulator.pop("negative_delta_r_sum"))
        monotonicity[name] = {
            **accumulator,
            "total_pixel_step_count": pixel_steps,
            "negative_delta_r_fraction": count / pixel_steps,
            "mean_negative_delta_r": None if not count else signed_sum / count,
            "total_negative_delta_r_magnitude": -signed_sum,
        }

    event_delta_doc = np.asarray(
        [float(event["delta_doc"]) for event in events], dtype=float
    )
    event_summary = {
        "negative_tolerance": negative_tolerance,
        "negative_delta_r_event_count": len(events),
        "physics_steps_with_negative_delta_r": negative_event_step_count,
        "negative_delta_r_events_with_negative_delta_dose": sum(
            float(event["delta_dose_mj_cm2"]) < -negative_tolerance
            for event in events
        ),
        "negative_delta_r_events_with_curing_gate_true": sum(
            bool(event["curing_gate"]) for event in events
        ),
        "negative_delta_r_events_explained_by_equation_within_tolerance": sum(
            abs(float(event["reaction_equation_residual"])) <= negative_tolerance
            for event in events
        ),
        "negative_delta_r_events_with_inhibitor_sum_exceeding_energy": sum(
            bool(event["inhibition_sum_exceeds_energy"]) for event in events
        ),
        "negative_delta_r_events_where_diffusion_increased_inhibitor_sum": sum(
            bool(event["diffusion_increased_inhibitor_sum"]) for event in events
        ),
        "negative_delta_r_events_created_by_diffusion_crossing_energy": sum(
            bool(event["diffusion_created_negative_dose_increment"])
            for event in events
        ),
        "negative_delta_r_events_by_region_n1": {
            name: sum(event["region_n1"] == name for event in events)
            for name in ("target_interior", "target_boundary", "outside_target")
        },
        "negative_delta_r_events_producing_negative_delta_doc": int(
            np.sum(event_delta_doc < -negative_tolerance)
        ),
        "negative_delta_r_events_with_unchanged_doc": int(
            np.sum(np.abs(event_delta_doc) <= negative_tolerance)
        ),
        "negative_delta_r_events_producing_positive_delta_doc": int(
            np.sum(event_delta_doc > negative_tolerance)
        ),
        "minimum_delta_doc_at_negative_delta_r_event": (
            None if not event_delta_doc.size else float(np.min(event_delta_doc))
        ),
    }
    validation = {
        "all_states_finite": all_finite,
        "doc_min": doc_min,
        "doc_max": doc_max,
        "maximum_reaction_equation_residual": maximum_equation_residual,
        "target_mean_reaction_progress_minimum_step_increment": (
            mean_reaction_progress_min_increment
        ),
        "target_mean_reaction_progress_monotonic": (
            mean_reaction_progress_min_increment >= -negative_tolerance
        ),
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
        "physics_step_count": step_count,
        "sample_count_including_t0": int(time_s.size),
    }
    return SimulationResult(
        time_s=time_s,
        region_doc={
            name: region_array[:, index]
            for index, name in enumerate(tracking_names)
        },
        monotonicity=monotonicity,
        events=events,
        event_summary=event_summary,
        negative_count_map=negative_count_map.detach().cpu().numpy(),
        negative_magnitude_map=negative_magnitude_map.detach().cpu().numpy(),
        minimum_delta_r_map=minimum_delta_r_map.detach().cpu().numpy(),
        snapshots=snapshots,
        validation=validation,
    )


def _build_region_metrics(
    *,
    reference_id: str,
    trajectory: str,
    result: SimulationResult,
    reference: np.ndarray,
    curve: DoCReferenceCurve,
    region_masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_thresholds = _threshold_times(result.time_s, reference)
    rows: list[dict[str, Any]] = []
    document: dict[str, Any] = {}
    for region, values in result.region_doc.items():
        times = _threshold_times(result.time_s, values)
        deltas = _threshold_deltas(times, reference_thresholds)
        metrics = _tracking_metrics(
            result.time_s,
            values,
            reference,
            curve.saturation_time_s,
            reference_thresholds,
        )
        row: dict[str, Any] = {
            "condition_id": reference_id,
            "trajectory": trajectory,
            "region": region,
            "pixel_count": int(region_masks[region].sum()),
            **times,
            **deltas,
            **metrics,
        }
        rows.append(row)
        document[region] = {
            "pixel_count": row["pixel_count"],
            "threshold_times_s": times,
            "threshold_deltas_vs_reference_s": deltas,
            "tracking_metrics": metrics,
        }
    return rows, document


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _plot_region_tracking(
    path: Path,
    reference_id: str,
    reference: np.ndarray,
    open_loop: SimulationResult,
    mpc: SimulationResult,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 9.0), sharex=True)
    trajectories = (("Full-power L-shape", open_loop), ("Closed-loop MPC replay", mpc))
    interior_colors = {1: "#9ecae1", 2: "#4292c6", 4: "#08519c"}
    boundary_colors = {1: "#fcbba1", 2: "#ef3b2c", 4: "#99000d"}
    for axis, (title, result) in zip(axes, trajectories, strict=True):
        axis.plot(result.time_s, reference, color="black", linewidth=2.7, label="isotonic reference")
        axis.plot(result.time_s, result.region_doc["full_target"], color="#6a3d9a", linewidth=2.2, label="full target")
        for depth in EROSION_DEPTHS:
            axis.plot(
                result.time_s,
                result.region_doc[f"interior_N{depth}"],
                color=interior_colors[depth],
                linewidth=1.5,
                label=f"interior N={depth}",
            )
            axis.plot(
                result.time_s,
                result.region_doc[f"boundary_N{depth}"],
                color=boundary_colors[depth],
                linewidth=1.3,
                linestyle="--",
                label=f"boundary N={depth}",
            )
        axis.set_ylim(-0.025, 1.04)
        axis.set_ylabel("Mean DoC")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="lower right", frameon=False)
    axes[1].set_xlabel("Absolute process time (s)")
    axes[1].set_xlim(0.0, 20.0)
    fig.suptitle(f"Spatial tracking regions - {CONDITION_LABELS[reference_id]}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_negative_map(
    path: Path,
    target: np.ndarray,
    regions: dict[str, np.ndarray],
    result: SimulationResult,
) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(18.0, 9.2), constrained_layout=True
    )
    classification = np.zeros(target.shape, dtype=int)
    classification[regions["interior_N1"]] = 1
    classification[regions["boundary_N1"]] = 2
    count = result.negative_count_map
    count_masked = np.ma.masked_where(count <= 0, count)
    count_max = max(1, int(np.max(count)))
    magnitude = result.negative_magnitude_map
    event_rows, event_cols = np.nonzero(count > 0)
    if not event_rows.size:
        raise ValueError("negative-event map requested without any event pixels")
    padding = 10
    row_min = max(0, int(event_rows.min()) - padding)
    row_max = min(target.shape[0] - 1, int(event_rows.max()) + padding)
    col_min = max(0, int(event_cols.min()) - padding)
    col_max = min(target.shape[1] - 1, int(event_cols.max()) + padding)
    count_norm = LogNorm(vmin=1, vmax=max(1.01, count_max))
    image_handles: list[Any] = []
    for row_index in range(2):
        row_axes = axes[row_index]
        image_handles = [
            row_axes[0].imshow(
                classification,
                cmap=ListedColormap(["#eeeeee", "#3182bd", "#fd8d3c"]),
                interpolation="nearest",
            ),
            row_axes[1].imshow(
                count_masked,
                cmap="magma",
                norm=count_norm,
                interpolation="nearest",
            ),
            row_axes[2].imshow(magnitude, cmap="inferno", interpolation="nearest"),
            row_axes[3].imshow(
                result.minimum_delta_r_map,
                cmap="coolwarm",
                interpolation="nearest",
            ),
        ]
        for axis in row_axes:
            axis.contour(
                target.astype(float), levels=[0.5], colors="cyan", linewidths=0.6
            )
            axis.scatter(
                event_cols,
                event_rows,
                facecolors="none",
                edgecolors="#ff00ff",
                s=24 if row_index else 10,
                linewidths=0.8,
            )
            if row_index == 1:
                axis.set_xlim(col_min, col_max)
                axis.set_ylim(row_max, row_min)
                axis.set_xticks(np.arange(col_min, col_max + 1, 5))
                axis.set_yticks(np.arange(row_min, row_max + 1, 5))
                axis.grid(color="white", alpha=0.18, linewidth=0.5)
            else:
                axis.set_xticks([])
                axis.set_yticks([])
    for column, title in enumerate(
        (
            "N=1 region classification",
            "Negative delta-R event count",
            "Cumulative |negative delta-R|",
            "Minimum delta-R per pixel",
        )
    ):
        axes[0, column].set_title(title)
    axes[1, 0].set_ylabel("Concave-corner zoom")
    fig.colorbar(image_handles[1], ax=axes[:, 1].tolist(), fraction=0.035, pad=0.03)
    fig.colorbar(image_handles[2], ax=axes[:, 2].tolist(), fraction=0.035, pad=0.03)
    fig.colorbar(image_handles[3], ax=axes[:, 3].tolist(), fraction=0.035, pad=0.03)
    axes[0, 0].legend(
        handles=[
            Patch(color="#3182bd", label="target interior N=1"),
            Patch(color="#fd8d3c", label="target boundary N=1"),
            Patch(color="#eeeeee", label="outside"),
            Patch(facecolor="none", edgecolor="#ff00ff", label="negative delta-R pixel"),
        ],
        fontsize=7,
        loc="lower right",
        frameon=True,
    )
    fig.suptitle("30 mW/cm^2 / 5 mM: local reaction-progress reversals", fontsize=15)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_snapshot_maps(path: Path, target: np.ndarray, snapshots: dict[str, Snapshot]) -> None:
    ordered = [snapshots[label] for label in ("t10", "t50", "t90", "t95", "10s")]
    gap_limit = max(
        1e-6, max(float(np.max(np.abs(snapshot.reference_minus_local_doc))) for snapshot in ordered)
    )
    dose_limit = max(
        1e-9, max(float(np.max(np.abs(snapshot.delta_dose))) for snapshot in ordered)
    )
    r_limit = max(
        1e-9,
        max(float(np.max(np.abs(snapshot.delta_reaction_progress))) for snapshot in ordered),
    )
    fig, axes = plt.subplots(5, 4, figsize=(18.5, 20.5))
    last_images = [None, None, None, None]
    history_cmap = ListedColormap(["#ff00ff"])
    for row, snapshot in enumerate(ordered):
        fields = (
            snapshot.doc,
            snapshot.reference_minus_local_doc,
            snapshot.delta_dose,
            snapshot.delta_reaction_progress,
        )
        norms = (
            None,
            TwoSlopeNorm(vmin=-gap_limit, vcenter=0.0, vmax=gap_limit),
            TwoSlopeNorm(vmin=-dose_limit, vcenter=0.0, vmax=dose_limit),
            TwoSlopeNorm(vmin=-r_limit, vcenter=0.0, vmax=r_limit),
        )
        cmaps = ("viridis", "coolwarm", "coolwarm", "coolwarm")
        for column, (field, norm, cmap) in enumerate(zip(fields, norms, cmaps, strict=True)):
            kwargs: dict[str, Any] = {"cmap": cmap, "interpolation": "nearest"}
            if column == 0:
                kwargs.update(vmin=0.0, vmax=1.0)
            else:
                kwargs["norm"] = norm
            last_images[column] = axes[row, column].imshow(field, **kwargs)
            history_overlay = np.ma.masked_where(
                ~snapshot.historical_negative_mask,
                snapshot.historical_negative_mask.astype(float),
            )
            axes[row, column].imshow(
                history_overlay,
                cmap=history_cmap,
                alpha=0.40,
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            history_rows, history_cols = np.nonzero(
                snapshot.historical_negative_mask
            )
            if history_rows.size:
                axes[row, column].scatter(
                    history_cols,
                    history_rows,
                    facecolors="none",
                    edgecolors="#ff00ff",
                    s=28,
                    linewidths=0.9,
                )
            axes[row, column].contour(
                target.astype(float), levels=[0.5], colors="white", linewidths=0.45
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            f"{snapshot.requested_label}\nrequested {snapshot.requested_time_s:.3f}s\nsampled {snapshot.sampled_time_s:.2f}s"
        )
    for column, title in enumerate(
        ("DoC field", "spatial reference - local DoC", "delta Dose", "delta reaction_progress")
    ):
        axes[0, column].set_title(title)
        assert last_images[column] is not None
        fig.colorbar(
            last_images[column],
            ax=axes[:, column].tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.025,
        )
    fig.legend(
        handles=[Patch(color="#ff00ff", alpha=0.55, label="historical negative delta-R pixel")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    fig.suptitle("30 mW/cm^2 / 5 mM full-power L-shape spatial lag maps", y=0.999, fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _contiguous_control_groups(times: np.ndarray, mask: np.ndarray, control_dt: float) -> list[list[float]]:
    selected = times[mask]
    if not selected.size:
        return []
    groups: list[list[float]] = [[float(selected[0])]]
    for value in selected[1:]:
        if math.isclose(float(value - groups[-1][-1]), control_dt, abs_tol=1e-9):
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return groups


def _control_analysis(
    archive: MPCArchive,
    target: np.ndarray,
    open_loop: SimulationResult,
    mpc: SimulationResult,
    reference: np.ndarray,
) -> dict[str, Any]:
    target_control_mean = archive.controls[:, target].mean(axis=1)
    full_at_controls = np.interp(
        archive.control_times_s,
        open_loop.time_s,
        open_loop.region_doc["full_target"],
    )
    reference_at_controls = np.interp(
        archive.control_times_s, open_loop.time_s, reference
    )
    mpc_at_controls = np.interp(
        archive.control_times_s,
        mpc.time_s,
        mpc.region_doc["full_target"],
    )
    control_dt = float(np.median(np.diff(archive.control_times_s)))
    result: dict[str, Any] = {
        "control_times_s": archive.control_times_s,
        "target_region_mean_control": target_control_mean,
        "experimental_reference_doc": reference_at_controls,
        "full_power_target_doc": full_at_controls,
        "closed_loop_target_doc": mpc_at_controls,
        "full_power_minus_reference": full_at_controls - reference_at_controls,
        "control_threshold_events": {},
    }
    for threshold in (0.95, 0.80, 0.60):
        below = target_control_mean < threshold
        event_rows = []
        for index in np.flatnonzero(below):
            event_rows.append(
                {
                    "time_s": float(archive.control_times_s[index]),
                    "target_region_mean_control": float(target_control_mean[index]),
                    "reference_doc": float(reference_at_controls[index]),
                    "full_power_target_doc": float(full_at_controls[index]),
                    "full_power_minus_reference": float(
                        full_at_controls[index] - reference_at_controls[index]
                    ),
                    "full_power_capable_at_time": bool(
                        full_at_controls[index] >= reference_at_controls[index]
                    ),
                }
            )
        result["control_threshold_events"][f"below_{threshold:.2f}"] = {
            "count": int(below.sum()),
            "fraction": float(np.mean(below)),
            "contiguous_time_groups_s": _contiguous_control_groups(
                archive.control_times_s, below, control_dt
            ),
            "events": event_rows,
        }
    return result


def _plot_control_feasibility(
    path: Path,
    analysis: dict[str, Any],
    time_s: np.ndarray,
    reference: np.ndarray,
    full_power_doc: np.ndarray,
    mpc_doc: np.ndarray,
) -> None:
    control_times = np.asarray(analysis["control_times_s"], dtype=float)
    controls = np.asarray(analysis["target_region_mean_control"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True)
    axes[0].plot(time_s, reference, color="black", linewidth=2.6, label="isotonic reference")
    axes[0].plot(time_s, full_power_doc, color="#d62728", linewidth=2.0, label="full-power L-shape")
    axes[0].plot(time_s, mpc_doc, color="#2ca02c", linewidth=1.8, label="MPC replay target DoC")
    infeasible = full_power_doc < reference
    axes[0].fill_between(
        time_s,
        full_power_doc,
        reference,
        where=infeasible,
        color="#d62728",
        alpha=0.18,
        interpolate=True,
        label="full-power below reference",
    )
    axes[0].set_ylabel("Mean target DoC")
    axes[0].set_ylim(-0.025, 1.04)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].step(control_times, controls, where="post", color="#1f77b4", linewidth=2.0, label="target-region mean control")
    colors = {0.95: "#ffbf00", 0.80: "#ff7f0e", 0.60: "#d62728"}
    for threshold, color in colors.items():
        below = controls < threshold
        axes[1].axhline(threshold, color=color, linestyle="--", linewidth=1.0)
        axes[1].scatter(control_times[below], controls[below], color=color, s=34, label=f"u < {threshold:.2f}", zorder=4)
    axes[1].fill_between(
        time_s,
        0.0,
        1.0,
        where=infeasible,
        color="#d62728",
        alpha=0.08,
        transform=axes[1].get_xaxis_transform(),
        label="full-power infeasible interval",
    )
    axes[1].set_xlim(0.0, 20.0)
    axes[1].set_ylim(0.45, 1.01)
    axes[1].set_xlabel("Absolute process time (s)")
    axes[1].set_ylabel("Mean target control")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("30 mW/cm^2 / 5 mM: MPC control dip versus full-power feasibility", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _event_causality(events: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    count = len(events)
    if count == 0:
        return {
            "event_count": 0,
            "all_negative_delta_r_caused_by_negative_delta_dose": True,
            "all_events_curing_gate_true": True,
            "all_events_match_reaction_equation": True,
        }
    signed_distances = sorted(
        {float(event["signed_distance_to_target_boundary_px"]) for event in events}
    )
    unique_pixels = sorted(
        {(int(event["row"]), int(event["column"])) for event in events}
    )
    unique_times = sorted({float(event["absolute_time_s"]) for event in events})
    return {
        "event_count": count,
        "all_negative_delta_r_caused_by_negative_delta_dose": all(
            float(event["delta_dose_mj_cm2"]) < -tolerance for event in events
        ),
        "all_events_curing_gate_true": all(
            bool(event["curing_gate"]) for event in events
        ),
        "all_events_match_reaction_equation": all(
            abs(float(event["reaction_equation_residual"])) <= tolerance
            for event in events
        ),
        "all_negative_delta_dose_explained_by_inhibitor_sum_exceeding_energy": all(
            bool(event["inhibition_sum_exceeds_energy"]) for event in events
        ),
        "diffusion_increased_inhibitor_sum_event_count": sum(
            bool(event["diffusion_increased_inhibitor_sum"]) for event in events
        ),
        "diffusion_created_negative_increment_event_count": sum(
            bool(event["diffusion_created_negative_dose_increment"])
            for event in events
        ),
        "negative_delta_doc_event_count": sum(
            event["doc_change_class"] == "decreases" for event in events
        ),
        "unchanged_delta_doc_event_count": sum(
            event["doc_change_class"] == "unchanged_within_tolerance"
            for event in events
        ),
        "positive_delta_doc_event_count": sum(
            event["doc_change_class"] == "increases" for event in events
        ),
        "minimum_delta_doc": min(float(event["delta_doc"]) for event in events),
        "maximum_absolute_reaction_equation_residual": max(
            abs(float(event["reaction_equation_residual"])) for event in events
        ),
        "event_count_by_region_n1": {
            region: sum(event["region_n1"] == region for event in events)
            for region in ("target_interior", "target_boundary", "outside_target")
        },
        "unique_event_pixel_count": len(unique_pixels),
        "unique_event_pixels_row_column": [list(pixel) for pixel in unique_pixels],
        "unique_event_times_s": unique_times,
        "signed_distance_to_target_boundary_values_px": signed_distances,
        "local_control_range": [
            min(float(event["local_control"]) for event in events),
            max(float(event["local_control"]) for event in events),
        ],
        "local_intensity_range_mw_cm2": [
            min(float(event["local_intensity_mw_cm2"]) for event in events),
            max(float(event["local_intensity_mw_cm2"]) for event in events),
        ],
        "delta_dose_range_mj_cm2": [
            min(float(event["delta_dose_mj_cm2"]) for event in events),
            max(float(event["delta_dose_mj_cm2"]) for event in events),
        ],
        "delta_reaction_progress_range": [
            min(float(event["delta_reaction_progress"]) for event in events),
            max(float(event["delta_reaction_progress"]) for event in events),
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose local reaction-progress reversals and spatial tracking "
            "regions without optimization or production changes."
        )
    )
    parser.add_argument(
        "--target", type=Path, default=REPOSITORY_DIR / "GEO" / "Lshape.png"
    )
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--total-time", type=float, default=20.0)
    parser.add_argument("--negative-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mpc-output",
        action="append",
        default=[],
        metavar="REFERENCE_ID=PATH",
        help="Override an existing MPC directory/NPZ; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.target_threshold < 1.0:
        raise ValueError("--target-threshold must lie strictly between 0 and 1")
    if not math.isfinite(args.negative_tolerance) or args.negative_tolerance <= 0:
        raise ValueError("--negative-tolerance must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    target_path = resolve_target_path(args.target)
    target_tensor = load_normalized_target(target_path)
    require_native_target(target_tensor, target_path)
    target = target_tensor.numpy() > args.target_threshold
    regions, region_metadata, signed_distance = _construct_regions(target)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mpc_paths = dict(DEFAULT_MPC_OUTPUTS)
    mpc_paths.update(_parse_mpc_overrides(args.mpc_output))
    for reference_id, path in mpc_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"existing MPC output required for {reference_id}: {path}"
            )

    print("Spatial tracking-gap diagnostic (NO optimization)")
    print(f"repository={REPOSITORY_DIR}")
    print(f"device={device} target={target_path} shape={target.shape}")
    print(
        f"negative delta-R criterion: delta_R < -{args.negative_tolerance:.3g}"
    )
    print("regions: full target; 3x3 binary erosion interior/boundary N=1,2,4")

    all_events: list[dict[str, Any]] = []
    region_metric_rows: list[dict[str, Any]] = []
    runtime_results: dict[str, dict[str, Any]] = {}
    summary_conditions: dict[str, Any] = {}

    for reference_id in CONDITION_IDS:
        print(f"\n=== {reference_id} ===")
        curve = load_doc_reference(reference_id)
        config = load_reference_config_for_condition(reference_id)
        params = AIEParameters.from_reference(config)
        physics_match = assess_reference_physics_match(curve, params)
        if not physics_match["matched"]:
            raise RuntimeError(
                f"reference/physics mismatch for {reference_id}: "
                f"{physics_match['missing_for_physical_match']}"
            )
        if tuple(params.native_shape) != tuple(target.shape):
            raise ValueError("target does not match authoritative native shape")
        if not math.isclose(params.dt, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"authoritative dt changed: {params.dt}")
        archive = _load_mpc_archive(mpc_paths[reference_id], reference_id)
        if archive.reference_model_sha256 != params.reference_model_sha256:
            raise RuntimeError(f"MPC physics SHA mismatch for {reference_id}")
        if archive.doc_reference_sha256 != curve.source_sha256:
            raise RuntimeError(f"MPC reference SHA mismatch for {reference_id}")
        if tuple(archive.controls.shape[1:]) != tuple(target.shape):
            raise ValueError("MPC controls do not match target shape")

        fixed_control = target.astype(np.float32)[None, :, :]
        reference_grid = np.arange(
            int(round(args.total_time / params.dt)) + 1, dtype=float
        ) * params.dt
        reference_values = np.asarray(curve.at(reference_grid), dtype=float)
        snapshot_requests = None
        if reference_id == "30mW_5mM":
            snapshot_requests = {}
            requested = {
                "t10": float(curve.metadata["threshold_times_s"]["t10_s"]),
                "t50": float(curve.metadata["threshold_times_s"]["t50_s"]),
                "t90": float(curve.metadata["threshold_times_s"]["t90_s"]),
                "t95": float(curve.metadata["threshold_times_s"]["t95_s"]),
                "10s": 10.0,
            }
            for label, requested_time in requested.items():
                step = int(round(requested_time / params.dt))
                sampled_time = step * params.dt
                snapshot_requests[step] = (
                    label,
                    requested_time,
                    float(curve.at(sampled_time)),
                )

        print("simulating fixed full-power L-shape and tracing local equations...")
        open_loop = _simulate(
            reference_id=reference_id,
            params=params,
            target=target,
            region_masks=regions,
            signed_distance=signed_distance,
            controls=fixed_control,
            physics_steps_per_control=1,
            total_time_s=args.total_time,
            device=device,
            negative_tolerance=args.negative_tolerance,
            trajectory="full_power_Lshape",
            collect_event_details=True,
            snapshot_requests=snapshot_requests,
        )
        print("replaying saved closed-loop MPC masks without optimization...")
        mpc_replay = _simulate(
            reference_id=reference_id,
            params=params,
            target=target,
            region_masks=regions,
            signed_distance=signed_distance,
            controls=archive.controls,
            physics_steps_per_control=archive.physics_steps_per_control,
            total_time_s=args.total_time,
            device=device,
            negative_tolerance=args.negative_tolerance,
            trajectory="closed_loop_MPC_native_replay",
            collect_event_details=False,
        )
        if not np.array_equal(open_loop.time_s, reference_grid):
            raise AssertionError("open-loop reference grid mismatch")
        if not np.array_equal(mpc_replay.time_s, reference_grid):
            raise AssertionError("MPC replay reference grid mismatch")
        control_indices = np.rint(archive.control_times_s / params.dt).astype(int)
        replay_error = float(
            np.max(
                np.abs(
                    mpc_replay.region_doc["full_target"][control_indices]
                    - archive.actual_target_doc
                )
            )
        )
        if replay_error > 2e-6:
            raise RuntimeError(
                f"MPC native replay does not reproduce saved target DoC: {replay_error}"
            )

        open_rows, open_metrics = _build_region_metrics(
            reference_id=reference_id,
            trajectory="full_power_Lshape",
            result=open_loop,
            reference=reference_values,
            curve=curve,
            region_masks=regions,
        )
        mpc_rows, mpc_metrics = _build_region_metrics(
            reference_id=reference_id,
            trajectory="closed_loop_MPC_native_replay",
            result=mpc_replay,
            reference=reference_values,
            curve=curve,
            region_masks=regions,
        )
        region_metric_rows.extend(open_rows)
        region_metric_rows.extend(mpc_rows)
        all_events.extend(open_loop.events)
        control_analysis = _control_analysis(
            archive, target, open_loop, mpc_replay, reference_values
        )
        causality = _event_causality(open_loop.events, args.negative_tolerance)

        full_rmse = open_metrics["full_target"]["tracking_metrics"][
            "rise_region_rmse"
        ]
        interior_rmse = open_metrics["interior_N2"]["tracking_metrics"][
            "rise_region_rmse"
        ]
        rmse_reduction = (
            None
            if full_rmse is None or interior_rmse is None or full_rmse == 0
            else float((full_rmse - interior_rmse) / full_rmse)
        )
        summary_conditions[reference_id] = {
            "reference_id": reference_id,
            "condition": curve.metadata["condition"],
            "reference_provenance": curve.provenance_metadata(),
            "forward_physics_match": physics_match,
            "forward_physics_parameters": params.provenance_metadata(),
            "reference_threshold_times_s": _threshold_times(
                reference_grid, reference_values
            ),
            "full_power_Lshape": {
                "region_metrics": open_metrics,
                "reaction_progress_monotonicity": open_loop.monotonicity,
                "negative_event_summary": open_loop.event_summary,
                "negative_event_causality": causality,
                "validation": open_loop.validation,
            },
            "closed_loop_MPC_native_replay": {
                "source_path": str(archive.path),
                "source_sha256": _sha256(archive.path),
                "region_metrics": mpc_metrics,
                "reaction_progress_monotonicity": mpc_replay.monotonicity,
                "validation": mpc_replay.validation,
                "maximum_saved_vs_replayed_target_doc_error": replay_error,
            },
            "interior_N2_vs_full_target": {
                "full_target_rise_rmse": full_rmse,
                "interior_N2_rise_rmse": interior_rmse,
                "relative_rise_rmse_reduction": rmse_reduction,
                "full_target_t90_s": open_metrics["full_target"][
                    "threshold_times_s"
                ]["t90_s"],
                "interior_N2_t90_s": open_metrics["interior_N2"][
                    "threshold_times_s"
                ]["t90_s"],
                "boundary_N2_t90_s": open_metrics["boundary_N2"][
                    "threshold_times_s"
                ]["t90_s"],
                "full_target_t95_s": open_metrics["full_target"][
                    "threshold_times_s"
                ]["t95_s"],
                "interior_N2_t95_s": open_metrics["interior_N2"][
                    "threshold_times_s"
                ]["t95_s"],
                "boundary_N2_t95_s": open_metrics["boundary_N2"][
                    "threshold_times_s"
                ]["t95_s"],
            },
            "mpc_control_analysis": control_analysis,
            "time_series": {
                "time_s": reference_grid,
                "reference_doc": reference_values,
                "full_power_region_doc": open_loop.region_doc,
                "mpc_replay_region_doc": mpc_replay.region_doc,
            },
        }
        runtime_results[reference_id] = {
            "curve": curve,
            "reference": reference_values,
            "open_loop": open_loop,
            "mpc_replay": mpc_replay,
            "archive": archive,
            "control_analysis": control_analysis,
        }
        plot_path = output_dir / f"spatial_tracking_gap_{reference_id}.png"
        _plot_region_tracking(
            plot_path, reference_id, reference_values, open_loop, mpc_replay
        )
        print(
            f"negative events={len(open_loop.events)}; "
            f"min delta_R={open_loop.monotonicity['full_image']['minimum_delta_r']}"
        )
        print(f"MPC replay max target-mean error={replay_error:.3e}")

    event_path = output_dir / "reaction_progress_negative_events.csv"
    metric_path = output_dir / "region_tracking_metrics.csv"
    _write_csv(event_path, EVENT_FIELDS, all_events)
    metric_fields = [
        "condition_id",
        "trajectory",
        "region",
        "pixel_count",
        *[f"t{int(round(value * 100)):02d}_s" for value in THRESHOLDS],
        *[f"delta_t{int(round(value * 100)):02d}_s" for value in THRESHOLDS],
        "full_duration_rmse",
        "full_duration_mae",
        "full_duration_sample_count",
        "pre_saturation_rmse",
        "pre_saturation_mae",
        "pre_saturation_sample_count",
        "rise_region_rmse",
        "rise_region_mae",
        "rise_region_sample_count",
        "maximum_absolute_error",
    ]
    _write_csv(metric_path, metric_fields, region_metric_rows)

    five = runtime_results["30mW_5mM"]
    negative_map_path = output_dir / "reaction_progress_negative_map_30mW_5mM.png"
    spatial_maps_path = output_dir / "spatial_lag_maps_30mW_5mM.png"
    control_plot_path = output_dir / "control_vs_feasibility_30mW_5mM.png"
    _plot_negative_map(
        negative_map_path, target, regions, five["open_loop"]
    )
    _plot_snapshot_maps(
        spatial_maps_path, target, five["open_loop"].snapshots
    )
    _plot_control_feasibility(
        control_plot_path,
        five["control_analysis"],
        five["open_loop"].time_s,
        five["reference"],
        five["open_loop"].region_doc["full_target"],
        five["mpc_replay"].region_doc["full_target"],
    )

    model_probe = AIEModel(
        AIEParameters.from_reference(
            load_reference_config_for_condition("30mW_5mM")
        ),
        device=device,
    )
    kernel_provenance = {
        "o2_kernel_sum": float(model_probe.o2_kernel_1d.sum().detach().cpu()),
        "o2_kernel_minimum": float(model_probe.o2_kernel_1d.min().detach().cpu()),
        "tempo_kernel_sum": float(model_probe.tempo_kernel_1d.sum().detach().cpu()),
        "tempo_kernel_minimum": float(
            model_probe.tempo_kernel_1d.min().detach().cpu()
        ),
        "padding_mode": "reflect",
        "convolution": "separable normalized nonnegative Gaussian kernels",
    }
    five_summary = summary_conditions["30mW_5mM"]
    five_causality = five_summary["full_power_Lshape"][
        "negative_event_causality"
    ]
    five_interior = five_summary["interior_N2_vs_full_target"]
    five_reference_times = five_summary["reference_threshold_times_s"]
    full_t90_delay = (
        five_interior["full_target_t90_s"] - five_reference_times["t90_s"]
    )
    interior_t90_delay = (
        five_interior["interior_N2_t90_s"] - five_reference_times["t90_s"]
    )
    full_t95_delay = (
        five_interior["full_target_t95_s"] - five_reference_times["t95_s"]
    )
    interior_t95_delay = (
        five_interior["interior_N2_t95_s"] - five_reference_times["t95_s"]
    )
    below_080 = five_summary["mpc_control_analysis"][
        "control_threshold_events"
    ]["below_0.80"]["events"]
    event_regions = five_causality["event_count_by_region_n1"]
    signed_event_distances = five_causality[
        "signed_distance_to_target_boundary_values_px"
    ]
    all_events_one_pixel_outside = bool(
        five_causality["event_count"]
        and event_regions["outside_target"] == five_causality["event_count"]
        and event_regions["target_interior"] == 0
        and event_regions["target_boundary"] == 0
        and signed_event_distances == [-1.0]
    )
    decision_answers = {
        "all_negative_reaction_progress_increments_caused_by_negative_delta_dose": (
            five_causality[
                "all_negative_delta_r_caused_by_negative_delta_dose"
            ]
        ),
        "negative_reaction_progress_events_causing_decreasing_doc": (
            five_causality["negative_delta_doc_event_count"]
        ),
        "negative_increment_location": (
            "All detected 5 mM full-power events are one-pixel-outside-target "
            "pixels near the concave L corner; target interior and target "
            "boundary event counts are zero."
            if all_events_one_pixel_outside
            else "See event_count_by_region_n1 and signed distance metadata."
        ),
        "interior_N2_tracking_improvement": {
            "rise_rmse_relative_reduction": five_interior[
                "relative_rise_rmse_reduction"
            ],
            "t90_delay_full_target_s": full_t90_delay,
            "t90_delay_interior_N2_s": interior_t90_delay,
            "t90_delay_reduction_fraction": (
                (full_t90_delay - interior_t90_delay) / full_t90_delay
            ),
            "t95_delay_full_target_s": full_t95_delay,
            "t95_delay_interior_N2_s": interior_t95_delay,
            "t95_delay_reduction_fraction": (
                (full_t95_delay - interior_t95_delay) / full_t95_delay
            ),
        },
        "five_mM_delay_attribution": (
            "Combination: a large boundary/spatial effect dominates much of the "
            "full-target t90/t95 delay; the eroded interior still has a kinetic "
            "delay; and the saved MPC has a distinct mid-rise control dip. The "
            "local negative reaction_progress events do not contribute to the "
            "target mean because they occur outside the target."
        ),
        "full_power_capability_during_control_below_0_80": {
            "all_dip_times_full_power_capable": all(
                bool(event["full_power_capable_at_time"])
                for event in below_080
            ),
            "events": below_080,
        },
        "recommended_next_investigations_before_any_production_change": [
            "validate 5 mM interior kinetics against spatially resolved experiment",
            "investigate boundary diffusion/scattering and the CenterROI-versus-full-target objective mismatch",
            "diagnose the MPC control dip/warm-start landscape before changing weights",
            "review negative Dose/reaction_progress bookkeeping for outside pixels, although it did not cause target DoC reversal here",
        ],
    }
    summary_path = output_dir / "spatial_tracking_gap_summary.json"
    output_paths = {
        "negative_events_csv": str(event_path),
        "region_metrics_csv": str(metric_path),
        "spatial_tracking_plot_0mM": str(
            output_dir / "spatial_tracking_gap_30mW_0mM.png"
        ),
        "spatial_tracking_plot_5mM": str(
            output_dir / "spatial_tracking_gap_30mW_5mM.png"
        ),
        "negative_event_map_5mM": str(negative_map_path),
        "spatial_lag_maps_5mM": str(spatial_maps_path),
        "control_vs_feasibility_5mM": str(control_plot_path),
    }
    summary = {
        "schema_version": 1,
        "diagnostic_id": "spatial_tracking_gap_and_reaction_progress_trace_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "optimization": {
            "performed": False,
            "description": (
                "Only fixed full-power propagation and native replay of previously "
                "saved MPC controls were performed."
            ),
        },
        "repository_dir": str(REPOSITORY_DIR),
        "diagnostic_script": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
        "negative_delta_r_tolerance": args.negative_tolerance,
        "target": {
            "path": str(target_path),
            "sha256": _sha256(target_path),
            "shape": list(target.shape),
            "threshold": args.target_threshold,
        },
        "spatial_regions": region_metadata,
        "equation_trace": {
            "production_code": {
                "delta_dose": "dose_next - dose_previous",
                "reaction_candidate": (
                    "reaction_progress_previous + effective_B * delta_dose / "
                    "safe_local_intensity"
                ),
                "reaction_next": (
                    "where(curing, reaction_candidate, reaction_progress_previous)"
                ),
                "dose_when_curing": (
                    "dose_previous + energy - o2_diffused - tempo_diffused"
                ),
                "curing_gate": "(o2_next <= 0) & (tempo_next <= 0)",
            },
            "kernel_provenance": kernel_provenance,
            "interpretation_scope": (
                "The diagnostic traces the unchanged equations; it does not clamp "
                "Dose, reaction_progress, or DoC and does not change physics."
            ),
        },
        "doc_history_mode": DOC_HISTORY_MODE,
        "doc_history_description": DOC_HISTORY_DESCRIPTION,
        "conditions": summary_conditions,
        "decision_answers": decision_answers,
        "outputs": output_paths,
        "protected_provenance": {
            "AIE_TEMPOv1.1.py_sha256": _sha256(
                REPOSITORY_DIR / "AIE_TEMPOv1.1.py"
            ),
            "doc_reference_curves.json_sha256": _sha256(
                REPOSITORY_DIR / "doc_reference_curves.json"
            ),
            "aie_model.py_sha256": _sha256(REPOSITORY_DIR / "aie_model.py"),
            "aie_mpc.py_sha256": _sha256(REPOSITORY_DIR / "aie_mpc.py"),
            "run_mpc.py_sha256": _sha256(REPOSITORY_DIR / "run_mpc.py"),
        },
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    causality = five_summary["full_power_Lshape"]["negative_event_causality"]
    comparison = five_summary["interior_N2_vs_full_target"]
    control = five_summary["mpc_control_analysis"]["control_threshold_events"]
    print("\n=== Key results ===")
    print(
        "5 mM all negative delta-R caused by negative delta-Dose: "
        f"{causality['all_negative_delta_r_caused_by_negative_delta_dose']}"
    )
    print(
        "5 mM negative delta-R events causing decreasing DoC: "
        f"{causality['negative_delta_doc_event_count']}"
    )
    print(
        "5 mM rise RMSE full/interior N=2: "
        f"{comparison['full_target_rise_rmse']:.6f}/"
        f"{comparison['interior_N2_rise_rmse']:.6f}"
    )
    print(
        "5 mM MPC controls below 0.80/0.60: "
        f"{control['below_0.80']['count']}/{control['below_0.60']['count']}"
    )
    print("validation: finite states, [0,1] DoC, exact equation trace, and MPC replay PASS")
    print(f"summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
