"""Reproducible fitting/export helpers for the experimental DoC reference.

This module analyzes the first data set in ``DoC curve.ipynb``.  Its source
sheet and columns retain a legacy ``20mW`` name, but the confirmed physical
condition is 30 mW/cm^2 and 0 mM TEMPO.  The fitted trajectory is a controller
reference only; it is never used as the AIE forward dynamics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


SOURCE_DATA_FILE = "260722_circles_TPEoac/data_exports/intensity_data_processed.xlsx"
SOURCE_SHEET = "20mW"
SOURCE_COLUMNS = {
    "T1": {
        "time": "20mW_0mM_TEMPO_T1_Time_s",
        "signal": "20mW_0mM_TEMPO_T1_Intensity",
    },
    "T2": {
        "time": "20mW_0mM_TEMPO_T2_Time_s",
        "signal": "20mW_0mM_TEMPO_T2_Intensity",
    },
}
CONDITION = {"intensity_mw_cm2": 30.0, "tempo_concentration_mM": 0.0}
CONDITION_LABEL = "0 mM TEMPO, 30 mW/cm^2"
LEGACY_NAME_NOTE = (
    "The source sheet and columns contain '20mW', a confirmed legacy naming "
    "typo; these measurements represent 30 mW/cm^2 and 0 mM TEMPO."
)
FILTER_RULE = (
    "Normalize each finite replicate by its own raw maximum. After the trace "
    "has reached >=0.95, find the first later sample <=0.50 whose one-sample "
    "drop is >=0.40; exclude that sample and the remaining acquisition suffix "
    "from fitting. Preserve all raw/normalized points for provenance and do "
    "not filter any earlier inhibition samples."
)
REFERENCE_CONSTRUCTION = (
    "Equal-replicate joint nonlinear least squares: concatenate T1/T2 residuals "
    "and scale each replicate by 1/sqrt(number of retained samples), so each "
    "replicate contributes equal mean-squared weight. Separate replicate fits "
    "are retained for diagnostics and replicate spread."
)


@dataclass(frozen=True)
class ReplicateData:
    label: str
    time_s: np.ndarray
    raw_signal: np.ndarray
    normalized_signal: np.ndarray
    fit_mask: np.ndarray
    artifact_start_index: int | None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    parameter_names: tuple[str, ...]
    function: Callable[..., np.ndarray]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    initial_guess: tuple[float, ...]


def delayed_avrami(
    time_s: np.ndarray, y0: float, plateau: float, t0_s: float, tau_s: float, n: float
) -> np.ndarray:
    """Delayed Weibull/Avrami curve with an explicit inhibition baseline."""

    time_s = np.asarray(time_s, dtype=float)
    elapsed = np.maximum(time_s - t0_s, 0.0)
    cured = y0 + (plateau - y0) * (1.0 - np.exp(-np.power(elapsed / tau_s, n)))
    return np.where(time_s <= t0_s, y0, cured)


def logistic(
    time_s: np.ndarray, y0: float, plateau: float, midpoint_s: float, scale_s: float
) -> np.ndarray:
    """Four-parameter monotonic logistic curve."""

    z = np.clip((np.asarray(time_s, dtype=float) - midpoint_s) / scale_s, -700, 700)
    return y0 + (plateau - y0) / (1.0 + np.exp(-z))


def gompertz(
    time_s: np.ndarray, y0: float, plateau: float, midpoint_s: float, scale_s: float
) -> np.ndarray:
    """Four-parameter monotonic Gompertz curve."""

    z = np.clip(-(np.asarray(time_s, dtype=float) - midpoint_s) / scale_s, -700, 700)
    return y0 + (plateau - y0) * np.exp(-np.exp(z))


def old_delayed_exponential(
    time_s: np.ndarray, a: float, b_per_s: float, c_s: float
) -> np.ndarray:
    """Previously used delayed single-exponential model, retained for comparison."""

    exponent = np.minimum(b_per_s * (c_s - np.asarray(time_s, dtype=float)), 0.0)
    return 1.0 - a * np.exp(exponent)


MODEL_SPECS = (
    ModelSpec(
        "old_delayed_exponential",
        ("a", "b_per_s", "c_s"),
        old_delayed_exponential,
        (0.0, 1e-4, 0.0),
        (1.2, 10.0, 10.0),
        (0.97, 0.7, 1.3),
    ),
    ModelSpec(
        "delayed_avrami",
        ("y0", "plateau", "t0_s", "tau_s", "shape_n"),
        delayed_avrami,
        (0.0, 0.85, 0.0, 0.05, 0.2),
        (0.10, 1.05, 8.0, 20.0, 8.0),
        (0.02, 0.98, 1.0, 2.5, 2.0),
    ),
    ModelSpec(
        "logistic",
        ("y0", "plateau", "midpoint_s", "scale_s"),
        logistic,
        (0.0, 0.85, 0.0, 0.05),
        (0.10, 1.05, 12.0, 10.0),
        (0.01, 0.98, 3.5, 0.8),
    ),
    ModelSpec(
        "gompertz",
        ("y0", "plateau", "midpoint_s", "scale_s"),
        gompertz,
        (0.0, 0.85, 0.0, 0.05),
        (0.10, 1.05, 12.0, 10.0),
        (0.01, 0.98, 3.0, 1.0),
    ),
)


def _finite_trace(time_s: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(time_s) & np.isfinite(signal)
    time_s = np.asarray(time_s[finite], dtype=float)
    signal = np.asarray(signal[finite], dtype=float)
    order = np.argsort(time_s, kind="stable")
    return time_s[order], signal[order]


def terminal_artifact_fit_mask(normalized_signal: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Apply the documented post-plateau abrupt-drop rule reproducibly."""

    normalized_signal = np.asarray(normalized_signal, dtype=float)
    fit_mask = np.ones(normalized_signal.shape, dtype=bool)
    reached_plateau = np.maximum.accumulate(normalized_signal) >= 0.95
    drops = np.r_[0.0, normalized_signal[:-1] - normalized_signal[1:]]
    candidates = np.flatnonzero(reached_plateau & (normalized_signal <= 0.50) & (drops >= 0.40))
    if candidates.size == 0:
        return fit_mask, None
    artifact_start = int(candidates[0])
    fit_mask[artifact_start:] = False
    return fit_mask, artifact_start


def load_replicates(repository_dir: Path | str = ".") -> dict[str, ReplicateData]:
    """Load the legacy-named first data set without altering the workbook."""

    repository_dir = Path(repository_dir)
    frame = pd.read_excel(repository_dir / SOURCE_DATA_FILE, sheet_name=SOURCE_SHEET)
    replicates: dict[str, ReplicateData] = {}
    for label, columns in SOURCE_COLUMNS.items():
        time_s, raw_signal = _finite_trace(
            frame[columns["time"]].to_numpy(dtype=float),
            frame[columns["signal"]].to_numpy(dtype=float),
        )
        maximum = float(np.max(raw_signal))
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"{label} has no positive finite normalization maximum")
        normalized = raw_signal / maximum
        fit_mask, artifact_start = terminal_artifact_fit_mask(normalized)
        replicates[label] = ReplicateData(
            label=label,
            time_s=time_s,
            raw_signal=raw_signal,
            normalized_signal=normalized,
            fit_mask=fit_mask,
            artifact_start_index=artifact_start,
        )
    return replicates


def _metrics(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> dict[str, float | int]:
    residual = np.asarray(predicted) - np.asarray(observed)
    n = residual.size
    sse = float(np.dot(residual, residual))
    rmse = math.sqrt(sse / n)
    mae = float(np.mean(np.abs(residual)))
    centered = observed - float(np.mean(observed))
    sst = float(np.dot(centered, centered))
    r_squared = 1.0 - sse / sst if sst > np.finfo(float).eps else float("nan")
    safe_sse = max(sse, np.finfo(float).tiny)
    aic = n * math.log(safe_sse / n) + 2 * parameter_count
    bic = n * math.log(safe_sse / n) + parameter_count * math.log(n)
    return {
        "sample_count": n,
        "parameter_count": parameter_count,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "aic": aic,
        "bic": bic,
    }


def _parameter_record(spec: ModelSpec, values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(spec.parameter_names, values)}


def fit_replicate(data: ReplicateData, spec: ModelSpec) -> dict[str, object]:
    """Fit one candidate to one filtered replicate."""

    time_s = data.time_s[data.fit_mask]
    observed = data.normalized_signal[data.fit_mask]
    parameters, covariance = curve_fit(
        spec.function,
        time_s,
        observed,
        p0=spec.initial_guess,
        bounds=(spec.lower_bounds, spec.upper_bounds),
        maxfev=100_000,
    )
    predicted = spec.function(time_s, *parameters)
    return {
        "parameters": _parameter_record(spec, parameters),
        "parameter_vector": parameters,
        "parameter_std_error": {
            name: float(value)
            for name, value in zip(spec.parameter_names, np.sqrt(np.diag(covariance)))
        },
        "metrics": _metrics(observed, predicted, len(parameters)),
    }


def fit_joint_equal_replicate(
    replicates: dict[str, ReplicateData], spec: ModelSpec
) -> dict[str, object]:
    """Joint fit in which T1 and T2 contribute equal replicate-average SSE."""

    times: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    sigma: list[np.ndarray] = []
    for data in replicates.values():
        retained_time = data.time_s[data.fit_mask]
        retained_signal = data.normalized_signal[data.fit_mask]
        times.append(retained_time)
        observed.append(retained_signal)
        sigma.append(np.full(retained_signal.shape, math.sqrt(retained_signal.size)))
    all_time = np.concatenate(times)
    all_observed = np.concatenate(observed)
    all_sigma = np.concatenate(sigma)
    parameters, covariance = curve_fit(
        spec.function,
        all_time,
        all_observed,
        p0=spec.initial_guess,
        bounds=(spec.lower_bounds, spec.upper_bounds),
        sigma=all_sigma,
        absolute_sigma=False,
        maxfev=100_000,
    )
    predicted = spec.function(all_time, *parameters)
    return {
        "parameters": _parameter_record(spec, parameters),
        "parameter_vector": parameters,
        "parameter_std_error": {
            name: float(value)
            for name, value in zip(spec.parameter_names, np.sqrt(np.diag(covariance)))
        },
        "metrics": _metrics(all_observed, predicted, len(parameters)),
    }


def threshold_times(
    function: Callable[..., np.ndarray], parameters: np.ndarray, *, end_s: float = 20.0
) -> dict[str, float | None]:
    """Interpolate first fitted crossings of 10/30/50/90 percent DoC."""

    dense_time = np.linspace(0.0, end_s, int(round(end_s / 0.001)) + 1)
    dense_doc = function(dense_time, *parameters)
    result: dict[str, float | None] = {}
    for percent in (10, 30, 50, 90):
        threshold = percent / 100.0
        indices = np.flatnonzero(dense_doc >= threshold)
        if indices.size == 0:
            result[f"t{percent}_s"] = None
            continue
        index = int(indices[0])
        if index == 0:
            crossing = dense_time[0]
        else:
            crossing = np.interp(
                threshold,
                dense_doc[index - 1 : index + 1],
                dense_time[index - 1 : index + 1],
            )
        result[f"t{percent}_s"] = float(crossing)
    return result


def analyze_reference_condition(repository_dir: Path | str = ".") -> dict[str, object]:
    """Fit all candidates and return notebook-ready diagnostics."""

    replicates = load_replicates(repository_dir)
    candidates: dict[str, object] = {}
    for spec in MODEL_SPECS:
        replicate_fits = {label: fit_replicate(data, spec) for label, data in replicates.items()}
        joint_fit = fit_joint_equal_replicate(replicates, spec)
        for label, result in replicate_fits.items():
            result["threshold_times_s"] = threshold_times(
                spec.function, result["parameter_vector"]
            )
        joint_fit["threshold_times_s"] = threshold_times(
            spec.function, joint_fit["parameter_vector"]
        )
        candidates[spec.model_id] = {
            "spec": spec,
            "replicate_fits": replicate_fits,
            "joint_fit": joint_fit,
        }

    # The delayed Avrami model is selected a priori from the candidate set when
    # its aggregate RMSE improves materially on the old fit. This avoids picking
    # a model solely from noisy last-decimal score differences.
    old_rmse = candidates["old_delayed_exponential"]["joint_fit"]["metrics"]["rmse"]
    avrami_rmse = candidates["delayed_avrami"]["joint_fit"]["metrics"]["rmse"]
    if not avrami_rmse <= 0.90 * old_rmse:
        raise RuntimeError(
            "delayed Avrami did not improve joint RMSE by the required 10%: "
            f"old={old_rmse:.6g}, Avrami={avrami_rmse:.6g}"
        )
    return {
        "replicates": replicates,
        "candidates": candidates,
        "selected_model_id": "delayed_avrami",
    }


def model_comparison_table(analysis: dict[str, object]) -> pd.DataFrame:
    """Return the quantitative joint-fit table displayed by the notebook."""

    rows = []
    for model_id, candidate in analysis["candidates"].items():
        metrics = candidate["joint_fit"]["metrics"]
        rows.append(
            {
                "model": model_id,
                "parameter_count": metrics["parameter_count"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "R2": metrics["r_squared"],
                "AIC": metrics["aic"],
                "BIC": metrics["bic"],
            }
        )
    return pd.DataFrame(rows).sort_values(["RMSE", "parameter_count"]).reset_index(drop=True)


def plot_fit_comparison(analysis: dict[str, object]):
    """Plot raw/filtered replicates, old/candidate fits, and production reference."""

    import matplotlib.pyplot as plt

    time_s = np.linspace(0.0, 20.0, 1001)
    figure, axis = plt.subplots(figsize=(11, 7))
    colors = {"T1": "tab:blue", "T2": "tab:orange"}
    for label, data in analysis["replicates"].items():
        axis.scatter(
            data.time_s[data.fit_mask],
            data.normalized_signal[data.fit_mask],
            s=12,
            alpha=0.45,
            color=colors[label],
            label=f"{label} raw (fit retained)",
        )
        if np.any(~data.fit_mask):
            axis.scatter(
                data.time_s[~data.fit_mask],
                data.normalized_signal[~data.fit_mask],
                s=65,
                marker="x",
                linewidth=2,
                color=colors[label],
                label=f"{label} excluded LED-off artifact",
            )
    styles = {
        "old_delayed_exponential": ("black", "--"),
        "delayed_avrami": ("tab:green", "-"),
        "logistic": ("tab:red", ":"),
        "gompertz": ("tab:purple", "-."),
    }
    for model_id, candidate in analysis["candidates"].items():
        spec = candidate["spec"]
        parameters = candidate["joint_fit"]["parameter_vector"]
        color, linestyle = styles[model_id]
        label = "old fit" if model_id == "old_delayed_exponential" else model_id
        axis.plot(
            time_s,
            spec.function(time_s, *parameters),
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    selected = analysis["candidates"][analysis["selected_model_id"]]
    selected_curve = selected["spec"].function(
        time_s, *selected["joint_fit"]["parameter_vector"]
    )
    axis.plot(
        time_s,
        selected_curve,
        color="tab:green",
        linewidth=4,
        alpha=0.35,
        label="selected production reference",
    )
    axis.set_xlim(0.0, 20.0)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalized DoC proxy")
    axis.set_title(CONDITION_LABEL)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    return figure, axis


def plot_replicate_diagnostics(analysis: dict[str, object]):
    """Plot separate selected-model fits for T1 and T2."""

    import matplotlib.pyplot as plt

    selected = analysis["candidates"][analysis["selected_model_id"]]
    spec = selected["spec"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, (label, data) in zip(axes, analysis["replicates"].items()):
        fit = selected["replicate_fits"][label]
        time_s = np.linspace(0.0, 20.0, 1001)
        axis.scatter(
            data.time_s[data.fit_mask],
            data.normalized_signal[data.fit_mask],
            s=13,
            alpha=0.5,
            label=f"{label} retained raw",
        )
        if np.any(~data.fit_mask):
            axis.scatter(
                data.time_s[~data.fit_mask],
                data.normalized_signal[~data.fit_mask],
                marker="x",
                s=70,
                linewidth=2,
                label="excluded artifact",
            )
        axis.plot(
            time_s,
            spec.function(time_s, *fit["parameter_vector"]),
            linewidth=2.5,
            label="separate delayed-Avrami fit",
        )
        axis.set_title(label)
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Normalized DoC proxy")
    figure.suptitle(CONDITION_LABEL)
    figure.tight_layout()
    return figure, axes


def _json_fit_record(result: dict[str, object]) -> dict[str, object]:
    return {
        "parameters": result["parameters"],
        "parameter_std_error": result["parameter_std_error"],
        "metrics": result["metrics"],
        "threshold_times_s": result["threshold_times_s"],
    }


def build_reference_export(
    analysis: dict[str, object], *, dt_s: float = 0.05, end_s: float = 20.0
) -> dict[str, object]:
    """Build the schema-v1 runtime reference from fitted variables."""

    if dt_s <= 0 or end_s <= 0:
        raise ValueError("reference dt and end time must be positive")
    step_count = int(round(end_s / dt_s))
    if not math.isclose(step_count * dt_s, end_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reference end time must be an integer multiple of dt")
    selected_id = str(analysis["selected_model_id"])
    selected = analysis["candidates"][selected_id]
    spec: ModelSpec = selected["spec"]
    joint = selected["joint_fit"]
    time_s = np.linspace(0.0, end_s, step_count + 1)
    joint_doc_unclamped = spec.function(time_s, *joint["parameter_vector"])
    doc_reference = np.clip(joint_doc_unclamped, 0.0, 1.0)
    if not np.isfinite(doc_reference).all() or np.any(np.diff(doc_reference) < -1e-12):
        raise ValueError("exported reference must be finite and monotonic nondecreasing")

    replicate_curves = np.stack(
        [
            spec.function(time_s, *fit["parameter_vector"])
            for fit in selected["replicate_fits"].values()
        ]
    )
    replicate_std = np.std(replicate_curves, axis=0, ddof=1)
    comparison = {}
    for model_id, candidate in analysis["candidates"].items():
        comparison[model_id] = {
            "parameter_count": len(candidate["spec"].parameter_names),
            "joint_equal_replicate_metrics": candidate["joint_fit"]["metrics"],
            "joint_parameters": candidate["joint_fit"]["parameters"],
        }

    replicates_json = {}
    filtering_json = {}
    for label, data in analysis["replicates"].items():
        excluded = np.flatnonzero(~data.fit_mask)
        filtering_json[label] = {
            "raw_finite_sample_count": int(data.time_s.size),
            "fit_sample_count": int(np.count_nonzero(data.fit_mask)),
            "excluded_sample_count": int(excluded.size),
            "artifact_start_index_zero_based": data.artifact_start_index,
            "excluded_times_s": [float(value) for value in data.time_s[excluded]],
            "excluded_normalized_values": [
                float(value) for value in data.normalized_signal[excluded]
            ],
        }
        replicates_json[label] = _json_fit_record(selected["replicate_fits"][label])

    return {
        "schema_version": 1,
        "reference_id": "doc_reference_30mWcm2_0mM_TEMPO_v1",
        "condition": CONDITION,
        "condition_label": CONDITION_LABEL,
        "source_notebook": "DoC curve.ipynb",
        "source_data_file": SOURCE_DATA_FILE,
        "legacy_source_names": {
            "sheet": SOURCE_SHEET,
            "columns": SOURCE_COLUMNS,
            "note": LEGACY_NAME_NOTE,
        },
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_fit_model": selected_id,
        "selected_fit_formula": (
            "y0 for t<=t0; y0+(plateau-y0)*(1-exp(-((t-t0)/tau)^n)) for t>t0"
        ),
        "selected_fit_parameters": joint["parameters"],
        "selected_fit_parameter_std_error": joint["parameter_std_error"],
        "replicate_fit_parameters": replicates_json,
        "fit_metrics": {
            "selected_joint_equal_replicate": joint["metrics"],
            "candidate_comparison": comparison,
        },
        "filtering": {"rule": FILTER_RULE, "replicates": filtering_json},
        "reference_construction_method": REFERENCE_CONSTRUCTION,
        "threshold_times_s": joint["threshold_times_s"],
        "replicate_uncertainty": {
            "definition": "sample standard deviation of the two separate fitted replicate curves",
            "doc_reference_std": [float(value) for value in replicate_std],
        },
        "reference_time_range_s": [0.0, float(end_s)],
        "reference_dt_s": float(dt_s),
        "time_s": [float(value) for value in time_s],
        "doc_reference": [float(value) for value in doc_reference],
    }


def write_reference_export(document: dict[str, object], path: Path | str) -> Path:
    """Write a deterministic, human-readable JSON artifact."""

    path = Path(path)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
