"""Saturation-aware fitting/export for experimental MPC DoC references.

The experimental curves are desired control trajectories only.  They never
replace the authoritative AIE forward dynamics or its reaction-progress state.
Raw workbook labels are retained verbatim for provenance.  The workbook's
``5mM`` labels are the scientifically correct 30 mW/cm^2, 5 mM condition.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, isotonic_regression


SOURCE_NOTEBOOK = "DoC curve.ipynb"
SOURCE_DATA_FILE = "260728_circles/data_exports/intensity_data_processed.xlsx"
SOURCE_SHEET = "30mW"
REFERENCE_SCHEMA_VERSION = 2
REFERENCE_ARTIFACT_ID = "doc_reference_curves_30mW_v2"
REFERENCE_DT_S = 0.05
REFERENCE_END_S = 20.0
ISOTONIC_COMMON_DT_S = 0.01
SATURATION_THRESHOLD = 0.98
SATURATION_CONSECUTIVE_SAMPLES = 4
SATURATION_RULE_ID = "sustained_near_unity_v1"
PRODUCTION_REFERENCE_METHOD = "equal_replicate_isotonic_linear_interp_v1"
PRODUCTION_MODEL_ID = "isotonic_monotonic_benchmark"
COMPACT_DIAGNOSTIC_MODEL_ID = "delayed_avrami_fixed_plateau"
PRODUCTION_INTERPOLATION_METHOD = "piecewise_linear"
SATURATION_RULE = (
    "Normalize each finite CenterROI trace by its own raw maximum. Saturation is "
    "confirmed at the fourth sample in the first run of four consecutive samples "
    "with normalized signal >= 0.98. Retain data through that confirmation sample "
    "for fitting; mark every later sample excluded_from_DoC_fit with reason "
    "late_post_saturation_signal_drift. The falling tail is preserved in the raw "
    "workbook and plots but is not interpreted as decreasing conversion."
)
REFERENCE_CONSTRUCTION = (
    "Interpolate each retained normalized replicate onto the same uniform 0.01 s "
    "absolute-time grid. Hold a replicate at DoC=1 after its own sustained-saturation "
    "confirmation, average T1 and T2 pointwise with weights 0.5/0.5, and apply "
    "increasing isotonic regression to that equal-replicate mean. Resample the "
    "monotonic isotonic result to the authoritative 0.05 s runtime grid with "
    "piecewise-linear interpolation, then set DoC exactly to 1 from the production "
    "saturation time. Compact nonlinear fits remain diagnostic only."
)


CONDITION_SPECS: dict[str, dict[str, object]] = {
    "30mW_0mM": {
        "condition_label": "30 mW/cm^2, 0 mM TEMPO",
        "intensity_mw_cm2": 30.0,
        "tempo_concentration_mM": 0.0,
        "replicates": {
            "T1": {
                "raw_replicate_label": "T3",
                "time_column": "30mW_0mM_TEMPO_T3_Time_s",
                "signal_column": "30mW_0mM_TEMPO_T3_CenterROI_Intensity",
            },
            "T2": {
                "raw_replicate_label": "T4",
                "time_column": "30mW_0mM_TEMPO_T4_Time_s",
                "signal_column": "30mW_0mM_TEMPO_T4_CenterROI_Intensity",
            },
        },
        "legacy_label_notes": [
            "The newest notebook-selected replicate pair is stored as raw T3/T4; "
            "production diagnostics call this pair T1/T2 while retaining T3/T4 here."
        ],
    },
    "30mW_5mM": {
        "condition_label": "30 mW/cm^2, 5 mM TEMPO",
        "intensity_mw_cm2": 30.0,
        "tempo_concentration_mM": 5.0,
        "replicates": {
            "T1": {
                "raw_replicate_label": "T2",
                "time_column": "30mW_5mM_TEMPO_T2_Time_s",
                "signal_column": "30mW_5mM_TEMPO_T2_CenterROI_Intensity",
            },
            "T2": {
                "raw_replicate_label": "T3",
                "time_column": "30mW_5mM_TEMPO_T3_Time_s",
                "signal_column": "30mW_5mM_TEMPO_T3_CenterROI_Intensity",
            },
        },
        "legacy_label_notes": [
            "The raw workbook sheet/columns and collaborator-confirmed scientific "
            "metadata agree: this is 30 mW/cm^2 and 5 mM TEMPO. Raw T2/T3 "
            "replicate names are retained verbatim."
        ],
    },
}


@dataclass(frozen=True)
class ReplicateData:
    condition_id: str
    label: str
    raw_replicate_label: str
    time_column: str
    signal_column: str
    time_s: np.ndarray
    raw_signal: np.ndarray
    normalized_signal: np.ndarray
    fit_mask: np.ndarray
    saturation_window_start_index: int
    saturation_confirmation_index: int

    @property
    def saturation_time_s(self) -> float:
        return float(self.time_s[self.saturation_confirmation_index])

    @property
    def retained_sample_count(self) -> int:
        return int(np.count_nonzero(self.fit_mask))

    @property
    def excluded_sample_count(self) -> int:
        return int(np.count_nonzero(~self.fit_mask))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    parameter_names: tuple[str, ...]
    function: Callable[..., np.ndarray]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    initial_guess: tuple[float, ...]
    model_type: str = "parametric"


def delayed_avrami_fixed_plateau(
    time_s: np.ndarray, y0: float, t0_s: float, tau_s: float, shape_n: float
) -> np.ndarray:
    """Delayed Avrami/Weibull curve with plateau fixed to one."""

    time_s = np.asarray(time_s, dtype=float)
    elapsed = np.maximum(time_s - t0_s, 0.0)
    rise = y0 + (1.0 - y0) * (
        1.0 - np.exp(-np.power(elapsed / tau_s, shape_n))
    )
    return np.where(time_s <= t0_s, y0, rise)


def gompertz_fixed_plateau(
    time_s: np.ndarray, y0: float, midpoint_s: float, scale_s: float
) -> np.ndarray:
    """Monotonic Gompertz curve with plateau fixed to one."""

    z = np.clip(-(np.asarray(time_s, dtype=float) - midpoint_s) / scale_s, -700, 700)
    return y0 + (1.0 - y0) * np.exp(-np.exp(z))


def richards_fixed_plateau(
    time_s: np.ndarray,
    y0: float,
    midpoint_s: float,
    scale_s: float,
    shape_nu: float,
) -> np.ndarray:
    """Richards/generalized logistic curve with plateau fixed to one."""

    z = np.clip(-(np.asarray(time_s, dtype=float) - midpoint_s) / scale_s, -700, 700)
    return y0 + (1.0 - y0) * np.power(1.0 + np.exp(z), -1.0 / shape_nu)


def old_delayed_exponential(
    time_s: np.ndarray, a: float, b_per_s: float, c_s: float
) -> np.ndarray:
    """Legacy notebook model, retained only for quantitative comparison."""

    exponent = np.minimum(b_per_s * (c_s - np.asarray(time_s, dtype=float)), 0.0)
    return 1.0 - a * np.exp(exponent)


MODEL_SPECS = (
    ModelSpec(
        "old_delayed_exponential",
        ("a", "b_per_s", "c_s"),
        old_delayed_exponential,
        (0.0, 1e-4, 0.0),
        (1.2, 10.0, 15.0),
        (0.98, 0.7, 1.0),
    ),
    ModelSpec(
        "delayed_avrami_fixed_plateau",
        ("y0", "t0_s", "tau_s", "shape_n"),
        delayed_avrami_fixed_plateau,
        (0.0, 0.0, 0.05, 0.2),
        (0.05, 12.0, 20.0, 10.0),
        (0.01, 0.5, 3.0, 2.0),
    ),
    ModelSpec(
        "gompertz_fixed_plateau",
        ("y0", "midpoint_s", "scale_s"),
        gompertz_fixed_plateau,
        (0.0, 0.0, 0.05),
        (0.05, 15.0, 20.0),
        (0.01, 3.0, 1.0),
    ),
    ModelSpec(
        "richards_fixed_plateau",
        ("y0", "midpoint_s", "scale_s", "shape_nu"),
        richards_fixed_plateau,
        (0.0, 0.0, 0.05, 0.1),
        (0.05, 15.0, 20.0, 10.0),
        (0.01, 3.0, 1.0, 1.0),
    ),
)
def _finite_trace(time_s: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(time_s) & np.isfinite(signal)
    time_s = np.asarray(time_s[finite], dtype=float)
    signal = np.asarray(signal[finite], dtype=float)
    order = np.argsort(time_s, kind="stable")
    time_s = time_s[order]
    signal = signal[order]
    if time_s.size < SATURATION_CONSECUTIVE_SAMPLES:
        raise ValueError("experimental trace is too short for saturation detection")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("experimental trace times must be strictly increasing")
    return time_s, signal


def detect_sustained_saturation(
    normalized_signal: np.ndarray,
    *,
    threshold: float = SATURATION_THRESHOLD,
    consecutive_samples: int = SATURATION_CONSECUTIVE_SAMPLES,
) -> tuple[np.ndarray, int, int]:
    """Return fit mask, sustained-window start, and confirmation indices."""

    normalized_signal = np.asarray(normalized_signal, dtype=float)
    hits = normalized_signal >= threshold
    run_counts = np.convolve(
        hits.astype(np.int64), np.ones(consecutive_samples, dtype=np.int64), mode="valid"
    )
    candidates = np.flatnonzero(run_counts == consecutive_samples)
    if candidates.size == 0:
        raise ValueError(
            f"trace never sustains normalized signal >= {threshold:g} for "
            f"{consecutive_samples} consecutive samples"
        )
    start = int(candidates[0])
    confirmation = start + consecutive_samples - 1
    fit_mask = np.arange(normalized_signal.size) <= confirmation
    return fit_mask, start, confirmation


def load_condition_replicates(
    condition_id: str, repository_dir: Path | str = "."
) -> dict[str, ReplicateData]:
    """Load one collaborator-confirmed condition without changing raw names."""

    if condition_id not in CONDITION_SPECS:
        raise KeyError(
            f"unknown DoC condition {condition_id!r}; expected one of "
            f"{sorted(CONDITION_SPECS)}"
        )
    repository_dir = Path(repository_dir)
    frame = pd.read_excel(repository_dir / SOURCE_DATA_FILE, sheet_name=SOURCE_SHEET)
    condition = CONDITION_SPECS[condition_id]
    replicates: dict[str, ReplicateData] = {}
    for label, source in condition["replicates"].items():
        time_column = str(source["time_column"])
        signal_column = str(source["signal_column"])
        missing = [column for column in (time_column, signal_column) if column not in frame]
        if missing:
            raise KeyError(f"{condition_id}/{label} missing exact workbook columns: {missing}")
        time_s, raw_signal = _finite_trace(
            frame[time_column].to_numpy(dtype=float),
            frame[signal_column].to_numpy(dtype=float),
        )
        maximum = float(np.max(raw_signal))
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"{condition_id}/{label} has no positive normalization maximum")
        normalized = raw_signal / maximum
        fit_mask, start, confirmation = detect_sustained_saturation(normalized)
        replicates[label] = ReplicateData(
            condition_id=condition_id,
            label=label,
            raw_replicate_label=str(source["raw_replicate_label"]),
            time_column=time_column,
            signal_column=signal_column,
            time_s=time_s,
            raw_signal=raw_signal,
            normalized_signal=normalized,
            fit_mask=fit_mask,
            saturation_window_start_index=start,
            saturation_confirmation_index=confirmation,
        )
    return replicates


def _metrics(
    observed: np.ndarray, predicted: np.ndarray, parameter_count: int
) -> dict[str, float | int]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    sample_count = int(residual.size)
    sse = float(np.dot(residual, residual))
    rmse = math.sqrt(sse / sample_count)
    mae = float(np.mean(np.abs(residual)))
    centered = observed - float(np.mean(observed))
    sst = float(np.dot(centered, centered))
    r_squared = 1.0 - sse / sst if sst > np.finfo(float).eps else float("nan")
    safe_mse = max(sse / sample_count, np.finfo(float).tiny)
    return {
        "sample_count": sample_count,
        "parameter_count": int(parameter_count),
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "aic": sample_count * math.log(safe_mse) + 2 * parameter_count,
        "bic": sample_count * math.log(safe_mse)
        + parameter_count * math.log(sample_count),
    }


def _parameter_record(spec: ModelSpec, values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(spec.parameter_names, values)}


def _first_interpolated_crossing(
    time_s: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(np.asarray(values) >= threshold)
    if indices.size == 0:
        return None
    index = int(indices[0])
    if index == 0:
        return float(time_s[0])
    y0, y1 = float(values[index - 1]), float(values[index])
    t0, t1 = float(time_s[index - 1]), float(time_s[index])
    if math.isclose(y0, y1, abs_tol=1e-15):
        return t1
    return t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)


def threshold_times_from_arrays(
    time_s: np.ndarray, values: np.ndarray, *, include_t99: bool = True
) -> dict[str, float | None]:
    percents = (10, 30, 50, 90, 95, 99) if include_t99 else (10, 30, 50, 90, 95)
    return {
        f"t{percent}_s": _first_interpolated_crossing(time_s, values, percent / 100.0)
        for percent in percents
    }


def empirical_threshold_times(data: ReplicateData) -> dict[str, float | None]:
    return threshold_times_from_arrays(
        data.time_s[data.fit_mask],
        data.normalized_signal[data.fit_mask],
        include_t99=True,
    )


def _evaluate_model_thresholds(
    function: Callable[[np.ndarray], np.ndarray], saturation_time_s: float
) -> dict[str, float | None]:
    dense_time = np.linspace(0.0, REFERENCE_END_S, 20_001)
    values = np.asarray(function(dense_time), dtype=float)
    values = np.maximum.accumulate(np.clip(values, 0.0, 1.0))
    values[dense_time >= saturation_time_s] = 1.0
    return threshold_times_from_arrays(dense_time, values)


def fit_replicate(data: ReplicateData, spec: ModelSpec) -> dict[str, object]:
    time_s = data.time_s[data.fit_mask]
    observed = data.normalized_signal[data.fit_mask]
    parameters, covariance = curve_fit(
        spec.function,
        time_s,
        observed,
        p0=spec.initial_guess,
        bounds=(spec.lower_bounds, spec.upper_bounds),
        maxfev=200_000,
    )
    predicted = spec.function(time_s, *parameters)
    fitted_thresholds = _evaluate_model_thresholds(
        lambda query: spec.function(query, *parameters), data.saturation_time_s
    )
    raw_thresholds = empirical_threshold_times(data)
    threshold_errors = {
        key: (
            None
            if value is None or fitted_thresholds[key] is None
            else float(fitted_thresholds[key] - value)
        )
        for key, value in raw_thresholds.items()
    }
    diagonal = np.diag(covariance)
    return {
        "parameters": _parameter_record(spec, parameters),
        "parameter_vector": parameters,
        "parameter_std_error": {
            name: float(math.sqrt(max(value, 0.0)))
            for name, value in zip(spec.parameter_names, diagonal)
        },
        "metrics": _metrics(observed, predicted, len(parameters)),
        "raw_threshold_times_s": raw_thresholds,
        "fitted_threshold_times_s": fitted_thresholds,
        "threshold_time_errors_s": threshold_errors,
    }


def _production_saturation_time(replicates: dict[str, ReplicateData]) -> float:
    mean_time = float(np.mean([data.saturation_time_s for data in replicates.values()]))
    return math.ceil(mean_time / REFERENCE_DT_S - 1e-12) * REFERENCE_DT_S


def fit_joint_equal_replicate(
    replicates: dict[str, ReplicateData], spec: ModelSpec
) -> dict[str, object]:
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
        maxfev=200_000,
    )
    predicted = spec.function(all_time, *parameters)
    per_replicate = []
    for data in replicates.values():
        x = data.time_s[data.fit_mask]
        y = data.normalized_signal[data.fit_mask]
        per_replicate.append(_metrics(y, spec.function(x, *parameters), len(parameters)))
    metrics = _metrics(all_observed, predicted, len(parameters))
    metrics["equal_replicate_rmse"] = math.sqrt(
        float(np.mean([entry["rmse"] ** 2 for entry in per_replicate]))
    )
    metrics["equal_replicate_mae"] = float(
        np.mean([entry["mae"] for entry in per_replicate])
    )
    production_saturation = _production_saturation_time(replicates)
    fitted_thresholds = _evaluate_model_thresholds(
        lambda query: spec.function(query, *parameters), production_saturation
    )
    central_raw_thresholds: dict[str, float | None] = {}
    for key in ("t10_s", "t30_s", "t50_s", "t90_s", "t95_s", "t99_s"):
        values = [empirical_threshold_times(data)[key] for data in replicates.values()]
        central_raw_thresholds[key] = (
            None if any(value is None for value in values) else float(np.mean(values))
        )
    threshold_errors = {
        key: (
            None
            if value is None or fitted_thresholds[key] is None
            else float(fitted_thresholds[key] - value)
        )
        for key, value in central_raw_thresholds.items()
    }
    finite_errors = [abs(value) for value in threshold_errors.values() if value is not None]
    metrics["central_threshold_time_mae_s"] = float(np.mean(finite_errors))
    return {
        "parameters": _parameter_record(spec, parameters),
        "parameter_vector": parameters,
        "parameter_std_error": {
            name: float(math.sqrt(max(value, 0.0)))
            for name, value in zip(spec.parameter_names, np.diag(covariance))
        },
        "metrics": metrics,
        "raw_central_threshold_times_s": central_raw_thresholds,
        "fitted_threshold_times_s": fitted_thresholds,
        "threshold_time_errors_s": threshold_errors,
    }


def _equal_replicate_isotonic(
    replicates: dict[str, ReplicateData], production_saturation_time_s: float
) -> dict[str, object]:
    """Construct an equal-replicate monotonic benchmark on absolute time."""

    common_time_s = np.round(
        np.arange(
            0.0,
            production_saturation_time_s + 0.5 * ISOTONIC_COMMON_DT_S,
            ISOTONIC_COMMON_DT_S,
        ),
        10,
    )
    replicate_common_values: dict[str, np.ndarray] = {}
    for label, data in replicates.items():
        retained_time = data.time_s[data.fit_mask]
        retained_values = data.normalized_signal[data.fit_mask]
        interpolated = np.interp(
            common_time_s,
            retained_time,
            retained_values,
            left=retained_values[0],
            right=1.0,
        )
        interpolated[common_time_s > data.saturation_time_s + 1e-12] = 1.0
        replicate_common_values[label] = np.clip(interpolated, 0.0, 1.0)

    if set(replicate_common_values) != {"T1", "T2"}:
        raise ValueError("equal-replicate production construction requires T1 and T2")
    equal_replicate_mean = 0.5 * (
        replicate_common_values["T1"] + replicate_common_values["T2"]
    )
    result = isotonic_regression(equal_replicate_mean, increasing=True)
    fitted = np.maximum.accumulate(np.clip(result.x, 0.0, 1.0))
    block_count = int(len(result.blocks) - 1)
    if not np.isfinite(fitted).all():
        raise RuntimeError("equal-replicate isotonic fit contains NaN or Inf")
    if np.any(np.diff(fitted) < -1e-12):
        raise RuntimeError("equal-replicate isotonic fit is not monotonic")
    if float(np.min(fitted)) < 0.0 or float(np.max(fitted)) > 1.0:
        raise RuntimeError("equal-replicate isotonic fit left [0,1]")

    def predict(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        return np.interp(
            query,
            common_time_s,
            fitted,
            left=fitted[0],
            right=fitted[-1],
        )

    metrics = _metrics(equal_replicate_mean, fitted, block_count)
    fitted_thresholds = _evaluate_model_thresholds(predict, production_saturation_time_s)
    central_raw = {}
    for key in ("t10_s", "t30_s", "t50_s", "t90_s", "t95_s", "t99_s"):
        values = [empirical_threshold_times(data)[key] for data in replicates.values()]
        central_raw[key] = (
            None if any(value is None for value in values) else float(np.mean(values))
        )
    errors = {
        key: (
            None
            if value is None or fitted_thresholds[key] is None
            else float(fitted_thresholds[key] - value)
        )
        for key, value in central_raw.items()
    }
    metrics["central_threshold_time_mae_s"] = float(
        np.mean([abs(value) for value in errors.values() if value is not None])
    )
    return {
        "model_id": PRODUCTION_MODEL_ID,
        "model_type": "nonparametric_benchmark",
        "parameter_count": block_count,
        "metrics": metrics,
        "fitted_threshold_times_s": fitted_thresholds,
        "threshold_time_errors_s": errors,
        "predict": predict,
        "common_time_s": common_time_s,
        "replicate_common_values": replicate_common_values,
        "equal_replicate_mean": equal_replicate_mean,
        "isotonic_values": fitted,
        "isotonic_block_count": block_count,
    }


def _production_curve(
    predict: Callable[[np.ndarray], np.ndarray], saturation_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.round(
        np.arange(0.0, REFERENCE_END_S + 0.5 * REFERENCE_DT_S, REFERENCE_DT_S), 10
    )
    values = np.asarray(predict(time_s), dtype=float)
    values = np.maximum.accumulate(np.clip(values, 0.0, 1.0))
    values[time_s >= saturation_time_s - 1e-12] = 1.0
    values = np.clip(values, 0.0, 1.0)
    return time_s, values


def analyze_reference_condition(
    condition_id: str, repository_dir: Path | str = "."
) -> dict[str, object]:
    """Fit diagnostic models and build the isotonic production reference."""

    replicates = load_condition_replicates(condition_id, repository_dir)
    production_saturation = _production_saturation_time(replicates)
    candidates: dict[str, dict[str, object]] = {}
    for spec in MODEL_SPECS:
        replicate_fits = {
            label: fit_replicate(data, spec) for label, data in replicates.items()
        }
        joint_fit = fit_joint_equal_replicate(replicates, spec)
        replicate_errors = [
            abs(error)
            for result in replicate_fits.values()
            for error in result["threshold_time_errors_s"].values()
            if error is not None
        ]
        joint_fit["metrics"]["replicate_threshold_time_mae_s"] = float(
            np.mean(replicate_errors)
        )
        candidates[spec.model_id] = {
            "model_id": spec.model_id,
            "model_type": spec.model_type,
            "spec": spec,
            "replicate_fits": replicate_fits,
            "joint_fit": joint_fit,
            "predict": lambda query, spec=spec, parameters=joint_fit[
                "parameter_vector"
            ]: spec.function(query, *parameters),
        }
    candidates[PRODUCTION_MODEL_ID] = _equal_replicate_isotonic(
        replicates, production_saturation
    )

    compact = candidates[COMPACT_DIAGNOSTIC_MODEL_ID]
    old = candidates["old_delayed_exponential"]
    compact_rmse = float(compact["joint_fit"]["metrics"]["rmse"])
    old_rmse = float(old["joint_fit"]["metrics"]["rmse"])
    if not compact_rmse <= 0.90 * old_rmse:
        raise RuntimeError(
            f"{condition_id}: diagnostic Avrami fit did not improve old RMSE by 10% "
            f"({compact_rmse:.6g} vs {old_rmse:.6g})"
        )
    if compact["joint_fit"]["metrics"]["replicate_threshold_time_mae_s"] > 0.20:
        raise RuntimeError(f"{condition_id}: Avrami diagnostic timing is unacceptable")
    production = candidates[PRODUCTION_MODEL_ID]
    if production["metrics"]["central_threshold_time_mae_s"] > 0.20:
        raise RuntimeError(f"{condition_id}: isotonic production timing is unacceptable")
    time_s, doc_reference = _production_curve(
        production["predict"],
        production_saturation,
    )
    if np.any(np.diff(doc_reference) < -1e-12):
        raise RuntimeError(f"{condition_id}: exported isotonic reference is nonmonotonic")
    if float(np.min(doc_reference)) < 0.0 or float(np.max(doc_reference)) > 1.0:
        raise RuntimeError(f"{condition_id}: exported isotonic reference left [0,1]")
    return {
        "condition_id": condition_id,
        "condition_spec": CONDITION_SPECS[condition_id],
        "replicates": replicates,
        "candidates": candidates,
        "selected_model_id": PRODUCTION_MODEL_ID,
        "compact_diagnostic_model_id": COMPACT_DIAGNOSTIC_MODEL_ID,
        "production_reference_method": PRODUCTION_REFERENCE_METHOD,
        "production_saturation_time_s": production_saturation,
        "time_s": time_s,
        "doc_reference": doc_reference,
    }


def analyze_all_conditions(repository_dir: Path | str = ".") -> dict[str, object]:
    return {
        condition_id: analyze_reference_condition(condition_id, repository_dir)
        for condition_id in CONDITION_SPECS
    }


def model_comparison_table(analysis: dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_id, candidate in analysis["candidates"].items():
        metrics = (
            candidate["metrics"]
            if candidate["model_type"] == "nonparametric_benchmark"
            else candidate["joint_fit"]["metrics"]
        )
        rows.append(
            {
                "model": model_id,
                "model_type": candidate["model_type"],
                "parameter_count": metrics["parameter_count"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "R2": metrics["r_squared"],
                "AIC": metrics["aic"],
                "BIC": metrics["bic"],
                "threshold_MAE_s": metrics["central_threshold_time_mae_s"],
                "selected": model_id == analysis["selected_model_id"],
            }
        )
    return pd.DataFrame(rows).sort_values(["RMSE", "parameter_count"]).reset_index(
        drop=True
    )


def reference_comparison_table(analyses: dict[str, object]) -> pd.DataFrame:
    rows = []
    for condition_id, analysis in analyses.items():
        selected = analysis["candidates"][analysis["selected_model_id"]]
        selected_metrics = selected["metrics"]
        thresholds = threshold_times_from_arrays(
            analysis["time_s"], analysis["doc_reference"]
        )
        rows.append(
            {
                "reference_id": condition_id,
                "production_method": analysis["production_reference_method"],
                **thresholds,
                "saturation_time_s": analysis["production_saturation_time_s"],
                "rise_duration_t10_to_t90_s": thresholds["t90_s"]
                - thresholds["t10_s"],
                "fit_rmse": selected_metrics["rmse"],
                "fit_mae": selected_metrics["mae"],
            }
        )
    return pd.DataFrame(rows).set_index("reference_id")


def _json_fit_record(result: dict[str, object]) -> dict[str, object]:
    return {
        "parameters": result["parameters"],
        "parameter_std_error": result["parameter_std_error"],
        "metrics": result["metrics"],
        "raw_threshold_times_s": result["raw_threshold_times_s"],
        "fitted_threshold_times_s": result["fitted_threshold_times_s"],
        "threshold_time_errors_s": result["threshold_time_errors_s"],
    }


def _candidate_export_record(
    model_id: str, candidate: dict[str, object], selected_model_id: str
) -> dict[str, object]:
    metrics = (
        candidate["metrics"]
        if candidate["model_type"] == "nonparametric_benchmark"
        else candidate["joint_fit"]["metrics"]
    )
    record = {
        "model_id": model_id,
        "model_type": candidate["model_type"],
        "parameter_count": metrics["parameter_count"],
        "metrics": metrics,
        "selected": model_id == selected_model_id,
    }
    if candidate["model_type"] != "nonparametric_benchmark":
        record["joint_fit_parameters"] = candidate["joint_fit"]["parameters"]
        record["fitted_threshold_times_s"] = candidate["joint_fit"][
            "fitted_threshold_times_s"
        ]
        record["threshold_time_errors_s"] = candidate["joint_fit"][
            "threshold_time_errors_s"
        ]
    else:
        record["fitted_threshold_times_s"] = candidate["fitted_threshold_times_s"]
        record["threshold_time_errors_s"] = candidate["threshold_time_errors_s"]
    return record


def build_reference_export(
    analyses: dict[str, object], repository_dir: Path | str = "."
) -> dict[str, object]:
    repository_dir = Path(repository_dir)
    source_path = repository_dir / SOURCE_DATA_FILE
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    references: dict[str, object] = {}
    for condition_id, analysis in analyses.items():
        condition = analysis["condition_spec"]
        production = analysis["candidates"][analysis["selected_model_id"]]
        compact = analysis["candidates"][analysis["compact_diagnostic_model_id"]]
        compact_joint = compact["joint_fit"]
        replicate_records = {}
        for label, data in analysis["replicates"].items():
            fit = compact["replicate_fits"][label]
            differences = np.diff(data.time_s)
            peak_index = int(np.argmax(data.normalized_signal))
            sustained_below = np.convolve(
                (data.normalized_signal[peak_index:] < 0.95).astype(np.int64),
                np.ones(5, dtype=np.int64),
                mode="valid",
            )
            decline_candidates = np.flatnonzero(sustained_below == 5)
            decline_indicator_time_s = (
                None
                if decline_candidates.size == 0
                else float(data.time_s[peak_index + int(decline_candidates[0])])
            )
            replicate_records[label] = {
                "raw_replicate_label": data.raw_replicate_label,
                "time_column": data.time_column,
                "signal_column": data.signal_column,
                "normalization": "raw CenterROI signal divided by this replicate's raw maximum",
                "original_time_range_s": [float(data.time_s[0]), float(data.time_s[-1])],
                "sampling_interval_s": {
                    "minimum": float(np.min(differences)),
                    "median": float(np.median(differences)),
                    "maximum": float(np.max(differences)),
                },
                "raw_sample_count": int(data.time_s.size),
                "raw_normalized_start": float(data.normalized_signal[0]),
                "raw_normalized_peak": float(np.max(data.normalized_signal)),
                "raw_normalized_final": float(data.normalized_signal[-1]),
                "raw_normalized_at_saturation_confirmation": float(
                    data.normalized_signal[data.saturation_confirmation_index]
                ),
                "retained_fit_sample_count": data.retained_sample_count,
                "excluded_post_saturation_sample_count": data.excluded_sample_count,
                "first_excluded_time_s": (
                    None
                    if data.saturation_confirmation_index + 1 >= data.time_s.size
                    else float(data.time_s[data.saturation_confirmation_index + 1])
                ),
                "late_decline_indicator": {
                    "diagnostic_rule": (
                        "first sample in the first post-maximum run of five "
                        "consecutive normalized samples below 0.95"
                    ),
                    "time_s": decline_indicator_time_s,
                },
                "saturation_window_start_time_s": float(
                    data.time_s[data.saturation_window_start_index]
                ),
                "saturation_confirmation_time_s": data.saturation_time_s,
                "excluded_reason": "late_post_saturation_signal_drift",
                "fit": _json_fit_record(fit),
                "fit_role": "compact_Avrami_diagnostic_only_not_MPC_reference",
            }
        common_replicates = production["replicate_common_values"]
        spread = np.std(
            np.vstack([common_replicates["T1"], common_replicates["T2"]]),
            axis=0,
            ddof=1,
        )
        threshold_times = threshold_times_from_arrays(
            analysis["time_s"], analysis["doc_reference"]
        )
        threshold_comparison = {}
        for label, data in analysis["replicates"].items():
            raw_thresholds = empirical_threshold_times(data)
            threshold_comparison[label] = {
                "raw_threshold_times_s": raw_thresholds,
                "production_threshold_times_s": threshold_times,
                "production_minus_raw_errors_s": {
                    key: (
                        None
                        if raw_value is None or threshold_times[key] is None
                        else float(threshold_times[key] - raw_value)
                    )
                    for key, raw_value in raw_thresholds.items()
                },
            }
        references[condition_id] = {
            "reference_id": condition_id,
            "condition": {
                "intensity_mw_cm2": condition["intensity_mw_cm2"],
                "tempo_concentration_mM": condition["tempo_concentration_mM"],
            },
            "condition_label": condition["condition_label"],
            "source_notebook": SOURCE_NOTEBOOK,
            "source_data_file": SOURCE_DATA_FILE,
            "source_data_sha256": source_sha256,
            "source_sheet": SOURCE_SHEET,
            "raw_source_columns": {
                label: {
                    "raw_replicate_label": data.raw_replicate_label,
                    "time": data.time_column,
                    "signal": data.signal_column,
                }
                for label, data in analysis["replicates"].items()
            },
            "legacy_label_notes": condition["legacy_label_notes"],
            "production_reference_method": analysis["production_reference_method"],
            "selected_fit_model": analysis["selected_model_id"],
            "selected_fit_formula": (
                "increasing isotonic regression of the pointwise 0.5*T1+0.5*T2 "
                "common-grid trajectory; piecewise-linear interpolation to 0.05 s; "
                "exactly 1 from production saturation"
            ),
            "selected_fit_parameters": {
                "increasing": True,
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "isotonic_block_count": production["isotonic_block_count"],
                "common_grid_dt_s": ISOTONIC_COMMON_DT_S,
            },
            "selected_fit_parameter_std_error": {},
            "fit_metrics": production["metrics"],
            "compact_diagnostic_fit": {
                "model_id": analysis["compact_diagnostic_model_id"],
                "role": "diagnostic_only_not_MPC_reference",
                "parameters": compact_joint["parameters"],
                "parameter_std_error": compact_joint["parameter_std_error"],
                "metrics": compact_joint["metrics"],
                "fitted_threshold_times_s": compact_joint[
                    "fitted_threshold_times_s"
                ],
            },
            "model_comparison": {
                model_id: _candidate_export_record(
                    model_id, candidate, analysis["selected_model_id"]
                )
                for model_id, candidate in analysis["candidates"].items()
            },
            "replicate_fits": replicate_records,
            "saturation_detection_rule": {
                "rule_id": SATURATION_RULE_ID,
                "normalized_threshold": SATURATION_THRESHOLD,
                "consecutive_samples": SATURATION_CONSECUTIVE_SAMPLES,
                "description": SATURATION_RULE,
            },
            "saturation_times_s": {
                **{
                    label: data.saturation_time_s
                    for label, data in analysis["replicates"].items()
                },
                "production": analysis["production_saturation_time_s"],
            },
            "post_saturation_reference_behavior": "doc_reference is exactly 1.0",
            "production_reference_construction_method": REFERENCE_CONSTRUCTION,
            "equal_replicate_construction": {
                "common_absolute_time_grid_start_s": 0.0,
                "common_absolute_time_grid_end_s": analysis[
                    "production_saturation_time_s"
                ],
                "common_absolute_time_grid_dt_s": ISOTONIC_COMMON_DT_S,
                "common_absolute_time_grid_point_count": int(
                    production["common_time_s"].size
                ),
                "replicate_weights": {"T1": 0.5, "T2": 0.5},
                "replicate_interpolation": "piecewise_linear_absolute_time",
                "post_replicate_saturation_behavior": (
                    "replicate DoC held exactly at 1 after its own sustained "
                    "saturation confirmation"
                ),
                "pointwise_combination": "0.5*T1 + 0.5*T2",
            },
            "isotonic_regression": {
                "increasing": True,
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "block_count": production["isotonic_block_count"],
            },
            "runtime_grid_interpolation_method": PRODUCTION_INTERPOLATION_METHOD,
            "threshold_times_s": threshold_times,
            "production_threshold_comparison_s": threshold_comparison,
            "replicate_spread": {
                "method": (
                    "pointwise standard deviation of common-grid interpolated "
                    "T1/T2 trajectories before equal-replicate averaging"
                ),
                "mean_std_doc": float(np.mean(spread)),
                "max_std_doc": float(np.max(spread)),
            },
            "reference_time_range_s": [0.0, REFERENCE_END_S],
            "dt_s": REFERENCE_DT_S,
            "time_s": [float(value) for value in analysis["time_s"]],
            "doc_reference": [float(value) for value in analysis["doc_reference"]],
        }
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "artifact_id": REFERENCE_ARTIFACT_ID,
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_notebook": SOURCE_NOTEBOOK,
        "source_data_file": SOURCE_DATA_FILE,
        "source_data_sha256": source_sha256,
        "authoritative_reference_dt_s": REFERENCE_DT_S,
        "references": references,
    }


def write_reference_export(
    document: dict[str, object], path: Path | str = "doc_reference_curves.json"
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def plot_condition_diagnostics(analysis: dict[str, object]):
    import matplotlib.pyplot as plt

    condition = analysis["condition_spec"]
    production = analysis["candidates"][analysis["selected_model_id"]]
    compact = analysis["candidates"][analysis["compact_diagnostic_model_id"]]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    colors = {"T1": "tab:blue", "T2": "tab:orange"}
    for label, data in analysis["replicates"].items():
        color = colors[label]
        axes[0].plot(
            data.time_s,
            data.normalized_signal,
            color=color,
            alpha=0.25,
            linewidth=1,
            label=f"{label} raw (workbook {data.raw_replicate_label})",
        )
        axes[0].scatter(
            data.time_s[data.fit_mask],
            data.normalized_signal[data.fit_mask],
            color=color,
            s=14,
            label=f"{label} retained",
        )
        axes[0].scatter(
            data.time_s[~data.fit_mask],
            data.normalized_signal[~data.fit_mask],
            facecolors="none",
            color=color,
            marker="x",
            s=20,
            alpha=0.75,
            label=f"{label} excluded drift",
        )
    compact_curve = np.asarray(compact["predict"](analysis["time_s"]), dtype=float)
    compact_curve = np.maximum.accumulate(np.clip(compact_curve, 0.0, 1.0))
    compact_curve[
        analysis["time_s"] >= analysis["production_saturation_time_s"] - 1e-12
    ] = 1.0
    axes[0].plot(
        analysis["time_s"],
        compact_curve,
        color="0.35",
        linewidth=2,
        linestyle="--",
        label="compact Avrami diagnostic (not MPC)",
    )
    axes[0].plot(
        production["common_time_s"],
        production["equal_replicate_mean"],
        color="tab:cyan",
        linewidth=1.2,
        alpha=0.8,
        label="equal-replicate measured mean",
    )
    axes[0].plot(
        production["common_time_s"],
        production["isotonic_values"],
        color="tab:purple",
        linewidth=2,
        linestyle=":",
        label="equal-replicate isotonic benchmark",
    )
    axes[0].plot(
        analysis["time_s"], analysis["doc_reference"], color="black", linewidth=2.8,
        label="FINAL MPC production: isotonic + linear interpolation",
    )
    axes[0].axhline(1.0, color="0.35", linewidth=1, linestyle=":")
    thresholds = threshold_times_from_arrays(analysis["time_s"], analysis["doc_reference"])
    for key, color in zip(
        ("t10_s", "t30_s", "t50_s", "t90_s"),
        ("#9467bd", "#8c564b", "#e377c2", "#7f7f7f"),
    ):
        axes[0].axvline(thresholds[key], color=color, linewidth=0.9, alpha=0.7)
    axes[0].axvline(
        analysis["production_saturation_time_s"], color="black", linestyle=":",
        linewidth=1.5, label="production t_sat",
    )
    axes[0].set_title("Experimental data and MPC production reference")

    styles = {
        "old_delayed_exponential": ("0.45", ":"),
        "delayed_avrami_fixed_plateau": ("tab:blue", "--"),
        "gompertz_fixed_plateau": ("tab:green", "--"),
        "richards_fixed_plateau": ("tab:red", "-."),
        "isotonic_monotonic_benchmark": ("tab:purple", (0, (3, 1, 1, 1))),
    }
    plot_time = np.linspace(0.0, REFERENCE_END_S, 1001)
    for model_id, candidate in analysis["candidates"].items():
        curve = np.asarray(candidate["predict"](plot_time), dtype=float)
        curve = np.maximum.accumulate(np.clip(curve, 0.0, 1.0))
        curve[plot_time >= analysis["production_saturation_time_s"]] = 1.0
        color, linestyle = styles[model_id]
        axes[1].plot(
            plot_time, curve, color=color, linestyle=linestyle,
            linewidth=2.5 if model_id == analysis["selected_model_id"] else 1.5,
            label=model_id.replace("_", " "),
        )
    axes[1].plot(
        analysis["time_s"],
        analysis["doc_reference"],
        color="black",
        linewidth=3.0,
        label="FINAL MPC isotonic reference",
    )
    axes[1].axhline(1.0, color="0.35", linewidth=1, linestyle=":")
    axes[1].set_title("Diagnostic model comparison (production is isotonic)")
    for axis in axes:
        axis.set_xlim(0.0, REFERENCE_END_S)
        axis.set_ylim(-0.02, 1.03)
        axis.set_xlabel("Absolute time since exposure start (s)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Normalized signal / DoC reference")
    figure.suptitle(condition["condition_label"].replace("cm^2", r"cm$^2$"))
    figure.tight_layout()
    return figure


def plot_reference_comparison(analyses: dict[str, object]):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6))
    for condition_id, analysis in analyses.items():
        axis.plot(
            analysis["time_s"], analysis["doc_reference"], linewidth=2.8,
            label=analysis["condition_spec"]["condition_label"].replace(
                "cm^2", r"cm$^2$"
            ),
        )
    axis.axhline(1.0, color="0.35", linestyle=":", linewidth=1)
    axis.set_xlim(0.0, REFERENCE_END_S)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("Absolute time since exposure start (s)")
    axis.set_ylabel("DoC reference")
    axis.set_title(r"Experimental DoC references at 30 mW/cm$^2$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def save_diagnostic_figures(
    analyses: dict[str, object], repository_dir: Path | str = "."
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    repository_dir = Path(repository_dir)
    paths: dict[str, Path] = {}
    for condition_id, analysis in analyses.items():
        path = repository_dir / f"doc_reference_{condition_id}_diagnostics.png"
        figure = plot_condition_diagnostics(analysis)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths[condition_id] = path
    comparison_path = repository_dir / "doc_reference_comparison_30mW.png"
    figure = plot_reference_comparison(analyses)
    figure.savefig(comparison_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    paths["comparison"] = comparison_path
    return paths


def main() -> None:
    repository_dir = Path(__file__).resolve().parent
    analyses = analyze_all_conditions(repository_dir)
    document = build_reference_export(analyses, repository_dir)
    export_path = write_reference_export(
        document, repository_dir / "doc_reference_curves.json"
    )
    figure_paths = save_diagnostic_figures(analyses, repository_dir)
    for condition_id, analysis in analyses.items():
        print(f"\n=== {analysis['condition_spec']['condition_label']} ===")
        print(model_comparison_table(analysis).to_string(index=False))
    print("\n=== Reference comparison ===")
    print(reference_comparison_table(analyses).to_string())
    print(f"\nWrote {export_path}")
    for label, path in figure_paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
