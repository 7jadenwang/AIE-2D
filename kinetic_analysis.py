"""Shared kinetic fitting for the 2026-10-01 DLP and FTIR datasets."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dlp_plateau import trim_terminal_decline


COLORS = {0: "blue", 1: "green", 3: "orange", 5: "red"}
MINIMUM_R_SQUARED = 0.90
MAX_NORMALIZED_AMPLITUDE = 2.0


def kinetic_curve(time, amplitude, rate, inhibition_time):
    """Saturating growth model used by the existing notebooks."""
    time = np.asarray(time, dtype=float)
    elapsed = np.maximum(time - inhibition_time, 0.0)
    return amplitude * (1.0 - np.exp(-rate * elapsed))


def fit_trace(time, signal, trim_decline=False):
    """Fit one trace after finite-value filtering and optional DLP trimming."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal) & (time >= 0)
    time, signal = time[valid], np.clip(signal[valid], 0.0, None)
    order = np.argsort(time)
    time, signal = time[order], signal[order]
    if trim_decline and signal.size:
        signal = trim_terminal_decline(signal)
        time = time[: signal.size]
    if signal.size < 4 or np.ptp(signal) <= 0 or signal.max() <= 0:
        raise ValueError("signal must contain at least four points with positive variation")
    normalized = signal / signal.max()
    onset_candidates = np.flatnonzero(normalized >= 0.02)
    onset_guess = float(time[onset_candidates[0]]) if onset_candidates.size else 0.0
    parameters, _ = curve_fit(
        kinetic_curve, time, normalized, p0=(1.0, 0.5, onset_guess),
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, float(time.max())]),
        maxfev=50000,
    )
    predicted = kinetic_curve(time, *parameters)
    residual = np.sum((normalized - predicted) ** 2)
    total = np.sum((normalized - normalized.mean()) ** 2)
    time_steps = np.diff(np.unique(time))
    time_resolution = float(np.median(time_steps[time_steps > 0]))
    return {
        "amplitude": float(parameters[0]), "rate": float(parameters[1]),
        "inhibition_time": float(parameters[2]),
        "r_squared": float(1.0 - residual / total), "n_points": int(time.size),
        "time_resolution": time_resolution,
    }


def _concentration_label(value):
    match = re.search(r"(\d+(?:\.\d+)?)\s*mM", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse TEMPO concentration from {value!r}")
    return f"{match.group(1)}mM TEMPO"


def _safe_fit(time, signal, trim_decline):
    try:
        fit = fit_trace(time, signal, trim_decline)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return {
            "amplitude": np.nan, "rate": np.nan, "inhibition_time": np.nan,
            "r_squared": np.nan, "n_points": 0, "time_resolution": np.nan,
            "fit_valid": False, "rate_valid": False,
            "inhibition_time_valid": False, "rejection_reason": f"fit_failed: {error}",
        }
    fit_valid = fit["r_squared"] >= MINIMUM_R_SQUARED
    rate_valid = fit_valid and fit["amplitude"] <= MAX_NORMALIZED_AMPLITUDE
    inhibition_valid = fit_valid and fit["inhibition_time"] >= fit["time_resolution"]
    reasons = []
    if not fit_valid:
        reasons.append("low_r_squared")
    if fit_valid and not rate_valid:
        reasons.append("rate_plateau_unresolved")
    if fit_valid and not inhibition_valid:
        reasons.append("inhibition_below_time_resolution")
    return {
        **fit, "fit_valid": fit_valid, "rate_valid": rate_valid,
        "inhibition_time_valid": inhibition_valid,
        "rejection_reason": "; ".join(reasons),
    }


def _mean_std_count(group, value_column, valid_column):
    values = group.loc[group[valid_column], value_column].to_numpy(dtype=float)
    if not values.size:
        return np.nan, np.nan, 0
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return float(np.mean(values)), standard_deviation, int(values.size)


def _summarize(fits):
    group_columns = ["concentration", "intensity_mw"]
    if "channel" in fits.columns:
        group_columns.append("channel")
    records = []
    for keys, group in fits.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rate_mean, rate_std, rate_count = _mean_std_count(group, "rate", "rate_valid")
        time_mean, time_std, time_count = _mean_std_count(
            group, "inhibition_time", "inhibition_time_valid"
        )
        fit_count = int(group.loc[group["fit_valid"], "replicate"].nunique())
        total = int(group["replicate"].nunique())
        valid_r_squared = group.loc[group["fit_valid"], "r_squared"]
        record = dict(zip(group_columns, keys))
        record.update({
            "rate_mean": rate_mean, "rate_std": rate_std,
            "inhibition_time_mean": time_mean, "inhibition_time_std": time_std,
            "r_squared_min": float(valid_r_squared.min()) if fit_count else np.nan,
            "n_replicates": fit_count, "n_rate_replicates": rate_count,
            "n_inhibition_replicates": time_count, "n_total": total,
            "n_rejected": total - fit_count,
        })
        records.append(record)
    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


def analyze_dlp_fluorescence(root):
    """Fit every retained DLP fluorescence trial below data/conditions."""
    files = sorted(Path(root).rglob("*_roi_timeseries.csv"))
    if not files:
        raise FileNotFoundError(f"No DLP ROI timeseries CSV files below {root}")
    records = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"concentration", "intensity_mw", "trial", "time_s", "center_mean_16bit"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for trial, trace in frame.groupby("trial", sort=True):
            fit = _safe_fit(trace["time_s"], trace["center_mean_16bit"], trim_decline=True)
            records.append({
                "concentration": _concentration_label(trace["concentration"].iloc[0]),
                "intensity_mw": float(trace["intensity_mw"].iloc[0]),
                "replicate": int(trial), "source_file": str(path), **fit,
            })
    fits = pd.DataFrame(records).sort_values(
        ["concentration", "intensity_mw", "replicate"]
    ).reset_index(drop=True)
    return fits, _summarize(fits)


def analyze_ftir_runs(root):
    """Fit fluorescence and FTIR DoC independently for every synchronized run."""
    files = sorted(Path(root).rglob("common_timescale.csv"))
    if not files:
        raise FileNotFoundError(f"No common_timescale.csv files below {root}")
    channels = {
        "DoC": ("ftir_smoothed_conversion_percent_interpolated", False),
        "Fluo": ("fluorescence_smoothed_8bit", True),
    }
    records = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"concentration", "intensity_mw", "run", "led_elapsed_s"}
        missing = required.union(name for name, _ in channels.values()).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for channel, (signal_column, trim_decline) in channels.items():
            fit = _safe_fit(frame["led_elapsed_s"], frame[signal_column], trim_decline)
            records.append({
                "concentration": _concentration_label(frame["concentration"].iloc[0]),
                "intensity_mw": float(frame["intensity_mw"].iloc[0]),
                "replicate": int(frame["run"].iloc[0]), "channel": channel,
                "source_file": str(path), **fit,
            })
    fits = pd.DataFrame(records).sort_values(
        ["concentration", "intensity_mw", "replicate", "channel"]
    ).reset_index(drop=True)
    return fits, _summarize(fits)


def _power_law(intensity, values):
    valid = np.isfinite(intensity) & np.isfinite(values) & (intensity > 0) & (values > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    exponent, log_coefficient = np.polyfit(
        np.log10(intensity[valid]), np.log10(values[valid]), 1
    )
    return float(10**log_coefficient), float(exponent)


def build_kinetic_datasets(summary, channel_label=None):
    """Build intensity-sorted dictionaries matching the comparison notebook."""
    datasets = {}
    for concentration, group in summary.groupby("concentration", sort=False):
        group = group.sort_values("intensity_mw")
        intensity = group["intensity_mw"].to_numpy(dtype=float)
        rate = group["rate_mean"].to_numpy(dtype=float)
        inhibition = group["inhibition_time_mean"].to_numpy(dtype=float)
        b_k, b_n = _power_law(intensity, rate)
        c_k, c_n = _power_law(intensity, inhibition)
        concentration_value = int(float(re.search(r"[\d.]+", concentration).group()))
        if channel_label:
            suffix = f"_{channel_label}"
            datasets[concentration] = {
                "intensity": intensity, f"B{suffix}": rate, f"C{suffix}": inhibition,
                f"B_K{suffix}": b_k, f"B_N{suffix}": b_n,
                f"C_K{suffix}": c_k, f"C_N{suffix}": c_n,
                "color": COLORS.get(concentration_value, "black"),
            }
        else:
            datasets[concentration] = {
                "intensity": intensity, "polymerization_rate": rate,
                "inhibition_time": inhibition, "B_K": b_k, "B_N": b_n,
                "C_K": c_k, "C_N": c_n,
                "color": COLORS.get(concentration_value, "black"),
            }
    return datasets
