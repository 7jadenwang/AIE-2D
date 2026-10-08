import importlib

import numpy as np
import pandas as pd
import pytest


def _analysis_module():
    """Import the requested feature while keeping the RED failure explicit."""
    try:
        return importlib.import_module("kinetic_analysis")
    except ModuleNotFoundError:
        pytest.fail("kinetic_analysis.py has not been implemented")


def _growth_trace(time, amplitude, rate, inhibition_time):
    elapsed = np.maximum(time - inhibition_time, 0.0)
    return amplitude * (1.0 - np.exp(-rate * elapsed))


def test_fit_trace_recovers_rate_and_inhibition_time():
    analysis = _analysis_module()
    time = np.linspace(0.0, 15.0, 151)
    signal = _growth_trace(time, 4.0, 0.45, 1.2)

    result = analysis.fit_trace(time, signal)

    assert result["rate"] == pytest.approx(0.45, rel=1e-3)
    assert result["inhibition_time"] == pytest.approx(1.2, abs=1e-3)
    assert result["r_squared"] > 0.999999


def test_fit_trace_rejects_a_zero_signal():
    analysis = _analysis_module()

    with pytest.raises(ValueError, match="positive variation"):
        analysis.fit_trace(np.arange(5.0), np.zeros(5))


def test_analyze_dlp_fluorescence_groups_retained_trials(tmp_path):
    analysis = _analysis_module()
    condition_dir = tmp_path / "0mMTEMPO" / "2mW"
    condition_dir.mkdir(parents=True)
    time = np.linspace(-1.0, 12.0, 131)
    frames = []
    for trial, amplitude in [(1, 100.0), (3, 140.0)]:
        signal = _growth_trace(time, amplitude, 0.5, 1.0)
        frames.append(pd.DataFrame({
            "concentration": "0mMTEMPO",
            "intensity_mw": 2,
            "trial": trial,
            "time_s": time,
            "center_mean_16bit": signal,
        }))
    frames.append(pd.DataFrame({
        "concentration": "0mMTEMPO",
        "intensity_mw": 2,
        "trial": 2,
        "time_s": time,
        "center_mean_16bit": 100.0 + np.sin(5.0 * time),
    }))
    frames.append(pd.DataFrame({
        "concentration": "0mMTEMPO",
        "intensity_mw": 2,
        "trial": 4,
        "time_s": time,
        "center_mean_16bit": _growth_trace(time, 100.0, 0.02, 1.0),
    }))
    pd.concat(frames).to_csv(
        condition_dir / "0mMTEMPO_2mW_roi_timeseries.csv", index=False
    )

    trial_fits, summary = analysis.analyze_dlp_fluorescence(tmp_path)

    assert trial_fits["replicate"].tolist() == [1, 2, 3, 4]
    assert trial_fits["fit_valid"].tolist() == [True, False, True, True]
    assert trial_fits["rate_valid"].tolist() == [True, False, True, False]
    assert trial_fits["inhibition_time_valid"].tolist() == [True, False, True, True]
    assert summary.loc[0, "concentration"] == "0mM TEMPO"
    assert summary.loc[0, "intensity_mw"] == 2
    assert summary.loc[0, "n_replicates"] == 3
    assert summary.loc[0, "n_rate_replicates"] == 2
    assert summary.loc[0, "n_inhibition_replicates"] == 3
    assert summary.loc[0, "n_rejected"] == 1
    assert summary.loc[0, "rate_mean"] == pytest.approx(0.5, rel=1e-3)
    assert summary.loc[0, "inhibition_time_mean"] == pytest.approx(1.0, abs=1e-3)


def test_analyze_ftir_runs_keeps_fluorescence_and_doc_separate(tmp_path):
    analysis = _analysis_module()
    data_dir = tmp_path / "1mM_TEMPO" / "5mW" / "r2" / "data"
    data_dir.mkdir(parents=True)
    time = np.linspace(-2.0, 12.0, 141)
    pd.DataFrame({
        "concentration": "1mM TEMPO",
        "intensity_mw": 5,
        "run": 2,
        "led_elapsed_s": time,
        "fluorescence_smoothed_8bit": _growth_trace(time, 80.0, 0.3, 2.0),
        "ftir_smoothed_conversion_percent_interpolated": _growth_trace(time, 95.0, 0.7, 0.0),
    }).to_csv(data_dir / "common_timescale.csv", index=False)

    run_fits, summary = analysis.analyze_ftir_runs(tmp_path)

    assert run_fits["channel"].tolist() == ["DoC", "Fluo"]
    assert len(summary) == 2
    doc = summary.loc[summary["channel"] == "DoC"].iloc[0]
    fluo = summary.loc[summary["channel"] == "Fluo"].iloc[0]
    assert doc["rate_mean"] == pytest.approx(0.7, rel=1e-3)
    assert np.isnan(doc["inhibition_time_mean"])
    assert doc["n_inhibition_replicates"] == 0
    assert fluo["rate_mean"] == pytest.approx(0.3, rel=1e-3)


def test_analyze_dlp_retains_an_all_invalid_condition(tmp_path):
    analysis = _analysis_module()
    condition_dir = tmp_path / "1mMTEMPO" / "3mW"
    condition_dir.mkdir(parents=True)
    pd.DataFrame({
        "concentration": "1mMTEMPO",
        "intensity_mw": 3,
        "trial": 1,
        "time_s": np.arange(5.0),
        "center_mean_16bit": np.zeros(5),
    }).to_csv(condition_dir / "1mMTEMPO_3mW_roi_timeseries.csv", index=False)

    trial_fits, summary = analysis.analyze_dlp_fluorescence(tmp_path)

    assert trial_fits.loc[0, "fit_valid"] == False  # noqa: E712
    assert trial_fits.loc[0, "rejection_reason"].startswith("fit_failed:")
    assert len(summary) == 1
    assert summary.loc[0, "n_total"] == 1
    assert summary.loc[0, "n_rejected"] == 1
    assert summary.loc[0, "n_rate_replicates"] == 0
    assert np.isnan(summary.loc[0, "rate_mean"])


def test_build_kinetic_datasets_sorts_intensity_and_fits_power_laws():
    analysis = _analysis_module()
    summary = pd.DataFrame({
        "concentration": ["0mM TEMPO", "0mM TEMPO"],
        "intensity_mw": [4, 2],
        "rate_mean": [2.0, 1.0],
        "inhibition_time_mean": [2.0, 4.0],
    })

    datasets = analysis.build_kinetic_datasets(summary, channel_label="Fluo")
    data = datasets["0mM TEMPO"]

    assert data["intensity"].tolist() == [2.0, 4.0]
    assert data["B_Fluo"].tolist() == [1.0, 2.0]
    assert data["C_Fluo"].tolist() == [4.0, 2.0]
    assert data["B_K_Fluo"] == pytest.approx(0.5)
    assert data["B_N_Fluo"] == pytest.approx(1.0)
    assert data["C_K_Fluo"] == pytest.approx(8.0)
    assert data["C_N_Fluo"] == pytest.approx(-1.0)
