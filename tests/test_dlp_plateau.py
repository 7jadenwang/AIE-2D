import numpy as np
import json
from pathlib import Path

import matplotlib

from dlp_plateau import trim_terminal_decline

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def test_trims_a_sustained_terminal_drop_after_the_last_valid_plateau_sample():
    """A terminal collapse must not be passed to the curve fit."""
    signal = np.array([10.0, 60.0, 100.0, 99.2, 98.5, 96.0, 94.0, 40.0])

    corrected = trim_terminal_decline(signal, tolerance=0.02, consecutive=3)

    np.testing.assert_allclose(corrected, [10.0, 60.0, 100.0, 99.2, 98.5])


def test_preserves_plateau_fluctuations_inside_the_tolerance_band():
    """Normal near-peak noise must remain visible in the processed trace."""
    signal = np.array([10.0, 60.0, 100.0, 99.4, 98.1, 99.0, 98.3])

    corrected = trim_terminal_decline(signal, tolerance=0.02, consecutive=3)

    np.testing.assert_allclose(corrected, signal)


def test_uses_a_top_range_median_instead_of_a_single_noisy_maximum():
    """One high fluctuation must not define where a valid trace is cut."""
    signal = np.array([10.0, 60.0, 100.0, 100.0, 100.0, 100.0, 130.0, 100.0, 96.0, 95.0, 94.0])

    corrected = trim_terminal_decline(signal, tolerance=0.02, consecutive=3)

    np.testing.assert_allclose(corrected, [10.0, 60.0, 100.0, 100.0, 100.0, 100.0, 130.0, 100.0])


def test_dlp_notebook_cell_executes_with_its_data_file():
    """The reusable DLP cell must import its correction helper itself."""
    notebook_path = Path("DoC curve.ipynb")
    with notebook_path.open(encoding="utf-8") as notebook_file:
        source = "".join(json.load(notebook_file)["cells"][2]["source"])

    original_show = plt.show
    plt.show = lambda: None
    namespace = {"__name__": "__main__"}
    try:
        exec(source, namespace)
    finally:
        plt.show = original_show

    assert namespace["signal1"].size == namespace["time1"].size
    assert namespace["signal1"].size < namespace["loc1"].shape[0]
    assert namespace["signal1"][-1] > 0.95 * namespace["signal1"].max()
