import contextlib
import io
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest


def _cell_source(notebook, cell_id):
    for cell in notebook.cells:
        if cell.get("id") == cell_id:
            return cell.source
    pytest.fail(f"Notebook cell {cell_id!r} is missing")


def test_ftir_notebook_creates_fluorescence_and_doc_figures():
    """Catch a missing channel figure or a mislabeled two-panel result."""
    notebook = nbformat.read("FTIRvsDLP_What is_the_relation.ipynb", as_version=4)
    dataset_source = _cell_source(notebook, "ftir-doc-fluo-20261001")
    figure_source = _cell_source(notebook, "ftir-doc-fluo-figures-20261001")
    scope = {"__name__": "__main__"}

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        exec(dataset_source, scope)
        exec(figure_source, scope)

    expected_titles = {
        "fig_ftir_fluo_20261001": [
            "FTIR Fluorescence: Rate vs Intensity",
            "FTIR Fluorescence: Inhibition Time vs Intensity",
        ],
        "fig_ftir_doc_20261001": [
            "FTIR DoC: Rate vs Intensity",
            "FTIR DoC: Inhibition Time vs Intensity",
        ],
    }
    for figure_name, titles in expected_titles.items():
        figure = scope[figure_name]
        assert len(figure.axes) == 2
        assert [axis.get_title() for axis in figure.axes] == titles
        assert [len(axis.lines) for axis in figure.axes] == [6, 6]
        assert all(axis.get_xscale() == "log" and axis.get_yscale() == "log" for axis in figure.axes)
        assert all(axis.get_legend() is not None for axis in figure.axes)

    synthetic = {
        "test": {
            "intensity": np.array([1.0, 2.0]),
            "B_Test": np.array([1.0, 2.0]),
            "B_K_Test": 1.0,
            "B_N_Test": 1.0,
            "C_Test": np.array([np.nan, np.nan]),
            "C_K_Test": np.nan,
            "C_N_Test": np.nan,
            "color": "black",
        }
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        _, empty_metric_axes = scope["plot_ftir_kinetics_20261001"](
            synthetic, "Test", "Synthetic"
        )
    assert len(empty_metric_axes[0].lines) == 2
    assert len(empty_metric_axes[1].lines) == 0
    assert empty_metric_axes[1].get_xlabel() == "Exposure Intensity (mW/cm$^2$)"
    assert empty_metric_axes[1].get_ylabel() == "Inhibition Time, C (s)"
    assert empty_metric_axes[1].get_xgridlines()[0].get_linestyle() == ":"
    assert empty_metric_axes[1].get_xgridlines()[0].get_alpha() == pytest.approx(0.4)

    plt.close("all")


def test_ftir_legends_show_complete_power_law_functions_for_each_concentration():
    """Catch legends that omit K or n from any B(I) or C(I) fit."""
    notebook = nbformat.read("FTIRvsDLP_What is_the_relation.ipynb", as_version=4)
    scope = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        exec(_cell_source(notebook, "ftir-doc-fluo-20261001"), scope)
        exec(_cell_source(notebook, "ftir-doc-fluo-figures-20261001"), scope)

    figure_specs = [
        ("fig_ftir_fluo_20261001", "Fluo_datasets_20261001", "Fluo"),
        ("fig_ftir_doc_20261001", "DoC_datasets_20261001", "DoC"),
    ]
    for figure_name, datasets_name, channel_key in figure_specs:
        figure = scope[figure_name]
        datasets = scope[datasets_name]
        assert set(datasets) == {"0mM TEMPO", "1mM TEMPO", "3mM TEMPO"}
        for axis, symbol in zip(figure.axes, ["B", "C"]):
            fit_labels = [
                line.get_label() for line in axis.lines if line.get_linestyle() == "--"
            ]
            pattern = re.compile(
                rf"^.+: \${symbol}\(I\)=[0-9.e+-]+I\^\{{-?[0-9.]+\}}\$, "
                rf"\$R\^2\$=-?[0-9.]+$"
            )
            assert len(fit_labels) == 3
            assert all(pattern.match(label) for label in fit_labels)
            for concentration, data in datasets.items():
                coefficient = data[f"{symbol}_K_{channel_key}"]
                exponent = data[f"{symbol}_N_{channel_key}"]
                equation = (
                    f"{concentration}: ${symbol}(I)={coefficient:.3g}"
                    f"I^{{{exponent:.2f}}}$, $R^2$="
                )
                assert sum(label.startswith(equation) for label in fit_labels) == 1

    plt.close("all")
