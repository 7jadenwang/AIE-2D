import contextlib
import io
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


def _cell_source(notebook, cell_id):
    for cell in notebook.cells:
        if cell.get("id") == cell_id:
            return cell.source
    raise AssertionError(f"Notebook cell {cell_id!r} is missing")


def test_dlp_legends_show_complete_power_law_functions_for_each_concentration():
    """Catch legends that omit K or n from any DLP B(I) or C(I) fit."""
    notebook = nbformat.read("Absorption-intensity fit.ipynb", as_version=4)
    scope = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        exec(_cell_source(notebook, "dlp-power-law-20261001"), scope)

    datasets = scope["DLP_Fluo_datasets_20261001"]
    assert set(datasets) == {
        "0mM TEMPO", "1mM TEMPO", "3mM TEMPO", "5mM TEMPO"
    }
    figure = scope["fig"]
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "DLP Fluorescence: Rate vs Intensity",
        "DLP Fluorescence: Inhibition Time vs Intensity",
    ]
    assert all(
        axis.get_xscale() == "log" and axis.get_yscale() == "log"
        for axis in figure.axes
    )
    for axis, symbol in zip(figure.axes, ["B", "C"]):
        fit_labels = [
            line.get_label() for line in axis.lines if line.get_linestyle() == "--"
        ]
        pattern = re.compile(
            rf"^.+: \${symbol}\(I\)=[0-9.e+-]+I\^\{{-?[0-9.]+\}}\$, "
            rf"\$R\^2\$=-?[0-9.]+$"
        )
        assert len(fit_labels) == 4
        assert all(pattern.match(label) for label in fit_labels)
        for concentration, data in datasets.items():
            equation = (
                f"{concentration}: ${symbol}(I)={data[f'{symbol}_K']:.3g}"
                f"I^{{{data[f'{symbol}_N']:.2f}}}$, $R^2$="
            )
            assert sum(label.startswith(equation) for label in fit_labels) == 1

    plt.close("all")
