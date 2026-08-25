# Repository structure

The repository root retains collaborator/reference sources, production controller code, and production calibration artifacts that are needed by normal workflows.

- `AIE_TEMPOv1.1.py` is collaborator-owned and read-only for our development workflow.
- `DoC curve.ipynb` contains the experimental DoC and tracking-reference processing.
- `doc_reference_curves.json` is the production time-varying DoC tracking-reference artifact.
- `aie_reference.py`, `aie_model.py`, `aie_mpc.py`, `run_mpc.py`, `doc_curve_fit.py`, `doc_reference.py`, and `mpc_metrics.py` implement the production adapter, physics, controller, fitting, loading, and metrics workflows.
- `GEO/` contains target geometry inputs.
- `260722_circles_TPEoac/` and `260728_circles/` contain experimental inputs and their source exports.
- `diagnostics/` contains standalone scientific diagnostic scripts; these are not production controller modules.
- `results/mpc/` contains generated MPC run directories.
- `results/diagnostics/` contains generated diagnostic figures, tables, and summaries grouped by diagnostic.

Generated material under `results/` is output, not source code. Small summaries may be retained for reproducibility, while large frame sequences and animations can be handled separately according to project archival policy.

Recommended commands from the repository root:

```powershell
python run_mpc.py --target GEO/Lshape.png --total-time 20 --doc-reference 30mW_0mM --output-dir results/mpc/my_run
python diagnostics/diagnose_open_loop_feasibility.py
python diagnostics/diagnose_spatial_tracking_gap.py
```
