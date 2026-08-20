"""Condition-aware loader for experimental MPC DoC trajectory references.

The loaded curve is a desired control reference.  It is intentionally separate
from the authoritative AIE forward physics and its reaction-progress history.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = REPOSITORY_DIR / "doc_reference_curves.json"
DEFAULT_REFERENCE_ID = "30mW_0mM"
DOC_REFERENCE_SCHEMA_VERSION = 2
PRODUCTION_REFERENCE_METHOD = "equal_replicate_isotonic_linear_interp_v1"
EXPECTED_CONDITIONS = {
    "30mW_0mM": {"intensity_mw_cm2": 30.0, "tempo_concentration_mM": 0.0},
    "30mW_5mM": {"intensity_mw_cm2": 30.0, "tempo_concentration_mM": 5.0},
}


class DoCReferenceError(ValueError):
    """Raised when a reference artifact is missing or scientifically invalid."""


@dataclass(frozen=True)
class DoCReferenceCurve:
    """Finite monotonic absolute-time DoC curve with endpoint hold."""

    time_s: np.ndarray
    doc_reference: np.ndarray
    metadata: dict[str, Any]
    source_path: str
    source_sha256: str

    @property
    def reference_id(self) -> str:
        return str(self.metadata["reference_id"])

    @property
    def start_time_s(self) -> float:
        return float(self.time_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def final_doc(self) -> float:
        return float(self.doc_reference[-1])

    @property
    def saturation_time_s(self) -> float:
        return float(self.metadata["saturation_times_s"]["production"])

    def at(
        self, process_time_s: float | Sequence[float] | np.ndarray
    ) -> float | np.ndarray:
        """Linearly interpolate at absolute process time with endpoint hold."""

        query = np.asarray(process_time_s, dtype=float)
        if not np.isfinite(query).all():
            raise ValueError("reference query times must be finite")
        values = np.interp(
            query,
            self.time_s,
            self.doc_reference,
            left=self.doc_reference[0],
            right=self.doc_reference[-1],
        )
        if query.ndim == 0:
            return float(values)
        return values

    def stage_times(
        self, current_process_time_s: float, control_dt_s: float, horizon: int
    ) -> np.ndarray:
        """Return future absolute control-boundary times for an MPC horizon."""

        if not math.isfinite(current_process_time_s) or current_process_time_s < 0:
            raise ValueError("current process time must be finite and nonnegative")
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return current_process_time_s + control_dt_s * np.arange(1, horizon + 1)

    def stage_values(
        self, current_process_time_s: float, control_dt_s: float, horizon: int
    ) -> np.ndarray:
        """Interpolate at absolute future stage times; never restart at zero."""

        return np.asarray(
            self.at(self.stage_times(current_process_time_s, control_dt_s, horizon)),
            dtype=float,
        )

    def provenance_metadata(self) -> dict[str, Any]:
        """Return compact, machine-readable calibration provenance."""

        return {
            "artifact_path": self.source_path,
            "artifact_sha256": self.source_sha256,
            "artifact_schema_version": self.metadata["schema_version"],
            "artifact_id": self.metadata["artifact_id"],
            "artifact_exported_at_utc": self.metadata["artifact_exported_at_utc"],
            "source_data_sha256": self.metadata["source_data_sha256"],
            "reference_id": self.reference_id,
            "condition": self.metadata["condition"],
            "condition_label": self.metadata["condition_label"],
            "selected_fit_model": self.metadata["selected_fit_model"],
            "selected_fit_parameters": self.metadata["selected_fit_parameters"],
            "fit_metrics": self.metadata["fit_metrics"],
            "production_reference_method": self.metadata[
                "production_reference_method"
            ],
            "equal_replicate_construction": self.metadata[
                "equal_replicate_construction"
            ],
            "isotonic_regression": self.metadata["isotonic_regression"],
            "runtime_grid_interpolation_method": self.metadata[
                "runtime_grid_interpolation_method"
            ],
            "compact_diagnostic_fit": self.metadata["compact_diagnostic_fit"],
            "saturation_detection_rule": self.metadata["saturation_detection_rule"],
            "saturation_times_s": self.metadata["saturation_times_s"],
            "threshold_times_s": self.metadata["threshold_times_s"],
            "reference_time_range_s": [self.start_time_s, self.end_time_s],
            "final_doc": self.final_doc,
            "raw_source_columns": self.metadata["raw_source_columns"],
            "legacy_label_notes": self.metadata["legacy_label_notes"],
        }


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DoCReferenceError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise DoCReferenceError(f"{label} must be finite")
    return number


def _read_document(path: Path | str) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"required DoC reference artifact is missing: {source_path}")
    raw = source_path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DoCReferenceError(f"invalid DoC reference JSON: {error}") from error
    if not isinstance(document, dict):
        raise DoCReferenceError("DoC reference document must be an object")
    return source_path, raw, document


def available_doc_reference_ids(
    path: Path | str = DEFAULT_REFERENCE_PATH,
) -> tuple[str, ...]:
    """Return validated top-level reference IDs without selecting a curve."""

    _, _, document = _read_document(path)
    if document.get("schema_version") != DOC_REFERENCE_SCHEMA_VERSION:
        raise DoCReferenceError(
            f"unsupported schema_version {document.get('schema_version')!r}; "
            f"expected {DOC_REFERENCE_SCHEMA_VERSION}"
        )
    references = document.get("references")
    if not isinstance(references, dict):
        raise DoCReferenceError("schema-v2 artifact must contain a references object")
    return tuple(sorted(references))


def _validate_reference(
    document: dict[str, Any],
    reference_id: str,
    source_path: Path,
    source_sha256: str,
) -> DoCReferenceCurve:
    top_required = {
        "schema_version",
        "artifact_id",
        "exported_at_utc",
        "source_notebook",
        "source_data_file",
        "source_data_sha256",
        "authoritative_reference_dt_s",
        "references",
    }
    missing_top = sorted(top_required - document.keys())
    if missing_top:
        raise DoCReferenceError(f"reference artifact is missing fields: {missing_top}")
    if document["schema_version"] != DOC_REFERENCE_SCHEMA_VERSION:
        raise DoCReferenceError(
            f"unsupported schema_version {document['schema_version']!r}; "
            f"expected {DOC_REFERENCE_SCHEMA_VERSION}"
        )
    if document["source_notebook"] != "DoC curve.ipynb":
        raise DoCReferenceError("source_notebook must be 'DoC curve.ipynb'")
    references = document["references"]
    if not isinstance(references, dict):
        raise DoCReferenceError("references must be an object")
    if reference_id not in references:
        raise DoCReferenceError(
            f"unknown DoC reference ID {reference_id!r}; available IDs: "
            f"{sorted(references)}"
        )
    if reference_id not in EXPECTED_CONDITIONS:
        raise DoCReferenceError(f"runtime has no condition contract for {reference_id!r}")
    reference = references[reference_id]
    if not isinstance(reference, dict):
        raise DoCReferenceError(f"reference {reference_id!r} must be an object")
    required = {
        "reference_id",
        "condition",
        "condition_label",
        "source_notebook",
        "source_data_file",
        "source_data_sha256",
        "source_sheet",
        "raw_source_columns",
        "legacy_label_notes",
        "production_reference_method",
        "selected_fit_model",
        "selected_fit_parameters",
        "fit_metrics",
        "model_comparison",
        "replicate_fits",
        "saturation_detection_rule",
        "saturation_times_s",
        "production_reference_construction_method",
        "equal_replicate_construction",
        "isotonic_regression",
        "runtime_grid_interpolation_method",
        "threshold_times_s",
        "production_threshold_comparison_s",
        "compact_diagnostic_fit",
        "reference_time_range_s",
        "dt_s",
        "time_s",
        "doc_reference",
    }
    missing = sorted(required - reference.keys())
    if missing:
        raise DoCReferenceError(f"reference {reference_id!r} is missing fields: {missing}")
    if reference["reference_id"] != reference_id:
        raise DoCReferenceError("reference map key and embedded reference_id disagree")
    condition = reference["condition"]
    if not isinstance(condition, dict):
        raise DoCReferenceError("condition must be an object")
    expected = EXPECTED_CONDITIONS[reference_id]
    for field, expected_value in expected.items():
        actual = _finite_float(condition.get(field), f"{reference_id}.{field}")
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise DoCReferenceError(
                f"{reference_id}.{field} must be {expected_value:g}, got {actual:g}"
            )
    raw_columns = reference["raw_source_columns"]
    if not isinstance(raw_columns, dict) or set(raw_columns) != {"T1", "T2"}:
        raise DoCReferenceError("raw_source_columns must preserve exact T1/T2 provenance")
    for label, columns in raw_columns.items():
        if not isinstance(columns, dict) or not all(
            isinstance(columns.get(key), str) and columns[key]
            for key in ("raw_replicate_label", "time", "signal")
        ):
            raise DoCReferenceError(f"{reference_id}/{label} source columns are incomplete")
    if not isinstance(reference["legacy_label_notes"], list):
        raise DoCReferenceError("legacy_label_notes must be a list")
    if not isinstance(reference["selected_fit_parameters"], dict) or not reference[
        "selected_fit_parameters"
    ]:
        raise DoCReferenceError("selected_fit_parameters must be a nonempty object")
    if reference["production_reference_method"] != PRODUCTION_REFERENCE_METHOD:
        raise DoCReferenceError(
            "production_reference_method must be "
            f"{PRODUCTION_REFERENCE_METHOD!r}"
        )
    if reference["selected_fit_model"] != "isotonic_monotonic_benchmark":
        raise DoCReferenceError(
            "production selected_fit_model must be isotonic_monotonic_benchmark"
        )
    if reference["runtime_grid_interpolation_method"] != "piecewise_linear":
        raise DoCReferenceError("runtime grid interpolation must be piecewise_linear")
    construction = reference["equal_replicate_construction"]
    if not isinstance(construction, dict) or construction.get(
        "replicate_weights"
    ) != {"T1": 0.5, "T2": 0.5}:
        raise DoCReferenceError("production reference must weight T1/T2 equally")
    isotonic = reference["isotonic_regression"]
    if (
        not isinstance(isotonic, dict)
        or isotonic.get("increasing") is not True
        or isotonic.get("lower_bound") != 0.0
        or isotonic.get("upper_bound") != 1.0
        or not isinstance(isotonic.get("block_count"), int)
        or isotonic["block_count"] < 1
    ):
        raise DoCReferenceError("isotonic production metadata is invalid")
    model_comparison = reference["model_comparison"]
    if not isinstance(model_comparison, dict):
        raise DoCReferenceError("model_comparison must be an object")
    selected_models = [
        model_id
        for model_id, model in model_comparison.items()
        if isinstance(model, dict) and model.get("selected") is True
    ]
    if selected_models != ["isotonic_monotonic_benchmark"]:
        raise DoCReferenceError("model comparison must select only isotonic production")
    compact = reference["compact_diagnostic_fit"]
    if (
        not isinstance(compact, dict)
        or compact.get("model_id") != "delayed_avrami_fixed_plateau"
        or compact.get("role") != "diagnostic_only_not_MPC_reference"
    ):
        raise DoCReferenceError("compact Avrami fit must be diagnostic only")
    rule = reference["saturation_detection_rule"]
    if not isinstance(rule, dict) or not all(
        key in rule for key in ("rule_id", "normalized_threshold", "consecutive_samples", "description")
    ):
        raise DoCReferenceError("saturation detection provenance is incomplete")

    try:
        time_s = np.asarray(reference["time_s"], dtype=float)
        doc_reference = np.asarray(reference["doc_reference"], dtype=float)
    except (TypeError, ValueError) as error:
        raise DoCReferenceError("time_s/doc_reference must be numeric arrays") from error
    if time_s.ndim != 1 or doc_reference.ndim != 1:
        raise DoCReferenceError("time_s/doc_reference must be one-dimensional")
    if time_s.size < 2 or time_s.size != doc_reference.size:
        raise DoCReferenceError("time_s/doc_reference must have equal length >= 2")
    if not np.isfinite(time_s).all() or not np.isfinite(doc_reference).all():
        raise DoCReferenceError("reference trajectory contains NaN or Inf")
    if np.any(np.diff(time_s) <= 0):
        raise DoCReferenceError("reference times must be strictly increasing")
    if float(np.min(doc_reference)) < 0.0 or float(np.max(doc_reference)) > 1.0:
        raise DoCReferenceError("doc_reference must remain in [0,1]")
    if np.any(np.diff(doc_reference) < -1e-12):
        raise DoCReferenceError("doc_reference must be monotonic nondecreasing")
    if not math.isclose(float(time_s[0]), 0.0, abs_tol=1e-12) or not math.isclose(
        float(time_s[-1]), 20.0, abs_tol=1e-12
    ):
        raise DoCReferenceError("runtime reference must cover exactly 0 to 20 s")
    declared_range = reference["reference_time_range_s"]
    if declared_range != [0.0, 20.0]:
        raise DoCReferenceError("reference_time_range_s must declare [0.0, 20.0]")
    dt_s = _finite_float(reference["dt_s"], "reference dt_s")
    if not np.allclose(np.diff(time_s), dt_s, rtol=0.0, atol=1e-10):
        raise DoCReferenceError("reference time grid is inconsistent with declared dt_s")
    saturation = _finite_float(
        reference["saturation_times_s"].get("production"), "production saturation"
    )
    if not 0.0 <= saturation <= 20.0:
        raise DoCReferenceError("production saturation time must lie in [0,20]")
    plateau = time_s >= saturation - 1e-12
    if not np.any(plateau) or not np.all(doc_reference[plateau] == 1.0):
        raise DoCReferenceError(
            "doc_reference must remain exactly 1.0 at and after production saturation"
        )
    if doc_reference[-1] != 1.0:
        raise DoCReferenceError("20 s reference value must be exactly 1.0")
    if reference["source_data_sha256"] != document["source_data_sha256"]:
        raise DoCReferenceError("reference/source artifact workbook hashes disagree")

    metadata = {
        **reference,
        "schema_version": document["schema_version"],
        "artifact_id": document["artifact_id"],
        "artifact_exported_at_utc": document["exported_at_utc"],
    }
    time_s.setflags(write=False)
    doc_reference.setflags(write=False)
    return DoCReferenceCurve(
        time_s=time_s,
        doc_reference=doc_reference,
        metadata=metadata,
        source_path=str(source_path),
        source_sha256=source_sha256,
    )


def load_doc_reference(
    reference_id: str = DEFAULT_REFERENCE_ID,
    path: Path | str = DEFAULT_REFERENCE_PATH,
) -> DoCReferenceCurve:
    """Load and validate a named experimental reference from schema v2."""

    if not isinstance(reference_id, str) or not reference_id:
        raise DoCReferenceError("reference_id must be a nonempty string")
    source_path, raw, document = _read_document(path)
    return _validate_reference(
        document, reference_id, source_path, hashlib.sha256(raw).hexdigest()
    )
