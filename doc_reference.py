"""Validated loader/interpolator for the experimental MPC DoC reference.

The reference trajectory is deliberately separate from ``aie_model``: it is a
desired control trajectory, never a replacement for the AIE forward physics.
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
DEFAULT_REFERENCE_PATH = REPOSITORY_DIR / "doc_reference_curve.json"
DOC_REFERENCE_SCHEMA_VERSION = 1
EXPECTED_INTENSITY_MW_CM2 = 30.0
EXPECTED_TEMPO_CONCENTRATION_MM = 0.0


class DoCReferenceError(ValueError):
    """Raised when the exported trajectory is missing or scientifically invalid."""


@dataclass(frozen=True)
class DoCReferenceCurve:
    """Finite monotonic time-domain DoC reference with endpoint hold."""

    time_s: np.ndarray
    doc_reference: np.ndarray
    metadata: dict[str, Any]
    source_path: str
    source_sha256: str

    @property
    def start_time_s(self) -> float:
        return float(self.time_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def final_doc(self) -> float:
        return float(self.doc_reference[-1])

    def at(self, process_time_s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
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
        """Return absolute future control-boundary times for one MPC horizon."""

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
        """Interpolate references at absolute future control-boundary times."""

        return np.asarray(
            self.at(self.stage_times(current_process_time_s, control_dt_s, horizon)),
            dtype=float,
        )

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "schema_version": self.metadata["schema_version"],
            "reference_id": self.metadata["reference_id"],
            "condition": self.metadata["condition"],
            "condition_label": self.metadata.get("condition_label", ""),
            "selected_fit_model": self.metadata["selected_fit_model"],
            "reference_time_range_s": [self.start_time_s, self.end_time_s],
            "final_doc": self.final_doc,
        }


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DoCReferenceError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise DoCReferenceError(f"{label} must be finite")
    return number


def _validate_document(document: dict[str, Any], source_path: Path, source_sha256: str) -> DoCReferenceCurve:
    required = {
        "schema_version",
        "reference_id",
        "condition",
        "source_notebook",
        "source_data_file",
        "legacy_source_names",
        "selected_fit_model",
        "selected_fit_parameters",
        "replicate_fit_parameters",
        "fit_metrics",
        "filtering",
        "reference_construction_method",
        "reference_time_range_s",
        "time_s",
        "doc_reference",
    }
    missing = sorted(required - document.keys())
    if missing:
        raise DoCReferenceError(f"DoC reference is missing required fields: {missing}")
    if document["schema_version"] != DOC_REFERENCE_SCHEMA_VERSION:
        raise DoCReferenceError(
            f"unsupported schema_version {document['schema_version']!r}; "
            f"expected {DOC_REFERENCE_SCHEMA_VERSION}"
        )
    if document["source_notebook"] != "DoC curve.ipynb":
        raise DoCReferenceError("source_notebook must be 'DoC curve.ipynb'")
    condition = document["condition"]
    if not isinstance(condition, dict):
        raise DoCReferenceError("condition must be an object")
    intensity = _finite_float(condition.get("intensity_mw_cm2"), "condition intensity")
    tempo = _finite_float(
        condition.get("tempo_concentration_mM"), "condition TEMPO concentration"
    )
    if not math.isclose(intensity, EXPECTED_INTENSITY_MW_CM2, abs_tol=1e-12):
        raise DoCReferenceError(
            f"reference intensity must be {EXPECTED_INTENSITY_MW_CM2:g} mW/cm^2, got {intensity:g}"
        )
    if not math.isclose(tempo, EXPECTED_TEMPO_CONCENTRATION_MM, abs_tol=1e-12):
        raise DoCReferenceError(
            f"reference TEMPO concentration must be {EXPECTED_TEMPO_CONCENTRATION_MM:g} mM, got {tempo:g}"
        )
    if not isinstance(document["reference_id"], str) or not document["reference_id"]:
        raise DoCReferenceError("reference_id must be a nonempty string")
    if not isinstance(document["selected_fit_model"], str) or not document["selected_fit_model"]:
        raise DoCReferenceError("selected_fit_model must be a nonempty string")

    try:
        time_s = np.asarray(document["time_s"], dtype=float)
        doc_reference = np.asarray(document["doc_reference"], dtype=float)
    except (TypeError, ValueError) as error:
        raise DoCReferenceError("time_s and doc_reference must be numeric arrays") from error
    if time_s.ndim != 1 or doc_reference.ndim != 1:
        raise DoCReferenceError("time_s and doc_reference must be one-dimensional")
    if time_s.size < 2 or time_s.size != doc_reference.size:
        raise DoCReferenceError("time_s/doc_reference must have equal length >= 2")
    if not np.isfinite(time_s).all() or not np.isfinite(doc_reference).all():
        raise DoCReferenceError("reference trajectory contains NaN or Inf")
    if np.any(np.diff(time_s) <= 0):
        raise DoCReferenceError("reference times must be strictly increasing")
    if float(doc_reference.min()) < 0.0 or float(doc_reference.max()) > 1.0:
        raise DoCReferenceError("doc_reference must remain in [0, 1]")
    if np.any(np.diff(doc_reference) < -1e-10):
        raise DoCReferenceError("doc_reference must be monotonic nondecreasing")
    declared_range = document["reference_time_range_s"]
    if not isinstance(declared_range, list) or len(declared_range) != 2:
        raise DoCReferenceError("reference_time_range_s must contain [start, end]")
    start = _finite_float(declared_range[0], "reference start time")
    end = _finite_float(declared_range[1], "reference end time")
    if not math.isclose(start, float(time_s[0]), abs_tol=1e-12) or not math.isclose(
        end, float(time_s[-1]), abs_tol=1e-12
    ):
        raise DoCReferenceError("declared reference range does not match time_s endpoints")
    if not math.isclose(start, 0.0, abs_tol=1e-12) or not math.isclose(
        end, 20.0, abs_tol=1e-12
    ):
        raise DoCReferenceError("runtime reference must cover exactly 0 to 20 s")

    time_s.setflags(write=False)
    doc_reference.setflags(write=False)
    return DoCReferenceCurve(
        time_s=time_s,
        doc_reference=doc_reference,
        metadata=document,
        source_path=str(source_path),
        source_sha256=source_sha256,
    )


def load_doc_reference(path: Path | str = DEFAULT_REFERENCE_PATH) -> DoCReferenceCurve:
    """Read and validate the production 30 mW/cm^2, 0 mM reference."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"required DoC reference is missing: {source_path}")
    raw = source_path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DoCReferenceError(f"invalid DoC reference JSON: {error}") from error
    if not isinstance(document, dict):
        raise DoCReferenceError("DoC reference document must be an object")
    return _validate_document(document, source_path, hashlib.sha256(raw).hexdigest())

