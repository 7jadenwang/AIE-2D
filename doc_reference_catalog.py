"""Versioned model-catalog implementation used by :mod:`doc_reference`.

Schema v2 is accepted as a reproducible legacy isotonic artifact.  Schema v3
is the multi-model catalog whose declared default is collaborator_original.
Migration from v2 is an in-memory view and never rewrites the source artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = REPOSITORY_DIR / "doc_reference_catalog.json"
LEGACY_REFERENCE_PATH = REPOSITORY_DIR / "doc_reference_curves.json"
DEFAULT_REFERENCE_ID = "30mW_0mM"
CURRENT_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (2, 3)
DEFAULT_CURVE_MODEL = "collaborator_original"
CURVE_MODEL_IDS = (
    "collaborator_original",
    "isotonic",
    "avrami",
    "gompertz",
    "richards",
)
COLLABORATOR_FORMULA = "1 - a * exp(minimum(b * (c - t), 0))"
EXPECTED_CONDITIONS = {
    "30mW_0mM": {"intensity_mw_cm2": 30.0, "tempo_concentration_mM": 0.0},
    "30mW_5mM": {"intensity_mw_cm2": 30.0, "tempo_concentration_mM": 5.0},
}


class DoCReferenceError(ValueError):
    """Raised when a reference artifact is missing or scientifically invalid."""


def _parameter(parameters: dict[str, float], *names: str) -> float:
    for name in names:
        if name in parameters:
            return float(parameters[name])
    raise DoCReferenceError(f"missing model parameter; expected one of {names}")


def _collaborator_original(t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    a, b, c = _parameter(p, "a"), _parameter(p, "b", "b_per_s"), _parameter(p, "c", "c_s")
    return 1.0 - a * np.exp(np.minimum(b * (c - t), 0.0))


def _avrami(t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    y0 = _parameter(p, "y0")
    t0, tau, n = _parameter(p, "t0_s", "t0"), _parameter(p, "tau_s", "tau"), _parameter(p, "shape_n", "n")
    elapsed = np.maximum(t - t0, 0.0)
    rise = y0 + (1.0 - y0) * (1.0 - np.exp(-np.power(elapsed / tau, n)))
    return np.where(t <= t0, y0, rise)


def _gompertz(t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    y0, midpoint, scale = _parameter(p, "y0"), _parameter(p, "midpoint_s", "midpoint"), _parameter(p, "scale_s", "scale")
    z = np.clip(-(t - midpoint) / scale, -700.0, 700.0)
    return y0 + (1.0 - y0) * np.exp(-np.exp(z))


def _richards(t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    y0 = _parameter(p, "y0")
    midpoint, scale, nu = _parameter(p, "midpoint_s", "midpoint"), _parameter(p, "scale_s", "scale"), _parameter(p, "shape_nu", "nu")
    z = np.clip(-(t - midpoint) / scale, -700.0, 700.0)
    return y0 + (1.0 - y0) * np.power(1.0 + np.exp(z), -1.0 / nu)


_PARAMETRIC_EVALUATORS = {
    "collaborator_original": _collaborator_original,
    "avrami": _avrami,
    "gompertz": _gompertz,
    "richards": _richards,
}


@dataclass(frozen=True)
class DoCReferenceCurve:
    """A selected curve model evaluated in absolute process time."""

    time_s: np.ndarray
    doc_reference: np.ndarray
    metadata: dict[str, Any]
    source_path: str
    source_sha256: str
    curve_model: str
    model_parameters: dict[str, float] | None = None

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
        return float(self.evaluate(self.end_time_s))

    @property
    def saturation_time_s(self) -> float | None:
        value = self.metadata.get("saturation_time_s")
        if value is None and isinstance(self.metadata.get("saturation_times_s"), dict):
            times = self.metadata["saturation_times_s"]
            value = times.get("production") or times.get("production_saturation_time_s")
        return None if value is None else float(value)

    def evaluate(self, process_time_s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate at absolute process time, with endpoint hold."""

        query = np.asarray(process_time_s, dtype=float)
        if not np.isfinite(query).all():
            raise ValueError("reference query times must be finite")
        if self.curve_model in _PARAMETRIC_EVALUATORS:
            if self.model_parameters is None:
                raise DoCReferenceError(f"{self.curve_model} parameters are unavailable")
            values = _PARAMETRIC_EVALUATORS[self.curve_model](query, self.model_parameters)
            values = np.where(query < self.start_time_s, self.doc_reference[0], values)
            values = np.where(query > self.end_time_s, self.doc_reference[-1], values)
        else:
            values = np.interp(query, self.time_s, self.doc_reference, left=self.doc_reference[0], right=self.doc_reference[-1])
        values = np.clip(values, 0.0, 1.0)
        return float(values) if query.ndim == 0 else np.asarray(values, dtype=float)

    at = evaluate

    def stage_times(self, current_process_time_s: float, control_dt_s: float, horizon: int) -> np.ndarray:
        if not math.isfinite(current_process_time_s) or current_process_time_s < 0:
            raise ValueError("current process time must be finite and nonnegative")
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return current_process_time_s + control_dt_s * np.arange(1, horizon + 1)

    def stage_values(self, current_process_time_s: float, control_dt_s: float, horizon: int) -> np.ndarray:
        return np.asarray(self.evaluate(self.stage_times(current_process_time_s, control_dt_s, horizon)), dtype=float)

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            "artifact_path": self.source_path,
            "artifact_sha256": self.source_sha256,
            "artifact_schema_version": int(self.metadata["schema_version"]),
            "source_schema_version": int(self.metadata.get("source_schema_version", self.metadata["schema_version"])),
            "artifact_id": self.metadata.get("artifact_id"),
            "reference_id": self.reference_id,
            "condition": self.metadata["condition"],
            "condition_label": self.metadata.get("condition_label"),
            "curve_model": self.curve_model,
            "curve_model_role": self.metadata.get("curve_model_role"),
            "model_formula": self.metadata.get("model_formula"),
            "model_parameters": self.model_parameters,
            "fit_metrics": self.metadata.get("fit_metrics", {}),
            "source_data_file": self.metadata.get("source_data_file"),
            "source_data_sha256": self.metadata.get("source_data_sha256"),
            "raw_source_columns": self.metadata.get("raw_source_columns", {}),
            "replicate_treatment": self.metadata.get("replicate_treatment"),
            "reference_time_range_s": [self.start_time_s, self.end_time_s],
            "final_doc": self.final_doc,
            "legacy_schema_migration": self.metadata.get("legacy_schema_migration"),
        }


DoCReference = DoCReferenceCurve


def _read_document(path: Path | str) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"required DoC reference artifact is missing: {source_path}")
    raw = source_path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DoCReferenceError(f"invalid DoC reference JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise DoCReferenceError(
            f"unsupported schema_version {document.get('schema_version') if isinstance(document, dict) else None!r}; supported versions are {SUPPORTED_SCHEMA_VERSIONS}"
        )
    return source_path, raw, document


def migrate_v2_to_v3(document: dict[str, Any]) -> dict[str, Any]:
    """Create a non-destructive runtime catalog view of schema v2."""

    if document.get("schema_version") != 2:
        raise DoCReferenceError("migrate_v2_to_v3 requires a schema-v2 document")
    migrated = copy.deepcopy(document)
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    migrated["source_schema_version"] = 2
    migrated["artifact_id"] = f"{document.get('artifact_id', 'doc_reference')}__runtime_v3_view"
    for reference in migrated.get("references", {}).values():
        comparison = reference.get("model_comparison", {})
        curve_models: dict[str, Any] = {
            "isotonic": {
                "model_id": "isotonic",
                "role": "legacy_schema_v2_default",
                "representation": "sampled_table",
                "time_s": reference.get("time_s"),
                "doc_reference": reference.get("doc_reference"),
                "metrics": comparison.get("isotonic_monotonic_benchmark", {}).get("metrics", reference.get("fit_metrics", {})),
            }
        }
        for public_id, legacy_id in {
            "avrami": "delayed_avrami_fixed_plateau",
            "gompertz": "gompertz_fixed_plateau",
            "richards": "richards_fixed_plateau",
        }.items():
            record = comparison.get(legacy_id)
            if isinstance(record, dict) and isinstance(record.get("joint_fit_parameters"), dict):
                curve_models[public_id] = {
                    "model_id": public_id,
                    "source_model_id": legacy_id,
                    "role": "historical_diagnostic",
                    "representation": "parametric",
                    "parameters": record["joint_fit_parameters"],
                    "metrics": record.get("metrics", {}),
                }
        reference["default_curve_model"] = "isotonic"
        reference["curve_models"] = curve_models
        reference["legacy_schema_migration"] = {
            "from_schema_version": 2,
            "method": "non_destructive_runtime_view_v1",
            "scientific_values_changed": False,
            "collaborator_original_available": False,
        }
    return migrated


def _catalog_document(document: dict[str, Any]) -> tuple[dict[str, Any], int]:
    source_schema = int(document["schema_version"])
    return (migrate_v2_to_v3(document), source_schema) if source_schema == 2 else (document, source_schema)


def available_doc_reference_ids(path: Path | str = DEFAULT_REFERENCE_PATH) -> tuple[str, ...]:
    _, _, source = _read_document(path)
    document, _ = _catalog_document(source)
    references = document.get("references")
    if not isinstance(references, dict):
        raise DoCReferenceError("reference artifact must contain a references object")
    return tuple(sorted(references))


def available_curve_models(reference_id: str = DEFAULT_REFERENCE_ID, path: Path | str = DEFAULT_REFERENCE_PATH) -> tuple[str, ...]:
    _, _, source = _read_document(path)
    document, _ = _catalog_document(source)
    references = document.get("references", {})
    if reference_id not in references:
        raise DoCReferenceError(f"unknown DoC reference ID {reference_id!r}; available IDs: {sorted(references)}")
    models = references[reference_id].get("curve_models")
    if not isinstance(models, dict):
        raise DoCReferenceError(f"reference {reference_id!r} has no curve_models catalog")
    return tuple(model for model in CURVE_MODEL_IDS if model in models)


def _numeric_vector(values: Any, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise DoCReferenceError(f"{label} must be numeric") from error
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise DoCReferenceError(f"{label} must be a finite one-dimensional array of length >= 2")
    return vector


def _validate_condition(reference_id: str, reference: dict[str, Any]) -> None:
    if reference.get("reference_id") != reference_id:
        raise DoCReferenceError("reference map key and embedded reference_id disagree")
    expected = EXPECTED_CONDITIONS.get(reference_id)
    condition = reference.get("condition")
    if expected is None or not isinstance(condition, dict):
        raise DoCReferenceError(f"runtime has no valid condition contract for {reference_id!r}")
    for field, expected_value in expected.items():
        try:
            actual = float(condition[field])
        except (KeyError, TypeError, ValueError) as error:
            raise DoCReferenceError(f"{reference_id}.{field} is missing or invalid") from error
        if not math.isfinite(actual) or not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise DoCReferenceError(f"{reference_id}.{field} must be {expected_value:g}, got {actual!r}")


def load_doc_reference(
    reference_id: str = DEFAULT_REFERENCE_ID,
    path: Path | str = DEFAULT_REFERENCE_PATH,
    curve_model: str | None = None,
) -> DoCReferenceCurve:
    """Load a selected condition/model from schema v2 or v3."""

    source_path, raw, source_document = _read_document(path)
    document, source_schema = _catalog_document(source_document)
    references = document.get("references")
    if not isinstance(references, dict) or reference_id not in references:
        available = sorted(references) if isinstance(references, dict) else []
        raise DoCReferenceError(f"unknown DoC reference ID {reference_id!r}; available IDs: {available}")
    reference = references[reference_id]
    if not isinstance(reference, dict):
        raise DoCReferenceError(f"reference {reference_id!r} must be an object")
    _validate_condition(reference_id, reference)
    models = reference.get("curve_models")
    selected = curve_model or reference.get("default_curve_model")
    if selected not in CURVE_MODEL_IDS:
        raise DoCReferenceError(f"unknown curve model {selected!r}; supported model IDs: {CURVE_MODEL_IDS}")
    if not isinstance(models, dict) or selected not in models:
        available = sorted(models) if isinstance(models, dict) else []
        raise DoCReferenceError(
            f"curve model {selected!r} is unavailable for {reference_id!r} in schema-v{source_schema}; available models: {available}"
        )
    model = models[selected]
    representation = model.get("representation") if isinstance(model, dict) else None
    parameters: dict[str, float] | None
    if representation == "sampled_table":
        time_s = _numeric_vector(model.get("time_s", reference.get("time_s")), "time_s")
        values = _numeric_vector(model.get("doc_reference", reference.get("doc_reference")), "doc_reference")
        parameters = None
    elif representation == "parametric":
        raw_parameters = model.get("parameters")
        if not isinstance(raw_parameters, dict) or not raw_parameters:
            raise DoCReferenceError(f"curve model {selected!r} has no parameters")
        parameters = {key: float(value) for key, value in raw_parameters.items()}
        range_s = model.get("reference_time_range_s", reference.get("reference_time_range_s", [0.0, 20.0]))
        dt_s = float(model.get("dt_s", reference.get("dt_s", document.get("authoritative_reference_dt_s", 0.05))))
        if not isinstance(range_s, list) or len(range_s) != 2 or dt_s <= 0:
            raise DoCReferenceError("parametric model has invalid time range/grid")
        time_s = np.round(np.arange(float(range_s[0]), float(range_s[1]) + 0.5 * dt_s, dt_s), 10)
        evaluator = _PARAMETRIC_EVALUATORS.get(selected)
        if evaluator is None:
            raise DoCReferenceError(f"no evaluator is implemented for {selected!r}")
        values = np.asarray(evaluator(time_s, parameters), dtype=float)
    else:
        raise DoCReferenceError(f"curve model {selected!r} has unsupported representation {representation!r}")
    if time_s.size != values.size or np.any(np.diff(time_s) <= 0):
        raise DoCReferenceError("reference times must be strictly increasing and match values")
    if not np.isfinite(values).all() or float(values.min()) < -1e-12 or float(values.max()) > 1.0 + 1e-12:
        raise DoCReferenceError("selected DoC curve must be finite and remain in [0,1]")
    if np.any(np.diff(values) < -1e-10):
        raise DoCReferenceError("selected DoC curve must be monotonic nondecreasing")
    values = np.clip(values, 0.0, 1.0)
    time_s.setflags(write=False)
    values.setflags(write=False)
    metadata = {
        **reference,
        "schema_version": int(document["schema_version"]),
        "source_schema_version": source_schema,
        "artifact_id": document.get("artifact_id"),
        "curve_model": selected,
        "curve_model_role": model.get("role"),
        "model_formula": model.get("formula"),
        "fit_metrics": model.get("metrics", {}),
        "source_data_sha256": reference.get("source_data_sha256", document.get("source_data_sha256")),
        "replicate_treatment": model.get("replicate_treatment", reference.get("production_reference_construction_method")),
    }
    return DoCReferenceCurve(
        time_s=time_s,
        doc_reference=values,
        metadata=metadata,
        source_path=str(source_path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        curve_model=selected,
        model_parameters=parameters,
    )
