"""Read-only adapter for the collaborator-owned AIE reference model.

``AIE_TEMPOv1.1.py`` is parsed as source and is never imported or executed.
Numeric assignments are evaluated by a deliberately small AST interpreter.
The active equation structure is validated separately: numeric-only changes
flow through automatically, while an unsupported physics edit fails loudly so
the differentiable implementation can be synchronized intentionally.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_DIR = Path(__file__).resolve().parent
REFERENCE_MODEL_PATH = REPOSITORY_DIR / "AIE_TEMPOv1.1.py"
DOC_FIT_PATH = REPOSITORY_DIR / "doc_fit_parameters.json"
DOC_FIT_SCHEMA_VERSION = 1
CONTROLLER_NATIVE_SHAPE = (300, 300)
SUPPORTED_MODEL_STRUCTURE_VERSION = 2
SUPPORTED_FORWARD_CONDITIONS = ("30mW_0mM", "30mW_5mM")


class ReferenceResolutionError(ValueError):
    """Raised when active reference source cannot be resolved safely."""


class UnsupportedReferencePhysicsError(ReferenceResolutionError):
    """Raised when reference equations no longer match the supported model."""


@dataclass(frozen=True)
class DoCFitReplicate:
    label: str
    source_column: str
    sample_count: int
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class DoCFitCalibration:
    calibration_id: str
    condition_id: str
    condition_label: str
    intensity_mw_cm2: float
    tempo_concentration_mM: float
    formula_id: str
    formula: str
    source_notebook: str
    source_data_file: str
    source_sheet: str
    exported_at_utc: str
    averaging_method: str
    replicates: tuple[DoCFitReplicate, ...]
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class ReferenceConfig:
    """Effective controller configuration resolved from read-only sources."""

    reference_model_source: str
    reference_model_path: str
    reference_model_sha256: str
    reference_structure_sha256: str
    model_structure_version: int
    native_shape: tuple[int, int]
    native_pixel_pitch_m: float
    projector_refinement: int
    dt: float
    total_simulation_time_s: float
    loss_sample_times_s: tuple[float, float, float]
    intensity_mw_cm2: float
    tempo_concentration_mM: float | None
    o2_diffusivity_m2_s: float
    tempo_diffusivity_m2_s: float
    o2_inhibition_mj_cm2: float
    total_inhibition_mj_cm2: float
    scattering_blur_size_m: float
    tempo_gaussian_sigma_scale: float
    o2_diffusion_enabled: bool
    tempo_diffusion_enabled: bool
    chain_growth_noise_std: float
    chain_growth_noise_enabled: bool
    b_slope: float
    b_intercept: float
    b_condition_label: str
    minimum_normalized_intensity: float
    division_epsilon: float
    mask_grayscale_max: float
    diffusion_kernel_size_formula: str
    scattering_kernel_size_formula: str
    scattering_sigma_formula: str
    doc_model_id: str
    doc_model_formula: str
    doc_fit_applied_to_governing_law: bool
    doc_calibration_source: str
    doc_calibration_path: str
    doc_calibration_sha256: str
    doc_fit_selection_status: str
    available_doc_fit_condition_ids: tuple[str, ...]
    doc_fit: DoCFitCalibration | None

    @property
    def tempo_inhibition_mj_cm2(self) -> float:
        return max(0.0, self.total_inhibition_mj_cm2 - self.o2_inhibition_mj_cm2)

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["native_shape"] = list(self.native_shape)
        metadata["loss_sample_times_s"] = list(self.loss_sample_times_s)
        metadata["available_doc_fit_condition_ids"] = list(
            self.available_doc_fit_condition_ids
        )
        return metadata


@dataclass
class ReferenceStepResult:
    o2: Any
    tempo: Any
    dose: Any
    doc: Any
    o2_diffused: Any
    tempo_diffused: Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReferenceResolutionError(
            f"{label} must be a finite number, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ReferenceResolutionError(f"{label} must be finite, got {result}")
    return result


def _safe_eval(node: ast.AST, values: dict[str, Any]) -> Any:
    """Evaluate only literals, names, arithmetic, and selected numeric calls."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)) or node.value is None:
            return node.value
    if isinstance(node, ast.Name):
        if node.id in values:
            return values[node.id]
        raise ReferenceResolutionError(f"unresolved name {node.id!r}")
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_safe_eval(item, values) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, values)
        right = _safe_eval(node.right, values)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        functions = {"float": float, "int": int, "max": max, "min": min}
        function = functions.get(node.func.id)
        if function is not None and not node.keywords:
            return function(*(_safe_eval(argument, values) for argument in node.args))
    raise ReferenceResolutionError(
        "unsupported non-literal expression: " + ast.unparse(node)
    )


def _bind_target(target: ast.AST, value: Any, values: dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
        values[target.id] = value
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
            raise ReferenceResolutionError(
                f"cannot safely unpack assignment target {ast.unparse(target)}"
            )
        for child, child_value in zip(target.elts, value):
            _bind_target(child, child_value, values)
        return
    raise ReferenceResolutionError(
        f"unsupported assignment target {ast.unparse(target)}"
    )


def _top_level_values(tree: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        try:
            value = _safe_eval(statement.value, values)
            for target in statement.targets:
                _bind_target(target, value, values)
        except ReferenceResolutionError:
            # Array/device/output assignments are intentionally ignored. Any
            # required physical value is checked explicitly below.
            continue
    return values


def _required_numeric(values: dict[str, Any], name: str) -> float:
    if name not in values:
        raise ReferenceResolutionError(
            f"required active reference parameter {name!r} could not be resolved"
        )
    return _finite_float(values[name], name)


def _assignment_nodes(tree: ast.Module, target_name: str) -> list[ast.Assign]:
    assignments: list[ast.Assign] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            assignments.append(node)
    return sorted(assignments, key=lambda item: (item.lineno, item.col_offset))


def _active_assignment(tree: ast.Module, target_name: str) -> ast.Assign:
    assignments = _assignment_nodes(tree, target_name)
    if not assignments:
        raise UnsupportedReferencePhysicsError(
            f"active equation assignment {target_name!r} was not found"
        )
    return assignments[-1]


def _ast_expression(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body


def _same_expression(actual: ast.AST, expected: ast.AST) -> bool:
    return ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def _expect_expression(tree: ast.Module, name: str, expected: str) -> ast.Assign:
    assignment = _active_assignment(tree, name)
    if not _same_expression(assignment.value, _ast_expression(expected)):
        raise UnsupportedReferencePhysicsError(
            f"unsupported active {name} equation: {ast.unparse(assignment.value)}; "
            f"expected {expected}"
        )
    return assignment


def _extract_clamp_min(node: ast.AST) -> float:
    values: list[float] = []
    for child in ast.walk(node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "clamp"
        ):
            continue
        for keyword in child.keywords:
            if keyword.arg == "min" and isinstance(keyword.value, ast.Constant):
                values.append(_finite_float(keyword.value.value, "clamp minimum"))
    if len(set(values)) != 1:
        raise UnsupportedReferencePhysicsError(
            f"expected one unambiguous local-intensity clamp minimum, found {values}"
        )
    return values[0]


def _extract_linear_b(
    tree: ast.Module, source_lines: list[str]
) -> tuple[float, float, ast.AST, str, ast.Assign]:
    candidates = [
        assignment
        for assignment in _assignment_nodes(tree, "B")
        if {"blur_mask", "intensity"}
        <= {
            node.id
            for node in ast.walk(assignment.value)
            if isinstance(node, ast.Name)
        }
    ]
    if len(candidates) != 1:
        raise UnsupportedReferencePhysicsError(
            f"expected one active local-intensity B assignment, found {len(candidates)}"
        )
    assignment = candidates[0]
    expression = assignment.value
    if not (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)):
        raise UnsupportedReferencePhysicsError(
            "active B relation is no longer a supported affine expression"
        )
    product, intercept_node = expression.left, expression.right
    if not (
        isinstance(product, ast.BinOp)
        and isinstance(product.op, ast.Mult)
        and isinstance(product.left, ast.Constant)
        and isinstance(intercept_node, ast.Constant)
    ):
        raise UnsupportedReferencePhysicsError(
            "active B relation must be slope * local_intensity + intercept"
        )
    slope = _finite_float(product.left.value, "B slope")
    intercept = _finite_float(intercept_node.value, "B intercept")
    line = source_lines[assignment.lineno - 1]
    comment = line.partition("#")[2].strip()
    label = f"source comment: {comment}" if comment else "unlabeled active B relation"
    return slope, intercept, product.right, label, assignment


def _extract_tempo_sigma_scale(tree: ast.Module) -> float:
    assignment = _active_assignment(tree, "TEMPO_kernel")
    call = assignment.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func) == "cv2.getGaussianKernel"
        and len(call.args) == 2
    ):
        raise UnsupportedReferencePhysicsError(
            "TEMPO Gaussian-kernel construction is unsupported"
        )
    sigma = call.args[1]
    if isinstance(sigma, ast.Name) and sigma.id == "TEMPO_sigma":
        return 1.0
    if (
        isinstance(sigma, ast.BinOp)
        and isinstance(sigma.op, ast.Mult)
        and isinstance(sigma.left, ast.Name)
        and sigma.left.id == "TEMPO_sigma"
        and isinstance(sigma.right, ast.Constant)
    ):
        return _finite_float(sigma.right.value, "TEMPO Gaussian sigma scale")
    raise UnsupportedReferencePhysicsError(
        "TEMPO Gaussian sigma must be TEMPO_sigma times one numeric scale"
    )


def _validate_equations(
    tree: ast.Module, source_lines: list[str]
) -> dict[str, Any]:
    slope, intercept, local_intensity, b_label, b_assignment = _extract_linear_b(
        tree, source_lines
    )
    if not (
        isinstance(local_intensity, ast.BinOp)
        and isinstance(local_intensity.op, ast.Mult)
        and isinstance(local_intensity.right, ast.Name)
        and local_intensity.right.id == "intensity"
        and isinstance(local_intensity.left, ast.BinOp)
        and isinstance(local_intensity.left.op, ast.Div)
        and isinstance(local_intensity.left.right, ast.Constant)
    ):
        raise UnsupportedReferencePhysicsError(
            "local intensity must be clamped grayscale / numeric scale * intensity"
        )
    grayscale_max = _finite_float(
        local_intensity.left.right.value, "projector grayscale scale"
    )
    if grayscale_max <= 0:
        raise ReferenceResolutionError("projector grayscale scale must be positive")
    grayscale_clamp_min = _extract_clamp_min(local_intensity)
    energy = _active_assignment(tree, "energy")
    if not (
        isinstance(energy.value, ast.BinOp)
        and isinstance(energy.value.op, ast.Mult)
        and isinstance(energy.value.right, ast.Name)
        and energy.value.right.id == "dt"
        and _same_expression(energy.value.left, local_intensity)
    ):
        raise UnsupportedReferencePhysicsError(
            "Energy must equal active normalized local intensity * intensity * dt"
        )
    _expect_expression(tree, "O2next", "torch.clamp(O2_diffused - energy, min=0)")
    _expect_expression(
        tree,
        "TEMPOnext",
        "torch.where(O2next <= 0, torch.clamp(TEMPO_diffused - energy, min=0), TEMPO_diffused)",
    )
    _expect_expression(
        tree,
        "Dosenext",
        "torch.where((O2next <= 0) & (TEMPOnext <= 0), Dose[-1] + energy - O2_diffused - TEMPO_diffused, Dose[-1])",
    )
    exposure_time = _active_assignment(tree, "t")
    if not (
        isinstance(exposure_time.value, ast.BinOp)
        and isinstance(exposure_time.value.op, ast.Div)
        and _same_expression(exposure_time.value.left, _ast_expression("Dosenext"))
        and _same_expression(exposure_time.value.right, local_intensity)
    ):
        raise UnsupportedReferencePhysicsError(
            "exposure time must equal Dosenext / active local intensity"
        )
    doc_assignment = _expect_expression(
        tree,
        "DoCnext",
        "torch.where((O2next <= 0) & (TEMPOnext <= 0), 1 - torch.exp(-(B * t).clamp(min=0)), DoC[-1])",
    )

    o2_assignment = _active_assignment(tree, "O2_diffused")
    o2_diffusion_enabled = "F.conv2d" in ast.unparse(o2_assignment.value)
    if o2_diffusion_enabled and not _same_expression(
        o2_assignment.value, _ast_expression("F.conv2d(O2_padded, O2_diff)[0, 0]")
    ):
        raise UnsupportedReferencePhysicsError(
            f"unsupported O2 convolution: {ast.unparse(o2_assignment.value)}"
        )
    if not o2_diffusion_enabled and not _same_expression(
        o2_assignment.value, _ast_expression("O2[-1]")
    ):
        raise UnsupportedReferencePhysicsError(
            f"unsupported O2 diffusion expression: {ast.unparse(o2_assignment.value)}"
        )
    tempo_assignment = _active_assignment(tree, "TEMPO_diffused")
    tempo_diffusion_enabled = "F.conv2d" in ast.unparse(tempo_assignment.value)
    if tempo_diffusion_enabled and not _same_expression(
        tempo_assignment.value,
        _ast_expression("F.conv2d(TEMPO_padded, TEMPO_diff)[0, 0]"),
    ):
        raise UnsupportedReferencePhysicsError(
            f"unsupported TEMPO convolution: {ast.unparse(tempo_assignment.value)}"
        )
    if not tempo_diffusion_enabled and not _same_expression(
        tempo_assignment.value, _ast_expression("TEMPO[-1]")
    ):
        raise UnsupportedReferencePhysicsError(
            "unsupported TEMPO diffusion expression: "
            + ast.unparse(tempo_assignment.value)
        )
    _expect_expression(tree, "blur_mask", "F.conv2d(opt_mask_padded, ls)[0, 0]")
    _expect_expression(
        tree,
        "opt_mask_padded",
        "F.pad(opt_mask_pre, pad=(ls_pad, ls_pad, ls_pad, ls_pad), mode='reflect')",
    )
    _expect_expression(
        tree,
        "O2_padded",
        "F.pad(O2_pre, pad=(O2_pad, O2_pad, O2_pad, O2_pad), mode='reflect')",
    )
    _expect_expression(
        tree,
        "TEMPO_padded",
        "F.pad(TEMPO_pre, pad=(TEMPO_pad, TEMPO_pad, TEMPO_pad, TEMPO_pad), mode='reflect')",
    )
    _expect_expression(
        tree,
        "O2_kernel_size",
        "int((O2_sigma - 0.8) / 0.3 + 1) * 2 + 1",
    )
    _expect_expression(
        tree,
        "TEMPO_kernel_size",
        "int((TEMPO_sigma - 0.8) / 0.3 + 1) * 2 + 1",
    )
    _expect_expression(
        tree,
        "ls_kernel_size",
        "int(blur_size / dx) if int(blur_size / dx) % 2 != 0 else int(blur_size / dx) + 1",
    )
    _expect_expression(
        tree,
        "ls_sigma",
        "0.3 * ((ls_kernel_size - 1) * 0.5 - 1) + 0.8",
    )
    _expect_expression(
        tree, "O2_kernel", "cv2.getGaussianKernel(O2_kernel_size, O2_sigma)"
    )
    _expect_expression(
        tree, "ls_kernel", "cv2.getGaussianKernel(ls_kernel_size, ls_sigma)"
    )
    for name, diffusivity in (
        ("O2_sigma", "O2_dfsvty"),
        ("TEMPO_sigma", "TEMPO_dfsvty"),
    ):
        sigma_assignments = _assignment_nodes(tree, name)
        expected_sigma = (
            f"(2 * {diffusivity} * dt) ** 0.5",
            f"{name} / dx",
        )
        if len(sigma_assignments) != 2 or any(
            not _same_expression(assignment.value, _ast_expression(expected))
            for assignment, expected in zip(sigma_assignments, expected_sigma)
        ):
            actual = [ast.unparse(item.value) for item in sigma_assignments]
            raise UnsupportedReferencePhysicsError(
                f"unsupported {name} construction: {actual}"
            )

    noise_enabled = any(
        isinstance(assignment.value, ast.BinOp)
        and isinstance(assignment.value.op, ast.Mult)
        and _same_expression(assignment.value.left, _ast_expression("B"))
        and _same_expression(assignment.value.right, _ast_expression("B_noise"))
        for assignment in _assignment_nodes(tree, "B")
    )
    if noise_enabled:
        _expect_expression(
            tree,
            "B_noise",
            "(1 + chainGrowth_noise_std * torch.randn(H, W, device=device)).clamp(min=0.001)",
        )
    selected = [
        energy.value,
        b_assignment.value,
        o2_assignment.value,
        tempo_assignment.value,
        _active_assignment(tree, "O2next").value,
        _active_assignment(tree, "TEMPOnext").value,
        _active_assignment(tree, "Dosenext").value,
        exposure_time.value,
        doc_assignment.value,
        _active_assignment(tree, "blur_mask").value,
        _active_assignment(tree, "O2_kernel_size").value,
        _active_assignment(tree, "TEMPO_kernel_size").value,
        _active_assignment(tree, "ls_kernel_size").value,
        _active_assignment(tree, "ls_sigma").value,
    ]
    structure_bytes = "\n".join(
        ast.dump(node, include_attributes=False) for node in selected
    ).encode("utf-8")
    return {
        "b_slope": slope,
        "b_intercept": intercept,
        "b_condition_label": b_label,
        "local_intensity": local_intensity,
        "minimum_grayscale": grayscale_clamp_min,
        "mask_grayscale_max": grayscale_max,
        "o2_diffusion_enabled": o2_diffusion_enabled,
        "tempo_diffusion_enabled": tempo_diffusion_enabled,
        "tempo_sigma_scale": _extract_tempo_sigma_scale(tree),
        "chain_growth_noise_enabled": noise_enabled,
        "structure_sha256": _sha256(structure_bytes),
        "doc_formula": ast.unparse(doc_assignment.value),
    }


def _load_doc_fit_catalog() -> tuple[str, tuple[DoCFitCalibration, ...]]:
    """Load, hash, and fully validate the notebook calibration export."""

    if not DOC_FIT_PATH.is_file():
        raise FileNotFoundError(
            f"required DoC calibration export is missing: {DOC_FIT_PATH}. "
            "Run the fit and export cells in DoC curve.ipynb."
        )
    raw = DOC_FIT_PATH.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceResolutionError(
            f"invalid DoC calibration JSON {DOC_FIT_PATH}: {error}"
        ) from error
    required_document = {
        "schema_version",
        "calibration_id",
        "source_notebook",
        "exported_at_utc",
        "averaging_method",
        "conditions",
    }
    missing = sorted(required_document - document.keys())
    if missing:
        raise ReferenceResolutionError(
            f"DoC calibration is missing required fields: {missing}"
        )
    if document["schema_version"] != DOC_FIT_SCHEMA_VERSION:
        raise ReferenceResolutionError(
            f"unsupported DoC schema_version {document['schema_version']!r}; "
            f"expected {DOC_FIT_SCHEMA_VERSION}"
        )
    if document["source_notebook"] != "DoC curve.ipynb":
        raise ReferenceResolutionError(
            "DoC source_notebook must be 'DoC curve.ipynb'"
        )
    try:
        datetime.fromisoformat(str(document["exported_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ReferenceResolutionError("DoC exported_at_utc is not ISO-8601") from error
    if not isinstance(document["conditions"], list) or not document["conditions"]:
        raise ReferenceResolutionError("DoC conditions must be a non-empty list")

    parsed: list[DoCFitCalibration] = []
    condition_ids: set[str] = set()
    for index, condition in enumerate(document["conditions"]):
        required_condition = {
            "condition_id",
            "condition_label",
            "intensity_mw_cm2",
            "tempo_concentration_mM",
            "formula_id",
            "formula",
            "source_data_file",
            "source_sheet",
            "replicates",
            "average",
        }
        if not isinstance(condition, dict):
            raise ReferenceResolutionError(f"DoC condition {index} must be an object")
        missing = sorted(required_condition - condition.keys())
        if missing:
            raise ReferenceResolutionError(
                f"DoC condition {index} is missing required fields: {missing}"
            )
        condition_id = str(condition["condition_id"])
        if not condition_id or condition_id in condition_ids:
            raise ReferenceResolutionError(
                f"empty or duplicate DoC condition_id {condition_id!r}"
            )
        condition_ids.add(condition_id)
        replicate_records = condition["replicates"]
        if not isinstance(replicate_records, list) or not replicate_records:
            raise ReferenceResolutionError(
                f"{condition_id}.replicates must be a non-empty list"
            )
        replicates: list[DoCFitReplicate] = []
        for replicate_index, replicate in enumerate(replicate_records):
            label = f"{condition_id}.replicates[{replicate_index}]"
            required_replicate = {
                "label",
                "source_column",
                "sample_count",
                "a",
                "b",
                "c",
            }
            if not isinstance(replicate, dict):
                raise ReferenceResolutionError(f"{label} must be an object")
            missing = sorted(required_replicate - replicate.keys())
            if missing:
                raise ReferenceResolutionError(f"{label} is missing fields: {missing}")
            sample_count = int(replicate["sample_count"])
            if sample_count < 1:
                raise ReferenceResolutionError(f"{label}.sample_count must be positive")
            replicates.append(
                DoCFitReplicate(
                    label=str(replicate["label"]),
                    source_column=str(replicate["source_column"]),
                    sample_count=sample_count,
                    a=_finite_float(replicate["a"], f"{label}.a"),
                    b=_finite_float(replicate["b"], f"{label}.b"),
                    c=_finite_float(replicate["c"], f"{label}.c"),
                )
            )
        average = condition["average"]
        if not isinstance(average, dict) or any(
            name not in average for name in ("a", "b", "c")
        ):
            raise ReferenceResolutionError(
                f"{condition_id}.average must contain a, b, and c"
            )
        averaged = {
            name: _finite_float(average[name], f"{condition_id}.average.{name}")
            for name in ("a", "b", "c")
        }
        for name in ("a", "b", "c"):
            calculated = sum(getattr(item, name) for item in replicates) / len(replicates)
            if not math.isclose(
                averaged[name], calculated, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ReferenceResolutionError(
                    f"{condition_id}.average.{name} does not match replicate mean"
                )
        parsed.append(
            DoCFitCalibration(
                calibration_id=str(document["calibration_id"]),
                condition_id=condition_id,
                condition_label=str(condition["condition_label"]),
                intensity_mw_cm2=_finite_float(
                    condition["intensity_mw_cm2"], f"{condition_id}.intensity"
                ),
                tempo_concentration_mM=_finite_float(
                    condition["tempo_concentration_mM"], f"{condition_id}.TEMPO"
                ),
                formula_id=str(condition["formula_id"]),
                formula=str(condition["formula"]),
                source_notebook=str(document["source_notebook"]),
                source_data_file=str(condition["source_data_file"]),
                source_sheet=str(condition["source_sheet"]),
                exported_at_utc=str(document["exported_at_utc"]),
                averaging_method=str(document["averaging_method"]),
                replicates=tuple(replicates),
                a=averaged["a"],
                b=averaged["b"],
                c=averaged["c"],
            )
        )
    return _sha256(raw), tuple(parsed)


def _select_doc_fit(
    catalog: tuple[DoCFitCalibration, ...],
    intensity_mw_cm2: float,
    tempo_concentration_mM: float | None,
) -> tuple[DoCFitCalibration | None, str]:
    if tempo_concentration_mM is None:
        return None, "not_selected:active_reference_has_no_numeric_TEMPO_condition"
    matches = [
        condition
        for condition in catalog
        if math.isclose(
            condition.intensity_mw_cm2,
            intensity_mw_cm2,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            condition.tempo_concentration_mM,
            tempo_concentration_mM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) == 1:
        return matches[0], "exact_intensity_and_tempo_match"
    if matches:
        return None, f"ambiguous_exact_match_count={len(matches)}"
    return None, (
        "no_exact_fit_for_active_reference_condition:"
        f"{intensity_mw_cm2:g}mW_{tempo_concentration_mM:g}mM_TEMPO"
    )


def load_reference_config() -> ReferenceConfig:
    """Parse and validate the active collaborator reference without executing it."""

    if not REFERENCE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"collaborator reference model is missing: {REFERENCE_MODEL_PATH}"
        )
    source_bytes = REFERENCE_MODEL_PATH.read_bytes()
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReferenceResolutionError(
            f"reference model must be UTF-8 source: {REFERENCE_MODEL_PATH}"
        ) from error
    try:
        tree = ast.parse(source, filename=str(REFERENCE_MODEL_PATH))
    except SyntaxError as error:
        raise ReferenceResolutionError(
            f"reference model cannot be parsed safely: {error}"
        ) from error

    values = _top_level_values(tree)
    equations = _validate_equations(tree, source.splitlines())
    dx = _required_numeric(values, "dx")
    dy = _required_numeric(values, "dy")
    if not math.isclose(dx, dy, rel_tol=0.0, abs_tol=1e-15):
        raise ReferenceResolutionError(
            f"anisotropic reference pitch is unsupported: dx={dx}, dy={dy}"
        )
    dt = _required_numeric(values, "dt")
    intensity = _required_numeric(values, "intensity")
    total_steps = _required_numeric(values, "total_steps")
    if not total_steps.is_integer() or total_steps < 1:
        raise ReferenceResolutionError(
            f"total_steps must be a positive integer, got {total_steps}"
        )
    loss_steps = tuple(
        _required_numeric(values, name)
        for name in ("tstepT0", "tstepT1", "tstepT2")
    )
    tempo_condition = next(
        (
            _finite_float(values[name], name)
            for name in (
                "TEMPO_concentration_mM",
                "TEMPO_concentration",
                "tempo_concentration_mM",
            )
            if name in values
        ),
        None,
    )
    calibration_sha, calibration_catalog = _load_doc_fit_catalog()
    doc_fit, selection_status = _select_doc_fit(
        calibration_catalog, intensity, tempo_condition
    )
    o2_inhibition = _required_numeric(values, "O2inhibition")
    total_inhibition = _required_numeric(values, "Totalinhibtion")
    tempo_inhibition = _required_numeric(values, "TEMPOinhibition")
    expected_tempo_inhibition = max(0.0, total_inhibition - o2_inhibition)
    if not math.isclose(
        tempo_inhibition,
        expected_tempo_inhibition,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReferenceResolutionError(
            "active TEMPOinhibition no longer equals "
            "max(0, Totalinhibtion - O2inhibition)"
        )

    return ReferenceConfig(
        reference_model_source=REFERENCE_MODEL_PATH.name,
        reference_model_path=str(REFERENCE_MODEL_PATH),
        reference_model_sha256=_sha256(source_bytes),
        reference_structure_sha256=equations["structure_sha256"],
        model_structure_version=SUPPORTED_MODEL_STRUCTURE_VERSION,
        native_shape=CONTROLLER_NATIVE_SHAPE,
        native_pixel_pitch_m=dx,
        projector_refinement=1,
        dt=dt,
        total_simulation_time_s=total_steps * dt,
        loss_sample_times_s=tuple(step * dt for step in loss_steps),
        intensity_mw_cm2=intensity,
        tempo_concentration_mM=tempo_condition,
        o2_diffusivity_m2_s=_required_numeric(values, "O2_dfsvty"),
        tempo_diffusivity_m2_s=_required_numeric(values, "TEMPO_dfsvty"),
        o2_inhibition_mj_cm2=o2_inhibition,
        total_inhibition_mj_cm2=total_inhibition,
        scattering_blur_size_m=_required_numeric(values, "blur_size"),
        tempo_gaussian_sigma_scale=equations["tempo_sigma_scale"],
        o2_diffusion_enabled=equations["o2_diffusion_enabled"],
        tempo_diffusion_enabled=equations["tempo_diffusion_enabled"],
        chain_growth_noise_std=_required_numeric(values, "chainGrowth_noise_std"),
        chain_growth_noise_enabled=equations["chain_growth_noise_enabled"],
        b_slope=equations["b_slope"],
        b_intercept=equations["b_intercept"],
        b_condition_label=equations["b_condition_label"],
        minimum_normalized_intensity=(
            equations["minimum_grayscale"] / equations["mask_grayscale_max"]
        ),
        division_epsilon=(
            equations["minimum_grayscale"]
            / equations["mask_grayscale_max"]
            * intensity
        ),
        mask_grayscale_max=equations["mask_grayscale_max"],
        diffusion_kernel_size_formula=ast.unparse(
            _active_assignment(tree, "O2_kernel_size").value
        ),
        scattering_kernel_size_formula=ast.unparse(
            _active_assignment(tree, "ls_kernel_size").value
        ),
        scattering_sigma_formula=ast.unparse(
            _active_assignment(tree, "ls_sigma").value
        ),
        doc_model_id="reference_intensity_dependent_B_v2",
        doc_model_formula=equations["doc_formula"],
        doc_fit_applied_to_governing_law=False,
        doc_calibration_source="DoC curve.ipynb -> doc_fit_parameters.json",
        doc_calibration_path=str(DOC_FIT_PATH),
        doc_calibration_sha256=calibration_sha,
        doc_fit_selection_status=selection_status,
        available_doc_fit_condition_ids=tuple(
            condition.condition_id for condition in calibration_catalog
        ),
        doc_fit=doc_fit,
    )


def _unique_commented_condition_value(
    source: str, pattern: str, label: str
) -> tuple[float, ...]:
    """Extract one explicitly labelled alternative without executing source."""

    matches = re.findall(pattern, source, flags=re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ReferenceResolutionError(
            f"expected one authoritative {label} comment, found {len(matches)}"
        )
    values = matches[0]
    if isinstance(values, str):
        values = (values,)
    return tuple(_finite_float(value, label) for value in values)


def load_reference_config_for_condition(reference_id: str) -> ReferenceConfig:
    """Resolve a validated condition configuration through the read-only adapter.

    The active collaborator script is the 30 mW / 0 mM configuration.  Its source
    also explicitly labels a 5 mM total-inhibition value and a 5 mM alternative
    B/intensity relation.  This selector validates those comments on every load,
    reuses the script's active generic TEMPO diffusivity, and never edits or
    executes ``AIE_TEMPOv1.1.py``.
    """

    if reference_id not in SUPPORTED_FORWARD_CONDITIONS:
        raise ReferenceResolutionError(
            f"unsupported forward-physics condition {reference_id!r}; supported: "
            f"{list(SUPPORTED_FORWARD_CONDITIONS)}"
        )
    base = load_reference_config()
    if not math.isclose(
        base.intensity_mw_cm2, 30.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ReferenceResolutionError(
            "condition selector requires authoritative intensity=30 mW/cm^2"
        )

    normalized_label = base.b_condition_label.lower().replace(" ", "")
    if reference_id == "30mW_0mM":
        if base.tempo_inhibition_mj_cm2 != 0.0 or "0mmtempo" not in normalized_label:
            raise ReferenceResolutionError(
                "active authoritative source is no longer the validated 0 mM "
                "TEMPO configuration"
            )
        return replace(
            base,
            tempo_concentration_mM=0.0,
            doc_fit_selection_status=(
                "not_selected:legacy_doc_fit_parameters_not_used_for_runtime_tracking"
            ),
        )

    source = REFERENCE_MODEL_PATH.read_text(encoding="utf-8")
    (total_inhibition,) = _unique_commented_condition_value(
        source,
        r"^\s*#\s*([-+0-9.eE]+)\s+for\s+5\s*mmol\s+TEMPO\s+concentration\s*$",
        "5 mM total inhibition",
    )
    b_slope, b_intercept = _unique_commented_condition_value(
        source,
        r"^\s*#\s*B\s*=\s*([-+0-9.eE]+)\s*\*\s*\([^\r\n]+\)\s*"
        r"\+\s*([-+0-9.eE]+)\s*#\s*5\s*mM\s*TEMPO\s*$",
        "5 mM B/intensity relation",
    )
    if total_inhibition <= base.o2_inhibition_mj_cm2:
        raise ReferenceResolutionError(
            "5 mM total inhibition must exceed the active O2 inhibition"
        )
    if b_slope <= 0.0 or b_intercept < 0.0:
        raise ReferenceResolutionError(
            "5 mM B/intensity coefficients must be physically nonnegative"
        )
    return replace(
        base,
        tempo_concentration_mM=5.0,
        total_inhibition_mj_cm2=total_inhibition,
        b_slope=b_slope,
        b_intercept=b_intercept,
        b_condition_label=(
            "5mMTEMPO (validated commented alternative in AIE_TEMPOv1.1.py)"
        ),
        doc_fit_selection_status=(
            "not_selected:legacy_doc_fit_parameters_not_used_for_runtime_tracking"
        ),
    )


def _full_gaussian_torch(field: Any, kernel_2d: Any) -> Any:
    import torch.nn.functional as functional

    padding = int(kernel_2d.shape[-1] // 2)
    if padding == 0:
        return field
    padded = functional.pad(
        field[None, None],
        (padding, padding, padding, padding),
        mode="reflect",
    )
    return functional.conv2d(padded, kernel_2d)[0, 0]


def reference_step_from_scattered_torch(
    *,
    o2: Any,
    tempo: Any,
    dose: Any,
    doc: Any,
    scattered_normalized_mask: Any,
    o2_kernel_2d: Any,
    tempo_kernel_2d: Any,
    chain_growth_multiplier: Any | None = None,
    config: ReferenceConfig | None = None,
) -> ReferenceStepResult:
    """Independent controller-side fixture mirroring validated active equations."""

    import torch

    reference = config or load_reference_config()
    local_intensity = scattered_normalized_mask.clamp_min(
        reference.minimum_normalized_intensity
    ) * reference.intensity_mw_cm2
    energy = local_intensity * reference.dt
    b = reference.b_slope * local_intensity + reference.b_intercept
    if chain_growth_multiplier is not None:
        b = b * chain_growth_multiplier
    elif reference.chain_growth_noise_std > 0:
        raise ReferenceResolutionError(
            "chain_growth_multiplier is required for nonzero reference B noise"
        )
    o2_diffused = (
        _full_gaussian_torch(o2, o2_kernel_2d)
        if reference.o2_diffusion_enabled
        else o2
    )
    tempo_diffused = (
        _full_gaussian_torch(tempo, tempo_kernel_2d)
        if reference.tempo_diffusion_enabled
        else tempo
    )
    o2_next = torch.clamp(o2_diffused - energy, min=0.0)
    tempo_next = torch.where(
        o2_next <= 0,
        torch.clamp(tempo_diffused - energy, min=0.0),
        tempo_diffused,
    )
    curing = (o2_next <= 0) & (tempo_next <= 0)
    dose_next = torch.where(
        curing,
        dose + energy - o2_diffused - tempo_diffused,
        dose,
    )
    exposure_time = dose_next / local_intensity.clamp_min(
        reference.division_epsilon
    )
    doc_candidate = 1.0 - torch.exp(-torch.clamp(b * exposure_time, min=0.0))
    doc_next = torch.where(curing, doc_candidate, doc)
    return ReferenceStepResult(
        o2=o2_next,
        tempo=tempo_next,
        dose=dose_next,
        doc=doc_next,
        o2_diffused=o2_diffused,
        tempo_diffused=tempo_diffused,
    )


def reference_step_torch(
    *,
    o2: Any,
    tempo: Any,
    dose: Any,
    doc: Any,
    normalized_mask: Any,
    scattering_kernel_2d: Any,
    o2_kernel_2d: Any,
    tempo_kernel_2d: Any,
    chain_growth_multiplier: Any | None = None,
    config: ReferenceConfig | None = None,
) -> ReferenceStepResult:
    """Apply scattering then the independent validated reference fixture."""

    scattered = _full_gaussian_torch(normalized_mask, scattering_kernel_2d)
    return reference_step_from_scattered_torch(
        o2=o2,
        tempo=tempo,
        dose=dose,
        doc=doc,
        scattered_normalized_mask=scattered,
        o2_kernel_2d=o2_kernel_2d,
        tempo_kernel_2d=tempo_kernel_2d,
        chain_growth_multiplier=chain_growth_multiplier,
        config=config,
    )
