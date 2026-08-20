"""Quantitative temporal, DoC, geometry, component, and hole metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


GEOMETRY_THRESHOLD_NOTE = (
    "Computational segmentation threshold only; not claimed as an experimentally "
    "calibrated gel-point threshold."
)


def _as_2d(array: Any, label: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {result.shape}")
    return result


def _mean_square_root(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(values))))) if values.size else 0.0


def temporal_tracking_metrics(actual: Any, reference: Any) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if actual.ndim != 1 or actual.shape != reference.shape or actual.size == 0:
        raise ValueError("actual/reference tracking arrays must be equal nonempty 1D arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("tracking arrays contain NaN or Inf")
    error = actual - reference
    return {
        "rmse": _mean_square_root(error),
        "mae": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def soft_doc_metrics(
    final_doc: Any, target: Any, reference_final_doc: float, target_threshold: float
) -> dict[str, float]:
    final_doc = _as_2d(final_doc, "final_doc").astype(float, copy=False)
    target = _as_2d(target, "target").astype(float, copy=False)
    if final_doc.shape != target.shape:
        raise ValueError("final_doc and target must have the same shape")
    target_region = target > target_threshold
    if not target_region.any():
        raise ValueError("target has no pixels above target_threshold")
    outside_region = ~target_region
    desired = target * float(reference_final_doc)
    error = final_doc - desired
    return {
        "full_image_rmse": _mean_square_root(error),
        "target_region_rmse": _mean_square_root(error[target_region]),
        "outside_region_rmse": _mean_square_root(error[outside_region]),
        "full_image_mae": float(np.mean(np.abs(error))),
    }


def threshold_geometry_metrics(
    final_doc: Any, target: Any, geometry_threshold: float, target_threshold: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    final_doc = _as_2d(final_doc, "final_doc").astype(float, copy=False)
    target = _as_2d(target, "target").astype(float, copy=False)
    if final_doc.shape != target.shape:
        raise ValueError("final_doc and target must have the same shape")
    if not 0.0 <= geometry_threshold <= 1.0:
        raise ValueError("geometry_threshold must lie in [0, 1]")
    if not 0.0 <= target_threshold <= 1.0:
        raise ValueError("target_threshold must lie in [0, 1]")
    target_binary = target > target_threshold
    cured = final_doc >= geometry_threshold
    target_area = int(np.count_nonzero(target_binary))
    if target_area == 0:
        raise ValueError("binary target is empty")
    image_area = int(target_binary.size)
    cured_area = int(np.count_nonzero(cured))
    true_positive = int(np.count_nonzero(cured & target_binary))
    false_positive = int(np.count_nonzero(cured & ~target_binary))
    false_negative = int(np.count_nonzero(~cured & target_binary))
    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union else 1.0
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice = 2 * true_positive / dice_denominator if dice_denominator else 1.0
    precision = true_positive / (true_positive + false_positive) if cured_area else 0.0
    recall = true_positive / target_area
    metrics = {
        "threshold": float(geometry_threshold),
        "threshold_interpretation": GEOMETRY_THRESHOLD_NOTE,
        "target_threshold": float(target_threshold),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "undercure_fraction": false_negative / target_area,
        "overcure_fraction_target_normalized": false_positive / target_area,
        "overcure_fraction_image": false_positive / image_area,
        "area_error_fraction": (cured_area - target_area) / target_area,
        "pixel_counts": {
            "image_area": image_area,
            "target_area": target_area,
            "cured_area": cured_area,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }
    return metrics, target_binary, cured


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def boundary_metrics(
    target_binary: Any, cured_binary: Any, pixel_pitch_um: float
) -> dict[str, Any]:
    target_binary = _as_2d(target_binary, "target_binary").astype(bool, copy=False)
    cured_binary = _as_2d(cured_binary, "cured_binary").astype(bool, copy=False)
    if target_binary.shape != cured_binary.shape:
        raise ValueError("target/cured masks must have the same shape")
    if not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0:
        raise ValueError("pixel_pitch_um must be finite and positive")
    target_boundary = _boundary(target_binary)
    cured_boundary = _boundary(cured_binary)
    if not target_boundary.any() and not cured_boundary.any():
        return {
            "status": "both_masks_empty",
            "mean_symmetric_distance_px": 0.0,
            "mean_symmetric_distance_um": 0.0,
            "p95_symmetric_distance_px": 0.0,
            "p95_symmetric_distance_um": 0.0,
        }
    if not target_boundary.any() or not cured_boundary.any():
        return {
            "status": "undefined_one_boundary_empty",
            "mean_symmetric_distance_px": None,
            "mean_symmetric_distance_um": None,
            "p95_symmetric_distance_px": None,
            "p95_symmetric_distance_um": None,
        }
    distance_to_target = ndimage.distance_transform_edt(~target_boundary)
    distance_to_cured = ndimage.distance_transform_edt(~cured_boundary)
    symmetric = np.concatenate(
        (distance_to_target[cured_boundary], distance_to_cured[target_boundary])
    )
    mean_px = float(np.mean(symmetric))
    p95_px = float(np.percentile(symmetric, 95))
    return {
        "status": "ok",
        "mean_symmetric_distance_px": mean_px,
        "mean_symmetric_distance_um": mean_px * pixel_pitch_um,
        "p95_symmetric_distance_px": p95_px,
        "p95_symmetric_distance_um": p95_px * pixel_pitch_um,
    }


def component_metrics(
    target_binary: Any, final_doc: Any, cured_binary: Any, pixel_pitch_um: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_binary = _as_2d(target_binary, "target_binary").astype(bool, copy=False)
    final_doc = _as_2d(final_doc, "final_doc").astype(float, copy=False)
    cured_binary = _as_2d(cured_binary, "cured_binary").astype(bool, copy=False)
    if target_binary.shape != final_doc.shape or target_binary.shape != cured_binary.shape:
        raise ValueError("component inputs must have the same shape")
    labels, count = ndimage.label(target_binary, structure=np.ones((3, 3), dtype=int))
    rows: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        region = labels == component_id
        values = final_doc[region]
        cured_fraction = float(np.mean(cured_binary[region]))
        area = int(values.size)
        rows.append(
            {
                "component_id": component_id,
                "area_pixels": area,
                "area_um2": area * pixel_pitch_um**2,
                "mean_final_doc": float(np.mean(values)),
                "p05_final_doc": float(np.percentile(values, 5)),
                "cured_fraction": cured_fraction,
                "undercure_fraction": 1.0 - cured_fraction,
            }
        )
    if not rows:
        return {
            "count": 0,
            "worst_mean_doc": None,
            "worst_mean_doc_component_id": None,
            "worst_cured_fraction": None,
            "worst_cured_fraction_component_id": None,
            "worst_undercure_fraction": None,
            "worst_undercure_component_id": None,
        }, rows
    worst_mean = min(rows, key=lambda row: row["mean_final_doc"])
    worst_cured = min(rows, key=lambda row: row["cured_fraction"])
    worst_undercure = max(rows, key=lambda row: row["undercure_fraction"])
    return {
        "count": count,
        "worst_mean_doc": worst_mean["mean_final_doc"],
        "worst_mean_doc_component_id": worst_mean["component_id"],
        "worst_cured_fraction": worst_cured["cured_fraction"],
        "worst_cured_fraction_component_id": worst_cured["component_id"],
        "worst_undercure_fraction": worst_undercure["undercure_fraction"],
        "worst_undercure_component_id": worst_undercure["component_id"],
    }, rows


def hole_metrics(
    target_binary: Any, final_doc: Any, cured_binary: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    target_binary = _as_2d(target_binary, "target_binary").astype(bool, copy=False)
    final_doc = _as_2d(final_doc, "final_doc").astype(float, copy=False)
    cured_binary = _as_2d(cured_binary, "cured_binary").astype(bool, copy=False)
    if target_binary.shape != final_doc.shape or target_binary.shape != cured_binary.shape:
        raise ValueError("hole inputs must have the same shape")
    holes = ndimage.binary_fill_holes(target_binary) & ~target_binary
    labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
    rows: list[dict[str, Any]] = []
    for hole_id in range(1, count + 1):
        region = labels == hole_id
        values = final_doc[region]
        rows.append(
            {
                "hole_id": hole_id,
                "area_pixels": int(values.size),
                "cured_fraction": float(np.mean(cured_binary[region])),
                "mean_doc": float(np.mean(values)),
                "p95_doc": float(np.percentile(values, 95)),
                "max_doc": float(np.max(values)),
            }
        )
    total_pixels = int(np.count_nonzero(holes))
    if total_pixels == 0:
        summary = {
            "count": 0,
            "total_hole_pixels": 0,
            "cured_fraction": 0.0,
            "mean_doc": 0.0,
            "p95_doc": 0.0,
            "max_doc": 0.0,
        }
    else:
        values = final_doc[holes]
        summary = {
            "count": count,
            "total_hole_pixels": total_pixels,
            "cured_fraction": float(np.mean(cured_binary[holes])),
            "mean_doc": float(np.mean(values)),
            "p95_doc": float(np.percentile(values, 95)),
            "max_doc": float(np.max(values)),
        }
    return summary, rows, holes


def calculate_final_metrics(
    final_doc: Any,
    target: Any,
    reference_final_doc: float,
    target_threshold: float,
    geometry_threshold: float,
    pixel_pitch_um: float,
) -> dict[str, Any]:
    """Calculate all final primary-grid metrics in one validated pass."""

    geometry, target_binary, cured_binary = threshold_geometry_metrics(
        final_doc, target, geometry_threshold, target_threshold
    )
    components, component_rows = component_metrics(
        target_binary, final_doc, cured_binary, pixel_pitch_um
    )
    holes, hole_rows, hole_mask = hole_metrics(target_binary, final_doc, cured_binary)
    return {
        "soft_doc": soft_doc_metrics(
            final_doc, target, reference_final_doc, target_threshold
        ),
        "geometry": geometry,
        "boundary": boundary_metrics(target_binary, cured_binary, pixel_pitch_um),
        "components": components,
        "component_rows": component_rows,
        "holes": holes,
        "hole_rows": hole_rows,
        "hole_mask": hole_mask,
        "target_binary": target_binary,
        "cured_binary": cured_binary,
    }

