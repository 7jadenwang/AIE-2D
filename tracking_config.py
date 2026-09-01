"""Central validation and scheduling for MPC target-side tracking specifications."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from doc_reference import DoCReferenceCurve


TRACKING_MODES = ("curve", "sampled-curve", "checkpoints")
SPATIAL_DEFINITIONS = ("pixelwise", "target-mean")
TRACKING_LOSSES = ("mse", "mae", "huber")
TRACKING_VARIABLES = ("doc", "reaction-progress")


class TrackingConfigurationError(ValueError):
    """Raised before optimization when a tracking specification is inconsistent."""


def doc_to_reaction_progress(required_doc: float) -> float:
    """Convert a finite DoC requirement in ``[0, 1)`` to reaction progress."""

    value = float(required_doc)
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise TrackingConfigurationError(
            "reaction-progress tracking requires every requested DoC to be "
            f"finite and in [0, 1); got {required_doc!r}"
        )
    return -math.log1p(-value)


def parse_float_list(text: str | None, *, label: str) -> tuple[float, ...]:
    if text is None or not text.strip():
        return ()
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise TrackingConfigurationError(f"{label} must be a comma-separated numeric list") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise TrackingConfigurationError(f"{label} must contain finite values")
    return values


def parse_checkpoints(text: str | None) -> tuple[tuple[float, float], ...]:
    if text is None or not text.strip():
        return ()
    records: list[tuple[float, float]] = []
    for item in text.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise TrackingConfigurationError(
                "--checkpoints must use 'time:DoC,time:DoC,...'"
            )
        try:
            records.append((float(fields[0]), float(fields[1])))
        except ValueError as error:
            raise TrackingConfigurationError("checkpoint times and DoC values must be numeric") from error
    return tuple(records)


def validate_grid_alignment(times_s: Sequence[float], control_dt_s: float, *, label: str) -> None:
    if not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise TrackingConfigurationError("control_dt_s must be finite and positive")
    for time_s in times_s:
        grid_index = round(float(time_s) / control_dt_s)
        aligned = grid_index * control_dt_s
        if not math.isclose(float(time_s), aligned, rel_tol=0.0, abs_tol=1e-9):
            raise TrackingConfigurationError(
                f"{label} time {time_s:g} s is not aligned to control_dt={control_dt_s:g} s"
            )


def evenly_spaced_tracking_times(start_s: float, end_s: float, count: int) -> tuple[float, ...]:
    if count < 2:
        raise TrackingConfigurationError("--num-tracking-points must be at least 2")
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or end_s <= start_s:
        raise TrackingConfigurationError("tracking-point start/end must satisfy 0 <= start < end")
    return tuple(float(value) for value in np.linspace(start_s, end_s, count))


def resolve_sampled_tracking_times(
    *,
    explicit_times_s: Sequence[float] = (),
    count: int | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Resolve sampled-curve times; explicit times have documented precedence."""

    explicit = tuple(float(value) for value in explicit_times_s)
    if explicit:
        warnings = (
            ("explicit --tracking-times take precedence over --num-tracking-points",)
            if count is not None
            else ()
        )
        return explicit, warnings
    if count is None:
        raise TrackingConfigurationError(
            "sampled-curve mode requires --tracking-times or --num-tracking-points with explicit --tracking-start/--tracking-end"
        )
    if start_s is None or end_s is None:
        raise TrackingConfigurationError(
            "--num-tracking-points requires explicit --tracking-start and --tracking-end"
        )
    return evenly_spaced_tracking_times(start_s, end_s, count), ()


@dataclass(frozen=True)
class StageTrackingSchedule:
    times_s: np.ndarray
    required_doc: np.ndarray
    active: np.ndarray
    point_weights: np.ndarray


@dataclass(frozen=True)
class TrackingSpecification:
    """One unambiguous tracking specification for an entire MPC run."""

    tracking_mode: str = "curve"
    reference_curve: DoCReferenceCurve | None = None
    point_times_s: tuple[float, ...] = ()
    point_values: tuple[float, ...] = ()
    point_weights: tuple[float, ...] = ()
    spatial_definition: str = "pixelwise"
    tracking_loss: str = "mse"
    huber_delta: float = 0.1
    source: str = "curve_artifact"
    tracking_variable: str = "doc"

    def __post_init__(self) -> None:
        if self.tracking_mode not in TRACKING_MODES:
            raise TrackingConfigurationError(f"tracking_mode must be one of {TRACKING_MODES}")
        if self.spatial_definition not in SPATIAL_DEFINITIONS:
            raise TrackingConfigurationError(f"tracking spatial definition must be one of {SPATIAL_DEFINITIONS}")
        if self.tracking_loss not in TRACKING_LOSSES:
            raise TrackingConfigurationError(f"tracking loss must be one of {TRACKING_LOSSES}")
        if self.tracking_variable not in TRACKING_VARIABLES:
            raise TrackingConfigurationError(
                f"tracking variable must be one of {TRACKING_VARIABLES}"
            )
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise TrackingConfigurationError("Huber delta must be finite and positive")
        if self.tracking_mode == "curve":
            if self.reference_curve is None:
                raise TrackingConfigurationError("curve mode requires a DoC reference")
            if self.point_times_s or self.point_values or self.point_weights:
                raise TrackingConfigurationError("curve mode must not define sparse tracking points")
            return
        if self.tracking_mode == "sampled-curve" and self.reference_curve is None:
            raise TrackingConfigurationError("sampled-curve mode requires a DoC reference")
        if self.tracking_mode == "checkpoints" and self.reference_curve is not None:
            raise TrackingConfigurationError("checkpoint mode must not load or mix a fitted curve")
        if not self.point_times_s:
            raise TrackingConfigurationError(f"{self.tracking_mode} mode requires tracking times")
        if len(self.point_times_s) != len(self.point_values):
            raise TrackingConfigurationError("tracking times and values must have equal length")
        if self.point_weights and len(self.point_weights) != len(self.point_times_s):
            raise TrackingConfigurationError("point weights must match the number of tracking points")
        if any(not math.isfinite(t) or t <= 0 for t in self.point_times_s):
            raise TrackingConfigurationError("sparse tracking times must be finite and positive")
        if tuple(sorted(self.point_times_s)) != self.point_times_s or len(set(self.point_times_s)) != len(self.point_times_s):
            raise TrackingConfigurationError("sparse tracking times must be strictly increasing and unique")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.point_values):
            raise TrackingConfigurationError("tracking DoC values must be finite and in [0,1]")
        if self.tracking_variable == "reaction-progress":
            for value in self.point_values:
                doc_to_reaction_progress(value)
        if self.point_weights and any(not math.isfinite(weight) or weight <= 0 for weight in self.point_weights):
            raise TrackingConfigurationError("point weights must be finite and positive")

    @classmethod
    def curve(
        cls,
        reference_curve: DoCReferenceCurve,
        *,
        spatial_definition: str = "pixelwise",
        tracking_loss: str = "mse",
        tracking_variable: str = "doc",
        huber_delta: float = 0.1,
    ) -> "TrackingSpecification":
        return cls(
            tracking_mode="curve",
            reference_curve=reference_curve,
            spatial_definition=spatial_definition,
            tracking_loss=tracking_loss,
            tracking_variable=tracking_variable,
            huber_delta=huber_delta,
            source="curve_artifact",
        )

    @classmethod
    def sampled_curve(
        cls,
        reference_curve: DoCReferenceCurve,
        times_s: Sequence[float],
        *,
        point_weights: Sequence[float] = (),
        spatial_definition: str = "pixelwise",
        tracking_loss: str = "mse",
        tracking_variable: str = "doc",
        huber_delta: float = 0.1,
    ) -> "TrackingSpecification":
        times = tuple(float(value) for value in times_s)
        values = tuple(float(value) for value in reference_curve.evaluate(np.asarray(times)))
        return cls(
            tracking_mode="sampled-curve",
            reference_curve=reference_curve,
            point_times_s=times,
            point_values=values,
            point_weights=tuple(float(value) for value in point_weights),
            spatial_definition=spatial_definition,
            tracking_loss=tracking_loss,
            tracking_variable=tracking_variable,
            huber_delta=huber_delta,
            source="curve_artifact_sampled_at_explicit_absolute_times",
        )

    @classmethod
    def checkpoints(
        cls,
        checkpoints: Sequence[tuple[float, float]],
        *,
        point_weights: Sequence[float] = (),
        spatial_definition: str = "pixelwise",
        tracking_loss: str = "mse",
        tracking_variable: str = "doc",
        huber_delta: float = 0.1,
    ) -> "TrackingSpecification":
        records = tuple((float(time_s), float(doc)) for time_s, doc in checkpoints)
        return cls(
            tracking_mode="checkpoints",
            point_times_s=tuple(time_s for time_s, _ in records),
            point_values=tuple(doc for _, doc in records),
            point_weights=tuple(float(value) for value in point_weights),
            spatial_definition=spatial_definition,
            tracking_loss=tracking_loss,
            tracking_variable=tracking_variable,
            huber_delta=huber_delta,
            source="collaborator_specification",
        )

    def validate_runtime(self, control_dt_s: float, total_time_s: float, horizon: int) -> tuple[str, ...]:
        if not math.isfinite(total_time_s) or total_time_s <= 0:
            raise TrackingConfigurationError("total process time must be finite and positive")
        if horizon < 1:
            raise TrackingConfigurationError("horizon must be positive")
        if self.tracking_mode == "curve":
            if self.tracking_variable == "reaction-progress":
                assert self.reference_curve is not None
                realized_steps = int(round(total_time_s / control_dt_s))
                last_predicted_time = total_time_s + (horizon - 1) * control_dt_s
                times = control_dt_s * np.arange(
                    1, realized_steps + horizon, dtype=float
                )
                if times.size and times[-1] > last_predicted_time + 1e-9:
                    raise AssertionError("dense tracking validation exceeded final horizon")
                for value in np.asarray(self.reference_curve.evaluate(times), dtype=float):
                    doc_to_reaction_progress(float(value))
            return ()
        validate_grid_alignment(self.point_times_s, control_dt_s, label=self.tracking_mode)
        if self.point_times_s[-1] > total_time_s + 1e-9:
            raise TrackingConfigurationError(
                f"last {self.tracking_mode} time {self.point_times_s[-1]:g} s exceeds total time {total_time_s:g} s"
            )
        gaps = np.diff(np.asarray(self.point_times_s, dtype=float))
        lookahead = horizon * control_dt_s
        warnings = []
        if gaps.size and float(np.max(gaps)) > lookahead + 1e-9:
            warnings.append(
                f"largest sparse tracking gap ({float(np.max(gaps)):.2f} s) exceeds prediction lookahead ({lookahead:.2f} s)"
            )
        return tuple(warnings)

    def stage_schedule(self, current_process_time_s: float, control_dt_s: float, horizon: int) -> StageTrackingSchedule:
        times = current_process_time_s + control_dt_s * np.arange(1, horizon + 1, dtype=float)
        if self.tracking_mode == "curve":
            assert self.reference_curve is not None
            values = np.asarray(self.reference_curve.evaluate(times), dtype=float)
            if self.tracking_variable == "reaction-progress":
                for value in values:
                    doc_to_reaction_progress(float(value))
            return StageTrackingSchedule(times, values, np.ones(horizon, dtype=bool), np.ones(horizon, dtype=float))
        values = np.zeros(horizon, dtype=float)
        active = np.zeros(horizon, dtype=bool)
        weights = np.zeros(horizon, dtype=float)
        configured_weights = self.point_weights or tuple(1.0 for _ in self.point_times_s)
        for point_time, point_value, point_weight in zip(self.point_times_s, self.point_values, configured_weights):
            matches = np.flatnonzero(np.isclose(times, point_time, rtol=0.0, atol=1e-9))
            if matches.size > 1:
                raise AssertionError("one sparse point matched multiple prediction stages")
            if matches.size == 1:
                index = int(matches[0])
                values[index], active[index], weights[index] = point_value, True, point_weight
        return StageTrackingSchedule(times, values, active, weights)

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            "tracking_mode": self.tracking_mode,
            "tracking_reference_type": "explicit_checkpoints" if self.tracking_mode == "checkpoints" else self.tracking_mode,
            "tracking_reference_source": self.source,
            "tracking_times_s": list(self.point_times_s),
            "tracking_values": list(self.point_values),
            "point_weights": list(self.point_weights or tuple(1.0 for _ in self.point_times_s)),
            "point_weighting": "explicit" if self.point_weights else "equal",
            "target_tracking_stage_policy": (
                "all_prediction_stages"
                if self.tracking_mode == "curve"
                else "only_exact_configured_absolute_times"
            ),
            "non_target_penalty_stage_policy": "all_prediction_stages",
            "spatial_definition": self.spatial_definition,
            "tracking_loss": self.tracking_loss,
            "tracking_variable": self.tracking_variable,
            "huber_delta": self.huber_delta if self.tracking_loss == "huber" else None,
            "curve_reference": None if self.reference_curve is None else self.reference_curve.provenance_metadata(),
        }
