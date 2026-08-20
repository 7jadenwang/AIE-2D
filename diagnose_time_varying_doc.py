"""Diagnose the legacy DoC law under time-varying projector intensity.

This script does not alter the collaborator reference or introduce fitted
coefficients. It exercises the production ``AIEModel`` from a shared,
post-inhibition state and compares its incremental history with the legacy
static-formula evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from aie_model import DOC_HISTORY_MODE, AIEModel, AIEState
from aie_reference import load_reference_config


@dataclass(frozen=True)
class TransitionResult:
    """Legacy and incremental-kinetics results for one common-state branch."""

    case: str
    mask_level: float
    current_intensity_mw_cm2: float
    dose_before_mj_cm2: float
    dose_after_mj_cm2: float
    delta_dose_mj_cm2: float
    added_energy_mj_cm2: float
    b_per_s: float
    reaction_progress_before: float
    reaction_progress_after: float
    reaction_progress_increment: float
    doc_before: float
    corrected_doc_after: float
    corrected_delta_doc: float
    legacy_apparent_exposure_time_s: float
    legacy_doc_after: float
    legacy_delta_doc: float


@dataclass(frozen=True)
class FixedDoseResult:
    """Direct evaluation of the active DoC expression at one fixed dose."""

    mask_level: float
    current_intensity_mw_cm2: float
    fixed_dose_mj_cm2: float
    apparent_exposure_time_s: float
    b_per_s: float
    b_times_exposure: float
    calculated_doc: float


def _clone_state(state: AIEState) -> AIEState:
    """Make an independent, detached copy for an exact branch reset."""

    multiplier = (
        state.chain_growth_multiplier.detach().clone()
        if state.chain_growth_multiplier is not None
        else None
    )
    return AIEState(
        o2=state.o2.detach().clone(),
        tempo=state.tempo.detach().clone(),
        dose=state.dose.detach().clone(),
        reaction_progress=state.reaction_progress.detach().clone(),
        doc=state.doc.detach().clone(),
        chain_growth_multiplier=multiplier,
    )


def _mean(field: torch.Tensor) -> float:
    return float(field.detach().mean().cpu())


def _validate_state(state: AIEState, label: str) -> None:
    """Fail clearly if the diagnostic encounters invalid model output."""

    for name, field in zip(
        ("O2", "TEMPO", "Dose", "reaction progress", "DoC"),
        state.tensors(),
    ):
        if not bool(torch.isfinite(field).all()):
            raise RuntimeError(f"{label} {name} contains NaN or Inf")
    doc_min = float(state.doc.detach().min().cpu())
    doc_max = float(state.doc.detach().max().cpu())
    if doc_min < -1e-6 or doc_max > 1.0 + 1e-6:
        raise RuntimeError(
            f"{label} DoC lies outside [0, 1]: min={doc_min}, max={doc_max}"
        )


def _uniform_mask(
    model: AIEModel, shape: tuple[int, int], level: float
) -> torch.Tensor:
    control_shape = model.control_shape_for(shape)
    return torch.full(
        control_shape,
        level,
        device=model.device,
        dtype=model.dtype,
    )


def _build_common_prehistory(
    model: AIEModel,
    shape: tuple[int, int],
    high_level: float,
    target_doc: float,
    max_steps: int,
) -> tuple[AIEState, int]:
    """Expose uniformly until inhibition is gone and curing is established."""

    state = model.initialize_state(shape)
    high_mask = _uniform_mask(model, shape, high_level)
    for step_index in range(1, max_steps + 1):
        state = model.step(state, high_mask)
        _validate_state(state, f"prehistory step {step_index}")
        ready = (
            _mean(state.o2) <= 1e-7
            and _mean(state.tempo) <= 1e-7
            and _mean(state.dose) > 0.0
            and _mean(state.doc) >= target_doc
        )
        if ready:
            if _mean(state.doc) >= 0.95:
                raise RuntimeError(
                    "The first prehistory state meeting the requested target is nearly "
                    "saturated; lower --prehistory-target-doc."
                )
            return _clone_state(state), step_index
    raise RuntimeError(
        "Could not create a cured, nonsaturated common state in "
        f"{max_steps} reference physics steps. Increase --max-prehistory-steps "
        "or adjust --prehistory-target-doc."
    )


def _run_transition(
    model: AIEModel,
    common_state: AIEState,
    case: str,
    mask_level: float,
) -> TransitionResult:
    """Advance one physical step from an exact copy of the common state."""

    state_before = _clone_state(common_state)
    mask = _uniform_mask(model, tuple(state_before.shape), mask_level)
    prepared = model.prepare_control(mask, state_before.shape)
    state_after = model.step_prepared(state_before, prepared)
    _validate_state(state_after, case)

    intensity = _mean(prepared.local_intensity)
    if intensity <= 0.0:
        raise RuntimeError(f"{case} produced nonpositive local intensity")
    dose_before = _mean(state_before.dose)
    dose_after = _mean(state_after.dose)
    doc_before = _mean(state_before.doc)
    corrected_doc_after = _mean(state_after.doc)
    reaction_progress_before = _mean(state_before.reaction_progress)
    reaction_progress_after = _mean(state_after.reaction_progress)

    effective_b = prepared.b
    if state_before.chain_growth_multiplier is not None:
        effective_b = effective_b * state_before.chain_growth_multiplier
    safe_intensity = prepared.local_intensity.clamp_min(
        model.params.division_epsilon
    )
    expected_progress_candidate = state_before.reaction_progress + (
        effective_b * (state_after.dose - state_before.dose) / safe_intensity
    )
    curing = (state_after.o2 <= 0) & (state_after.tempo <= 0)
    expected_progress = torch.where(
        curing, expected_progress_candidate, state_before.reaction_progress
    )
    torch.testing.assert_close(
        state_after.reaction_progress,
        expected_progress,
        rtol=1e-6,
        atol=1e-7,
    )
    legacy_exposure = state_after.dose / safe_intensity
    legacy_doc_candidate = 1.0 - torch.exp(
        -torch.clamp(effective_b * legacy_exposure, min=0.0)
    )
    legacy_doc = torch.where(curing, legacy_doc_candidate, state_before.doc)
    legacy_doc_after = _mean(legacy_doc)
    return TransitionResult(
        case=case,
        mask_level=mask_level,
        current_intensity_mw_cm2=intensity,
        dose_before_mj_cm2=dose_before,
        dose_after_mj_cm2=dose_after,
        delta_dose_mj_cm2=dose_after - dose_before,
        added_energy_mj_cm2=_mean(prepared.energy),
        b_per_s=_mean(prepared.b),
        reaction_progress_before=reaction_progress_before,
        reaction_progress_after=reaction_progress_after,
        reaction_progress_increment=(
            reaction_progress_after - reaction_progress_before
        ),
        doc_before=doc_before,
        corrected_doc_after=corrected_doc_after,
        corrected_delta_doc=corrected_doc_after - doc_before,
        legacy_apparent_exposure_time_s=dose_after / intensity,
        legacy_doc_after=legacy_doc_after,
        legacy_delta_doc=legacy_doc_after - doc_before,
    )


def _run_fixed_dose_check(
    model: AIEModel,
    shape: tuple[int, int],
    fixed_dose: float,
    mask_levels: tuple[float, ...],
) -> list[FixedDoseResult]:
    """Evaluate 1-exp(-B(I)*Dose/I) while holding Dose fixed."""

    results: list[FixedDoseResult] = []
    for mask_level in mask_levels:
        mask = _uniform_mask(model, shape, mask_level)
        prepared = model.prepare_control(mask, shape)
        intensity = _mean(prepared.local_intensity)
        b_value = _mean(prepared.b)
        apparent_time = fixed_dose / intensity
        exponent = b_value * apparent_time
        calculated_doc = 1.0 - math.exp(-max(exponent, 0.0))
        values = (intensity, b_value, apparent_time, exponent, calculated_doc)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"fixed-Dose evaluation is nonfinite at mask level {mask_level}"
            )
        results.append(
            FixedDoseResult(
                mask_level=mask_level,
                current_intensity_mw_cm2=intensity,
                fixed_dose_mj_cm2=fixed_dose,
                apparent_exposure_time_s=apparent_time,
                b_per_s=b_value,
                b_times_exposure=exponent,
                calculated_doc=calculated_doc,
            )
        )
    return results


def _print_transition_table(results: list[TransitionResult]) -> None:
    print("\nState-transition check (one reference physical step per branch)")
    print(
        f"{'case':<14} {'I':>10} {'B(I)':>9} {'added_E':>10} {'Dose_before':>13} "
        f"{'Dose_after':>12} {'delta_Dose':>12} {'R_before':>11} "
        f"{'R_after':>11} {'delta_R':>11}"
    )
    for result in results:
        print(
            f"{result.case:<14} {result.current_intensity_mw_cm2:10.6f} "
            f"{result.b_per_s:9.6f} "
            f"{result.added_energy_mj_cm2:10.6f} "
            f"{result.dose_before_mj_cm2:13.6f} "
            f"{result.dose_after_mj_cm2:12.6f} "
            f"{result.delta_dose_mj_cm2:12.6f} "
            f"{result.reaction_progress_before:11.6f} "
            f"{result.reaction_progress_after:11.6f} "
            f"{result.reaction_progress_increment:11.6f}"
        )

    print("\nLegacy recomputation versus incremental reaction progress")
    print(
        f"{'case':<14} {'DoC_before':>12} {'legacy_Dose/I':>14} "
        f"{'legacy_after':>13} {'legacy_delta':>13} "
        f"{'corrected_after':>16} {'corrected_delta':>16}"
    )
    for result in results:
        print(
            f"{result.case:<14} {result.doc_before:12.6f} "
            f"{result.legacy_apparent_exposure_time_s:14.6f} "
            f"{result.legacy_doc_after:13.6f} "
            f"{result.legacy_delta_doc:13.6f} "
            f"{result.corrected_doc_after:16.6f} "
            f"{result.corrected_delta_doc:16.6f}"
        )


def _print_fixed_dose_table(results: list[FixedDoseResult]) -> None:
    print("\nDirect formula check (accumulated Dose held fixed)")
    print(
        f"{'mask':>7} {'current_I':>12} {'fixed_Dose':>12} {'Dose/I':>12} "
        f"{'B(I)':>10} {'B*Dose/I':>12} {'calculated_DoC':>16}"
    )
    for result in results:
        print(
            f"{result.mask_level:7.3f} "
            f"{result.current_intensity_mw_cm2:12.6f} "
            f"{result.fixed_dose_mj_cm2:12.6f} "
            f"{result.apparent_exposure_time_s:12.6f} "
            f"{result.b_per_s:10.6f} {result.b_times_exposure:12.6f} "
            f"{result.calculated_doc:16.6f}"
        )


def _save_csv(
    path: Path,
    transitions: list[TransitionResult],
    direct_results: list[FixedDoseResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "case",
        "mask_level",
        "current_intensity_mw_cm2",
        "dose_before_mj_cm2",
        "dose_after_mj_cm2",
        "delta_dose_mj_cm2",
        "added_energy_mj_cm2",
        "fixed_dose_mj_cm2",
        "apparent_exposure_time_s",
        "legacy_apparent_exposure_time_s",
        "b_per_s",
        "b_times_exposure",
        "reaction_progress_before",
        "reaction_progress_after",
        "reaction_progress_increment",
        "doc_before",
        "corrected_doc_after",
        "corrected_delta_doc",
        "legacy_doc_after",
        "legacy_delta_doc",
        "calculated_doc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in transitions:
            row = {"section": "state_transition", **asdict(result)}
            writer.writerow(row)
        for result in direct_results:
            row = {"section": "fixed_dose", "case": "", **asdict(result)}
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify the current Dose/current-intensity DoC behavior."
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--medium", type=float, default=0.3)
    parser.add_argument("--low", type=float, default=0.1)
    parser.add_argument("--prehistory-target-doc", type=float, default=0.60)
    parser.add_argument(
        "--max-prehistory-steps",
        type=int,
        default=None,
        help="Default: the number of steps in the reference simulation duration.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "time_varying_doc_diagnostic.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.grid_size < 3:
        raise ValueError("--grid-size must be at least 3")
    if not 0.0 < args.low < args.medium < args.high <= 1.0:
        raise ValueError("require 0 < low < medium < high <= 1")
    if not 0.0 < args.prehistory_target_doc < 0.95:
        raise ValueError("--prehistory-target-doc must lie strictly between 0 and 0.95")

    reference = load_reference_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = AIEModel(device=device)
    shape = (args.grid_size, args.grid_size)
    max_steps = args.max_prehistory_steps
    if max_steps is None:
        max_steps = math.ceil(reference.total_simulation_time_s / reference.dt)
    if max_steps < 1:
        raise ValueError("--max-prehistory-steps must be at least 1")

    print("Time-varying DoC diagnostic (legacy versus incremental MPC history)")
    print(f"Reference model: {reference.reference_model_source}")
    print(f"Reference SHA256: {reference.reference_model_sha256}")
    print(f"device={device} grid={shape} dt={reference.dt:.9g} s")
    print(f"DoC history mode: {DOC_HISTORY_MODE}")
    print(
        f"reference intensity={reference.intensity_mw_cm2:.9g} mW/cm^2; "
        f"B(I)={reference.b_slope:.9g}*I+{reference.b_intercept:.9g}"
    )
    print(
        f"O2 inhibition={reference.o2_inhibition_mj_cm2:.9g} mJ/cm^2; "
        f"TEMPO inhibition={reference.tempo_inhibition_mj_cm2:.9g} mJ/cm^2"
    )

    with torch.no_grad():
        common_state, prehistory_steps = _build_common_prehistory(
            model,
            shape,
            args.high,
            args.prehistory_target_doc,
            max_steps,
        )
        print("\nCommon prehistory state")
        print(
            f"steps={prehistory_steps} elapsed={prehistory_steps * reference.dt:.6f} s "
            f"high_mask={args.high:.3f} O2={_mean(common_state.o2):.6f} "
            f"TEMPO={_mean(common_state.tempo):.6f} "
            f"Dose={_mean(common_state.dose):.6f} "
            f"R={_mean(common_state.reaction_progress):.6f} "
            f"DoC={_mean(common_state.doc):.6f}"
        )

        transitions = [
            _run_transition(model, common_state, "HIGH->HIGH", args.high),
            _run_transition(model, common_state, "HIGH->MEDIUM", args.medium),
            _run_transition(model, common_state, "HIGH->LOW", args.low),
        ]
        direct_results = _run_fixed_dose_check(
            model,
            shape,
            _mean(common_state.dose),
            (args.high, args.medium, args.low),
        )

    _print_transition_table(transitions)
    _print_fixed_dose_table(direct_results)
    _save_csv(args.csv.resolve(), transitions, direct_results)

    high_result, medium_result, low_result = transitions
    legacy_apparent_time_increases = (
        medium_result.legacy_apparent_exposure_time_s
        > high_result.legacy_apparent_exposure_time_s
        and low_result.legacy_apparent_exposure_time_s
        > medium_result.legacy_apparent_exposure_time_s
    )
    legacy_low_jump_ratio = low_result.legacy_delta_doc / max(
        abs(high_result.legacy_delta_doc), 1e-15
    )
    corrected_low_is_larger = (
        low_result.corrected_delta_doc > high_result.corrected_delta_doc
    )
    corrected_medium_is_larger = (
        medium_result.corrected_delta_doc > high_result.corrected_delta_doc
    )
    common_history_is_identical = all(
        math.isclose(
            result.reaction_progress_before,
            high_result.reaction_progress_before,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for result in transitions
    )
    if not common_history_is_identical:
        raise AssertionError("branch reset changed historical reaction progress")
    if corrected_medium_is_larger or corrected_low_is_larger:
        raise AssertionError(
            "lower-current branch still produced a larger corrected DoC increment"
        )

    print("\nInterpretation")
    print(
        "1. Legacy behavior still shows Dose/current_intensity increasing as "
        f"current intensity falls: {legacy_apparent_time_increases}."
    )
    print(
        "2. Legacy LOW/HIGH delta-DoC ratio from the same prehistory is "
        f"{legacy_low_jump_ratio:.3f}; this reproduces the diagnosed artifact."
    )
    print(
        "3. Historical reaction progress is identical before every branch: "
        f"{common_history_is_identical}. Only each new Dose increment changes R."
    )
    print(
        "4. Under incremental history, LOW produces a larger DoC jump than HIGH: "
        f"{corrected_low_is_larger}."
    )
    print(
        "5. The corrected transition does not divide historical Dose by the new "
        "current intensity, so lowering intensity no longer reinterprets history."
    )
    print(
        "6. With a constant mask, summing B(I)*delta_Dose/I telescopes to "
        "B(I)*Dose/I and therefore retains the static reference law."
    )
    print(f"\nsaved diagnostic CSV: {args.csv.resolve()}")


if __name__ == "__main__":
    main()

