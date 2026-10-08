"""Plateau correction for DLP fluorescence time series."""

import numpy as np


def trim_terminal_decline(
    signal, tolerance=0.02, consecutive=3, candidate_fraction=0.05,
    minimum_candidates=5, keep_after=0,
):
    """Trim a sustained decline after the measured plateau.

    The plateau is the median of the highest raw-signal candidates. Values
    within ``tolerance`` of it remain unchanged; a run of ``consecutive``
    lower values marks the terminal decline. ``keep_after`` retains that many
    raw points beginning at the first point of the confirmed decline.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    if isinstance(keep_after, bool) or not isinstance(keep_after, (int, np.integer)):
        raise ValueError("keep_after must be a non-negative integer")
    if keep_after < 0:
        raise ValueError("keep_after must be a non-negative integer")
    if not 0 < candidate_fraction <= 1:
        raise ValueError("candidate_fraction must be in (0, 1]")
    if minimum_candidates < 1:
        raise ValueError("minimum_candidates must be positive")
    if values.size == 0:
        return values.copy()

    corrected = values.copy()
    candidate_count = min(
        values.size, max(minimum_candidates, int(np.ceil(values.size * candidate_fraction)))
    )
    plateau_candidates = np.partition(values, values.size - candidate_count)[-candidate_count:]
    plateau = float(np.median(plateau_candidates))
    lower_bound = plateau * (1 - tolerance)
    plateau_start = int(np.flatnonzero(values >= lower_bound)[0])
    run_length = 0

    for index in range(plateau_start + 1, values.size):
        run_length = run_length + 1 if values[index] < lower_bound else 0
        if run_length >= consecutive:
            decline_start = index - consecutive + 1
            return corrected[: decline_start + keep_after]
    return corrected
