import numpy as np

from dlp_plateau import trim_terminal_decline


def test_keep_after_retains_points_from_the_detected_decline_start():
    """A confirmed three-point decline starts at index 5 and keeps two points."""
    signal = np.array([0.0, 0.5, 1.0, 0.99, 0.8, 0.7, 0.6, 0.5])

    trimmed = trim_terminal_decline(
        signal, tolerance=0.02, consecutive=3, keep_after=2
    )

    np.testing.assert_array_equal(trimmed, signal[:7])
