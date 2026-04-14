"""
CUSUM (Cumulative Sum) change-point detector.

Plain English: Imagine you're watching a machine that usually produces 1% defects.
CUSUM keeps a running score: every time the defect rate is higher than expected,
the score goes up; when it drops back down, the score resets to zero. When the
score gets big enough (crosses a threshold), you declare the machine has degraded.

For quantum circuits: the bit-flip rate plays the role of "defect rate." When
the device noise increases, the bit-flip rate rises, and CUSUM accumulates
evidence until it is confident enough to raise an alarm.

Reference: Page, E. S. (1954). Continuous inspection schemes.
           Biometrika, 41(1/2), 100-115.
"""

import numpy as np
from typing import Optional


def run_cusum(
    bit_flip_rates: np.ndarray,
    target_mean: float,
    allowance: float,
    threshold: float,
) -> dict:
    """
    Run the one-sided CUSUM algorithm to detect an upward shift in bit-flip rate.

    CUSUM computes a running score at each timestep t:

        S_t = max(0, S_{t-1} + (x_t - target_mean - allowance))

    The max(..., 0) resets the score when observations drop back to normal —
    we only care about sustained upward shifts. A changepoint is declared the
    first time S_t exceeds the threshold.

    The 'allowance' (also called k in the literature) controls sensitivity:
    observations that deviate by less than allowance are essentially ignored.
    A good starting value is half the expected shift size.

    Args:
        bit_flip_rates: 1D array of bit-flip rates (floats in [0, 1]),
            one value per timestep. Produced by extract_bit_flip_rate().
        target_mean: Expected bit-flip rate *before* the change. Set to the
            known (or estimated) baseline error rate, e.g. 0.01 for 1%.
        allowance: Slack parameter (k). Shifts smaller than this are absorbed
            without growing the score. If you expect the rate to jump from
            1% to 5%, use allowance = 0.02 (half the shift of 0.04).
        threshold: Detection threshold (h). When the score exceeds this, an
            alarm is raised. Larger values mean fewer false alarms but slower
            detection. Typical useful range: 0.05 to 0.5.

    Returns:
        dict with keys:
            'scores'      : 1D float array of CUSUM scores, shape (n_timesteps,).
            'detected_at' : int timestep of first detection, or None if no alarm.
            'threshold'   : the threshold used (echoed back for plotting convenience).

    Example:
        >>> rates = np.array([0.01] * 50 + [0.05] * 50)
        >>> result = run_cusum(rates, target_mean=0.01, allowance=0.02, threshold=0.1)
        >>> print(result['detected_at'])  # Should be close to 50
    """
    n = len(bit_flip_rates)
    scores = np.zeros(n)
    detected_at: Optional[int] = None

    for t in range(1, n):
        # Accumulate how far above (target_mean + allowance) the current rate is.
        # If the rate is normal or below, the max resets the score to zero —
        # CUSUM "forgets" past low readings and focuses on sustained high ones.
        scores[t] = max(0.0, scores[t - 1] + (bit_flip_rates[t] - target_mean - allowance))

        if detected_at is None and scores[t] > threshold:
            detected_at = t

    return {
        "scores": scores,
        "detected_at": detected_at,
        "threshold": threshold,
    }
