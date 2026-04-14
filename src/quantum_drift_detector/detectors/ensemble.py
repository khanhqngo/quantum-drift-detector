"""
Ensemble change-point detector.

Plain English: Instead of trusting a single detective to find the suspect,
we hire three and let them vote.  Each individual detector has a different
personality:

  - CUSUM is a fast accumulator — it notices sustained upward trends early.
  - BOCD is a Bayesian reasoner — it updates a full probability model of where
    the changepoint might be.
  - The Window detector is a non-parametric comparator — it needs no assumptions
    about the noise model, just compares two windows of data directly.

By combining all three, the ensemble inherits the speed of CUSUM, the
principled uncertainty of BOCD, and the model-free flexibility of the Window
detector.  Their individual weaknesses tend to cancel out.

Two combination strategies are supported:

  soft  (default): Normalize each detector's score to [0, 1] and take a
        weighted average.  Raise an alarm when the combined score exceeds a
        threshold.  This is a graded signal — the higher the score, the more
        evidence of a changepoint.

  hard  (majority vote): Each detector independently raises a binary alarm at
        each timestep.  Raise a combined alarm when at least `min_votes` out of
        3 detectors agree.  This suppresses individual false alarms — a fluke
        spike in one detector won't trigger the ensemble.
"""

import numpy as np
from typing import Optional

from .cusum import run_cusum
from .bayesian import run_bocd
from .window import run_window_detector


# ---------------------------------------------------------------------------
# Default parameter sets for each member detector.
# These are conservative "good defaults" that work across a wide range of
# scenarios without knowing the true shift size in advance.
# target_mean / mu_0 = None means "auto-estimate from the first 20% of data."
# ---------------------------------------------------------------------------

_DEFAULT_CUSUM: dict = {
    "target_mean": None,   # auto-estimated
    "allowance": 0.01,
    "threshold": 0.10,
}

_DEFAULT_BOCD: dict = {
    "hazard_rate": 0.01,
    "mu_0": None,          # auto-estimated
    "obs_var": 2.5e-5,
    "prior_var": 0.01,
    "short_run_window": 5,
    "threshold": 0.30,
}

_DEFAULT_WINDOW: dict = {
    "window_size": 10,
    "threshold": 0.05,
}


def _merge(defaults: dict, overrides: Optional[dict]) -> dict:
    """Merge user-supplied overrides into a copy of the defaults dict."""
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return params


def run_ensemble(
    bit_flip_rates: np.ndarray,
    cusum_params: Optional[dict] = None,
    bocd_params: Optional[dict] = None,
    window_params: Optional[dict] = None,
    threshold: float = 0.5,
    method: str = "soft",
    min_votes: int = 2,
    weights: Optional[list] = None,
) -> dict:
    """
    Combine CUSUM, BOCD, and the sliding-window detector into an ensemble.

    The ensemble auto-estimates the baseline bit-flip rate from the first 20%
    of observations, so you don't need to pass the true pre-change rate.

    Soft voting (method='soft'):
        Each detector's raw score is normalised to [0, 1] by dividing by its
        own threshold (capped at 1.0).  The three normalised scores are then
        averaged (optionally with weights).  An alarm is raised when this
        combined score exceeds `threshold`.

    Hard voting (method='hard'):
        Each detector's binary alarm signal at each timestep is extracted.  An
        ensemble alarm fires when at least `min_votes` detectors agree at the
        same timestep.

    Args:
        bit_flip_rates : 1D array of bit-flip rates, shape (n_timesteps,).
        cusum_params   : Override any key in the CUSUM default params dict.
                         Recognised keys: target_mean, allowance, threshold.
        bocd_params    : Override any key in the BOCD default params dict.
                         Recognised keys: hazard_rate, mu_0, obs_var, prior_var,
                         short_run_window, threshold.
        window_params  : Override any key in the Window default params dict.
                         Recognised keys: window_size, threshold.
        threshold      : Ensemble-level detection threshold.
                         For soft voting: combined score must exceed this.
                         Not used for hard voting (use min_votes instead).
        method         : 'soft' (weighted average) or 'hard' (majority vote).
        min_votes      : Minimum number of member detectors that must agree to
                         trigger the ensemble alarm (only used when method='hard').
        weights        : List of three floats [w_cusum, w_bocd, w_window] for
                         soft voting.  Defaults to equal weights [1, 1, 1].

    Returns:
        dict with keys:
            'combined_scores'    : 1D float array in [0, 1], shape (n_timesteps,).
            'cusum_scores_norm'  : CUSUM scores normalised to [0, 1].
            'bocd_scores'        : BOCD short-run probabilities (already [0, 1]).
            'window_scores_norm' : Window KL scores normalised to [0, 1].
            'individual_alarms'  : dict {'cusum', 'bocd', 'window'} → detected_at
                                   (int or None) for each member detector.
            'detected_at'        : int timestep of first ensemble alarm, or None.
            'threshold'          : the threshold used (for plotting convenience).
            'method'             : the combination method used.

    Example:
        >>> rates = np.array([0.01] * 50 + [0.06] * 50, dtype=float)
        >>> result = run_ensemble(rates)
        >>> print(result['detected_at'])   # Should be close to 50
    """
    n = len(bit_flip_rates)

    if method not in ("soft", "hard"):
        raise ValueError(f"method must be 'soft' or 'hard', got '{method}'")

    if weights is None:
        weights = [1.0, 1.0, 1.0]
    if len(weights) != 3:
        raise ValueError("weights must have exactly 3 elements [w_cusum, w_bocd, w_window]")
    w = np.array(weights, dtype=float)
    w = w / w.sum()   # normalise to sum-to-one

    # ── Auto-estimate baseline from first 20 % of data ───────────────────────
    n_baseline = max(10, n // 5)
    baseline = float(np.mean(bit_flip_rates[:n_baseline]))

    # ── Build per-detector parameter dicts ───────────────────────────────────
    cp = _merge(_DEFAULT_CUSUM, cusum_params)
    if cp["target_mean"] is None:
        cp["target_mean"] = baseline

    bp = _merge(_DEFAULT_BOCD, bocd_params)
    if bp["mu_0"] is None:
        bp["mu_0"] = baseline

    wp = _merge(_DEFAULT_WINDOW, window_params)

    # ── Run member detectors ─────────────────────────────────────────────────
    cusum_result  = run_cusum(
        bit_flip_rates,
        target_mean=cp["target_mean"],
        allowance=cp["allowance"],
        threshold=cp["threshold"],
    )
    bocd_result   = run_bocd(
        bit_flip_rates,
        hazard_rate=bp["hazard_rate"],
        mu_0=bp["mu_0"],
        obs_var=bp["obs_var"],
        prior_var=bp["prior_var"],
        short_run_window=bp["short_run_window"],
        threshold=bp["threshold"],
    )
    window_result = run_window_detector(
        bit_flip_rates,
        window_size=wp["window_size"],
        threshold=wp["threshold"],
    )

    # ── Normalise scores to [0, 1] ───────────────────────────────────────────
    # CUSUM: divide by its threshold and cap at 1.
    cusum_norm  = np.clip(cusum_result["scores"]  / cp["threshold"], 0.0, 1.0)
    # BOCD: already probabilities in [0, 1].
    bocd_norm   = bocd_result["changepoint_probs"]
    # Window: divide by its threshold and cap at 1.
    window_norm = np.clip(window_result["scores"] / wp["threshold"], 0.0, 1.0)

    # ── Combine ───────────────────────────────────────────────────────────────
    detected_at: Optional[int] = None

    if method == "soft":
        combined = w[0] * cusum_norm + w[1] * bocd_norm + w[2] * window_norm
        # Find the first timestep where combined score exceeds threshold.
        alarm_indices = np.where(combined > threshold)[0]
        detected_at = int(alarm_indices[0]) if len(alarm_indices) > 0 else None

    else:  # hard voting
        # Binary alarm at each timestep for each member.
        cusum_alarm  = (cusum_result["scores"]           > cp["threshold"]).astype(float)
        bocd_alarm   = (bocd_result["changepoint_probs"] > bp["threshold"]).astype(float)
        window_alarm = (window_result["scores"]          > wp["threshold"]).astype(float)

        vote_count = cusum_alarm + bocd_alarm + window_alarm  # 0, 1, 2, or 3
        combined   = vote_count / 3.0                         # normalise to [0, 1]

        alarm_indices = np.where(vote_count >= min_votes)[0]
        detected_at = int(alarm_indices[0]) if len(alarm_indices) > 0 else None

    return {
        "combined_scores":    combined,
        "cusum_scores_norm":  cusum_norm,
        "bocd_scores":        bocd_norm,
        "window_scores_norm": window_norm,
        "individual_alarms": {
            "cusum":  cusum_result["detected_at"],
            "bocd":   bocd_result["detected_at"],
            "window": window_result["detected_at"],
        },
        "detected_at": detected_at,
        "threshold":   threshold,
        "method":      method,
    }
