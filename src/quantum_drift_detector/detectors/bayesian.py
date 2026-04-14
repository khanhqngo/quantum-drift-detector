"""
Bayesian Online Change-Point Detection (BOCD).

Plain English: At every timestep, the algorithm maintains a probability
distribution over the "run length" — how many steps the current noise regime
has been running.  Think of it as a table with a column for each possible age
of the current regime (0, 1, 2, ... steps since the last change).

During a quiet period, almost all probability sits in one column
(the long-running stable regime).  When a shift happens:
1. The data no longer fits the established regime's prediction.
2. Probability mass flows into the "recently started" columns (short run lengths).
3. We detect this as: P(run age is very small) suddenly jumps.

Mathematical model:
    - Observations: x_t (bit-flip rate) ~ Normal(mu, obs_var) within each regime.
      obs_var represents measurement-shot noise; for N shots at rate p,
      obs_var ≈ p(1-p)/N.
    - Prior on regime mean:  mu ~ Normal(mu_0, prior_var).
      prior_var encodes how large a jump between regimes you expect.
    - Hazard rate H: prior probability that a changepoint occurs at any given step.
    - Detection signal: P(current run length ≤ K), denoted the "short-run
      probability."  In a stable regime this is ≈ (K+1)·H.  Right after a real
      changepoint it spikes toward 1.0.

Reference: Adams, R. P. & MacKay, D. J. C. (2007).
           Bayesian Online Changepoint Detection. arXiv:0710.3742.
"""

import numpy as np
from scipy import stats
from typing import Optional


def run_bocd(
    bit_flip_rates: np.ndarray,
    hazard_rate: float = 0.01,
    mu_0: float = 0.02,
    obs_var: float = 2.5e-5,
    prior_var: float = 0.01,
    short_run_window: int = 5,
    threshold: float = 0.3,
) -> dict:
    """
    Run Bayesian Online Change-Point Detection on a bit-flip rate time series.

    Maintains the full run-length posterior at every timestep and returns the
    "short-run probability" P(run_length ≤ short_run_window) as the detection
    signal.  This quantity is low (≈ hazard_rate * short_run_window) in a
    stable regime and spikes toward 1.0 for several timesteps after a genuine
    changepoint.

    Setting the parameters (plain-English guide):
        mu_0             : Prior guess for the baseline bit-flip rate (e.g. 0.02
                           for 2%).
        obs_var          : Within-regime observation variance (shot noise, σ²).
                           For N shots at rate p: obs_var ≈ p(1-p)/N.
                           With N=1000, p=0.02: obs_var ≈ 2e-5.
        prior_var        : Variance of possible jump sizes (τ²).  If rates can
                           shift by up to ±0.1 (10 pp), set prior_var = 0.01.
                           Larger → more sensitive to small shifts.
        hazard_rate      : P(changepoint at any given step).
                           Set to 1 / (expected average segment length).
        short_run_window : K in P(r_t ≤ K).  Controls the width of the spike
                           after a changepoint.  5–10 is a good range.
        threshold        : Alarm when P(r_t ≤ K) exceeds this.
                           Should be above the stable-regime baseline of
                           approximately (K+1) * hazard_rate.

    Args:
        bit_flip_rates   : 1D array of bit-flip rates, one value per timestep.
        hazard_rate      : Prior changepoint probability per step (H).
        mu_0             : Prior mean for each regime's bit-flip rate.
        obs_var          : Observation variance within a regime (σ²).
        prior_var        : Prior variance on the regime mean (τ²).
        short_run_window : K — accumulate probability over run lengths 0..K.
        threshold        : Alarm when short-run probability exceeds this.

    Returns:
        dict with keys:
            'changepoint_probs' : 1D array — P(r_t ≤ K) at each timestep.
            'detected_at'       : int timestep of first alarm, or None.
            'threshold'         : the threshold used (for plotting convenience).

    Example:
        >>> rates = np.array([0.01] * 50 + [0.05] * 50, dtype=float)
        >>> result = run_bocd(rates, mu_0=0.01, obs_var=1e-5, prior_var=0.01)
        >>> print(result['detected_at'])  # Should be close to 50
    """
    n = len(bit_flip_rates)
    changepoint_probs = np.zeros(n)
    detected_at: Optional[int] = None

    # log_weights[r] = log P(R_{t-1} = r | x_{1:t-1}).
    # Grows by one element each timestep.  Normalised after each update.
    log_weights = np.array([0.0])   # At t=0: run length 0 with probability 1.

    # Sufficient statistics for the Normal-Normal conjugate update.
    # For run of length r: ns[r] observations summing to sums[r].
    #   Posterior precision:  prec_r   = 1/prior_var + ns[r]/obs_var
    #   Posterior mean:       mu_r     = (mu_0/prior_var + sums[r]/obs_var) / prec_r
    #   Predictive variance:  pred_var = obs_var + 1/prec_r
    ns   = np.array([0.0])
    sums = np.array([0.0])

    for t in range(n):
        x = bit_flip_rates[t]

        # ── Step 1: predictive log-probabilities ─────────────────────────
        prec_r   = 1.0 / prior_var + ns / obs_var
        mu_r     = (mu_0 / prior_var + sums / obs_var) / prec_r
        pred_var = obs_var + 1.0 / prec_r

        log_preds = stats.norm.logpdf(x, loc=mu_r, scale=np.sqrt(pred_var))

        # ── Step 2: growth and changepoint messages ──────────────────────
        # log P(R_{t-1}=r | past) + log P(x_t | R_{t-1}=r)
        log_joint = log_weights + log_preds

        # Growth:       P(R_t = r+1) ∝ P(R_{t-1}=r) * P(x_t|r) * (1-H)
        log_growth = log_joint + np.log(1.0 - hazard_rate)

        # Changepoint:  P(R_t = 0)   ∝ [sum_r P(R_{t-1}=r)*P(x_t|r)] * H
        log_cp = np.logaddexp.reduce(log_joint) + np.log(hazard_rate)

        # ── Step 3: assemble and normalise ───────────────────────────────
        new_log_weights = np.empty(len(log_weights) + 1)
        new_log_weights[0]  = log_cp
        new_log_weights[1:] = log_growth

        log_norm = np.logaddexp.reduce(new_log_weights)
        new_log_weights -= log_norm

        # ── Step 4: detection signal = P(run length ≤ K) ────────────────
        # This is low (~(K+1)*H) in a stable regime and spikes toward 1.0
        # for K timesteps immediately following a genuine changepoint.
        # We only report once there are strictly more states than K+1, so that
        # the window does not trivially cover all possible run lengths.
        if len(new_log_weights) > short_run_window + 1:
            short_run_prob = float(np.exp(
                np.logaddexp.reduce(new_log_weights[:short_run_window + 1])
            ))
        else:
            short_run_prob = 0.0
        changepoint_probs[t] = short_run_prob

        if detected_at is None and short_run_prob > threshold:
            detected_at = t

        # ── Step 5: update sufficient statistics ─────────────────────────
        new_ns   = np.empty(len(ns) + 1)
        new_sums = np.empty(len(sums) + 1)

        new_ns[0]   = 0.0       # fresh start: reset to prior
        new_sums[0] = 0.0

        new_ns[1:]   = ns   + 1.0   # existing runs gain one observation
        new_sums[1:] = sums + x

        log_weights = new_log_weights
        ns          = new_ns
        sums        = new_sums

    return {
        "changepoint_probs": changepoint_probs,
        "detected_at": detected_at,
        "threshold": threshold,
    }
