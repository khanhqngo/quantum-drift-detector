"""
Evaluation utilities for benchmarking change-point detectors.

This module provides:
  - evaluate_single_trial : score one detector run against a known changepoint.
  - run_benchmark         : batch evaluation across many scenarios and trials.

Plain English: A "scenario" is a recipe for generating data (which simulator,
which parameters, how long, where the true changepoint is).  For each scenario
we run the experiment many times (trials) with different random seeds, apply
all four detectors, and record whether each detector found the changepoint and
how quickly.

The resulting DataFrame can then be sliced, grouped, and plotted to answer
questions like:
  - Which detector works best when noise is highly correlated?
  - How does detection delay change as the shift size shrinks?
  - Does the ensemble outperform any single method?
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Single-trial metrics
# ---------------------------------------------------------------------------

def evaluate_single_trial(
    detected_at: Optional[int],
    true_cp: int,
    tolerance: int = 15,
) -> dict:
    """
    Compute detection metrics for one detector run.

    We say a detection is a TRUE POSITIVE if the alarm fires within
    `tolerance` timesteps AFTER the true changepoint.  Anything before the
    changepoint counts as a FALSE ALARM.  Silence after the changepoint
    counts as a MISSED detection.

    Args:
        detected_at : The timestep at which the detector raised its alarm,
                      or None if no alarm was raised.
        true_cp     : The true changepoint index (0-based).
        tolerance   : Maximum allowed delay (timesteps after true_cp) for a
                      detection to count as a true positive.

    Returns:
        dict with keys:
            'detected'    : bool — True if alarm within [true_cp, true_cp+tolerance].
            'false_alarm' : bool — True if alarm fired before true_cp.
            'missed'      : bool — True if no alarm after true_cp.
            'delay'       : int or None — timesteps after true_cp until alarm
                            (None if not a true positive).

    Example:
        >>> evaluate_single_trial(detected_at=55, true_cp=50, tolerance=15)
        {'detected': True, 'false_alarm': False, 'missed': False, 'delay': 5}

        >>> evaluate_single_trial(detected_at=None, true_cp=50, tolerance=15)
        {'detected': False, 'false_alarm': False, 'missed': True, 'delay': None}
    """
    if detected_at is None:
        return {"detected": False, "false_alarm": False, "missed": True, "delay": None}

    if detected_at < true_cp:
        return {"detected": False, "false_alarm": True, "missed": False, "delay": None}

    delay = detected_at - true_cp
    if delay <= tolerance:
        return {"detected": True, "false_alarm": False, "missed": False, "delay": delay}
    else:
        # Alarm fired, but too late to be called a true positive.
        return {"detected": False, "false_alarm": False, "missed": True, "delay": None}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    scenarios: list,
    detector_runners: dict,
    n_trials: int = 30,
    tolerance: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run all detectors across many scenarios and collect per-trial metrics.

    Each scenario defines a simulation setup.  For each trial we generate
    fresh data, extract the bit-flip rate, run every detector, and score the
    result using evaluate_single_trial.

    Args:
        scenarios : List of dicts.  Each dict must have:
                        'name'       : str — human-readable label.
                        'noise_type' : str — one of 'depolarizing', 'dephasing',
                                       'correlated', etc.
                        'sim_class'  : class — simulator class (uninstantiated).
                        'sim_kwargs' : dict — kwargs passed to the constructor.
                        'n_timesteps': int — total length of the time series.
                        'n_shots'    : int — measurement shots per timestep.
                        'true_cp'    : int — the true changepoint index.
                    Optional key:
                        'meta'       : dict of extra columns to add to output
                                       (e.g. {'shift_size': 0.04, 'phi': 0.5}).

        detector_runners : dict mapping detector name (str) to a callable that
                           accepts a 1D bit_flip_rates array and returns a dict
                           containing a 'detected_at' key (int or None).
                           Example:
                               {
                                 'CUSUM':    lambda r: run_cusum(r, ...),
                                 'BOCD':     lambda r: run_bocd(r, ...),
                                 'Window':   lambda r: run_window_detector(r, ...),
                                 'Ensemble': lambda r: run_ensemble(r),
                               }

        n_trials    : Number of independent trials per scenario.
        tolerance   : Maximum delay (timesteps) for a TP detection.
        random_seed : Base random seed; each trial uses seed + trial_index.

    Returns:
        pandas DataFrame with one row per (scenario × detector × trial):
            scenario, noise_type, detector, trial,
            detected, false_alarm, missed, delay,
            + any extra columns from scenario['meta'].

    Example:
        >>> df = run_benchmark(scenarios, runners, n_trials=10)
        >>> df.groupby(['noise_type', 'detector'])['detected'].mean()
    """
    from .features.extractors import extract_bit_flip_rate

    rows = []

    for sc in scenarios:
        name       = sc["name"]
        noise_type = sc["noise_type"]
        sim_class  = sc["sim_class"]
        sim_kwargs = sc["sim_kwargs"]
        n_ts       = sc["n_timesteps"]
        n_sh       = sc["n_shots"]
        true_cp    = sc["true_cp"]
        meta       = sc.get("meta", {})

        for trial in range(n_trials):
            rng = np.random.default_rng(random_seed + trial)
            # Inject the rng seed into the simulator via numpy's global seed.
            np.random.seed(random_seed + trial)

            sim  = sim_class(**sim_kwargs)
            data = sim.generate_data(n_timesteps=n_ts, n_shots=n_sh)
            rates = extract_bit_flip_rate(data)

            for det_name, runner in detector_runners.items():
                result   = runner(rates)
                det_at   = result.get("detected_at", None)
                metrics  = evaluate_single_trial(det_at, true_cp, tolerance)

                row = {
                    "scenario":    name,
                    "noise_type":  noise_type,
                    "detector":    det_name,
                    "trial":       trial,
                    "detected_at": det_at,
                    **metrics,
                    **meta,
                }
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-trial results into recall, false-alarm rate, and mean delay.

    Args:
        df : DataFrame returned by run_benchmark.

    Returns:
        Summary DataFrame grouped by (scenario, noise_type, detector) with:
            'recall'      : fraction of trials where the change was detected.
            'far'         : false alarm rate (fraction of trials with alarm
                            before changepoint).
            'mean_delay'  : mean delay (in timesteps) over true-positive trials.
                            NaN if no true positives.
            'n_trials'    : total number of trials.

    Example:
        >>> summary = summarise_benchmark(df)
        >>> print(summary.sort_values('recall', ascending=False).head(10))
    """
    def _agg(g):
        recall     = g["detected"].mean()
        far        = g["false_alarm"].mean()
        delays     = g.loc[g["detected"], "delay"].dropna()
        mean_delay = delays.mean() if len(delays) > 0 else float("nan")
        return pd.Series({
            "recall":     recall,
            "far":        far,
            "mean_delay": mean_delay,
            "n_trials":   len(g),
        })

    group_cols = [c for c in ["scenario", "noise_type", "detector"]
                  if c in df.columns]
    return df.groupby(group_cols).apply(_agg).reset_index()
