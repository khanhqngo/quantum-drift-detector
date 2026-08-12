"""
verify_conclusion_numbers.py
============================
Run this script to get the exact numbers that should appear in the conclusion:

    "Showed standard detectors miss X–Y% of drift events under non-Markovian
     noise vs. <Z% miss rate under independent noise, with
     autocorrelation-aware features recovering W% of lost performance
     across 100+ evaluation runs."

Usage (from the project root with the venv active):

    python verify_conclusion_numbers.py

Output: a printed table plus a ready-to-paste conclusion sentence.
"""

import sys
import io
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output so box-drawing characters work on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from quantum_drift_detector.simulators import (
    DepolarizingSimulator, CorrelatedNoiseSimulator
)
from quantum_drift_detector.detectors import (
    run_cusum, run_bocd, run_window_detector, run_ensemble
)
from quantum_drift_detector.features import (
    extract_bit_flip_rate, extract_rolling_autocorr
)
from quantum_drift_detector.utils import (
    run_benchmark, summarise_benchmark, evaluate_single_trial
)

# ─── Shared constants ─────────────────────────────────────────────────────────
N_TRIALS  = 50     # more trials → tighter estimates (takes ~2 min)
N_TS      = 200    # timesteps per trial
N_SH      = 500    # shots per timestep
TRUE_CP   = 100    # changepoint at midpoint
TOLERANCE = 15     # timesteps after true_cp that still count as TP
SEED      = 42

# ─── Standard detector runners ───────────────────────────────────────────────
STANDARD_RUNNERS = {
    "CUSUM":    lambda r: run_cusum(r,
                    target_mean=float(np.mean(r[:20])),
                    allowance=0.01, threshold=0.10),
    "BOCD":     lambda r: run_bocd(r,
                    mu_0=float(np.mean(r[:20])),
                    obs_var=2.5e-5, prior_var=0.01, threshold=0.30),
    "Window":   lambda r: run_window_detector(r,
                    window_size=10, threshold=0.05),
    "Ensemble": lambda r: run_ensemble(r, method="soft", threshold=0.50),
}
INDIVIDUAL_DETECTORS = ["CUSUM", "BOCD", "Window"]

print("=" * 65)
print("CLAIM 1 & 2 — Miss rates: independent vs non-Markovian noise")
print("=" * 65)

# ─── Experiment A: phi sweep, fixed shift Δμ = 0.04 ─────────────────────────
# We test phi = 0.0 (independent) and phi ∈ {0.7, 0.8, 0.9} (non-Markovian)

PHI_INDEPENDENT    = [0.0]
PHI_NON_MARKOVIAN  = [0.7, 0.8, 0.9]
DELTA_MU           = 0.04

def make_corr_scenario(phi, name=None):
    return dict(
        name=name or f"phi={phi:.1f}",
        noise_type="correlated",
        sim_class=CorrelatedNoiseSimulator,
        sim_kwargs=dict(mu_pre=0.01, mu_post=0.01 + DELTA_MU,
                        phi=phi, sigma=0.003, changepoint=TRUE_CP),
        n_timesteps=N_TS, n_shots=N_SH, true_cp=TRUE_CP,
        meta=dict(phi=phi, shift_size=DELTA_MU),
    )

scenarios_A = (
    [make_corr_scenario(p) for p in PHI_INDEPENDENT] +
    [make_corr_scenario(p) for p in PHI_NON_MARKOVIAN]
)

print(f"\nRunning {len(scenarios_A)} phi values × {len(STANDARD_RUNNERS)} "
      f"detectors × {N_TRIALS} trials …", flush=True)

df_A = run_benchmark(scenarios_A, STANDARD_RUNNERS,
                     n_trials=N_TRIALS, tolerance=TOLERANCE, random_seed=SEED)

# ── Compute miss rates ───────────────────────────────────────────────────────
summary_A = summarise_benchmark(df_A)
summary_A["miss_rate"] = 1.0 - summary_A["recall"]

# Attach phi for grouping
phi_map = {s["name"]: s["meta"]["phi"] for s in scenarios_A}
summary_A["phi"] = summary_A["scenario"].map(phi_map)

# Individual detectors only (ensemble is the "fix", not the baseline)
ind_A = summary_A[summary_A["detector"].isin(INDIVIDUAL_DETECTORS)]

# Claim 2: miss rate under INDEPENDENT noise (phi=0)
ind_miss = (
    ind_A[ind_A["phi"] == 0.0]
    .groupby("detector")["miss_rate"].mean()
)
independent_miss_mean  = float(ind_miss.mean())
independent_miss_range = (float(ind_miss.min()), float(ind_miss.max()))

# Claim 1: miss rate under NON-MARKOVIAN noise (phi >= 0.7)
nm_miss = (
    ind_A[ind_A["phi"] >= 0.7]
    .groupby("detector")["miss_rate"].mean()
)
nonmarkov_miss_mean  = float(nm_miss.mean())
nonmarkov_miss_range = (float(nm_miss.min()), float(nm_miss.max()))

print("\n── Miss rates by detector and phi ──")
print(ind_A[["scenario", "detector", "recall", "miss_rate"]]
      .sort_values(["scenario", "detector"])
      .to_string(index=False))

print(f"\n★ CLAIM 2 — Independent noise (phi=0):")
print(f"   Individual detector miss rates: {independent_miss_range[0]:.1%} – "
      f"{independent_miss_range[1]:.1%}  (mean {independent_miss_mean:.1%})")

print(f"\n★ CLAIM 1 — Non-Markovian noise (phi=0.7–0.9):")
print(f"   Individual detector miss rates: {nonmarkov_miss_range[0]:.1%} – "
      f"{nonmarkov_miss_range[1]:.1%}  (mean {nonmarkov_miss_mean:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CLAIM 3 — ACF-aware recovery under non-Markovian noise")
print("=" * 65)

# The ACF-aware experiment uses a DIFFERENT scenario:
# the CORRELATION STRUCTURE itself changes at the changepoint
# (phi: 0.0 → 0.8) while the mean stays fixed.
#
# In this scenario, the standard detectors (which look at the mean rate)
# struggle — but a CUSUM run on the rolling ACF signal detects the change,
# because ACF rises from 0 to ~0.8 after the changepoint.
#
# We compare:
#   - Standard ensemble (bit_flip_rate input)
#   - ACF-aware detector  (rolling_autocorr input → CUSUM)
#   - ACF-enhanced ensemble (combine both)

class CorrelationShiftSimulator:
    """
    Generates data where the CORRELATION STRUCTURE changes at the changepoint
    while the mean bit-flip rate stays approximately constant.

    Pre-change:  phi=0    (independent, Markovian noise)
    Post-change: phi=0.8  (strongly correlated, non-Markovian)
    Mean mu stays at 0.03 throughout.

    This is the scenario where standard detectors are blind (no mean shift)
    but ACF-based detection can see the change.
    """
    def __init__(self, mu=0.03, phi_pre=0.0, phi_post=0.8,
                 sigma=0.005, changepoint=100):
        self.mu         = mu
        self.phi_pre    = phi_pre
        self.phi_post   = phi_post
        self.sigma      = sigma
        self.changepoint = changepoint

    def generate_data(self, n_timesteps, n_shots):
        eps = np.zeros(n_timesteps)
        eps[0] = self.mu
        for t in range(1, n_timesteps):
            phi = self.phi_pre if t < self.changepoint else self.phi_post
            eps[t] = np.clip(
                self.mu + phi * (eps[t - 1] - self.mu) + self.sigma * np.random.randn(),
                0.0, 1.0,
            )
        return np.random.binomial(1, eps[:, None].repeat(n_shots, axis=1)).astype(int)


# ── Detector runners for correlation-shift experiment ────────────────────────
ACF_WINDOW = 20   # rolling ACF window size (same as used in Notebook 04)

def run_acf_cusum(bit_flip_rates):
    """Run CUSUM on the rolling lag-1 autocorrelation signal."""
    acf = extract_rolling_autocorr(bit_flip_rates, window=ACF_WINDOW, lag=1)
    # Baseline ACF ≈ 0 for Markovian noise; rises after shift to correlated.
    # CUSUM on acf with target=0 and small allowance detects the rise.
    return run_cusum(acf, target_mean=0.0, allowance=0.05, threshold=0.3)


def run_acf_ensemble(bit_flip_rates):
    """Soft-vote ensemble that combines the standard ensemble with ACF-CUSUM."""
    std = run_ensemble(bit_flip_rates, method="soft", threshold=0.50)
    acf_res = run_acf_cusum(bit_flip_rates)

    # Normalise ACF-CUSUM score to [0,1]
    acf_norm = np.clip(acf_res["scores"] / 0.3, 0.0, 1.0)
    std_score = std["combined_scores"]

    # Combined: equal weight between standard ensemble and ACF signal
    combined = 0.5 * std_score + 0.5 * acf_norm

    alarm_idx = np.where(combined > 0.50)[0]
    detected_at = int(alarm_idx[0]) if len(alarm_idx) > 0 else None

    return {"detected_at": detected_at, "combined_scores": combined}


RUNNERS_B = {
    "Standard ensemble":  lambda r: run_ensemble(r, method="soft", threshold=0.50),
    "ACF-CUSUM":          run_acf_cusum,
    "ACF-enhanced ensemble": run_acf_ensemble,
}

# Build scenario list manually (CorrelationShiftSimulator is not a BaseSimulator
# subclass, so we call run_benchmark with a thin wrapper)
print(f"\nRunning correlation-shift experiment: {N_TRIALS} trials …", flush=True)

rows_B = []
for trial in range(N_TRIALS):
    np.random.seed(SEED + trial)
    sim  = CorrelationShiftSimulator(mu=0.03, phi_pre=0.0, phi_post=0.8,
                                     sigma=0.005, changepoint=TRUE_CP)
    data  = sim.generate_data(N_TS, N_SH)
    rates = extract_bit_flip_rate(data)

    for det_name, runner in RUNNERS_B.items():
        result  = runner(rates)
        det_at  = result.get("detected_at", None)
        metrics = evaluate_single_trial(det_at, TRUE_CP, TOLERANCE)
        rows_B.append({"detector": det_name, "trial": trial,
                        "detected_at": det_at, **metrics})

df_B = pd.DataFrame(rows_B)

summary_B = (
    df_B.groupby("detector")
    .agg(
        recall        = ("detected",    "mean"),
        miss_rate     = ("missed",      "mean"),
        false_alarm   = ("false_alarm", "mean"),
        mean_delay    = ("delay",       lambda x: x.dropna().mean()),
        n_trials      = ("trial",       "count"),
    )
    .reindex(["Standard ensemble", "ACF-CUSUM", "ACF-enhanced ensemble"])
)

print("\n── Correlation-shift results ──")
print(summary_B.round(3).to_string())

std_recall = summary_B.loc["Standard ensemble", "recall"]
acf_recall = summary_B.loc["ACF-enhanced ensemble", "recall"]
acf_far    = summary_B.loc["ACF-enhanced ensemble", "false_alarm"]

# "Lost performance" is the gap between the standard ensemble and perfect
# recall (1.0) — NOT the recall it already achieved.
gap        = 1.0 - std_recall
recovery   = (acf_recall - std_recall)    # gain from ACF enhancement
recovery_pct = recovery / max(gap, 1e-6) * 100 if gap > 0 else 0

print(f"\n★ CLAIM 3 — ACF recovery:")
print(f"   Standard ensemble recall:      {std_recall:.1%}")
print(f"   ACF-enhanced ensemble recall:  {acf_recall:.1%}")
print(f"   Absolute recall gain:          +{recovery:.1%}")
print(f"   Recovery of lost performance:  {recovery_pct:.1f}%")
print(f"   ACF-enhanced false-alarm rate: {acf_far:.1%}")

if acf_far > 0.20:
    print("\n   ⚠  WARNING: the ACF-enhanced detector's false-alarm rate is high.")
    print("      A detector that fires before the changepoint in most trials is")
    print("      not a usable detector, and its recall gain is NOT a real result.")
    print("      Calibrate the rolling-ACF null distribution before claiming this.")

# ─────────────────────────────────────────────────────────────────────────────
total_trials = len(df_A) + len(df_B)

print("\n" + "=" * 65)
print("READY-TO-PASTE CONCLUSION SENTENCE")
print("=" * 65)

# Per-detector degradation is the defensible claim: comparing the SAME detector
# against itself at phi=0 vs phi=0.9.  Averaging miss rates ACROSS detectors
# hides the fact that some are weak even under independent noise.
per_det = (
    ind_A.pivot_table(index="detector", columns="phi", values="miss_rate")
)
worst_det   = (per_det[0.9] - per_det[0.0]).idxmax()
worst_indep = per_det.loc[worst_det, 0.0]
worst_corr  = per_det.loc[worst_det, 0.9]
best_det    = per_det[0.9].idxmin()
best_corr   = per_det.loc[best_det, 0.9]

print("\n── Per-detector miss rate: independent (phi=0) vs correlated (phi=0.9) ──")
print(per_det[[0.0, 0.9]].rename(columns={0.0: "phi=0.0", 0.9: "phi=0.9"})
      .to_string(float_format=lambda v: f"{v:.0%}"))

print(f"""
Benchmarked CUSUM, Bayesian online change-point detection, and a windowed-KL
detector across {total_trials} evaluation runs on simulated quantum noise, showing
that AR(1) temporal correlation raises {worst_det}'s miss rate from
{worst_indep:.0%} to {worst_corr:.0%} (phi 0 -> 0.9) while {best_det} degrades to only
{best_corr:.0%}, isolating which change-point assumptions survive non-Markovian noise.
""")

print("=" * 65)
print("NOTES ON WHAT IS AND IS NOT SUPPORTED")
print("=" * 65)
print(f"  SUPPORTED  : per-detector degradation, phi=0 -> phi=0.9")
print(f"               {worst_det}: {worst_indep:.0%} -> {worst_corr:.0%} miss")
print(f"               {best_det}: {per_det.loc[best_det, 0.0]:.0%} -> {best_corr:.0%} miss")
print(f"  SUPPORTED  : standard ensemble is blind to a pure correlation-structure")
print(f"               change ({std_recall:.0%} recall) — expected, it watches the mean")
print(f"  NOT YET    : the ACF-aware 'fix'. Recall {acf_recall:.0%} but false-alarm")
print(f"               rate {acf_far:.0%}. Do not cite as a recovery result.")
print(f"  CAUTION    : averaging miss rate across detectors ({independent_miss_mean:.0%} at phi=0)")
print(f"               hides that Window is already weak on independent noise.")
print(f"\n  (Based on {N_TRIALS} trials per scenario, tolerance={TOLERANCE} timesteps,")
print(f"   N={N_SH} shots, T={N_TS} timesteps, true changepoint at t={TRUE_CP})")
