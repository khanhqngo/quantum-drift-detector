# Quantum Drift Detector

A Python toolkit for detecting when a quantum computer's noise behavior suddenly changes over time, using classical statistical change-point detection methods.

## Overview

Quantum devices drift. Calibration decays, control electronics warm up, and two-level-system defects wander in and out of resonance — so the error rate a circuit sees at 9am is not the one it sees at 5pm. This package simulates streams of quantum measurement data from circuits whose noise level shifts at unknown times, then detects those shift points using several classical statistical methods, and benchmarks how well each one holds up.

The interesting question is not "can you detect a step change in a noisy mean" (you can), but **what happens when the noise is temporally correlated** — i.e. non-Markovian, as real superconducting hardware tends to be. That is where standard detectors quietly fall apart, and quantifying that failure is the main result here.

## Installation

```bash
pip install -e .
```

For development (tests, notebooks):
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from quantum_drift_detector.simulators import DepolarizingSimulator
from quantum_drift_detector.features import extract_bit_flip_rate
from quantum_drift_detector.detectors import run_cusum, run_ensemble

# Simulate 100 timesteps x 1000 shots, with the error rate jumping 1% -> 5% at t=50
sim = DepolarizingSimulator(error_rate_pre=0.01, error_rate_post=0.05, changepoint=50)
data = sim.generate_data(n_timesteps=100, n_shots=1000)

# Collapse each timestep's shots into a single bit-flip rate
rates = extract_bit_flip_rate(data)

# Detect with CUSUM
result = run_cusum(rates, target_mean=0.01, allowance=0.02, threshold=0.10)
print(result["detected_at"])        # -> 55

# Or let the ensemble auto-estimate the baseline for you
ens = run_ensemble(rates, method="soft", threshold=0.50)
print(ens["detected_at"])           # -> 50
```

Plotting:

```python
from quantum_drift_detector.visualization import plot_drift_detection

plot_drift_detection(data, true_changepoint=50, detected_changepoint=ens["detected_at"])
```

## What's Implemented

### Simulators (`simulators/`)
| Class | Models |
|---|---|
| `DepolarizingSimulator` | Step change in depolarizing error rate |
| `DephasingSimulator` | Phase-damping noise with a drifting coherence time |
| `CorrelatedNoiseSimulator` | AR(1) non-Markovian noise — error rate itself is autocorrelated with parameter `phi` |

### Detectors (`detectors/`)
| Function | Method |
|---|---|
| `run_cusum` | One-sided cumulative sum on the bit-flip rate |
| `run_bocd` | Bayesian online change-point detection (run-length posterior) |
| `run_window_detector` | Sliding-window KL divergence between adjacent windows |
| `run_ensemble` | Soft/hard voting over the three above, with auto-estimated baseline |

### Features (`features/`)
`extract_bit_flip_rate`, `extract_entropy`, `extract_rolling_mean`, `extract_rolling_variance`, `extract_rolling_autocorr`, `extract_features`

### Evaluation (`utils.py`)
`evaluate_single_trial`, `run_benchmark`, `summarise_benchmark` — a scenario/trial harness that sweeps simulators × detectors × seeds and scores each run for recall, false-alarm rate, and detection delay.

## Results

All numbers below come from `verify_conclusion_numbers.py`, which is reproducible from a clean checkout:

```bash
python verify_conclusion_numbers.py
```

**Setup:** 50 trials per scenario, 200 timesteps, 500 shots/timestep, true changepoint at t=100, mean shift Δμ = 0.04 (1% → 5%). A detection counts as a true positive only if the alarm fires within 15 timesteps *after* the changepoint; anything earlier is a false alarm. 950 evaluation runs total.

### Miss rate vs. noise correlation

`phi` is the AR(1) autocorrelation of the underlying error rate. `phi = 0` is independent (Markovian) noise; `phi = 0.9` is strongly correlated, which is the regime real hardware lives in.

| `phi` | CUSUM | BOCD | Window (KL) |
|---:|---:|---:|---:|
| 0.0 (independent) | **0%** | 24% | 76% |
| 0.7 | **0%** | 68% | 74% |
| 0.8 | **2%** | 76% | 78% |
| 0.9 | **20%** | 90% | 90% |

**Takeaways:**

1. **Temporal correlation degrades every detector, but not equally.** BOCD is the most fragile — its miss rate goes from 24% to 90% as `phi` rises from 0 to 0.9. It assumes i.i.d. observations around a piecewise-constant mean, so autocorrelated wander looks to it like an ongoing series of small changepoints, which flattens the run-length posterior and prevents any single alarm from crossing threshold.
2. **CUSUM is by far the most robust** at this shift size, staying at ≤2% miss rate up to `phi = 0.8` and only degrading to 20% at `phi = 0.9`. Its cumulative sum integrates over the correlated fluctuations rather than being confused by them.
3. **The sliding-window KL detector is weak regardless of correlation** (74–90% miss). At a 0.04 shift with a 10-timestep window it is simply underpowered — this is a sensitivity limitation, not a non-Markovian one.

### Open problem: detecting a change in correlation structure

A harder scenario, also in the verification script: the *mean* error rate stays fixed at 3%, but the correlation structure changes at t=100 (`phi`: 0 → 0.8). Nothing about the average rate changes — only its temporal texture.

| Detector | Recall | False-alarm rate |
|---|---:|---:|
| Standard ensemble | 2% | 0% |
| ACF-CUSUM (CUSUM on rolling lag-1 autocorrelation) | 6% | 90% |

The standard ensemble is essentially blind here (2% recall), which is the expected and correct result — it watches the mean, and the mean did not move.

The autocorrelation-aware detector is **not yet a working fix.** Its nominal recall improvement (2% → 6%) is swamped by a 90% false-alarm rate: it fires before the true changepoint in 9 of 10 trials, so it is closer to an always-on alarm than a detector. The rolling-ACF estimator over a 20-sample window is too noisy at baseline, and the fixed `target_mean=0.0` / `allowance=0.05` CUSUM parameters do not account for that variance. Making this work needs a properly calibrated null distribution for the ACF statistic — that is the next piece of work, and the current numbers should not be read as a positive result.

## Testing

```bash
pytest tests/ -q
```

62 tests currently pass, covering simulator output shapes and statistics, all four detectors, and every feature extractor.

## Project Structure

```
src/quantum_drift_detector/
├── simulators/      # Quantum circuit noise simulation
├── detectors/       # Change-point detection algorithms
├── features/        # Statistical feature extraction from measurement data
├── visualization/   # Plotting utilities
└── utils.py         # Benchmark + evaluation harness
notebooks/           # Tutorials, walkthroughs, and result reproduction
tests/               # Unit tests
verify_conclusion_numbers.py   # Reproduces every number in the Results section
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_intro_quantum_noise.ipynb` | What quantum noise is and how it shows up in measurement data |
| `02_simulating_drift.ipynb` | Building drift scenarios with the simulators |
| `03_detection_methods.ipynb` | CUSUM, BOCD, and windowed KL from first principles |
| `04_correlated_noise.ipynb` | Non-Markovian noise and where standard detectors break |
| `05_comparison_and_results.ipynb` | Full benchmark sweep and comparison |

## Status

Phases 1–4 complete: simulators, detectors, feature extraction, ensemble, and the evaluation harness are all implemented and tested.

Next up:
- [ ] Calibrated null distribution for the rolling-ACF statistic, to make correlation-structure detection usable
- [ ] Multi-qubit and crosstalk-aware drift scenarios
- [ ] Validation against real device data (e.g. IBM Quantum calibration logs)

## License

MIT License
