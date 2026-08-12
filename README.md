# Quantum Drift Detector

A Python toolkit for detecting when a quantum computer's noise behavior suddenly changes over time, using classical statistical change-point detection methods.

## Overview

This package simulates streams of quantum measurement data from circuits whose noise level shifts at unknown times, then detects those shift points using multiple classical statistical methods. It's designed for researchers studying quantum device stability and noise characterization.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from quantum_drift_detector.simulators import DepolarizingSimulator
from quantum_drift_detector.detectors import CusumDetector
from quantum_drift_detector.visualization import plot_drift_detection

# Simulate data with a changepoint at t=50
sim = DepolarizingSimulator(error_rate_pre=0.01, error_rate_post=0.05, changepoint=50)
data = sim.generate_data(n_timesteps=100, n_shots=1000)

# Extract features
from quantum_drift_detector.features import extract_bit_flip_rate
rates = extract_bit_flip_rate(data)

# Plot results
plot_drift_detection(data, true_changepoint=50)
```

## Current Status

### Phase 1: Foundation (Complete)
- [x] Project structure and packaging
- [x] Depolarizing noise simulator
- [x] Bit-flip rate feature extraction
- [x] Basic plotting utilities
- [x] Unit tests
- [x] Tutorial notebooks (01_intro, 02_simulating_drift)

### Phase 2: Detection Algorithms (Next)
- [ ] CUSUM detector
- [ ] Bayesian Online Change-Point Detection
- [ ] Sliding-window KL-divergence detector
- [ ] Comparative evaluation

## Project Structure

- `simulators/`: Quantum circuit noise simulation
- `detectors/`: Change-point detection algorithms
- `features/`: Statistical feature extraction from measurement data
- `visualization/`: Plotting utilities

## Notebooks

See the `notebooks/` directory for tutorials and examples.

## License

MIT License
