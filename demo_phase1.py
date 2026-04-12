#!/usr/bin/env python3
"""
Demo script for Phase 1: Basic quantum drift detection setup.

This script demonstrates:
1. Simulating quantum measurement data with a changepoint
2. Extracting the bit-flip rate feature
3. Plotting the results
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our package
from quantum_drift_detector.simulators import DepolarizingSimulator
from quantum_drift_detector.features import extract_bit_flip_rate
from quantum_drift_detector.visualization import plot_drift_detection

def main():
    print("Quantum Drift Detector - Phase 1 Demo")
    print("=" * 40)

    # Create simulator with changepoint at t=50
    # Before: 1% error rate, After: 5% error rate
    sim = DepolarizingSimulator(error_rate_pre=0.01, error_rate_post=0.05, changepoint=50)

    # Generate data: 100 timesteps, 1000 shots each
    print("Generating simulated quantum measurement data...")
    data = sim.generate_data(n_timesteps=100, n_shots=1000)
    print(f"Data shape: {data.shape}")

    # Extract bit-flip rate
    print("Extracting bit-flip rate feature...")
    bit_flip_rates = extract_bit_flip_rate(data)
    cp = sim.changepoint
    print(f"Mean bit-flip rate — pre: {bit_flip_rates[:cp].mean():.3f}, post: {bit_flip_rates[cp:].mean():.3f}")

    # Plot the results
    print("Plotting results...")
    plot_drift_detection(data, true_changepoint=50)

    print("Demo complete! You should see a plot showing the changepoint.")

if __name__ == "__main__":
    main()