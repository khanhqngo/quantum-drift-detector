"""Plotting utilities for drift detection results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_bit_flip_rate_over_time(bit_flip_rates: np.ndarray,
                                true_changepoint: Optional[int] = None,
                                detected_changepoint: Optional[int] = None,
                                title: str = "Bit-Flip Rate Over Time"):
    """
    Plot the bit-flip rate time series with optional changepoint markers.

    Args:
        bit_flip_rates: 1D array of bit-flip rates over time.
        true_changepoint: Timestep of the true changepoint (if known).
        detected_changepoint: Timestep of the detected changepoint.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    timesteps = np.arange(len(bit_flip_rates))

    plt.plot(timesteps, bit_flip_rates, 'b-', alpha=0.7, label='Bit-flip rate')

    if true_changepoint is not None:
        plt.axvline(x=true_changepoint, color='red', linestyle='--',
                   label=f'True changepoint (t={true_changepoint})')

    if detected_changepoint is not None:
        plt.axvline(x=detected_changepoint, color='green', linestyle='--',
                   label=f'Detected changepoint (t={detected_changepoint})')

    plt.xlabel('Time step')
    plt.ylabel('Bit-flip rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drift_detection(data: np.ndarray,
                        true_changepoint: Optional[int] = None,
                        detected_changepoint: Optional[int] = None):
    """
    Convenience function to plot drift detection results.

    Extracts bit-flip rate from data and plots it with changepoint markers.

    Args:
        data: 2D array of measurement outcomes (n_timesteps, n_shots).
        true_changepoint: True changepoint timestep.
        detected_changepoint: Detected changepoint timestep.
    """
    from ..features import extract_bit_flip_rate

    bit_flip_rates = extract_bit_flip_rate(data)
    plot_bit_flip_rate_over_time(bit_flip_rates, true_changepoint, detected_changepoint)