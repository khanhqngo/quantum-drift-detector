"""Feature extraction from quantum measurement data."""

import numpy as np
from typing import Dict


def extract_bit_flip_rate(data: np.ndarray) -> np.ndarray:
    """
    Extract the bit-flip rate (fraction of 1s) at each timestep.

    This is the simplest feature: for each timestep, count what fraction
    of the measurement shots gave a 1 instead of a 0. If the circuit is
    supposed to prepare |0⟩, then this rate tells us how noisy the device is.

    Args:
        data: 2D array of shape (n_timesteps, n_shots) with 0/1 outcomes.

    Returns:
        1D array of shape (n_timesteps,) with bit-flip rates.

    Example:
        >>> data = np.array([[0, 0, 1], [1, 1, 0]])  # 2 timesteps, 3 shots each
        >>> rates = extract_bit_flip_rate(data)
        >>> print(rates)  # [0.333, 0.667]
    """
    return np.mean(data, axis=1)


def extract_features(data: np.ndarray, feature_names: list[str] = None) -> Dict[str, np.ndarray]:
    """
    Extract multiple features from measurement data.

    Args:
        data: 2D array of shape (n_timesteps, n_shots) with 0/1 outcomes.
        feature_names: List of feature names to extract. If None, extracts all available.

    Returns:
        Dictionary mapping feature names to 1D arrays of shape (n_timesteps,).
    """
    if feature_names is None:
        feature_names = ["bit_flip_rate"]

    features = {}

    if "bit_flip_rate" in feature_names:
        features["bit_flip_rate"] = extract_bit_flip_rate(data)

    return features