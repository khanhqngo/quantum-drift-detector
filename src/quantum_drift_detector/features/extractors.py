"""Feature extraction from quantum measurement data."""

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Phase 1 features
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 3 features
# ---------------------------------------------------------------------------

def extract_entropy(data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Extract the Shannon entropy of the binary outcome distribution at each timestep.

    Plain English: Shannon entropy measures how "uncertain" the measurement
    outcomes are. If a coin always lands heads (p = 0 or p = 1), entropy = 0 —
    there is no surprise. If the coin is perfectly fair (p = 0.5), entropy = 1
    bit — maximum uncertainty. For quantum noise monitoring:
      - Low entropy → outcomes are predictable → low (or very high) noise
      - High entropy → outcomes are near 50/50 → noise has scrambled the qubit

    For a Bernoulli distribution with parameter p (the bit-flip rate):
        H(p) = −p·log₂(p) − (1−p)·log₂(1−p)

    Note: entropy is symmetric around p = 0.5 and returns the same value for
    p and 1−p, so it is most useful as a COMPLEMENT to the bit-flip rate rather
    than a replacement.

    Args:
        data: 2D array of shape (n_timesteps, n_shots) with 0/1 outcomes.
        eps : Small constant to avoid log(0) when p = 0 or p = 1.

    Returns:
        1D array of shape (n_timesteps,) with entropy values in [0, 1] bits.

    Example:
        >>> data = np.array([[0,0,0,0], [1,1,0,0], [1,1,1,1]])
        >>> h = extract_entropy(data)
        >>> print(h.round(3))  # [0.0, 1.0, 0.0]
    """
    rates = extract_bit_flip_rate(data)
    p = np.clip(rates, eps, 1.0 - eps)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def extract_rolling_mean(
    bit_flip_rates: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute a rolling (moving) mean of the bit-flip rate.

    Plain English: instead of looking at a single noisy timestep, average the
    last `window` timesteps.  This smooths out shot-to-shot randomness and
    makes underlying trends more visible.  A changepoint shows up as a step
    in this smoothed signal.

    Entries before the first full window are computed over the available data
    (equivalent to pandas' min_periods=1).

    Args:
        bit_flip_rates: 1D array of bit-flip rates, shape (n_timesteps,).
        window        : Number of timesteps to average over.

    Returns:
        1D array of shape (n_timesteps,) with rolling mean values.

    Example:
        >>> rates = np.array([0.01, 0.01, 0.05, 0.05])
        >>> extract_rolling_mean(rates, window=2)
        array([0.01, 0.01, 0.03, 0.05])
    """
    n = len(bit_flip_rates)
    result = np.zeros(n)
    for t in range(n):
        start = max(0, t - window + 1)
        result[t] = np.mean(bit_flip_rates[start : t + 1])
    return result


def extract_rolling_variance(
    bit_flip_rates: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute a rolling variance of the bit-flip rate.

    Plain English: variance measures how much the bit-flip rate bounces around
    within a time window.  Correlated (non-Markovian) noise tends to have
    LOWER short-window variance than independent noise of the same mean —
    because correlated noise is "sticky" and doesn't jump around as much.
    A changepoint often appears as a spike in rolling variance as the signal
    transitions between two levels.

    Entries with fewer than 2 data points return 0.

    Args:
        bit_flip_rates: 1D array of bit-flip rates, shape (n_timesteps,).
        window        : Size of the rolling window.

    Returns:
        1D array of shape (n_timesteps,) with rolling variance values.

    Example:
        >>> rates = np.array([0.01, 0.01, 0.05, 0.05])
        >>> extract_rolling_variance(rates, window=2)
        array([0.   , 0.   , 0.0008, 0.   ])
    """
    n = len(bit_flip_rates)
    result = np.zeros(n)
    for t in range(n):
        start = max(0, t - window + 1)
        segment = bit_flip_rates[start : t + 1]
        result[t] = np.var(segment) if len(segment) >= 2 else 0.0
    return result


def extract_rolling_autocorr(
    bit_flip_rates: np.ndarray,
    window: int = 20,
    lag: int = 1,
) -> np.ndarray:
    """
    Compute the rolling lag-k autocorrelation of the bit-flip rate.

    Plain English: autocorrelation at lag k asks "how similar is the signal
    today to what it was k steps ago?"  For independent (Markovian) noise,
    ACF(k > 0) ≈ 0 — past and present are unrelated.  For correlated
    (non-Markovian) noise, ACF(1) > 0 — the current error rate is related to
    the previous one.

    Tracking the rolling ACF over time lets us detect a CHANGE IN CORRELATION
    STRUCTURE — not just a change in mean level.  This is the key insight for
    the Giarmatzi / Tonekaboni group's research: they want to know when the
    temporal correlation regime changes.

    Args:
        bit_flip_rates: 1D array of bit-flip rates, shape (n_timesteps,).
        window        : Number of timesteps in each rolling window.
                        Must be at least lag + 2.
        lag           : Autocorrelation lag (default 1 = consecutive timesteps).

    Returns:
        1D float array of shape (n_timesteps,).  Entries before the first full
        window are set to 0.  Values range from −1 to +1.

    Example:
        >>> import numpy as np
        >>> rates = np.array([0.01]*10 + [0.05]*10, dtype=float)
        >>> acf = extract_rolling_autocorr(rates, window=5, lag=1)
        >>> acf.shape
        (20,)
    """
    n = len(bit_flip_rates)
    result = np.zeros(n)

    for t in range(window - 1, n):
        segment = bit_flip_rates[t - window + 1 : t + 1]

        x = segment[: window - lag]   # x_t
        y = segment[lag:]             # x_{t+lag}

        # Pearson correlation between x and its lagged version.
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())

        if denom < 1e-12:
            result[t] = 0.0   # constant window → no meaningful autocorrelation
        else:
            result[t] = float((x_c * y_c).sum() / denom)

    return result


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def extract_features(
    data: np.ndarray,
    feature_names: Optional[list] = None,
    window: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Extract multiple features from raw measurement data.

    Args:
        data         : 2D array of shape (n_timesteps, n_shots) with 0/1 outcomes.
        feature_names: List of feature names to extract.  Pass None to extract
                       all available features.  Supported names:
                           'bit_flip_rate', 'entropy',
                           'rolling_mean', 'rolling_variance', 'rolling_autocorr'
        window       : Window size used by rolling features.

    Returns:
        Dictionary mapping feature names to 1D arrays of shape (n_timesteps,).

    Example:
        >>> data = np.random.randint(0, 2, (20, 100))
        >>> feats = extract_features(data, ['bit_flip_rate', 'entropy'])
        >>> feats.keys()
        dict_keys(['bit_flip_rate', 'entropy'])
    """
    available = [
        "bit_flip_rate",
        "entropy",
        "rolling_mean",
        "rolling_variance",
        "rolling_autocorr",
    ]
    if feature_names is None:
        feature_names = available

    rates = extract_bit_flip_rate(data)
    features: Dict[str, np.ndarray] = {}

    if "bit_flip_rate" in feature_names:
        features["bit_flip_rate"] = rates
    if "entropy" in feature_names:
        features["entropy"] = extract_entropy(data)
    if "rolling_mean" in feature_names:
        features["rolling_mean"] = extract_rolling_mean(rates, window=window)
    if "rolling_variance" in feature_names:
        features["rolling_variance"] = extract_rolling_variance(rates, window=window)
    if "rolling_autocorr" in feature_names:
        features["rolling_autocorr"] = extract_rolling_autocorr(rates, window=window * 2)

    return features
