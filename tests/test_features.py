"""
Tests for Phase 3 feature extractors.

Each test checks shape, dtype, boundary values, and known analytical results.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_zeros(n_timesteps: int = 20, n_shots: int = 50) -> np.ndarray:
    """All outcomes are 0 — perfect qubit, no noise."""
    return np.zeros((n_timesteps, n_shots), dtype=int)


def _all_ones(n_timesteps: int = 20, n_shots: int = 50) -> np.ndarray:
    """All outcomes are 1 — fully flipped qubit."""
    return np.ones((n_timesteps, n_shots), dtype=int)


def _half_half(n_timesteps: int = 20, n_shots: int = 100) -> np.ndarray:
    """Exactly half 0s and half 1s at every timestep — maximum uncertainty."""
    data = np.zeros((n_timesteps, n_shots), dtype=int)
    data[:, n_shots // 2 :] = 1
    return data


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

class TestExtractEntropy:
    def test_shape(self):
        from quantum_drift_detector.features import extract_entropy
        data = np.random.randint(0, 2, (30, 200))
        result = extract_entropy(data)
        assert result.shape == (30,)

    def test_all_zeros_gives_zero_entropy(self):
        from quantum_drift_detector.features import extract_entropy
        result = extract_entropy(_all_zeros())
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_all_ones_gives_zero_entropy(self):
        from quantum_drift_detector.features import extract_entropy
        result = extract_entropy(_all_ones())
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_half_half_gives_one_bit_entropy(self):
        from quantum_drift_detector.features import extract_entropy
        result = extract_entropy(_half_half())
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_entropy_in_range(self):
        from quantum_drift_detector.features import extract_entropy
        data = np.random.randint(0, 2, (50, 100))
        result = extract_entropy(data)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Rolling mean
# ---------------------------------------------------------------------------

class TestExtractRollingMean:
    def test_shape(self):
        from quantum_drift_detector.features import extract_rolling_mean
        rates = np.random.rand(40)
        result = extract_rolling_mean(rates, window=5)
        assert result.shape == (40,)

    def test_constant_series(self):
        from quantum_drift_detector.features import extract_rolling_mean
        rates = np.full(20, 0.03)
        result = extract_rolling_mean(rates, window=5)
        np.testing.assert_allclose(result, 0.03, atol=1e-10)

    def test_window_of_one_is_identity(self):
        from quantum_drift_detector.features import extract_rolling_mean
        rates = np.array([0.01, 0.05, 0.03, 0.07])
        result = extract_rolling_mean(rates, window=1)
        np.testing.assert_array_equal(result, rates)

    def test_known_two_step_average(self):
        from quantum_drift_detector.features import extract_rolling_mean
        rates = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        result = extract_rolling_mean(rates, window=2)
        expected = np.array([0.0, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Rolling variance
# ---------------------------------------------------------------------------

class TestExtractRollingVariance:
    def test_shape(self):
        from quantum_drift_detector.features import extract_rolling_variance
        rates = np.random.rand(40)
        result = extract_rolling_variance(rates, window=5)
        assert result.shape == (40,)

    def test_constant_series_zero_variance(self):
        from quantum_drift_detector.features import extract_rolling_variance
        rates = np.full(20, 0.03)
        result = extract_rolling_variance(rates, window=5)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_non_negative(self):
        from quantum_drift_detector.features import extract_rolling_variance
        rates = np.random.rand(30)
        result = extract_rolling_variance(rates, window=5)
        assert np.all(result >= 0.0)

    def test_first_entry_is_zero(self):
        from quantum_drift_detector.features import extract_rolling_variance
        rates = np.array([0.01, 0.05, 0.03])
        result = extract_rolling_variance(rates, window=5)
        assert result[0] == 0.0  # only one value in window → no variance


# ---------------------------------------------------------------------------
# Rolling autocorrelation
# ---------------------------------------------------------------------------

class TestExtractRollingAutocorr:
    def test_shape(self):
        from quantum_drift_detector.features import extract_rolling_autocorr
        rates = np.random.rand(50)
        result = extract_rolling_autocorr(rates, window=10, lag=1)
        assert result.shape == (50,)

    def test_values_in_range(self):
        from quantum_drift_detector.features import extract_rolling_autocorr
        rates = np.random.rand(60)
        result = extract_rolling_autocorr(rates, window=10, lag=1)
        assert np.all(result >= -1.0 - 1e-9)
        assert np.all(result <= 1.0 + 1e-9)

    def test_early_entries_are_zero(self):
        from quantum_drift_detector.features import extract_rolling_autocorr
        rates = np.random.rand(50)
        result = extract_rolling_autocorr(rates, window=10, lag=1)
        # Entries before the first full window should be 0.
        assert np.all(result[:9] == 0.0)

    def test_perfectly_correlated_series(self):
        """An AR(1) series with phi=0.99 should have ACF(1) near +1."""
        from quantum_drift_detector.features import extract_rolling_autocorr
        np.random.seed(0)
        rates = np.zeros(100)
        rates[0] = 0.05
        for t in range(1, 100):
            rates[t] = 0.05 + 0.99 * (rates[t - 1] - 0.05) + 0.001 * np.random.randn()
        result = extract_rolling_autocorr(rates, window=30, lag=1)
        # The latter half should show strong positive autocorrelation.
        assert result[50:].mean() > 0.5

    def test_independent_series_near_zero_acf(self):
        """i.i.d. noise should give ACF ≈ 0 on average (not always in short windows)."""
        from quantum_drift_detector.features import extract_rolling_autocorr
        np.random.seed(42)
        rates = np.random.uniform(0.01, 0.05, 200)
        result = extract_rolling_autocorr(rates, window=30, lag=1)
        # Average over valid (non-zero) entries should be close to 0.
        valid = result[29:]
        assert abs(valid.mean()) < 0.2


# ---------------------------------------------------------------------------
# New simulators — shape and statistical sanity
# ---------------------------------------------------------------------------

class TestDephasingSimulator:
    def test_shape(self):
        from quantum_drift_detector.simulators import DephasingSimulator
        sim = DephasingSimulator(gamma_pre=0.2, gamma_post=1.5, changepoint=5)
        data = sim.generate_data(10, 100)
        assert data.shape == (10, 100)

    def test_binary_outcomes(self):
        from quantum_drift_detector.simulators import DephasingSimulator
        sim = DephasingSimulator()
        data = sim.generate_data(20, 200)
        assert np.all((data == 0) | (data == 1))

    def test_bit_flip_rate_bounded_by_half(self):
        """Dephasing bit-flip rate is bounded in [0, 0.5]."""
        from quantum_drift_detector.simulators import DephasingSimulator
        from quantum_drift_detector.features import extract_bit_flip_rate
        # Use extreme gamma to push rate toward 0.5.
        sim = DephasingSimulator(gamma_pre=0.0, gamma_post=10.0, changepoint=50)
        data = sim.generate_data(100, 2000)
        rates = extract_bit_flip_rate(data)
        assert np.all(rates <= 0.5 + 0.05)  # allow small statistical fluctuation

    def test_pre_rate_below_post_rate(self):
        from quantum_drift_detector.simulators import DephasingSimulator
        from quantum_drift_detector.features import extract_bit_flip_rate
        np.random.seed(1)
        sim = DephasingSimulator(gamma_pre=0.1, gamma_post=2.0, changepoint=50)
        data = sim.generate_data(100, 2000)
        rates = extract_bit_flip_rate(data)
        assert rates[:50].mean() < rates[50:].mean()


class TestCorrelatedNoiseSimulator:
    def test_shape(self):
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        sim = CorrelatedNoiseSimulator(mu_pre=0.01, mu_post=0.05, phi=0.8)
        data = sim.generate_data(10, 100)
        assert data.shape == (10, 100)

    def test_binary_outcomes(self):
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        sim = CorrelatedNoiseSimulator()
        data = sim.generate_data(20, 200)
        assert np.all((data == 0) | (data == 1))

    def test_invalid_phi_raises(self):
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        with pytest.raises(ValueError):
            CorrelatedNoiseSimulator(phi=1.5)

    def test_generate_error_rates_shape(self):
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        sim = CorrelatedNoiseSimulator(phi=0.5)
        rates = sim.generate_error_rates(50)
        assert rates.shape == (50,)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_phi_zero_resembles_markovian(self):
        """phi=0 should give autocorrelation near 0 in the generated series."""
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        from quantum_drift_detector.features import extract_rolling_autocorr
        np.random.seed(42)
        sim = CorrelatedNoiseSimulator(mu_pre=0.05, mu_post=0.05, phi=0.0, sigma=0.01)
        rates = sim.generate_error_rates(200)
        acf = extract_rolling_autocorr(rates, window=30, lag=1)
        assert abs(acf[50:].mean()) < 0.25  # near zero on average

    def test_high_phi_gives_positive_autocorrelation(self):
        """phi=0.9 should give ACF(1) clearly above 0."""
        from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
        from quantum_drift_detector.features import extract_rolling_autocorr
        np.random.seed(99)
        sim = CorrelatedNoiseSimulator(mu_pre=0.03, mu_post=0.03, phi=0.9, sigma=0.005)
        rates = sim.generate_error_rates(200)
        acf = extract_rolling_autocorr(rates, window=30, lag=1)
        assert acf[50:].mean() > 0.4
