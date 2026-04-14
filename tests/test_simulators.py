"""Tests for quantum circuit simulators."""

import numpy as np
import pytest

def test_package_import():
    """Test that the package can be imported."""
    import quantum_drift_detector
    assert quantum_drift_detector.__version__ == "0.1.0"

def test_simulators_import():
    """Test that simulators module can be imported."""
    from quantum_drift_detector import simulators
    assert simulators is not None

def test_depolarizing_simulator():
    """Test the depolarizing simulator generates correct shape data."""
    from quantum_drift_detector.simulators import DepolarizingSimulator

    sim = DepolarizingSimulator(error_rate_pre=0.0, error_rate_post=1.0, changepoint=5)
    data = sim.generate_data(n_timesteps=10, n_shots=100)

    assert data.shape == (10, 100)
    assert data.dtype == int
    assert np.all((data == 0) | (data == 1))  # All values are 0 or 1

    # Check that changepoint is visible in means
    pre_mean = np.mean(data[:5, :])
    post_mean = np.mean(data[5:, :])

    assert pre_mean < 0.1  # Should be close to 0
    assert post_mean > 0.9  # Should be close to 1

def test_bit_flip_rate_extraction():
    """Test bit-flip rate extraction."""
    from quantum_drift_detector.features import extract_bit_flip_rate

    # Create test data: first half all 0s, second half all 1s
    data = np.zeros((10, 5))
    data[5:, :] = 1

    rates = extract_bit_flip_rate(data)

    expected = np.array([0.0] * 5 + [1.0] * 5)
    np.testing.assert_array_equal(rates, expected)


# ---------------------------------------------------------------------------
# Phase 3 simulator tests
# ---------------------------------------------------------------------------

def test_dephasing_simulator_import():
    """DephasingSimulator can be imported."""
    from quantum_drift_detector.simulators import DephasingSimulator
    assert DephasingSimulator is not None


def test_dephasing_simulator_shape():
    """DephasingSimulator returns correct shape."""
    from quantum_drift_detector.simulators import DephasingSimulator
    sim = DephasingSimulator(gamma_pre=0.2, gamma_post=1.5, changepoint=5)
    data = sim.generate_data(n_timesteps=10, n_shots=100)
    assert data.shape == (10, 100)
    assert data.dtype == int
    assert np.all((data == 0) | (data == 1))


def test_correlated_simulator_import():
    """CorrelatedNoiseSimulator can be imported."""
    from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
    assert CorrelatedNoiseSimulator is not None


def test_correlated_simulator_shape():
    """CorrelatedNoiseSimulator returns correct shape."""
    from quantum_drift_detector.simulators import CorrelatedNoiseSimulator
    sim = CorrelatedNoiseSimulator(mu_pre=0.01, mu_post=0.05, phi=0.5)
    data = sim.generate_data(n_timesteps=10, n_shots=100)
    assert data.shape == (10, 100)
    assert data.dtype == int
    assert np.all((data == 0) | (data == 1))