"""
Tests for Phase 2 change-point detection algorithms.

Each test uses a synthetic bit-flip rate series with a large, obvious shift
(0.0 → 1.0) to check correctness of shape, dtype, and detection logic.
Threshold / sensitivity tuning is left to the notebooks.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obvious_shift():
    """
    A bit-flip rate series that jumps from 0.0 to 1.0 at timestep 50.
    This is an extreme shift that every detector should catch easily.
    """
    rates = np.array([0.0] * 50 + [1.0] * 50, dtype=float)
    return rates, 50  # (series, true_changepoint)


@pytest.fixture
def no_shift():
    """A constant series with no changepoint."""
    return np.full(100, 0.02, dtype=float)


# ---------------------------------------------------------------------------
# CUSUM tests
# ---------------------------------------------------------------------------

class TestCUSUM:
    def test_returns_correct_keys(self, obvious_shift):
        from quantum_drift_detector.detectors import run_cusum
        rates, _ = obvious_shift
        result = run_cusum(rates, target_mean=0.0, allowance=0.1, threshold=1.0)
        assert set(result.keys()) == {"scores", "detected_at", "threshold"}

    def test_scores_shape(self, obvious_shift):
        from quantum_drift_detector.detectors import run_cusum
        rates, _ = obvious_shift
        result = run_cusum(rates, target_mean=0.0, allowance=0.1, threshold=1.0)
        assert result["scores"].shape == (100,)

    def test_scores_non_negative(self, obvious_shift):
        from quantum_drift_detector.detectors import run_cusum
        rates, _ = obvious_shift
        result = run_cusum(rates, target_mean=0.0, allowance=0.1, threshold=1.0)
        assert np.all(result["scores"] >= 0.0)

    def test_detects_obvious_shift(self, obvious_shift):
        from quantum_drift_detector.detectors import run_cusum
        rates, true_cp = obvious_shift
        result = run_cusum(rates, target_mean=0.0, allowance=0.1, threshold=1.0)
        assert result["detected_at"] is not None
        # Detection should be close to the true changepoint (within 15 steps)
        assert abs(result["detected_at"] - true_cp) <= 15

    def test_no_false_alarm_on_flat_series(self, no_shift):
        from quantum_drift_detector.detectors import run_cusum
        # With a generous threshold, a flat series should produce no alarm.
        result = run_cusum(no_shift, target_mean=0.02, allowance=0.01, threshold=10.0)
        assert result["detected_at"] is None

    def test_threshold_echoed(self, obvious_shift):
        from quantum_drift_detector.detectors import run_cusum
        rates, _ = obvious_shift
        result = run_cusum(rates, target_mean=0.0, allowance=0.1, threshold=0.5)
        assert result["threshold"] == 0.5


# ---------------------------------------------------------------------------
# Bayesian BOCD tests
# ---------------------------------------------------------------------------

class TestBOCD:
    def test_returns_correct_keys(self, obvious_shift):
        from quantum_drift_detector.detectors import run_bocd
        rates, _ = obvious_shift
        result = run_bocd(rates)
        assert set(result.keys()) == {"changepoint_probs", "detected_at", "threshold"}

    def test_changepoint_probs_shape(self, obvious_shift):
        from quantum_drift_detector.detectors import run_bocd
        rates, _ = obvious_shift
        result = run_bocd(rates)
        assert result["changepoint_probs"].shape == (100,)

    def test_changepoint_probs_in_range(self, obvious_shift):
        from quantum_drift_detector.detectors import run_bocd
        rates, _ = obvious_shift
        result = run_bocd(rates)
        probs = result["changepoint_probs"]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0 + 1e-9)  # allow tiny float rounding

    def test_detects_obvious_shift(self, obvious_shift):
        from quantum_drift_detector.detectors import run_bocd
        rates, true_cp = obvious_shift
        # For the extreme 0→1 shift, set prior_var wide enough that 1.0 is not
        # ruled out under the fresh-start hypothesis (obs_var=1e-6, prior_var=0.25
        # gives predictive std ≈ 0.5, so z ≈ 2 which is reasonable).
        result = run_bocd(rates, mu_0=0.0, obs_var=1e-6, prior_var=0.25, threshold=0.3)
        assert result["detected_at"] is not None
        assert abs(result["detected_at"] - true_cp) <= 15

    def test_no_false_alarm_on_flat_series(self, no_shift):
        from quantum_drift_detector.detectors import run_bocd
        # With a very high threshold, a perfectly flat series should not trigger.
        result = run_bocd(no_shift, mu_0=0.02, obs_var=1e-5, prior_var=0.01, threshold=0.9)
        assert result["detected_at"] is None


# ---------------------------------------------------------------------------
# Sliding-window tests
# ---------------------------------------------------------------------------

class TestWindowDetector:
    def test_returns_correct_keys(self, obvious_shift):
        from quantum_drift_detector.detectors import run_window_detector
        rates, _ = obvious_shift
        result = run_window_detector(rates, window_size=10, threshold=0.05)
        assert set(result.keys()) == {"scores", "detected_at", "threshold"}

    def test_scores_shape(self, obvious_shift):
        from quantum_drift_detector.detectors import run_window_detector
        rates, _ = obvious_shift
        result = run_window_detector(rates, window_size=10, threshold=0.05)
        assert result["scores"].shape == (100,)

    def test_scores_non_negative(self, obvious_shift):
        from quantum_drift_detector.detectors import run_window_detector
        rates, _ = obvious_shift
        result = run_window_detector(rates, window_size=10, threshold=0.05)
        assert np.all(result["scores"] >= 0.0)

    def test_detects_obvious_shift(self, obvious_shift):
        from quantum_drift_detector.detectors import run_window_detector
        rates, true_cp = obvious_shift
        result = run_window_detector(rates, window_size=10, threshold=0.05)
        assert result["detected_at"] is not None
        # Window detector is offset by window_size; allow generous margin.
        assert abs(result["detected_at"] - true_cp) <= 20

    def test_no_detection_on_flat_series(self, no_shift):
        from quantum_drift_detector.detectors import run_window_detector
        # With a high threshold, a perfectly flat series has zero KL divergence.
        result = run_window_detector(no_shift, window_size=10, threshold=10.0)
        assert result["detected_at"] is None

    def test_boundary_scores_are_zero(self, obvious_shift):
        from quantum_drift_detector.detectors import run_window_detector
        rates, _ = obvious_shift
        W = 10
        result = run_window_detector(rates, window_size=W, threshold=0.05)
        # Scores outside the valid range should be exactly 0.
        assert result["scores"][0] == 0.0
        assert result["scores"][-1] == 0.0


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_returns_correct_keys(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        result = run_ensemble(rates)
        expected_keys = {
            "combined_scores", "cusum_scores_norm", "bocd_scores",
            "window_scores_norm", "individual_alarms", "detected_at",
            "threshold", "method",
        }
        assert expected_keys.issubset(result.keys())

    def test_combined_scores_shape(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        result = run_ensemble(rates)
        assert result["combined_scores"].shape == (100,)

    def test_combined_scores_in_range(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        result = run_ensemble(rates)
        scores = result["combined_scores"]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_soft_detects_obvious_shift(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, true_cp = obvious_shift
        result = run_ensemble(rates, method="soft", threshold=0.5)
        assert result["detected_at"] is not None
        assert abs(result["detected_at"] - true_cp) <= 20

    def test_hard_detects_obvious_shift(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, true_cp = obvious_shift
        result = run_ensemble(rates, method="hard", min_votes=2)
        assert result["detected_at"] is not None
        assert abs(result["detected_at"] - true_cp) <= 20

    def test_individual_alarms_keys(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        result = run_ensemble(rates)
        assert set(result["individual_alarms"].keys()) == {"cusum", "bocd", "window"}

    def test_no_detection_on_flat_series(self, no_shift):
        from quantum_drift_detector.detectors import run_ensemble
        # A very flat series with a high threshold should not alarm.
        result = run_ensemble(
            no_shift,
            cusum_params={"threshold": 10.0},
            bocd_params={"threshold": 0.99},
            window_params={"threshold": 100.0},
            threshold=0.99,
            method="soft",
        )
        assert result["detected_at"] is None

    def test_invalid_method_raises(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        with pytest.raises(ValueError):
            run_ensemble(rates, method="unknown")

    def test_invalid_weights_raises(self, obvious_shift):
        from quantum_drift_detector.detectors import run_ensemble
        rates, _ = obvious_shift
        with pytest.raises(ValueError):
            run_ensemble(rates, weights=[1.0, 1.0])  # wrong length
