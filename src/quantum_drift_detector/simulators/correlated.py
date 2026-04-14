"""
Correlated (non-Markovian) noise simulator.

Plain English — what is Markovian vs. non-Markovian noise?

MARKOVIAN noise (what Phase 1 simulated):
  Each timestep's noise is completely independent of every other timestep.
  Like flipping a (possibly biased) coin at every step — yesterday's outcome
  tells you nothing about today's.

NON-MARKOVIAN (correlated) noise (what this module simulates):
  The noise at time t is influenced by what happened at t-1, t-2, … — like
  the weather: today's temperature is related to yesterday's.

  In a quantum device this happens because:
    - Environmental fluctuations (magnetic field noise, charge noise) are
      themselves correlated processes — they don't jump randomly every
      microsecond.
    - Feedback loops in control electronics have memory.
    - Thermal fluctuations drift slowly over time.

  This is exactly the "non-Markovian" noise that Giarmatzi / Tonekaboni study.
  The key challenge: correlated noise makes change-point detection HARDER because
  the signal drifts smoothly rather than jumping sharply.

Mathematical model — AR(1) autoregressive process:

    ε_t = μ_t + φ · (ε_{t-1} − μ_t) + σ · η_t

  where
    ε_t   = error rate at timestep t  (clipped to [ε_min, ε_max])
    μ_t   = long-run mean of the error rate (shifts at the changepoint)
    φ     = autocorrelation coefficient  (0 = Markovian, near 1 = highly correlated)
    σ     = innovation standard deviation (controls within-regime volatility)
    η_t   ~ Normal(0, 1) = independent noise added each step

  When φ = 0, ε_t = μ_t + σ·η_t (independent noise, same as Markovian).
  When φ → 1, ε_t barely changes from step to step (very "sticky" noise).

  At the changepoint the mean shifts μ_pre → μ_post, but the autoregressive
  smoothing means the actual error rate drifts toward the new mean gradually
  rather than jumping instantly.  This is the key difficulty for detectors.

Reference: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015).
           Time Series Analysis: Forecasting and Control (5th ed.).
"""

import numpy as np
from typing import Optional
from .base import BaseSimulator


class CorrelatedNoiseSimulator(BaseSimulator):
    """
    Simulates quantum measurement data where the underlying error rate follows
    a correlated AR(1) process that shifts mean at a changepoint.

    The error rate ε_t (used to generate shot outcomes each timestep) evolves
    according to an AR(1) model.  The changepoint shifts the long-run mean from
    mu_pre to mu_post, but the AR(1) smoothing makes the transition gradual.

    Key parameter:
        phi (φ): autocorrelation.
            φ = 0.0 → independent noise (Markovian, like DepolarizingSimulator).
            φ = 0.8 → strongly correlated — errors take many steps to drift to
                       the new mean after the changepoint.
            φ must satisfy |φ| < 1 for the process to be stationary.
    """

    def __init__(
        self,
        mu_pre: float = 0.01,
        mu_post: float = 0.05,
        phi: float = 0.8,
        sigma: float = 0.005,
        changepoint: int = 50,
        epsilon_min: float = 0.0,
        epsilon_max: float = 1.0,
    ):
        """
        Initialise the correlated noise simulator.

        Args:
            mu_pre      : Long-run mean error rate before the changepoint.
            mu_post     : Long-run mean error rate after the changepoint.
            phi         : AR(1) autocorrelation coefficient.  Must be in (-1, 1).
                          Typical values: 0.0 (Markovian), 0.5 (moderate
                          correlation), 0.8 (strong correlation).
            sigma       : Standard deviation of the per-step innovation η_t.
                          Controls how much ε_t fluctuates around its mean.
                          A good default: ~10–20% of (mu_post - mu_pre).
            changepoint : Timestep when the long-run mean shifts.
            epsilon_min : Lower clip for the error rate (default 0).
            epsilon_max : Upper clip for the error rate (default 1).

        Example:
            >>> sim = CorrelatedNoiseSimulator(mu_pre=0.01, mu_post=0.05, phi=0.8)
            >>> data = sim.generate_data(n_timesteps=100, n_shots=1000)
            >>> data.shape
            (100, 1000)
        """
        super().__init__(changepoint)
        if not (-1.0 < phi < 1.0):
            raise ValueError(f"phi must be in (-1, 1) for stationarity; got {phi}.")

        self.mu_pre = mu_pre
        self.mu_post = mu_post
        self.phi = phi
        self.sigma = sigma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max

    def generate_data(self, n_timesteps: int, n_shots: int) -> np.ndarray:
        """
        Generate measurement data from a circuit with AR(1) correlated noise.

        The error rate ε_t evolves as an AR(1) process around a mean that
        shifts at the changepoint.  At each timestep, n_shots binary outcomes
        are drawn from Binomial(1, ε_t).

        Args:
            n_timesteps : Number of timesteps to simulate.
            n_shots     : Number of shots per timestep.

        Returns:
            2D integer array of shape (n_timesteps, n_shots) with 0/1 outcomes.
            Unlike DepolarizingSimulator, the bit-flip rate changes smoothly
            rather than jumping sharply at the changepoint.

        Example:
            >>> sim = CorrelatedNoiseSimulator(mu_pre=0.01, mu_post=0.05, phi=0.8)
            >>> data = sim.generate_data(100, 1000)
            >>> # The bit-flip rate drifts gradually after the changepoint
        """
        data = np.zeros((n_timesteps, n_shots), dtype=int)

        # Initialise ε_0 at the pre-change mean (the process starts in equilibrium).
        epsilon = self.mu_pre

        for t in range(n_timesteps):
            # Long-run mean for this timestep (shifts at changepoint).
            mu_t = self.mu_pre if t < self.changepoint else self.mu_post

            # AR(1) update: drift toward mu_t, then add innovation noise.
            innovation = self.sigma * np.random.randn()
            epsilon = mu_t + self.phi * (epsilon - mu_t) + innovation

            # Clip to a valid probability range.
            epsilon = float(np.clip(epsilon, self.epsilon_min, self.epsilon_max))

            # Generate n_shots binary outcomes with this error probability.
            data[t, :] = np.random.binomial(1, epsilon, n_shots)

        return data

    def generate_error_rates(self, n_timesteps: int) -> np.ndarray:
        """
        Generate only the latent AR(1) error rate trajectory (no shot noise).

        Useful for visualising the underlying process without measurement noise.

        Args:
            n_timesteps : Number of timesteps.

        Returns:
            1D float array of shape (n_timesteps,) with the true ε_t values.

        Example:
            >>> sim = CorrelatedNoiseSimulator(mu_pre=0.01, mu_post=0.05, phi=0.8)
            >>> rates = sim.generate_error_rates(100)
            >>> rates.shape
            (100,)
        """
        rates = np.zeros(n_timesteps)
        epsilon = self.mu_pre

        for t in range(n_timesteps):
            mu_t = self.mu_pre if t < self.changepoint else self.mu_post
            innovation = self.sigma * np.random.randn()
            epsilon = mu_t + self.phi * (epsilon - mu_t) + innovation
            epsilon = float(np.clip(epsilon, self.epsilon_min, self.epsilon_max))
            rates[t] = epsilon

        return rates
