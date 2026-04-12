"""Simulator for depolarizing noise that shifts at a changepoint."""

import numpy as np
from typing import Optional
from .base import BaseSimulator


class DepolarizingSimulator(BaseSimulator):
    """
    Simulates a quantum circuit with depolarizing noise that changes at a specific timestep.

    Depolarizing noise means that with some probability, the qubit gets randomized
    (like flipping a coin that was supposed to land heads). This changes the fraction
    of 1s we measure when we prepare |0⟩ and measure it.
    """

    def __init__(self,
                 error_rate_pre: float = 0.01,
                 error_rate_post: float = 0.05,
                 changepoint: int = 50):
        """
        Initialize the depolarizing noise simulator.

        Args:
            error_rate_pre: Probability of bit flip before the changepoint.
            error_rate_post: Probability of bit flip after the changepoint.
            changepoint: Timestep when the error rate changes.
        """
        super().__init__(changepoint)
        self.error_rate_pre = error_rate_pre
        self.error_rate_post = error_rate_post

    def generate_data(self, n_timesteps: int, n_shots: int) -> np.ndarray:
        """
        Generate measurement data from a simple quantum circuit with changing noise.

        The circuit: prepare |0⟩, apply identity (no gate), measure.
        Noise: depolarizing channel that flips the bit with probability error_rate.

        Args:
            n_timesteps: Number of time steps to simulate.
            n_shots: Number of measurement shots per timestep.

        Returns:
            A 2D array of shape (n_timesteps, n_shots) with 0/1 outcomes.
        """
        data = np.zeros((n_timesteps, n_shots), dtype=int)

        for t in range(n_timesteps):
            # Determine error rate for this timestep
            error_rate = self.error_rate_pre if t < self.changepoint else self.error_rate_post

            # Generate shots: 0 with probability (1 - error_rate), 1 with probability error_rate
            # This simulates preparing |0⟩ and having it flip to |1⟩ with probability error_rate
            outcomes = np.random.binomial(1, error_rate, n_shots)
            data[t, :] = outcomes

        return data