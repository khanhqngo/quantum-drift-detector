"""Base classes for quantum circuit simulators."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseSimulator(ABC):
    """Abstract base class for quantum circuit simulators that generate measurement data."""

    def __init__(self, changepoint: int):
        """
        Initialize the simulator.

        Args:
            changepoint: The timestep at which the noise regime changes.
        """
        self.changepoint = changepoint

    @abstractmethod
    def generate_data(self, n_timesteps: int, n_shots: int) -> np.ndarray:
        """
        Generate measurement data over time.

        Args:
            n_timesteps: Number of time steps to simulate.
            n_shots: Number of measurement shots per timestep.

        Returns:
            A 2D array of shape (n_timesteps, n_shots) containing 0/1 measurement outcomes.
        """
        pass