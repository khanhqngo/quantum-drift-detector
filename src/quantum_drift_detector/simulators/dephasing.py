"""
Dephasing noise simulator (Ramsey experiment model).

Plain English: A qubit has two distinct properties that noise can disturb:
  (1) WHICH STATE it is in — 0 or 1.  Depolarizing noise (Phase 1) corrupts this.
  (2) its PHASE — a subtle quantum property invisible in a single measurement,
      but revealed when the qubit is put into superposition and "probed."
      Dephasing noise corrupts (2) while leaving (1) untouched.

An analogy: imagine two clocks that are supposed to tick in perfect sync.
Dephasing is like someone randomly nudging one clock forward or backward —
the individual tick rate (which state) is unchanged, but the synchrony
(phase) is lost.

To DETECT dephasing, physicists use the **Ramsey experiment**:
  Step 1. Prepare the qubit in |0⟩ (clock pointing "up").
  Step 2. Apply a Hadamard gate (H): puts the qubit into a superposition
          |+⟩ = (|0⟩ + |1⟩)/√2 — like balancing the coin perfectly on its edge.
  Step 3. Wait (during which dephasing may randomise the phase).
  Step 4. Apply H again — this converts phase information back into a measurable
          0/1 outcome.
  Step 5. Measure.

Without any noise, step 4 perfectly undoes step 2 → always measure 0.
With dephasing (rate γ per timestep), the phase is partially scrambled, and:

    P(measure 1) = (1 − e^{−γ}) / 2

This is bounded in [0, 0.5].  Unlike depolarizing (which can push P(1) up to 1),
dephasing can only make the qubit "half-random" at worst.  This different
signature is physically meaningful and useful for characterising noise types.

Dephasing is particularly relevant to:
  - Superconducting qubits (characterised by T1 and T2 times)
  - Neutral-atom platforms (Infleqtion, QuEra) where T2 ≪ T1
  - Any platform where gate noise scrambles phase faster than it flips states
"""

import numpy as np
from typing import Optional
from .base import BaseSimulator


class DephasingSimulator(BaseSimulator):
    """
    Simulates a Ramsey experiment with dephasing noise that shifts at a changepoint.

    The dephasing rate γ changes from gamma_pre to gamma_post at the changepoint.
    Measurement outcomes follow:
        P(1 | γ) = (1 − exp(−γ)) / 2

    The bit-flip rate is bounded in [0, 0.5], which distinguishes dephasing from
    depolarizing noise (where P(1) ∈ [0, 1]).
    """

    def __init__(
        self,
        gamma_pre: float = 0.2,
        gamma_post: float = 1.5,
        changepoint: int = 50,
    ):
        """
        Initialise the dephasing noise simulator.

        Args:
            gamma_pre   : Dephasing rate before the changepoint.
                          γ = 0 → no dephasing (always measure 0).
                          γ = ∞ → maximal dephasing (P(1) → 0.5).
                          Physically: γ ≈ Δt / T2 where T2 is the coherence time.
            gamma_post  : Dephasing rate after the changepoint.
            changepoint : Timestep when the dephasing rate changes.

        Example:
            >>> sim = DephasingSimulator(gamma_pre=0.2, gamma_post=1.5, changepoint=50)
            >>> data = sim.generate_data(n_timesteps=100, n_shots=1000)
            >>> data.shape
            (100, 1000)
        """
        super().__init__(changepoint)
        self.gamma_pre = gamma_pre
        self.gamma_post = gamma_post

        # Pre-compute the bit-flip probabilities from the Ramsey formula.
        # P(1) = (1 - exp(-γ)) / 2
        self.p_pre  = (1.0 - np.exp(-gamma_pre))  / 2.0
        self.p_post = (1.0 - np.exp(-gamma_post)) / 2.0

    def generate_data(self, n_timesteps: int, n_shots: int) -> np.ndarray:
        """
        Generate Ramsey experiment measurements with shifting dephasing noise.

        At each timestep the Ramsey circuit is run n_shots times.  Before the
        changepoint the dephasing rate is gamma_pre; after, it is gamma_post.

        Args:
            n_timesteps : Number of timesteps to simulate.
            n_shots     : Number of shots (circuit repetitions) per timestep.

        Returns:
            2D integer array of shape (n_timesteps, n_shots) with 0/1 outcomes.
            Values near 0 indicate low dephasing; values approaching 0.5 indicate
            strong dephasing.

        Example:
            >>> sim = DephasingSimulator(gamma_pre=0.2, gamma_post=1.5, changepoint=50)
            >>> data = sim.generate_data(100, 1000)
            >>> pre_rate  = data[:50].mean()   # should be ≈ p_pre  ≈ 0.09
            >>> post_rate = data[50:].mean()   # should be ≈ p_post ≈ 0.39
        """
        data = np.zeros((n_timesteps, n_shots), dtype=int)

        for t in range(n_timesteps):
            # Choose bit-flip probability based on which side of the changepoint we are.
            p = self.p_pre if t < self.changepoint else self.p_post

            # Simulate n_shots Ramsey experiments.
            # Each shot: measure 1 with probability p, measure 0 with probability 1-p.
            data[t, :] = np.random.binomial(1, p, n_shots)

        return data
