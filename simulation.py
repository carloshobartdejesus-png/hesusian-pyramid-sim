"""
simulation.py
=============
Recursive simulation engine for the Hesusian Pyramid / ECC model.

Wraps the pure equations in :mod:`equations` into an iterable time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from equations import (
    HPState,
    conceptual_development,
    effective_behavior,
    environmental_update,
    stimulus_generation,
    degree_activations,
)
from utils import SCALE_MAX


@dataclass
class SimulationResult:
    """Container holding the full trajectory of the simulation."""

    states: list[HPState]
    activations: list[list[float]]
    C_series: np.ndarray
    A_series: np.ndarray
    I_series: np.ndarray
    S_series: np.ndarray


def run_simulation(
    B: float,
    S0: float,
    E: float,
    R: float,
    F: float,
    eta: float,
    lam: float,
    iterations: int,
) -> SimulationResult:
    """Run the recursive HP/ECC simulation.

    Parameters
    ----------
    B : float
        Biological Capacity (0..100).
    S0 : float
        Initial Informational Stimuli (0..100).
    E, R, F : float
        Experience, Reasoning, Reflection (0..100).
    eta : float
        Hesusian Influence Coefficient (0..1).
    lam : float
        Environmental Decay Constant (0..1).
    iterations : int
        Number of recursive iterations.

    Returns
    -------
    SimulationResult
        Time series of states, activations, and individual variable arrays.
    """
    states: list[HPState] = []
    activations: list[list[float]] = []

    # Initial environment baseline — seeded by the initial stimulus so the
    # cycle has something to decay from.
    I = float(stimulus_generation(S0 * 0.5))
    S = float(S0)

    for t in range(int(iterations)):
        C = conceptual_development(B, S, E, R, F)
        A_eff = effective_behavior(C, eta)
        state = HPState(
            t=t, B=B, S=S, E=E, R=R, F=F, eta=eta, C=C, A=A_eff, I=I
        )
        states.append(state)
        activations.append(degree_activations(state))

        # --- ECC recursion -------------------------------------------------
        I = environmental_update(I, A_eff, lam)       # I_{t+1}
        S = stimulus_generation(I)                     # S_{t+1}

    n = len(states)
    return SimulationResult(
        states=states,
        activations=activations,
        C_series=np.array([s.C for s in states], dtype=float),
        A_series=np.array([s.A for s in states], dtype=float),
        I_series=np.array([s.I for s in states], dtype=float),
        S_series=np.array([s.S for s in states], dtype=float),
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_statistics(result: SimulationResult) -> dict:
    """Compute summary statistics for the statistics panel."""
    C = result.C_series
    A = result.A_series  # effective behavior (eta * tanh(C) * SCALE_MAX)
    I = result.I_series
    S = result.S_series
    n = len(C)

    growth_rate = (
        float((C[-1] - C[0]) / C[0]) if C[0] != 0 else float("nan")
    )
    env_growth = (
        float((I[-1] - I[0]) / max(abs(I[0]), 1e-9))
        if n else float("nan")
    )

    return {
        "max_C": float(np.max(C)) if n else 0.0,
        "avg_C": float(np.mean(C)) if n else 0.0,
        "final_C": float(C[-1]) if n else 0.0,
        "growth_rate": growth_rate,
        "env_growth": env_growth,
        "avg_behavior": float(np.mean(A)) if n else 0.0,
        "final_env": float(I[-1]) if n else 0.0,
        "final_stimulus": float(S[-1]) if n else 0.0,
    }