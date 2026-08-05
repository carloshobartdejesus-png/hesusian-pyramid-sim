"""
equations.py
============
Mathematical core of the Hesusian Pyramid (HP) and the Environmental
Conceptualization Cycle (ECC).

Every function is pure (no side effects) and documented so the equations
can be cross-referenced with the theoretical framework documentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from utils import SCALE_MAX, sigmoid, to_unit

# --------------------------------------------------------------------------- #
# Weights for the conceptual-development composite
# --------------------------------------------------------------------------- #
W_S: float = 0.30  # Informational Stimuli
W_E: float = 0.25  # Experience
W_R: float = 0.25  # Reasoning
W_F: float = 0.20  # Reflection
assert abs((W_S + W_E + W_R + W_F) - 1.0) < 1e-9, "Weights must sum to 1."


@dataclass
class HPState:
    """Snapshot of every dynamic variable at a single iteration.

    Attributes
    ----------
    t : int
        Iteration index.
    B, S, E, R, F : float
        Input variables on the canonical 0..100 scale.
    eta : float
        Hesusian Influence Coefficient (0..1).
    C : float
        Conceptual Development.
    A : float
        Behavior output.
    I : float
        Environmental Information.
    """

    t: int
    B: float
    S: float
    E: float
    R: float
    F: float
    eta: float
    C: float
    A: float
    I: float


# --------------------------------------------------------------------------- #
# Core equations
# --------------------------------------------------------------------------- #
def conceptual_development(B: float, S: float, E: float, R: float, F: float) -> float:
    """Compute Conceptual Development *C*.

    .. math::

        C_t = B\\,(0.30\\,S_t + 0.25\\,E_t + 0.25\\,R_t + 0.20\\,F_t)

    All inputs are on the 0..100 canonical scale; the weighted sum is
    normalized back to that scale before modulation by *B*.

    Parameters
    ----------
    B, S, E, R, F : float
        Biological Capacity, Stimuli, Experience, Reasoning, Reflection.

    Returns
    -------
    float
        Conceptual Development on the 0..100 scale.
    """
    composite = (
        W_S * to_unit(S)
        + W_E * to_unit(E)
        + W_R * to_unit(R)
        + W_F * to_unit(F)
    )  # now in 0..1
    return float(to_unit(B) * composite)  # 0..100


def behavior(C: float) -> float:
    """Behavioral output via hyperbolic tangent.

    .. math::

        A_t = \\tanh(C_t)

    The argument is normalized to [0, 1] so tanh saturates gracefully.

    Returns
    -------
    float
        Behavior in (0, 1).
    """
    return float(math.tanh(to_unit(C)))


def environmental_update(I: float, A: float, lam: float) -> float:
    """Recursive environmental-information update.

    .. math::

        I_{t+1} = (1 - \\lambda)\\,I_t + \\eta\\,A_t

    Here *A* already incorporates the Hesusian coefficient (see
    :func:`effective_behavior`), so this function applies the decay +
    injection purely.

    Parameters
    ----------
    I : float
        Current environmental information.
    A : float
        Effective behavior contribution (eta * tanh(C), scaled).
    lam : float
        Environmental decay constant, 0 < lambda < 1.

    Returns
    -------
    float
        Updated environment value.
    """
    return float((1.0 - lam) * I + A)


def effective_behavior(C: float, eta: float) -> float:
    """Behavior scaled by the Hesusian Influence Coefficient.

    The ECC injects ``eta * tanh(C)`` into the environment.  We scale it
    to the canonical 0..100 scale so it composes cleanly with *I*.

    Returns
    -------
    float
        Scaled behavior contribution on 0..100.
    """
    return float(eta * behavior(C) * SCALE_MAX)


def stimulus_generation(I_next: float) -> float:
    """Generate next-iteration stimulus from environment.

    .. math::

        S_{t+1} = \\log(1 + I_{t+1})

    The raw log is normalized back to the 0..100 slider scale by dividing
    by the theoretical ceiling ``log(1 + SCALE_MAX)``.

    Returns
    -------
    float
        Stimulus on 0..100 scale.
    """
    raw = math.log1p(max(0.0, I_next))
    ceiling = math.log1p(SCALE_MAX)
    return float(raw / ceiling * SCALE_MAX) if ceiling > 0 else 0.0


# --------------------------------------------------------------------------- #
# Degree-of-Conceptualization activation
# --------------------------------------------------------------------------- #
def degree_activations(state: HPState) -> list[float]:
    """Compute smooth activation levels for the six Hesusian Degrees.

    Each degree uses a *smooth* sigmoid interpolation (never a binary
    threshold).  Lower degrees activate first; higher degrees emerge
    progressively as conceptual development (and related variables) grow.

    Dependencies (per the framework specification)
    -----------------------------------------------
    * Degree I  — Informational Recognition        : S
    * Degree II — Associative Conceptualization    : S, E
    * Degree III— Structured Understanding         : C
    * Degree IV — Reflective Conceptualization      : F
    * Degree V  — Generative Conceptualization      : C, R, F
    * Degree VI — Environmental Conceptualization   : C, eta

    Returns
    -------
    list[float]
        Six activation values in [0, 1], Degree I .. Degree VI.
    """
    S_u = to_unit(state.S)
    E_u = to_unit(state.E)
    R_u = to_unit(state.R)
    F_u = to_unit(state.F)
    C_u = to_unit(state.C)
    eta = state.eta

    # Degree I — primarily Stimuli
    d1 = sigmoid(state.S, k=0.06, x0=15.0)

    # Degree II — Stimuli + Experience (geometric mean for joint dependency)
    d2 = sigmoid(math.sqrt(S_u * E_u) * SCALE_MAX, k=0.06, x0=25.0)

    # Degree III — Conceptual Development
    d3 = sigmoid(state.C, k=0.06, x0=35.0)

    # Degree IV — strongly Reflection
    d4 = sigmoid(state.F, k=0.07, x0=45.0)

    # Degree V — C, R, F (weighted geometric composite)
    v5 = (C_u ** 0.4) * (R_u ** 0.3) * (F_u ** 0.3)
    d5 = sigmoid(v5 * SCALE_MAX, k=0.06, x0=55.0)

    # Degree VI — C and eta
    v6 = (C_u ** 0.6) * (eta ** 0.4)
    d6 = sigmoid(v6 * SCALE_MAX, k=0.06, x0=65.0)

    return [d1, d2, d3, d4, d5, d6]


# --------------------------------------------------------------------------- #
# Metadata for UI / LaTeX rendering
# --------------------------------------------------------------------------- #
DEGREE_INFO: list[dict] = [
    {"roman": "I",   "name": "Informational Recognition",     "dep": "S"},
    {"roman": "II",  "name": "Associative Conceptualization", "dep": "S, E"},
    {"roman": "III", "name": "Structured Understanding",      "dep": "C"},
    {"roman": "IV",  "name": "Reflective Conceptualization",   "dep": "F"},
    {"roman": "V",   "name": "Generative Conceptualization",   "dep": "C, R, F"},
    {"roman": "VI",  "name": "Environmental Conceptualization", "dep": "C, η"},
]


def latex_forms(state: HPState, lam: float) -> list[dict]:
    """Return symbolic + numerical LaTeX strings for every equation.

    Each dict has keys: ``label``, ``symbolic``, ``numeric``, ``result``.
    """
    B, S, E, R, F = state.B, state.S, state.E, state.R, state.F
    eta = state.eta
    A = behavior(state.C)
    eff = effective_behavior(state.C, eta)
    I_next = environmental_update(state.I, eff, lam)
    S_next = stimulus_generation(I_next)

    return [
        {
            "label": "Conceptual Development",
            "symbolic": r"C_t = B\,(0.30\,S_t + 0.25\,E_t + 0.25\,R_t + 0.20\,F_t)",
            "numeric": (
                f"C_{{{state.t}}} = {B:.2f}\\,(0.30\\times{S:.2f} + "
                f"0.25\\times{E:.2f} + 0.25\\times{R:.2f} + 0.20\\times{F:.2f})"
            ),
            "result": f"C_{{{state.t}}} = {state.C:.4f}",
        },
        {
            "label": "Behavior",
            "symbolic": r"A_t = \tanh(C_t)",
            "numeric": f"A_{{{state.t}}} = \\tanh({state.C:.4f})",
            "result": f"A_{{{state.t}}} = {A:.6f}",
        },
        {
            "label": "Effective Behavior (ECC injection)",
            "symbolic": r"A^{\text{eff}}_t = \eta\,A_t",
            "numeric": f"A^{{\\text{{eff}}}}_{{{state.t}}} = {eta:.4f}\\times{A:.6f}\\times{SCALE_MAX:.0f}",
            "result": f"A^{{\\text{{eff}}}}_{{{state.t}}} = {eff:.4f}",
        },
        {
            "label": "Environmental Update",
            "symbolic": r"I_{t+1} = (1-\lambda)\,I_t + \eta\,A_t",
            "numeric": (
                f"I_{{{state.t+1}}} = (1-{lam:.4f})\\times{state.I:.4f} + {eff:.4f}"
            ),
            "result": f"I_{{{state.t+1}}} = {I_next:.4f}",
        },
        {
            "label": "Stimulus Generation",
            "symbolic": r"S_{t+1} = \log(1 + I_{t+1})",
            "numeric": f"S_{{{state.t+1}}} = \\log(1 + {I_next:.4f})",
            "result": f"S_{{{state.t+1}}} = {S_next:.4f}",
        },
    ]