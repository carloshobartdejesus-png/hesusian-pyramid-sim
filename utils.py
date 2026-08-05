"""
utils.py
========
Shared utility functions for the Hesusian Pyramid (HP) and Environmental
Conceptualization Cycle (ECC) simulator.

Contents
--------
* SCALE_MAX : float
    Canonical scale ceiling for all model variables (0 .. 100).
* sigmoid(x, k, x0) : float
    Smooth logistic interpolation used for degree activation.
* normalize(value, lo, hi) : float
    Linear rescale of *value* from [lo, hi] to [0, SCALE_MAX].
* clamp(x, lo, hi) : float
    Bounding helper.
* smoothstep(edge0, edge1, x) : float
    Hermite-smooth interpolation between two thresholds.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SCALE_MAX: float = 100.0
"""Canonical ceiling for every slider / model variable (0 .. 100)."""


# --------------------------------------------------------------------------- #
# Interpolation primitives
# --------------------------------------------------------------------------- #
def sigmoid(x: float, k: float = 0.1, x0: float = 50.0) -> float:
    """Numerically stable logistic activation.

    Parameters
    ----------
    x : float
        Input value (same scale as *SCALE_MAX*).
    k : float, default 0.1
        Steepness of the transition.
    x0 : float, default 50.0
        Midpoint of the activation curve.

    Returns
    -------
    float
        Activation in the open interval (0, 1).
    """
    # Branch-safe stable sigmoid
    if x >= x0:
        z = np.exp(-k * (x - x0))
        return float(1.0 / (1.0 + z))
    z = np.exp(k * (x - x0))
    return float(z / (1.0 + z))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite-smooth clamped interpolation between two edges.

    Parameters
    ----------
    edge0, edge1 : float
        Lower and upper thresholds.
    x : float
        Value to interpolate.

    Returns
    -------
    float
        Smooth value in [0, 1].
    """
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def normalize(value: float, lo: float = 0.0, hi: float = SCALE_MAX) -> float:
    """Linearly rescale *value* from [lo, hi] to [0, SCALE_MAX].

    Parameters
    ----------
    value : float
        Raw input.
    lo, hi : float
        Source range bounds.

    Returns
    -------
    float
        Value expressed on the canonical [0, SCALE_MAX] scale.
    """
    if hi == lo:
        return 0.0
    return float(clamp((value - lo) / (hi - lo) * SCALE_MAX, 0.0, SCALE_MAX))


def clamp(x: float, lo: float, hi: float) -> float:
    """Bound *x* to [lo, hi]."""
    return float(max(lo, min(hi, x)))


def to_unit(value: float) -> float:
    """Convert a canonical-scale value to [0, 1]."""
    return float(clamp(value / SCALE_MAX, 0.0, 1.0))