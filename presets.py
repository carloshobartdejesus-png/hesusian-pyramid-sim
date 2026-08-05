"""
presets.py
==========
Pre-configured parameter sets for the Hesusian Pyramid simulator.

The first entry is always ``Custom (Manual Input)`` — the conventional
placeholder that leaves every slider at the user's chosen values.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Preset:
    """A named bundle of simulation parameters."""
    name: str
    description: str
    B: float
    S: float
    E: float
    R: float
    F: float
    eta: float
    lam: float
    iterations: int


# --------------------------------------------------------------------------- #
# Preset catalogue
# --------------------------------------------------------------------------- #
PRESETS: list[Preset] = [
    Preset(
        name="Custom (Manual Input)",
        description="User-controlled configuration with no preset applied.",
        B=50, S=50, E=50, R=50, F=50, eta=0.5, lam=0.3, iterations=30,
    ),
    Preset(
        name="Novice Learner",
        description="Low capacity, minimal stimuli, little experience — early Degree I–II.",
        B=25, S=20, E=15, R=20, F=10, eta=0.1, lam=0.6, iterations=30,
    ),
    Preset(
        name="Dedicated Student",
        description="Moderate capacity with rich stimuli and growing reflection.",
        B=55, S=60, E=45, R=55, F=45, eta=0.25, lam=0.4, iterations=40,
    ),
    Preset(
        name="Reflective Practitioner",
        description="High reflection, strong reasoning — Degree IV–V emergence.",
        B=65, S=70, E=75, R=70, F=80, eta=0.4, lam=0.35, iterations=40,
    ),
    Preset(
        name="Generative Innovator",
        description="Balanced high capacity enabling original synthesis (Degree V).",
        B=80, S=75, E=80, R=85, F=85, eta=0.55, lam=0.3, iterations=50,
    ),
    Preset(
        name="Environmental Mentor",
        description="Very high eta — strong ECC feedback, Degree VI dominance.",
        B=85, S=80, E=85, R=80, F=80, eta=0.9, lam=0.2, iterations=50,
    ),
    Preset(
        name="Stagnant Environment",
        description="High decay suppresses ECC — conceptual plateau.",
        B=50, S=40, E=40, R=40, F=30, eta=0.15, lam=0.9, iterations=40,
    ),
    Preset(
        name="Information Flood",
        description="Overwhelming stimuli with low reflection — breadth without depth.",
        B=45, S=95, E=30, R=50, F=20, eta=0.3, lam=0.25, iterations=40,
    ),
]


def preset_by_name(name: str) -> Preset | None:
    """Look up a preset by name, returning ``None`` if not found."""
    for p in PRESETS:
        if p.name == name:
            return p
    return None


def preset_names() -> list[str]:
    """Return ordered list of preset names (Custom first)."""
    return [p.name for p in PRESETS]