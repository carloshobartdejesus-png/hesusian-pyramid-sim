"""
visualization.py
================
All Plotly charts and the animated Hesusian Pyramid rendering.

Keeps visualization logic fully separated from simulation logic and UI.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from equations import HPState, DEGREE_INFO, latex_forms
from simulation import SimulationResult, compute_statistics
from utils import SCALE_MAX

# --------------------------------------------------------------------------- #
# Shared layout constants
# --------------------------------------------------------------------------- #
_CHART_BG = "rgba(0,0,0,0)"
_GRID_COLOR = "rgba(0,0,0,0.08)"
_FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color="#1f2937")
_MARGINS = dict(l=50, r=30, t=50, b=45)


def _base_layout(title: str, **kw) -> go.Layout:
    """Return a clean journal-style Plotly layout."""
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        font=_FONT,
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_CHART_BG,
        margin=_MARGINS,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False),
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1. Animated Hesusian Pyramid
# --------------------------------------------------------------------------- #
def pyramid_figure(activations: list[float]) -> go.Figure:
    """Render the six-degree Hesusian Pyramid as a heat-map stacked chart.

    Parameters
    ----------
    activations : list[float]
        Six activation values [0,1] for Degrees I..VI.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    n_levels = 6
    # Trapezoid widths: bottom wide, top narrow
    max_w = 1.0
    min_w = 0.35
    widths = [max_w - (max_w - min_w) * (i / (n_levels - 1)) for i in range(n_levels)]
    # Degrees rendered bottom (I) to top (VI)
    for i in range(n_levels - 1, -1, -1):
        deg = DEGREE_INFO[i]
        act = activations[i]
        y_base = (n_levels - 1 - i)
        w = widths[i]
        x_center = 0.5

        # Heat color from cool (low) to warm (high)
        r = int(30 + act * 200)
        g = int(60 + act * 120)
        b = int(120 - act * 80)
        color = f"rgb({r},{g},{b})"
        opacity = 0.35 + 0.65 * act

        # Trapezoid corners
        x0 = x_center - w / 2
        x1 = x_center + w / 2
        # Account for trapezoid shape (slightly narrower top of each block)
        inset = 0.04
        xs = [x0, x1, x1 - inset, x0 + inset, x0]
        ys = [y_base, y_base, y_base + 1, y_base + 1, y_base]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            fillcolor=color,
            line=dict(color="white", width=2),
            opacity=opacity,
            mode="lines",
            showlegend=False,
            hovertemplate=(
                f"<b>Degree {deg['roman']} — {deg['name']}</b><br>"
                f"Activation: {act:.1%}<br>"
                f"Depends on: {deg['dep']}<extra></extra>"
            ),
            text=f"Degree {deg['roman']}",
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(
            visible=False,
            range=[-0.2, n_levels + 0.2],
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_CHART_BG,
        height=420,
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Recursive Dynamics Graph
# --------------------------------------------------------------------------- #
def recursive_dynamics_figure(result: SimulationResult) -> go.Figure:
    """Plot Iteration vs Conceptual Development (C)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(result.C_series))),
        y=result.C_series,
        mode="lines+markers",
        name="C — Conceptual Development",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=5, color="#2563eb"),
    ))
    fig.update_layout(_base_layout("Recursive Dynamics — Conceptual Development"))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="C", range=[0, SCALE_MAX * 1.05])
    return fig


# --------------------------------------------------------------------------- #
# 3. Environmental Conceptualization Cycle Graph
# --------------------------------------------------------------------------- #
def ecc_figure(result: SimulationResult) -> go.Figure:
    """Plot Environmental Information (I) and Stimuli (S) together."""
    fig = go.Figure()
    x = list(range(len(result.I_series)))
    fig.add_trace(go.Scatter(
        x=x, y=result.I_series,
        mode="lines+markers", name="I — Environmental Information",
        line=dict(color="#0d9488", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=result.S_series,
        mode="lines+markers", name="S — Informational Stimuli",
        line=dict(color="#d97706", width=2.5, dash="dot"),
    ))
    fig.update_layout(_base_layout("Environmental Conceptualization Cycle (ECC)"))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Value")
    return fig


# --------------------------------------------------------------------------- #
# 4. Multi-Line Dynamics Graph
# --------------------------------------------------------------------------- #
def multiline_dynamics_figure(result: SimulationResult) -> go.Figure:
    """Plot C, Behavior, Environment, and Stimuli on one graph."""
    x = list(range(len(result.C_series)))
    traces = [
        ("Conceptual Development (C)", result.C_series, "#2563eb"),
        ("Behavior (A)", result.A_series, "#dc2626"),
        ("Environment (I)", result.I_series, "#0d9488"),
        ("Stimuli (S)", result.S_series, "#d97706"),
    ]
    fig = go.Figure()
    for name, y, color in traces:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig.update_layout(_base_layout("Multi-Line Dynamics"))
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Value")
    return fig


# --------------------------------------------------------------------------- #
# 5. Radar Chart
# --------------------------------------------------------------------------- #
def radar_figure(state: HPState) -> go.Figure:
    """Radar chart of current input-variable values."""
    categories = [
        "Biological Capacity",
        "Informational Stimuli",
        "Experience",
        "Reasoning",
        "Reflection",
        "Hesusian η",
    ]
    values = [state.B, state.S, state.E, state.R, state.F, state.eta * SCALE_MAX]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Current State",
        line=dict(color="#2563eb", width=2),
        fillcolor="rgba(37,99,235,0.18)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, SCALE_MAX], tickfont=dict(size=10)),
            bgcolor=_CHART_BG,
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    return fig


# --------------------------------------------------------------------------- #
# 8. Phase Portrait
# --------------------------------------------------------------------------- #
def phase_portrait_figure(result: SimulationResult) -> go.Figure:
    """Phase portrait: Environment (I) vs Conceptual Development (C)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.I_series,
        y=result.C_series,
        mode="lines+markers",
        name="Phase Trajectory",
        line=dict(color="#7c3aed", width=2),
        marker=dict(size=5, color="#7c3aed"),
    ))
    # Direction arrow start -> end
    if len(result.I_series) >= 2:
        fig.add_trace(go.Scatter(
            x=[result.I_series[0]],
            y=[result.C_series[0]],
            mode="markers",
            marker=dict(size=14, color="#16a34a", symbol="circle"),
            name="Start",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[result.I_series[-1]],
            y=[result.C_series[-1]],
            mode="markers",
            marker=dict(size=14, color="#dc2626", symbol="x"),
            name="End",
            showlegend=True,
        ))
    fig.update_layout(_base_layout("Phase Portrait — Environment vs Conceptual Development"))
    fig.update_xaxes(title_text="Environment (I)")
    fig.update_yaxes(title_text="Conceptual Development (C)")
    return fig