"""
Hesusian Pyramid (HP) & Environmental Cycle of Conceptualization (ECC)
A discrete-time state-space feedback simulator.

Run:
    pip install streamlit plotly networkx numpy pandas
    streamlit run app.py
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Hesusian Pyramid | ECC Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0e1117",
    "panel": "#161a23",
    "grid": "#262c38",
    "text": "#e6e8ee",
    "muted": "#8b93a7",
    "accent": "#4c8dff",
    "edge": "#3d4657",
}

TIERS = [
    ("I", "Perceptual Conceptualization", 0, 15, "#f7d154"),
    ("II", "Associative Conceptualization", 16, 35, "#f5b942"),
    ("III", "Abstract Conceptualization", 36, 55, "#f09232"),
    ("IV", "Reflective Conceptualization", 56, 75, "#e87028"),
    ("V", "Generative Conceptualization", 76, 90, "#e04f24"),
    ("VI", "Environmental Conceptualization", 91, 10**9, "#d3241f"),
]

TIER_QUESTIONS = {
    "I": "What am I experiencing?",
    "II": "How are these concepts related?",
    "III": "What general principle explains these relationships?",
    "IV": "Should these concepts be revised?",
    "V": "How can these concepts be transformed to produce new understanding?",
    "VI": "How can these concepts contribute to the conceptual development of others?",
}

NODE_ORDER = ["C_t", "A_t", "I_t1", "S_t1", "C_t1"]
NODE_LABELS = {
    "C_t": "C<sub>t</sub>",
    "A_t": "A<sub>t</sub>",
    "I_t1": "I<sub>t+1</sub>",
    "S_t1": "S<sub>t+1</sub>",
    "C_t1": "C<sub>t+1</sub>",
}
NODE_TITLES = {
    "C_t": "Conceptual State",
    "A_t": "Behavioral Expression",
    "I_t1": "Informational Environment",
    "S_t1": "Stimulus Input",
    "C_t1": "Updated State",
}
NODE_POS = {
    "C_t": (0.00, 0.00),
    "A_t": (0.95, 0.62),
    "I_t1": (1.55, -0.30),
    "S_t1": (0.75, -1.05),
    "C_t1": (-0.55, -0.82),
}
EDGES = [
    ("C_t", "A_t", "g(C<sub>t</sub>)"),
    ("A_t", "I_t1", "I<sub>t</sub> + &#951;A<sub>t</sub>"),
    ("I_t1", "S_t1", "h(I<sub>t+1</sub>)"),
    ("S_t1", "C_t1", "f(B, S, E, R, F)"),
    ("C_t1", "C_t", "t &#8592; t+1"),
]


# --------------------------------------------------------------------------- #
# Simulation core
# --------------------------------------------------------------------------- #

def tier_for(c: float):
    for roman, name, lo, hi, color in TIERS:
        if lo <= c <= hi:
            return roman, name, lo, hi, color
    return TIERS[0]


def step(state: dict, p: dict) -> dict:
    """One discrete-time transition t -> t+1."""
    C_t = float(state["C_t"])
    I_t = float(state["I_t"])

    A_t = C_t * p["action_tendency"]                      # A_t = g(C_t)
    I_next = I_t + (p["eta"] * A_t)                       # I_{t+1} = I_t + eta*A_t
    S_next = I_next * p["stimulus_clarity"]               # S_{t+1} = h(I_{t+1})
    delta = p["B"] * (S_next / 100.0) * (1.0 + (p["R_t"] + p["F_t"]) / 200.0)
    C_next = C_t + delta                                  # C_{t+1}

    return {
        "t": int(state["t"]) + 1,
        "C_t": float(np.clip(C_next, 0.0, 1e9)),
        "I_t": float(I_next),
        "A_t": float(A_t),
        "S_t": float(S_next),
        "delta": float(delta),
        "prev_C": C_t,
    }


def init_state(I0: float = 10.0, C0: float = 5.0) -> dict:
    return {"t": 0, "C_t": C0, "I_t": I0, "A_t": 0.0, "S_t": 0.0, "delta": 0.0, "prev_C": C0}


def blank_history(state: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: state[k] for k in ("t", "C_t", "I_t", "A_t", "S_t")}])


if "state" not in st.session_state:
    st.session_state.state = init_state()
    st.session_state.history = blank_history(st.session_state.state)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### System Parameters")
    st.caption("Structural constants of the conceptualization process.")

    B = st.slider("Biological Capacity (B)", 0, 100, 80, 1)
    E_t = st.slider("Experience (E\u209c)", 0, 100, 50, 1)
    R_t = st.slider("Reasoning (R\u209c)", 0, 100, 50, 1)
    F_t = st.slider("Reflection (F\u209c)", 0, 100, 50, 1)

    st.markdown("### Coupling Coefficients")
    eta = st.slider("Hesusian Influence Coefficient (\u03b7)", 0.0, 1.0, 0.5, 0.01)
    action_tendency = st.slider("Action Tendency", 0.0, 1.0, 0.35, 0.01)
    stimulus_clarity = st.slider("Stimulus Clarity", 0.0, 1.0, 0.40, 0.01)

    st.markdown("### Initial Conditions")
    C0 = st.number_input("C\u2080 (initial conceptual state)", 0.0, 500.0, 5.0, 1.0)
    I0 = st.number_input("I\u2080 (initial informational environment)", 0.0, 500.0, 10.0, 1.0)

    st.markdown("### Execution")
    c1, c2 = st.columns(2)
    run_1 = c1.button("Run 1 Step", use_container_width=True)
    run_50 = c2.button("Simulate 50 Steps", use_container_width=True)
    reset = st.button("Reset System", use_container_width=True)

params = {
    "B": B,
    "E_t": E_t,
    "R_t": R_t,
    "F_t": F_t,
    "eta": eta,
    "action_tendency": action_tendency,
    "stimulus_clarity": stimulus_clarity,
}

if reset:
    st.session_state.state = init_state(I0=I0, C0=C0)
    st.session_state.history = blank_history(st.session_state.state)

if run_1 or run_50:
    n = 50 if run_50 else 1
    rows = []
    s = st.session_state.state
    for _ in range(n):
        s = step(s, params)
        rows.append({k: s[k] for k in ("t", "C_t", "I_t", "A_t", "S_t")})
    st.session_state.state = s
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame(rows)], ignore_index=True
    )

state = st.session_state.state
hist = st.session_state.history


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def base_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(size=14, color=PALETTE["text"])) if title else None,
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, Segoe UI, system-ui, sans-serif", size=12),
        margin=dict(l=40, r=24, t=48 if title else 16, b=32),
    )
    return fig


def ecc_graph(state: dict, p: dict) -> go.Figure:
    G = nx.DiGraph()
    for n in NODE_ORDER:
        G.add_node(n, pos=NODE_POS[n])
    for a, b, lbl in EDGES:
        G.add_edge(a, b, label=lbl)

    values = {
        "C_t": state["prev_C"] if state["t"] > 0 else state["C_t"],
        "A_t": state["A_t"],
        "I_t1": state["I_t"],
        "S_t1": state["S_t"],
        "C_t1": state["C_t"],
    }
    vmax = max(1e-6, max(abs(v) for v in values.values()))

    # Flow magnitude per edge, normalised -> width + colour intensity
    flows = {
        ("C_t", "A_t"): abs(values["A_t"]),
        ("A_t", "I_t1"): abs(p["eta"] * values["A_t"]),
        ("I_t1", "S_t1"): abs(values["S_t1"]),
        ("S_t1", "C_t1"): abs(state["delta"]),
        ("C_t1", "C_t"): abs(values["C_t1"]),
    }
    fmax = max(1e-6, max(flows.values()))

    fig = go.Figure()

    def lerp_color(t: float) -> str:
        t = float(np.clip(t, 0.0, 1.0))
        c0 = (61, 70, 87)
        c1 = (76, 141, 255)
        rgb = tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
        return "rgb(%d,%d,%d)" % rgb

    for (a, b, lbl) in EDGES:
        x0, y0 = NODE_POS[a]
        x1, y1 = NODE_POS[b]
        inten = flows[(a, b)] / fmax
        width = 1.2 + 5.0 * inten
        color = lerp_color(inten)

        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=width,
            arrowcolor=color, standoff=26, startstandoff=26, opacity=0.95,
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_annotation(
            x=mx, y=my, text=lbl, showarrow=False,
            font=dict(size=10, color=PALETTE["muted"]),
            bgcolor=PALETTE["panel"], borderpad=2, opacity=0.95,
        )

    xs = [NODE_POS[n][0] for n in NODE_ORDER]
    ys = [NODE_POS[n][1] for n in NODE_ORDER]
    sizes = [40 + 34 * (abs(values[n]) / vmax) for n in NODE_ORDER]
    texts = [f"{NODE_LABELS[n]}<br><span style='font-size:10px'>{values[n]:,.2f}</span>"
             for n in NODE_ORDER]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=sizes,
            color=[abs(values[n]) for n in NODE_ORDER],
            colorscale=[[0, "#1d2637"], [0.5, "#2e5aa8"], [1, "#4c8dff"]],
            line=dict(color="#5f6b82", width=1.4),
            showscale=False,
        ),
        text=texts, textposition="middle center",
        textfont=dict(size=13, color="#ffffff"),
        hovertext=[f"{NODE_TITLES[n]}<br>value = {values[n]:,.4f}" for n in NODE_ORDER],
        hoverinfo="text",
    ))

    base_layout(fig, 430, "Dynamic Causal Feedback Network \u2014 ECC Loop")
    fig.update_xaxes(visible=False, range=[-1.0, 2.2])
    fig.update_yaxes(visible=False, range=[-1.6, 1.15], scaleanchor="x", scaleratio=1)
    fig.add_annotation(
        x=-0.95, y=1.05, xanchor="left", showarrow=False,
        text=f"\u03b7 = {p['eta']:.2f}   |   step t = {state['t']}",
        font=dict(size=11, color=PALETTE["muted"]),
    )
    return fig


def pyramid_figure(c_value: float) -> go.Figure:
    roman_active, _, _, _, _ = tier_for(c_value)
    fig = go.Figure()

    n = len(TIERS)
    top_half, bottom_half = 0.30, 1.0
    for idx, (roman, name, lo, hi, color) in enumerate(reversed(TIERS)):
        # idx 0 = bottom tier VI
        y0, y1 = idx, idx + 1
        w0 = bottom_half - (bottom_half - top_half) * (idx / n)
        w1 = bottom_half - (bottom_half - top_half) * ((idx + 1) / n)
        active = roman == roman_active
        hi_txt = "\u221e" if hi > 10**8 else str(hi)

        fig.add_trace(go.Scatter(
            x=[-w0, w0, w1, -w1, -w0],
            y=[y0, y0, y1, y1, y0],
            fill="toself",
            fillcolor=color,
            opacity=1.0 if active else 0.28,
            line=dict(color="#ffffff" if active else "rgba(255,255,255,0.15)",
                      width=2.4 if active else 0.8),
            mode="lines",
            hovertemplate=(f"<b>Tier {roman} \u2014 {name}</b><br>"
                           f"C\u209c range: {lo}\u2013{hi_txt}<br>"
                           f"{TIER_QUESTIONS[roman]}<extra></extra>"),
            showlegend=False,
        ))
        label = f"<b>{roman}</b>  {name.split()[0]}"
        fig.add_annotation(
            x=0, y=(y0 + y1) / 2, text=label, showarrow=False,
            font=dict(size=12 if active else 11,
                      color="#111318" if active else "rgba(230,232,238,0.55)"),
        )

    base_layout(fig, 430, "State-Threshold Tier Graph \u2014 Hesusian Pyramid")
    fig.update_xaxes(visible=False, range=[-1.15, 1.15])
    fig.update_yaxes(visible=False, range=[-0.25, n + 0.35])
    fig.add_annotation(
        x=0, y=-0.15, showarrow=False,
        text=f"Active tier {roman_active} \u00b7 C\u209c = {c_value:,.2f}",
        font=dict(size=11, color=PALETTE["muted"]),
    )
    return fig


def series_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color, label in (
        ("C_t", "#4c8dff", "C\u209c \u00b7 Conceptual Development"),
        ("I_t", "#f09232", "I\u209c \u00b7 Informational Environment"),
        ("A_t", "#57c78a", "A\u209c \u00b7 Behavioral Expression"),
    ):
        fig.add_trace(go.Scatter(
            x=df["t"], y=df[col], mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate="t=%{x}<br>%{y:,.3f}<extra>" + label + "</extra>",
        ))
    base_layout(fig, 380, "State Trajectories Across Time Steps")
    fig.update_layout(
        margin=dict(l=56, r=24, t=88, b=44),
        legend=dict(orientation="h", y=1.16, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_xaxes(title="discrete time step t", gridcolor=PALETTE["grid"],
                     zeroline=False, linecolor=PALETTE["grid"])
    fig.update_yaxes(title="magnitude", gridcolor=PALETTE["grid"],
                     zeroline=False, linecolor=PALETTE["grid"], type="linear")
    return fig


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      .hp-sub {color:#8b93a7; font-size:0.9rem; margin-top:-0.6rem;}
      .hp-rule {border:0; border-top:1px solid #262c38; margin:1.1rem 0 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Hesusian Pyramid \u2014 Environmental Cycle of Conceptualization")
st.markdown(
    '<p class="hp-sub">Discrete-time state-space simulation of recursive conceptual '
    "development. \u03a6 = f \u2218 h \u2218 g</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="hp-rule"/>', unsafe_allow_html=True)

roman, tier_name, lo, hi, tier_color = tier_for(state["C_t"])

m = st.columns(6)
m[0].metric("Time step (t)", state["t"])
m[1].metric("C\u209c", f"{state['C_t']:,.2f}", f"{state['delta']:+,.2f}" if state["t"] else None)
m[2].metric("A\u209c", f"{state['A_t']:,.2f}")
m[3].metric("I\u209c", f"{state['I_t']:,.2f}")
m[4].metric("S\u209c", f"{state['S_t']:,.2f}")
m[5].metric("Active tier", f"{roman}", tier_name.split()[0])

left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.plotly_chart(ecc_graph(state, params), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("###### Governing equations")
    st.latex(r"A_t = g(C_t) = C_t \cdot \alpha")
    st.latex(r"I_{t+1} = I_t + \eta A_t")
    st.latex(r"S_{t+1} = h(I_{t+1}) = I_{t+1} \cdot \sigma")
    st.latex(r"C_{t+1} = C_t + B\left(\frac{S_{t+1}}{100}\right)\left(1 + \frac{R_t + F_t}{200}\right)")

with right:
    st.plotly_chart(pyramid_figure(state["C_t"]), use_container_width=True,
                    config={"displayModeBar": False})
    st.plotly_chart(series_figure(hist), use_container_width=True,
                    config={"displayModeBar": False})

with st.expander("Tier definitions and diagnostic questions"):
    st.dataframe(
        pd.DataFrame([
            {
                "Tier": r,
                "Stage": n_,
                "C\u209c range": f"{lo_}\u2013" + ("\u221e" if hi_ > 10**8 else str(hi_)),
                "Diagnostic question": TIER_QUESTIONS[r],
                "Active": "\u25cf" if r == roman else "",
            }
            for r, n_, lo_, hi_, _ in TIERS
        ]),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Simulation log"):
    st.dataframe(hist.round(4), hide_index=True, use_container_width=True, height=280)
    st.download_button(
        "Export trajectory (CSV)",
        hist.to_csv(index=False).encode("utf-8"),
        file_name="ecc_trajectory.csv",
        mime="text/csv",
    )
