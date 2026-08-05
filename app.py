"""
app.py
======
Streamlit entry point for the Hesusian Pyramid (HP) & Environmental
Conceptualization Cycle (ECC) interactive simulator.

Run with::

    streamlit run app.py

Dependencies:  streamlit, plotly, numpy, pandas, streamlit-katex
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from equations import HPState, DEGREE_INFO, latex_forms
from simulation import run_simulation, compute_statistics
from visualization import (
    pyramid_figure,
    recursive_dynamics_figure,
    ecc_figure,
    multiline_dynamics_figure,
    radar_figure,
    phase_portrait_figure,
)
from presets import PRESETS, preset_by_name, preset_names
from utils import SCALE_MAX

# Optional LaTeX rendering
try:
    from streamlit_katex import st_katex  # type: ignore
    HAS_KATEX = True
except Exception:
    HAS_KATEX = False

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Hesusian Pyramid Simulator",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Custom CSS — journal-like, minimalist, light theme
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    h1, h2, h3 { font-family: 'Inter', 'Helvetica Neue', sans-serif; letter-spacing: -0.02em; }
    .stMetric { background: #ffffff; border: 1px solid rgba(0,0,0,0.06);
                border-radius: 12px; padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .eq-card { background:#ffffff; border:1px solid rgba(0,0,0,0.06);
               border-radius:12px; padding:1rem 1.25rem; margin-bottom:0.75rem;
               box-shadow:0 1px 3px rgba(0,0,0,0.04); }
    .degree-tag { font-size:0.72rem; font-weight:600; color:#6b7280;
                  text-transform:uppercase; letter-spacing:0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_latex(tex: str) -> None:
    """Render a LaTeX string, with graceful fallback."""
    if HAS_KATEX:
        st_katex(tex)
    else:
        st.markdown(f"`{tex}`")


# --------------------------------------------------------------------------- #
# Session state for preset tracking
# --------------------------------------------------------------------------- #
if "preset_label" not in st.session_state:
    st.session_state["preset_label"] = "Custom (Manual Input)"
if "last_preset" not in st.session_state:
    st.session_state["last_preset"] = None


# --------------------------------------------------------------------------- #
# Sidebar — simulation controls
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🔺 Hesusian Pyramid")
st.sidebar.caption("Interactive HP / ECC Research Simulator")

st.sidebar.markdown("### Presets")
preset_name = st.sidebar.selectbox(
    "Configuration preset",
    options=preset_names(),
    index=0,
    key="preset_select",
    help="Selecting a preset updates every slider. The first option is always Custom.",
)

default = preset_by_name(preset_name) or PRESETS[0]

# If preset changed, load its values
if st.session_state.get("last_preset") != preset_name:
    st.session_state["last_preset"] = preset_name
    st.session_state["preset_label"] = preset_name
    # force defaults
    for k, v in dict(B=default.B, S=default.S, E=default.E, R=default.R,
                     F=default.F, eta=default.eta, lam=default.lam,
                     iterations=default.iterations).items():
        st.session_state[f"slider_{k}"] = v

st.sidebar.markdown("### Variables")

B = st.sidebar.slider("Biological Capacity (B)", 0.0, SCALE_MAX, float(default.B), 1.0, key="slider_B")
S = st.sidebar.slider("Informational Stimuli (S)", 0.0, SCALE_MAX, float(default.S), 1.0, key="slider_S")
E = st.sidebar.slider("Experience (E)", 0.0, SCALE_MAX, float(default.E), 1.0, key="slider_E")
R = st.sidebar.slider("Reasoning (R)", 0.0, SCALE_MAX, float(default.R), 1.0, key="slider_R")
F = st.sidebar.slider("Reflection (F)", 0.0, SCALE_MAX, float(default.F), 1.0, key="slider_F")
eta = st.sidebar.slider("Hesusian Influence Coefficient (η)", 0.0, 1.0, float(default.eta), 0.01, key="slider_eta")
lam = st.sidebar.slider("Environmental Decay Constant (λ)", 0.01, 0.99, float(default.lam), 0.01, key="slider_lam")
iterations = st.sidebar.slider("Number of Iterations", 5, 150, int(default.iterations), 5, key="slider_iterations")

# Detect manual modification -> relabel preset
loaded = preset_by_name(preset_name)
if loaded is not None and preset_name != "Custom (Manual Input)":
    if any([
        B != loaded.B, S != loaded.S, E != loaded.E, R != loaded.R,
        F != loaded.F, eta != loaded.eta, lam != loaded.lam, iterations != loaded.iterations,
    ]):
        st.session_state["preset_label"] = "Custom (Modified)"
    else:
        st.session_state["preset_label"] = preset_name

st.sidebar.markdown("---")
st.sidebar.caption(f"**Active preset:** {st.session_state['preset_label']}")


# --------------------------------------------------------------------------- #
# Run simulation
# --------------------------------------------------------------------------- #
result = run_simulation(B, S, E, R, F, eta, lam, iterations)
stats = compute_statistics(result)
current_state: HPState = result.states[-1]
current_acts = result.activations[-1]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown("# 🔺 Hesusian Pyramid & Environmental Conceptualization Cycle")
st.markdown(
    "An interactive research simulator for recursive conceptual development, "
    "environmental feedback, and the six Degrees of Conceptualization."
)
st.markdown(f"*Active configuration:* **{st.session_state['preset_label']}**")
st.markdown("---")

# --------------------------------------------------------------------------- #
# Row 1 — Pyramid + Radar
# --------------------------------------------------------------------------- #
col_pyramid, col_radar = st.columns([2, 1])

with col_pyramid:
    st.markdown("### Hesusian Pyramid — Degrees of Conceptualization")
    st.plotly_chart(pyramid_figure(current_acts), use_container_width=True)

    # Degree labels with activation percentages
    label_cols = st.columns(6)
    for i, col in enumerate(label_cols):
        deg = DEGREE_INFO[i]
        act = current_acts[i]
        col.markdown(
            f"<div class='degree-tag'>Deg {deg['roman']}</div>"
            f"<div style='font-weight:700;font-size:1.05rem;color:#111827'>{act:.0%}</div>"
            f"<div style='font-size:0.7rem;color:#6b7280'>{deg['name']}</div>",
            unsafe_allow_html=True,
        )

with col_radar:
    st.markdown("### Input Profile")
    st.plotly_chart(radar_figure(current_state), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Row 2 — Recursive Dynamics + ECC
# --------------------------------------------------------------------------- #
col_dyn, col_ecc = st.columns(2)
with col_dyn:
    st.markdown("### Recursive Dynamics")
    st.plotly_chart(recursive_dynamics_figure(result), use_container_width=True)

with col_ecc:
    st.markdown("### Environmental Conceptualization Cycle")
    st.plotly_chart(ecc_figure(result), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Row 3 — Multi-line + Phase portrait
# --------------------------------------------------------------------------- #
col_multi, col_phase = st.columns(2)
with col_multi:
    st.markdown("### Multi-Line Dynamics")
    st.caption("Click legend entries to hide individual variables.")
    st.plotly_chart(multiline_dynamics_figure(result), use_container_width=True)

with col_phase:
    st.markdown("### Phase Portrait")
    st.caption("Long-term recursive system behavior: Environment vs Conceptual Development.")
    st.plotly_chart(phase_portrait_figure(result), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Row 4 — Mathematical panel + Statistics
# --------------------------------------------------------------------------- #
col_math, col_stats = st.columns([3, 2])

with col_math:
    st.markdown("### Mathematical Panel")
    forms = latex_forms(current_state, lam)
    for f in forms:
        st.markdown(f"<div class='eq-card'><div class='degree-tag'>{f['label']}</div></div>", unsafe_allow_html=True)
        render_latex(f["symbolic"])
        render_latex(f["numeric"])
        render_latex(f["result"])
        st.markdown("---")

with col_stats:
    st.markdown("### Statistics")
    c1, c2 = st.columns(2)
    c1.metric("Max C", f"{stats['max_C']:.2f}")
    c2.metric("Average C", f"{stats['avg_C']:.2f}")
    c1.metric("Final C", f"{stats['final_C']:.2f}")
    c2.metric("Growth Rate", f"{stats['growth_rate']:.2%}")
    c1.metric("Environmental Growth", f"{stats['env_growth']:.2%}")
    c2.metric("Average Behavior", f"{stats['avg_behavior']:.2f}")
    c1.metric("Final Environment", f"{stats['final_env']:.2f}")
    c2.metric("Final Stimulus", f"{stats['final_stimulus']:.2f}")

st.markdown("---")
st.caption(
    "Hesusian Pyramid (HP) & Environmental Conceptualization Cycle (ECC) — "
    "theoretical framework simulator. All values are on a canonical 0–100 scale."
)