"""Streamlit UI components for the stress testing app.

Components:
    chat_panel(messages, send_callback)
    shocks_panel(shocks, on_change)        -> editable 10-factor vector
    results_panel(predictions, summary)
    analog_panel(curated, nearest)
    market_state_panel(state)
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from src.factors import FACTOR_NAMES, FACTORS_BY_NAME


# Display format per factor — change-style factors are integers/decimals,
# return-style factors are 4-decimal floats.
_RETURN_FACTORS = {"mkt_excess", "smb", "hml", "mom", "oil", "dxy"}


def _factor_format(name: str) -> str:
    return "%.4f" if name in _RETURN_FACTORS else "%.2f"


def chat_panel(messages: list[dict],
               send_callback: Callable[[str], None]) -> None:
    """Render chat history and a send box."""
    st.subheader("Scenario advisor")
    user_input = st.chat_input("Describe a scenario...")
    if user_input:
        send_callback(user_input)
    container = st.container(height=400, key="chat_container")
    with container:
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])


def shocks_panel(shocks: dict[str, float],
                 on_change: Callable[[dict[str, float]], None]) -> dict[str, float]:
    """Editable shocks panel.

    Renders one number input per factor with unit/range hints. The widget
    keys (``shock_<fname>``) are the source of truth for what's displayed.
    Callers (e.g. the LLM tool-call handler) should write directly to those
    keys to update the displayed values; we then mirror them back into the
    canonical ``shocks`` dict via ``on_change``.
    """
    st.subheader("Factor shocks (3-month)")

    # Initialize widget state from the canonical shocks dict on first render.
    # Streamlit uses the widget key as source of truth on subsequent reruns,
    # so we must seed it here rather than via st.number_input(value=...).
    for fname in FACTOR_NAMES:
        wk = f"shock_{fname}"
        if wk not in st.session_state:
            st.session_state[wk] = float(shocks.get(fname, 0.0))

    cols = st.columns(2)
    for i, fname in enumerate(FACTOR_NAMES):
        f = FACTORS_BY_NAME[fname]
        col = cols[i % 2]
        with col:
            st.number_input(
                fname,
                help=f"{f.description}\nUnit: {f.unit}\nTypical 3M: {f.typical_3m_range}",
                key=f"shock_{fname}",
                format=_factor_format(fname),
            )

    edited = {fname: float(st.session_state[f"shock_{fname}"])
              for fname in FACTOR_NAMES}
    if edited != shocks:
        on_change(edited)
    return edited


def _format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def results_panel(predictions: pd.DataFrame, summary: dict) -> None:
    """Render per-stock predictions and aggregations."""
    st.subheader("Predicted 3M excess returns")
    st.caption(
        "Excess return = predicted 3M return minus 3M risk-free. "
        "Negative = underperforms cash; positive = outperforms cash."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median", _format_pct(summary["median"]))
    c2.metric("Mean", _format_pct(summary["mean"]))
    c3.metric("Worst", _format_pct(summary["min"]))
    c4.metric("Best", _format_pct(summary["max"]))

    tab_top, tab_sector, tab_factors, tab_table = st.tabs(
        ["Top movers", "By sector", "Factor drivers", "Full table"]
    )

    with tab_top:
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**Worst 10**")
            df = pd.DataFrame(summary["worst"])
            df["predicted_excess_return"] = df["predicted_excess_return"].map(_format_pct)
            st.dataframe(df, hide_index=True, use_container_width=True)
        with col_r:
            st.write("**Best 10**")
            df = pd.DataFrame(summary["best"])
            df["predicted_excess_return"] = df["predicted_excess_return"].map(_format_pct)
            st.dataframe(df, hide_index=True, use_container_width=True)

    with tab_sector:
        sec = pd.DataFrame(summary["by_sector"])
        sec["mean"] = sec["mean"].map(_format_pct)
        sec["median"] = sec["median"].map(_format_pct)
        sec = sec.rename(columns={"count": "n"})
        st.dataframe(
            sec.sort_values("median", ascending=False, key=lambda s: s.str.rstrip("%").astype(float)),
            hide_index=True, use_container_width=True,
        )

    with tab_factors:
        st.write("Cross-sectional mean contribution per factor (decimal return units)")
        ft = pd.DataFrame(
            list(summary["factor_totals"].items()),
            columns=["factor", "mean_contribution"],
        )
        ft["mean_contribution"] = ft["mean_contribution"].round(4)
        st.dataframe(
            ft.sort_values("mean_contribution"),
            hide_index=True, use_container_width=True,
        )

    with tab_table:
        display = predictions.copy()
        display["predicted_excess_return"] = (display["predicted_excess_return"] * 100).round(2)
        st.dataframe(display, hide_index=True, use_container_width=True, height=400)


def analog_panel(curated: list[dict], nearest: list) -> None:
    """Display side-by-side curated and quantitative historical analogs."""
    st.subheader("Historical analogs")
    col_curated, col_nearest = st.columns(2)
    with col_curated:
        st.write("**Curated events (closest match)**")
        if not curated:
            st.caption("No curated analogs matched.")
        for e in curated:
            with st.expander(f"{e['name']} ({e['date']})  ·  d={e.get('distance', 0):.2f}"):
                st.write(e.get("summary", ""))
                if e.get("lessons"):
                    st.caption(e["lessons"])
    with col_nearest:
        st.write("**Closest 3M windows in history**")
        if not nearest:
            st.caption("No historical windows matched.")
        for a in nearest:
            label = f"{a.window_end.date()}  ·  d={a.distance:.2f}"
            with st.expander(label):
                st.write({k: round(v, 4) for k, v in a.factors.items()})


def market_state_panel(state: dict) -> None:
    """Render the current market snapshot used as LLM context."""
    st.caption("Current market state (snapshot at app startup)")
    df = pd.DataFrame(list(state.items()), columns=["indicator", "value"])
    st.dataframe(df, hide_index=True, use_container_width=True)
