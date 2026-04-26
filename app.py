"""Streamlit entry point for the financial stress testing tool.

Run:
    streamlit run app.py

Env vars (set in Codespaces secrets / HF Space variables / .env):
    LLM_PROVIDER          gemini | anthropic | groq  (default: gemini)
    GEMINI_API_KEY        required if LLM_PROVIDER=gemini
    ANTHROPIC_API_KEY     required if LLM_PROVIDER=anthropic
    GROQ_API_KEY          required if LLM_PROVIDER=groq
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.factors import FACTOR_NAMES, validate_shocks, zero_shocks
from src.llm import get_provider, propose_factor_shocks_tool
from src.model import load_model, predict, summarize_predictions
from src.retrieval import (
    find_curated_analogs,
    find_nearest_windows,
    load_curated_analogs,
    load_factor_history,
)
from src.ui import (
    analog_panel,
    chat_panel,
    market_state_panel,
    results_panel,
    shocks_panel,
)


load_dotenv()

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PROMPTS_DIR = ROOT / "prompts"


# ---- App config ----

st.set_page_config(
    page_title="Financial Stress Testing",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Cached loaders ----

@st.cache_resource
def _artifacts():
    return load_model(MODEL_DIR)


@st.cache_resource
def _factor_history():
    return load_factor_history(MODEL_DIR)


@st.cache_resource
def _curated():
    return load_curated_analogs(MODEL_DIR)


@st.cache_resource
def _llm_provider():
    return get_provider()


@st.cache_resource
def _system_prompt() -> str:
    base = (PROMPTS_DIR / "scenario_advisor.md").read_text(encoding="utf-8")
    analog_block = "\n\n## Curated historical analogs\n\n" + json.dumps(
        _curated(), indent=2
    )
    return base + analog_block


@st.cache_data(ttl=60 * 60)
def _market_snapshot() -> dict:
    """Placeholder market snapshot.

    For now this is a stub so the app boots without network. A later
    iteration should hook this to live FRED/yfinance pulls (e.g. last
    SPX close, 10Y, oil, VIX).
    """
    return {
        "snapshot_taken_at": datetime.now(timezone.utc).isoformat(),
        "note": "stub snapshot — wire to FRED/yfinance in next iteration",
    }


# ---- State init ----

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("shocks", zero_shocks())
    ss.setdefault("predictions", None)
    ss.setdefault("summary", None)
    ss.setdefault("nearest", [])
    ss.setdefault("curated_match", [])


# ---- Actions ----

def _on_user_message(text: str) -> None:
    """Send the user's message to the LLM, append the response, and capture
    any propose_factor_shocks tool call into the editable shocks state.
    """
    ss = st.session_state
    ss["messages"].append({"role": "user", "content": text})

    try:
        provider = _llm_provider()
        resp = provider.chat(
            ss["messages"],
            _system_prompt(),
            [propose_factor_shocks_tool()],
        )
    except Exception as exc:  # noqa: BLE001
        ss["messages"].append({
            "role": "assistant",
            "content": f"_LLM error: {exc}_",
        })
        return

    ss["messages"].append({"role": "assistant", "content": resp.get("text", "")})

    for call in resp.get("tool_calls", []):
        if call["name"] != "propose_factor_shocks":
            continue
        new_shocks = {f: 0.0 for f in FACTOR_NAMES}
        for k, v in call["arguments"].items():
            if k in FACTOR_NAMES:
                try:
                    new_shocks[k] = float(v)
                except (TypeError, ValueError):
                    pass
        # Update both the canonical state and each widget key so the
        # shocks panel reflects the LLM's proposal on the next rerun.
        # (Streamlit number_input widgets use their key as source of truth
        # after first render, so the canonical dict alone isn't enough.)
        ss["shocks"] = new_shocks
        for fname, val in new_shocks.items():
            ss[f"shock_{fname}"] = val
        # Show a small confirmation in the chat that the panel was updated.
        ss["messages"].append({
            "role": "assistant",
            "content": "_Shocks panel updated — review and edit, then click **Run model**._",
        })


def _on_run() -> None:
    """Run inference with current shocks."""
    ss = st.session_state
    valid, msgs = validate_shocks(ss["shocks"])
    if not valid:
        errors = [m for m in msgs if m.startswith("ERROR")]
        st.error("Invalid shocks: " + "; ".join(errors))
        return
    preds = predict(ss["shocks"], _artifacts())
    ss["predictions"] = preds
    ss["summary"] = summarize_predictions(preds)
    ss["nearest"] = find_nearest_windows(ss["shocks"], _factor_history(), k=5)
    ss["curated_match"] = find_curated_analogs(ss["shocks"], _curated(), k=3)


def _on_shocks_edit(edited: dict[str, float]) -> None:
    st.session_state["shocks"] = edited


def _on_reset() -> None:
    ss = st.session_state
    ss["messages"] = []
    ss["shocks"] = zero_shocks()
    ss["predictions"] = None
    ss["summary"] = None
    ss["nearest"] = []
    ss["curated_match"] = []
    # Clear the widget-keyed shock values too so the panel resets visually.
    for fname in FACTOR_NAMES:
        ss[f"shock_{fname}"] = 0.0


# ---- Layout ----

def main() -> None:
    _init_state()

    st.title("Financial Stress Testing")
    st.caption(
        "LLM scenario advisor + per-stock linear factor model · educational use only"
    )

    with st.sidebar:
        market_state_panel(_market_snapshot())
        st.divider()
        st.write(f"**LLM provider:** {os.environ.get('LLM_PROVIDER', 'gemini')}")
        try:
            st.write(f"**Stocks loaded:** {len(_artifacts().coefficients)}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load model artifacts: {exc}")
        st.divider()
        st.button("Reset session", on_click=_on_reset, use_container_width=True)

    left, right = st.columns([5, 7])

    with left:
        chat_panel(st.session_state["messages"], _on_user_message)
        st.divider()
        shocks_panel(st.session_state["shocks"], _on_shocks_edit)
        st.button(
            "Run model",
            on_click=_on_run,
            type="primary",
            use_container_width=True,
        )

    with right:
        if st.session_state["predictions"] is not None:
            results_panel(
                st.session_state["predictions"],
                st.session_state["summary"],
            )
            st.divider()
            analog_panel(
                st.session_state["curated_match"],
                st.session_state["nearest"],
            )
        else:
            st.info(
                "Describe a scenario in chat or edit shocks directly, "
                "then click **Run model**."
            )


if __name__ == "__main__":
    main()
