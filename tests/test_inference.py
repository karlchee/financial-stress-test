"""Smoke tests for the inference and retrieval layers.

Run from repo root:
    pytest tests/

These tests exercise the math layer only — no LLM keys required. They assume
calibration artifacts already exist under models/.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.factors import FACTOR_NAMES, zero_shocks
from src.model import load_model, predict, summarize_predictions
from src.retrieval import (
    find_curated_analogs,
    find_nearest_windows,
    load_curated_analogs,
    load_factor_history,
)


# ---- model.py ----

def test_zero_shocks_equal_intercept():
    """With all zero shocks, predictions should equal each stock's intercept."""
    art = load_model("models")
    pred = predict(zero_shocks(), art)
    diff = (pred["predicted_excess_return"] - pred["intercept"]).abs()
    assert diff.max() < 1e-12


def test_zero_shocks_no_intercept():
    """With all zero shocks and no intercept, predictions should be zero."""
    art = load_model("models")
    pred = predict(zero_shocks(), art, use_intercept=False)
    assert (pred["predicted_excess_return"].abs() < 1e-12).all()
    # Intercept column should still be present
    assert "intercept" in pred.columns


def test_predict_shape_and_columns():
    art = load_model("models")
    pred = predict(zero_shocks(), art)
    for col in ("ticker", "name", "sector", "predicted_excess_return",
                "intercept", "n_obs", "r2"):
        assert col in pred.columns, f"missing {col}"
    for fname in FACTOR_NAMES:
        assert f"contrib_{fname}" in pred.columns
    assert len(pred) > 100  # we expect ~491 stocks


def test_oil_shock_helps_energy_more_than_staples():
    art = load_model("models")
    shocks = zero_shocks()
    shocks["oil"] = 0.50
    pred = predict(shocks, art)
    energy = pred[pred["sector"] == "Energy"]
    staples = pred[pred["sector"] == "Consumer Staples"]
    assert energy["contrib_oil"].mean() > staples["contrib_oil"].mean()
    assert energy["contrib_oil"].mean() > 0


def test_invalid_shocks_raises():
    art = load_model("models")
    with pytest.raises(ValueError):
        predict({"oil": 0.10}, art)  # missing all other factors


def test_summarize_predictions_keys():
    art = load_model("models")
    pred = predict(zero_shocks(), art)
    summary = summarize_predictions(pred)
    for k in ("n_stocks", "mean", "median", "std", "min", "max",
              "worst", "best", "by_sector", "factor_totals"):
        assert k in summary
    assert len(summary["worst"]) == 10
    assert len(summary["best"]) == 10


# ---- retrieval.py ----

def test_nearest_windows_returns_k_sorted():
    hist = load_factor_history("models")
    shocks = zero_shocks()
    shocks["oil"] = 0.50
    nearest = find_nearest_windows(shocks, hist, k=5)
    assert len(nearest) == 5
    distances = [a.distance for a in nearest]
    assert distances == sorted(distances)


def test_curated_analogs_returns_k_sorted():
    curated = load_curated_analogs("models")
    shocks = zero_shocks()
    shocks["mkt_excess"] = -0.20
    shocks["vix"] = 30
    matches = find_curated_analogs(shocks, curated, k=3)
    assert len(matches) == 3
    distances = [m["distance"] for m in matches]
    assert distances == sorted(distances)
    for m in matches:
        assert "name" in m and "date" in m


def test_lehman_matches_severe_drawdown():
    """A severe-drawdown shock vector should pull GFC-era windows in."""
    curated = load_curated_analogs("models")
    shocks = {
        "mkt_excess": -0.25,
        "vix": 40,
        "oil": -0.50
    }
    matches = find_curated_analogs(shocks, curated, k=3)
    names = {m["name"] for m in matches}
    # Lehman is the obvious analog — should be in the top 3
    assert any("Lehman" in n for n in names), f"got {names}"
