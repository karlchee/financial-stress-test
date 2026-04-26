"""Inference: shocks -> per-stock predicted 3M excess returns + factor attribution.

Loads calibration artifacts and applies a user-supplied shock vector. The model
predicts 3-month *excess* returns (over rf) because that is what was fit during
calibration. To convert to raw 3M returns, add back the prevailing 3M risk-free
rate.

NOTE on column naming: in `coefficients.parquet` the column named ``alpha`` is
the regression *intercept*, not the regularization parameter. (The name is
inherited from the calibration script.) We rename it to ``intercept`` in the
inference output to avoid confusion.

Public API:
    load_model(model_dir)          -> ModelArtifacts
    predict(shocks, artifacts)     -> DataFrame (per-stock predictions)
    summarize_predictions(preds)   -> dict (UI-friendly aggregates)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.factors import FACTOR_NAMES, validate_shocks


@dataclass(frozen=True)
class ModelArtifacts:
    """Loaded calibration artifacts."""
    coefficients: pd.DataFrame    # ticker, alpha (=intercept), n_obs, r2, beta_<factor>
    metadata: pd.DataFrame        # ticker, name, sector


def load_model(model_dir: str | Path = "models") -> ModelArtifacts:
    """Load calibration artifacts from disk."""
    model_dir = Path(model_dir)
    coef = pd.read_parquet(model_dir / "coefficients.parquet")
    meta = pd.read_csv(model_dir / "stock_metadata.csv")
    return ModelArtifacts(coefficients=coef, metadata=meta)


def predict(shocks: dict[str, float],
            artifacts: ModelArtifacts) -> pd.DataFrame:
    """Apply shock vector to each stock's betas. Returns per-stock predictions
    with full factor attribution.

    Returns columns:
      ticker, name, sector, predicted_excess_return,
      contrib_<factor> for each factor, intercept, n_obs, r2

    NaN betas (from sparse-factor handling in calibration) are treated as 0
    for that stock -- equivalent to assuming the stock has no exposure to a
    factor we couldn't reliably fit for it.

    Output is sorted ascending by predicted_excess_return (worst to best).
    """
    valid, msgs = validate_shocks(shocks)
    if not valid:
        errors = [m for m in msgs if m.startswith("ERROR")]
        raise ValueError(f"Invalid shock vector: {errors}")

    coef = artifacts.coefficients
    meta = artifacts.metadata

    # Per-factor contributions: shock x beta. Treat NaN beta as 0.
    contribs: dict[str, pd.Series] = {}
    for fname in FACTOR_NAMES:
        beta = coef[f"beta_{fname}"].fillna(0.0)
        contribs[f"contrib_{fname}"] = beta * float(shocks[fname])
    contrib_df = pd.DataFrame(contribs)

    # Predicted excess return = intercept + sum of contributions
    pred = coef["alpha"].astype(float) + contrib_df.sum(axis=1)

    out = pd.DataFrame({
        "ticker": coef["ticker"].values,
        "predicted_excess_return": pred.values,
        "intercept": coef["alpha"].astype(float).values,
        "n_obs": coef["n_obs"].values,
        "r2": coef["r2"].values,
    })
    for c in contrib_df.columns:
        out[c] = contrib_df[c].values

    # Attach metadata (name, sector)
    out = out.merge(meta[["ticker", "name", "sector"]], on="ticker", how="left")

    front = ["ticker", "name", "sector", "predicted_excess_return"]
    contrib_cols = [f"contrib_{f}" for f in FACTOR_NAMES]
    diag_cols = ["intercept", "n_obs", "r2"]
    return (
        out[front + contrib_cols + diag_cols]
        .sort_values("predicted_excess_return")
        .reset_index(drop=True)
    )


def summarize_predictions(predictions: pd.DataFrame, top_k: int = 10) -> dict:
    """Produce summary stats for UI display.

    Returns a dict with: n_stocks, mean/median/std/min/max of predicted
    excess return, top_k worst and best, sector aggregates, and the
    cross-sectional mean factor contribution (so we can see which factor
    is driving the scenario in aggregate).
    """
    pred_col = "predicted_excess_return"
    contrib_cols = [c for c in predictions.columns if c.startswith("contrib_")]

    return {
        "n_stocks": int(len(predictions)),
        "mean": float(predictions[pred_col].mean()),
        "median": float(predictions[pred_col].median()),
        "std": float(predictions[pred_col].std()),
        "min": float(predictions[pred_col].min()),
        "max": float(predictions[pred_col].max()),
        "worst": predictions.nsmallest(top_k, pred_col)[
            ["ticker", "name", "sector", pred_col]
        ].to_dict(orient="records"),
        "best": predictions.nlargest(top_k, pred_col)[
            ["ticker", "name", "sector", pred_col]
        ].to_dict(orient="records"),
        "by_sector": (
            predictions.groupby("sector")[pred_col]
            .agg(["mean", "median", "count"])
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "factor_totals": {
            c.replace("contrib_", ""): float(predictions[c].mean())
            for c in contrib_cols
        },
    }
