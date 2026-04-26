"""Historical analog retrieval over the long-history factor dataset.

Two retrieval modes:

    find_nearest_windows(shocks, history, k=5)
        Quantitative: nearest historical 3M windows to a given shock vector,
        using normalized Euclidean distance. Each factor is divided by its
        historical std before distance is computed, so a 1-std move in oil
        counts the same as a 1-std move in ust10y bps. Windows where fewer
        than half of the supplied factors have non-null values are skipped.

    find_curated_analogs(shocks, curated, k=3)
        Qualitative: handcrafted historical events (Iraq/Kuwait, GFC, COVID,
        SVB, etc.) loaded from models/analogs.json. Used as few-shot reasoning
        anchors for the LLM advisor. Same normalized-Euclidean approach,
        with stds computed from the curated set itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors import FACTOR_NAMES


@dataclass(frozen=True)
class HistoricalAnalog:
    """A historical 3M window matched against a candidate shock vector."""
    window_end: pd.Timestamp
    distance: float
    factors: dict[str, float]      # actual factor values for this window
    matched_factors: list[str]     # which factors participated in the distance


# ---- Loaders ----

def load_factor_history(model_dir: str | Path = "models") -> pd.DataFrame:
    """Load the long-history retrieval dataset (factor_history.parquet)."""
    return pd.read_parquet(Path(model_dir) / "factor_history.parquet")


def load_curated_analogs(model_dir: str | Path = "models") -> list[dict]:
    """Load the curated analogs from analogs.json."""
    path = Path(model_dir) / "analogs.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])


# ---- Internals ----

def _normalized_distance(candidates: pd.DataFrame,
                         shock_vec: pd.Series,
                         min_avail_ratio: float = 0.5) -> pd.Series:
    """Per-row normalized Euclidean distance to shock_vec.

    Both sides are divided by per-factor stds computed from `candidates`
    (skipping NaN). For each row, only factors that are non-null in both
    the shock vector and the row participate. Rows where fewer than
    `min_avail_ratio` of the supplied factors have non-null values get NaN.
    The returned distance is the *average* per-factor squared difference,
    sqrt'd, so rows with fewer matched factors are not penalized for
    coverage alone.
    """
    stds = candidates.std(skipna=True).replace(0, np.nan)
    cand_norm = candidates.divide(stds, axis=1)
    shock_norm = shock_vec.divide(stds)

    diff = cand_norm.subtract(shock_norm, axis=1)
    sq = diff.pow(2)
    n_avail = sq.notna().sum(axis=1)
    sum_sq = sq.sum(axis=1, skipna=True)
    min_n = max(1, int(min_avail_ratio * len(shock_vec)))

    avg_sq = np.where(
        n_avail >= min_n,
        sum_sq / n_avail.replace(0, 1),
        np.nan,
    )
    return pd.Series(np.sqrt(avg_sq), index=candidates.index)


# ---- Public API ----

def find_nearest_windows(shocks: dict[str, float],
                         history: pd.DataFrame,
                         k: int = 5) -> list[HistoricalAnalog]:
    """Find the k nearest historical 3M windows to the given shock vector.

    Only factors present in the shock vector and (per row) non-null in
    `history` participate in the distance.
    """
    factors_in_shock = [f for f in FACTOR_NAMES if f in shocks]
    H = history[factors_in_shock]
    shock_vec = pd.Series({f: shocks[f] for f in factors_in_shock})

    distances = _normalized_distance(H, shock_vec)
    nearest = distances.dropna().nsmallest(k)

    out: list[HistoricalAnalog] = []
    for ts, d in nearest.items():
        row = H.loc[ts]
        avail = row.dropna().index.tolist()
        out.append(HistoricalAnalog(
            window_end=ts,
            distance=float(d),
            factors={f: float(row[f]) for f in avail},
            matched_factors=avail,
        ))
    return out


def find_curated_analogs(shocks: dict[str, float],
                         curated: list[dict],
                         k: int = 3) -> list[dict]:
    """Find the k closest curated analogs to the shock vector.

    Returns each matched analog dict augmented with a `distance` field,
    sorted ascending by distance.
    """
    if not curated:
        return []

    rows = [evt["shocks"] for evt in curated]
    A = pd.DataFrame(rows).reindex(columns=FACTOR_NAMES)

    factors_in_shock = [f for f in FACTOR_NAMES if f in shocks]
    A_sub = A[factors_in_shock]
    shock_vec = pd.Series({f: shocks[f] for f in factors_in_shock})

    distances = _normalized_distance(A_sub, shock_vec).values

    augmented: list[dict] = []
    for evt, d in zip(curated, distances):
        if np.isnan(d):
            continue
        e = dict(evt)
        e["distance"] = float(d)
        augmented.append(e)

    augmented.sort(key=lambda e: e["distance"])
    return augmented[:k]
