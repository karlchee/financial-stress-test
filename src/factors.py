"""Factor schema: definitions, units, validation.

Single source of truth for the 12-factor model. Used by:
  - scripts/calibrate.py  (regression feature columns)
  - src/model.py          (inference column ordering)
  - src/llm.py            (LLM tool-use schema)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    """A single factor in the model."""
    name: str
    description: str
    unit: str                            # human-readable (for LLM + UI)
    direction: str                       # "return" or "change"
    typical_3m_range: tuple[float, float]  # heuristic sanity bounds


# Order matters — used as column order in calibration and inference matrices.
FACTORS: list[Factor] = [
    Factor(
        name="mkt_excess",
        description="S&P 500 total return minus risk-free rate, cumulative over 3 months",
        unit="decimal (e.g. -0.10 means -10%)",
        direction="return",
        typical_3m_range=(-0.30, 0.20),
    ),
    Factor(
        name="smb",
        description="Fama-French Small-Minus-Big size factor, cumulative over 3 months",
        unit="decimal",
        direction="return",
        typical_3m_range=(-0.10, 0.10),
    ),
    Factor(
        name="hml",
        description="Fama-French High-Minus-Low value factor, cumulative over 3 months",
        unit="decimal",
        direction="return",
        typical_3m_range=(-0.10, 0.15),
    ),
    Factor(
        name="mom",
        description="Carhart momentum factor (Up-minus-Down), cumulative over 3 months",
        unit="decimal",
        direction="return",
        typical_3m_range=(-0.20, 0.20),
    ),
    Factor(
        name="oil",
        description="WTI crude oil spot price 3-month return",
        unit="decimal (e.g. 0.50 means +50%)",
        direction="return",
        typical_3m_range=(-0.45, 0.60),
    ),
    Factor(
        name="dxy",
        description="Trade-weighted broad USD index 3-month return",
        unit="decimal",
        direction="return",
        typical_3m_range=(-0.08, 0.12),
    ),
    Factor(
        name="ust10y",
        description="10-year US Treasury yield change over 3 months",
        unit="basis points (e.g. 50 means +50bps)",
        direction="change",
        typical_3m_range=(-150, 150),
    ),
    Factor(
        name="slope",
        description="10Y minus 2Y yield curve slope change over 3 months",
        unit="basis points",
        direction="change",
        typical_3m_range=(-100, 100),
    ),
    Factor(
        name="breakeven",
        description="10Y breakeven inflation rate change over 3 months",
        unit="basis points",
        direction="change",
        typical_3m_range=(-80, 80),
    ),
    Factor(
        name="ig_spread",
        description="ICE BofA US Investment Grade corporate OAS change over 3 months",
        unit="basis points",
        direction="change",
        typical_3m_range=(-50, 250),
    ),
    Factor(
        name="hy_spread",
        description="ICE BofA US High Yield corporate OAS change over 3 months",
        unit="basis points",
        direction="change",
        typical_3m_range=(-200, 700),
    ),
    Factor(
        name="vix",
        description="VIX index level change over 3 months",
        unit="points (e.g. 10 means +10 vol points)",
        direction="change",
        typical_3m_range=(-20, 50),
    ),
]


FACTOR_NAMES: list[str] = [f.name for f in FACTORS]
FACTORS_BY_NAME: dict[str, Factor] = {f.name: f for f in FACTORS}


def validate_shocks(shocks: dict[str, float]) -> tuple[bool, list[str]]:
    """Validate a shock vector against the schema.

    Returns (is_valid, list_of_messages). Messages include both errors
    (missing/extra/non-numeric factors) and soft warnings (values 3x outside
    typical range).
    """
    msgs: list[str] = []
    hard_errors = 0

    for name in FACTOR_NAMES:
        if name not in shocks:
            msgs.append(f"ERROR: missing factor '{name}'")
            hard_errors += 1
            continue
        v = shocks[name]
        if not isinstance(v, (int, float)):
            msgs.append(f"ERROR: '{name}' is not numeric (got {type(v).__name__})")
            hard_errors += 1
            continue
        lo, hi = FACTORS_BY_NAME[name].typical_3m_range
        if v < 3 * lo or v > 3 * hi:
            msgs.append(
                f"WARNING: '{name}'={v} is outside 3x typical range [{lo}, {hi}]"
            )

    extra = set(shocks) - set(FACTOR_NAMES)
    if extra:
        msgs.append(f"ERROR: unknown factors {sorted(extra)}")
        hard_errors += 1

    return hard_errors == 0, msgs


def zero_shocks() -> dict[str, float]:
    """Convenience: a baseline (no-shock) factor vector."""
    return {name: 0.0 for name in FACTOR_NAMES}
