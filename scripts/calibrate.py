#!/usr/bin/env python
"""Calibrate per-stock multi-factor model on overlapping 3-month returns.

Pulls:
  - S&P 500 constituents from Wikipedia
  - Monthly stock prices from yfinance
  - Macro factors from FRED (oil, USD, rates, spreads, VIX)
  - Style factors from Ken French's data library (SMB, HML, momentum)

Fits ridge regression per stock on overlapping 3M aggregates of monthly data.
Also builds a separate long-history retrieval dataset (back to 1970) used by
the LLM at inference time to find historical analogs.

Outputs (under --output, default models/):
  coefficients.parquet     - alpha, betas, R2, n_obs per stock
  stock_metadata.csv       - tickers, names, GICS sectors
  factor_history.parquet   - long-history 3M factor windows (1970-present)
  calibration_report.md    - diagnostic summary

Usage:
  python scripts/calibrate.py                          # full S&P 500, 15y window
  python scripts/calibrate.py --years 10 --limit 30    # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge

# Allow running as `python scripts/calibrate.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.factors import FACTOR_NAMES  # noqa: E402

warnings.filterwarnings("ignore")


# ---- Configuration ----

# Calibration data: dense, modern, used to fit per-stock betas
FRED_SERIES = {
    "oil_wti":      "DCOILWTICO",     # WTI spot, daily, 1986-
    "dxy":          "DTWEXBGS",       # Trade-weighted USD broad index, 2006-
    "ust10y":       "DGS10",          # 10Y Treasury constant maturity, 1962-
    "ust2y":        "DGS2",           # 2Y Treasury constant maturity, 1976-
    "breakeven10y": "T10YIE",         # 10Y breakeven inflation, 2003-
    "vix":          "VIXCLS",         # VIX close, 1990-
}

# Long-history retrieval data: covers as far back as each series allows.
# Used by LLM for analog lookup. Older periods will have nulls for series
# that didn't exist (VIX pre-1990, spreads pre-1996, breakeven pre-2003,
# slope pre-1976). Oil and DXY are spliced from longer-history alternatives.
LONG_HISTORY_START = pd.Timestamp("1970-01-01")
LONG_FRED_SERIES = {
    "oil_wti":      "DCOILWTICO",     # 1986- (primary, post-deregulation spot)
    "oil_ppi":      "WPU0561",        # 1947- (PPI Crude Petroleum, monthly proxy)
    "dxy_broad":    "DTWEXBGS",       # 2006- (primary, modern broad index)
    "dxy_major":    "DTWEXM",         # 1973-2020 (major-currencies index, discontinued)
    "ust10y":       "DGS10",
    "ust2y":        "DGS2",
    "breakeven10y": "T10YIE",
    "vix":          "VIXCLS",
}

DEFAULT_YEARS = 15
DEFAULT_MIN_MONTHS = 60
DEFAULT_RIDGE_ALPHA = 0.000001


# ---- Helpers ----

def _to_month_start(idx) -> pd.DatetimeIndex:
    """Normalize any date-like index to month-start timestamps."""
    return pd.to_datetime(idx).to_period("M").to_timestamp(how="start")


# ---- Data loaders ----

def load_sp500_tickers() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia.

    Returns columns: ticker, name, sector. Tickers use Yahoo's '-' convention
    (e.g. 'BRK-B' rather than 'BRK.B').
    """
    import requests
    from io import StringIO
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0].rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
    })
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector"]].reset_index(drop=True)


def load_monthly_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Download monthly auto-adjusted close prices.

    Returns wide DataFrame indexed by month-start, columns are tickers.
    Tickers with no data are dropped silently.
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    data = yf.download(
        tickers, start=start, end=end,
        interval="1mo", auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        # default yfinance returns ('field', 'ticker')
        closes = data["Close"]
    else:
        # single ticker
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.dropna(axis=1, how="all")
    closes.index = _to_month_start(closes.index)
    return closes


def load_fred(series: dict[str, str], start, end) -> pd.DataFrame:
    """Load FRED daily series, ffilled, indexed by date."""
    from pandas_datareader import data as pdr
    frames = []
    for col, code in series.items():
        try:
            s = pdr.DataReader(code, "fred", start, end)
            s.columns = [col]
            frames.append(s)
        except Exception as exc:
            print(f"  WARN: FRED fetch failed for {code} ({col}): {exc}")
    if not frames:
        raise RuntimeError("No FRED series loaded -- check network and try again.")
    return pd.concat(frames, axis=1).ffill()


def load_french(start, end) -> pd.DataFrame:
    """Load Fama-French 3 factors + Carhart momentum, monthly, in decimals.

    Returns columns: mkt_excess_ff, smb, hml, rf, mom
    Index: month-start timestamps.
    """
    from pandas_datareader import data as pdr
    ff = pdr.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)[0]
    mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start, end)[0]
    out = ff.join(mom)
    out.columns = [c.strip() for c in out.columns]
    out = out.rename(columns={
        "Mkt-RF": "mkt_excess_ff",
        "SMB":    "smb",
        "HML":    "hml",
        "RF":     "rf",
        "Mom":    "mom",
    })
    out = out.div(100.0)  # percent -> decimal
    out.index = _to_month_start(out.index.to_timestamp(how="start"))
    return out


def load_sp500_index(start, end) -> pd.Series:
    """S&P 500 monthly close (^GSPC) for market-excess factor."""
    sp = yf.download("^GSPC", start=start, end=end, interval="1mo",
                     auto_adjust=True, progress=False)
    s = sp["Close"].squeeze()
    s.index = _to_month_start(s.index)
    s.name = "sp500"
    return s


# ---- Factor construction ----

def build_monthly_factors(fred_d: pd.DataFrame, french: pd.DataFrame,
                          sp500_m: pd.Series) -> pd.DataFrame:
    """Construct monthly factor changes/returns aligned on month-start.

    Returns DataFrame with FACTOR_NAMES columns plus 'rf' for excess-return
    construction downstream.
    """
    fred_m = fred_d.resample("MS").last().ffill()
    fred_m.index = _to_month_start(fred_m.index)
    
    # Convert French PeriodIndex to DatetimeIndex for alignment
    french_ts = french.copy()
    if hasattr(french_ts.index, 'to_timestamp'):
        french_ts.index = _to_month_start(french_ts.index.to_timestamp(how="start"))
    french_ts = french_ts.reindex(fred_m.index)

    f = pd.DataFrame(index=fred_m.index)
    f["oil"]       = fred_m["oil_wti"].pct_change()
    f["dxy"]       = fred_m["dxy"].pct_change()
    f["ust10y"]    = fred_m["ust10y"].diff() * 100        # percent -> bps
    f["slope"]     = (fred_m["ust10y"] - fred_m["ust2y"]).diff() * 100
    f["breakeven"] = fred_m["breakeven10y"].diff() * 100
    f["vix"]       = fred_m["vix"].diff()                  # already in points

    # Style factors from Ken French (already monthly, decimal)
    for c in ("smb", "hml", "mom"):
        f[c] = french_ts[c]
    rf = french_ts["rf"]

    # Market excess: SP500 monthly return - RF
    sp_ret = sp500_m.pct_change()
    sp_ret = sp_ret.reindex(f.index)
    rf_aligned = rf.reindex(f.index)
    f["mkt_excess"] = sp_ret - rf_aligned
    f["rf"] = rf_aligned

    # Reorder to canonical FACTOR_NAMES + rf
    return f[FACTOR_NAMES + ["rf"]]


def aggregate_3m_overlap(monthly: pd.DataFrame,
                          return_cols: list[str],
                          change_cols: list[str]) -> pd.DataFrame:
    """Compound returns / sum changes over trailing 3M windows.

    Each row at date t represents the cumulative shock from t-3 to t (inclusive
    of three monthly observations). Strict NaN propagation: if any of the
    three monthly inputs is null, the aggregated value is null. This matters
    for the long-history dataset where some factors have null values in
    older periods.
    """
    out = pd.DataFrame(index=monthly.index)
    for c in return_cols:
        out[c] = (1 + monthly[c]).rolling(3).apply(np.prod, raw=True) - 1
    for c in change_cols:
        # np.sum (raw) propagates NaN, unlike pandas .sum() which may skip them
        out[c] = monthly[c].rolling(3).apply(np.sum, raw=True)
    return out


def splice_returns(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Compute monthly returns within each series, then prefer primary.

    For each month, returns primary's pct_change if the underlying primary
    series had non-null values both this month and last; otherwise falls back
    to secondary's pct_change. This avoids constructing a spurious return at
    the splice point (which would compare two different price levels).
    """
    primary_ret = primary.pct_change()
    secondary_ret = secondary.pct_change()
    return primary_ret.where(primary_ret.notna(), secondary_ret)


def build_long_monthly_factors(fred_d: pd.DataFrame, french: pd.DataFrame,
                                sp500_m: pd.Series) -> pd.DataFrame:
    """Construct monthly factor changes/returns over the full long history.

    Differs from build_monthly_factors in that:
      - Oil is spliced from WTI (post-1986) backed by PPI Crude Petroleum (1947-).
      - DXY is spliced from Broad TWI (post-2006) backed by Major TWI (1973-2020).
      - Older periods retain nulls for VIX (pre-1990), credit spreads (pre-1996),
        breakeven (pre-2003), and slope (pre-1976).

    Returns DataFrame with FACTOR_NAMES columns ordered canonically.
    """
    fred_m = fred_d.resample("MS").last()
    fred_m.index = _to_month_start(fred_m.index)
    
    # Convert French PeriodIndex to DatetimeIndex for alignment
    french_ts = french.copy()
    if hasattr(french_ts.index, 'to_timestamp'):
        french_ts.index = _to_month_start(french_ts.index.to_timestamp(how="start"))
    french_ts = french_ts.reindex(fred_m.index)

    f = pd.DataFrame(index=fred_m.index)

    # Multi-source splicing for max history
    f["oil"] = splice_returns(fred_m["oil_wti"], fred_m["oil_ppi"])
    f["dxy"] = splice_returns(fred_m["dxy_broad"], fred_m["dxy_major"])

    # Yield-based factors (null pre-series-start)
    f["ust10y"]    = fred_m["ust10y"].diff() * 100
    f["slope"]     = (fred_m["ust10y"] - fred_m["ust2y"]).diff() * 100
    f["breakeven"] = fred_m["breakeven10y"].diff() * 100
    f["vix"]       = fred_m["vix"].diff()

    # Style factors from Ken French (back to 1926, no nulls)
    for c in ("smb", "hml", "mom"):
        f[c] = french_ts[c]

    # Market excess: SP500 monthly return - RF
    sp_ret = sp500_m.pct_change()
    sp_ret = sp_ret.reindex(f.index)
    rf_aligned = french_ts["rf"].reindex(f.index)
    f["mkt_excess"] = sp_ret - rf_aligned

    return f[FACTOR_NAMES]


def build_long_history_dataset(end: pd.Timestamp,
                                output_dir: Path) -> pd.DataFrame:
    """Build and save the long-history retrieval dataset.

    Saves models/factor_history.parquet -- overlapping 3M factor windows from
    1970 to today. Used by Phase 2's retrieval module to find historical
    analogs for LLM-proposed scenarios.
    """
    print(f"      loading long-history series since {LONG_HISTORY_START.date()}...")
    long_fred = load_fred(LONG_FRED_SERIES, LONG_HISTORY_START, end)
    long_french = load_french(LONG_HISTORY_START, end)
    long_sp500 = load_sp500_index(LONG_HISTORY_START, end)

    print(f"      building monthly factors with multi-source splicing...")
    long_factors_m = build_long_monthly_factors(long_fred, long_french, long_sp500)

    return_cols = ["mkt_excess", "smb", "hml", "mom", "oil", "dxy"]
    change_cols = ["ust10y", "slope", "breakeven", "vix"]
    long_factor_3m = aggregate_3m_overlap(long_factors_m, return_cols, change_cols)

    out_path = output_dir / "factor_history.parquet"
    long_factor_3m.to_parquet(out_path)
    return long_factor_3m


# ---- Calibration ----

def fit_per_stock(stock_y: pd.DataFrame, factor_X: pd.DataFrame,
                  alpha: float, min_obs: int) -> pd.DataFrame:
    """One ridge regression per stock. Returns coefficient table.

    Columns: ticker, alpha, n_obs, r2, beta_<factor> for each factor.
    Stocks with fewer than `min_obs` aligned observations are skipped.
    
    Handles missing data intelligently:
      - Always drops rows where stock y is NaN
      - Always drops rows where core factors (mkt_excess, smb, hml, mom, oil, dxy) are NaN
      - For optional factors with significant gaps (if >50% missing),
        drops sparse factor columns entirely rather than dropping rows
      - Fills remaining NaN in non-core factors with forward-fill
    """
    # Core factors that must be non-null (foundational market + style factors plus commodities)
    core_factors = ["mkt_excess", "smb", "hml", "mom", "oil", "dxy"]
    
    # Optional factors that may be sparse (credit spreads especially)
    optional_factors = ["ust10y", "slope", "breakeven", "vix"]
    
    rows = []
    for ticker in stock_y.columns:
        df = pd.concat([stock_y[ticker].rename("y"), factor_X], axis=1)
        
        # Drop rows where stock y or core factors are NaN
        df = df.dropna(subset=["y"] + core_factors)
        
        # Check which optional factors have too many missing values (>50%)
        factors_to_use = core_factors.copy()
        for opt_factor in optional_factors:
            if opt_factor in df.columns:
                nan_ratio = df[opt_factor].isna().sum() / len(df)
                if nan_ratio < 0.5:
                    # This factor has <50% missing, include it
                    factors_to_use.append(opt_factor)
        
        # Forward-fill remaining NaN in factors we're using
        for f in factors_to_use:
            if f in df.columns and f not in core_factors:
                df[f] = df[f].ffill()
        
        # Drop any remaining NaN rows (shouldn't be many after ffill)
        df = df.dropna(subset=["y"] + factors_to_use)
        
        if len(df) < min_obs:
            continue
        
        X = df[factors_to_use].values
        y = df["y"].values
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X, y)
        y_pred = m.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        row = {
            "ticker": ticker,
            "alpha":  float(m.intercept_),
            "n_obs":  int(len(df)),
            "r2":     float(r2),
        }
        # Record betas for all FACTOR_NAMES (use NaN for factors not in the model)
        factor_idx = {f: i for i, f in enumerate(factors_to_use)}
        for fname in FACTOR_NAMES:
            if fname in factor_idx:
                row[f"beta_{fname}"] = float(m.coef_[factor_idx[fname]])
            else:
                row[f"beta_{fname}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Reporting ----

def write_report(coefs: pd.DataFrame, output_dir: Path,
                 years: int, min_obs: int, alpha: float,
                 metadata: pd.DataFrame,
                 long_factor_3m: pd.DataFrame | None = None) -> None:
    """Write calibration_report.md with diagnostics."""
    today = pd.Timestamp.today().date().isoformat()
    lines: list[str] = []
    lines.append(f"# Calibration Report\n\n")
    lines.append(f"Generated: {today}\n\n")
    lines.append("## Configuration\n\n")
    lines.append(f"- Window: {years} years of monthly data\n")
    lines.append(f"- Aggregation: overlapping 3-month windows\n")
    lines.append(f"- Estimator: Ridge regression (alpha={alpha})\n")
    lines.append(f"- Minimum observations per stock: {min_obs}\n")
    lines.append(f"- Stocks fit: **{len(coefs)}**\n\n")

    if coefs.empty:
        lines.append("No stocks passed the minimum-observations threshold.\n")
        (output_dir / "calibration_report.md").write_text("".join(lines), encoding="utf-8")
        return

    lines.append("## R-squared distribution\n\n")
    r2 = coefs["r2"].dropna()
    lines.append(f"- Mean: {r2.mean():.3f}\n")
    lines.append(f"- Median: {r2.median():.3f}\n")
    lines.append(f"- IQR: {r2.quantile(0.25):.3f} - {r2.quantile(0.75):.3f}\n")
    lines.append(f"- Min / Max: {r2.min():.3f} / {r2.max():.3f}\n\n")

    lines.append("## Top 10 best-fit stocks\n\n")
    top = coefs.nlargest(10, "r2")[["ticker", "n_obs", "r2"]].copy()
    top["r2"] = top["r2"].round(3)
    lines.append(top.to_markdown(index=False) + "\n\n")

    lines.append("## Bottom 10 worst-fit stocks\n\n")
    bot = coefs.nsmallest(10, "r2")[["ticker", "n_obs", "r2"]].copy()
    bot["r2"] = bot["r2"].round(3)
    lines.append(bot.to_markdown(index=False) + "\n\n")

    lines.append("## Cross-sectional mean factor sensitivities\n\n")
    beta_cols = [c for c in coefs.columns if c.startswith("beta_")]
    means = coefs[beta_cols].mean().round(4).to_frame("mean_beta")
    medians = coefs[beta_cols].median().round(4)
    means["median_beta"] = medians.values
    lines.append(means.to_markdown() + "\n\n")

    if "sector" in metadata.columns:
        merged = coefs.merge(metadata[["ticker", "sector"]], on="ticker", how="left")
        if merged["sector"].notna().any():
            lines.append("## Mean R-squared by GICS sector\n\n")
            by_sec = (
                merged.groupby("sector")["r2"]
                .agg(["mean", "count"])
                .round(3)
                .sort_values("mean", ascending=False)
            )
            lines.append(by_sec.to_markdown() + "\n\n")

            lines.append("## Mean oil exposure by GICS sector (sanity check)\n\n")
            if "beta_oil" in merged.columns:
                oil_by_sec = (
                    merged.groupby("sector")["beta_oil"]
                    .mean()
                    .round(3)
                    .sort_values(ascending=False)
                    .to_frame("mean_beta_oil")
                )
                lines.append(oil_by_sec.to_markdown() + "\n\n")

    # Long-history retrieval dataset coverage
    if long_factor_3m is not None and not long_factor_3m.empty:
        lines.append("## Long-history retrieval dataset (factor_history.parquet)\n\n")
        lines.append(f"- Total 3M windows: {len(long_factor_3m)}\n")
        lines.append(f"- Date range: {long_factor_3m.index.min().date()} to {long_factor_3m.index.max().date()}\n\n")
        lines.append("### Coverage by factor\n\n")
        cov_rows = []
        for fac in long_factor_3m.columns:
            col = long_factor_3m[fac].dropna()
            cov_rows.append({
                "factor": fac,
                "non_null_windows": len(col),
                "earliest": col.index.min().date().isoformat() if len(col) else "n/a",
                "latest": col.index.max().date().isoformat() if len(col) else "n/a",
            })
        cov_df = pd.DataFrame(cov_rows)
        lines.append(cov_df.to_markdown(index=False) + "\n")

    (output_dir / "calibration_report.md").write_text("".join(lines), encoding="utf-8")


# ---- Main ----

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"Years of history (default: {DEFAULT_YEARS})")
    parser.add_argument("--output", type=Path, default=ROOT / "models",
                        help="Output directory for artifacts")
    parser.add_argument("--min-months", type=int, default=DEFAULT_MIN_MONTHS,
                        help=f"Min monthly observations per stock (default: {DEFAULT_MIN_MONTHS})")
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA,
                        help=f"Ridge regularization (default: {DEFAULT_RIDGE_ALPHA})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Subset to first N tickers (for debugging)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)

    print(f"[1/7] Loading S&P 500 constituents...")
    universe = load_sp500_tickers()
    if args.limit:
        universe = universe.head(args.limit)
    tickers = universe["ticker"].tolist()
    print(f"      {len(tickers)} tickers")

    print(f"[2/7] Downloading {args.years}y of monthly stock prices...")
    prices = load_monthly_prices(tickers, args.years)
    rets = prices.pct_change()
    median_cov = int(rets.notna().sum().median())
    print(f"      shape={prices.shape}, median ticker coverage={median_cov} months")

    print(f"[3/7] Loading FRED + Ken French + S&P 500 index (calibration window)...")
    fred_d = load_fred(FRED_SERIES, start, end)
    french = load_french(start, end)
    sp500_m = load_sp500_index(start, end)
    print(f"      FRED: {fred_d.shape}, French: {french.shape}, SP500: {len(sp500_m)} months")

    print(f"[4/7] Building monthly factor returns...")
    factors_m = build_monthly_factors(fred_d, french, sp500_m)
    print(f"      shape={factors_m.shape}, non-null rows={factors_m.dropna().shape[0]}")

    print(f"[5/7] Aggregating to 3M overlapping windows...")
    return_cols = ["mkt_excess", "smb", "hml", "mom", "oil", "dxy"]
    change_cols = ["ust10y", "slope", "breakeven", "vix"]
    factor_3m = aggregate_3m_overlap(factors_m, return_cols, change_cols)
    rf_3m = (1 + factors_m["rf"]).rolling(3).apply(np.prod, raw=True) - 1
    stock_3m = (1 + rets).rolling(3).apply(np.prod, raw=True) - 1
    stock_3m_excess = stock_3m.sub(rf_3m, axis=0)
    print(f"      factor_3m={factor_3m.shape}, stock_3m_excess={stock_3m_excess.shape}")

    print(f"[6/7] Fitting per-stock ridge regressions...")
    coefs = fit_per_stock(stock_3m_excess, factor_3m, args.ridge_alpha, args.min_months)
    skipped = len(tickers) - len(coefs)
    print(f"      {len(coefs)} stocks fit, {skipped} skipped (insufficient history)")

    # Persist calibration artifacts
    coefs.to_parquet(args.output / "coefficients.parquet", index=False)
    universe.to_csv(args.output / "stock_metadata.csv", index=False)

    print(f"[7/7] Building long-history factor dataset for retrieval...")
    long_factor_3m = build_long_history_dataset(end, args.output)
    print(f"      shape={long_factor_3m.shape}")
    print(f"      Coverage by factor (non-null 3M windows | earliest):")
    for fac in FACTOR_NAMES:
        col = long_factor_3m[fac].dropna()
        n = len(col)
        earliest = col.index.min().date().isoformat() if n else "n/a"
        print(f"        {fac:12s}: {n:4d} | {earliest}")

    write_report(coefs, args.output, args.years, args.min_months,
                 args.ridge_alpha, universe, long_factor_3m)

    print()
    print(f"Done. Artifacts written to {args.output}/")
    print(f"  - coefficients.parquet    ({len(coefs)} rows)")
    print(f"  - stock_metadata.csv      ({len(universe)} rows)")
    print(f"  - factor_history.parquet  ({len(long_factor_3m)} 3M windows since 1970)")
    print(f"  - calibration_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
