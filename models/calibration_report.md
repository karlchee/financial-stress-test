# Calibration Report

Generated: 2026-04-26

## Configuration

- Window: 15 years of monthly data
- Aggregation: overlapping 3-month windows
- Estimator: Ridge regression (alpha=1e-06)
- Minimum observations per stock: 60
- Stocks fit: **491**

## R-squared distribution

- Mean: 0.403
- Median: 0.387
- IQR: 0.298 - 0.509
- Min / Max: 0.017 / 0.812

## Top 10 best-fit stocks

| ticker   |   n_obs |    r2 |
|:---------|--------:|------:|
| CFG      |     135 | 0.812 |
| PRU      |     175 | 0.779 |
| FITB     |     175 | 0.776 |
| KEY      |     175 | 0.762 |
| HBAN     |     175 | 0.752 |
| MET      |     175 | 0.74  |
| PNC      |     175 | 0.72  |
| MS       |     175 | 0.719 |
| AMP      |     175 | 0.718 |
| JPM      |     175 | 0.716 |

## Bottom 10 worst-fit stocks

| ticker   |   n_obs |    r2 |
|:---------|--------:|------:|
| KDP      |     175 | 0.017 |
| LLY      |     175 | 0.05  |
| ERIE     |     175 | 0.079 |
| CNC      |     175 | 0.08  |
| REGN     |     175 | 0.084 |
| CTRA     |     175 | 0.084 |
| HRL      |     175 | 0.085 |
| PCG      |     175 | 0.089 |
| SJM      |     175 | 0.113 |
| FISV     |     175 | 0.113 |

## Cross-sectional mean factor sensitivities

|                 |   mean_beta |   median_beta |
|:----------------|------------:|--------------:|
| beta_mkt_excess |      0.9097 |        0.8138 |
| beta_smb        |      0.2732 |        0.2014 |
| beta_hml        |      0.18   |        0.2414 |
| beta_mom        |      0.0216 |        0.0456 |
| beta_oil        |      0.0083 |        0.0009 |
| beta_dxy        |     -0.139  |       -0.0829 |
| beta_ust10y     |     -0      |       -0.0001 |
| beta_slope      |     -0.0001 |       -0.0001 |
| beta_breakeven  |     -0      |       -0      |
| beta_vix        |     -0.0004 |       -0.0008 |

## Mean R-squared by GICS sector

| sector                 |   mean |   count |
|:-----------------------|-------:|--------:|
| Energy                 |  0.515 |      21 |
| Financials             |  0.514 |      74 |
| Industrials            |  0.442 |      77 |
| Real Estate            |  0.439 |      31 |
| Materials              |  0.438 |      26 |
| Information Technology |  0.395 |      70 |
| Consumer Discretionary |  0.391 |      48 |
| Communication Services |  0.367 |      23 |
| Utilities              |  0.316 |      30 |
| Health Care            |  0.3   |      56 |
| Consumer Staples       |  0.258 |      35 |

## Mean oil exposure by GICS sector (sanity check)

| sector                 |   mean_beta_oil |
|:-----------------------|----------------:|
| Energy                 |           0.269 |
| Consumer Staples       |           0.024 |
| Information Technology |           0.018 |
| Materials              |           0.011 |
| Utilities              |           0.006 |
| Real Estate            |           0.003 |
| Health Care            |          -0.004 |
| Financials             |          -0.008 |
| Communication Services |          -0.014 |
| Consumer Discretionary |          -0.015 |
| Industrials            |          -0.032 |

## Long-history retrieval dataset (factor_history.parquet)

- Total 3M windows: 676
- Date range: 1970-01-01 to 2026-04-01

### Coverage by factor

| factor     |   non_null_windows | earliest   | latest     |
|:-----------|-------------------:|:-----------|:-----------|
| mkt_excess |                491 | 1985-04-01 | 2026-02-01 |
| smb        |                672 | 1970-03-01 | 2026-02-01 |
| hml        |                672 | 1970-03-01 | 2026-02-01 |
| mom        |                672 | 1970-03-01 | 2026-02-01 |
| oil        |                673 | 1970-04-01 | 2026-04-01 |
| dxy        |                637 | 1973-04-01 | 2026-04-01 |
| ust10y     |                673 | 1970-04-01 | 2026-04-01 |
| slope      |                596 | 1976-09-01 | 2026-04-01 |
| breakeven  |                277 | 2003-04-01 | 2026-04-01 |
| vix        |                433 | 1990-04-01 | 2026-04-01 |
