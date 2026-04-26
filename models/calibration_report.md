# Calibration Report

Generated: 2026-04-26

## Configuration

- Window: 15 years of monthly data
- Aggregation: overlapping 3-month windows
- Estimator: Ridge regression (alpha=1.0)
- Minimum observations per stock: 60
- Stocks fit: **491**

## R-squared distribution

- Mean: 0.310
- Median: 0.294
- IQR: 0.217 - 0.400
- Min / Max: 0.009 / 0.654

## Top 10 best-fit stocks

| ticker   |   n_obs |    r2 |
|:---------|--------:|------:|
| HAL      |     175 | 0.654 |
| PRU      |     175 | 0.621 |
| SLB      |     175 | 0.618 |
| EOG      |     175 | 0.614 |
| MET      |     175 | 0.614 |
| EMR      |     175 | 0.591 |
| COP      |     175 | 0.589 |
| TRGP     |     175 | 0.589 |
| AMP      |     175 | 0.583 |
| INVH     |     106 | 0.568 |

## Bottom 10 worst-fit stocks

| ticker   |   n_obs |    r2 |
|:---------|--------:|------:|
| KDP      |     175 | 0.009 |
| LLY      |     175 | 0.035 |
| PCG      |     175 | 0.049 |
| ERIE     |     175 | 0.057 |
| REGN     |     175 | 0.061 |
| CTRA     |     175 | 0.064 |
| HRL      |     175 | 0.064 |
| CNC      |     175 | 0.068 |
| SJM      |     175 | 0.071 |
| KR       |     175 | 0.078 |

## Cross-sectional mean factor sensitivities

|                 |   mean_beta |   median_beta |
|:----------------|------------:|--------------:|
| beta_mkt_excess |      0.1634 |        0.1541 |
| beta_smb        |      0.0756 |        0.0656 |
| beta_hml        |      0.0299 |        0.0476 |
| beta_mom        |     -0.0162 |       -0.0073 |
| beta_oil        |      0.0224 |        0.0157 |
| beta_dxy        |     -0.0215 |       -0.0176 |
| beta_ust10y     |     -0.0003 |       -0.0004 |
| beta_slope      |      0.0002 |        0.0001 |
| beta_breakeven  |      0.0009 |        0.0008 |
| beta_vix        |     -0.0044 |       -0.0043 |

## Mean R-squared by GICS sector

| sector                 |   mean |   count |
|:-----------------------|-------:|--------:|
| Energy                 |  0.452 |      21 |
| Financials             |  0.385 |      74 |
| Materials              |  0.374 |      26 |
| Real Estate            |  0.347 |      31 |
| Industrials            |  0.346 |      77 |
| Information Technology |  0.297 |      70 |
| Consumer Discretionary |  0.286 |      48 |
| Communication Services |  0.278 |      23 |
| Utilities              |  0.243 |      30 |
| Health Care            |  0.222 |      56 |
| Consumer Staples       |  0.184 |      35 |

## Mean oil exposure by GICS sector (sanity check)

| sector                 |   mean_beta_oil |
|:-----------------------|----------------:|
| Energy                 |           0.235 |
| Materials              |           0.035 |
| Information Technology |           0.027 |
| Consumer Staples       |           0.024 |
| Financials             |           0.018 |
| Real Estate            |           0.017 |
| Health Care            |           0.013 |
| Utilities              |           0.013 |
| Communication Services |           0.003 |
| Consumer Discretionary |          -0.002 |
| Industrials            |          -0.007 |

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
