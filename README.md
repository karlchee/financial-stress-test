---
title: Financial Stress Testing
sdk: streamlit
sdk_version: 1.34.0
app_file: app.py
pinned: false
license: mit
---

# Financial Stress Testing Tool

LLM-guided stress testing for the S&P 500. The user describes a scenario in plain English (e.g. an oil shock or banking stress); an LLM advisor refines it into a vector of 3-month factor shocks; a per-stock linear factor model maps those shocks to predicted stock-level returns with factor attribution.

NTU SCTP Data Science and AI capstone project.

## Architecture

```
[User scenario in plain English]
        |
        v
[LLM advisor — Gemini / Claude / Groq]
   * asks clarifying questions
   * anchors on historical analogs (Gulf War, COVID, SVB, etc.)
   * emits structured factor shocks via tool-use
   * user can override shocks directly in the UI
        |
        v
[Pre-trained linear factor model]
   * per-stock ridge regression
   * 12 macro/style factors
   * fit on 15y monthly data, overlapping 3M windows
        |
        v
[Predicted 3M returns + factor attribution per stock]
```

## Quick start (Codespaces)

1. Click **Code -> Codespaces -> Create codespace on main**. The devcontainer installs dependencies automatically (~2 min on first launch).
2. Set Codespaces secrets at the GitHub user level (Settings -> Codespaces -> Secrets), grant access to this repo:
   - `GEMINI_API_KEY` — get from https://aistudio.google.com/app/apikey (free)
3. Calibrate the model (one-time, refresh occasionally):

   ```bash
   python scripts/calibrate.py --years 15
   ```

   For a quick smoke test with 30 stocks:
   ```bash
   python scripts/calibrate.py --years 10 --limit 30
   ```

4. Launch the app:
   ```bash
   streamlit run app.py
   ```

   Codespaces forwards port 8501 to a public HTTPS URL — viewable on a phone.

5. Commit the calibration artifacts:
   ```bash
   git add models/ && git commit -m "calibrate model" && git push
   ```

## Deployment to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space (SDK: Streamlit, hardware: CPU basic = free).
2. Add `GEMINI_API_KEY` to Space Settings -> Variables and secrets.
3. Connect to this GitHub repo, or use the included GitHub Action to auto-mirror on push.
4. Push to `main` — the Space rebuilds and deploys automatically.

## Project structure

```
.
├── app.py                       Streamlit entry point
├── src/
│   ├── factors.py               12-factor schema (single source of truth)
│   ├── llm.py                   LLM provider abstraction (Gemini/Anthropic/Groq)
│   ├── model.py                 Inference: shocks -> per-stock predictions
│   └── ui.py                    Streamlit UI components
├── prompts/
│   └── scenario_advisor.md      LLM system prompt
├── scripts/
│   └── calibrate.py             Per-stock model calibration
├── models/                      Calibration artifacts (committed to repo)
│   ├── coefficients.parquet
│   ├── stock_metadata.csv
│   ├── analogs.json             Curated historical analogs
│   └── calibration_report.md    Diagnostics
├── tests/
├── .devcontainer/               Codespaces config
└── .github/workflows/           Optional auto-deploy to HF Space
```

## Methodology

**Horizon.** 3 months. Calibration aggregates monthly returns into overlapping 3-month windows so that fitted betas map 3-month factor shocks directly to 3-month stock returns.

**Factors (12).** Carhart 4-factor (market excess, SMB, HML, momentum) plus 8 macro: WTI oil, broad USD, UST 10Y yield, 2s10s slope, 10Y breakeven inflation, IG corporate spread, HY corporate spread, VIX. See `src/factors.py` for the schema with units and typical ranges.

**Estimator.** Ridge regression per stock (default `α=1.0`) on 15-year monthly window; minimum 60 monthly observations required. Ridge keeps coefficients well-behaved when factors correlate (which they do in stress periods).

**Universe.** Current S&P 500 constituents scraped from Wikipedia. Stocks without sufficient history are skipped.

**LLM role.** The LLM never produces stock-level numbers. It only proposes structured factor shocks (with reasoning) which the user can accept, edit, or refine through conversation. The user has final authority.

**Data sources.** All free: yfinance (S&P 500 prices, ^GSPC), FRED via pandas-datareader (oil/USD/rates/spreads/VIX), Ken French data library (SMB/HML/momentum).

## Disclaimer

Educational project. Not investment advice.
"# financial-stress-test" 
