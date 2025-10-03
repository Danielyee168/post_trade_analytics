# Post-Trade Analytics (PTA)

This repository packages a lightweight post-trade analytics stack for Crypto.com research. It ingests venue fills (see `sample_trades.csv`), rebuilds symbol-level inventory, and generates portfolio diagnostics that highlight cash usage, exposure, and realized P&L under multiple accounting treatments. Junior analysts can follow the guided notebook to understand how raw executions become risk-aware insights, while the CLI pipeline automates daily reporting.

## What the Toolkit Provides
- **Data ingestion**: `src/pta_analysis.load_trades` parses exchange exports with Decimal precision and validates schema expectations.
- **Accounting engine**: `PTAEngine` replays trades through two P&L methods—Weighted Average Cost (smooths entry price) and FIFO (lot-by-lot matching)—to compare realizations and residual exposure.
- **Analytics outputs**: utilities resample portfolio curves, build per-symbol summaries, and (optionally) render Plotly dashboards or static charts for reporting.
- **Walkthrough**: `pta_walkthrough.ipynb` breaks the workflow into teachable steps with commentary and interpretation guidance.

## Quickstart
1. Create a Python 3.11 environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run linting/formatting before committing:
   ```bash
   ruff check src
   black src notebooks
   ```
3. Execute the ingest pipeline and produce reports (defaults to `data/raw/sample_trades.csv`):
   ```bash
   python src/pta_analysis.py --output-dir reports --session-date 2024-01-01
   ```
4. Explore the interactive walkthrough (optional):
   ```bash
   jupyter notebook notebooks/pta_walkthrough.ipynb
   ```

## Accounting Methods
- **Weighted Average Cost (WAC)**: maintains a rolling average entry price, realizing P&L when positions flip direction and smoothing frequent churn.
- **First-In-First-Out (FIFO)**: matches exits to the earliest open lots, releasing P&L according to the original trade sequence and making inventory layering explicit.

Compare method outputs in the generated reports or notebook interpretation cell to understand how methodology selection impacts risk narratives.
