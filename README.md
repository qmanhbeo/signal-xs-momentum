# Cross-Sectional Momentum in Vietnamese Equities

A fully reproducible research repository for the paper:
> **Cross-Sectional Momentum in Vietnamese Equities: A Machine-Learning Walk-Forward Study (2012-2026)**

## Key Results (Walk-Forward 2015-2025)

| Metric | Value |
|--------|-------|
| Mean daily IC (Spearman) | +0.1515 |
| Sig. years (p<0.05) | 11/11 |
| Gross / trade | +0.575% excess vs VNINDEX |
| Net / trade (-0.30% RT) | +0.275% |
| Net Sharpe (annual) | 2.19 |
| 2026 OOS IC | +0.1343 (pre-committed range [0.10, 0.13]) |
| 2026 OOS verdict | Weak pass: IC strong, portfolio still cost-sensitive |

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate momentum-paper

# 2. Run the full pipeline (fetch -> features -> model -> portfolio -> robustness -> 2026 holdout -> post-freeze extensions -> outputs -> manuscript)
python main.py

# Skip data fetch if you already have data/
python main.py --skip-fetch

# Re-run everything ignoring cached outputs
python main.py --force

# Run only specific steps
python main.py --steps 3,4,5
```

## Expected Output

```text
Step 1/8: Fetching data... 393 stocks + VIX + VNINDEX
Step 2/8: Data summary... ~974k rows
Step 3/8: Walk-forward 2015-2025... IC avg ~0.152
Step 4/8: Portfolio simulation... ~140 trades/yr, gross ~0.58%/trade
Step 5/8: Robustness tests... 4/4 passed
Step 6/8: 2026 holdout... IC +0.134 WEAK PASS
Step 7/9: Post-freeze extensions... holding grid + no-VIX cost stress + no-VIX 2026 rerun
Step 8/9: Generating figures + tables... 6 figures, 6 tables
Step 9/9: Compiling manuscript...
Done. See output/ and manuscript/paper.pdf
```

## Repository Structure

```text
signal-xs-momentum/
|-- README.md
|-- main.py
|-- requirements.txt
|-- environment.yml
|
|-- src/
|   |-- fetch.py
|   |-- features.py
|   |-- model.py
|   |-- portfolio.py
|   `-- utils.py
|
|-- analysis/
|   |-- 01_data_summary.py
|   |-- 02_signal_ic.py
|   |-- 03_portfolio.py
|   |-- 04_robustness.py
|   |-- 05_holdout_2026.py
|   `-- 06_outputs.py
|
|-- data/
|   |-- stock/
|   `-- vix/
|
|-- output/
|   |-- figures/
|   |-- tables/
|   `-- results/
|
`-- manuscript/
    |-- paper.tex
    `-- paper.bib
```

## Methodology Summary

**Universe:** 393 HOSE stocks, daily OHLCV, 2012-01-01 to 2026-03-18

**Features (35):** MA distance, RSI, MACD, Bollinger Bands, ATR, candle structure,
stochastic oscillator, ADX, realized volatility, and relative/market returns vs VNINDEX.
No fundamentals or alternative data.

**Target:** Triple barrier +/-1% relative to VNINDEX over 5 days.

**Model:** XGBoost classifier (depth=3, lr=0.05, n=500) trained cross-sectionally
on all available stocks with expanding-window walk-forward.

**Portfolio:** Top-5 stocks by predicted probability, square weighting,
VIX filter [15, 30], MIN_PROBA >= 0.55, E=1 H=1 execution.

**Cost:** 0.30% round-trip assumed.

## Reproducibility Notes

- All results are deterministic (`random_state=42` in XGBoost).
- The 2026 pre-committed expectations were documented in
  `notes/session_23_2026_holdout.md` in the parent research repo before the holdout was run.
- Walk-forward ensures zero look-ahead: test year data is never seen during training.

## Later Credibility Updates

The compact paper pipeline above is the frozen publication candidate. The repo now
also includes a reproducible supplemental step, `main.py --steps 7`, that runs
three later credibility checks:

- Executable no-VIX holding-period grid: `H=1` is best by annual net at `0.30%` RT
  (`+59.32%`), ahead of `H=2` (`+44.60%`) and `H=5` (`+21.66%`).
- No-VIX cost stress: exact break-even for daily `H=1` is `0.54%` RT; weekly `H=5`
  breaks even at `0.73%` RT.
- No-VIX 2026 rerun: IC stays strong at `+0.1323`, square-weight gross/trade is
  `+0.198%`, and net/trade is `-0.102%` at `0.30%` RT.

These results are saved in `output/results/postfreeze_extensions.json` and summarized
for the manuscript in `output/tables/tab_postfreeze.tex`.

## Dependencies

- Python 3.11
- vnstock==3.4.2
- xgboost>=2.0.0
- pandas>=2.0.0, numpy>=1.24.0, scipy>=1.10.0
- matplotlib>=3.7.0, pyarrow>=14.0.0
- yfinance>=0.2.40

## License

MIT
