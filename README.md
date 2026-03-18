# Cross-Sectional Momentum in Vietnamese Equities

A fully reproducible research repository for the paper:
> **Cross-Sectional Momentum in Vietnamese Equities: A Machine-Learning Walk-Forward Study (2012–2026)**

## Key Results (Walk-Forward 2015–2025)

| Metric | Value |
|--------|-------|
| Mean daily IC (Spearman) | +0.137 |
| Sig. years (p<0.05) | 10/11 |
| Gross / trade | +0.40% excess vs VNINDEX |
| Net / trade (−0.30% RT) | +0.26% |
| Net Sharpe (annual) | 4.58 |
| 2026 OOS IC | +0.132 ✓ (pre-committed range [0.10, 0.13]) |

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate momentum-paper

# 2. Run the full pipeline (fetch → features → model → portfolio → robustness → 2026 holdout → outputs → manuscript)
python main.py

# Skip data fetch if you already have data/
python main.py --skip-fetch

# Re-run everything ignoring cached outputs
python main.py --force

# Run only specific steps
python main.py --steps 3,4,5
```

## Expected Output

```
Step 1/8: Fetching data... 393 stocks + VIX + VNINDEX
Step 2/8: Data summary... ~974k rows
Step 3/8: Walk-forward 2015-2025... IC avg ~0.137
Step 4/8: Portfolio simulation... ~45 trades/yr, gross ~0.40%/trade
Step 5/8: Robustness tests... 4/4 passed
Step 6/8: 2026 holdout... IC +0.132 PASS
Step 7/8: Generating figures + tables... 6 figures, 5 tables
Step 8/8: Compiling manuscript...
Done. See output/ and manuscript/paper.pdf
```

## Repository Structure

```
market-momentum-analysis/
├── README.md
├── main.py                    # One-click runner
├── requirements.txt
├── environment.yml            # conda env (python 3.11)
│
├── src/
│   ├── fetch.py               # Data fetching (vnstock + yfinance)
│   ├── features.py            # 35 production features + triple-barrier target
│   ├── model.py               # XGBoost cross-sectional walk-forward
│   ├── portfolio.py           # Portfolio simulation + metrics
│   └── utils.py               # Constants, path helpers
│
├── analysis/
│   ├── 01_data_summary.py     # Universe coverage stats
│   ├── 02_signal_ic.py        # IC by year
│   ├── 03_portfolio.py        # Portfolio annual returns
│   ├── 04_robustness.py       # 4 robustness tests
│   ├── 05_holdout_2026.py     # 2026 out-of-sample test
│   └── 06_outputs.py          # Figures + LaTeX tables
│
├── data/                      # Auto-populated (git-ignored)
│   ├── stock/                 # gia_lich_su_{TICKER}_1D.parquet (393 stocks)
│   └── vix/                   # vix_daily.parquet
│
├── output/                    # Auto-generated (git-ignored)
│   ├── figures/               # PDF + PNG figures
│   ├── tables/                # .tex table fragments
│   └── results/               # JSON/CSV intermediate results
│
└── manuscript/
    ├── paper.tex              # Full LaTeX manuscript
    ├── paper.bib              # BibTeX references
    └── figures/               # (symlink target for paper.tex)
```

## Methodology Summary

**Universe:** 393 HOSE stocks, daily OHLCV, 2012-01-01 to 2026-03-18

**Features (35):** MA distance, RSI, MACD, Bollinger Bands, ATR, candle structure,
stochastic oscillator, ADX, realized volatility, and relative/market returns vs VNINDEX.
No fundamentals or alternative data.

**Target:** Triple barrier ±1% relative to VNINDEX over 5 days.

**Model:** XGBoost classifier (depth=3, lr=0.05, n=500) trained cross-sectionally
on all available stocks with expanding-window walk-forward.

**Portfolio:** Top-5 stocks by predicted probability, square weighting,
VIX filter [15, 30], MIN_PROBA ≥ 0.55, E=1 H=1 execution.

**Cost:** 0.30% round-trip assumed.

## Reproducibility Notes

- Data is not committed (git-ignored). Re-fetch with `python main.py` or Step 1.
- All results are deterministic (random_state=42 in XGBoost).
- The 2026 pre-committed expectations were documented in
  `notes/session_23_2026_holdout.md` before the holdout was run.
- Walk-forward ensures zero look-ahead: test year data is never seen during training.

## Dependencies

- Python 3.11
- vnstock==3.4.2
- xgboost>=2.0.0
- pandas>=2.0.0, numpy>=1.24.0, scipy>=1.10.0
- matplotlib>=3.7.0, pyarrow>=14.0.0
- yfinance>=0.2.40

## License

MIT
