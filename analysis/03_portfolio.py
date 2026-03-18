"""
03_portfolio.py — Top-5 square-weighted portfolio simulation 2015–2025.

Reads xs_probas.parquet + raw OHLC data.
Saves: output/results/portfolio_annual.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import PROBAS_FILE, RESULTS_DIR, safe_print, ensure_dirs
from src.portfolio import run_portfolio


def run():
    ensure_dirs()
    safe_print("\n--- 03: Portfolio Simulation (E=1 H=1, square weighting) ---")

    if not PROBAS_FILE.exists():
        safe_print(f"  [ERROR] {PROBAS_FILE} not found — run Step 3 first.")
        return None

    probas = pd.read_parquet(PROBAS_FILE)
    result = run_portfolio(
        probas=probas,
        entry_lag=1,
        out_path=RESULTS_DIR / "trades_e1h1.csv",
        start_year=2015,
    )

    out = RESULTS_DIR / "portfolio_annual.json"
    out.write_text(json.dumps(result["annual"], indent=2))
    safe_print(f"  Saved {out}")
    return result


if __name__ == "__main__":
    run()
