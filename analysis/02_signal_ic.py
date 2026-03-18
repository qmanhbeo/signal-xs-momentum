"""
02_signal_ic.py — IC walk-forward 2015–2025 (main result).

Reads xs_probas.parquet (from Step 3 / src/model.py) and computes:
  - Daily Spearman IC per year
  - Annual mean IC ± SE
  - t-statistic and p-value per year

Saves: output/results/ic_by_year.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp

from src.utils import PROBAS_FILE, RESULTS_DIR, safe_print, ensure_dirs


def compute_ic(probas: pd.DataFrame) -> list:
    """Return list of {year, ic_mean, ic_se, n_days, n_tickers, t_stat, p_val}."""
    rows = []
    for year in sorted(probas["year"].unique()):
        yr = probas[probas["year"] == year]
        daily_ics = []
        for _, day in yr.groupby("time"):
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["xs_proba"], day["actual"])
            if not np.isnan(ic):
                daily_ics.append(ic)
        if not daily_ics:
            continue
        arr     = np.array(daily_ics)
        ic_mean = float(np.mean(arr))
        ic_se   = float(np.std(arr) / np.sqrt(len(arr)))
        t_stat  = float(ic_mean / (ic_se + 1e-9))
        p_val   = float(ttest_1samp(arr, 0).pvalue)
        rows.append({
            "year":       int(year),
            "ic_mean":    round(ic_mean, 4),
            "ic_se":      round(ic_se, 4),
            "n_days":     len(daily_ics),
            "n_tickers":  int(yr["ticker"].nunique()),
            "t_stat":     round(t_stat, 2),
            "p_val":      round(p_val, 4),
        })
    return rows


def run():
    ensure_dirs()
    safe_print("\n--- 02: Signal IC by Year ---")

    if not PROBAS_FILE.exists():
        safe_print(f"  [ERROR] {PROBAS_FILE} not found — run Step 3 first.")
        return None

    probas = pd.read_parquet(PROBAS_FILE)
    probas["time"] = pd.to_datetime(probas["time"])

    ic_rows = compute_ic(probas)

    safe_print(f"\n  {'year':>5}  {'IC':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  {'n_days':>7}  {'n_tickers':>10}")
    safe_print("  " + "-" * 60)
    for r in ic_rows:
        sig = "*" if r["p_val"] < 0.05 else " "
        safe_print(f"  {r['year']:>5}  {r['ic_mean']:>+8.4f}  {r['ic_se']:>8.4f}  "
                   f"{r['t_stat']:>6.1f}  {r['p_val']:>7.4f}{sig}  "
                   f"{r['n_days']:>7}  {r['n_tickers']:>10}")

    avg_ic = np.mean([r["ic_mean"] for r in ic_rows])
    n_sig  = sum(1 for r in ic_rows if r["p_val"] < 0.05)
    safe_print(f"\n  Average IC: {avg_ic:+.4f}")
    safe_print(f"  Sig years (p<0.05): {n_sig}/{len(ic_rows)}")

    out = RESULTS_DIR / "ic_by_year.json"
    out.write_text(json.dumps(ic_rows, indent=2))
    safe_print(f"  Saved {out}")
    return ic_rows


if __name__ == "__main__":
    run()
