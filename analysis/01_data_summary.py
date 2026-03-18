"""
01_data_summary.py — Universe coverage, date ranges, missing data statistics.

Saves: output/results/data_summary.json
"""
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import STOCK_DIR, VNINDEX_FILE, RESULTS_DIR, safe_print, ensure_dirs


def run():
    ensure_dirs()
    safe_print("\n--- 01: Data Summary ---")

    files = sorted(glob.glob(str(STOCK_DIR / "gia_lich_su_*_1D.parquet")))
    tickers = [Path(f).stem.replace("gia_lich_su_", "").replace("_1D", "")
               for f in files]
    tickers = [t for t in tickers if t not in ("VNINDEX", "VN30")]

    records = []
    for f, ticker in zip(files, tickers):
        if ticker in ("VNINDEX", "VN30"):
            continue
        try:
            df = pd.read_parquet(f, columns=["time", "close"])
            df["time"] = pd.to_datetime(df["time"])
            records.append({
                "ticker":   ticker,
                "n_rows":   len(df),
                "date_min": str(df["time"].min().date()),
                "date_max": str(df["time"].max().date()),
            })
        except Exception:
            pass

    all_df = pd.DataFrame(records)
    safe_print(f"  Total tickers: {len(all_df)}")
    safe_print(f"  Median rows:   {all_df['n_rows'].median():.0f}")
    safe_print(f"  Min date:      {all_df['date_min'].min()}")
    safe_print(f"  Max date:      {all_df['date_max'].max()}")

    # Per-year obs count (from VNINDEX for consistency)
    vni = pd.read_parquet(VNINDEX_FILE, columns=["time"])
    vni["time"] = pd.to_datetime(vni["time"])
    year_counts = vni.groupby(vni["time"].dt.year).size().reset_index(name="n_days")

    # Stocks available each year (n_rows >= 50 that year)
    year_stocks = {}
    for rec in records:
        for y in range(2012, 2027):
            if rec["date_min"] <= f"{y}-06-30" and rec["date_max"] >= f"{y}-01-01":
                year_stocks[y] = year_stocks.get(y, 0) + 1

    safe_print(f"\n  Year   N_stocks  VNI_days")
    safe_print("  " + "-" * 30)
    for row in year_counts.itertuples():
        y = int(row.time)
        if y < 2012:
            continue
        safe_print(f"  {y:>5}  {year_stocks.get(y, 0):>8}  {row.n_days:>8}")

    summary = {
        "n_tickers":     len(all_df),
        "median_rows":   int(all_df["n_rows"].median()),
        "date_min":      all_df["date_min"].min(),
        "date_max":      all_df["date_max"].max(),
        "year_stocks":   year_stocks,
        "tickers_gt300": int((all_df["n_rows"] >= 300).sum()),
    }
    out = RESULTS_DIR / "data_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    safe_print(f"\n  Saved {out}")
    return summary


if __name__ == "__main__":
    run()
