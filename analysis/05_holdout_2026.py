"""
05_holdout_2026.py — 2026 out-of-sample holdout test.

Pre-committed expectations (from session_23_2026_holdout.md, set before running):
  IC:          [0.10, 0.13]  | FAIL if < 0.05
  net/trade:   [+0.00%, +0.25%] | FAIL if < -0.20%

Pipeline:
  1. Load all 393 stocks + features
  2. Train XGBoost on years < 2026 (same config as src/model.py)
  3. Score year == 2026
  4. Compute IC and portfolio vs expectations

Saves: output/results/holdout_2026.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from src.utils import (STOCK_DIR, VIX_FILE, VNINDEX_FILE, RESULTS_DIR,
                       START_DATE, TOP_N, MIN_PROBA, VIX_MIN, VIX_MAX,
                       COST_RT, safe_print, ensure_dirs)

warnings.filterwarnings("ignore")

TEST_YEAR = 2026
MIN_ROWS  = 300

# Pre-committed expectations
EXPECTED_IC_LOW   = 0.10
EXPECTED_IC_HIGH  = 0.13
EXPECTED_NET_LOW  = 0.000
EXPECTED_NET_HIGH = 0.0025
FAIL_IC           = 0.05
FAIL_NET          = -0.002


def _load_vni():
    df = pd.read_parquet(VNINDEX_FILE, columns=["time", "open"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _load_prices_e1h1(tickers, vni):
    vni_open  = vni["open"].values.astype(float)
    vni_times = vni["time"].values
    vni_idx   = {t: i for i, t in enumerate(vni_times)}
    ticker_set = set(tickers)
    records = []
    for sf in sorted(STOCK_DIR.glob("gia_lich_su_*_1D.parquet")):
        ticker = sf.name.replace("gia_lich_su_", "").replace("_1D.parquet", "")
        if ticker not in ticker_set or ticker == "VNINDEX":
            continue
        try:
            df = pd.read_parquet(sf, columns=["time", "open"])
        except Exception:
            continue
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["ticker"] = ticker
        open_arr  = df["open"].values.astype(float)
        times_arr = df["time"].values
        n = len(df)
        ret_arr = np.full(n, np.nan)
        for i in range(n):
            vi = vni_idx.get(times_arr[i])
            if vi is None or i + 2 >= n or vi + 2 >= len(vni_open):
                continue
            p_in, p_out = open_arr[i + 1], open_arr[i + 2]
            v_in, v_out = vni_open[vi + 1], vni_open[vi + 2]
            if p_in <= 0 or v_in <= 0:
                continue
            ret_arr[i] = (p_out / p_in - 1.0) - (v_out / v_in - 1.0)
        df["fwd_e1_h1"] = ret_arr
        records.append(df[["time", "ticker", "fwd_e1_h1"]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _square_weights(probas):
    w = np.maximum(probas - 0.5, 1e-6) ** 2
    return w / w.sum()


def run():
    ensure_dirs()
    safe_print("\n--- 05: 2026 Out-of-Sample Holdout ---")
    safe_print(f"Pre-committed expectations:")
    safe_print(f"  IC:        [{EXPECTED_IC_LOW:.2f}, {EXPECTED_IC_HIGH:.2f}]  | FAIL < {FAIL_IC:.2f}")
    safe_print(f"  net/trade: [{EXPECTED_NET_LOW*100:+.2f}%, {EXPECTED_NET_HIGH*100:+.2f}%] | "
               f"FAIL < {FAIL_NET*100:.2f}%")

    from src.features import load_all_stocks, FEATURE_COLS, TARGET_COL
    from src.model import _get_model, _clean

    safe_print(f"\nLoading features for all stocks (train<{TEST_YEAR}, test={TEST_YEAR})...")
    all_df = load_all_stocks(start_date=START_DATE, min_rows=MIN_ROWS)
    all_df["_year"] = pd.to_datetime(all_df["time"]).dt.year

    feat_cols = [c for c in FEATURE_COLS if c in all_df.columns]
    target    = TARGET_COL

    train_df = all_df[all_df["_year"] < TEST_YEAR].copy()
    test_df  = all_df[all_df["_year"] == TEST_YEAR].copy()

    safe_print(f"  Train: {len(train_df):,} rows  Test: {len(test_df):,} rows "
               f"({test_df['ticker'].nunique()} tickers)")

    if len(test_df) < 50:
        safe_print(f"\n[ERROR] Only {len(test_df)} rows for {TEST_YEAR}. "
                   f"Fetch more data for 2026 first.")
        return None

    X_tr, y_tr, _ = _clean(train_df[feat_cols], train_df[target])
    X_te, y_te, valid_te = _clean(test_df[feat_cols], test_df[target])
    test_clean = test_df[valid_te.values]

    n_val = max(500, int(len(X_tr) * 0.12))
    X_t, y_t = X_tr.iloc[:-n_val], y_tr.iloc[:-n_val]
    X_v, y_v = X_tr.iloc[-n_val:], y_tr.iloc[-n_val:]

    safe_print(f"\nTraining XGBoost on {len(X_tr):,} rows ...")
    model = _get_model()
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
    safe_print("  Training complete.")

    probas_arr = model.predict_proba(X_te)[:, 1]
    actuals    = y_te.values.astype(int)

    probas_df = pd.DataFrame({
        "time":     pd.to_datetime(test_clean["time"].values),
        "ticker":   test_clean["ticker"].values,
        "xs_proba": probas_arr.round(4),
        "actual":   actuals,
    })

    n_tickers = probas_df["ticker"].nunique()
    n_days    = probas_df["time"].nunique()
    safe_print(f"\n  Scored: {n_tickers} tickers | {n_days} days | "
               f"{probas_df['time'].min().date()} to {probas_df['time'].max().date()}")

    # IC
    daily_ics = []
    for _, day in probas_df.groupby("time"):
        if len(day) < 5:
            continue
        ic, _ = spearmanr(day["xs_proba"], day["actual"])
        if not np.isnan(ic):
            daily_ics.append(ic)

    ic_2026 = float(np.mean(daily_ics)) if daily_ics else float("nan")
    safe_print(f"\n  IC_2026 = {ic_2026:+.4f}  (expect [{EXPECTED_IC_LOW:.2f}, {EXPECTED_IC_HIGH:.2f}])")
    if not np.isnan(ic_2026):
        if ic_2026 >= EXPECTED_IC_LOW:
            ic_verdict = "IN-RANGE"
        elif ic_2026 >= FAIL_IC:
            ic_verdict = "BELOW-RANGE"
        else:
            ic_verdict = "FAIL"
        safe_print(f"  IC verdict: {ic_verdict}")

    # Portfolio
    vix = pd.read_parquet(VIX_FILE)[["time", "close"]].rename(columns={"close": "vix"})
    vix["time"] = pd.to_datetime(vix["time"])
    merged = probas_df.merge(vix, on="time", how="left")
    merged["vix"] = merged["vix"].ffill()

    tickers_2026 = probas_df["ticker"].unique().tolist()
    vni    = _load_vni()
    prices = _load_prices_e1h1(tickers_2026, vni)
    merged = merged.merge(prices, on=["time", "ticker"], how="left")

    sq_trades = []
    for date, day in merged.groupby("time"):
        vix_val = day["vix"].iloc[0]
        if not (VIX_MIN <= vix_val <= VIX_MAX):
            continue
        top = day.nlargest(TOP_N, "xs_proba")
        if len(top) < TOP_N or top["xs_proba"].min() < MIN_PROBA:
            continue
        valid = top.dropna(subset=["fwd_e1_h1"])
        if len(valid) < 2:
            continue
        w = _square_weights(valid["xs_proba"].values)
        sq_trades.append(float((w * valid["fwd_e1_h1"].values).sum()))

    safe_print(f"\n  Trade days with signal: {len(sq_trades)}")

    net_verdict = "N/A"
    net_sq = float("nan")
    if sq_trades:
        gross_sq = float(np.mean(sq_trades))
        net_sq   = gross_sq - COST_RT
        safe_print(f"  Gross/trade: {gross_sq*100:>+.3f}%")
        safe_print(f"  Net/trade:   {net_sq*100:>+.3f}%  "
                   f"(expect [{EXPECTED_NET_LOW*100:+.2f}%, {EXPECTED_NET_HIGH*100:+.2f}%])")
        if net_sq >= EXPECTED_NET_LOW:
            net_verdict = "IN-RANGE"
        elif net_sq >= FAIL_NET:
            net_verdict = "BELOW-RANGE"
        else:
            net_verdict = "FAIL"
        safe_print(f"  Portfolio verdict: {net_verdict}")

    # Overall verdict
    ic_ok  = not np.isnan(ic_2026) and ic_2026 >= FAIL_IC
    net_ok = bool(sq_trades) and net_sq >= FAIL_NET
    if ic_ok and net_ok:
        if (ic_2026 >= EXPECTED_IC_LOW and net_sq >= EXPECTED_NET_LOW):
            overall = "STRONG PASS"
        else:
            overall = "WEAK PASS"
    else:
        overall = "FAIL"
    safe_print(f"\n  OVERALL: {overall}")

    result = {
        "ic_2026":        round(ic_2026, 4),
        "ic_verdict":     ic_verdict if not np.isnan(ic_2026) else "N/A",
        "net_sq":         round(net_sq, 4) if not np.isnan(net_sq) else None,
        "net_verdict":    net_verdict,
        "n_trade_days":   len(sq_trades),
        "n_tickers":      n_tickers,
        "n_days":         n_days,
        "overall_verdict":overall,
        "expected_ic":    [EXPECTED_IC_LOW, EXPECTED_IC_HIGH],
        "expected_net":   [EXPECTED_NET_LOW, EXPECTED_NET_HIGH],
    }
    out = RESULTS_DIR / "holdout_2026.json"
    out.write_text(json.dumps(result, indent=2))
    safe_print(f"\n  Saved {out}")
    return result


if __name__ == "__main__":
    run()
