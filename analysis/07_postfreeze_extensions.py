"""
07_postfreeze_extensions.py - supplementary Session 26 credibility checks.

These checks are not part of the frozen paper pipeline's main claims, but they
strengthen execution realism and the interpretation of the 2026 evidence:

1. Executable no-VIX holding-period grid (H=1/2/3/5)
2. No-VIX cost stress for daily H=1 vs weekly H=5
3. 2026 no-VIX live-validation rerun with saved probabilities

Saves:
- output/results/postfreeze_extensions.json
- output/results/holdout_2026_novix_probas.parquet
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import FEATURE_COLS, TARGET_COL, load_all_stocks
from src.model import _clean, _get_model
from src.utils import (
    COST_RT,
    MIN_PROBA,
    RESULTS_DIR,
    START_DATE,
    STOCK_DIR,
    TOP_N,
    TRAIN_END,
    VNINDEX_FILE,
    ensure_dirs,
    safe_print,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

TEST_YEAR = 2026
HOLD_PERIODS = (1, 2, 3, 5)
START_YEAR = 2015


def _load_vni() -> pd.DataFrame:
    df = pd.read_parquet(VNINDEX_FILE, columns=["time", "open"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _load_prices_e1(tickers: list[str], holds: tuple[int, ...]) -> pd.DataFrame:
    """Compute open-to-open excess returns keyed by signal date for each hold."""
    vni = _load_vni()
    vni_open = vni["open"].values.astype(float)
    vni_times = vni["time"].values
    vni_idx = {t: i for i, t in enumerate(vni_times)}

    ticker_set = set(tickers)
    records = []

    for stock_file in sorted(STOCK_DIR.glob("gia_lich_su_*_1D.parquet")):
        ticker = stock_file.name.replace("gia_lich_su_", "").replace("_1D.parquet", "")
        if ticker not in ticker_set or ticker == "VNINDEX":
            continue
        try:
            df = pd.read_parquet(stock_file, columns=["time", "open"])
        except Exception:
            continue

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["ticker"] = ticker

        open_arr = df["open"].values.astype(float)
        times_arr = df["time"].values
        n_rows = len(df)

        for hold in holds:
            col = f"fwd_h{hold}"
            arr = np.full(n_rows, np.nan)
            for i in range(n_rows):
                vni_pos = vni_idx.get(times_arr[i])
                if vni_pos is None:
                    continue
                stock_exit = i + 1 + hold
                vni_exit = vni_pos + 1 + hold
                if stock_exit >= n_rows or vni_exit >= len(vni_open):
                    continue
                p_in = open_arr[i + 1]
                p_out = open_arr[stock_exit]
                v_in = vni_open[vni_pos + 1]
                v_out = vni_open[vni_exit]
                if p_in <= 0 or v_in <= 0:
                    continue
                arr[i] = (p_out / p_in - 1.0) - (v_out / v_in - 1.0)
            df[col] = arr

        records.append(df[["time", "ticker"] + [f"fwd_h{h}" for h in holds]])

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _square_weights(probas: np.ndarray) -> np.ndarray:
    w = np.maximum(probas - 0.5, 1e-6) ** 2
    return w / w.sum()


def _build_signal_returns(merged: pd.DataFrame, hold: int) -> dict[int, list[float]]:
    """Non-overlapping signal returns for a fixed holding period, no VIX filter."""
    col = f"fwd_h{hold}"
    per_year: dict[int, list[float]] = {}

    for year, grp in merged.groupby(merged["time"].dt.year):
        if year < START_YEAR or year > TRAIN_END:
            continue
        dates = sorted(grp["time"].unique())
        last_signal_idx = -999
        signal_returns = []

        for di, date in enumerate(dates):
            if di - last_signal_idx < hold:
                continue
            day = grp[grp["time"] == date]
            top = day.nlargest(TOP_N, "xs_proba")
            if len(top) < TOP_N or top["xs_proba"].min() < MIN_PROBA:
                continue
            realized = top[col].dropna().values
            if len(realized) == 0:
                continue
            signal_returns.append(float(np.mean(realized)))
            last_signal_idx = di

        if signal_returns:
            per_year[int(year)] = signal_returns

    return per_year


def _summarize_trade_stream(per_year: dict[int, list[float]], cost_rt: float = COST_RT) -> dict | None:
    all_trades = [ret for yearly in per_year.values() for ret in yearly]
    if not all_trades:
        return None

    avg_gross_trade = float(np.mean(all_trades))
    yearly_gross = [float(np.sum(vals)) for vals in per_year.values()]
    yearly_counts = [len(vals) for vals in per_year.values()]
    yearly_net = [float(np.sum(vals) - len(vals) * cost_rt) for vals in per_year.values()]

    avg_gross_annual = float(np.mean(yearly_gross))
    avg_trades_yr = float(np.mean(yearly_counts))
    avg_net_annual = float(np.mean(yearly_net))
    breakeven = avg_gross_annual / avg_trades_yr if avg_trades_yr > 0 else None

    return {
        "avg_gross_trade": round(avg_gross_trade, 6),
        "avg_gross_annual": round(avg_gross_annual, 6),
        "avg_trades_yr": round(avg_trades_yr, 2),
        "avg_net_annual_0_30": round(avg_net_annual, 6),
        "exact_breakeven_rt": round(breakeven, 6) if breakeven is not None else None,
        "positive_years_at_0_30": int(sum(1 for val in yearly_net if val > 0)),
        "total_years": len(yearly_net),
    }


def run_holding_period_grid(probas: pd.DataFrame) -> dict:
    safe_print("\n  Supplement A: Executable holding-period grid (no VIX)")
    tickers = probas["ticker"].unique().tolist()
    prices = _load_prices_e1(tickers, HOLD_PERIODS)
    merged = probas.merge(prices, on=["time", "ticker"], how="left")

    rows = []
    for hold in HOLD_PERIODS:
        summary = _summarize_trade_stream(_build_signal_returns(merged, hold))
        if not summary:
            continue
        rows.append({"hold": hold, **summary})
        safe_print(
            f"    H={hold}: gross/trade={summary['avg_gross_trade']*100:+.3f}% | "
            f"ann_net@0.30={summary['avg_net_annual_0_30']*100:+.2f}% | "
            f"breakeven={summary['exact_breakeven_rt']*100:.2f}%"
        )

    best = max(rows, key=lambda row: row["avg_net_annual_0_30"], default=None)
    return {
        "best_hold_by_ann_net_0_30": best["hold"] if best else None,
        "rows": rows,
    }


def run_cost_stress_no_vix(probas: pd.DataFrame) -> dict:
    safe_print("\n  Supplement B: No-VIX cost stress")
    tickers = probas["ticker"].unique().tolist()
    prices = _load_prices_e1(tickers, (1, 5))
    merged = probas.merge(prices, on=["time", "ticker"], how="left")
    daily = _summarize_trade_stream(_build_signal_returns(merged, 1), cost_rt=COST_RT)
    weekly = _summarize_trade_stream(_build_signal_returns(merged, 5), cost_rt=COST_RT)

    if daily:
        safe_print(
            f"    Daily H=1 breakeven={daily['exact_breakeven_rt']*100:.2f}% RT | "
            f"ann_net@0.30={daily['avg_net_annual_0_30']*100:+.2f}%"
        )
    if weekly:
        safe_print(
            f"    Weekly H=5 breakeven={weekly['exact_breakeven_rt']*100:.2f}% RT | "
            f"ann_net@0.30={weekly['avg_net_annual_0_30']*100:+.2f}%"
        )

    return {"daily_h1": daily, "weekly_h5": weekly}


def _compute_daily_ics(df: pd.DataFrame) -> list[float]:
    ics = []
    for _, day in df.groupby("time"):
        if len(day) < 5:
            continue
        ic, _ = spearmanr(day["xs_proba"], day["actual"])
        if not np.isnan(ic):
            ics.append(float(ic))
    return ics


def run_holdout_no_vix() -> dict:
    safe_print("\n  Supplement C: 2026 no-VIX live validation rerun")
    all_df = load_all_stocks(start_date=START_DATE)
    all_df["_year"] = pd.to_datetime(all_df["time"]).dt.year

    feat_cols = [c for c in FEATURE_COLS if c in all_df.columns]
    train_df = all_df[all_df["_year"] < TEST_YEAR].copy()
    test_df = all_df[all_df["_year"] == TEST_YEAR].copy()

    x_train, y_train, _ = _clean(train_df[feat_cols], train_df[TARGET_COL])
    x_test, y_test, valid_test = _clean(test_df[feat_cols], test_df[TARGET_COL])
    test_clean = test_df[valid_test.values].copy()

    n_val = max(500, int(len(x_train) * 0.12))
    x_tr, y_tr = x_train.iloc[:-n_val], y_train.iloc[:-n_val]
    x_val, y_val = x_train.iloc[-n_val:], y_train.iloc[-n_val:]

    model = _get_model()
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
    probas = model.predict_proba(x_test)[:, 1]

    probas_2026 = pd.DataFrame(
        {
            "time": pd.to_datetime(test_clean["time"].values),
            "year": TEST_YEAR,
            "ticker": test_clean["ticker"].values,
            "xs_proba": probas.round(4),
            "actual": y_test.values.astype(int),
        }
    ).sort_values(["time", "ticker"]).reset_index(drop=True)

    probas_path = RESULTS_DIR / "holdout_2026_novix_probas.parquet"
    probas_2026.to_parquet(probas_path, index=False)

    ic_days = _compute_daily_ics(probas_2026)
    ic_2026 = float(np.mean(ic_days)) if ic_days else float("nan")

    prices = _load_prices_e1(probas_2026["ticker"].unique().tolist(), (1,))
    merged = probas_2026.merge(prices, on=["time", "ticker"], how="left")

    ew_trades = []
    sq_trades = []
    for _, day in merged.groupby("time"):
        top = day.nlargest(TOP_N, "xs_proba")
        if len(top) < TOP_N or top["xs_proba"].min() < MIN_PROBA:
            continue
        valid = top.dropna(subset=["fwd_h1"])
        if len(valid) < 2:
            continue
        rets = valid["fwd_h1"].values
        probs = valid["xs_proba"].values
        ew_trades.append(float(np.mean(rets)))
        sq_trades.append(float((_square_weights(probs) * rets).sum()))

    def trade_summary(values: list[float]) -> dict | None:
        if not values:
            return None
        gross = float(np.mean(values))
        return {
            "trades": len(values),
            "gross_per_trade": round(gross, 6),
            "net_per_trade": round(gross - COST_RT, 6),
        }

    eq = trade_summary(ew_trades)
    sq = trade_summary(sq_trades)
    safe_print(
        f"    IC={ic_2026:+.4f} | EW gross={eq['gross_per_trade']*100:+.3f}% | "
        f"SQ gross={sq['gross_per_trade']*100:+.3f}%"
    )

    verdict = "fail"
    if not np.isnan(ic_2026):
        if ic_2026 >= 0.10 and sq and sq["gross_per_trade"] > 0:
            verdict = "weak_pass" if sq["net_per_trade"] < 0 else "pass"
        elif ic_2026 >= 0.05:
            verdict = "weak_pass"

    return {
        "ic_2026": round(ic_2026, 6) if not np.isnan(ic_2026) else None,
        "n_ic_days": len(ic_days),
        "n_tickers": int(probas_2026["ticker"].nunique()),
        "n_days": int(probas_2026["time"].nunique()),
        "date_min": probas_2026["time"].min().strftime("%Y-%m-%d"),
        "date_max": probas_2026["time"].max().strftime("%Y-%m-%d"),
        "equal_weight": eq,
        "square_weight": sq,
        "verdict": verdict,
        "probas_path": str(probas_path),
    }


def run():
    ensure_dirs()
    safe_print("\n--- 07: Post-Freeze Credibility Extensions ---")

    probas_path = RESULTS_DIR / "xs_probas.parquet"
    if not probas_path.exists():
        safe_print(f"  [ERROR] {probas_path} not found - run Step 3 first.")
        return None

    probas = pd.read_parquet(probas_path)
    probas["time"] = pd.to_datetime(probas["time"])

    holding_grid = run_holding_period_grid(probas)
    cost_stress = run_cost_stress_no_vix(probas)
    holdout_novix = run_holdout_no_vix()

    payload = {
        "holding_period_grid_no_vix": holding_grid,
        "cost_stress_no_vix": cost_stress,
        "holdout_2026_no_vix": holdout_novix,
    }

    out = RESULTS_DIR / "postfreeze_extensions.json"
    out.write_text(json.dumps(payload, indent=2))
    safe_print(f"\n  Saved {out}")
    return payload


if __name__ == "__main__":
    run()
