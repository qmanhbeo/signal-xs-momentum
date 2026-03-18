"""
src/portfolio.py — Portfolio simulation and performance metrics.

Strategy:
  - Each trading day: rank all stocks by xs_proba
  - Long top-5 if ALL have proba >= MIN_PROBA AND VIX in [VIX_MIN, VIX_MAX]
  - Square weighting: w_i = (proba_i - 0.5)^2 / sum
  - E=1 H=1: entry at T+1 open, exit at T+2 open (excess vs VNINDEX)
  - Cost: COST_RT per round-trip

Returns: trade-level log, annual summary, Sharpe.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (STOCK_DIR, VIX_FILE, VNINDEX_FILE, PROBAS_FILE,
                       TOP_N, MIN_PROBA, VIX_MIN, VIX_MAX, COST_RT,
                       safe_print, ensure_dirs)


def _load_vni():
    df = pd.read_parquet(VNINDEX_FILE, columns=["time", "open"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _load_prices_e1h1(tickers: list, vni: pd.DataFrame) -> pd.DataFrame:
    """
    Load E=1 H=1 excess returns: enter T+1 open, exit T+2 open, vs VNINDEX.
    Returns (time, ticker, fwd_e1_h1).
    """
    vni_open  = vni["open"].values.astype(float)
    vni_times = vni["time"].values
    vni_idx   = {t: i for i, t in enumerate(vni_times)}

    ticker_set = set(tickers)
    records    = []

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


def _load_prices_e2h1(tickers: list, vni: pd.DataFrame) -> pd.DataFrame:
    """E=2 H=1: enter T+2 open, exit T+3 open."""
    vni_open  = vni["open"].values.astype(float)
    vni_times = vni["time"].values
    vni_idx   = {t: i for i, t in enumerate(vni_times)}
    ticker_set = set(tickers)
    records    = []
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
            if vi is None or i + 3 >= n or vi + 3 >= len(vni_open):
                continue
            p_in, p_out = open_arr[i + 2], open_arr[i + 3]
            v_in, v_out = vni_open[vi + 2], vni_open[vi + 3]
            if p_in <= 0 or v_in <= 0:
                continue
            ret_arr[i] = (p_out / p_in - 1.0) - (v_out / v_in - 1.0)
        df["fwd_e2_h1"] = ret_arr
        records.append(df[["time", "ticker", "fwd_e2_h1"]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _square_weights(probas: np.ndarray) -> np.ndarray:
    w = np.maximum(probas - 0.5, 1e-6) ** 2
    return w / w.sum()


def run_portfolio(probas: pd.DataFrame = None,
                  entry_lag: int = 1,
                  out_path: Path = None,
                  start_year: int = 2015,
                  end_year: int = None,
                  force: bool = False) -> dict:
    """
    Simulate square-weighted top-N portfolio.

    Parameters
    ----------
    probas      : DataFrame with (time, ticker, xs_proba, actual). Loaded from
                  PROBAS_FILE if None.
    entry_lag   : 1 = E=1 H=1 (default), 2 = E=2 H=1.
    out_path    : where to save trade log (CSV). If None, not saved.
    start_year  : first year to simulate (default 2015).
    end_year    : last year to simulate (default: max year in probas).

    Returns dict with keys:
      trades      : list of per-day gross excess returns
      annual      : list of per-year {year, gross, net, n_trades}
      sharpe_net  : Sharpe on annual net returns
      avg_gross   : average gross per trade
      avg_net     : average net per trade
    """
    ensure_dirs()

    if probas is None:
        probas = pd.read_parquet(PROBAS_FILE)

    probas = probas.copy()
    probas["time"] = pd.to_datetime(probas["time"])

    # Load VIX
    vix = pd.read_parquet(VIX_FILE)[["time", "close"]].rename(columns={"close": "vix"})
    vix["time"] = pd.to_datetime(vix["time"])
    probas = probas.merge(vix, on="time", how="left")
    probas["vix"] = probas["vix"].ffill()

    tickers = probas["ticker"].unique().tolist()
    vni     = _load_vni()

    safe_print(f"  Loading E={entry_lag} H=1 prices for {len(tickers)} tickers ...")
    if entry_lag == 1:
        prices = _load_prices_e1h1(tickers, vni)
        fwd_col = "fwd_e1_h1"
    else:
        prices = _load_prices_e2h1(tickers, vni)
        fwd_col = "fwd_e2_h1"

    merged = probas.merge(prices, on=["time", "ticker"], how="left")

    if end_year is None:
        end_year = int(merged["time"].dt.year.max())

    ew_all  = []
    sq_all  = []
    annual  = []

    for year in range(start_year, end_year + 1):
        yr_df = merged[merged["time"].dt.year == year]
        ew_yr, sq_yr = [], []

        for date, day in yr_df.groupby("time"):
            vix_val = day["vix"].iloc[0]
            if not (VIX_MIN <= vix_val <= VIX_MAX):
                continue
            top = day.nlargest(TOP_N, "xs_proba")
            if len(top) < TOP_N or top["xs_proba"].min() < MIN_PROBA:
                continue
            valid = top.dropna(subset=[fwd_col])
            if len(valid) < 2:
                continue
            rets   = valid[fwd_col].values
            pr     = valid["xs_proba"].values
            ew_yr.append(float(rets.mean()))
            sq_yr.append(float((_square_weights(pr) * rets).sum()))

        if not sq_yr:
            continue

        gross_sq = np.mean(sq_yr)
        net_sq   = gross_sq - COST_RT
        annual.append({
            "year":     year,
            "gross_ew": float(np.mean(ew_yr)),
            "net_ew":   float(np.mean(ew_yr) - COST_RT),
            "gross_sq": gross_sq,
            "net_sq":   net_sq,
            "n_trades": len(sq_yr),
        })
        ew_all.extend(ew_yr)
        sq_all.extend(sq_yr)

    if not annual:
        return {"trades": [], "annual": [], "sharpe_net": 0.0,
                "avg_gross": 0.0, "avg_net": 0.0}

    ann_nets  = [r["net_sq"] for r in annual]
    sh_net    = float(np.mean(ann_nets) / (np.std(ann_nets) + 1e-9))
    avg_gross = float(np.mean([r["gross_sq"] for r in annual]))
    avg_net   = float(np.mean(ann_nets))

    # Print summary
    safe_print(f"\n  {'year':>5}  {'gross/trade':>12}  {'net/trade':>11}  {'trades':>7}")
    safe_print("  " + "-" * 45)
    for r in annual:
        safe_print(f"  {r['year']:>5}  {r['gross_sq']*100:>+11.3f}%  "
                   f"{r['net_sq']*100:>+10.3f}%  {r['n_trades']:>7}")
    safe_print(f"\n  Avg gross/trade: {avg_gross*100:>+.3f}%")
    safe_print(f"  Avg net/trade:   {avg_net*100:>+.3f}%")
    safe_print(f"  Sharpe (net):    {sh_net:.2f}")

    if out_path:
        pd.DataFrame({"gross": sq_all}).to_csv(out_path, index=False)

    return {
        "trades":      sq_all,
        "annual":      annual,
        "sharpe_net":  sh_net,
        "avg_gross":   avg_gross,
        "avg_net":     avg_net,
    }
