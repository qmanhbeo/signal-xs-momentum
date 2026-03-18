"""
src/features.py — Feature engineering for the momentum paper pipeline.

Produces the exact 35 production features used in the walk-forward model:

Group                   Features
────────────────────────────────────────────────────────
MA distance (4)         ma_dist_{5,10,20,50}
RSI (1)                 rsi_14
MACD (3)                macd_line, macd_signal_line, macd_hist
Bollinger (2)           bb_width, bb_pct
ATR (1)                 atr_14
Candle structure (4)    candle_body, upper_wick, lower_wick, gap
Stochastic (2)          stoch_k, stoch_d
ADX (2)                 adx, di_diff
Realized vol (4)        hist_vol_{5,10,20}, intraday_range
Rel return (6)          rel_ret_{1,2,3,5,10,20}d
Market return (6)       idx_ret_{1,2,3,5,10,20}d
────────────────────────────────────────────────────────
Total                   35

Target: future_rel_barrier_5d — triple barrier ±1% relative to VNINDEX over 5 days.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils import VNINDEX_FILE, BARRIER_PCT, HORIZON, START_DATE


FEATURE_COLS = (
    ["ma_dist_5", "ma_dist_10", "ma_dist_20", "ma_dist_50"]
    + ["rsi_14"]
    + ["macd_line", "macd_signal_line", "macd_hist"]
    + ["bb_width", "bb_pct"]
    + ["atr_14"]
    + ["candle_body", "upper_wick", "lower_wick", "gap"]
    + ["stoch_k", "stoch_d"]
    + ["adx", "di_diff"]
    + ["hist_vol_5", "hist_vol_10", "hist_vol_20", "intraday_range"]
    + [f"rel_ret_{l}d" for l in [1, 2, 3, 5, 10, 20]]
    + [f"idx_ret_{l}d" for l in [1, 2, 3, 5, 10, 20]]
)

TARGET_COL = f"future_rel_barrier_{HORIZON}d"
REL_LAGS   = [1, 2, 3, 5, 10, 20]


# ── Individual feature builders ───────────────────────────────────────────────

def _ma_dist(df: pd.DataFrame) -> pd.DataFrame:
    for w in [5, 10, 20, 50]:
        df[f"ma_dist_{w}"] = df["close"] / df["close"].rolling(w).mean() - 1
    return df


def _rsi(df: pd.DataFrame) -> pd.DataFrame:
    p = 14
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=p - 1, min_periods=p).mean()
    al    = loss.ewm(com=p - 1, min_periods=p).mean()
    rs    = ag / al.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    return df


def _macd(df: pd.DataFrame) -> pd.DataFrame:
    fast, slow, sig = 12, 26, 9
    ef   = df["close"].ewm(span=fast, adjust=False).mean()
    es   = df["close"].ewm(span=slow, adjust=False).mean()
    line = (ef - es) / df["close"]
    df["macd_line"]        = line
    df["macd_signal_line"] = line.ewm(span=sig, adjust=False).mean()
    df["macd_hist"]        = line - df["macd_signal_line"]
    return df


def _bb(df: pd.DataFrame) -> pd.DataFrame:
    p, ns = 20, 2
    ma  = df["close"].rolling(p).mean()
    std = df["close"].rolling(p).std()
    upper = ma + ns * std
    lower = ma - ns * std
    df["bb_width"] = (upper - lower) / ma
    df["bb_pct"]   = (df["close"] - lower) / (upper - lower)
    return df


def _atr(df: pd.DataFrame) -> pd.DataFrame:
    p  = 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=p - 1, min_periods=p).mean() / df["close"]
    return df


def _candle(df: pd.DataFrame) -> pd.DataFrame:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body"] = (df["close"] - df["open"]) / hl
    df["upper_wick"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / hl
    df["lower_wick"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / hl
    df["gap"]         = df["open"] / df["close"].shift(1) - 1
    return df


def _stoch(df: pd.DataFrame) -> pd.DataFrame:
    p  = 14
    hh = df["high"].rolling(p).max()
    ll = df["low"].rolling(p).min()
    df["stoch_k"] = (df["close"] - ll) / (hh - ll).replace(0, np.nan) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


def _adx(df: pd.DataFrame) -> pd.DataFrame:
    p    = 14
    hl   = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr  = tr.ewm(com=p - 1, min_periods=p).mean()
    pdi  = pdm.ewm(com=p - 1, min_periods=p).mean() / atr * 100
    mdi  = mdm.ewm(com=p - 1, min_periods=p).mean() / atr * 100
    dx   = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    df["adx"]     = dx.ewm(com=p - 1, min_periods=p).mean() / 100
    df["di_diff"] = (pdi - mdi) / 100
    return df


def _hist_vol(df: pd.DataFrame) -> pd.DataFrame:
    daily_ret = df["close"] / df["close"].shift(1) - 1
    for w in [5, 10, 20]:
        df[f"hist_vol_{w}"] = daily_ret.rolling(w).std()
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
    return df


def _market_relative(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Add excess returns vs VNINDEX and raw VNINDEX returns."""
    merged = df.merge(idx.rename(columns={"close": "_idx"}), on="time", how="left")
    merged["_idx"] = merged["_idx"].ffill()
    for lag in REL_LAGS:
        fpt = merged["close"] / merged["close"].shift(lag) - 1
        ix  = merged["_idx"]  / merged["_idx"].shift(lag)  - 1
        merged[f"rel_ret_{lag}d"] = fpt - ix
        merged[f"idx_ret_{lag}d"] = ix
    # Put new columns back into df (keep df index intact)
    for col in [f"rel_ret_{l}d" for l in REL_LAGS] + [f"idx_ret_{l}d" for l in REL_LAGS]:
        df[col] = merged[col].values
    return df


def _triple_barrier_target(df: pd.DataFrame, idx: pd.DataFrame,
                            barrier_pct: float = BARRIER_PCT,
                            horizon: int = HORIZON) -> pd.DataFrame:
    """
    Triple barrier label relative to VNINDEX.
      1 if stock excess return hits +barrier_pct within horizon days
      0 if it hits -barrier_pct first
      fixed-time direction on timeout (no NaN rows dropped by design)
    """
    merged = df.merge(idx.rename(columns={"close": "_idx"}), on="time", how="left")
    merged["_idx"] = merged["_idx"].ffill()

    fpt_arr = merged["close"].values
    idx_arr = merged["_idx"].values
    n       = len(merged)
    labels  = np.full(n, np.nan)

    for i in range(n - horizon):
        ef = fpt_arr[i]
        ei = idx_arr[i]
        if ef <= 0 or ei <= 0:
            continue
        label = np.nan
        for j in range(1, horizon + 1):
            rel = fpt_arr[i + j] / ef - idx_arr[i + j] / ei
            if rel >= barrier_pct:
                label = 1.0
                break
            elif rel <= -barrier_pct:
                label = 0.0
                break
        if np.isnan(label):
            rel_h = fpt_arr[i + horizon] / ef - idx_arr[i + horizon] / ei
            label = float(rel_h > 0)
        labels[i] = label

    df[TARGET_COL] = labels
    return df


# ── Public interface ───────────────────────────────────────────────────────────

def build_features_for_ticker(df_raw: pd.DataFrame,
                               vnindex: Optional[pd.DataFrame] = None,
                               barrier_pct: float = BARRIER_PCT,
                               horizon: int = HORIZON) -> pd.DataFrame:
    """
    Apply all 35 features + triple-barrier target to a single ticker's raw OHLCV.
    df_raw must have columns: time, open, high, low, close, volume.
    Returns a clean DataFrame with FEATURE_COLS + TARGET_COL (NaN rows dropped).
    """
    if vnindex is None:
        vnindex = pd.read_parquet(VNINDEX_FILE)[["time", "close"]]
        vnindex["time"] = pd.to_datetime(vnindex["time"])

    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Technical features
    df = _ma_dist(df)
    df = _rsi(df)
    df = _macd(df)
    df = _bb(df)
    df = _atr(df)
    df = _candle(df)
    df = _stoch(df)
    df = _adx(df)
    df = _hist_vol(df)
    df = _market_relative(df, vnindex)

    # Target
    df = _triple_barrier_target(df, vnindex, barrier_pct, horizon)

    # Drop rows with NaN in features or target
    keep = FEATURE_COLS + [TARGET_COL, "time"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna().reset_index(drop=True)
    return df


def load_all_stocks(start_date: str = START_DATE,
                    min_rows: int = 300) -> pd.DataFrame:
    """
    Load + feature-engineer all 393 HOSE stocks.
    Returns a single DataFrame with columns: time, ticker, <features>, <target>.
    """
    from pathlib import Path
    from src.utils import STOCK_DIR
    import glob

    vnindex = pd.read_parquet(VNINDEX_FILE)[["time", "close"]]
    vnindex["time"] = pd.to_datetime(vnindex["time"])

    files  = sorted(glob.glob(str(STOCK_DIR / "gia_lich_su_*_1D.parquet")))
    frames = []
    skipped = []
    for f in files:
        ticker = Path(f).stem.replace("gia_lich_su_", "").replace("_1D", "")
        if ticker in ("VNINDEX", "VN30"):
            continue
        try:
            raw = pd.read_parquet(f)
            raw["time"] = pd.to_datetime(raw["time"])
            raw = raw[raw["time"] >= start_date].reset_index(drop=True)
            if len(raw) < min_rows:
                skipped.append(ticker)
                continue
            feat = build_features_for_ticker(raw, vnindex)
            if len(feat) < 100:
                skipped.append(ticker)
                continue
            feat["ticker"] = ticker
            frames.append(feat)
        except Exception as e:
            skipped.append(ticker)
    if not frames:
        raise RuntimeError("No stock data found — run Step 1 (fetch) first.")
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(frames)} stocks, {len(all_df):,} rows "
          f"({len(skipped)} skipped, min_rows={min_rows})")
    return all_df
