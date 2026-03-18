"""
src/model.py — Cross-sectional XGBoost walk-forward.

For each test_year in 2015..TRAIN_END:
  - Train on all stocks × all prior years
  - Score test year → save (time, ticker, xs_proba, actual) rows
  - Compute daily Spearman IC for that year

Outputs: output/results/xs_probas.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from src.utils import (PROBAS_FILE, TRAIN_END, START_DATE, safe_print, ensure_dirs)
from src.features import FEATURE_COLS, TARGET_COL


MODEL_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)


def _get_model():
    from xgboost import XGBClassifier
    return XGBClassifier(**MODEL_PARAMS)


def _clean(X: pd.DataFrame, y: pd.Series):
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    valid = y.notna() & np.isfinite(y) & (y.abs() < 10)
    return X[valid], y[valid], valid


def run_walk_forward(all_df: pd.DataFrame = None,
                     start_year: int = 2015,
                     train_end: int = TRAIN_END,
                     out_path: Path = None,
                     force: bool = False) -> pd.DataFrame:
    """
    Cross-sectional walk-forward 2015..train_end.
    Returns probas DataFrame (time, ticker, xs_proba, actual, year).
    Saves to out_path (default: output/results/xs_probas.parquet).
    """
    ensure_dirs()
    if out_path is None:
        out_path = PROBAS_FILE

    if not force and Path(out_path).exists():
        safe_print(f"  xs_probas already exists at {out_path} — skipping. Use force=True to rerun.")
        return pd.read_parquet(out_path)

    if all_df is None:
        from src.features import load_all_stocks
        safe_print("  Loading all stocks for walk-forward ...")
        all_df = load_all_stocks(start_date=START_DATE)

    all_df = all_df.copy()
    all_df["time"]  = pd.to_datetime(all_df["time"])
    all_df["_year"] = all_df["time"].dt.year

    feat_cols = [c for c in FEATURE_COLS if c in all_df.columns]
    target    = TARGET_COL if TARGET_COL in all_df.columns else "future_rel_barrier_5d"

    safe_print(f"\n  Walk-forward {start_year}..{train_end} | {len(feat_cols)} features")
    safe_print(f"  {'year':>5}  {'n_train':>8}  {'n_test':>7}  {'n_tickers':>10}  {'IC':>8}")
    safe_print("  " + "-" * 50)

    all_records = []
    for test_year in range(start_year, train_end + 1):
        train_mask = all_df["_year"] < test_year
        test_mask  = all_df["_year"] == test_year

        train_df = all_df[train_mask]
        test_df  = all_df[test_mask]

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        X_train_raw = train_df[feat_cols]
        y_train_raw = train_df[target]
        X_tr, y_tr, _ = _clean(X_train_raw, y_train_raw)

        X_test_raw = test_df[feat_cols]
        y_test_raw = test_df[target]
        X_te, y_te, valid_te = _clean(X_test_raw, y_test_raw)
        test_df_clean = test_df[valid_te.values]

        # Chronological val split for early stopping (last 12% of train)
        n_val = max(50, int(len(X_tr) * 0.12))
        X_t, y_t = X_tr.iloc[:-n_val], y_tr.iloc[:-n_val]
        X_v, y_v = X_tr.iloc[-n_val:], y_tr.iloc[-n_val:]

        model = _get_model()
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        probas = model.predict_proba(X_te)[:, 1]

        # Daily IC
        ic_list = []
        tmp = pd.DataFrame({
            "time":     pd.to_datetime(test_df_clean["time"].values),
            "xs_proba": probas,
            "actual":   y_te.values.astype(int),
        })
        for _, day in tmp.groupby("time"):
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["xs_proba"], day["actual"])
            if not np.isnan(ic):
                ic_list.append(ic)
        ic_mean = np.mean(ic_list) if ic_list else float("nan")

        safe_print(f"  {test_year:>5}  {len(X_tr):>8,}  {len(X_te):>7,}  "
                   f"{test_df_clean['ticker'].nunique():>10}  {ic_mean:>+8.4f}")

        for tick, t, p, a in zip(test_df_clean["ticker"].values,
                                  test_df_clean["time"].values,
                                  probas,
                                  y_te.values.astype(int)):
            all_records.append({
                "time":     t,
                "year":     test_year,
                "ticker":   tick,
                "xs_proba": round(float(p), 4),
                "actual":   int(a),
            })

    result = pd.DataFrame(all_records)
    result["time"] = pd.to_datetime(result["time"])
    result.to_parquet(out_path, index=False)
    safe_print(f"\n  Saved {len(result):,} rows to {out_path}")
    return result
