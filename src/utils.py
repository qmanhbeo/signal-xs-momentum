"""Shared helpers: safe print, parquet IO, path constants."""
import sys
import os
from pathlib import Path

# Repository root (one level above this file)
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR    = REPO_ROOT / "data"
STOCK_DIR   = DATA_DIR / "stock"
VIX_DIR     = DATA_DIR / "vix"
OUTPUT_DIR  = REPO_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR  = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"

VNINDEX_FILE = STOCK_DIR / "gia_lich_su_VNINDEX_1D.parquet"
VIX_FILE     = VIX_DIR   / "vix_daily.parquet"
PROBAS_FILE  = RESULTS_DIR / "xs_probas.parquet"

# ── Global constants ──────────────────────────────────────────────────────────
CUTOFF_DATE  = "2026-03-18"
START_DATE   = "2012-01-01"
TRAIN_END    = 2025
TEST_YEAR    = 2026
N_STOCKS_MIN = 300
TOP_N        = 5
MIN_PROBA    = 0.55
VIX_MIN      = 15.0
VIX_MAX      = 30.0
COST_RT      = 0.003    # 0.30% round-trip
BARRIER_PCT  = 0.01     # ±1% triple barrier
HORIZON      = 5        # 5-day forward window


def safe_print(*args, **kwargs):
    """Print with UTF-8 encoding, falling back to ASCII on narrow terminals."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args)
        print(msg.encode("ascii", "replace").decode("ascii"), **kwargs)


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [STOCK_DIR, VIX_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
