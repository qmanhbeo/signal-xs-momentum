"""
04_robustness.py — Four validation tests.

1. Label permutation  — shuffle actual labels → IC should collapse to ~0
2. Sector neutrality  — demean actual by sector-day → IC drop < 15% = PASS
3. Mean-reversion     — -past_5d_excess IC vs model IC (expect ~33%)
4. Execution timing   — E=1 vs E=2 net Sharpe (E=2 must be negative or near-zero)

Saves: output/results/robustness.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from src.utils import (PROBAS_FILE, STOCK_DIR, VNINDEX_FILE, VIX_FILE,
                       RESULTS_DIR, safe_print, ensure_dirs)
from src.portfolio import run_portfolio

warnings.filterwarnings("ignore", category=RuntimeWarning)

START_YEAR = 2015

# Sector mapping (167 / 393 tickers classified into 15 named sectors)
SECTOR_MAP = {
    # Banks (22)
    "ACB": "Bank", "BID": "Bank", "CTG": "Bank", "EIB": "Bank",
    "HDB": "Bank", "KLB": "Bank", "LPB": "Bank", "MBB": "Bank",
    "MSB": "Bank", "NAB": "Bank", "OCB": "Bank", "PVB": "Bank",
    "SHB": "Bank", "SSB": "Bank", "STB": "Bank", "TCB": "Bank",
    "TPB": "Bank", "TVB": "Bank", "VAB": "Bank", "VCB": "Bank",
    "VIB": "Bank", "VPB": "Bank",
    # Securities (12)
    "AGR": "Securities", "BSI": "Securities", "CTS": "Securities",
    "FTS": "Securities", "HCM": "Securities", "ORS": "Securities",
    "SSI": "Securities", "TVS": "Securities", "VCI": "Securities",
    "VDS": "Securities", "VIX": "Securities", "VND": "Securities",
    # Insurance (5)
    "BIC": "Insurance", "BMI": "Insurance", "BVH": "Insurance",
    "MIG": "Insurance", "PGI": "Insurance",
    # Real Estate (29)
    "AGG": "RealEstate", "BCG": "RealEstate", "BCM": "RealEstate",
    "CRE": "RealEstate", "DIG": "RealEstate", "DPG": "RealEstate",
    "DXG": "RealEstate", "DXS": "RealEstate", "FIR": "RealEstate",
    "HDC": "RealEstate", "KBC": "RealEstate", "KDH": "RealEstate",
    "LDG": "RealEstate", "NHA": "RealEstate", "NLG": "RealEstate",
    "NRC": "RealEstate", "NVL": "RealEstate", "PDR": "RealEstate",
    "QCG": "RealEstate", "SCR": "RealEstate", "SIP": "RealEstate",
    "SJS": "RealEstate", "SZC": "RealEstate", "TDC": "RealEstate",
    "TDH": "RealEstate", "VHM": "RealEstate", "VIC": "RealEstate",
    "VPH": "RealEstate", "VRC": "RealEstate",
    # Steel / Metal (8)
    "DTL": "Steel", "HPG": "Steel", "HSG": "Steel", "NHH": "Steel",
    "NKG": "Steel", "SMC": "Steel", "TLH": "Steel", "VGS": "Steel",
    # Oil & Gas (5)
    "BSR": "OilGas", "GAS": "OilGas", "PVD": "OilGas",
    "PVP": "OilGas", "PVT": "OilGas",
    # Power (14)
    "BTP": "Power", "BWE": "Power", "CHP": "Power", "EVE": "Power",
    "GEE": "Power", "GEG": "Power", "GEX": "Power", "NT2": "Power",
    "PC1": "Power", "POW": "Power", "PPC": "Power", "REE": "Power",
    "TBC": "Power", "VSH": "Power",
    # Food & Consumer (16)
    "AAM": "FoodConsumer", "ABT": "FoodConsumer", "ACL": "FoodConsumer",
    "ANV": "FoodConsumer", "CLC": "FoodConsumer", "FMC": "FoodConsumer",
    "HAG": "FoodConsumer", "IDI": "FoodConsumer", "KDC": "FoodConsumer",
    "LSS": "FoodConsumer", "MSN": "FoodConsumer", "SAB": "FoodConsumer",
    "SBT": "FoodConsumer", "TAC": "FoodConsumer", "VHC": "FoodConsumer",
    "VNM": "FoodConsumer",
    # Pharma (6)
    "DHG": "Pharma", "DMC": "Pharma", "IMP": "Pharma",
    "OPC": "Pharma", "PHC": "Pharma", "TRA": "Pharma",
    # Technology (5)
    "ADG": "Technology", "CMG": "Technology", "ELC": "Technology",
    "FPT": "Technology", "SGT": "Technology",
    # Construction (16)
    "ACC": "Construction", "BCE": "Construction", "C32": "Construction",
    "C47": "Construction", "CCC": "Construction", "CCI": "Construction",
    "CII": "Construction", "CTD": "Construction", "CTI": "Construction",
    "CTR": "Construction", "FCN": "Construction", "HBC": "Construction",
    "HHV": "Construction", "LCG": "Construction", "SC5": "Construction",
    "TV2": "Construction",
    # Rubber / Agriculture (8)
    "BAF": "RubberAgri", "CSM": "RubberAgri", "DPR": "RubberAgri",
    "DRC": "RubberAgri", "GVR": "RubberAgri", "HRC": "RubberAgri",
    "PHR": "RubberAgri", "TRC": "RubberAgri",
    # Plastics (3)
    "AAA": "Plastics", "BMP": "Plastics", "NTP": "Plastics",
    # Shipping / Logistics (8)
    "GMD": "Shipping", "GSP": "Shipping", "HAH": "Shipping",
    "HTV": "Shipping", "VOS": "Shipping", "VSC": "Shipping",
    "VTO": "Shipping", "VTP": "Shipping",
    # Textiles (4)
    "MSH": "Textiles", "STK": "Textiles", "STG": "Textiles", "TNG": "Textiles",
}


def _daily_ic(df: pd.DataFrame, sig: str, out: str, min_n: int = 10) -> list:
    ics = []
    for _, day in df.groupby("time"):
        sub = day[[sig, out]].dropna()
        if len(sub) < min_n:
            continue
        ic, _ = spearmanr(sub[sig], sub[out])
        if not np.isnan(ic):
            ics.append(float(ic))
    return ics


def test_permutation(probas: pd.DataFrame) -> dict:
    """Shuffle actual labels → IC should collapse to ~0."""
    safe_print("\n  Test 1: Label Permutation")
    rng = np.random.default_rng(42)
    p2  = probas.copy()
    p2["actual_perm"] = rng.permutation(p2["actual"].values)

    ic_raw  = _daily_ic(p2, "xs_proba", "actual")
    ic_perm = _daily_ic(p2, "xs_proba", "actual_perm")

    m_raw  = float(np.mean(ic_raw))
    m_perm = float(np.mean(ic_perm))
    safe_print(f"    IC_raw:  {m_raw:+.4f}")
    safe_print(f"    IC_perm: {m_perm:+.4f} (expect ~0)")
    verdict = "PASS" if abs(m_perm) < 0.02 else "FAIL"
    safe_print(f"    Verdict: {verdict}")
    return {"ic_raw": m_raw, "ic_perm": m_perm, "verdict": verdict}


def test_sector_neutral(probas: pd.DataFrame) -> dict:
    """Sector-neutral IC: demean actual by sector-day."""
    safe_print("\n  Test 2: Sector Neutrality")
    p2 = probas.copy()
    p2["sector"] = p2["ticker"].map(SECTOR_MAP).fillna("Other")

    # Demean within sector-day
    def _neutral(df):
        result = pd.Series(np.nan, index=df.index)
        for (_, sector), grp in df.groupby(["time", "sector"]):
            result.loc[grp.index] = grp["actual"] - grp["actual"].mean()
        return result

    p2["actual_neutral"] = _neutral(p2)
    ic_raw     = _daily_ic(p2, "xs_proba", "actual")
    ic_neutral = _daily_ic(p2, "xs_proba", "actual_neutral")

    m_raw = float(np.mean(ic_raw))
    m_neu = float(np.mean(ic_neutral))
    drop  = (m_raw - m_neu) / m_raw * 100 if m_raw > 0 else float("nan")
    safe_print(f"    IC_raw:     {m_raw:+.4f}")
    safe_print(f"    IC_neutral: {m_neu:+.4f}")
    safe_print(f"    IC drop:    {drop:.1f}%  (threshold <15%)")
    verdict = "PASS" if drop < 15 else ("MARGINAL" if drop < 30 else "FAIL")
    safe_print(f"    Verdict: {verdict}")
    return {"ic_raw": m_raw, "ic_neutral": m_neu, "ic_drop_pct": round(drop, 1),
            "verdict": verdict}


def test_mean_reversion(probas: pd.DataFrame) -> dict:
    """MR baseline: IC of -past_5d_excess vs model IC."""
    safe_print("\n  Test 3: Mean-Reversion Baseline")

    tickers = probas["ticker"].unique().tolist()
    closes_list = []
    for sf in sorted(STOCK_DIR.glob("gia_lich_su_*_1D.parquet")):
        ticker = sf.name.replace("gia_lich_su_", "").replace("_1D.parquet", "")
        if ticker not in set(tickers) or ticker == "VNINDEX":
            continue
        try:
            df = pd.read_parquet(sf, columns=["time", "close"])
            df["time"] = pd.to_datetime(df["time"])
            df["ticker"] = ticker
            closes_list.append(df[["time", "ticker", "close"]])
        except Exception:
            pass

    if not closes_list:
        safe_print("    [SKIP] No close prices found.")
        return {"verdict": "SKIP"}

    closes = pd.concat(closes_list, ignore_index=True)
    vni    = (pd.read_parquet(VNINDEX_FILE, columns=["time", "close"])
              .rename(columns={"close": "vni"}))
    vni["time"] = pd.to_datetime(vni["time"])

    wide = closes.pivot(index="time", columns="ticker", values="close").sort_index()
    vni_s = vni.set_index("time")["vni"].reindex(wide.index).ffill()

    mr5 = -(np.log(wide / wide.shift(5)).sub(np.log(vni_s / vni_s.shift(5)), axis=0))
    mr5_long = mr5.stack().reset_index()
    mr5_long.columns = ["time", "ticker", "mr5"]

    merged = probas[probas["time"].dt.year >= START_YEAR].merge(
        mr5_long, on=["time", "ticker"], how="left")

    ic_model = _daily_ic(merged, "xs_proba", "actual", min_n=20)
    ic_mr5   = _daily_ic(merged, "mr5", "actual", min_n=20)

    m_model = float(np.mean(ic_model))
    m_mr5   = float(np.mean(ic_mr5))
    ratio   = m_mr5 / m_model if m_model != 0 else float("nan")
    safe_print(f"    IC_model: {m_model:+.4f}")
    safe_print(f"    IC_mr5:   {m_mr5:+.4f}  ({ratio*100:.1f}% of model IC)")
    verdict = ("PASS" if ratio < 0.40 else
               "MODERATE" if ratio < 0.70 else "CONCERN")
    safe_print(f"    Verdict: {verdict}  (threshold <40%=PASS, <70%=MODERATE)")
    return {"ic_model": m_model, "ic_mr5": m_mr5, "ratio_pct": round(ratio * 100, 1),
            "verdict": verdict}


def test_execution_timing(probas: pd.DataFrame) -> dict:
    """E=1 vs E=2 net Sharpe — E=2 must be negative or near-zero."""
    safe_print("\n  Test 4: Execution Timing (E=1 vs E=2)")
    res_e1 = run_portfolio(probas=probas, entry_lag=1, start_year=START_YEAR)
    res_e2 = run_portfolio(probas=probas, entry_lag=2, start_year=START_YEAR)

    sh1 = res_e1["sharpe_net"]
    sh2 = res_e2["sharpe_net"]
    safe_print(f"    E=1 net Sharpe: {sh1:.2f}")
    safe_print(f"    E=2 net Sharpe: {sh2:.2f}  (expect negative)")
    verdict = ("PASS" if sh2 < 0 else
               "MARGINAL" if sh2 < 0.5 else "FAIL")
    safe_print(f"    Verdict: {verdict}")
    return {"sh_e1": sh1, "sh_e2": sh2, "verdict": verdict}


def run():
    ensure_dirs()
    safe_print("\n--- 04: Robustness Tests ---")

    if not PROBAS_FILE.exists():
        safe_print(f"  [ERROR] {PROBAS_FILE} not found — run Step 3 first.")
        return None

    probas = pd.read_parquet(PROBAS_FILE)
    probas["time"] = pd.to_datetime(probas["time"])
    probas = probas[probas["time"].dt.year >= START_YEAR].copy()

    r1 = test_permutation(probas)
    r2 = test_sector_neutral(probas)
    r3 = test_mean_reversion(probas)
    r4 = test_execution_timing(probas)

    results = {
        "permutation":      r1,
        "sector_neutral":   r2,
        "mean_reversion":   r3,
        "execution_timing": r4,
    }

    n_pass = sum(1 for r in [r1, r2, r3, r4] if r.get("verdict") == "PASS")
    safe_print(f"\n  Summary: {n_pass}/4 tests passed")

    out = RESULTS_DIR / "robustness.json"
    out.write_text(json.dumps(results, indent=2))
    safe_print(f"  Saved {out}")
    return results


if __name__ == "__main__":
    run()
