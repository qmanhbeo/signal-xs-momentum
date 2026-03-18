"""
src/fetch.py — Data fetching for the momentum paper pipeline.

Two data sources:
  1. HOSE stock daily OHLCV via vnstock VCI  →  data/stock/gia_lich_su_{TICKER}_1D.parquet
  2. VIX (^VIX) via yfinance                 →  data/vix/vix_daily.parquet

Idempotent: a file is only re-fetched if it is missing or its last row predates
CUTOFF_DATE.
"""
import socket
import time
import warnings
import pandas as pd
from pathlib import Path

from src.utils import (STOCK_DIR, VIX_DIR, VNINDEX_FILE, VIX_FILE,
                       CUTOFF_DATE, START_DATE, safe_print, ensure_dirs)

warnings.filterwarnings("ignore")

# ── Complete 393-ticker HOSE universe ─────────────────────────────────────────
HOSE_TICKERS = [
    "AAA","AAM","AAT","ABR","ABS","ABT","ACB","ACC","ACG","ACL","ADG","ADP",
    "ADS","AFX","AGG","AGR","ANT","ANV","APG","APH","ASG","ASM","ASP","AST",
    "BAF","BCE","BCG","BCM","BFC","BHN","BIC","BID","BKG","BMC","BMI","BMP",
    "BRC","BSI","BSR","BTP","BTT","BVH","BWE","C32","C47","CCC","CCI","CCL",
    "CDC","CHP","CIG","CII","CKG","CLC","CLL","CLW","CMG","CMV","CMX","CNG",
    "COM","CRC","CRE","CRV","CSM","CSV","CTD","CTF","CTG","CTI","CTR","CTS",
    "CVT","D2D","DAH","DAT","DBC","DBD","DBT","DC4","DCL","DCM","DGC","DGW",
    "DHA","DHC","DHG","DHM","DIG","DLG","DMC","DPG","DPM","DPR","DQC","DRC",
    "DRH","DRL","DSC","DSE","DSN","DTA","DTL","DTT","DVP","DXG","DXS","DXV",
    "EIB","ELC","EVE","EVF","EVG","FCM","FCN","FDC","FIR","FIT","FMC","FPT",
    "FRT","FTS","GAS","GDT","GEE","GEG","GEX","GHC","GIL","GMD","GMH","GSP",
    "GTA","GVR","HAG","HAH","HAR","HAS","HAX","HCD","HCM","HDB","HDC","HDG",
    "HHP","HHS","HHV","HID","HII","HMC","HNA","HPG","HPX","HQC","HRC","HSG",
    "HSL","HT1","HTG","HTI","HTL","HTN","HTV","HU1","HUB","HVH","HVN","ICT",
    "IDI","IJC","ILB","IMP","ITC","ITD","JVC","KBC","KDC","KDH","KHG","KHP",
    "KLB","KMR","KOS","L10","LAF","LBM","LCG","LDG","LGC","LGL","LHG","LIX",
    "LM8","LPB","LSS","MBB","MCM","MCP","MDG","MHC","MIG","MSB","MSH","MSN",
    "MWG","NAB","NAF","NAV","NBB","NCT","NHA","NHH","NHT","NKG","NLG","NNC",
    "NO1","NSC","NT2","NTC","NTL","NVL","NVT","OCB","OGC","OPC","ORS","PAC",
    "PAN","PC1","PDN","PDR","PDV","PET","PGC","PGD","PGI","PGV","PHC","PHR",
    "PIT","PJT","PLP","PLX","PMG","PNC","PNJ","POW","PPC","PTB","PTC","PTL",
    "PVB","PVD","PVP","PVT","QCG","QNP","RAL","REE","RYG","S4A","SAB","SAM",
    "SAV","SBA","SBG","SBT","SBV","SC5","SCR","SCS","SFC","SFG","SFI","SGN",
    "SGR","SGT","SHA","SHB","SHI","SHP","SIP","SJD","SJS","SKG","SMA","SMB",
    "SMC","SPM","SRC","SRF","SSB","SSC","SSI","ST8","STB","STG","STK","SVC",
    "SVD","SVT","SZC","SZL","TAL","TBC","TCB","TCD","TCH","TCI","TCL","TCM",
    "TCO","TCR","TCT","TCX","TDC","TDG","TDH","TDM","TDP","TDW","TEG","THG",
    "TIP","TIX","TLD","TLG","TLH","TMP","TMS","TMT","TN1","TNC","TNH","TNI",
    "TNT","TPB","TPC","TRA","TRC","TSA","TSC","TTA","TTE","TTF","TV2","TVB",
    "TVS","TVT","TYA","UIC","VAB","VAF","VCA","VCB","VCF","VCG","VCI","VCK",
    "VDP","VDS","VFG","VGC","VHC","VHM","VIB","VIC","VID","VIP","VIX","VJC",
    "VMD","VND","VNE","VNG","VNL","VNM","VNS","VOS","VPB","VPD","VPG","VPH",
    "VPI","VPL","VPS","VPX","VRC","VRE","VSC","VSH","VSI","VTB","VTO","VTP",
    "VVS","YBM","YEG",
]


def _is_fresh(path: Path, cutoff: str) -> bool:
    """Return True if file exists and last row date >= cutoff."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["time"])
        last = pd.to_datetime(df["time"]).max()
        return str(last.date()) >= cutoff
    except Exception:
        return False


def _fetch_vnstock(ticker: str, dest: Path, cutoff: str, delay: float) -> bool:
    """Fetch one ticker via vnstock VCI and save as parquet. Returns True on success."""
    if _is_fresh(dest, cutoff):
        return True
    try:
        socket.setdefaulttimeout(30)
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.history(start=START_DATE, end=cutoff, interval="1D")
        if df is None or len(df) == 0:
            return False
        df.to_parquet(dest, index=False)
        time.sleep(delay)
        return True
    except Exception as e:
        safe_print(f"  [WARN] {ticker}: {e}")
        return False


def fetch_all(data_dir: Path = None, cutoff: str = CUTOFF_DATE, delay: float = 0.3):
    """
    Download all 393 HOSE stocks + VNINDEX (vnstock) and VIX (yfinance).
    Skips files whose last row >= cutoff.
    """
    ensure_dirs()
    if data_dir is None:
        stock_dir = STOCK_DIR
        vix_dir   = VIX_DIR
    else:
        stock_dir = Path(data_dir) / "stock"
        vix_dir   = Path(data_dir) / "vix"
        stock_dir.mkdir(parents=True, exist_ok=True)
        vix_dir.mkdir(parents=True, exist_ok=True)

    tickers_all = HOSE_TICKERS + ["VNINDEX"]
    n_total = len(tickers_all)
    n_ok = 0
    n_skip = 0
    failed = []

    safe_print(f"\nFetching {n_total} tickers (cutoff {cutoff}) ...")
    for i, ticker in enumerate(tickers_all):
        dest = stock_dir / f"gia_lich_su_{ticker}_1D.parquet"
        if _is_fresh(dest, cutoff):
            n_skip += 1
            n_ok += 1
            continue
        ok = _fetch_vnstock(ticker, dest, cutoff, delay)
        if ok:
            n_ok += 1
            safe_print(f"  [{i+1}/{n_total}] {ticker}: OK")
        else:
            failed.append(ticker)
            safe_print(f"  [{i+1}/{n_total}] {ticker}: FAILED")

    safe_print(f"\nStocks: {n_ok}/{n_total} ok ({n_skip} skipped, {len(failed)} failed)")
    if failed:
        safe_print(f"  Failed: {', '.join(failed[:20])}")

    # VIX via yfinance
    safe_print(f"\nFetching VIX via yfinance ...")
    vix_dest = vix_dir / "vix_daily.parquet"
    if _is_fresh(vix_dest, cutoff):
        safe_print("  VIX: already up to date, skipping.")
    else:
        try:
            import yfinance as yf
            vix = yf.download("^VIX", start="2000-01-01", end=cutoff,
                              auto_adjust=False, progress=False)
            if vix is not None and len(vix) > 0:
                # Extract Close column before any index reset to avoid MultiIndex issues
                if isinstance(vix.columns, pd.MultiIndex):
                    # yfinance >=0.2.x returns MultiIndex (field, ticker)
                    close_series = vix["Close"].squeeze()  # Series
                else:
                    close_series = vix["Close"]
                vix = pd.DataFrame({
                    "time":  pd.to_datetime(close_series.index),
                    "close": close_series.values,
                }).dropna()
                vix.to_parquet(vix_dest, index=False)
                safe_print(f"  VIX: {len(vix)} rows saved.")
            else:
                safe_print("  VIX: no data returned.")
        except Exception as e:
            safe_print(f"  VIX: ERROR — {e}")

    safe_print("\nFetch complete.")


if __name__ == "__main__":
    fetch_all()
