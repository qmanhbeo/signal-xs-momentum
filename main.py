"""
main.py — One-click runner for the momentum paper pipeline.

Steps:
  1/8  fetch_all()          — download data (393 stocks + VIX + VNINDEX)
  2/8  build_features()     — generate feature parquets (in-memory; no separate cache)
  3/8  run_walk_forward()   — train + score 2015-2025
  4/8  run_portfolio()      — simulate portfolio
  5/8  run_robustness()     — 4 validation tests
  6/8  run_2026_holdout()   — out-of-sample test
  7/8  generate_outputs()   — figures + LaTeX tables
  8/8  compile_manuscript() — pdflatex if available

Each step is idempotent: output files are skipped if they already exist.
Pass --force to re-run all steps.

Usage:
    python main.py
    python main.py --force
    python main.py --skip-fetch        # skip Step 1 (data already present)
    python main.py --steps 3,4,5       # run only specific steps
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (PROBAS_FILE, RESULTS_DIR, ensure_dirs, safe_print,
                       CUTOFF_DATE)


def step_header(n: int, total: int, desc: str):
    safe_print(f"\n{'='*65}")
    safe_print(f"Step {n}/{total}: {desc}")
    safe_print("="*65)


def _import_analysis(filename: str):
    """Import an analysis module by filename (handles digit-prefixed names)."""
    import importlib.util
    path = Path(__file__).resolve().parent / "analysis" / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step1_fetch(force: bool = False):
    step_header(1, 8, "Fetching data (393 stocks + VIX + VNINDEX)")
    from src.fetch import fetch_all
    fetch_all(cutoff=CUTOFF_DATE)


def step2_features(force: bool = False):
    step_header(2, 8, "Data summary + feature engineering check")
    out = RESULTS_DIR / "data_summary.json"
    if not force and out.exists():
        safe_print(f"  data_summary.json exists — skipping.")
        return
    from src.utils import STOCK_DIR, VNINDEX_FILE
    import glob
    n_stocks = len(glob.glob(str(STOCK_DIR / "gia_lich_su_*_1D.parquet"))) - 2
    safe_print(f"  {n_stocks} stock files found in {STOCK_DIR}")
    if not VNINDEX_FILE.exists():
        safe_print("  [ERROR] VNINDEX file missing — run Step 1 first.")
        sys.exit(1)
    _import_analysis("01_data_summary.py").run()
    safe_print("  Feature engineering runs inline during Step 3.")


def step3_walk_forward(force: bool = False):
    step_header(3, 8, "Walk-forward 2015-2025 (XGBoost cross-sectional)")
    if not force and PROBAS_FILE.exists():
        safe_print(f"  xs_probas.parquet already exists — skipping. Use --force to rerun.")
        return
    from src.features import load_all_stocks
    from src.model import run_walk_forward
    safe_print("  Loading all stocks (this may take several minutes) ...")
    all_df = load_all_stocks()
    run_walk_forward(all_df=all_df, force=force)


def step4_portfolio(force: bool = False):
    step_header(4, 8, "Signal IC + Portfolio simulation")
    # IC by year (reads xs_probas.parquet)
    ic_out = RESULTS_DIR / "ic_by_year.json"
    if force or not ic_out.exists():
        _import_analysis("02_signal_ic.py").run()
    else:
        safe_print(f"  ic_by_year.json exists — skipping IC computation.")
    # Portfolio simulation
    port_out = RESULTS_DIR / "portfolio_annual.json"
    if not force and port_out.exists():
        safe_print(f"  portfolio_annual.json exists — skipping.")
        return
    _import_analysis("03_portfolio.py").run()


def step5_robustness(force: bool = False):
    step_header(5, 8, "Robustness tests (4 tests)")
    out = RESULTS_DIR / "robustness.json"
    if not force and out.exists():
        safe_print(f"  robustness.json exists — skipping.")
        return
    _import_analysis("04_robustness.py").run()


def step6_holdout(force: bool = False):
    step_header(6, 8, "2026 out-of-sample holdout")
    out = RESULTS_DIR / "holdout_2026.json"
    if not force and out.exists():
        safe_print(f"  holdout_2026.json exists — skipping.")
        return
    _import_analysis("05_holdout_2026.py").run()


def step7_outputs(force: bool = False):
    step_header(7, 8, "Generating figures and LaTeX tables")
    _import_analysis("06_outputs.py").run()


def step8_manuscript():
    step_header(8, 8, "Compiling manuscript (pdflatex)")
    tex = Path("manuscript/paper.tex")
    if not tex.exists():
        safe_print("  manuscript/paper.tex not found — skipping.")
        return
    try:
        manuscript_dir = Path(__file__).resolve().parent / "manuscript"
        # Run twice: first pass builds aux, second pass resolves references
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                capture_output=True, text=True,
                cwd=str(manuscript_dir)
            )
        if result.returncode == 0:
            safe_print("  pdflatex: compiled successfully → manuscript/paper.pdf")
        else:
            # Show first error from log
            log = (manuscript_dir / "paper.log").read_text(errors="replace")
            first_err = next((l for l in log.splitlines() if l.startswith("!")), "unknown error")
            safe_print(f"  pdflatex error: {first_err}")
            safe_print(f"  Full log: manuscript/paper.log")
    except FileNotFoundError:
        safe_print("  pdflatex not found — skipping PDF compilation.")
        safe_print("  Install TeX Live / MiKTeX to compile the manuscript.")


def main():
    parser = argparse.ArgumentParser(
        description="Market momentum paper pipeline — one-click runner"
    )
    parser.add_argument("--force",       action="store_true",
                        help="Re-run all steps, ignoring cached outputs")
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip Step 1 (data fetching)")
    parser.add_argument("--steps",       type=str, default="",
                        help="Comma-separated list of steps to run, e.g. '3,4,5'")
    args = parser.parse_args()

    ensure_dirs()

    # Determine which steps to run
    if args.steps:
        to_run = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
    else:
        to_run = set(range(1, 9))

    if args.skip_fetch:
        to_run.discard(1)

    step_map = {
        1: lambda: step1_fetch(args.force),
        2: lambda: step2_features(args.force),
        3: lambda: step3_walk_forward(args.force),
        4: lambda: step4_portfolio(args.force),
        5: lambda: step5_robustness(args.force),
        6: lambda: step6_holdout(args.force),
        7: lambda: step7_outputs(args.force),
        8: lambda: step8_manuscript(),
    }

    for step_num in sorted(to_run):
        if step_num in step_map:
            try:
                step_map[step_num]()
            except Exception as e:
                safe_print(f"\n[ERROR] Step {step_num} failed: {e}")
                import traceback
                traceback.print_exc()
                safe_print("Continuing to next step ...")

    safe_print("\n" + "="*65)
    safe_print("Pipeline complete.")
    safe_print("  Results:  output/results/")
    safe_print("  Figures:  output/figures/")
    safe_print("  Tables:   output/tables/")
    safe_print("  Paper:    manuscript/paper.pdf  (if pdflatex available)")
    safe_print("="*65)


if __name__ == "__main__":
    main()
