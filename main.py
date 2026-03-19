"""
One-click runner for the signal-xs-momentum paper pipeline.

Steps:
  1/9  fetch_all()          - download data (393 stocks + VIX + VNINDEX)
  2/9  build_features()     - data summary + feature engineering check
  3/9  run_walk_forward()   - train + score 2015-2025
  4/9  run_portfolio()      - signal IC + portfolio simulation
  5/9  run_robustness()     - four validation tests
  6/9  run_2026_holdout()   - out-of-sample holdout
  7/9  run_postfreeze()     - supplementary credibility checks
  8/9  generate_outputs()   - figures + LaTeX tables
  9/9  compile_manuscript() - pdflatex if available
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import CUTOFF_DATE, PROBAS_FILE, RESULTS_DIR, ensure_dirs, safe_print


def step_header(n: int, total: int, desc: str):
    safe_print(f"\n{'=' * 65}")
    safe_print(f"Step {n}/{total}: {desc}")
    safe_print("=" * 65)


def _import_analysis(filename: str):
    import importlib.util

    path = Path(__file__).resolve().parent / "analysis" / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step1_fetch(force: bool = False):
    step_header(1, 9, "Fetching data (393 stocks + VIX + VNINDEX)")
    from src.fetch import fetch_all

    fetch_all(cutoff=CUTOFF_DATE)


def step2_features(force: bool = False):
    step_header(2, 9, "Data summary + feature engineering check")
    out = RESULTS_DIR / "data_summary.json"
    if not force and out.exists():
        safe_print("  data_summary.json exists - skipping.")
        return

    import glob
    from src.utils import STOCK_DIR, VNINDEX_FILE

    n_stocks = len(glob.glob(str(STOCK_DIR / "gia_lich_su_*_1D.parquet"))) - 2
    safe_print(f"  {n_stocks} stock files found in {STOCK_DIR}")
    if not VNINDEX_FILE.exists():
        safe_print("  [ERROR] VNINDEX file missing - run Step 1 first.")
        sys.exit(1)

    _import_analysis("01_data_summary.py").run()
    safe_print("  Feature engineering runs inline during Step 3.")


def step3_walk_forward(force: bool = False):
    step_header(3, 9, "Walk-forward 2015-2025 (XGBoost cross-sectional)")
    if not force and PROBAS_FILE.exists():
        safe_print("  xs_probas.parquet already exists - skipping. Use --force to rerun.")
        return

    from src.features import load_all_stocks
    from src.model import run_walk_forward

    safe_print("  Loading all stocks (this may take several minutes) ...")
    all_df = load_all_stocks()
    run_walk_forward(all_df=all_df, force=force)


def step4_portfolio(force: bool = False):
    step_header(4, 9, "Signal IC + Portfolio simulation")
    ic_out = RESULTS_DIR / "ic_by_year.json"
    if force or not ic_out.exists():
        _import_analysis("02_signal_ic.py").run()
    else:
        safe_print("  ic_by_year.json exists - skipping IC computation.")

    port_out = RESULTS_DIR / "portfolio_annual.json"
    if not force and port_out.exists():
        safe_print("  portfolio_annual.json exists - skipping.")
        return
    _import_analysis("03_portfolio.py").run()


def step5_robustness(force: bool = False):
    step_header(5, 9, "Robustness tests (4 tests)")
    out = RESULTS_DIR / "robustness.json"
    if not force and out.exists():
        safe_print("  robustness.json exists - skipping.")
        return
    _import_analysis("04_robustness.py").run()


def step6_holdout(force: bool = False):
    step_header(6, 9, "2026 out-of-sample holdout")
    out = RESULTS_DIR / "holdout_2026.json"
    if not force and out.exists():
        safe_print("  holdout_2026.json exists - skipping.")
        return
    _import_analysis("05_holdout_2026.py").run()


def step7_postfreeze(force: bool = False):
    step_header(7, 9, "Post-freeze credibility extensions")
    out = RESULTS_DIR / "postfreeze_extensions.json"
    if not force and out.exists():
        safe_print("  postfreeze_extensions.json exists - skipping.")
        return
    _import_analysis("07_postfreeze_extensions.py").run()


def step8_outputs(force: bool = False):
    step_header(8, 9, "Generating figures and LaTeX tables")
    _import_analysis("06_outputs.py").run()


def step9_manuscript():
    step_header(9, 9, "Compiling manuscript (pdflatex)")
    tex = Path("manuscript/paper.tex")
    if not tex.exists():
        safe_print("  manuscript/paper.tex not found - skipping.")
        return

    try:
        manuscript_dir = Path(__file__).resolve().parent / "manuscript"
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                capture_output=True,
                text=True,
                cwd=str(manuscript_dir),
            )
        if result.returncode == 0:
            safe_print("  pdflatex: compiled successfully -> manuscript/paper.pdf")
        else:
            log = (manuscript_dir / "paper.log").read_text(errors="replace")
            first_err = next((line for line in log.splitlines() if line.startswith("!")), "unknown error")
            safe_print(f"  pdflatex error: {first_err}")
            safe_print("  Full log: manuscript/paper.log")
    except FileNotFoundError:
        safe_print("  pdflatex not found - skipping PDF compilation.")
        safe_print("  Install TeX Live / MiKTeX to compile the manuscript.")


def main():
    parser = argparse.ArgumentParser(description="signal-xs-momentum one-click pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run all steps, ignoring cached outputs")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip Step 1 (data fetching)")
    parser.add_argument("--steps", type=str, default="", help="Comma-separated steps to run, e.g. '3,4,5'")
    args = parser.parse_args()

    ensure_dirs()

    if args.steps:
        to_run = {int(step.strip()) for step in args.steps.split(",") if step.strip()}
    else:
        to_run = set(range(1, 10))

    if args.skip_fetch:
        to_run.discard(1)

    step_map = {
        1: lambda: step1_fetch(args.force),
        2: lambda: step2_features(args.force),
        3: lambda: step3_walk_forward(args.force),
        4: lambda: step4_portfolio(args.force),
        5: lambda: step5_robustness(args.force),
        6: lambda: step6_holdout(args.force),
        7: lambda: step7_postfreeze(args.force),
        8: lambda: step8_outputs(args.force),
        9: step9_manuscript,
    }

    for step_num in sorted(to_run):
        if step_num not in step_map:
            continue
        try:
            step_map[step_num]()
        except Exception as exc:
            safe_print(f"\n[ERROR] Step {step_num} failed: {exc}")
            import traceback

            traceback.print_exc()
            safe_print("Continuing to next step ...")

    safe_print("\n" + "=" * 65)
    safe_print("Pipeline complete.")
    safe_print("  Results:  output/results/")
    safe_print("  Figures:  output/figures/")
    safe_print("  Tables:   output/tables/")
    safe_print("  Paper:    manuscript/paper.pdf  (if pdflatex available)")
    safe_print("=" * 65)


if __name__ == "__main__":
    main()
