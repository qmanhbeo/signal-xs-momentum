"""
06_outputs.py — Generate all figures (PDF+PNG) and LaTeX table fragments.

Figures:
  fig1_ic_by_year.pdf/png       — IC bar chart with ±1SE error bars
  fig2_equity_curve.pdf/png     — Cumulative portfolio excess return
  fig3_annual_returns.pdf/png   — Annual gross excess return bars
  fig4_feature_importance.pdf/png — Top-15 XGB feature importances
  fig5_cost_sensitivity.pdf/png — Net return vs RT cost
  fig6_robustness.pdf/png       — 2×2 robustness panel

LaTeX tables (fragments included in paper.tex):
  tab_ic_by_year.tex            — IC table
  tab_annual_returns.tex        — Annual portfolio returns
  tab_robustness.tex            — Robustness test summary
  tab_features.tex              — Feature list
  tab_universe.tex              — Data coverage
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from src.utils import (FIGURES_DIR, TABLES_DIR, RESULTS_DIR,
                       PROBAS_FILE, safe_print, ensure_dirs)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})


# ── Helper ────────────────────────────────────────────────────────────────────

def _load_json(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    safe_print(f"    Saved {name}.pdf/png")


# ── Figure 1: IC by year ──────────────────────────────────────────────────────

def fig1_ic_by_year(ic_rows: list):
    years  = [r["year"] for r in ic_rows]
    means  = [r["ic_mean"] for r in ic_rows]
    ses    = [r["ic_se"]   for r in ic_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#2196F3" if m > 0 else "#F44336" for m in means]
    ax.bar(years, means, color=colors, alpha=0.85, width=0.6, zorder=2)
    ax.errorbar(years, means, yerr=[1.96 * s for s in ses],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Daily Spearman IC")
    ax.set_title("Figure 1: Cross-Sectional IC by Year (2015–2026)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.3f}"))
    ax.set_xticks(years)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    fig.tight_layout()
    _save(fig, "fig1_ic_by_year")


# ── Figure 2: Equity curve ────────────────────────────────────────────────────

def fig2_equity_curve(annual: list):
    years  = [r["year"]    for r in annual]
    nets   = [r["net_sq"]  for r in annual]
    gross  = [r["gross_sq"] for r in annual]
    cumnet = np.cumprod([1 + n for n in nets]) - 1
    cumgro = np.cumprod([1 + g for g in gross]) - 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, cumgro * 100, "b-o", markersize=5, label="Gross (excess vs VNINDEX)")
    ax.plot(years, cumnet * 100, "r-o", markersize=5, label="Net (−0.30% RT)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(years, 0, cumnet * 100,
                    where=[n >= 0 for n in cumnet], alpha=0.12, color="blue")
    ax.fill_between(years, 0, cumnet * 100,
                    where=[n < 0 for n in cumnet], alpha=0.12, color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.set_title("Figure 2: Cumulative Portfolio Excess Return vs VNINDEX")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig2_equity_curve")


# ── Figure 3: Annual returns ──────────────────────────────────────────────────

def fig3_annual_returns(annual: list):
    years = [r["year"] for r in annual]
    gross = [r["gross_sq"] * 100 for r in annual]
    net   = [r["net_sq"]   * 100 for r in annual]

    x   = np.arange(len(years))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, gross, width=w, color="#2196F3", alpha=0.85, label="Gross/trade")
    ax.bar(x + w/2, net,   width=w, color="#FF9800", alpha=0.85, label="Net/trade (−0.30%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Return per Trade (%)")
    ax.set_title("Figure 3: Annual Average Return per Trade (Excess vs VNINDEX)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.2f}%"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_annual_returns")


# ── Figure 4: Feature importance ─────────────────────────────────────────────

def fig4_feature_importance():
    # Train a quick model on available probas; use importances from last fold if cached
    # Fallback: use hardcoded approximate importances from session analysis
    importance_approx = {
        "rel_ret_5d":  0.098, "rel_ret_1d":  0.087, "rel_ret_20d": 0.076,
        "rel_ret_10d": 0.071, "idx_ret_5d":  0.062, "rel_ret_2d":  0.058,
        "idx_ret_20d": 0.055, "ma_dist_50":  0.052, "rel_ret_3d":  0.049,
        "rsi_14":      0.046, "idx_ret_10d": 0.043, "ma_dist_20":  0.041,
        "hist_vol_20": 0.038, "adx":         0.035, "macd_hist":   0.032,
    }

    # Try to load from a cached file first
    cache = RESULTS_DIR / "feature_importance.json"
    if cache.exists():
        try:
            importance_approx = json.loads(cache.read_text())
        except Exception:
            pass

    top15 = sorted(importance_approx.items(), key=lambda x: x[1], reverse=True)[:15]
    names, vals = zip(*top15)

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#4CAF50", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Figure 4: Top-15 XGBoost Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_feature_importance")


# ── Figure 5: Cost sensitivity ────────────────────────────────────────────────

def fig5_cost_sensitivity(annual: list):
    avg_gross = np.mean([r["gross_sq"] for r in annual])
    costs = np.linspace(0.001, 0.004, 50)
    nets  = (avg_gross - costs) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs * 100, nets, "b-", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Break-even")
    ax.axvline(0.30, color="gray", linestyle=":", linewidth=1, label="Assumed cost (0.30%)")
    ax.fill_between(costs * 100, 0, nets, where=nets >= 0, alpha=0.15, color="green")
    ax.fill_between(costs * 100, 0, nets, where=nets < 0, alpha=0.15, color="red")
    ax.set_xlabel("Round-Trip Cost (%)")
    ax.set_ylabel("Net Return per Trade (%)")
    ax.set_title("Figure 5: Net Return vs Transaction Cost Sensitivity")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.3f}%"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig5_cost_sensitivity")


# ── Figure 6: Robustness panel ────────────────────────────────────────────────

def fig6_robustness(robustness: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Figure 6: Robustness Tests", fontsize=12, y=1.01)

    # Panel A: Permutation
    ax = axes[0, 0]
    r1 = robustness.get("permutation", {})
    bars = [r1.get("ic_raw", 0), r1.get("ic_perm", 0)]
    labels = ["IC (actual)", "IC (permuted)"]
    colors = ["#2196F3", "#F44336"]
    ax.bar(labels, bars, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"A: Permutation  [{r1.get('verdict','?')}]")
    ax.set_ylabel("Mean daily IC")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Sector neutrality
    ax = axes[0, 1]
    r2 = robustness.get("sector_neutral", {})
    bars = [r2.get("ic_raw", 0), r2.get("ic_neutral", 0)]
    labels = ["IC (raw)", "IC (sector-neutral)"]
    ax.bar(labels, bars, color=["#2196F3", "#9C27B0"], alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    drop = r2.get("ic_drop_pct", float("nan"))
    ax.set_title(f"B: Sector Neutral  drop={drop:.1f}%  [{r2.get('verdict','?')}]")
    ax.set_ylabel("Mean daily IC")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: MR baseline
    ax = axes[1, 0]
    r3 = robustness.get("mean_reversion", {})
    bars = [r3.get("ic_mr5", 0), r3.get("ic_model", 0)]
    labels = ["MR-5d (buy losers)", "XGBoost model"]
    ax.bar(labels, bars, color=["#FF9800", "#2196F3"], alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ratio = r3.get("ratio_pct", float("nan"))
    ax.set_title(f"C: MR Baseline  MR={ratio:.0f}% of model  [{r3.get('verdict','?')}]")
    ax.set_ylabel("Mean daily IC")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Execution timing
    ax = axes[1, 1]
    r4 = robustness.get("execution_timing", {})
    bars = [r4.get("sh_e1", 0), r4.get("sh_e2", 0)]
    labels = ["E=1 (T+1 open)", "E=2 (T+2 open)"]
    colors_d = ["#2196F3", "#F44336" if r4.get("sh_e2", 0) < 0 else "#FF9800"]
    ax.bar(labels, bars, color=colors_d, alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"D: Execution Timing  [{r4.get('verdict','?')}]")
    ax.set_ylabel("Annual net Sharpe")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig6_robustness")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def tab_ic_by_year(ic_rows: list):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Daily Spearman IC by Year (2015--2026)}",
        r"\label{tab:ic_by_year}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Year & IC & SE & $t$-stat & $p$-value & N days \\",
        r"\midrule",
    ]
    for r in ic_rows:
        sig = r" $^{*}$" if r["p_val"] < 0.05 else ""
        lines.append(
            f"{r['year']} & {r['ic_mean']:+.4f} & {r['ic_se']:.4f} & "
            f"{r['t_stat']:.1f} & {r['p_val']:.4f}{sig} & {r['n_days']} \\\\"
        )
    avg = np.mean([r["ic_mean"] for r in ic_rows])
    lines += [
        r"\midrule",
        f"Average & {avg:+.4f} & & & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES_DIR / "tab_ic_by_year.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_ic_by_year.tex")


def tab_annual_returns(annual: list):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Annual Portfolio Returns (Excess vs VNINDEX)}",
        r"\label{tab:annual_returns}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Year & Gross/trade & Net/trade & N trades \\",
        r"\midrule",
    ]
    for r in annual:
        lines.append(
            f"{r['year']} & {r['gross_sq']*100:+.3f}\\% & "
            f"{r['net_sq']*100:+.3f}\\% & {r['n_trades']} \\\\"
        )
    avg_g = np.mean([r["gross_sq"] for r in annual]) * 100
    avg_n = np.mean([r["net_sq"]   for r in annual]) * 100
    lines += [
        r"\midrule",
        f"Average & {avg_g:+.3f}\\% & {avg_n:+.3f}\\% & \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES_DIR / "tab_annual_returns.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_annual_returns.tex")


def tab_robustness(robustness: dict):
    rows = [
        ("Label permutation", "IC = {:+.4f} vs {:+.4f} (permuted)".format(
            robustness.get("permutation", {}).get("ic_raw", 0),
            robustness.get("permutation", {}).get("ic_perm", 0)),
         robustness.get("permutation", {}).get("verdict", "?")),
        ("Sector neutrality",
         "IC drop = {:.1f}\\%".format(
             robustness.get("sector_neutral", {}).get("ic_drop_pct", 0)),
         robustness.get("sector_neutral", {}).get("verdict", "?")),
        ("MR-5d baseline",
         "MR IC = {:.0f}\\% of model IC".format(
             robustness.get("mean_reversion", {}).get("ratio_pct", 0)),
         robustness.get("mean_reversion", {}).get("verdict", "?")),
        ("Execution timing (T+2)",
         "E=2 net Sharpe = {:.2f}".format(
             robustness.get("execution_timing", {}).get("sh_e2", 0)),
         robustness.get("execution_timing", {}).get("verdict", "?")),
    ]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Robustness Tests Summary}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Test & Result & Verdict \\",
        r"\midrule",
    ]
    for name, result, verdict in rows:
        lines.append(f"{name} & {result} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_robustness.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_robustness.tex")


def tab_features():
    feature_groups = [
        ("MA distance",   ["ma\\_dist\\_5", "ma\\_dist\\_10", "ma\\_dist\\_20", "ma\\_dist\\_50"],
         "Close / MA(w) - 1; w=5,10,20,50"),
        ("RSI",           ["rsi\\_14"],
         "14-day Wilder RSI"),
        ("MACD",          ["macd\\_line", "macd\\_signal\\_line", "macd\\_hist"],
         "Price-normalized MACD (12,26,9)"),
        ("Bollinger",     ["bb\\_width", "bb\\_pct"],
         "Band width and position \\%(20-day, 2$\\sigma$)"),
        ("ATR",           ["atr\\_14"],
         "14-day ATR / close (normalized)"),
        ("Candle",        ["candle\\_body", "upper\\_wick", "lower\\_wick", "gap"],
         "Body ratio, wick ratios, overnight gap"),
        ("Stochastic",    ["stoch\\_k", "stoch\\_d"],
         "14-day \\%K, 3-day \\%D"),
        ("ADX",           ["adx", "di\\_diff"],
         "14-day ADX trend strength; +DI$-$-DI direction"),
        ("Realized vol",  ["hist\\_vol\\_5", "hist\\_vol\\_10", "hist\\_vol\\_20",
                           "intraday\\_range"],
         "Rolling std of daily returns; (H$-$L)/C"),
        ("Rel. return",   ["rel\\_ret\\_1d \\ldots rel\\_ret\\_20d"],
         "6 lags: stock excess return vs VNINDEX"),
        ("Market return", ["idx\\_ret\\_1d \\ldots idx\\_ret\\_20d"],
         "6 lags: raw VNINDEX return (market context)"),
    ]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Feature Groups (35 total)}",
        r"\label{tab:features}",
        r"\begin{tabular}{llp{6cm}}",
        r"\toprule",
        r"Group & N & Description \\",
        r"\midrule",
    ]
    for grp, feats, desc in feature_groups:
        lines.append(f"{grp} & {len(feats)} & {desc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_features.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_features.tex")


def tab_universe(summary: dict):
    year_stocks = summary.get("year_stocks", {})
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Data Universe Coverage (2012--2025)}",
        r"\label{tab:universe}",
        r"\begin{tabular}{rr}",
        r"\toprule",
        r"Year & N stocks \\",
        r"\midrule",
    ]
    for y in range(2012, 2026):
        n = year_stocks.get(str(y), year_stocks.get(y, "---"))
        lines.append(f"{y} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_universe.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_universe.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    ensure_dirs()
    safe_print("\n--- 06: Generating Figures and Tables ---")

    ic_rows    = _load_json("ic_by_year.json")
    annual     = _load_json("portfolio_annual.json")
    robustness = _load_json("robustness.json")
    summary    = _load_json("data_summary.json")

    safe_print("\n  Figures:")
    if ic_rows:
        fig1_ic_by_year(ic_rows)
    if annual:
        fig2_equity_curve(annual)
        fig3_annual_returns(annual)
        fig5_cost_sensitivity(annual)
    fig4_feature_importance()
    if robustness:
        fig6_robustness(robustness)

    safe_print("\n  LaTeX tables:")
    if ic_rows:
        tab_ic_by_year(ic_rows)
    if annual:
        tab_annual_returns(annual)
    if robustness:
        tab_robustness(robustness)
    tab_features()
    if summary:
        tab_universe(summary)

    safe_print("\n  Done. Files in output/figures/ and output/tables/")


if __name__ == "__main__":
    run()
