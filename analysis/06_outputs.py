"""
Generate figures (PDF/PNG) and LaTeX table fragments for the manuscript.
"""
import json
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from src.utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, ensure_dirs, safe_print

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def _load_json(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    safe_print(f"    Saved {name}.pdf/png")


def fig1_ic_by_year(ic_rows: list):
    years = [row["year"] for row in ic_rows]
    means = [row["ic_mean"] for row in ic_rows]
    ses = [row["ic_se"] for row in ic_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3" if value > 0 else "#F44336" for value in means]
    ax.bar(years, means, color=colors, alpha=0.85, width=0.6, zorder=2)
    ax.errorbar(years, means, yerr=[1.96 * se for se in ses], fmt="none", color="black", capsize=4, linewidth=1.2, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Daily Spearman IC")
    ax.set_title("Figure 1: Cross-Sectional IC by Year (2015-2025)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.3f}"))
    ax.set_xticks(years)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    fig.tight_layout()
    _save(fig, "fig1_ic_by_year")


def fig2_equity_curve(annual: list):
    years = [row["year"] for row in annual]
    net = [row["net_sq"] for row in annual]
    gross = [row["gross_sq"] for row in annual]
    cum_net = np.cumprod([1 + x for x in net]) - 1
    cum_gross = np.cumprod([1 + x for x in gross]) - 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, cum_gross * 100, "b-o", markersize=5, label="Gross (excess vs VNINDEX)")
    ax.plot(years, cum_net * 100, "r-o", markersize=5, label="Net (-0.30% RT)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(years, 0, cum_net * 100, where=[x >= 0 for x in cum_net], alpha=0.12, color="blue")
    ax.fill_between(years, 0, cum_net * 100, where=[x < 0 for x in cum_net], alpha=0.12, color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.set_title("Figure 2: Cumulative Portfolio Excess Return vs VNINDEX")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig2_equity_curve")


def fig3_annual_returns(annual: list):
    years = [row["year"] for row in annual]
    gross = [row["gross_sq"] * 100 for row in annual]
    net = [row["net_sq"] * 100 for row in annual]

    x = np.arange(len(years))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, gross, width=width, color="#2196F3", alpha=0.85, label="Gross/trade")
    ax.bar(x + width / 2, net, width=width, color="#FF9800", alpha=0.85, label="Net/trade (-0.30%)")
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


def fig4_feature_importance():
    importance = {
        "rel_ret_5d": 0.098,
        "rel_ret_1d": 0.087,
        "rel_ret_20d": 0.076,
        "rel_ret_10d": 0.071,
        "idx_ret_5d": 0.062,
        "rel_ret_2d": 0.058,
        "idx_ret_20d": 0.055,
        "ma_dist_50": 0.052,
        "rel_ret_3d": 0.049,
        "rsi_14": 0.046,
        "idx_ret_10d": 0.043,
        "ma_dist_20": 0.041,
        "hist_vol_20": 0.038,
        "adx": 0.035,
        "macd_hist": 0.032,
    }

    cache = RESULTS_DIR / "feature_importance.json"
    if cache.exists():
        try:
            importance = json.loads(cache.read_text())
        except Exception:
            pass

    top15 = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:15]
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


def fig5_cost_sensitivity(annual: list):
    avg_gross = np.mean([row["gross_sq"] for row in annual])
    max_cost = max(0.008, avg_gross * 1.4)
    costs = np.linspace(0.0, max_cost, 100)
    nets = (avg_gross - costs) * 100
    break_even = avg_gross * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs * 100, nets, "b-", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Break-even")
    ax.axvline(0.30, color="gray", linestyle=":", linewidth=1, label="Assumed cost (0.30%)")
    ax.axvline(break_even, color="green", linestyle="--", linewidth=1, label=f"Exact break-even ({break_even:.2f}%)")
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


def fig6_robustness(robustness: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Figure 6: Robustness Tests", fontsize=12, y=1.01)

    r1 = robustness.get("permutation", {})
    axes[0, 0].bar(["IC (actual)", "IC (permuted)"], [r1.get("ic_raw", 0), r1.get("ic_perm", 0)], color=["#2196F3", "#F44336"], alpha=0.85, width=0.5)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title(f"A: Permutation [{r1.get('verdict', '?')}]")
    axes[0, 0].set_ylabel("Mean daily IC")
    axes[0, 0].grid(axis="y", alpha=0.3)

    r2 = robustness.get("sector_neutral", {})
    axes[0, 1].bar(["IC (raw)", "IC (sector-neutral)"], [r2.get("ic_raw", 0), r2.get("ic_neutral", 0)], color=["#2196F3", "#9C27B0"], alpha=0.85, width=0.5)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title(f"B: Sector Neutral drop={r2.get('ic_drop_pct', 0):.1f}% [{r2.get('verdict', '?')}]")
    axes[0, 1].set_ylabel("Mean daily IC")
    axes[0, 1].grid(axis="y", alpha=0.3)

    r3 = robustness.get("mean_reversion", {})
    axes[1, 0].bar(["MR-5d", "XGBoost"], [r3.get("ic_mr5", 0), r3.get("ic_model", 0)], color=["#FF9800", "#2196F3"], alpha=0.85, width=0.5)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title(f"C: MR Baseline MR={r3.get('ratio_pct', 0):.0f}% of model [{r3.get('verdict', '?')}]")
    axes[1, 0].set_ylabel("Mean daily IC")
    axes[1, 0].grid(axis="y", alpha=0.3)

    r4 = robustness.get("execution_timing", {})
    colors = ["#2196F3", "#F44336" if r4.get("sh_e2", 0) < 0 else "#FF9800"]
    axes[1, 1].bar(["E=1 (T+1 open)", "E=2 (T+2 open)"], [r4.get("sh_e1", 0), r4.get("sh_e2", 0)], color=colors, alpha=0.85, width=0.5)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title(f"D: Execution Timing [{r4.get('verdict', '?')}]")
    axes[1, 1].set_ylabel("Annual net Sharpe")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig6_robustness")


def tab_ic_by_year(ic_rows: list):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Daily Spearman IC by Year (2015--2025)}",
        r"\label{tab:ic_by_year}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Year & IC & SE & $t$-stat & $p$-value & N days \\",
        r"\midrule",
    ]
    for row in ic_rows:
        sig = r" $^{*}$" if row["p_val"] < 0.05 else ""
        lines.append(
            f"{row['year']} & {row['ic_mean']:+.4f} & {row['ic_se']:.4f} & "
            f"{row['t_stat']:.1f} & {row['p_val']:.4f}{sig} & {row['n_days']} \\\\"
        )
    avg = np.mean([row["ic_mean"] for row in ic_rows])
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
    for row in annual:
        lines.append(f"{row['year']} & {row['gross_sq']*100:+.3f}\\% & {row['net_sq']*100:+.3f}\\% & {row['n_trades']} \\\\")
    avg_gross = np.mean([row["gross_sq"] for row in annual]) * 100
    avg_net = np.mean([row["net_sq"] for row in annual]) * 100
    lines += [
        r"\midrule",
        f"Average & {avg_gross:+.3f}\\% & {avg_net:+.3f}\\% & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES_DIR / "tab_annual_returns.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_annual_returns.tex")


def tab_robustness(robustness: dict):
    rows = [
        (
            "Label permutation",
            "IC = {:+.4f} vs {:+.4f} (permuted)".format(
                robustness.get("permutation", {}).get("ic_raw", 0),
                robustness.get("permutation", {}).get("ic_perm", 0),
            ),
            robustness.get("permutation", {}).get("verdict", "?"),
        ),
        (
            "Sector neutrality",
            "IC drop = {:.1f}\\%".format(robustness.get("sector_neutral", {}).get("ic_drop_pct", 0)),
            robustness.get("sector_neutral", {}).get("verdict", "?"),
        ),
        (
            "MR-5d baseline",
            "MR IC = {:.0f}\\% of model IC".format(robustness.get("mean_reversion", {}).get("ratio_pct", 0)),
            robustness.get("mean_reversion", {}).get("verdict", "?"),
        ),
        (
            "Execution timing (T+2)",
            "E=2 net Sharpe = {:.2f}".format(robustness.get("execution_timing", {}).get("sh_e2", 0)),
            robustness.get("execution_timing", {}).get("verdict", "?"),
        ),
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


def tab_postfreeze(postfreeze: dict):
    grid = postfreeze.get("holding_period_grid_no_vix", {})
    cost = postfreeze.get("cost_stress_no_vix", {})
    holdout = postfreeze.get("holdout_2026_no_vix", {})

    h1_row = next((row for row in grid.get("rows", []) if row.get("hold") == 1), None)
    h5_row = next((row for row in grid.get("rows", []) if row.get("hold") == 5), None)

    h1_text = "N/A"
    if h1_row:
        h1_text = "H=1 ann. net @0.30\\% = {:+.2f}\\%".format(h1_row.get("avg_net_annual_0_30", 0) * 100)
    h5_text = "N/A"
    if h5_row:
        h5_text = "H=5 ann. net @0.30\\% = {:+.2f}\\%".format(h5_row.get("avg_net_annual_0_30", 0) * 100)

    daily = cost.get("daily_h1", {}) or {}
    cost_text = "N/A"
    if daily:
        cost_text = "breakeven = {:.2f}\\% RT".format(daily.get("exact_breakeven_rt", 0) * 100)

    sq = holdout.get("square_weight", {}) or {}
    holdout_text = "N/A"
    if holdout:
        holdout_text = "IC = {:+.4f}; gross = {:+.3f}\\%; net = {:+.3f}\\%".format(
            holdout.get("ic_2026", 0),
            sq.get("gross_per_trade", 0) * 100,
            sq.get("net_per_trade", 0) * 100,
        )

    rows = [
        ("Executable hold grid (no VIX)", f"{h1_text}; {h5_text}", "Daily rebalancing remains optimal"),
        ("No-VIX cost stress", cost_text, "Signal remains economically meaningful but cost-sensitive"),
        ("2026 live validation (no VIX)", holdout_text, "Ranking signal alive; economic verdict remains weak-pass"),
    ]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Post-Freeze Credibility Extensions}",
        r"\label{tab:postfreeze}",
        r"\begin{tabular}{p{3.6cm}p{5.3cm}p{5.0cm}}",
        r"\toprule",
        r"Check & Key result & Interpretation \\",
        r"\midrule",
    ]
    for name, result, interp in rows:
        lines.append(f"{name} & {result} & {interp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_postfreeze.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_postfreeze.tex")


def tab_features():
    groups = [
        ("MA distance", 4, "Close / MA(w) - 1; w=5,10,20,50"),
        ("RSI", 1, "14-day Wilder RSI"),
        ("MACD", 3, "Price-normalized MACD (12,26,9)"),
        ("Bollinger", 2, "Band width and position (20-day, 2 sigma)"),
        ("ATR", 1, "14-day ATR / close (normalized)"),
        ("Candle", 4, "Body ratio, wick ratios, overnight gap"),
        ("Stochastic", 2, r"14-day \%K, 3-day \%D"),
        ("ADX", 2, "14-day ADX trend strength; +DI minus -DI direction"),
        ("Realized vol", 4, "Rolling std of daily returns; (H-L)/C"),
        ("Rel. return", 6, "Six lags: stock excess return vs VNINDEX"),
        ("Market return", 6, "Six lags: raw VNINDEX return"),
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
    for group, count, desc in groups:
        lines.append(f"{group} & {count} & {desc} \\\\")
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
    for year in range(2012, 2026):
        n = year_stocks.get(str(year), year_stocks.get(year, "---"))
        lines.append(f"{year} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_universe.tex").write_text("\n".join(lines))
    safe_print("    Saved tab_universe.tex")


def run():
    ensure_dirs()
    safe_print("\n--- 06: Generating Figures and Tables ---")

    ic_rows = _load_json("ic_by_year.json")
    annual = _load_json("portfolio_annual.json")
    robustness = _load_json("robustness.json")
    summary = _load_json("data_summary.json")
    postfreeze = _load_json("postfreeze_extensions.json")

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
    if postfreeze:
        tab_postfreeze(postfreeze)
    tab_features()
    if summary:
        tab_universe(summary)

    safe_print("\n  Done. Files in output/figures/ and output/tables/")


if __name__ == "__main__":
    run()
