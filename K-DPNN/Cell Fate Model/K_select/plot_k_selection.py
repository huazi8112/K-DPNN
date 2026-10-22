"""从独立A/B/C评价文件绘制K选择图，不预设被选中的K。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_summary(folder: Path) -> tuple[dict, list[dict], int]:
    """从逐条件记录复算图示数值，并核对汇总文件。"""
    selection = json.loads((folder / "budget_selection.json").read_text(encoding="utf-8"))
    if selection.get("new_independent_ABC_SDE_data") is not True:
        raise ValueError("输入必须来自独立A/B/C方案")
    threshold = float(selection["threshold_H"])
    coverage = float(selection["coverage_required"])
    count = int(selection["n_conditions"])
    rank = math.ceil(count * coverage - 1e-12)
    candidates = sorted(map(int, selection["candidate_K"]))
    details = read_csv(folder / "abc_million_gmm_details.csv")
    recorded = {int(row["K"]): row for row in read_csv(folder / "coverage_summary_by_K.csv")}
    verified = []
    condition_ids = None
    for k in candidates:
        rows = [row for row in details if int(row["K"]) == k]
        ids = {row["condition_id"] for row in rows}
        if len(rows) != count or len(ids) != count:
            raise ValueError(f"K={k}的条件记录不完整或重复")
        if condition_ids is not None and ids != condition_ids:
            raise ValueError("不同K必须评价同一组代表条件")
        condition_ids = ids
        if any(row["converged"].lower() != "true" for row in rows):
            raise ValueError(f"K={k}包含未收敛的拟合")
        if any(int(row[key]) != 1_000_000 for row in rows for key in ("n_A", "n_B", "n_C")):
            raise ValueError("A/B/C每组应包含100万个状态点")
        errors = np.array([(float(row["H_GMM_B"]) + float(row["H_GMM_C"])) / 2 for row in rows])
        if not np.isfinite(errors).all() or not np.allclose(
            errors, [float(row["H_GMM_mean"]) for row in rows], rtol=0, atol=1e-14
        ):
            raise ValueError("条件级平均Hellinger距离不匹配")
        rank_error = float(np.sort(errors)[rank - 1])
        passed = int(np.count_nonzero(errors <= threshold))
        summary = recorded[k]
        if (abs(rank_error - float(summary["coverage_rank_H_27th"])) > 1e-12
                or passed != int(summary["n_pass_H_le_0_10"])
                or abs(passed / count - float(summary["pass_fraction"])) > 1e-12):
            raise ValueError(f"K={k}的逐条件记录与汇总不匹配")
        verified.append({"K": k, "rank_H": rank_error,
                         "n_pass": passed, "pass_fraction": passed / count})
    passing = [row["K"] for row in verified if row["n_pass"] >= rank]
    selected = min(passing) if passing else None
    if selected != selection["selected_K"]:
        raise ValueError("按精度预算重新计算的K与保存值不匹配")
    return selection, verified, rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path,
                        help="绘图输出目录；默认使用结果目录下的figures")
    args = parser.parse_args()
    selection, rows, rank = load_summary(args.results_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "savefig.facecolor": "white",
    })
    threshold = float(selection["threshold_H"])
    coverage = float(selection["coverage_required"])
    count = int(selection["n_conditions"])
    selected = selection["selected_K"]
    ks = np.array([row["K"] for row in rows])
    errors = np.array([row["rank_H"] for row in rows])
    rates = np.array([row["pass_fraction"] * 100 for row in rows])
    blue, green = "#2376AC", "#267447"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
    for ax in axes:
        if selected is not None:
            ax.axvspan(selected - 0.14, selected + 0.14, facecolor=green, alpha=0.10, zorder=0)
        ax.set_xlim(min(ks) - 0.28, max(ks) + 0.28)
        ax.set_xticks(ks)
        ax.set_xlabel("Gaussian component count, $K$")
        ax.grid(axis="y", linewidth=0.5, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.8, length=3)
    line, = axes[0].plot(ks, errors, "o-", color=blue, linewidth=1.5,
                         markersize=5.5, label="Independent A/B/C experiment", zorder=3)
    budget = axes[0].axhline(threshold, color="#333333", linestyle="--", linewidth=1,
                             label=f"$H$ budget = {threshold:g}")
    axes[0].set_title("A  Representation error", loc="left", fontweight="bold", pad=9)
    axes[0].set_ylabel(f"Hellinger distance at rank {rank}")
    axes[0].set_ylim(0, max(float(errors.max()), threshold) * 1.23)
    for k, value in zip(ks, errors):
        axes[0].annotate(f"{value:.4f}", (k, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=9,
                         color=green if k == selected else "#222222")
    handles = [line, budget]
    if selected is not None:
        handles.append(Patch(facecolor=green, alpha=0.12, label=f"Selected $K$ = {selected}"))
    axes[0].legend(handles=handles, loc="lower left", frameon=False)
    axes[1].plot(ks, rates, "o-", color=blue, linewidth=1.5, markersize=5.5, zorder=3)
    axes[1].axhline(coverage * 100, color="#333333", linestyle="--", linewidth=1,
                    label=f"Required coverage = {coverage:.0%}")
    axes[1].set_title("B  Empirical pass rate", loc="left", fontweight="bold", pad=9)
    axes[1].set_ylabel(f"Conditions with $H \\leq {threshold:g}$ (%)")
    axes[1].set_ylim(0, 113)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])
    for row in rows:
        axes[1].annotate(f"{row['pass_fraction']:.1%}\n({row['n_pass']}/{count})",
                         (row["K"], row["pass_fraction"] * 100), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=9,
                         color=green if row["K"] == selected else "#222222")
    axes[1].legend(loc="lower left", frameon=False)
    for ax in axes:
        for tick in ax.get_xticklabels():
            if tick.get_text() == str(selected):
                tick.set_color(green)
                tick.set_fontweight("bold")
    output = args.output_dir or (args.results_dir / "figures")
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "accuracy_budget_k_prior_ABC"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"已保存K选择图：{stem.resolve()} (.png / .pdf)")


if __name__ == "__main__":
    main()
