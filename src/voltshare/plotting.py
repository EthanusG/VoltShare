from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import ensure_dir
from .metrics import BATTERY_SECURITY_THRESHOLDS


def _bar(summary: pd.DataFrame, metric: str, title: str, output: Path) -> None:
    pivot = summary.pivot(index="scenario", columns="policy", values=metric)
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.title(title)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Scenario")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _battery_security(summary: pd.DataFrame, output: Path) -> None:
    metrics = [f"battery_security_coverage_{threshold}" for threshold in BATTERY_SECURITY_THRESHOLDS]
    scenarios = list(summary["scenario"].drop_duplicates())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        scenario_rows = summary[summary["scenario"] == scenario]
        for _, row in scenario_rows.iterrows():
            values = [row[metric] for metric in metrics]
            ax.plot(BATTERY_SECURITY_THRESHOLDS, values, marker="o", label=row["policy"])
        ax.set_title(f"{scenario} demand")
        ax.set_xlabel("Final battery threshold (%)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Coverage rate")
    axes[-1].legend(loc="best")
    fig.suptitle("Layered battery security coverage")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_figures(summary: pd.DataFrame, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    specs = [
        ("average_charging_completion_rate", "Average charging completion by scenario"),
        ("basic_reserve_achievement_rate", "Basic reserve achievement by scenario"),
        ("jain_fairness_index", "Fairness by scenario"),
        ("potential_acceptance_pressure", "Potential acceptance pressure by scenario"),
        ("cooperation_participation_rate", "Cooperation participation by scenario"),
    ]
    for metric, title in specs:
        _bar(summary, metric, title, out / f"{metric}.png")
    _battery_security(summary, out / "battery_security_coverage.png")


def _paper_high_metric_bars(summary: pd.DataFrame, output: Path) -> None:
    high = summary[summary["scenario"] == "high"].copy()
    metrics = [
        "average_charging_completion_rate",
        "exceptional_demand_satisfaction_rate",
        "jain_fairness_index",
        "potential_acceptance_pressure",
    ]
    labels = [
        "Avg. charging\ncompletion",
        "Exceptional-demand\nsatisfaction",
        "Jain fairness",
        "Acceptance\npressure",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.6))
    colors = {
        "equal_share": "#8da0cb",
        "fcfs": "#fc8d62",
        "proportional_share": "#66c2a5",
        "voltshare": "#222222",
    }
    for ax, metric, label in zip(axes, metrics, labels):
        values = high[f"{metric}_mean"]
        errors = high[f"{metric}_std"]
        ax.bar(
            high["policy"],
            values,
            yerr=errors,
            capsize=3,
            color=[colors.get(policy, "#999999") for policy in high["policy"]],
        )
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("High-demand scenario: mean ± std across random seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _paper_high_battery_security(summary: pd.DataFrame, output: Path) -> None:
    high = summary[summary["scenario"] == "high"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for _, row in high.iterrows():
        means = [row[f"battery_security_coverage_{threshold}_mean"] for threshold in BATTERY_SECURITY_THRESHOLDS]
        stds = [row[f"battery_security_coverage_{threshold}_std"] for threshold in BATTERY_SECURITY_THRESHOLDS]
        ax.plot(BATTERY_SECURITY_THRESHOLDS, means, marker="o", label=row["policy"])
        ax.fill_between(
            BATTERY_SECURITY_THRESHOLDS,
            [max(0.0, mean - std) for mean, std in zip(means, stds)],
            [min(1.0, mean + std) for mean, std in zip(means, stds)],
            alpha=0.12,
        )
    ax.set_title("High-demand layered battery security coverage")
    ax.set_xlabel("Final battery threshold (%)")
    ax.set_ylabel("Coverage rate")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_paper_figures(summary_mean_std: pd.DataFrame, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    _paper_high_metric_bars(summary_mean_std, out / "paper_high_metric_bars.png")
    _paper_high_battery_security(summary_mean_std, out / "paper_high_battery_security.png")
