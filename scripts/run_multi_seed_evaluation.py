#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voltshare.config import ensure_dir, load_yaml
from voltshare.data_generation import generate_all
from voltshare.metrics import BATTERY_SECURITY_THRESHOLDS
from voltshare.plotting import write_paper_figures
from voltshare.simulation import run_all_policies


PAPER_METRICS = [
    "average_charging_completion_rate",
    "basic_reserve_achievement_rate",
    "average_time_to_basic_reserve_hours",
    "exceptional_demand_satisfaction_rate",
    "jain_fairness_index",
    "potential_acceptance_pressure",
    *[f"battery_security_coverage_{threshold}" for threshold in BATTERY_SECURITY_THRESHOLDS],
]


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_summary(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = summary_by_seed.groupby(["scenario", "policy"], as_index=False)
    mean = grouped[PAPER_METRICS].mean()
    std = grouped[PAPER_METRICS].std(ddof=0)
    rows = []
    for _, mean_row in mean.iterrows():
        scenario = mean_row["scenario"]
        policy = mean_row["policy"]
        std_row = std[(std["scenario"] == scenario) & (std["policy"] == policy)].iloc[0]
        row = {"scenario": scenario, "policy": policy}
        for metric in PAPER_METRICS:
            row[f"{metric}_mean"] = float(mean_row[metric])
            row[f"{metric}_std"] = float(std_row[metric])
            row[metric] = _format_mean_std(float(mean_row[metric]), float(std_row[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed VoltShare evaluation.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--weights", default="outputs/results/best_weights.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--work-data", default="outputs/multi_seed/generated")
    parser.add_argument("--results", default="outputs/paper")
    parser.add_argument("--figures", default="outputs/paper/figures")
    args = parser.parse_args()

    config = load_yaml(args.config)
    with Path(args.weights).open("r", encoding="utf-8") as handle:
        weights = json.load(handle)

    data_root = ensure_dir(args.work_data)
    result_dir = ensure_dir(args.results)
    all_summaries = []
    all_sessions = []
    for seed in args.seeds:
        seed_data = ensure_dir(data_root / f"seed_{seed}")
        generate_all(config, "all", seed, seed_data)
        session_results, summary = run_all_policies(seed_data, args.config, weights=weights, seed=seed)
        session_results["seed"] = seed
        summary["seed"] = seed
        all_sessions.append(session_results)
        all_summaries.append(summary)

    session_table = pd.concat(all_sessions, ignore_index=True)
    summary_by_seed = pd.concat(all_summaries, ignore_index=True)
    aggregate = aggregate_summary(summary_by_seed)

    session_table.to_csv(result_dir / "multi_seed_session_results.csv", index=False)
    summary_by_seed.to_csv(result_dir / "multi_seed_summary_by_seed.csv", index=False)
    aggregate.to_csv(result_dir / "paper_summary_mean_std.csv", index=False)
    write_paper_figures(aggregate, args.figures)

    print(f"Wrote multi-seed evaluation to {result_dir}")
    print(aggregate[["scenario", "policy", *PAPER_METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
