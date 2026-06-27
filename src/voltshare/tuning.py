from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_yaml
from .simulation import run_all_policies


def sample_weights(search_space: dict, rng: np.random.Generator) -> dict[str, float]:
    sampled = {
        name: float(rng.uniform(bounds[0], bounds[1]))
        for name, bounds in search_space["weights"].items()
    }
    total = sum(value for value in sampled.values() if value > 0)
    return {name: value / total for name, value in sampled.items()}


def balanced_objective(summary: pd.DataFrame, objective_weights: dict) -> float:
    rows = summary[summary["policy"] == "voltshare"]
    if rows.empty:
        return -1e9
    if rows["physical_violation_count"].sum() > 0:
        return -1e9
    score = 0.0
    for metric, weight in objective_weights.items():
        if metric not in rows:
            continue
        value = float(rows[metric].mean())
        if metric == "average_time_to_basic_reserve_hours":
            value = min(value / 6.0, 1.0)
        score += float(weight) * value
    return score


def tune_weights(
    data_dir: str | Path,
    config_path: str | Path,
    search_path: str | Path,
    trials: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    search_space = load_yaml(search_path)
    rng = np.random.default_rng(seed)
    rows = []
    best_score = -1e18
    best_weights = {}
    for trial in range(trials):
        weights = sample_weights(search_space, rng)
        _, summary = run_all_policies(data_dir, config_path, weights=weights, seed=seed + trial)
        score = balanced_objective(summary, search_space.get("objective", {}))
        row = {"trial": trial, "objective_score": score, **weights}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, pd.DataFrame(rows)
