#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voltshare.config import ensure_dir
from voltshare.plotting import write_figures
from voltshare.simulation import run_all_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoltShare policy simulations.")
    parser.add_argument("--data", default="data/generated")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="outputs/results")
    parser.add_argument("--figures", default="outputs/figures")
    args = parser.parse_args()

    weights = None
    if args.weights:
        with Path(args.weights).open("r", encoding="utf-8") as handle:
            weights = json.load(handle)

    results, summary = run_all_policies(args.data, args.config, weights=weights, seed=args.seed)
    results_dir = ensure_dir(args.results)
    results.to_csv(results_dir / "session_results.csv", index=False)
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    write_figures(summary, args.figures)
    print(f"Wrote results to {results_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
