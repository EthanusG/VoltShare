#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voltshare.config import ensure_dir, load_yaml
from voltshare.tuning import tune_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune VoltShare score coefficients.")
    parser.add_argument("--data", default="data/generated")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--search", default="config/tuning_search.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output", default="outputs/results")
    args = parser.parse_args()

    config = load_yaml(args.config)
    trials = args.trials or int(config.get("tuning", {}).get("trials", 120))
    best, table = tune_weights(args.data, args.config, args.search, trials, args.seed)
    out = ensure_dir(args.output)
    table.to_csv(out / "tuning_results.csv", index=False)
    with (out / "best_weights.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(f"Wrote tuned weights to {out / 'best_weights.json'}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
