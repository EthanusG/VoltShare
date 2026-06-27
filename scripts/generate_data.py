#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voltshare.config import load_yaml
from voltshare.data_generation import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic VoltShare charging data.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--scenario", default="all", choices=["all", "low", "medium", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/generated")
    args = parser.parse_args()

    config = load_yaml(args.config)
    generate_all(config, args.scenario, args.seed, args.output)
    print(f"Generated data in {args.output}")


if __name__ == "__main__":
    main()
