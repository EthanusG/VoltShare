from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voltshare.tuning import balanced_objective, sample_weights


def test_sample_weights_are_normalized():
    space = {"weights": {"a": [0.1, 1.0], "b": [0.1, 1.0], "c": [0.1, 1.0]}}
    weights = sample_weights(space, np.random.default_rng(1))
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_balanced_objective_rejects_physical_violations():
    summary = pd.DataFrame(
        [
            {
                "policy": "voltshare",
                "average_charging_completion_rate": 1.0,
                "physical_violation_count": 1,
            }
        ]
    )
    assert balanced_objective(summary, {"average_charging_completion_rate": 1.0}) < -1e8
