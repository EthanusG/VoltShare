from __future__ import annotations

import numpy as np
import pandas as pd


BATTERY_SECURITY_THRESHOLDS = tuple(range(10, 80, 10))


def jain_index(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    denominator = arr.size * float(np.sum(arr**2))
    if denominator <= 0:
        return 1.0
    return float((np.sum(arr) ** 2) / denominator)


def summarize_sessions(results: pd.DataFrame, baseline_policy: str = "equal_share") -> pd.DataFrame:
    baseline = results[results["policy"] == baseline_policy][
        ["scenario", "session_id", "service_ratio"]
    ].rename(columns={"service_ratio": "baseline_service_ratio"})
    merged = results.merge(baseline, on=["scenario", "session_id"], how="left")
    merged["relative_loss"] = (
        merged["baseline_service_ratio"].fillna(0) - merged["service_ratio"]
    ).clip(lower=0)

    rows: list[dict[str, float | str]] = []
    for (scenario, policy), group in merged.groupby(["scenario", "policy"], sort=True):
        service = group["service_ratio"].to_numpy()
        exceptional = group[group["exception_request"]]
        row: dict[str, float | int | str] = {
            "scenario": scenario,
            "policy": policy,
            "sessions": int(len(group)),
            "average_charging_completion_rate": float(group["final_battery_ratio"].mean()),
            "full_request_completion_rate_diagnostic": float(group["completed"].mean()),
            "basic_reserve_achievement_rate": float(group["reserve_satisfied"].mean()),
            "average_time_to_basic_reserve_hours": float(
                group["time_to_basic_reserve_hours"].mean()
            ),
            "exceptional_demand_satisfaction_rate": float(
                exceptional["completed"].mean() if len(exceptional) else 1.0
            ),
            "jain_fairness_index": jain_index(service),
            "potential_acceptance_pressure": float(group["relative_loss"].mean()),
            "cooperation_participation_rate": float(group["cooperation_offer"].mean()),
            "cooperation_credit_used": float(group["cooperation_credit_used"].mean()),
            "physical_violation_count": int(group["physical_violation_count"].sum()),
        }
        for threshold in BATTERY_SECURITY_THRESHOLDS:
            row[f"battery_security_coverage_{threshold}"] = float(
                (group["final_battery_ratio"] >= threshold / 100).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)
