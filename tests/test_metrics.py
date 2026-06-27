from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voltshare.metrics import jain_index, summarize_sessions


def test_jain_index_is_one_for_equal_distribution():
    assert jain_index([1, 1, 1, 1]) == 1.0


def test_jain_index_is_lower_for_unequal_distribution():
    assert jain_index([1, 0, 0, 0]) < 0.4


def test_jain_index_handles_zero_distribution():
    assert np.isclose(jain_index([0, 0, 0]), 1.0)


def test_average_charging_completion_rate_uses_final_battery_ratio():
    results = pd.DataFrame(
        [
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 1,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.80,
                "completed": True,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 2,
                "service_ratio": 0.5,
                "final_battery_ratio": 0.40,
                "completed": False,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
        ]
    )

    summary = summarize_sessions(results)

    assert np.isclose(summary.loc[0, "average_charging_completion_rate"], 0.60)
    assert np.isclose(summary.loc[0, "full_request_completion_rate_diagnostic"], 0.50)


def test_summary_uses_pdf_aligned_primary_metrics():
    results = pd.DataFrame(
        [
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 1,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.80,
                "completed": True,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.25,
                "exception_request": True,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            }
        ]
    )

    summary = summarize_sessions(results)

    assert {
        "average_charging_completion_rate",
        "basic_reserve_achievement_rate",
        "average_time_to_basic_reserve_hours",
        "exceptional_demand_satisfaction_rate",
        "jain_fairness_index",
        "potential_acceptance_pressure",
    }.issubset(summary.columns)


def test_summary_reports_layered_battery_security_coverage():
    results = pd.DataFrame(
        [
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 1,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.15,
                "completed": False,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 2,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.35,
                "completed": False,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 3,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.75,
                "completed": True,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
        ]
    )

    summary = summarize_sessions(results)

    assert np.isclose(summary.loc[0, "battery_security_coverage_10"], 1.0)
    assert np.isclose(summary.loc[0, "battery_security_coverage_30"], 2 / 3)
    assert np.isclose(summary.loc[0, "battery_security_coverage_70"], 1 / 3)


def test_average_time_to_basic_reserve_uses_session_values():
    results = pd.DataFrame(
        [
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 1,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.50,
                "completed": False,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 0.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 2,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.50,
                "completed": False,
                "reserve_satisfied": True,
                "time_to_basic_reserve_hours": 2.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
            {
                "scenario": "stress",
                "policy": "equal_share",
                "session_id": 3,
                "service_ratio": 1.0,
                "final_battery_ratio": 0.50,
                "completed": False,
                "reserve_satisfied": False,
                "time_to_basic_reserve_hours": 4.0,
                "exception_request": False,
                "cooperation_offer": False,
                "cooperation_credit_used": 0.0,
                "physical_violation_count": 0,
            },
        ]
    )

    summary = summarize_sessions(results)

    assert np.isclose(summary.loc[0, "average_time_to_basic_reserve_hours"], 2.0)
