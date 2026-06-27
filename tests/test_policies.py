from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voltshare.models import CommunityConfig, SessionState
from voltshare.policies import EqualSharePolicy, PolicyContext, VoltSharePolicy


def make_config() -> CommunityConfig:
    return CommunityConfig(
        scenario="test",
        timestep_minutes=15,
        reserve_kwh=8.0,
        max_charging_rate_kw=7.0,
        battery_capacity_kwh=60.0,
        community_power_cap_kw=10.0,
    )


def make_session(session_id: int, user_id: int, initial: float, requested: float = 12.0) -> SessionState:
    return SessionState(
        session_id=session_id,
        user_id=user_id,
        scenario="test",
        day=0,
        weekday=0,
        arrival_step=0,
        departure_step=20,
        requested_energy_kwh=requested,
        initial_energy_kwh=initial,
        max_rate_kw=7.0,
        exception_request=False,
        cooperation_offer=False,
        offered_energy_kwh=0.0,
        predicted_energy_kwh=12.0,
        predicted_duration_steps=20,
    )


def make_context() -> PolicyContext:
    return PolicyContext(
        config=make_config(),
        weights={
            "remaining_demand": 0.3,
            "urgency": 0.25,
            "fairness": 0.2,
            "acceptance": 0.1,
            "exception_penalty": 0.05,
            "cooperation_credit": 0.1,
        },
        cooperation={"max_score_bonus": 0.18, "max_credit": 1.0},
        exception={"frequent_request_threshold": 3, "penalty_per_extra_request": 0.06, "max_penalty": 0.24},
        user_service_ratio={},
        baseline_service_ratio={},
        exception_counts={},
        cooperation_credit={},
    )


def test_equal_share_never_exceeds_power_cap():
    context = make_context()
    sessions = [make_session(1, 1, 5.0), make_session(2, 2, 5.0)]
    allocation = EqualSharePolicy().allocate(sessions, 0, context)
    assert sum(allocation.values()) <= context.config.community_power_cap_kw
    assert all(value <= 7.0 for value in allocation.values())


def test_voltshare_prioritizes_below_reserve_users():
    context = make_context()
    low = make_session(1, 1, 2.0)
    high = make_session(2, 2, 20.0)
    allocation = VoltSharePolicy().allocate([low, high], 0, context)
    assert allocation[low.session_id] >= allocation[high.session_id]


def test_cooperation_credit_increases_score():
    context = make_context()
    session = make_session(1, 1, 15.0)
    policy = VoltSharePolicy()
    without_credit = policy._score(session, 0, context)
    context.cooperation_credit[1] = 1.0
    with_credit = policy._score(session, 0, context)
    assert with_credit > without_credit


def test_voltshare_protects_reasonable_exception_requests():
    context = make_context()
    context.exception["emergency_cap_fraction"] = 0.5
    context.weights["fairness"] = 0.45
    context.weights["urgency"] = 0.10

    historically_underserved = make_session(1, 1, 18.0, requested=20.0)
    urgent = make_session(2, 2, 18.0, requested=12.0)
    urgent.exception_request = True
    urgent.departure_step = 8
    context.user_service_ratio = {1: 0.1, 2: 1.0}

    allocation = VoltSharePolicy().allocate([historically_underserved, urgent], 0, context)

    assert allocation[urgent.session_id] >= context.config.community_power_cap_kw * 0.5


def test_score_sharpness_concentrates_residual_allocation():
    policy = VoltSharePolicy()
    urgent_context = make_context()
    urgent_context.weights = {
        "remaining_demand": 0.0,
        "urgency": 1.0,
        "fairness": 0.0,
        "acceptance": 0.0,
        "exception_penalty": 0.0,
        "cooperation_credit": 0.0,
    }
    urgent = make_session(1, 1, 18.0, requested=12.0)
    relaxed = make_session(2, 2, 18.0, requested=12.0)
    urgent.max_rate_kw = 20.0
    relaxed.max_rate_kw = 20.0
    urgent.departure_step = 4
    relaxed.departure_step = 40

    normal = policy.allocate([urgent, relaxed], 0, urgent_context)
    urgent_context.exception["score_sharpness"] = 3.0
    sharpened = policy.allocate([urgent, relaxed], 0, urgent_context)

    assert sharpened[urgent.session_id] > normal[urgent.session_id]
