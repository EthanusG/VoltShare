from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .config import load_yaml
from .metrics import summarize_sessions
from .models import CommunityConfig, SessionState
from .policies import POLICIES, PolicyContext
from .predictors import add_predictions


def _load_generated_data(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    data_path = Path(data_dir)
    sessions = pd.read_csv(data_path / "sessions.csv")
    with (data_path / "community_config.json").open("r", encoding="utf-8") as handle:
        community = json.load(handle)
    return sessions, community


def _config_for_scenario(config_dict: dict, scenario: str) -> CommunityConfig:
    raw = dict(config_dict[scenario])
    allowed = {
        "scenario",
        "timestep_minutes",
        "reserve_kwh",
        "max_charging_rate_kw",
        "battery_capacity_kwh",
        "community_power_cap_kw",
    }
    return CommunityConfig(**{key: value for key, value in raw.items() if key in allowed})


def _session_states(sessions: pd.DataFrame) -> list[SessionState]:
    states = []
    for row in sessions.to_dict("records"):
        states.append(
            SessionState(
                session_id=int(row["session_id"]),
                user_id=int(row["user_id"]),
                scenario=str(row["scenario"]),
                day=int(row["day"]),
                weekday=int(row["weekday"]),
                arrival_step=int(row["arrival_step"]),
                departure_step=int(row["departure_step"]),
                requested_energy_kwh=float(row["requested_energy_kwh"]),
                initial_energy_kwh=float(row["initial_energy_kwh"]),
                max_rate_kw=float(row["max_rate_kw"]),
                exception_request=bool(row["exception_request"]),
                cooperation_offer=bool(row["cooperation_offer"]),
                offered_energy_kwh=float(row["offered_energy_kwh"]),
                predicted_energy_kwh=float(row["predicted_energy_kwh"]),
                predicted_duration_steps=float(row["predicted_duration_steps"]),
            )
        )
    return states


def _initial_context(
    config: CommunityConfig,
    paper_config: dict,
    weights: dict[str, float],
    baseline_service_ratio: dict[int, float] | None = None,
) -> PolicyContext:
    return PolicyContext(
        config=config,
        weights=weights,
        cooperation=paper_config.get("cooperation", {}),
        exception=paper_config.get("exception", {}),
        user_service_ratio={},
        baseline_service_ratio=baseline_service_ratio or {},
        exception_counts={},
        cooperation_credit={},
    )


def _decay_cooperation_credit(context: PolicyContext, states: list[SessionState]) -> None:
    decay = float(context.cooperation.get("daily_decay", 0.92))
    users = {state.user_id for state in states}
    for user_id in users:
        context.cooperation_credit[user_id] = context.cooperation_credit.get(user_id, 0.0) * decay


def _apply_cooperation_credit(context: PolicyContext, session: SessionState) -> None:
    if not session.cooperation_offer:
        return
    if session.current_total_energy_kwh < context.config.reserve_kwh:
        return
    credit_per_kwh = float(context.cooperation.get("credit_per_kwh", 0.08))
    max_credit = float(context.cooperation.get("max_credit", 1.0))
    current = context.cooperation_credit.get(session.user_id, 0.0)
    gained = session.offered_energy_kwh * credit_per_kwh
    context.cooperation_credit[session.user_id] = min(max_credit, current + gained)


def _update_user_history(context: PolicyContext, session: SessionState) -> None:
    service_ratio = min(1.0, session.delivered_kwh / max(session.requested_energy_kwh, 1e-9))
    previous = context.user_service_ratio.get(session.user_id, service_ratio)
    context.user_service_ratio[session.user_id] = 0.75 * previous + 0.25 * service_ratio
    if session.exception_request:
        context.exception_counts[session.user_id] = context.exception_counts.get(session.user_id, 0) + 1
    _apply_cooperation_credit(context, session)


def _time_to_basic_reserve_hours(state: SessionState, config: CommunityConfig) -> float:
    if state.initial_energy_kwh >= config.reserve_kwh:
        return 0.0
    if state.reserve_achievement_step is None:
        return (state.departure_step - state.arrival_step) * config.timestep_hours
    return (state.reserve_achievement_step - state.arrival_step) * config.timestep_hours


def run_policy(
    sessions: pd.DataFrame,
    community_configs: dict,
    paper_config: dict,
    policy_name: str,
    weights: dict[str, float],
    baseline_service_ratio: dict[int, float] | None = None,
) -> pd.DataFrame:
    policy = POLICIES[policy_name]()
    result_rows = []

    for scenario, group in sessions.groupby("scenario", sort=False):
        scenario_config = _config_for_scenario(community_configs, scenario)
        states = _session_states(group)
        context = _initial_context(scenario_config, paper_config, weights, baseline_service_ratio)
        start = min(state.arrival_step for state in states)
        end = max(state.departure_step for state in states)
        by_id = {state.session_id: state for state in states}
        last_day = None
        physical_violations = 0
        processed_history: set[int] = set()
        for state in states:
            if state.initial_energy_kwh >= scenario_config.reserve_kwh:
                state.reserve_achievement_step = state.arrival_step

        for step in range(start, end + 1):
            day = step // 96
            if last_day is None or day != last_day:
                _decay_cooperation_credit(context, states)
                last_day = day

            active = [state for state in states if state.is_active(step)]
            if not active:
                continue
            allocation = policy.allocate(active, step, context)
            total_kw = sum(allocation.values())
            if total_kw > scenario_config.community_power_cap_kw + 1e-6:
                physical_violations += 1

            for session_id, kw in allocation.items():
                state = by_id[session_id]
                if kw > state.max_rate_kw + 1e-6:
                    physical_violations += 1
                state.delivered_kwh += min(
                    state.remaining_kwh, kw * scenario_config.timestep_hours
                    )
                if (
                    state.reserve_achievement_step is None
                    and state.current_total_energy_kwh >= scenario_config.reserve_kwh
                ):
                    state.reserve_achievement_step = step
                if policy_name == "voltshare":
                    used = min(
                        context.cooperation_credit.get(state.user_id, 0.0),
                        float(context.cooperation.get("max_score_bonus", 0.18)),
                    )
                    state.cooperation_credit_used += used

            if policy_name == "voltshare":
                for state in states:
                    if state.session_id in processed_history:
                        continue
                    if state.departure_step <= step:
                        _update_user_history(context, state)
                        processed_history.add(state.session_id)

        for state in states:
            if policy_name == "voltshare" and state.session_id not in processed_history:
                _update_user_history(context, state)
                processed_history.add(state.session_id)

            service_ratio = min(1.0, state.delivered_kwh / max(state.requested_energy_kwh, 1e-9))
            final_battery_ratio = min(
                1.0,
                state.current_total_energy_kwh / max(scenario_config.battery_capacity_kwh, 1e-9),
            )
            result_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy_name,
                    "session_id": state.session_id,
                    "user_id": state.user_id,
                    "day": state.day,
                    "weekday": state.weekday,
                    "requested_energy_kwh": state.requested_energy_kwh,
                    "delivered_kwh": round(state.delivered_kwh, 4),
                    "service_ratio": service_ratio,
                    "final_battery_ratio": final_battery_ratio,
                    "completed": state.delivered_kwh >= state.requested_energy_kwh - 1e-6,
                    "reserve_satisfied": state.current_total_energy_kwh >= scenario_config.reserve_kwh,
                    "time_to_basic_reserve_hours": _time_to_basic_reserve_hours(
                        state, scenario_config
                    ),
                    "exception_request": state.exception_request,
                    "cooperation_offer": state.cooperation_offer,
                    "cooperation_credit_used": state.cooperation_credit_used,
                    "physical_violation_count": physical_violations,
                }
            )
    return pd.DataFrame(result_rows)


def run_all_policies(
    data_dir: str | Path,
    config_path: str | Path,
    weights: dict[str, float] | None = None,
    policies: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions, community_configs = _load_generated_data(data_dir)
    sessions = add_predictions(sessions, seed=seed)
    paper_config = load_yaml(config_path)
    weights = weights or paper_config.get("policy_weights", {})
    policy_names = policies or ["equal_share", "fcfs", "proportional_share", "voltshare"]

    equal_results = run_policy(
        sessions,
        community_configs,
        paper_config,
        "equal_share",
        weights,
    )
    baseline_by_user = (
        equal_results.groupby(["scenario", "user_id"])["service_ratio"].mean().to_dict()
    )

    all_results = [equal_results]
    for policy in policy_names:
        if policy == "equal_share":
            continue
        all_results.append(
            run_policy(
                deepcopy(sessions),
                community_configs,
                paper_config,
                policy,
                weights,
                baseline_service_ratio=baseline_by_user,
            )
        )
    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_sessions(results)
    return results, summary
