from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_dir


STEPS_PER_DAY = 96


def _clip_step(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def generate_scenario(config: dict, scenario: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = config["simulation"]
    spec = config["scenarios"][scenario]
    rng = np.random.default_rng(seed)

    users_count = int(spec.get("users", base["users"]))
    days = int(spec.get("days", base["days"]))
    max_rate = float(base["max_charging_rate_kw"])
    battery_capacity = float(base["battery_capacity_kwh"])
    duration_multiplier = float(spec.get("duration_multiplier", 1.0))
    energy_multiplier = float(spec.get("energy_multiplier", 1.0))
    initial_min = float(spec.get("initial_energy_min_kwh", 8.0))
    initial_max_fraction = float(spec.get("initial_energy_max_fraction", 0.55))
    exception_duration_low = float(spec.get("exception_duration_multiplier_low", 0.45))
    exception_duration_high = float(spec.get("exception_duration_multiplier_high", 0.75))
    exception_extra_low = float(spec.get("exception_extra_kwh_low", 8.0))
    exception_extra_high = float(spec.get("exception_extra_kwh_high", 18.0))

    users = []
    sessions = []
    session_id = 0
    for user_id in range(users_count):
        weekday_arrival = rng.normal(74, 6, size=7).clip(62, 88)
        weekday_duration = rng.normal(44, 8, size=7).clip(24, 64)
        weekday_energy = rng.normal(17, 5, size=7).clip(7, 34)
        users.append(
            {
                "user_id": user_id,
                "home_charger_kw": max_rate,
                "habit_arrival_step_mean": float(np.mean(weekday_arrival)),
                "habit_duration_steps_mean": float(np.mean(weekday_duration)),
                "habit_energy_kwh_mean": float(np.mean(weekday_energy)),
            }
        )
        for day in range(days):
            weekday = day % 7
            drift = 0.08 * day
            if rng.random() > float(spec["session_probability"]):
                continue
            arrival_in_day = _clip_step(
                rng.normal(weekday_arrival[weekday] + drift, 4.5), 54, 93
            )
            duration = _clip_step(
                rng.normal(weekday_duration[weekday] * duration_multiplier, 7), 8, 72
            )
            requested = float(max(4, rng.normal(weekday_energy[weekday] * energy_multiplier, 4.0)))
            exception_request = bool(rng.random() < float(spec["exception_probability"]))
            if exception_request:
                duration = _clip_step(
                    duration * rng.uniform(exception_duration_low, exception_duration_high), 6, 48
                )
                requested = float(min(52, requested + rng.uniform(exception_extra_low, exception_extra_high)))

            non_urgent = duration > 42 and requested < 24
            cooperation_offer = bool(
                non_urgent and rng.random() < float(spec["cooperation_probability"])
            )
            offered = float(rng.uniform(1.5, 5.0)) if cooperation_offer else 0.0
            initial_energy = float(rng.uniform(initial_min, battery_capacity * initial_max_fraction))

            arrival_step = day * STEPS_PER_DAY + arrival_in_day
            departure_step = arrival_step + duration
            sessions.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "scenario": scenario,
                    "day": day,
                    "weekday": weekday,
                    "arrival_step": arrival_step,
                    "departure_step": departure_step,
                    "arrival_step_of_day": arrival_in_day,
                    "duration_steps": duration,
                    "requested_energy_kwh": round(requested, 3),
                    "initial_energy_kwh": round(initial_energy, 3),
                    "max_rate_kw": max_rate,
                    "exception_request": exception_request,
                    "cooperation_offer": cooperation_offer,
                    "offered_energy_kwh": round(offered, 3),
                }
            )
            session_id += 1

    scenario_config = {
        "scenario": scenario,
        "users": users_count,
        "days": days,
        "timestep_minutes": int(base["timestep_minutes"]),
        "reserve_kwh": float(base["reserve_kwh"]),
        "max_charging_rate_kw": max_rate,
        "battery_capacity_kwh": battery_capacity,
        "community_power_cap_kw": float(spec["community_power_cap_kw"]),
    }
    return pd.DataFrame(users), pd.DataFrame(sessions), scenario_config


def generate_all(config: dict, scenario: str, seed: int, output_dir: str | Path) -> None:
    output = ensure_dir(output_dir)
    scenarios = list(config["scenarios"]) if scenario == "all" else [scenario]

    all_users = []
    all_sessions = []
    all_configs = {}
    for index, name in enumerate(scenarios):
        users, sessions, scenario_config = generate_scenario(config, name, seed + index * 101)
        all_users.append(users.assign(scenario=name))
        all_sessions.append(sessions)
        all_configs[name] = scenario_config

    pd.concat(all_users, ignore_index=True).to_csv(output / "users.csv", index=False)
    pd.concat(all_sessions, ignore_index=True).to_csv(output / "sessions.csv", index=False)

    import json

    with (output / "community_config.json").open("w", encoding="utf-8") as handle:
        json.dump(all_configs, handle, indent=2, ensure_ascii=False)
