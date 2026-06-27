from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voltshare.data_generation import generate_scenario


def test_scenario_stress_parameters_shape_generated_sessions():
    base_config = {
        "simulation": {
            "timestep_minutes": 15,
            "reserve_kwh": 8.0,
            "max_charging_rate_kw": 7.0,
            "battery_capacity_kwh": 60.0,
            "users": 12,
            "days": 8,
        },
        "scenarios": {
            "normal": {
                "users": 12,
                "days": 8,
                "session_probability": 1.0,
                "community_power_cap_kw": 80.0,
                "exception_probability": 0.0,
                "cooperation_probability": 0.0,
            },
            "stress": {
                "users": 12,
                "days": 8,
                "session_probability": 1.0,
                "community_power_cap_kw": 80.0,
                "exception_probability": 0.0,
                "cooperation_probability": 0.0,
                "duration_multiplier": 0.72,
                "energy_multiplier": 1.35,
                "initial_energy_min_kwh": 3.0,
                "initial_energy_max_fraction": 0.32,
            },
        },
    }

    _, normal, _ = generate_scenario(base_config, "normal", seed=7)
    _, stress, _ = generate_scenario(base_config, "stress", seed=7)

    assert stress["duration_steps"].mean() < normal["duration_steps"].mean()
    assert stress["requested_energy_kwh"].mean() > normal["requested_energy_kwh"].mean()
    assert stress["initial_energy_kwh"].min() < 8.0
