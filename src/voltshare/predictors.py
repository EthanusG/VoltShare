from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = [
    "user_id",
    "weekday",
    "arrival_step_of_day",
    "recent_energy_mean",
    "recent_duration_mean",
    "recent_completion_mean",
    "recent_exception_rate",
    "recent_cooperation_rate",
]


def add_history_features(sessions: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    rows = []
    for _, group in sessions.sort_values(["user_id", "arrival_step"]).groupby("user_id"):
        history: list[dict] = []
        for row in group.to_dict("records"):
            same_weekday = [h for h in history if h["weekday"] == row["weekday"]]
            basis = same_weekday[-window:] if same_weekday else history[-window:]
            if basis:
                energy = np.mean([h["requested_energy_kwh"] for h in basis])
                duration = np.mean([h["duration_steps"] for h in basis])
                completion = np.mean([1.0 for _ in basis])
                exception = np.mean([float(h["exception_request"]) for h in basis])
                cooperation = np.mean([float(h["cooperation_offer"]) for h in basis])
            else:
                energy = sessions["requested_energy_kwh"].mean()
                duration = sessions["duration_steps"].mean()
                completion = 1.0
                exception = sessions["exception_request"].mean()
                cooperation = sessions["cooperation_offer"].mean()
            current = dict(row)
            current.update(
                {
                    "recent_energy_mean": float(energy),
                    "recent_duration_mean": float(duration),
                    "recent_completion_mean": float(completion),
                    "recent_exception_rate": float(exception),
                    "recent_cooperation_rate": float(cooperation),
                }
            )
            rows.append(current)
            history.append(row)
    return pd.DataFrame(rows)


def add_predictions(sessions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    featured = add_history_features(sessions)
    result_frames = []
    for scenario, group in featured.groupby("scenario", sort=False):
        group = group.sort_values("arrival_step").copy()
        midpoint_day = int(group["day"].quantile(0.35))
        train = group[group["day"] <= midpoint_day]
        if len(train) < 20:
            group["predicted_energy_kwh"] = group["recent_energy_mean"]
            group["predicted_duration_steps"] = group["recent_duration_mean"]
            result_frames.append(group)
            continue
        x_train = train[FEATURE_COLUMNS]
        energy_model = RandomForestRegressor(n_estimators=80, random_state=seed, min_samples_leaf=3)
        duration_model = RandomForestRegressor(n_estimators=80, random_state=seed + 7, min_samples_leaf=3)
        energy_model.fit(x_train, train["requested_energy_kwh"])
        duration_model.fit(x_train, train["duration_steps"])
        group["predicted_energy_kwh"] = energy_model.predict(group[FEATURE_COLUMNS]).clip(1, 60)
        group["predicted_duration_steps"] = duration_model.predict(group[FEATURE_COLUMNS]).clip(4, 96)
        result_frames.append(group)
    return pd.concat(result_frames, ignore_index=True)
