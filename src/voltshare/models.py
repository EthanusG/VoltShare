from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommunityConfig:
    scenario: str
    timestep_minutes: int
    reserve_kwh: float
    max_charging_rate_kw: float
    battery_capacity_kwh: float
    community_power_cap_kw: float

    @property
    def timestep_hours(self) -> float:
        return self.timestep_minutes / 60.0


@dataclass
class SessionState:
    session_id: int
    user_id: int
    scenario: str
    day: int
    weekday: int
    arrival_step: int
    departure_step: int
    requested_energy_kwh: float
    initial_energy_kwh: float
    max_rate_kw: float
    exception_request: bool
    cooperation_offer: bool
    offered_energy_kwh: float
    predicted_energy_kwh: float
    predicted_duration_steps: float
    delivered_kwh: float = 0.0
    reserve_achievement_step: int | None = None
    cooperation_credit_used: float = 0.0

    @property
    def target_energy_kwh(self) -> float:
        return min(self.requested_energy_kwh, max(0.0, self.requested_energy_kwh))

    @property
    def current_total_energy_kwh(self) -> float:
        return self.initial_energy_kwh + self.delivered_kwh

    @property
    def remaining_kwh(self) -> float:
        return max(0.0, self.requested_energy_kwh - self.delivered_kwh)

    def is_active(self, step: int) -> bool:
        return self.arrival_step <= step < self.departure_step and self.remaining_kwh > 1e-9
