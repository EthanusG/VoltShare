from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CommunityConfig, SessionState


@dataclass
class PolicyContext:
    config: CommunityConfig
    weights: dict[str, float]
    cooperation: dict[str, float]
    exception: dict[str, float]
    user_service_ratio: dict[int, float]
    baseline_service_ratio: dict[int, float]
    exception_counts: dict[int, int]
    cooperation_credit: dict[int, float]


class BasePolicy:
    name = "base"

    def allocate(self, active: list[SessionState], step: int, context: PolicyContext) -> dict[int, float]:
        raise NotImplementedError

    @staticmethod
    def _cap_allocation(active: list[SessionState], context: PolicyContext, proposed: dict[int, float]) -> dict[int, float]:
        capped = {}
        total = 0.0
        for session in active:
            kw = max(0.0, float(proposed.get(session.session_id, 0.0)))
            kw = min(kw, session.max_rate_kw)
            kw = min(kw, session.remaining_kwh / context.config.timestep_hours)
            capped[session.session_id] = kw
            total += kw
        if total > context.config.community_power_cap_kw + 1e-9 and total > 0:
            scale = context.config.community_power_cap_kw / total
            capped = {session_id: kw * scale for session_id, kw in capped.items()}
        return capped


class EqualSharePolicy(BasePolicy):
    name = "equal_share"

    def allocate(self, active: list[SessionState], step: int, context: PolicyContext) -> dict[int, float]:
        if not active:
            return {}
        share = context.config.community_power_cap_kw / len(active)
        return self._cap_allocation(active, context, {s.session_id: share for s in active})


class FCFSPolicy(BasePolicy):
    name = "fcfs"

    def allocate(self, active: list[SessionState], step: int, context: PolicyContext) -> dict[int, float]:
        remaining = context.config.community_power_cap_kw
        allocation = {}
        for session in sorted(active, key=lambda item: (item.arrival_step, item.session_id)):
            kw = min(session.max_rate_kw, session.remaining_kwh / context.config.timestep_hours, remaining)
            allocation[session.session_id] = max(0.0, kw)
            remaining -= max(0.0, kw)
            if remaining <= 1e-9:
                break
        return self._cap_allocation(active, context, allocation)


class ProportionalSharePolicy(BasePolicy):
    name = "proportional_share"

    def allocate(self, active: list[SessionState], step: int, context: PolicyContext) -> dict[int, float]:
        if not active:
            return {}
        weights = np.array([max(s.remaining_kwh, 0.1) for s in active], dtype=float)
        weights = weights / weights.sum()
        proposed = {
            session.session_id: context.config.community_power_cap_kw * float(weight)
            for session, weight in zip(active, weights)
        }
        return self._cap_allocation(active, context, proposed)


class VoltSharePolicy(BasePolicy):
    name = "voltshare"

    def allocate(self, active: list[SessionState], step: int, context: PolicyContext) -> dict[int, float]:
        if not active:
            return {}
        allocation = {session.session_id: 0.0 for session in active}
        remaining_cap = context.config.community_power_cap_kw

        below_reserve = [
            session
            for session in active
            if session.current_total_energy_kwh < context.config.reserve_kwh
        ]
        for session in sorted(below_reserve, key=lambda item: item.current_total_energy_kwh):
            if remaining_cap <= 1e-9:
                break
            kw = min(session.max_rate_kw, session.remaining_kwh / context.config.timestep_hours, remaining_cap)
            allocation[session.session_id] = kw
            remaining_cap -= kw

        if remaining_cap > 1e-9:
            remaining_cap -= self._allocate_exception_layer(active, allocation, step, context, remaining_cap)

        residual = [
            session
            for session in active
            if self._available_rate(session, context) - allocation[session.session_id] > 1e-9
        ]
        if residual and remaining_cap > 1e-9:
            scores = np.array([self._score(session, step, context) for session in residual], dtype=float)
            scores = np.maximum(scores, 0.01)
            scores = scores ** max(1.0, float(context.exception.get("score_sharpness", 1.0)))
            scores = scores / scores.sum()
            for session, share in zip(residual, scores):
                allocation[session.session_id] += remaining_cap * float(share)

        return self._cap_allocation(active, context, allocation)

    def _allocate_exception_layer(
        self,
        active: list[SessionState],
        allocation: dict[int, float],
        step: int,
        context: PolicyContext,
        remaining_cap: float,
    ) -> float:
        emergency_fraction = float(context.exception.get("emergency_cap_fraction", 0.0))
        emergency_budget = min(remaining_cap, context.config.community_power_cap_kw * emergency_fraction)
        if emergency_budget <= 1e-9:
            return 0.0

        candidates = [
            session
            for session in active
            if session.exception_request
            and self._exception_trust(session, context) > 0
            and self._available_rate(session, context) - allocation[session.session_id] > 1e-9
        ]
        if not candidates:
            return 0.0

        scores = np.array(
            [
                max(0.05, self._urgency(session, step, context) * self._exception_trust(session, context))
                for session in candidates
            ],
            dtype=float,
        )
        scores = scores / scores.sum()
        spent = 0.0
        for session, share in zip(candidates, scores):
            available = self._available_rate(session, context) - allocation[session.session_id]
            kw = min(available, emergency_budget * float(share))
            allocation[session.session_id] += kw
            spent += kw
        return spent

    def _score(self, session: SessionState, step: int, context: PolicyContext) -> float:
        cfg = context.config
        weights = context.weights
        remaining_need = min(1.0, session.remaining_kwh / max(session.predicted_energy_kwh, 1.0))
        urgency = self._urgency(session, step, context)

        service = context.user_service_ratio.get(session.user_id, 1.0)
        average_service = np.mean(list(context.user_service_ratio.values())) if context.user_service_ratio else 1.0
        fairness = max(0.0, (average_service - service) / max(average_service, 0.05))

        baseline = context.baseline_service_ratio.get(
            (context.config.scenario, session.user_id),
            context.baseline_service_ratio.get(session.user_id, service),
        )
        acceptance = max(0.0, baseline - service)

        exception_penalty = self._exception_penalty(session, context)
        if session.exception_request:
            urgency = min(1.0, urgency + 0.18)

        credit = min(
            float(context.cooperation.get("max_score_bonus", 0.18)),
            context.cooperation_credit.get(session.user_id, 0.0)
            * float(context.cooperation.get("max_score_bonus", 0.18)),
        )

        return (
            weights.get("remaining_demand", 0.0) * remaining_need
            + weights.get("urgency", 0.0) * urgency
            + weights.get("fairness", 0.0) * fairness
            + weights.get("acceptance", 0.0) * acceptance
            + weights.get("cooperation_credit", 0.0) * credit
            - weights.get("exception_penalty", 0.0) * exception_penalty
        )

    def _available_rate(self, session: SessionState, context: PolicyContext) -> float:
        return min(session.max_rate_kw, session.remaining_kwh / context.config.timestep_hours)

    def _urgency(self, session: SessionState, step: int, context: PolicyContext) -> float:
        steps_left = max(1.0, session.departure_step - step)
        required_rate = session.remaining_kwh / (steps_left * context.config.timestep_hours)
        return min(1.0, required_rate / max(session.max_rate_kw, 0.1))

    def _exception_penalty(self, session: SessionState, context: PolicyContext) -> float:
        exception_count = context.exception_counts.get(session.user_id, 0)
        threshold = int(context.exception.get("frequent_request_threshold", 3))
        extra = max(0, exception_count - threshold)
        return min(
            float(context.exception.get("max_penalty", 0.24)),
            extra * float(context.exception.get("penalty_per_extra_request", 0.06)),
        )

    def _exception_trust(self, session: SessionState, context: PolicyContext) -> float:
        max_penalty = float(context.exception.get("max_penalty", 0.24))
        if max_penalty <= 0:
            return 1.0
        return max(0.0, 1.0 - self._exception_penalty(session, context) / max_penalty)


POLICIES = {
    "equal_share": EqualSharePolicy,
    "fcfs": FCFSPolicy,
    "proportional_share": ProportionalSharePolicy,
    "voltshare": VoltSharePolicy,
}
