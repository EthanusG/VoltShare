<h1 align="center">
  <img src="./icon.png" alt="VoltShare" width="128" />
  <br>
  VoltShare
  <br>
</h1>

<h3 align="center">
Smarter charging, fairer power
</h3>

<p align="center">
VoltShare is a simulator for ML-assisted dynamic EV charging power allocation, for residential communities with limited electrical capacity.
</p>

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Generate Synthetic Data

```bash
.venv/bin/python scripts/generate_data.py --scenario all --seed 42
```

This writes:

- `data/generated/users.csv`
- `data/generated/sessions.csv`
- `data/generated/community_config.json`

## Tune VoltShare Weights

```bash
.venv/bin/python scripts/tune_weights.py --data data/generated --seed 42 --trials 120
```

This writes:

- `outputs/results/best_weights.json`
- `outputs/results/tuning_results.csv`

## Run Simulation

```bash
.venv/bin/python scripts/run_simulation.py --data data/generated --weights outputs/results/best_weights.json
```

This writes:

- `outputs/results/session_results.csv`
- `outputs/results/summary_metrics.csv`
- figures under `outputs/figures/`

## Run Multi-Seed Paper Evaluation

```bash
.venv/bin/python scripts/run_multi_seed_evaluation.py --seeds 42 43 44 45 46
```

This writes:

- `outputs/paper/multi_seed_summary_by_seed.csv`
- `outputs/paper/paper_summary_mean_std.csv`
- paper-ready figures under `outputs/paper/figures/`

## Policies

- `equal_share`: divides available power equally among active users.
- `fcfs`: allocates by arrival order until the power cap is reached.
- `proportional_share`: allocates proportionally to remaining demand.
- `voltshare`: applies reserve protection, learned demand scoring, exceptional-demand handling, cooperation credit, and physical constraints.

## Metrics

- `average_charging_completion_rate`: average final battery ratio when charging ends.
- `battery_security_coverage_10` ... `battery_security_coverage_70`: layered battery security coverage, showing the share of cars above each final-battery threshold.
- `basic_reserve_achievement_rate`: share of sessions reaching the basic reserve before departure.
- `average_time_to_basic_reserve_hours`: average time from arrival until the vehicle reaches the basic reserve.
- `exceptional_demand_satisfaction_rate`: share of exceptional-demand users fully served before departure.
- `jain_fairness_index`: Jain index over users' service ratios.
- `potential_acceptance_pressure`: average service loss relative to the equal-share baseline.
- `full_request_completion_rate_diagnostic`: binary full-request completion, retained only as a diagnostic.
- `cooperation_participation_rate`, `cooperation_credit_used`, and `physical_violation_count`: implementation diagnostics.

##  Tests

```bash
.venv/bin/python -m pytest
```
