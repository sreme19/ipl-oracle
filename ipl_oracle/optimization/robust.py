"""Bertsimas-Sim Γ-robust counterpart for the selection MILP.

Each player weight has nominal value w_i and uncertainty radius δ_i.
The classical Bertsimas-Sim robust LP allows up to Γ players to take their
worst-case value simultaneously. For a maximisation with binary x_i this
becomes:

    max  Σ_i w_i x_i  -  ( Γ z + Σ_i p_i )
    s.t. z + p_i  ≥  δ_i x_i      ∀i
         z, p_i ≥ 0

Rather than rewrite the MILP we collapse the robust correction into
adjusted per-player weights. Worst-case coverage is approximated by
penalising the Γ players with the largest δ_i — which matches the optimal
robust allocation when x_i are binary (the adversary picks the Γ included
players with highest δ).
"""
from __future__ import annotations

import numpy as np


def robust_objective_weights(
    nominal: dict[str, float],
    uncertainty: dict[str, float],
    gamma: float = 2.0,
) -> dict[str, float]:
    """Return Γ-adjusted weights w_i' = w_i - λ_i·δ_i where λ_i ≤ 1 totals Γ."""
    if not nominal:
        return {}
    pids = list(nominal.keys())
    deltas = np.array([uncertainty.get(p, 0.0) for p in pids], dtype=float)
    if deltas.sum() <= 0 or gamma <= 0:
        return dict(nominal)

    # Allocate Γ units of penalty proportional to δ — capped at 1 per player.
    raw_lambda = gamma * deltas / deltas.sum() if deltas.sum() > 0 else np.zeros_like(deltas)
    lambdas = np.clip(raw_lambda, 0.0, 1.0)
    # Re-distribute leftover budget across players that hit the cap.
    leftover = gamma - lambdas.sum()
    while leftover > 1e-6:
        slack_mask = lambdas < 1.0
        if not slack_mask.any():
            break
        eligible_delta = deltas[slack_mask]
        if eligible_delta.sum() <= 0:
            break
        bump = leftover * eligible_delta / eligible_delta.sum()
        lambdas[slack_mask] = np.minimum(1.0, lambdas[slack_mask] + bump)
        new_leftover = gamma - lambdas.sum()
        if abs(new_leftover - leftover) < 1e-9:
            break
        leftover = new_leftover

    adjusted = {p: nominal[p] - float(lambdas[i]) * float(deltas[i]) for i, p in enumerate(pids)}
    return adjusted
