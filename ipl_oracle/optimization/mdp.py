"""Coarse MDP value-iteration for chase win probability.

State: (overs_remaining, wickets_lost, runs_remaining)
Action: implicit (no chooser — strategy baked into transition rates)
Reward: terminal — win iff runs_remaining ≤ 0 before overs run out / 10 wkts.

Per-ball transition (independent draws):
  - dismissal with probability p_w   ⇒ wickets += 1
  - runs ~ Multinomial over {0,1,2,3,4,6} with rates fitted from team form

Bellman: V(s) = E[V(s')]; with deterministic transitions over discrete state
this is exact when we discretise overs & wickets. We discretise overs into
balls (120) and wickets into 0..10; runs run over an explicit grid.

Used by the simulator agent as a deterministic complement to Monte Carlo —
mostly to get analytical par-score curves and toss-decision win probs.
"""
from __future__ import annotations

import numpy as np

# Fixed scoring distribution; coefficients can be calibrated from Cricsheet later.
_RUN_OUTCOMES = np.array([0, 1, 2, 3, 4, 6])
_BASE_DIST = np.array([0.40, 0.36, 0.06, 0.005, 0.10, 0.075])  # sums to 1.0
# Pre-clip to numerical safety
_BASE_DIST = _BASE_DIST / _BASE_DIST.sum()


def _scoring_dist(runs_per_ball: float) -> np.ndarray:
    """Re-weight the base scoring distribution to hit a target mean."""
    base_mean = float(_RUN_OUTCOMES @ _BASE_DIST)
    if base_mean <= 0:
        return _BASE_DIST
    scale = runs_per_ball / base_mean
    weighted = _BASE_DIST * np.where(_RUN_OUTCOMES > 0, scale, 1.0)
    weighted = np.clip(weighted, 1e-6, None)
    return weighted / weighted.sum()


def value_iteration_win_prob(
    target_runs: int,
    overs: float,
    runs_per_ball: float,
    dismissal_prob: float,
    max_wickets: int = 10,
) -> float:
    """Probability of chasing target_runs within `overs` overs and 10 wkts.

    Solved by backward induction over balls. Runs grid is bounded by target+1
    so the table stays tractable (~2000×11 states for typical T20 chases).
    """
    n_balls = int(round(overs * 6))
    if n_balls <= 0 or target_runs <= 0:
        return 0.0
    runs_dist = _scoring_dist(runs_per_ball)
    p_w = float(np.clip(dismissal_prob, 0.0, 0.99))
    # V[w, r] = P(win | wickets w, runs needed r, balls left = current_step).
    # We iterate from balls_left = 0 up to n_balls.
    R = target_runs + 1
    W = max_wickets + 1
    V = np.zeros((W, R), dtype=np.float64)
    # Terminal at balls_left = 0: win iff runs needed ≤ 0 (impossible for r≥1).
    V[:, 0] = 1.0

    for _ in range(n_balls):
        V_new = np.zeros_like(V)
        # Wicket transition: w → w+1 (loss if w+1 == max_wickets — set to 0).
        V_w = np.zeros_like(V)
        V_w[: W - 1, :] = V[1:W, :]  # next-state value if wicket falls
        # Run transitions: r → max(r - run, 0) with the calibrated dist.
        V_r = np.zeros_like(V)
        for run, prob in zip(_RUN_OUTCOMES, runs_dist, strict=True):
            shifted = np.zeros_like(V)
            if run == 0:
                shifted = V
            else:
                shifted[:, run:] = V[:, :-run] if run < R else 0
                # if run >= R we hit r==0 column directly (win):
                shifted[:, 0] = 1.0
            V_r += prob * shifted
        V_new = p_w * V_w + (1.0 - p_w) * V_r
        V_new[:, 0] = 1.0  # absorbing win state
        V_new[max_wickets, :] = 0.0  # absorbing loss when all out
        V = V_new

    return float(V[0, target_runs])
