"""Mixed-integer linear program for XI selection.

The selector solves:

    max  Σ_i  w_i · x_i
    s.t. Σ_i x_i = 11
         Σ_{i∈WK}    x_i ≥ 1
         Σ_{i∈BOWL∪AR} x_i ≥ 4
         Σ_{i∈BAT∪WK∪AR} x_i ≥ 5
         Σ_{i∈overseas} x_i ≤ 4
         x_i ∈ {0, 1}

w_i is a per-player objective weight assembled by the caller from form,
matchup advantage, venue α/β, and (in robust mode) Γ-budgeted worst-case.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pulp

from ..schemas import Player, PlayerRole


@dataclass
class MILPConfig:
    n_players: int = 11
    min_wicketkeepers: int = 1
    min_bowlers: int = 4
    min_batters: int = 5
    max_overseas: int = 4
    time_limit_s: int = 5


@dataclass
class MILPResult:
    selected: list[Player]
    excluded: list[Player]
    objective_value: float
    solve_time_ms: float
    status: str


class SelectionMILP:
    def __init__(self, config: MILPConfig | None = None):
        self.config = config or MILPConfig()

    def solve(
        self,
        squad: list[Player],
        weights: dict[str, float],
        must_include: list[str] | None = None,
        must_exclude: list[str] | None = None,
    ) -> MILPResult:
        cfg = self.config
        prob = pulp.LpProblem("xi_selection", pulp.LpMaximize)
        x = {p.player_id: pulp.LpVariable(f"x_{p.player_id}", cat="Binary") for p in squad}

        prob += pulp.lpSum(weights.get(p.player_id, 0.0) * x[p.player_id] for p in squad)

        prob += pulp.lpSum(x.values()) == cfg.n_players, "team_size"

        wk = [p for p in squad if p.role == PlayerRole.WICKET_KEEPER]
        bowlers = [p for p in squad if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER)]
        batters = [
            p for p in squad
            if p.role in (PlayerRole.BATSMAN, PlayerRole.ALL_ROUNDER, PlayerRole.WICKET_KEEPER)
        ]
        overseas = [p for p in squad if p.is_overseas]

        if wk:
            prob += pulp.lpSum(x[p.player_id] for p in wk) >= cfg.min_wicketkeepers, "min_wk"
        if bowlers:
            prob += pulp.lpSum(x[p.player_id] for p in bowlers) >= cfg.min_bowlers, "min_bowl"
        if batters:
            prob += pulp.lpSum(x[p.player_id] for p in batters) >= cfg.min_batters, "min_bat"
        if overseas:
            prob += pulp.lpSum(x[p.player_id] for p in overseas) <= cfg.max_overseas, "max_ovs"

        for pid in must_include or []:
            if pid in x:
                prob += x[pid] == 1, f"force_in_{pid}"
        for pid in must_exclude or []:
            if pid in x:
                prob += x[pid] == 0, f"force_out_{pid}"

        t0 = time.time()
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg.time_limit_s)
        prob.solve(solver)
        elapsed_ms = (time.time() - t0) * 1000

        selected, excluded = [], []
        for p in squad:
            if pulp.value(x[p.player_id]) and pulp.value(x[p.player_id]) > 0.5:
                selected.append(p)
            else:
                excluded.append(p)

        return MILPResult(
            selected=selected,
            excluded=excluded,
            objective_value=float(pulp.value(prob.objective) or 0.0),
            solve_time_ms=elapsed_ms,
            status=pulp.LpStatus[prob.status],
        )
