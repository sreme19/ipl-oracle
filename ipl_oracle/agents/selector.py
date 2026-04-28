"""Prescriptive selector — picks our XI given opponent forecast and conditions."""
from __future__ import annotations

import numpy as np

from ..optimization.bayesian import Posterior, thompson_weights
from ..optimization.milp import MILPConfig, SelectionMILP
from ..optimization.robust import robust_objective_weights
from ..schemas import (
    ConditionsVector,
    OpponentForecast,
    OptimizationMode,
    Player,
    PlayerRole,
    Squad,
    XIOptimization,
)
from .scout import ScoutAgent
from .strategist import StrategyDecision


class SelectorAgent:
    def __init__(self):
        self.solver = SelectionMILP(MILPConfig())

    def select(
        self,
        own_squad: Squad,
        own_posteriors: dict[str, Posterior],
        conditions: ConditionsVector,
        opponent: OpponentForecast,
        decision: StrategyDecision,
        must_include: list[str] | None = None,
        must_exclude: list[str] | None = None,
    ) -> XIOptimization:
        threat_per_batter: dict[str, float] = {}
        for edge in opponent.key_threats:
            threat_per_batter[edge.batter_id] = (
                threat_per_batter.get(edge.batter_id, 0.0) + edge.weight
            )
        max_threat = max(threat_per_batter.values(), default=1.0) or 1.0

        # Bowler matchup advantage: more "phase-strength gap" = higher value.
        opponent_bat_weak_phase = min(
            opponent.batting_phase_strengths,
            key=opponent.batting_phase_strengths.get,
        )

        nominal: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for p in own_squad.players:
            run_w = ScoutAgent.base_runs_weight(p)
            wkt_w = ScoutAgent.base_wickets_weight(p)
            base = conditions.alpha * run_w + conditions.beta * wkt_w * 150.0

            if p.role in (PlayerRole.BATSMAN, PlayerRole.ALL_ROUNDER, PlayerRole.WICKET_KEEPER):
                threat_penalty = (threat_per_batter.get(p.player_id, 0.0) / max_threat) * 0.15 * base
                base -= threat_penalty
            if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER):
                phase_bonus = self._phase_bonus(p, opponent_bat_weak_phase)
                base += phase_bonus * base * 0.10

            posterior = own_posteriors[p.player_id]
            nominal[p.player_id] = base * posterior.mean
            uncertainty[p.player_id] = base * posterior.std

        if decision.mode == OptimizationMode.ROBUST:
            weights = robust_objective_weights(nominal, uncertainty, gamma=decision.gamma)
        elif decision.mode == OptimizationMode.BAYESIAN:
            posterior_pulls = thompson_weights(own_posteriors)
            weights = {
                pid: nominal[pid] * (posterior_pulls.get(pid, 1.0) / max(own_posteriors[pid].mean, 1e-3))
                for pid in nominal
            }
        else:
            weights = nominal

        baseline = self._baseline(own_squad.players, conditions)
        result = self.solver.solve(
            own_squad.players,
            weights,
            must_include=must_include,
            must_exclude=must_exclude,
        )
        improvement = (
            ((result.objective_value - baseline) / baseline * 100.0)
            if baseline > 0 else 0.0
        )
        slot_reasons = self._explain(result.selected, opponent, conditions, threat_per_batter)
        return XIOptimization(
            selected_xi=result.selected,
            objective_value=result.objective_value,
            baseline_value=baseline,
            improvement_pct=improvement,
            mode=decision.mode,
            solve_time_ms=result.solve_time_ms,
            excluded=result.excluded,
            slot_reasons=slot_reasons,
        )

    @staticmethod
    def _phase_bonus(p: Player, weak_phase: str) -> float:
        sr_or_econ = p.bowling_econ or 8.0
        if weak_phase == "powerplay" and sr_or_econ < 8.0:
            return 1.0
        if weak_phase == "death" and sr_or_econ < 9.0:
            return 1.0
        if weak_phase == "middle" and sr_or_econ < 7.5:
            return 1.0
        return 0.3

    @staticmethod
    def _baseline(players: list[Player], conditions: ConditionsVector) -> float:
        ranked = sorted(
            players,
            key=lambda p: conditions.alpha * ScoutAgent.base_runs_weight(p)
                         + conditions.beta * ScoutAgent.base_wickets_weight(p) * 150.0,
            reverse=True,
        )
        return float(np.sum([
            conditions.alpha * ScoutAgent.base_runs_weight(p)
            + conditions.beta * ScoutAgent.base_wickets_weight(p) * 150.0
            for p in ranked[:11]
        ]))

    @staticmethod
    def _explain(
        selected: list[Player],
        opponent: OpponentForecast,
        conditions: ConditionsVector,
        threat_per_batter: dict[str, float],
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        weak_phase = min(opponent.batting_phase_strengths, key=opponent.batting_phase_strengths.get)
        for p in selected:
            if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER):
                reasons[p.player_id] = (
                    f"bowling pick — attack opponent's weakest phase ({weak_phase})"
                )
            elif p.role == PlayerRole.WICKET_KEEPER:
                reasons[p.player_id] = "wk slot — meets role minimum"
            else:
                threat = threat_per_batter.get(p.player_id, 0.0)
                if threat < 1.0:
                    reasons[p.player_id] = "low matchup threat from opponent attack"
                else:
                    reasons[p.player_id] = (
                        f"top-of-order value despite matchup threat ({threat:.1f})"
                    )
        return reasons
