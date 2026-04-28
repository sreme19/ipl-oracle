"""Opponent agents — predicts opponent XI then forecasts their match-shape.

Two stages:

1. OpponentSelector runs the same MILP as our selector but with weights
   driven by opponent form + venue α/β alone (no matchup advantage — we
   don't yet know who they're playing for).

2. OpponentStrategist projects the predicted XI's expected score and
   bowling-phase profile, and emits a bipartite threat graph against our
   squad. Output feeds the Strategist + Selector for our XI.
"""
from __future__ import annotations

import numpy as np

from ..optimization.milp import MILPConfig, SelectionMILP
from ..schemas import (
    ConditionsVector,
    OpponentForecast,
    OptimizationMode,
    Player,
    PlayerRole,
    Squad,
    ThreatEdge,
    XIOptimization,
)
from .scout import ScoutAgent


class OpponentSelector:
    def __init__(self, scout: ScoutAgent | None = None):
        self.scout = scout or ScoutAgent()
        self.solver = SelectionMILP(MILPConfig())

    def predict(self, opponent: Squad, conditions: ConditionsVector) -> XIOptimization:
        posteriors = self.scout.evaluate(opponent.players)
        weights: dict[str, float] = {}
        for p in opponent.players:
            post = posteriors[p.player_id]
            run_w = ScoutAgent.base_runs_weight(p)
            wkt_w = ScoutAgent.base_wickets_weight(p)
            base = conditions.alpha * run_w + conditions.beta * wkt_w * 150.0
            weights[p.player_id] = base * post.mean

        baseline = self._baseline(opponent.players, conditions)
        result = self.solver.solve(opponent.players, weights)
        improvement = (
            ((result.objective_value - baseline) / baseline * 100.0)
            if baseline > 0 else 0.0
        )
        return XIOptimization(
            selected_xi=result.selected,
            objective_value=result.objective_value,
            baseline_value=baseline,
            improvement_pct=improvement,
            mode=OptimizationMode.DETERMINISTIC,
            solve_time_ms=result.solve_time_ms,
            excluded=result.excluded,
            slot_reasons={p.player_id: "predicted by form + venue fit" for p in result.selected},
        )

    @staticmethod
    def _baseline(players: list[Player], conditions: ConditionsVector) -> float:
        ranked = sorted(
            players,
            key=lambda p: conditions.alpha * ScoutAgent.base_runs_weight(p)
                         + conditions.beta * ScoutAgent.base_wickets_weight(p) * 150.0,
            reverse=True,
        )
        top = ranked[:11]
        return sum(
            conditions.alpha * ScoutAgent.base_runs_weight(p)
            + conditions.beta * ScoutAgent.base_wickets_weight(p) * 150.0
            for p in top
        )


class OpponentStrategist:
    def forecast(
        self,
        own_squad: Squad,
        predicted_xi: XIOptimization,
        conditions: ConditionsVector,
    ) -> OpponentForecast:
        xi = predicted_xi.selected_xi
        batters = [
            p for p in xi
            if p.role in (PlayerRole.BATSMAN, PlayerRole.ALL_ROUNDER, PlayerRole.WICKET_KEEPER)
        ]
        bowlers = [p for p in xi if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER)]

        run_rate = float(np.mean([p.expected_runs_per_ball or 1.1 for p in batters]) or 1.1)
        run_rate *= conditions.alpha * 2.0
        expected_score = run_rate * 120.0
        std = expected_score * 0.12
        score_ci = (max(expected_score - 1.96 * std, 0.0), expected_score + 1.96 * std)

        # Phase strengths — coarse tags from per-bowler economy.
        phases = {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
        for p in bowlers:
            econ = p.bowling_econ or 8.0
            strength = max(0.0, 10.0 - econ) / 10.0
            phases["powerplay"] += 0.3 * strength
            phases["middle"] += 0.5 * strength
            phases["death"] += 0.2 * strength
        # Normalize across phases for comparability.
        total = sum(phases.values()) or 1.0
        phases = {k: v / total for k, v in phases.items()}

        bat_phases = {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
        for p in batters:
            sr = p.batting_sr or 130.0
            tag = "death" if sr >= 150 else "middle" if sr >= 130 else "powerplay"
            bat_phases[tag] += 1.0
        total = sum(bat_phases.values()) or 1.0
        bat_phases = {k: v / total for k, v in bat_phases.items()}

        threats = self._threat_graph(own_squad, xi, conditions)
        return OpponentForecast(
            predicted_xi=predicted_xi,
            expected_score=expected_score,
            expected_score_ci=score_ci,
            bowling_phase_strengths=phases,
            batting_phase_strengths=bat_phases,
            key_threats=threats,
        )

    def _threat_graph(
        self,
        own_squad: Squad,
        opponent_xi: list[Player],
        conditions: ConditionsVector,
    ) -> list[ThreatEdge]:
        """Bipartite threat scoring: opponent bowlers vs our batters."""
        own_batters = [
            p for p in own_squad.players
            if p.role in (PlayerRole.BATSMAN, PlayerRole.ALL_ROUNDER, PlayerRole.WICKET_KEEPER)
        ]
        opp_bowlers = [p for p in opponent_xi if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER)]

        edges: list[ThreatEdge] = []
        for batter in own_batters:
            for bowler in opp_bowlers:
                bat_yield = batter.expected_runs_per_ball or 1.0
                bowl_dismissal = bowler.expected_wickets_per_ball or 0.04
                runs_per_ball = bat_yield * (1.0 + 0.1 * (conditions.alpha - 0.5))
                dismissal = bowl_dismissal * (1.0 + 0.2 * (conditions.beta - 0.5))
                weight = float(0.6 * dismissal * 100.0 + 0.4 * (8.0 - runs_per_ball * 6.0))
                level = "high" if weight > 5.0 else "medium" if weight > 2.5 else "low"
                edge = ThreatEdge(
                    batter_id=batter.player_id,
                    bowler_id=bowler.player_id,
                    weight=weight,
                    threat_level=level,
                    expected_runs_per_ball=float(runs_per_ball),
                    dismissal_rate=float(dismissal),
                )
                edges.append(edge)
        edges.sort(key=lambda e: e.weight, reverse=True)
        return edges[:10]
