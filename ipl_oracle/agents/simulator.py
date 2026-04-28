"""Simulator agent — Monte Carlo + MDP value iteration for win probability."""
from __future__ import annotations

import numpy as np

from ..io import StateStore
from ..optimization.calibration import PlattScaler
from ..optimization.mdp import value_iteration_win_prob
from ..schemas import (
    ConditionsVector,
    OpponentForecast,
    Player,
    PlayerRole,
    TossDecision,
    WinProbability,
    XIOptimization,
)


class SimulatorAgent:
    def __init__(self, state: StateStore | None = None):
        self.state = state
        self.scaler = PlattScaler()
        if state is not None:
            history = state.calibration_history()
            if history:
                pred = [h[0] for h in history]
                act = [h[1] for h in history]
                self.scaler.fit(pred, act)

    @staticmethod
    def _team_runs_per_ball(xi: list[Player]) -> float:
        batters = [
            p for p in xi
            if p.role in (PlayerRole.BATSMAN, PlayerRole.ALL_ROUNDER, PlayerRole.WICKET_KEEPER)
        ]
        if not batters:
            return 1.0
        return float(np.mean([(p.expected_runs_per_ball or 1.1) * p.form_score for p in batters]))

    @staticmethod
    def _team_dismissal_per_ball(xi: list[Player]) -> float:
        bowlers = [p for p in xi if p.role in (PlayerRole.BOWLER, PlayerRole.ALL_ROUNDER)]
        if not bowlers:
            return 0.04
        return float(np.mean([(p.expected_wickets_per_ball or 0.04) * p.form_score for p in bowlers]))

    def simulate(
        self,
        own_xi: XIOptimization,
        opponent: OpponentForecast,
        conditions: ConditionsVector,
        sample_size: int = 10000,
        seed: int | None = 17,
    ) -> tuple[WinProbability, TossDecision]:
        rng = np.random.default_rng(seed)
        own_run = self._team_runs_per_ball(own_xi.selected_xi) * conditions.alpha * 2.0
        opp_run = self._team_runs_per_ball(opponent.predicted_xi.selected_xi) * conditions.alpha * 2.0
        own_diss = self._team_dismissal_per_ball(own_xi.selected_xi) * conditions.beta * 2.0
        opp_diss = self._team_dismissal_per_ball(opponent.predicted_xi.selected_xi) * conditions.beta * 2.0

        own_score = self._innings_score(rng, own_run, opp_diss, sample_size)
        opp_score = self._innings_score(rng, opp_run, own_diss, sample_size)
        wins_bowling_first = (opp_score < own_score).mean()
        # If we bat first, opponent chases our score — model by reusing own_score as target.
        wins_batting_first = self._batting_first_win_prob(
            rng, own_run, opp_run, own_diss, opp_diss, sample_size
        )

        # MDP cross-check using mean target.
        mdp_chase_prob = value_iteration_win_prob(
            target_runs=int(round(float(own_score.mean()))),
            overs=20.0,
            runs_per_ball=opp_run,
            dismissal_prob=own_diss,
        )
        # Blend MC and MDP for batting-first prob (simple average — both are noisy).
        blended_bat_first = float(0.5 * wins_batting_first + 0.5 * (1.0 - mdp_chase_prob))

        toss_dec = self._toss(blended_bat_first, float(wins_bowling_first), conditions)
        chosen = max(blended_bat_first, float(wins_bowling_first))
        ci = self._wilson_ci(chosen, sample_size)
        calibrated = self.scaler.is_fitted
        if calibrated:
            chosen = self.scaler.transform(chosen)

        win_prob = WinProbability(
            win_probability=float(chosen),
            confidence_interval=ci,
            sample_size=sample_size,
            calibrated=calibrated,
            expected_runs=float(own_score.mean()),
            expected_runs_ci=(
                float(np.percentile(own_score, 2.5)),
                float(np.percentile(own_score, 97.5)),
            ),
        )
        return win_prob, toss_dec

    def _innings_score(
        self,
        rng: np.random.Generator,
        runs_per_ball: float,
        dismissal_prob: float,
        n: int,
    ) -> np.ndarray:
        scores = np.zeros(n, dtype=np.int32)
        wickets = np.zeros(n, dtype=np.int32)
        balls = 120
        for _ in range(balls):
            alive = wickets < 10
            run_draw = rng.poisson(lam=runs_per_ball, size=n)
            wkt_draw = rng.binomial(1, min(max(dismissal_prob / 6.0, 0.0), 0.5), size=n)
            scores += np.where(alive, run_draw, 0)
            wickets += np.where(alive, wkt_draw, 0)
        return scores

    def _batting_first_win_prob(
        self,
        rng: np.random.Generator,
        own_rpb: float,
        opp_rpb: float,
        own_diss: float,
        opp_diss: float,
        n: int,
    ) -> float:
        own = self._innings_score(rng, own_rpb, opp_diss, n)
        opp_chase = self._innings_score(rng, opp_rpb, own_diss, n)
        return float((opp_chase < own).mean())

    @staticmethod
    def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
        if n <= 0:
            return (max(0.0, p - 0.05), min(1.0, p + 0.05))
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (float(max(0.0, center - half)), float(min(1.0, center + half)))

    @staticmethod
    def _toss(bat_first: float, bowl_first: float, conditions: ConditionsVector) -> TossDecision:
        if bowl_first > bat_first + 0.02:
            decision = "bowl"
            rationale = (
                f"chase win-prob {bowl_first:.2%} > set win-prob {bat_first:.2%}; "
                f"chasing advantage at venue {conditions.chasing_advantage:.0%}"
            )
        else:
            decision = "bat"
            rationale = (
                f"set win-prob {bat_first:.2%} ≥ chase win-prob {bowl_first:.2%}; "
                f"par score {conditions.par_score:.0f} suggests posting first"
            )
        return TossDecision(
            decision=decision,
            rationale=rationale,
            win_prob_batting_first=float(bat_first),
            win_prob_bowling_first=float(bowl_first),
        )
