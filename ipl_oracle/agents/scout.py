"""Scout agent — combines bundled stats with EWM form posterior from state."""
from __future__ import annotations

from ..io import StateStore
from ..optimization.bayesian import Posterior, bayesian_form_posterior
from ..schemas import Player, PlayerRole


class ScoutAgent:
    def __init__(self, state: StateStore | None = None):
        self.state = state

    def evaluate(self, players: list[Player]) -> dict[str, Posterior]:
        """Return per-player posterior over (run-yield × form)."""
        out: dict[str, Posterior] = {}
        for p in players:
            prior_mean = p.form_score
            prior_var = max(p.form_variance, 1e-3)
            obs, obs_var = prior_mean, prior_var
            if self.state is not None:
                stored = self.state.get_form(p.player_id)
                if stored is not None:
                    obs, obs_var = stored
            out[p.player_id] = bayesian_form_posterior(prior_mean, prior_var, obs, obs_var)
        return out

    @staticmethod
    def base_runs_weight(p: Player) -> float:
        """Approx runs contribution per 120 balls weighted by role."""
        if p.role == PlayerRole.BOWLER:
            return 0.05 * p.batting_sr / 100.0 if p.batting_sr else 0.0
        return p.expected_runs_per_ball * 120.0

    @staticmethod
    def base_wickets_weight(p: Player) -> float:
        return p.expected_wickets_per_ball * 24.0  # 4 overs ≈ 24 balls
