"""Strategist meta-agent — picks which optimization mode to use."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..optimization.bayesian import Posterior
from ..schemas import OpponentForecast, OptimizationMode


@dataclass
class StrategyDecision:
    mode: OptimizationMode
    gamma: float
    rationale: str
    own_uncertainty: float
    opponent_uncertainty: float


class StrategistAgent:
    """Inspects own/opponent uncertainty and selects an optimization mode.

    Heuristic:
      - mean opponent threat-edge weight std > 1.5  → robust MILP (Γ=2..3)
      - mean own form posterior std > 0.20          → bayesian (Thompson)
      - else                                        → deterministic
    """

    def decide(
        self,
        own_posteriors: dict[str, Posterior],
        opponent_forecast: OpponentForecast,
    ) -> StrategyDecision:
        own_std = float(np.mean([p.std for p in own_posteriors.values()]) or 0.0)
        opp_std = float(
            np.std([e.weight for e in opponent_forecast.key_threats]) or 0.0
        )
        if opp_std > 1.5:
            gamma = float(min(3.0, max(1.5, opp_std)))
            return StrategyDecision(
                mode=OptimizationMode.ROBUST,
                gamma=gamma,
                rationale=(
                    f"opponent threat-edge stdev {opp_std:.2f} exceeds 1.50 — "
                    f"running Γ={gamma:.1f} robust MILP to absorb worst-case matchups"
                ),
                own_uncertainty=own_std,
                opponent_uncertainty=opp_std,
            )
        if own_std > 0.20:
            return StrategyDecision(
                mode=OptimizationMode.BAYESIAN,
                gamma=0.0,
                rationale=(
                    f"own form posterior stdev {own_std:.3f} > 0.20 — "
                    f"Thompson-sampling weights for exploration"
                ),
                own_uncertainty=own_std,
                opponent_uncertainty=opp_std,
            )
        return StrategyDecision(
            mode=OptimizationMode.DETERMINISTIC,
            gamma=0.0,
            rationale=(
                f"both own ({own_std:.3f}) and opponent ({opp_std:.2f}) "
                f"uncertainty within tolerance — deterministic MILP suffices"
            ),
            own_uncertainty=own_std,
            opponent_uncertainty=opp_std,
        )
