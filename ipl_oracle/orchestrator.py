"""Sequential orchestrator that wires every agent together."""
from __future__ import annotations

from .agents.conditions import ConditionsAgent
from .agents.fixture import FixtureAgent
from .agents.narrator import NarratorAgent, build_trace
from .agents.opponent import OpponentSelector, OpponentStrategist
from .agents.scout import ScoutAgent
from .agents.selector import SelectorAgent
from .agents.simulator import SimulatorAgent
from .agents.strategist import StrategistAgent
from .io import DataLoader, StateStore
from .schemas import OracleRequest, OracleResult


class Orchestrator:
    def __init__(
        self,
        loader: DataLoader | None = None,
        state: StateStore | None = None,
        narrator: NarratorAgent | None = None,
    ):
        self.loader = loader or DataLoader()
        self.state = state
        self.fixture_agent = FixtureAgent(self.loader)
        self.conditions_agent = ConditionsAgent()
        self.scout_agent = ScoutAgent(self.state)
        self.opp_selector = OpponentSelector(self.scout_agent)
        self.opp_strategist = OpponentStrategist()
        self.strategist_agent = StrategistAgent()
        self.selector_agent = SelectorAgent()
        self.simulator_agent = SimulatorAgent(self.state)
        self.narrator = narrator

    def run(self, request: OracleRequest) -> OracleResult:
        team = request.team.upper()
        fixture = self.fixture_agent.resolve(
            team=team,
            opponent=request.opponent,
            venue=request.venue,
            match_date=request.match_date,
        )
        opponent_code = fixture.away_team if fixture.home_team == team else fixture.home_team

        own_squad = self.loader.load_squad(team)
        opp_squad = self.loader.load_squad(opponent_code)
        venue = self.loader.load_venue(fixture.venue)

        conditions = self.conditions_agent.evaluate(venue, request.formation_bias)
        opp_predicted = self.opp_selector.predict(opp_squad, conditions)
        forecast = self.opp_strategist.forecast(own_squad, opp_predicted, conditions)
        own_posteriors = self.scout_agent.evaluate(own_squad.players)
        decision = self.strategist_agent.decide(own_posteriors, forecast)

        own_xi = self.selector_agent.select(
            own_squad=own_squad,
            own_posteriors=own_posteriors,
            conditions=conditions,
            opponent=forecast,
            decision=decision,
            must_include=request.must_include,
            must_exclude=request.must_exclude,
        )

        # Refinement loop: if MC CI is too wide, re-run selector with tighter robust Γ.
        win_prob, toss = self.simulator_agent.simulate(
            own_xi=own_xi,
            opponent=forecast,
            conditions=conditions,
            sample_size=request.sample_size,
        )
        ci_width = win_prob.confidence_interval[1] - win_prob.confidence_interval[0]
        refinement_attempts = 0
        while ci_width > 0.18 and refinement_attempts < 3:
            refinement_attempts += 1
            decision.gamma = min(decision.gamma + 1.0, 4.0)
            decision.mode = decision.mode  # unchanged unless we want to flip
            own_xi = self.selector_agent.select(
                own_squad=own_squad,
                own_posteriors=own_posteriors,
                conditions=conditions,
                opponent=forecast,
                decision=decision,
                must_include=request.must_include,
                must_exclude=request.must_exclude,
            )
            win_prob, toss = self.simulator_agent.simulate(
                own_xi=own_xi,
                opponent=forecast,
                conditions=conditions,
                sample_size=request.sample_size,
            )
            ci_width = win_prob.confidence_interval[1] - win_prob.confidence_interval[0]

        partial = OracleResult(
            own_team=team,
            opponent_team=opponent_code,
            fixture=fixture,
            venue=venue,
            conditions=conditions,
            opponent_forecast=forecast,
            own_xi=own_xi,
            win_probability=win_prob,
            toss=toss,
            narrative="",
            decision_trace={
                "strategy": {
                    "mode": decision.mode.value,
                    "gamma": decision.gamma,
                    "rationale": decision.rationale,
                    "own_uncertainty": decision.own_uncertainty,
                    "opponent_uncertainty": decision.opponent_uncertainty,
                    "refinement_attempts": refinement_attempts,
                },
            },
        )
        narrative = ""
        if self.narrator is not None:
            narrative = self.narrator.narrate(build_trace(partial))
        return partial.model_copy(update={"narrative": narrative})
