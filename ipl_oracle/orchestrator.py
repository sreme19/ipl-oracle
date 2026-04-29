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
from .optimization.bayesian import Posterior
from .schemas import (
    ConditionsVector,
    OracleRequest,
    OracleResult,
    Player,
    TossDecision,
    WinProbability,
    XIOptimization,
)


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

        # Initial simulation.
        win_prob, toss = self.simulator_agent.simulate(
            own_xi=own_xi,
            opponent=forecast,
            conditions=conditions,
            sample_size=request.sample_size,
        )

        # MC feedback loop: try marginal player swaps, adopt those that improve win prob.
        own_xi, win_prob, toss, fb_rounds, fb_delta = self._mc_feedback_loop(
            own_posteriors=own_posteriors,
            conditions=conditions,
            forecast=forecast,
            current_xi=own_xi,
            current_win_prob=win_prob,
            current_toss=toss,
            must_include=request.must_include,
            sample_size=request.sample_size,
        )

        # CI-width loop: if estimate is still too uncertain, tighten robust Γ and re-select.
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
                "mc_feedback": {
                    "rounds": fb_rounds,
                    "win_prob_delta": round(fb_delta, 4),
                },
            },
        )

        run_id = ""
        if self.state is not None:
            run_id = self.state.log_run(partial)


        narrative = ""
        if self.narrator is not None:
            narrative = self.narrator.narrate(build_trace(partial))
        return partial.model_copy(update={"narrative": narrative, "run_id": run_id})

    # ------------------------------------------------------------------ MC feedback loop

    def _mc_feedback_loop(
        self,
        own_posteriors: dict[str, Posterior],
        conditions: ConditionsVector,
        forecast,
        current_xi: XIOptimization,
        current_win_prob: WinProbability,
        current_toss: TossDecision,
        must_include: list[str],
        sample_size: int,
    ) -> tuple[XIOptimization, WinProbability, TossDecision, int, float]:
        """Swap marginal players to improve win probability.

        Tries replacing the bottom-3 MILP-scored players with same-role bench
        players. Adopts any swap that lifts win prob by > 0.5%. Repeats up to
        5 rounds. Candidates are simulated at 2k samples; the winner is
        re-simulated at full sample_size before being returned.
        """
        _DELTA = 0.005       # minimum meaningful improvement
        _MAX_ROUNDS = 5
        _SWAP_SAMPLES = 2000

        must_include_ids = set(must_include or [])
        best_xi = current_xi
        best_prob = current_win_prob.win_probability
        total_delta = 0.0
        rounds_run = 0
        any_swap = False

        for _ in range(_MAX_ROUNDS):
            # Identify swappable selected players and rank by proxy score ascending.
            swappable = [
                p for p in best_xi.selected_xi
                if p.player_id not in must_include_ids
            ]
            if not swappable:
                break

            marginal = sorted(
                swappable,
                key=lambda p: self._proxy_score(p, conditions, own_posteriors),
            )[:3]

            selected_ids = {p.player_id for p in best_xi.selected_xi}
            bench = [p for p in best_xi.excluded if p.player_id not in selected_ids]

            candidates = [
                (out_p, in_p)
                for out_p in marginal
                for in_p in bench
                if in_p.role == out_p.role
            ]
            if not candidates:
                break

            round_best_xi: XIOptimization | None = None
            round_best_prob = best_prob

            for out_p, in_p in candidates:
                candidate_xi = self._swap_xi(best_xi, out_p, in_p)
                cand_wp, _ = self.simulator_agent.simulate(
                    own_xi=candidate_xi,
                    opponent=forecast,
                    conditions=conditions,
                    sample_size=_SWAP_SAMPLES,
                    seed=None,
                )
                if cand_wp.win_probability > round_best_prob + _DELTA:
                    round_best_prob = cand_wp.win_probability
                    round_best_xi = candidate_xi

            rounds_run += 1

            if round_best_xi is None:
                break  # converged

            total_delta += round_best_prob - best_prob
            best_prob = round_best_prob
            best_xi = round_best_xi
            any_swap = True

        if not any_swap:
            return current_xi, current_win_prob, current_toss, rounds_run, 0.0

        # Re-simulate winner at full fidelity.
        final_wp, final_toss = self.simulator_agent.simulate(
            own_xi=best_xi,
            opponent=forecast,
            conditions=conditions,
            sample_size=sample_size,
            seed=17,
        )
        return best_xi, final_wp, final_toss, rounds_run, total_delta

    @staticmethod
    def _proxy_score(
        p: Player,
        conditions: ConditionsVector,
        posteriors: dict[str, Posterior],
    ) -> float:
        run_w = ScoutAgent.base_runs_weight(p)
        wkt_w = ScoutAgent.base_wickets_weight(p)
        base = conditions.alpha * run_w + conditions.beta * wkt_w * 150.0
        return base * posteriors[p.player_id].mean

    @staticmethod
    def _swap_xi(
        xi: XIOptimization,
        out_player: Player,
        in_player: Player,
    ) -> XIOptimization:
        new_selected = [p for p in xi.selected_xi if p.player_id != out_player.player_id] + [in_player]
        new_excluded = [p for p in xi.excluded if p.player_id != in_player.player_id] + [out_player]
        return xi.model_copy(update={
            "selected_xi": new_selected,
            "excluded": new_excluded,
            "slot_reasons": {},
            "note": None,
        })
