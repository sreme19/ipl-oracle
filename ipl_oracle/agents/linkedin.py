"""LinkedIn agent — turns the OracleResult into a crisp shareable post.

Deterministic templating from the decision trace. No LLM call, so this
agent works without an Anthropic key. Numbers, names, and tactical points
are all pulled from the trace produced by the rest of the pipeline.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..schemas import OracleResult, Player, PlayerRole

REPO_URL = "https://github.com/sreme19/ipl-oracle"

PHASE_LABEL = {
    "powerplay": "powerplay (1–6)",
    "middle": "middle overs (7–15)",
    "death": "death overs (16–20)",
}


def _weakest_phase(phase_strengths: dict[str, float]) -> tuple[str, float]:
    return min(phase_strengths.items(), key=lambda kv: kv[1])


def _strongest_phase(phase_strengths: dict[str, float]) -> tuple[str, float]:
    return max(phase_strengths.items(), key=lambda kv: kv[1])


def _top_threat_bowler(threats: list) -> tuple[str | None, int, int]:
    if not threats:
        return None, 0, 0
    top = threats[:8]
    counts = Counter(t.bowler_id for t in top)
    bid, n = counts.most_common(1)[0]
    return bid, n, len(top)


def _coin_flip_pairs(
    selected: Iterable[Player], excluded: Iterable[Player]
) -> list[tuple[Player, Player]]:
    """Pairs of (in, out) batters/all-rounders whose form posteriors overlap.

    Two players are a 'coin flip' when their form means are within ~half a
    pooled standard deviation: the Bayesian model can't reliably tell them
    apart, so the choice between them is essentially noise.
    """
    sel = list(selected)
    exc = list(excluded)
    pairs: list[tuple[Player, Player]] = []
    used: set[str] = set()
    for s in sel:
        if s.role in (PlayerRole.WICKET_KEEPER, PlayerRole.BOWLER):
            continue
        best: tuple[Player, float] | None = None
        for e in exc:
            if e.player_id in used or e.role != s.role:
                continue
            mu_diff = abs(s.form_score - e.form_score)
            sigma = ((s.form_variance + e.form_variance) ** 0.5) or 1e-6
            ratio = mu_diff / sigma
            if ratio < 0.5 and (best is None or ratio < best[1]):
                best = (e, ratio)
        if best is not None:
            pairs.append((s, best[0]))
            used.add(best[0].player_id)
    return pairs[:2]


class LinkedInAgent:
    """Generates a crisp LinkedIn post from an OracleResult."""

    def __init__(self, repo_url: str = REPO_URL):
        self.repo_url = repo_url

    def compose(self, result: OracleResult) -> str:
        own = result.own_team
        opp = result.opponent_team
        fx = result.fixture
        wp = result.win_probability
        toss = result.toss
        of = result.opponent_forecast
        own_xi = result.own_xi

        weak_bowl_phase, _ = _weakest_phase(of.bowling_phase_strengths)
        weak_bat_phase, weak_bat_strength = _weakest_phase(of.batting_phase_strengths)
        strong_bowl_phase, strong_bowl_strength = _strongest_phase(of.bowling_phase_strengths)

        top_bowler_id, top_bowler_edges, total_top_edges = _top_threat_bowler(of.key_threats)
        opp_xi_by_id = {p.player_id: p for p in of.predicted_xi.selected_xi}
        top_bowler_name = (
            opp_xi_by_id[top_bowler_id].name
            if top_bowler_id and top_bowler_id in opp_xi_by_id
            else None
        )

        keeper = next(
            (p for p in own_xi.selected_xi if p.role == PlayerRole.WICKET_KEEPER), None
        )
        all_rounders = [p for p in own_xi.selected_xi if p.role == PlayerRole.ALL_ROUNDER]
        bowlers = [p for p in own_xi.selected_xi if p.role == PlayerRole.BOWLER]
        opener_bowler = (
            max(bowlers, key=lambda p: p.expected_wickets_per_ball) if bowlers else None
        )

        flips = _coin_flip_pairs(own_xi.selected_xi, own_xi.excluded)

        lines: list[str] = []
        lines.append("🚀 Let's Optimize the World — Starting with the IPL")
        lines.append("")
        lines.append(
            "Built **ipl-oracle**: a multi-agent IPL XI optimizer. Give it a team code, "
            "get back a recommended XI, a calibrated win probability, a toss call, and a "
            "written brief. One local CLI command — no frontend, no cloud."
        )
        lines.append("")
        lines.append(
            "Ten typed agents wired with Pydantic contracts. XI selection is an MILP "
            "(PuLP/CBC) with a Bertsimas–Sim Γ-robust counterpart for opponent uncertainty "
            "and a Bayesian Normal-Normal form posterior with Thompson sampling. Win prob "
            "comes from 10,000 Monte Carlo ball-by-ball rollouts, cross-checked against an "
            "MDP grid; Platt scaling calibrates once real outcomes are logged."
        )
        lines.append("")
        lines.append(f"🔗 Code: {self.repo_url}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            f"📊 **Test run: {own} vs {opp} at {fx.venue}, {fx.match_date.isoformat()}**"
        )
        lines.append("")
        lines.append(
            f"Win probability **{wp.win_probability:.1%}** "
            f"(CI {wp.confidence_interval[0]:.1%}–{wp.confidence_interval[1]:.1%}) · "
            f"Expected runs **{wp.expected_runs:.0f}** · "
            f"Mode: {own_xi.mode.value.capitalize()}."
        )
        lines.append("")
        lines.append(f"What the model tells {own}:")
        lines.append("")

        if toss.decision == "bat":
            lines.append(
                f"🏏 **Bat first.** Set win-prob ({toss.win_prob_batting_first:.1%}) > "
                f"chase win-prob ({toss.win_prob_bowling_first:.1%}). Par here is "
                f"{result.conditions.par_score:.0f} — post, don't chase."
            )
        else:
            lines.append(
                f"🏏 **Bowl first.** Chase win-prob ({toss.win_prob_bowling_first:.1%}) > "
                f"set win-prob ({toss.win_prob_batting_first:.1%}). Par here is "
                f"{result.conditions.par_score:.0f}."
            )
        lines.append("")

        if keeper and all_rounders:
            ar_names = ", ".join(p.name for p in all_rounders[:5])
            lines.append(
                f"🧱 **Five-bowler shape, not a sixth batter.** {keeper.name} as the lone "
                f"keeper, then stack bowling-capable slots: {ar_names}. With par at "
                f"{result.conditions.par_score:.0f}, batting depth doesn't pay."
            )
            lines.append("")

        if weak_bat_phase == "powerplay" or weak_bowl_phase == "powerplay":
            attacker = opener_bowler.name if opener_bowler else "your strike bowler"
            lines.append(
                f"⚡ **Open with {attacker}.** {opp}'s weakest batting phase is the "
                f"{PHASE_LABEL[weak_bat_phase]} (strength {weak_bat_strength:.2f}/1.00) — "
                "that's where leverage is concentrated. Strike there."
            )
            lines.append("")

        if top_bowler_name and top_bowler_edges >= 3:
            lines.append(
                f"🎯 **Defuse {top_bowler_name} early.** {top_bowler_edges} of the top "
                f"{total_top_edges} matchup-threat edges run through him. Force him onto "
                "the openers; rotate strike rather than going aerial."
            )
            lines.append("")

        lines.append(
            f"🛡️ **Don't force the {PHASE_LABEL[strong_bowl_phase]}.** {opp}'s strongest "
            f"bowling phase scores {strong_bowl_strength:.2f}. Settle for tempo there, "
            "then accelerate at the death."
        )
        lines.append("")

        if flips:
            pair_str = "; ".join(f"{a.name}/{b.name}" for a, b in flips)
            lines.append(
                f"🎲 **{pair_str} are coin-flips.** The Bayesian model can't separate "
                "them — their form posteriors overlap by more than half a pooled "
                "standard deviation. Pick on fitness."
            )
            lines.append("")

        lines.append(
            "The optimizer doesn't know who's hot in the dressing room. "
            "It knows which matchups to fear."
        )
        lines.append("")
        lines.append("#IPL2026 #Optimization #MILP #BayesianInference")

        return "\n".join(lines)
