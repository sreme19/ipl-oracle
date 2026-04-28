"""Fixture agent — finds the next match for the requested team."""
from __future__ import annotations

from datetime import date

from ..io import DataLoader
from ..schemas import Fixture


class FixtureAgent:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def resolve(
        self,
        team: str,
        opponent: str | None = None,
        venue: str | None = None,
        match_date: date | None = None,
    ) -> Fixture:
        team = team.upper()
        if opponent and venue and match_date:
            return Fixture(
                match_id=f"{team}-{opponent.upper()}-{match_date.isoformat()}",
                match_date=match_date,
                home_team=team,
                away_team=opponent.upper(),
                venue=venue,
                season=match_date.year,
            )
        fx = self.loader.find_next_fixture(team, on_or_after=match_date)
        if opponent:
            opponent = opponent.upper()
            other = fx.away_team if fx.home_team == team else fx.home_team
            if other != opponent:
                # User explicitly asked for a different opponent — synthesize fixture
                return Fixture(
                    match_id=f"{team}-{opponent}-{fx.match_date.isoformat()}",
                    match_date=fx.match_date,
                    home_team=team,
                    away_team=opponent,
                    venue=venue or fx.venue,
                    season=fx.season,
                )
        if venue:
            fx = fx.model_copy(update={"venue": venue})
        return fx
