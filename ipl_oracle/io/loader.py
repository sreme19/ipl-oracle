"""Reads bundled JSON datasets (squads, players, venues, fixtures)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

from ..schemas import Fixture, Player, PlayerRole, Squad, Venue


class DataNotFoundError(LookupError):
    pass


def _resolve_data_dir(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("IPL_ORACLE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    pkg_data = Path(__file__).resolve().parent.parent / "data"
    if pkg_data.exists():
        return pkg_data
    return Path("data").resolve()


class DataLoader:
    def __init__(self, data_dir: str | os.PathLike | None = None):
        self.data_dir = _resolve_data_dir(data_dir)

    def _read_json(self, *parts: str) -> dict:
        path = self.data_dir.joinpath(*parts)
        if not path.exists():
            raise DataNotFoundError(f"Missing data file: {path}")
        with path.open() as f:
            return json.load(f)

    def load_squad(self, team_code: str) -> Squad:
        raw = self._read_json("squads", f"{team_code.upper()}.json")
        players = [self._parse_player(p) for p in raw["players"]]
        return Squad(team_code=raw["team_code"], name=raw["name"], players=players)

    def load_venue(self, venue_name: str) -> Venue:
        raw = self._read_json("venues", "venues.json")
        for v in raw["venues"]:
            if v["name"].lower() == venue_name.lower() or venue_name.lower() in v["name"].lower():
                return Venue(**v)
        raise DataNotFoundError(f"Unknown venue: {venue_name}")

    def list_venues(self) -> list[Venue]:
        raw = self._read_json("venues", "venues.json")
        return [Venue(**v) for v in raw["venues"]]

    def load_fixtures(self) -> list[Fixture]:
        raw = self._read_json("fixtures", "fixtures.json")
        out = []
        for f in raw["fixtures"]:
            if isinstance(f["match_date"], str):
                f["match_date"] = datetime.fromisoformat(f["match_date"]).date()
            out.append(Fixture(**f))
        return out

    def find_next_fixture(self, team_code: str, on_or_after: date | None = None) -> Fixture:
        team_code = team_code.upper()
        ref = on_or_after or date.today()
        upcoming = [
            f for f in self.load_fixtures()
            if team_code in (f.home_team, f.away_team)
            and f.match_date >= ref
            and f.home_team != "TBD"
            and f.away_team != "TBD"
        ]
        if not upcoming:
            raise DataNotFoundError(
                f"No upcoming fixtures found for {team_code} on or after {ref.isoformat()}. "
                f"The season may be complete or the fixture list may need updating."
            )
        return min(upcoming, key=lambda f: f.match_date)

    def _parse_player(self, raw: dict) -> Player:
        return Player(
            player_id=raw["player_id"],
            name=raw["name"],
            role=PlayerRole(raw["role"]),
            is_overseas=raw.get("is_overseas", False),
            batting_avg=float(raw.get("batting_avg", 0.0)),
            batting_sr=float(raw.get("batting_sr", 0.0)),
            bowling_avg=float(raw.get("bowling_avg", 0.0)),
            bowling_econ=float(raw.get("bowling_econ", 0.0)),
            expected_runs_per_ball=float(raw.get("expected_runs_per_ball", 0.0)),
            expected_wickets_per_ball=float(raw.get("expected_wickets_per_ball", 0.0)),
            form_score=float(raw.get("form_score", 1.0)),
            form_variance=float(raw.get("form_variance", 0.1)),
            matches=int(raw.get("matches", 0)),
        )
