"""Fetch IPL match results from Cricsheet's open JSON archive.

Cricsheet (cricsheet.org) publishes match-by-match JSON within hours of
completion and is freely accessible — unlike the ESPNcricinfo API which
blocks programmatic access.
"""
from __future__ import annotations

import datetime
import io
import json
import zipfile
from dataclasses import dataclass

import requests

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"

# Canonical team name spellings used by Cricsheet → our internal code.
_TEAM_CODE_MAP: dict[str, str] = {
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Gujarat Titans": "GT",
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Delhi Daredevils": "DC",
    "Punjab Kings": "PBKS",
    "Kings XI Punjab": "PBKS",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Lucknow Super Giants": "LSG",
}

# Reverse map: our code → set of Cricsheet name variants.
_CODE_TO_NAMES: dict[str, set[str]] = {}
for _name, _code in _TEAM_CODE_MAP.items():
    _CODE_TO_NAMES.setdefault(_code, set()).add(_name)


@dataclass
class MatchResult:
    toss: str           # "bat" | "bowl" — from the perspective of `team`
    runs_for: int       # runs scored by `team`
    runs_against: int   # runs scored by the opponent
    won: bool
    match_title: str    # human-readable label for confirmation


class CricinfoFetchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_match_result(team: str, opponent: str, match_date: datetime.date) -> MatchResult:
    """Return the result of an IPL match via Cricsheet."""
    team_u = team.upper()
    opp_u = opponent.upper()

    our_names = _CODE_TO_NAMES.get(team_u)
    opp_names = _CODE_TO_NAMES.get(opp_u)
    if not our_names:
        raise CricinfoFetchError(
            f"Unknown team code: {team_u!r}. "
            "Add it to ipl_oracle/io/cricinfo.py:_TEAM_CODE_MAP."
        )
    if not opp_names:
        raise CricinfoFetchError(
            f"Unknown team code: {opp_u!r}. "
            "Add it to ipl_oracle/io/cricinfo.py:_TEAM_CODE_MAP."
        )

    zip_bytes = _download_zip()
    match_data = _find_match(zip_bytes, our_names | opp_names, our_names, match_date)
    return _parse(match_data, our_names)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _download_zip() -> bytes:
    try:
        resp = requests.get(CRICSHEET_URL, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        raise CricinfoFetchError(
            f"Failed to download Cricsheet archive ({CRICSHEET_URL}): {exc}"
        ) from exc


def _find_match(
    zip_bytes: bytes,
    both_names: set[str],
    our_names: set[str],
    match_date: datetime.date,
) -> dict:
    date_str = match_date.isoformat()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as f:
                data = json.load(f)
            info = data.get("info", {})
            if date_str not in info.get("dates", []):
                continue
            teams: list[str] = info.get("teams", [])
            if len({t for t in teams if t in both_names}) == 2:
                return data

    raise CricinfoFetchError(
        f"Match not found in Cricsheet archive for {match_date}. "
        "Cricsheet usually publishes data within hours of match completion; "
        "if the match just finished, try again later."
    )


def _parse(data: dict, our_names: set[str]) -> MatchResult:
    info = data["info"]
    teams: list[str] = info["teams"]

    our_full = next((t for t in teams if t in our_names), None)
    opp_full = next((t for t in teams if t not in our_names), None)
    if our_full is None or opp_full is None:
        raise CricinfoFetchError("Could not identify team names in Cricsheet match data.")

    # Toss
    toss_info = info.get("toss", {})
    toss_winner = toss_info.get("winner", "")
    toss_decision_raw = toss_info.get("decision", "bat")  # "bat" | "field"
    toss_decision = "bat" if toss_decision_raw == "bat" else "bowl"
    if toss_winner == our_full:
        toss = toss_decision
    else:
        toss = "bowl" if toss_decision == "bat" else "bat"

    # Runs per innings
    runs_map: dict[str, int] = {}
    for inn in data.get("innings", []):
        batting_team = inn.get("team", "")
        total = sum(
            delivery["runs"]["total"]
            for over in inn.get("overs", [])
            for delivery in over.get("deliveries", [])
        )
        runs_map[batting_team] = total

    runs_for = runs_map.get(our_full, 0)
    runs_against = runs_map.get(opp_full, 0)

    # Outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", "")
    won = winner == our_full

    our_code = _TEAM_CODE_MAP.get(our_full, our_full)
    opp_code = _TEAM_CODE_MAP.get(opp_full, opp_full)
    title = f"{our_code} vs {opp_code} on {info['dates'][0]}"

    return MatchResult(
        toss=toss,
        runs_for=runs_for,
        runs_against=runs_against,
        won=won,
        match_title=title,
    )
