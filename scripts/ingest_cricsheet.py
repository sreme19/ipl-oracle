"""Refresh ipl-oracle data from Cricsheet's IPL JSON archive.

Usage:
    python scripts/ingest_cricsheet.py --output ipl_oracle/data --seasons 2023 2024

The script downloads `ipl_json.zip` from cricsheet.org, parses every match
file, then writes:

    {output}/squads/<TEAM>.json     latest XI per franchise + role tags
    {output}/players/<PID>.json     aggregate per-ball stats
    {output}/venues/venues.json     venue names + observed scoring
    {output}/h2h/h2h_matrix.json    pairwise expected-runs / dismissal rate

Future fixtures are NOT in Cricsheet — those are sourced separately
(see fixtures.json which is hand-curated from the IPL website).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"

TEAM_CODE_MAP = {
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Punjab Kings": "PBKS",
    "Kings XI Punjab": "PBKS",
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Delhi Daredevils": "DC",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
}


def _download(url: str) -> bytes:
    print(f"[ingest] fetching {url}", file=sys.stderr)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content


def _iter_matches(buf: bytes, seasons: set[int] | None):
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as f:
                data = json.load(f)
            info = data.get("info", {})
            season = info.get("season")
            if isinstance(season, str) and "/" in season:
                season = int(season.split("/")[0])
            try:
                season_int = int(season)
            except (TypeError, ValueError):
                continue
            if seasons and season_int not in seasons:
                continue
            yield data


def _classify_role(player_stats: dict) -> str:
    balls_faced = player_stats["balls_faced"]
    balls_bowled = player_stats["balls_bowled"]
    is_wk = player_stats["wk_actions"] >= 5
    if is_wk:
        return "wicket_keeper"
    if balls_bowled > balls_faced * 0.4 and balls_bowled > 60:
        if balls_faced > 60:
            return "all_rounder"
        return "bowler"
    if balls_faced > balls_bowled * 1.5 or balls_bowled <= 60:
        return "batsman"
    return "all_rounder"


def _process(matches, output: Path) -> None:
    player_agg: dict[str, dict] = defaultdict(lambda: {
        "name": None,
        "matches": 0,
        "runs": 0,
        "balls_faced": 0,
        "wickets_taken": 0,
        "balls_bowled": 0,
        "runs_conceded": 0,
        "wk_actions": 0,
        "team_seen": defaultdict(int),
    })
    venue_agg: dict[str, dict] = defaultdict(lambda: {
        "city": "",
        "first_innings_scores": [],
        "second_innings_scores": [],
    })
    last_xi_per_team: dict[str, tuple[str, list[str]]] = {}
    h2h: dict[str, dict] = defaultdict(lambda: {
        "balls": 0, "runs": 0, "outs": 0,
    })

    for match in matches:
        info = match.get("info", {})
        venue = info.get("venue")
        city = info.get("city", "")
        if venue:
            venue_agg[venue]["city"] = city
        match_date = (info.get("dates") or [""])[0]
        players_per_team = info.get("players", {}) or {}
        registry = (info.get("registry") or {}).get("people", {}) or {}

        for team_name, names in players_per_team.items():
            for n in names:
                pid = registry.get(n) or n.replace(" ", "_")
                player_agg[pid]["name"] = n
                player_agg[pid]["matches"] += 1
                player_agg[pid]["team_seen"][TEAM_CODE_MAP.get(team_name, team_name)] += 1

        for team_name, names in players_per_team.items():
            code = TEAM_CODE_MAP.get(team_name)
            if not code:
                continue
            ids = [registry.get(n) or n.replace(" ", "_") for n in names]
            prev = last_xi_per_team.get(code)
            if not prev or match_date >= prev[0]:
                last_xi_per_team[code] = (match_date, ids)

        innings = match.get("innings") or []
        for idx, inn in enumerate(innings):
            total = 0
            for over in inn.get("overs") or []:
                for ball in over.get("deliveries") or []:
                    batter = ball.get("batter")
                    bowler = ball.get("bowler")
                    runs = (ball.get("runs") or {}).get("batter", 0) or 0
                    extras = (ball.get("runs") or {}).get("extras", 0) or 0
                    total += runs + extras
                    bid = registry.get(batter) or (batter or "").replace(" ", "_")
                    bw_id = registry.get(bowler) or (bowler or "").replace(" ", "_")
                    if bid:
                        player_agg[bid]["balls_faced"] += 1
                        player_agg[bid]["runs"] += runs
                        player_agg[bid]["name"] = batter
                    if bw_id:
                        player_agg[bw_id]["balls_bowled"] += 1
                        player_agg[bw_id]["runs_conceded"] += runs + extras
                        player_agg[bw_id]["name"] = bowler
                    if ball.get("wickets"):
                        if bw_id:
                            player_agg[bw_id]["wickets_taken"] += len(ball["wickets"])
                        for w in ball["wickets"]:
                            kind = (w.get("kind") or "").lower()
                            fielders = w.get("fielders") or []
                            for fielder in fielders:
                                fname = fielder.get("name") if isinstance(fielder, dict) else fielder
                                fid = registry.get(fname) or (fname or "").replace(" ", "_")
                                if "stumped" in kind or "caught" in kind and "wicket" in kind:
                                    player_agg[fid]["wk_actions"] += 1
                        h2h_key = f"{bid}__vs__{bw_id}"
                        h2h[h2h_key]["balls"] += 1
                        h2h[h2h_key]["runs"] += runs
                        h2h[h2h_key]["outs"] += len(ball.get("wickets") or [])
                    elif bid and bw_id:
                        h2h_key = f"{bid}__vs__{bw_id}"
                        h2h[h2h_key]["balls"] += 1
                        h2h[h2h_key]["runs"] += runs
            if venue:
                if idx == 0:
                    venue_agg[venue]["first_innings_scores"].append(total)
                elif idx == 1:
                    venue_agg[venue]["second_innings_scores"].append(total)

    output.mkdir(parents=True, exist_ok=True)
    (output / "squads").mkdir(exist_ok=True)
    (output / "players").mkdir(exist_ok=True)
    (output / "venues").mkdir(exist_ok=True)
    (output / "h2h").mkdir(exist_ok=True)

    # Write players
    for pid, agg in player_agg.items():
        balls_faced = max(agg["balls_faced"], 1)
        balls_bowled = max(agg["balls_bowled"], 1)
        sr = (agg["runs"] / balls_faced) * 100 if agg["balls_faced"] else 0.0
        econ = (agg["runs_conceded"] / balls_bowled) * 6 if agg["balls_bowled"] else 0.0
        bowl_avg = (agg["runs_conceded"] / agg["wickets_taken"]) if agg["wickets_taken"] else 0.0
        primary_team = max(agg["team_seen"], key=agg["team_seen"].get) if agg["team_seen"] else ""
        out = {
            "player_id": pid,
            "name": agg["name"],
            "team": primary_team,
            "role": _classify_role(agg),
            "matches": agg["matches"],
            "batting_sr": round(sr, 2),
            "batting_avg": round(agg["runs"] / max(1, agg["matches"]), 2),
            "bowling_econ": round(econ, 2),
            "bowling_avg": round(bowl_avg, 2),
            "expected_runs_per_ball": round(agg["runs"] / balls_faced, 4) if agg["balls_faced"] else 0.0,
            "expected_wickets_per_ball": round(agg["wickets_taken"] / balls_bowled, 4) if agg["balls_bowled"] else 0.0,
        }
        (output / "players" / f"{pid}.json").write_text(json.dumps(out, indent=2))

    # Write squads (latest XI per team — minimal seed; user can expand to full roster)
    for code, (_, ids) in last_xi_per_team.items():
        roster = []
        for pid in ids:
            pf = output / "players" / f"{pid}.json"
            if pf.exists():
                roster.append(json.loads(pf.read_text()))
        squad = {
            "team_code": code,
            "name": code,
            "players": roster,
        }
        (output / "squads" / f"{code}.json").write_text(json.dumps(squad, indent=2))

    # Write venues
    venues_list = []
    for name, agg in venue_agg.items():
        scores = agg["first_innings_scores"]
        chases = agg["second_innings_scores"]
        if scores:
            par = sum(scores) / len(scores)
        else:
            par = 170.0
        chasing_wr = sum(1 for s, c in zip(scores, chases, strict=False) if c >= s) / max(1, min(len(scores), len(chases)))
        venues_list.append({
            "name": name,
            "city": agg["city"],
            "boundary_straight_m": 70.0,
            "boundary_square_m": 65.0,
            "par_first_innings": round(par, 1),
            "chasing_win_rate": round(chasing_wr, 3),
            "dew_factor": 0.5 if "Chennai" in (agg["city"] or "") or "Hyderabad" in (agg["city"] or "") else 0.3,
            "pitch_type": "neutral",
        })
    (output / "venues" / "venues.json").write_text(json.dumps({"venues": venues_list}, indent=2))

    (output / "h2h" / "h2h_matrix.json").write_text(json.dumps(h2h, indent=2))

    print(
        f"[ingest] wrote {len(player_agg)} players, "
        f"{len(last_xi_per_team)} squads, {len(venue_agg)} venues",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ipl_oracle/data"))
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    parser.add_argument("--archive", type=Path, default=None,
                        help="Use a local copy of ipl_json.zip instead of downloading")
    args = parser.parse_args()
    seasons = set(args.seasons) if args.seasons else None
    if args.archive:
        buf = args.archive.read_bytes()
    else:
        buf = _download(CRICSHEET_URL)
    matches = list(_iter_matches(buf, seasons))
    print(f"[ingest] parsed {len(matches)} matches", file=sys.stderr)
    _process(matches, args.output)


if __name__ == "__main__":
    main()
