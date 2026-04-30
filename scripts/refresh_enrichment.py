"""Refresh analyst enrichment data and team briefing markdown files."""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

TEAM_CODES = ["CSK", "DC", "GT", "KKR", "LSG", "MI", "PBKS", "RCB", "RR", "SRH"]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


TEAM_INSIGHTS = {
    "CSK": {
        "analyst_take": "Analysts frame CSK as a spin-first side that can still control games on slower surfaces, with concern that conservative powerplay starts can create scoreboard pressure later.",
        "strengths": [
            "Chepauk spin squeeze with layered options",
            "Late-innings finishing experience",
            "Tactical matchups against right-heavy batting units",
        ],
        "risks": [
            "Powerplay scoring lag compared with league trend",
            "If surfaces flatten, bowling advantage narrows",
        ],
        "style_tags": ["spin-to-win", "control-middle-overs", "experienced-finishers"],
    },
    "DC": {
        "analyst_take": "Commentary positions DC as high-variance: explosive top-order options and improved pace resources, but strongly dependent on turning early momentum into middle-over control.",
        "strengths": [
            "Aggressive batting intent around anchor options",
            "Improved pace depth in attack",
            "Flexible batting combinations",
        ],
        "risks": [
            "Collapse risk when top-order assault fails",
            "Death-over execution swings results sharply",
        ],
        "style_tags": ["high-variance", "pace-upside", "aggressive-top-order"],
    },
    "GT": {
        "analyst_take": "Experts generally describe GT as playoff-caliber and balanced, powered by reliable top-order runs and a bowling group that tends to absorb disruptions.",
        "strengths": [
            "Consistent top-order run engine",
            "Bowling depth with powerplay and death options",
            "Composure in chases and pressure games",
        ],
        "risks": [
            "Top-order over-dependence if early wickets fall",
            "Middle-order acceleration can stall under spin choke",
        ],
        "style_tags": ["balanced", "top-order-driven", "bowling-resilience"],
    },
    "KKR": {
        "analyst_take": "KKR are seen as a momentum team with retained core quality, though analysts debate whether batting tempo can stay as explosive after personnel churn.",
        "strengths": [
            "Power-hitting ceiling on true surfaces",
            "Multiple seam and hit-the-deck options",
            "Strong familiarity with home conditions",
        ],
        "risks": [
            "Tempo mismatch if anchor-heavy lineups are used",
            "Inconsistency against high-quality legspin",
        ],
        "style_tags": ["momentum-team", "home-surge", "power-batting"],
    },
    "LSG": {
        "analyst_take": "LSG are typically viewed as strategically flexible with broad bowling options and a high-upside top six, but recurrent questions remain around opening stability.",
        "strengths": [
            "Bowling variety across pace and spin",
            "Adaptable composition by venue",
            "Capable finishers for close chases",
        ],
        "risks": [
            "Opening partnerships can be unstable",
            "Can become over-reliant on overseas batting output",
        ],
        "style_tags": ["flexible", "bowling-heavy", "matchup-driven"],
    },
    "MI": {
        "analyst_take": "Panels repeatedly point to MI's peaking pattern: elite pace quality and tournament know-how often make them dangerous at the business end.",
        "strengths": [
            "High-end pace trio for powerplay and death",
            "Explosive middle-order shot-making",
            "Strong record handling knockout pressure",
        ],
        "risks": [
            "Balance shifts when overseas exits or injuries hit",
            "Dependence on pace unit can expose spin gaps",
        ],
        "style_tags": ["peaking-team", "elite-pace", "knockout-experience"],
    },
    "PBKS": {
        "analyst_take": "PBKS commentary now emphasizes their Indian core as a differentiator, with clearer role ownership and a more coherent tactical identity.",
        "strengths": [
            "Indian batting core in strong form",
            "Frontline wicket-taking left-arm pace",
            "Clearer captain-coach blueprint",
        ],
        "risks": [
            "Lower-order stability tested if early wickets fall",
            "Bowling depth can thin when all-rounders are unavailable",
        ],
        "style_tags": ["indian-core", "defined-roles", "upward-trajectory"],
    },
    "RCB": {
        "analyst_take": "RCB are seen as more balanced than prior cycles due to stronger new-ball options, but batting volatility remains a recurring caution.",
        "strengths": [
            "Improved powerplay bowling control",
            "High-impact top-order aggression",
            "Finishing firepower in overs 16-20",
        ],
        "risks": [
            "Middle-order volatility under early pressure",
            "Spin resources can be stretched on turners",
        ],
        "style_tags": ["rebalanced-squad", "powerplay-discipline", "death-hitting"],
    },
    "RR": {
        "analyst_take": "RR are framed as tactically brave: Indian top-order heavy and bowling-refreshed, with upside if key stars remain available and middle-over plans hold.",
        "strengths": [
            "Indian top-order strike-rate potential",
            "Refreshed seam-spin bowling mix",
            "Tactical flexibility through impact substitutions",
        ],
        "risks": [
            "Limited overseas batting backups",
            "Availability and workload management of pace spearheads",
        ],
        "style_tags": ["indian-top-heavy", "bowling-refresh", "fitness-sensitive"],
    },
    "SRH": {
        "analyst_take": "SRH continue to be tagged as one of the most explosive batting units, while transitions between batting and bowling phases are treated as the key swing factor.",
        "strengths": [
            "Explosive top-order scoring bands",
            "Ability to post above-par totals quickly",
            "Strong matchup pressure in powerplay",
        ],
        "risks": [
            "Middle-over control can dip after aggressive starts",
            "Less-tested bench depth in some roles",
        ],
        "style_tags": ["explosive-top", "high-scoring", "pressure-powerplay"],
    },
}

FALLBACK_SOURCES = [
    {
        "publisher": "www.espncricinfo.com",
        "url": "https://www.espncricinfo.com/story/ipl-auction-2025-how-the-ten-teams-stack-up-after-the-mega-auction-1461630",
        "title": "IPL auction 2025 - How the ten teams stack up after the mega auction",
        "note": "Fallback source used when live scraping is unavailable.",
    },
    {
        "publisher": "www.espncricinfo.com",
        "url": "https://www.espncricinfo.com/story/ipl-2025-playoffs-how-punjab-kings-rcb-gujarat-titans-mumbai-indians-stack-up-1487758",
        "title": "IPL 2025 playoffs - team stack up analysis",
        "note": "Fallback source used when live scraping is unavailable.",
    },
    {
        "publisher": "www.skysports.com",
        "url": "https://www.skysports.com/cricket/news/12040/13372752/indian-premier-league-2025-the-final-four-teams-are-locked-in-but-who-will-come-out-strongest",
        "title": "Indian Premier League 2025 final four analysis",
        "note": "Fallback source used when live scraping is unavailable.",
    },
    {
        "publisher": "sportstar.thehindu.com",
        "url": "https://sportstar.thehindu.com/cricket/ipl/ipl-news/rcb-ipl-2025-squad-key-players-and-strategy/article69356690.ece",
        "title": "Royal Challengers Bengaluru IPL 2025 Preview",
        "note": "Fallback source used when live scraping is unavailable.",
    },
    {
        "publisher": "sports.ndtv.com",
        "url": "https://sports.ndtv.com/ipl-2025/ipl-2025-mid-season-review-batters-continue-to-dominate-rise-of-the-underachievers-and-no-home-advantage-8243184",
        "title": "IPL 2025 mid-season review",
        "note": "Fallback source used when live scraping is unavailable.",
    },
]


def _parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parent.parent / "ipl_oracle" / "data"
    parser = argparse.ArgumentParser(description="Refresh analyst enrichment for IPL Oracle.")
    parser.add_argument("--season", type=int, default=2026, help="Season year to generate.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Data directory containing squads/fixtures/venues.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=8,
        help="Maximum number of web sources to keep in metadata.",
    )
    return parser.parse_args()


def _search_duckduckgo(query: str, max_results: int = 5) -> list[str]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException:
        return []
    links = re.findall(r'href="(//duckduckgo.com/l/\?uddg=[^"]+)"', resp.text)
    out: list[str] = []
    for link in links:
        parsed = urllib.parse.urlparse("https:" + link)
        q = urllib.parse.parse_qs(parsed.query)
        uddg = q.get("uddg", [])
        if not uddg:
            continue
        target = urllib.parse.unquote(uddg[0])
        if target not in out:
            out.append(target)
        if len(out) >= max_results:
            break
    return out


def _fetch_title(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException:
        return "Unknown title"
    m = re.search(r"<title>(.*?)</title>", resp.text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return "Unknown title"
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title[:180] if title else "Unknown title"


def _collect_sources(max_sources: int) -> list[dict]:
    queries = [
        "IPL 2025 team preview ESPNcricinfo analysis",
        "IPL 2025 playoffs analysis Sky Sports cricket",
        "IPL 2025 mid season tactical trends",
        "IPL 2025 key players team strengths weaknesses",
    ]
    seen: set[str] = set()
    sources: list[dict] = []
    for query in queries:
        for url in _search_duckduckgo(query, max_results=4):
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                {
                    "publisher": urllib.parse.urlparse(url).netloc,
                    "url": url,
                    "title": _fetch_title(url),
                    "note": f"Auto-collected from query: {query}",
                }
            )
            if len(sources) >= max_sources:
                return sources
    if sources:
        return sources
    return FALLBACK_SOURCES[:max_sources]


def _top_players_by_team(data_dir: Path) -> dict[str, list[dict]]:
    watchlist: dict[str, list[dict]] = {}
    for code in TEAM_CODES:
        squad = json.loads((data_dir / "squads" / f"{code}.json").read_text())
        ranked = sorted(
            squad["players"],
            key=lambda p: (
                float(p.get("form_score", 1.0)),
                float(p.get("batting_sr", 0.0)),
                float(p.get("expected_wickets_per_ball", 0.0)),
            ),
            reverse=True,
        )
        watchlist[code] = [
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "role": p["role"],
                "analyst_focus": "Role-critical in likely XIs; monitor phase impact and current form.",
            }
            for p in ranked[:9]
        ]
    return watchlist


def _fixture_commentary(
    fixtures: list[dict],
    venues: dict[str, dict],
    team_insights: dict[str, dict],
) -> list[dict]:
    pitch_line = {
        "batting": "Batting-friendly conditions raise the ceiling for top-order intent and death-over hitting.",
        "spin": "Spin-friendly conditions reward rotation and matchup management against quality spin.",
        "bowling": "Bowling-friendly conditions amplify powerplay wicket value and disciplined hard lengths.",
        "neutral": "Neutral surfaces shift edge to middle-over execution and death control.",
    }
    out: list[dict] = []
    for fixture in fixtures:
        if fixture["home_team"] == "TBD" or fixture["away_team"] == "TBD" or fixture["venue"] == "TBD":
            continue
        home = fixture["home_team"]
        away = fixture["away_team"]
        venue = venues.get(fixture["venue"], {})
        par = float(venue.get("par_first_innings", 175.0))
        dew = float(venue.get("dew_factor", 0.4))
        chase = float(venue.get("chasing_win_rate", 0.5))
        pitch = str(venue.get("pitch_type", "neutral"))
        out.append(
            {
                "match_id": fixture["match_id"],
                "match_date": fixture["match_date"],
                "home_team": home,
                "away_team": away,
                "venue": fixture["venue"],
                "insight_commentary": (
                    f"{team_insights[home]['analyst_take']} {team_insights[away]['analyst_take']} "
                    f"{pitch_line[pitch]} Expected par first innings is around {par:.0f}; "
                    f"dew factor {dew:.2f} and chasing bias {chase:.2f} keep toss strategy venue-sensitive."
                ),
                "watch_for": [
                    f"{home}: {team_insights[home]['strengths'][0]}",
                    f"{away}: {team_insights[away]['strengths'][0]}",
                    f"Venue dynamic: par {par:.0f}, dew {dew:.2f}, chase bias {chase:.2f}",
                ],
            }
        )
    return out


def _write_briefings(out_dir: Path, season: int, team_insights: dict, watchlist: dict, league_trends: list[str]) -> None:
    briefings_dir = out_dir / f"briefings_{season}"
    briefings_dir.mkdir(parents=True, exist_ok=True)
    for code in TEAM_CODES:
        insight = team_insights[code]
        players = watchlist[code]
        md = [
            f"# {code} Analyst Briefing ({season})",
            "",
            "## Analyst Take",
            insight["analyst_take"],
            "",
            "## Strengths",
            *(f"- {s}" for s in insight["strengths"]),
            "",
            "## Risks",
            *(f"- {r}" for r in insight["risks"]),
            "",
            "## Style Tags",
            ", ".join(insight["style_tags"]),
            "",
            "## Key Players Watchlist",
            *(f"- {p['name']} ({p['role']}) — {p['analyst_focus']}" for p in players),
            "",
            "## League Trends Context",
            *(f"- {t}" for t in league_trends),
            "",
        ]
        (briefings_dir / f"{code}.md").write_text("\n".join(md))


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir
    fixtures = json.loads((data_dir / "fixtures" / "fixtures.json").read_text())["fixtures"]
    venues = {
        v["name"]: v for v in json.loads((data_dir / "venues" / "venues.json").read_text())["venues"]
    }
    sources = _collect_sources(args.max_sources)

    league_trends = [
        "Powerplay scoring remains elevated league-wide; early intent defines par trajectories.",
        "Away teams stay competitive; home advantage is weaker without strong venue specialization.",
        "Chasing remains favorable on high-dew surfaces and batting-friendly grounds.",
        "Middle-over bowling plans mixing pace-off and hard lengths are repeatedly cited as difference-makers.",
    ]
    watchlist = _top_players_by_team(data_dir)
    fixture_notes = _fixture_commentary(fixtures, venues, TEAM_INSIGHTS)

    payload = {
        "season": args.season,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "Auto-refreshed via web source discovery + local fixture/venue mapping. Commentary is synthesized planning guidance.",
        "sources": sources,
        "league_trends": league_trends,
        "team_insights": TEAM_INSIGHTS,
        "key_players_watchlist": watchlist,
        "fixture_commentary": fixture_notes,
    }

    out_dir = data_dir / "enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"analyst_insights_{args.season}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    _write_briefings(out_dir, args.season, TEAM_INSIGHTS, watchlist, league_trends)

    print(f"wrote {out_path}")
    print(f"fixtures_with_commentary {len(fixture_notes)}")
    print(f"sources_collected {len(sources)}")


if __name__ == "__main__":
    main()
