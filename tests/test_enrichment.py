from ipl_oracle.io import DataLoader


def test_loader_reads_team_insight():
    loader = DataLoader()
    insight = loader.load_team_insight("RCB")
    assert "analyst_take" in insight
    assert isinstance(insight.get("strengths"), list)


def test_loader_reads_fixture_commentary():
    loader = DataLoader()
    commentary = loader.find_fixture_commentary("2026-IPL-42")
    assert commentary["home_team"] == "GT"
    assert commentary["away_team"] == "RCB"
    assert commentary["venue"] == "Narendra Modi Stadium"


def test_loader_reads_briefing_markdown():
    loader = DataLoader()
    briefings = loader.list_team_briefings()
    assert "RCB.md" in briefings
    md = loader.load_team_briefing_markdown("RCB")
    assert "# RCB Analyst Briefing (2026)" in md
