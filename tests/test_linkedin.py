from datetime import date

from ipl_oracle.agents.linkedin import LinkedInAgent
from ipl_oracle.io import DataLoader, StateStore
from ipl_oracle.orchestrator import Orchestrator
from ipl_oracle.schemas import OracleRequest


def test_linkedin_post_contains_grounded_numbers(tmp_path):
    loader = DataLoader()
    state = StateStore(path=tmp_path / "state.db")
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    req = OracleRequest(
        team="RCB",
        opponent="GT",
        venue="Narendra Modi Stadium",
        match_date=date(2026, 4, 30),
        sample_size=500,
    )
    result = orch.run(req)
    post = LinkedInAgent().compose(result)

    assert "Let's Optimize the World" in post
    assert "ipl-oracle" in post
    assert "github.com/sreme19/ipl-oracle" in post
    assert "RCB" in post and "GT" in post
    assert "Narendra Modi Stadium" in post
    assert "2026-04-30" in post

    # Win prob is rendered as a percentage with one decimal.
    pct = f"{result.win_probability.win_probability:.1%}"
    assert pct in post

    # Toss decision verb appears.
    assert ("Bat first" in post) or ("Bowl first" in post)

    # Hashtags closing block.
    assert "#IPL2026" in post
    assert "#MILP" in post


def test_linkedin_repo_url_override(tmp_path):
    loader = DataLoader()
    state = StateStore(path=tmp_path / "state.db")
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    req = OracleRequest(
        team="RCB",
        opponent="GT",
        venue="Narendra Modi Stadium",
        match_date=date(2026, 4, 30),
        sample_size=500,
    )
    result = orch.run(req)
    post = LinkedInAgent(repo_url="https://example.com/x").compose(result)
    assert "https://example.com/x" in post
    assert "github.com/sreme19/ipl-oracle" not in post
