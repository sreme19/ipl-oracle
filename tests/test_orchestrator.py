from datetime import date

from ipl_oracle.io import DataLoader, StateStore
from ipl_oracle.orchestrator import Orchestrator
from ipl_oracle.schemas import OracleRequest


def test_full_pipeline_runs_without_narrator(tmp_path):
    loader = DataLoader()
    state = StateStore(path=tmp_path / "state.db")
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    req = OracleRequest(
        team="RCB",
        opponent="PBKS",
        venue="M Chinnaswamy Stadium",
        match_date=date(2026, 4, 15),
        sample_size=500,
    )
    result = orch.run(req)

    assert result.fixture.home_team == "RCB"
    assert result.fixture.away_team == "PBKS"
    assert len(result.own_xi.selected_xi) == 11
    assert len(result.opponent_forecast.predicted_xi.selected_xi) == 11
    assert 0.0 <= result.win_probability.win_probability <= 1.0
    assert result.toss.decision in ("bat", "bowl")
    assert result.narrative == ""  # narrator disabled


def test_pipeline_resolves_next_fixture_for_team(tmp_path):
    loader = DataLoader()
    state = StateStore(path=tmp_path / "state.db")
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    # Provide full override so fixture lookup never depends on date.today()
    req = OracleRequest(
        team="RCB",
        opponent="MI",
        venue="M Chinnaswamy Stadium",
        match_date=date(2026, 4, 12),
        sample_size=500,
    )
    result = orch.run(req)
    assert result.fixture.home_team == "RCB" or result.fixture.away_team == "RCB"
    assert result.fixture.match_date == date(2026, 4, 12)
