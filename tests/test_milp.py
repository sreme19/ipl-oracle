from ipl_oracle.io import DataLoader
from ipl_oracle.optimization.milp import SelectionMILP


def test_milp_selects_eleven_with_constraints():
    loader = DataLoader()
    squad = loader.load_squad("RCB")
    weights = {p.player_id: 1.0 + 0.1 * idx for idx, p in enumerate(squad.players)}
    res = SelectionMILP().solve(squad.players, weights)
    assert res.status == "Optimal"
    assert len(res.selected) == 11

    # role constraints
    roles = [p.role.value for p in res.selected]
    assert roles.count("wicket_keeper") >= 1
    assert sum(1 for r in roles if r in ("bowler", "all_rounder")) >= 4
    assert sum(1 for p in res.selected if p.is_overseas) <= 4


def test_milp_force_in_and_out():
    loader = DataLoader()
    squad = loader.load_squad("RCB")
    weights = {p.player_id: 1.0 for p in squad.players}
    res = SelectionMILP().solve(
        squad.players,
        weights,
        must_include=["rcb_kohli"],
        must_exclude=["rcb_maxwell"],
    )
    pids = {p.player_id for p in res.selected}
    assert "rcb_kohli" in pids
    assert "rcb_maxwell" not in pids
