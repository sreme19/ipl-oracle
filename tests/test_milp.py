from ipl_oracle.io import DataLoader
from ipl_oracle.optimization.milp import MILPConfig, SelectionMILP
from ipl_oracle.schemas import Player, PlayerRole


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


def test_milp_caps_wicketkeepers_at_one():
    """Squad with multiple keepers must not return more than one in the XI."""
    loader = DataLoader()
    squad = loader.load_squad("RCB")
    keepers = [p for p in squad.players if p.role == PlayerRole.WICKET_KEEPER]
    assert len(keepers) >= 2, "RCB sample squad expected to have multiple keepers"
    # weight keepers very high to tempt the optimizer to pick both
    weights = {p.player_id: 100.0 if p.role == PlayerRole.WICKET_KEEPER else 1.0
               for p in squad.players}
    res = SelectionMILP().solve(squad.players, weights)
    selected_keepers = [p for p in res.selected if p.role == PlayerRole.WICKET_KEEPER]
    assert len(selected_keepers) == 1


def test_milp_relaxes_when_overseas_cap_makes_eleven_infeasible():
    """Squad with too few domestic players should fall back to ≤ n_players."""
    # 5 domestic + 8 overseas, max_overseas=4 → max feasible XI = 9.
    # Mirrors the curated DC sample squad that triggered the original bug.
    squad = (
        [Player(player_id=f"d{i}", name=f"Dom{i}", role=PlayerRole.BATSMAN, is_overseas=False)
         for i in range(4)]
        + [Player(player_id="d_wk", name="DomWK", role=PlayerRole.WICKET_KEEPER, is_overseas=False)]
        + [Player(player_id=f"o{i}", name=f"Ovs{i}", role=PlayerRole.ALL_ROUNDER, is_overseas=True)
           for i in range(8)]
    )
    weights = {p.player_id: 1.0 for p in squad}
    res = SelectionMILP(MILPConfig(n_players=11, max_overseas=4)).solve(squad, weights)
    assert res.status == "Optimal"
    assert res.relaxed is True
    assert "cannot field" in res.relaxation_reason
    assert len(res.selected) < 11
    assert sum(1 for p in res.selected if p.is_overseas) <= 4
    assert sum(1 for p in res.selected if p.role == PlayerRole.WICKET_KEEPER) <= 1
