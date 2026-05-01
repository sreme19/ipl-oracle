# Schema Reference

Every Pydantic data model in `ipl-oracle`, the JSON data file formats on disk, and how to extend the dataset. All models live in `schemas.py`.

---

## Enumerations

### `PlayerRole`

```
"batsman" | "bowler" | "all_rounder" | "wicket_keeper"
```

Used in MILP constraints: wicket-keeper constraint checks `role == "wicket_keeper"`, bowling constraint checks `role in {"bowler", "all_rounder"}`, batting constraint checks `role in {"batsman", "all_rounder", "wicket_keeper"}`.

### `OptimizationMode`

```
"deterministic" | "robust" | "bayesian"
```

Set by `StrategistAgent` and recorded in `XIOptimization.mode` and `StateStore.runs`.

---

## Input Models

### `OracleRequest`

Entry contract for the `run` command.

| Field | Type | Default | Description |
|---|---|---|---|
| `team` | `str` | required | Team code (e.g. `"RCB"`) |
| `opponent` | `str \| None` | `None` | Override opponent; resolved from fixture if absent |
| `venue` | `str \| None` | `None` | Override venue; resolved from fixture if absent |
| `match_date` | `date \| None` | `None` | Override date; resolved from fixture if absent |
| `formation_bias` | `"batting" \| "balanced" \| "bowling"` | `"balanced"` | Shifts α/β weights ±0.05–0.10 |
| `must_include` | `list[str]` | `[]` | Player IDs locked into XI |
| `must_exclude` | `list[str]` | `[]` | Player IDs locked out of XI |
| `sample_size` | `int` | `10000` | Monte Carlo rollouts |

---

### `Player`

One player in a squad, with form and performance stats.

| Field | Type | Default | Description |
|---|---|---|---|
| `player_id` | `str` | required | Unique ID (e.g. `"rcb_kohli"`) |
| `name` | `str` | required | Display name |
| `role` | `PlayerRole` | required | Batting/bowling classification |
| `is_overseas` | `bool` | `False` | Counts toward the 4-overseas cap |
| `batting_avg` | `float` | `0.0` | Career batting average |
| `batting_sr` | `float` | `0.0` | Career strike rate |
| `bowling_avg` | `float` | `0.0` | Career bowling average |
| `bowling_econ` | `float` | `0.0` | Career economy rate |
| `expected_runs_per_ball` | `float` | `0.0` | Expected scoring rate [0, ∞) |
| `expected_wickets_per_ball` | `float` | `0.0` | Expected wicket rate [0, 1) |
| `form_score` | `float` | `1.0` | Multiplier on expected metrics (1.0 = average) |
| `form_variance` | `float` | `0.1` | Uncertainty in form estimate |
| `matches` | `int` | `0` | Career matches played |

`form_score` and `form_variance` are seeded from the squad JSON. They are updated by `StateStore.update_form_ewm()` after each retro run.

---

### `Squad`

A team's full playing pool.

| Field | Type | Description |
|---|---|---|
| `team_code` | `str` | e.g. `"RCB"` |
| `name` | `str` | Full team name |
| `players` | `list[Player]` | All squad members (typically 20–25) |

---

### `Venue`

Stadium metadata used by `ConditionsAgent`.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Stadium name |
| `city` | `str` | City |
| `boundary_straight_m` | `float` | Straight boundary in metres |
| `boundary_square_m` | `float` | Square boundary in metres |
| `par_first_innings` | `float` | Expected first-innings par score |
| `chasing_win_rate` | `float` [0,1] | Historical win rate when chasing |
| `dew_factor` | `float` [0,1] | Dew impact on evening matches |
| `pitch_type` | `str` | `"batting"` \| `"bowling"` \| `"spin"` \| `"neutral"` |

---

### `Fixture`

A scheduled match.

| Field | Type | Description |
|---|---|---|
| `match_id` | `str` | Unique identifier (e.g. `"2026-IPL-01"`) |
| `match_date` | `date` | ISO date |
| `home_team` | `str` | Home team code |
| `away_team` | `str` | Away team code |
| `venue` | `str` | Venue name (matched against `venues.json`) |
| `season` | `int` | IPL season year |

---

## Intermediate Models

### `ConditionsVector`

Output of `ConditionsAgent`. Drives MILP weight assembly.

| Field | Type | Description |
|---|---|---|
| `venue` | `str` | Venue name |
| `alpha` | `float` | Batting weight (with formation bias applied) |
| `beta` | `float` | Bowling weight; α + β = 1 |
| `size_factor` | `float` | Boundary ratio (0.85–1.20) |
| `par_score` | `float` | First-innings par runs |
| `dew_factor` | `float` [0,1] | Dew impact |
| `chasing_advantage` | `float` | Historical chase win rate |
| `pitch_type` | `str` | Pitch classification |
| `notes` | `list[str]` | Coaching notes |

---

### `ThreatEdge`

A batter-bowler matchup risk from the bipartite threat graph.

| Field | Type | Description |
|---|---|---|
| `batter_id` | `str` | Our player ID |
| `bowler_id` | `str` | Opponent player ID |
| `weight` | `float` | Composite threat magnitude |
| `threat_level` | `"low" \| "medium" \| "high"` | Classification |
| `expected_runs_per_ball` | `float` | Expected scoring rate in this matchup |
| `dismissal_rate` | `float` [0,1] | P(dismissal per ball) by this bowler vs this batter |

---

### `Posterior` (dataclass)

Output of `ScoutAgent` per player — Bayesian form posterior.

| Field | Type | Description |
|---|---|---|
| `mean` | `float` | μ_post — posterior mean form score |
| `variance` | `float` | σ_post² |
| `std` | `float` | √σ_post² |
| `sample()` | method | Draw from N(mean, variance) for Thompson sampling |

---

### `StrategyDecision`

Output of `StrategistAgent`.

| Field | Type | Description |
|---|---|---|
| `mode` | `OptimizationMode` | Selected MILP mode |
| `gamma` | `float` | Γ for robust mode; 0 otherwise |
| `rationale` | `str` | Human-readable explanation |

---

### `XIOptimization`

Output of both `OpponentSelector` (predicted opponent XI) and `SelectorAgent` (our XI).

| Field | Type | Description |
|---|---|---|
| `selected_xi` | `list[Player]` | 11 selected players |
| `objective_value` | `float` | MILP objective at optimum |
| `baseline_value` | `float` | Top-11-by-weight greedy baseline |
| `improvement_pct` | `float` | `(objective − baseline) / baseline × 100` |
| `mode` | `OptimizationMode` | Mode used |
| `solve_time_ms` | `float` | Solver wall time |
| `excluded` | `list[Player]` | Bench players |
| `slot_reasons` | `dict[str, str]` | `{player_id: reason_string}` |
| `note` | `Optional[str]` | Constraint relaxation note |

---

### `OpponentForecast`

Output of `OpponentStrategist`.

| Field | Type | Description |
|---|---|---|
| `predicted_xi` | `XIOptimization` | Opponent predicted XI |
| `expected_score` | `float` | Predicted opponent total |
| `expected_score_ci` | `tuple[float, float]` | 95% CI |
| `bowling_phase_strengths` | `dict[str, float]` | `{"powerplay": f, "middle": f, "death": f}` |
| `batting_phase_strengths` | `dict[str, float]` | Same for batting |
| `key_threats` | `list[ThreatEdge]` | Sorted by weight descending |

---

## Output Models

### `WinProbability`

| Field | Type | Description |
|---|---|---|
| `win_probability` | `float` [0,1] | Blended MC + MDP estimate |
| `confidence_interval` | `tuple[float, float]` | 95% Wilson CI |
| `sample_size` | `int` | Rollouts used |
| `calibrated` | `bool` | True if Platt scaler applied |
| `expected_runs` | `float` | Our team's expected first-innings score |
| `expected_runs_ci` | `tuple[float, float]` | 2.5th / 97.5th percentile |

### `TossDecision`

| Field | Type | Description |
|---|---|---|
| `decision` | `"bat" \| "bowl"` | Recommended call |
| `rationale` | `str` | e.g. `"bat 82.9% wins vs bowl 66.7%"` |
| `win_prob_batting_first` | `float` | Win probability if we bat first |
| `win_prob_bowling_first` | `float` | Win probability if we bowl first |

### `OracleResult`

Full pre-match output.

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | UUID (empty if not logged to StateStore) |
| `own_team` | `str` | Our team code |
| `opponent_team` | `str` | Opponent code |
| `fixture` | `Fixture` | Match details |
| `venue` | `Venue` | Venue details |
| `conditions` | `ConditionsVector` | Pitch/venue conditions |
| `opponent_forecast` | `OpponentForecast` | Predicted opponent XI + score + threats |
| `own_xi` | `XIOptimization` | Our selected XI + reasons |
| `win_probability` | `WinProbability` | MC + MDP estimate |
| `toss` | `TossDecision` | Toss recommendation |
| `narrative` | `str` | Claude brief or template |
| `decision_trace` | `dict` | Strategy mode, MC feedback, enrichment |

---

## JSON Data File Formats

### Squad file — `data/squads/<TEAM>.json`

```json
{
  "team_code": "RCB",
  "name": "Royal Challengers Bengaluru",
  "players": [
    {
      "player_id": "rcb_kohli",
      "name": "Virat Kohli",
      "role": "batsman",
      "is_overseas": false,
      "batting_avg": 38.6,
      "batting_sr": 138.4,
      "bowling_avg": 0.0,
      "bowling_econ": 0.0,
      "expected_runs_per_ball": 1.38,
      "expected_wickets_per_ball": 0.0,
      "form_score": 1.05,
      "form_variance": 0.08,
      "matches": 252
    }
  ]
}
```

All 10 IPL teams are required: `RCB`, `MI`, `CSK`, `KKR`, `DC`, `RR`, `SRH`, `PBKS`, `GT`, `LSG`.

### Venues file — `data/venues/venues.json`

```json
{
  "venues": [
    {
      "name": "M Chinnaswamy Stadium",
      "city": "Bengaluru",
      "boundary_straight_m": 67.0,
      "boundary_square_m": 60.0,
      "par_first_innings": 195.0,
      "chasing_win_rate": 0.55,
      "dew_factor": 0.6,
      "pitch_type": "batting"
    }
  ]
}
```

### Fixtures file — `data/fixtures/fixtures.json`

```json
{
  "season": 2026,
  "fixtures": [
    {
      "match_id": "2026-IPL-01",
      "match_date": "2026-03-28",
      "home_team": "RCB",
      "away_team": "SRH",
      "venue": "M Chinnaswamy Stadium",
      "season": 2026
    }
  ]
}
```

---

## Extending the Dataset

### Adding a new player to a squad

1. Add an entry to `data/squads/<TEAM>.json` with all required `Player` fields.
2. Set `form_score = 1.0` and `form_variance = 0.1` as sensible defaults.
3. Set `expected_runs_per_ball` and `expected_wickets_per_ball` from career stats (batting_sr / 600 and 1 / bowling_avg / 6 are rough starting points).
4. Run `ipl-oracle squad <TEAM>` to verify the player appears and validates.

### Adding a new venue

1. Add a `Venue` entry to `data/venues/venues.json`.
2. Ensure `name` exactly matches how it appears in `fixtures.json`.
3. Set `par_first_innings` from recent match data at that ground.

### Refreshing fixtures from Cricsheet

```bash
python scripts/ingest_cricsheet.py
```

This fetches the current season's match list from cricsheet.org and regenerates `fixtures.json`. Squad form scores are **not** overwritten — they must be updated manually or via `retro` runs. See [Data and Refresh](Data-and-Refresh).
