# Multi-Agent System

Documents the 10-agent sequential pipeline, each agent's contract, the two post-selection feedback loops, state persistence, and the narrator integration. For algorithm internals see [Technical Reference](Technical-Reference). For the layer stack see [Architecture](Architecture).

---

## Overview

`ipl-oracle` uses a plain sequential orchestrator (`Orchestrator` in `orchestrator.py`), not LangGraph. There is no shared TypedDict — state flows through method arguments. Each agent is a class with a single primary method.

```
OracleRequest
    │
    ▼
Orchestrator.run()  ──► 10 agents in fixed order ──► OracleResult
```

---

## Agent Pipeline

### Step 1 — `FixtureAgent`

**File:** `agents/fixture.py`

**Question:** What is the next match for this team?

| | |
|---|---|
| **Inputs** | `team: str`, `opponent: str \| None`, `venue: str \| None`, `match_date: date \| None` |
| **Outputs** | `Fixture` |
| **Logic** | If all four provided → synthesise fixture directly. Otherwise → load `fixtures.json`, find next upcoming match for the team by date. Raises if no fixture found. |

---

### Step 2 — Data Loading

**File:** `io/loader.py`

Not an agent — pure data loading. Loads and validates three JSON files:

| Loaded | Schema | Source |
|---|---|---|
| Own squad | `Squad` | `data/squads/{team}.json` |
| Opponent squad | `Squad` | `data/squads/{opponent}.json` |
| Venue | `Venue` | `data/venues/venues.json` (looked up by name) |

---

### Step 3 — `ConditionsAgent`

**File:** `agents/conditions.py`

**Question:** What do pitch and venue conditions mean for batting vs bowling weights?

| | |
|---|---|
| **Inputs** | `venue: Venue`, `formation_bias: "batting" \| "balanced" \| "bowling"` |
| **Outputs** | `ConditionsVector` |
| **Logic** | Maps venue geometry (boundary sizes), pitch type, and dew factor to α (batting weight) and β (bowling weight) such that α + β = 1. Formation bias shifts the α/β split by ±0.05–0.10. |

`ConditionsVector` fields:

| Field | Description |
|---|---|
| `alpha` | Run-scoring weight [0, 1] |
| `beta` | Bowling/defensive weight [0, 1]; α + β = 1 |
| `size_factor` | Boundary ratio (0.85–1.20) |
| `par_score` | First-innings par runs for this venue |
| `dew_factor` | Dew impact on chasing (0 = no dew, 1 = heavy dew) |
| `chasing_advantage` | Historical chase win rate at this venue |
| `pitch_type` | `"batting"` \| `"bowling"` \| `"spin"` \| `"neutral"` |
| `notes` | List of coaching notes |

---

### Step 4 — `OpponentSelector`

**File:** `agents/opponent.py`

**Question:** Which XI will the opponent likely field?

| | |
|---|---|
| **Inputs** | `opponent_squad: Squad`, `conditions: ConditionsVector` |
| **Outputs** | `XIOptimization` (predicted opponent XI) |
| **Logic** | Runs deterministic MILP on the opponent squad using bundled form scores and venue weights. No Bayesian or robust adjustments — opponent uncertainty is handled downstream by `OpponentStrategist`. |

---

### Step 5 — `OpponentStrategist`

**File:** `agents/opponent.py`

**Question:** What score will the opponent post, and where is our batting lineup exposed?

| | |
|---|---|
| **Inputs** | `own_squad: Squad`, `predicted_xi: XIOptimization`, `conditions: ConditionsVector` |
| **Outputs** | `OpponentForecast` |
| **Logic** | Estimates expected score from predicted XI form scores and venue par score. Builds bipartite threat graph (our batters × their bowlers). Classifies phase strengths (powerplay / middle / death). |

`OpponentForecast` fields:

| Field | Description |
|---|---|
| `predicted_xi` | `XIOptimization` from Step 4 |
| `expected_score` | Predicted opponent total runs |
| `expected_score_ci` | 95% CI |
| `bowling_phase_strengths` | `{"powerplay": float, "middle": float, "death": float}` |
| `batting_phase_strengths` | Same for batting |
| `key_threats` | `list[ThreatEdge]` — batter-bowler exposure edges |

---

### Step 6 — `ScoutAgent`

**File:** `agents/scout.py`

**Question:** What is each own-squad player's current estimated form?

| | |
|---|---|
| **Inputs** | `players: list[Player]`, `state_store: StateStore \| None` |
| **Outputs** | `dict[str, Posterior]` — one posterior per player |
| **Logic** | For each player: prior = (Player.form_score, Player.form_variance). If StateStore has a recent EWM observation, compute conjugate Normal-Normal posterior. Otherwise posterior = prior (as a Posterior dataclass). |

---

### Step 7 — `StrategistAgent`

**File:** `agents/strategist.py`

**Question:** Should we use deterministic, robust, or Bayesian MILP for our XI selection?

| | |
|---|---|
| **Inputs** | `own_posteriors: dict[str, Posterior]`, `opponent_forecast: OpponentForecast` |
| **Outputs** | `StrategyDecision` |

Mode selection logic:

| Condition | Mode selected | Γ |
|---|---|---|
| Threat edge std > 1.5 | `robust` | Scaled 1.5–3.0 by std |
| Any own posterior std > 0.20 | `bayesian` | N/A |
| Otherwise | `deterministic` | 0 |

`StrategyDecision` fields:

| Field | Description |
|---|---|
| `mode` | `OptimizationMode` |
| `gamma` | Γ for robust mode (0 otherwise) |
| `rationale` | Human-readable explanation |

---

### Step 8 — `SelectorAgent`

**File:** `agents/selector.py`

**Question:** Which 11 players should we field?

| | |
|---|---|
| **Inputs** | `own_squad`, `own_posteriors`, `conditions`, `forecast`, `decision`, `must_include`, `must_exclude` |
| **Outputs** | `XIOptimization` |
| **Logic** | Assembles per-player MILP weights: `form × venue_weights + matchup_adjustment`. Applies robust or Bayesian weight adjustments based on `StrategyDecision`. Runs MILP solver. Assigns slot reasons. |

**Weight assembly pipeline:**

```
base_weight    = form_score × (α × batting_weight + β × bowling_weight)
matchup_adj    = −max_threat_exposure × bowler_form_score     [for batters]
robust_adj     = −λᵢ × δᵢ                                    [if robust mode]
bayesian_draw  = sample from N(μ_post, σ_post²)               [if bayesian mode]

final_weight   = base_weight + matchup_adj [+ robust_adj | bayesian_draw]
```

---

### Step 9 — `SimulatorAgent`

**File:** `agents/simulator.py`

**Question:** What is our win probability and toss recommendation?

| | |
|---|---|
| **Inputs** | `our_xi: XIOptimization`, `opp_forecast: OpponentForecast`, `conditions: ConditionsVector`, `sample_size: int` |
| **Outputs** | `(WinProbability, TossDecision)` |
| **Logic** | Computes per-team runs_per_ball and dismissal_prob from XI form scores. Runs MC (both batting/bowling first scenarios). Blends MC with MDP chase probability. Applies Platt scaler if ≥5 calibration points. |

---

### Step 10 — MC Feedback Loop

**File:** `orchestrator.py` → `_run_mc_feedback_loop()`

Not a separate agent — an orchestrator-level loop that uses `SelectorAgent` + `SimulatorAgent` internally.

**Contract:**

| | |
|---|---|
| **Inputs** | Current XI, win_prob, toss; bench players; state_store |
| **Outputs** | Refined XI, win_prob, toss; `rounds` count; `delta` improvement |
| **Termination** | Δwin_prob ≤ 0.5% for all swaps, or 5 rounds reached |

Each candidate swap: 2,000 rollouts (vs 10,000 full fidelity) for speed.

---

### Step 11 — CI Width Loop

**File:** `orchestrator.py` → `_run_ci_width_loop()`

| | |
|---|---|
| **Inputs** | Current XI, win_prob (with CI); current Γ |
| **Outputs** | Converged (XI, win_prob, toss) |
| **Termination** | CI width ≤ 0.18, or 3 attempts |

Each attempt: increases Γ → re-runs SelectorAgent + full-fidelity SimulatorAgent.

---

### Step 12 — `NarratorAgent`

**File:** `agents/narrator.py`

**Question:** What is the plain-English strategy brief?

| | |
|---|---|
| **Inputs** | Full trace dict (fixture, conditions, forecast, our XI, win prob, toss, strategy decision, analyst enrichment if available) |
| **Outputs** | `narrative: str` (350–500 words) |
| **Model** | `claude-opus-4-5` (or template fallback if no API key) |

**System prompt role:** Cricket analyst briefing team management before a match. Tone: confident, analytical, numbers-first. Covers: why this XI, what the conditions favour, how the opponent threatens us, the toss call, and the risk scenario.

---

## OracleResult — Full Output Contract

```python
class OracleResult(BaseModel):
    run_id:             str            # UUID (empty if not persisted)
    own_team:           str
    opponent_team:      str
    fixture:            Fixture
    venue:              Venue
    conditions:         ConditionsVector
    opponent_forecast:  OpponentForecast  # predicted XI + score + threats
    own_xi:             XIOptimization    # selected XI + reasons
    win_probability:    WinProbability    # MC + MDP + optional Platt
    toss:               TossDecision      # bat/bowl + per-scenario probs
    narrative:          str               # Claude brief or template
    decision_trace:     dict              # analyst_enrichment, strategy, mc_feedback details
```

---

## `decision_trace` Structure

The `decision_trace` dict carries the full audit trail:

```python
{
    "analyst_enrichment": {
        "league_trends": str,
        "team_insights": {TEAM: {...}},
        "key_players_watchlist": [...]
    },
    "strategy": {
        "mode": "robust" | "deterministic" | "bayesian",
        "gamma": float,
        "rationale": str
    },
    "mc_feedback": {
        "rounds": int,
        "delta": float,          # total win_prob improvement from swaps
        "swaps": [...]           # list of player IDs swapped in/out
    }
}
```

---

## SQLite State — Full Schema

```sql
CREATE TABLE form (
    player_id   TEXT PRIMARY KEY,
    form_score  REAL NOT NULL,
    form_variance REAL NOT NULL,
    updated_at  REAL NOT NULL      -- Unix timestamp
);

CREATE TABLE calibration (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted   REAL NOT NULL,     -- raw win probability
    actual      REAL NOT NULL,     -- 1 = won, 0 = lost
    created_at  REAL NOT NULL
);

CREATE TABLE runs (
    run_id             TEXT PRIMARY KEY,
    run_at             REAL NOT NULL,
    team               TEXT NOT NULL,
    opponent           TEXT NOT NULL,
    venue              TEXT NOT NULL,
    strategy_mode      TEXT NOT NULL,
    gamma              REAL NOT NULL,
    selected_xi        TEXT NOT NULL,  -- JSON array of player IDs
    win_probability    REAL NOT NULL,
    ci_lower           REAL NOT NULL,
    ci_upper           REAL NOT NULL,
    ci_width           REAL NOT NULL,
    toss_decision      TEXT NOT NULL,
    refinement_rounds  INTEGER NOT NULL,
    mc_feedback_rounds INTEGER NOT NULL DEFAULT 0,
    mc_feedback_delta  REAL NOT NULL DEFAULT 0.0,
    solve_time_ms      REAL NOT NULL,
    sample_size        INTEGER NOT NULL,
    calibrated         INTEGER NOT NULL,  -- 0/1 boolean
    outcome_recorded   INTEGER NOT NULL DEFAULT 0
);
```

Default path: `~/.ipl-oracle/state.db`. Override with `IPL_ORACLE_STATE_DB` env var.
