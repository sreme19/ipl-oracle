# Architecture

System structure of `ipl-oracle`: layer stack, module dependency map, data flow, and the sequential pipeline design. For algorithm internals see [Technical Reference](Technical-Reference). For pipeline agent details see [Multi-Agent System](Multi-Agent-System).

---

## Layer Stack

```
┌──────────────────────────────────────────────────┐
│               CLI  (cli.py)                       │
│  Typer — run / retro / fixtures / squad /         │
│  enrichment-* / history / record-outcome /        │
│  linkedin / backfill                              │
├──────────────────────────────────────────────────┤
│            Orchestrator  (orchestrator.py)         │
│  Sequential agent coordinator                     │
│  MC Feedback Loop  ·  CI Width Loop               │
├──────────────────────────────────────────────────┤
│              Agent Layer  (agents/)               │
│  fixture · conditions · scout · opponent          │
│  strategist · selector · simulator                │
│  narrator · linkedin                              │
├──────────────────────────────────────────────────┤
│        Optimisation Algorithms (optimization/)    │
│  milp · robust · bayesian · mdp · calibration     │
├──────────────────────────────────────────────────┤
│   Data Models + I/O  (schemas.py / io/loader.py) │
│  Pydantic contracts · JSON loading                │
├──────────────────────────────────────────────────┤
│       Persistence  (io/state.py — SQLite)         │
│  form · calibration · runs tables                 │
├──────────────────────────────────────────────────┤
│         Data Layer  (ipl_oracle/data/)            │
│  squads/ · fixtures/ · venues/ · enrichment/      │
└──────────────────────────────────────────────────┘
```

**Key distinction from dhurandhar-oracle / got-oracle:** `ipl-oracle` does **not** use LangGraph. The orchestrator is a plain Python class (`Orchestrator`) that calls agents sequentially. State passes through method arguments, not a shared TypedDict.

---

## Module Dependency Map

```
cli.py
  └── orchestrator.py (Orchestrator.run / Orchestrator.retro)
        ├── agents/fixture.py         (FixtureAgent)
        ├── agents/conditions.py      (ConditionsAgent)
        ├── agents/opponent.py        (OpponentSelector, OpponentStrategist)
        ├── agents/scout.py           (ScoutAgent)
        ├── agents/strategist.py      (StrategistAgent)
        ├── agents/selector.py        (SelectorAgent)
        │     └── optimization/milp.py
        │     └── optimization/robust.py
        │     └── optimization/bayesian.py
        ├── agents/simulator.py       (SimulatorAgent)
        │     └── optimization/mdp.py
        │     └── optimization/calibration.py
        ├── agents/narrator.py        (NarratorAgent — Claude API)
        └── agents/linkedin.py        (LinkedInAgent — template)

io/loader.py   ←  JSON → Pydantic (squads, venues, fixtures)
io/state.py    ←  SQLite StateStore (form, calibration, runs)
io/cricinfo.py ←  Cricsheet API fetcher (retro mode)
schemas.py     ←  all Pydantic data contracts
```

---

## Data Flow — Pre-Match Run

```
User: ipl-oracle run --team RCB
         │
         ▼
      cli.py: run()
         │  OracleRequest = {team, opponent?, venue?, formation_bias, must_include, ...}
         ▼
   Orchestrator.run(request, state_store, narrator)
         │
         ▼ Step 1
   FixtureAgent.resolve(team, opponent?, venue?, match_date?)
   → Fixture (match_id, home/away teams, venue, date)
         │
         ▼ Step 2
   DataLoader.load_squad(team) + load_squad(opponent) + load_venue(venue)
   → Squad, Squad, Venue
         │
         ▼ Step 3
   ConditionsAgent.evaluate(venue, formation_bias)
   → ConditionsVector (α, β, par_score, dew_factor, pitch_type, notes)
         │
         ▼ Step 4
   OpponentSelector.predict(opponent_squad, conditions)
   → XIOptimization (predicted opponent XI, deterministic MILP)
         │
         ▼ Step 5
   OpponentStrategist.forecast(own_squad, predicted_xi, conditions)
   → OpponentForecast (expected_score, phase_strengths, key_threats=[ThreatEdge])
         │
         ▼ Step 6
   ScoutAgent.evaluate(own_squad.players, state_store)
   → dict[player_id → Posterior] (Bayesian form posteriors)
         │
         ▼ Step 7
   StrategistAgent.decide(own_posteriors, opponent_forecast)
   → StrategyDecision (mode: deterministic|robust|bayesian, gamma, rationale)
         │
         ▼ Step 8
   SelectorAgent.select(own_squad, posteriors, conditions, forecast, decision, must_include, must_exclude)
   → XIOptimization (our XI, slot_reasons)
         │
         ▼ Step 9
   SimulatorAgent.simulate(our_xi, opp_forecast, conditions, sample_size=10000)
   → (WinProbability, TossDecision)
         │
         ▼ Step 10 — MC Feedback Loop (up to 5 rounds)
   For each of 3 weakest players:
     try bench swap (same role) → simulate at 2k rollouts
     if Δwin_prob > 0.5%: adopt swap, continue
   → Updated (XI, win_prob, toss)
         │
         ▼ Step 11 — CI Width Loop (up to 3 attempts)
   If CI_width > 0.18:
     tighten Γ → re-select + re-simulate at full fidelity
   → Converged result
         │
         ▼ Step 12 (optional)
   StateStore.log_run(partial_result) → run_id (UUID)
         │
         ▼ Step 13 (optional — if narrator provided and ANTHROPIC_API_KEY set)
   NarratorAgent.narrate(trace_dict)
   → narrative (string, 350–500 words)
         │
         ▼
   OracleResult (complete)
         │
         ▼
   cli.py: renders Rich tables + narrative panel
```

---

## Data Flow — Post-Match Retro

```
User: ipl-oracle retro --team RCB --opponent GT --won --runs-for 202 --runs-against 195
         │
         ▼
   Orchestrator.retro(request, outcome)
         │
         ▼ (optional) CricinfoFetcher.fetch_result()  ← --fetch-result flag
         │
         ▼
   Re-run Step 1–9 against the same fixture parameters
         │
         ▼
   Compare predictions vs actuals:
     predicted_win_prob vs actual outcome (1/0)
     predicted_toss vs actual toss
     predicted_score vs runs_for / runs_against
         │
         ▼
   StateStore.add_calibration_point(predicted, actual)  ← if --log-calibration
   StateStore.update_form_ewm(player_id, observation)   ← for each player
         │
         ▼
   Render RCA report (Match Summary → Prediction Accuracy → What Went Wrong)
```

---

## Three Optimization Modes

`StrategistAgent` selects the mode before `SelectorAgent` runs. The MILP objective is the same in all modes — only how weights are computed differs.

| Mode | When triggered | Weight assembly | Effect |
|---|---|---|---|
| `deterministic` | Default; low uncertainty | `form_score × venue_weight + matchup_adjustment` | Straight MILP on best-estimate form |
| `robust` | Opponent threat std > 1.5 | Bertsimas-Sim penalty subtracts `λᵢ × δᵢ` for Γ uncertain players | Conservative; prefers consistent players |
| `bayesian` | Own-squad posterior std > 0.20 | Thompson sample from N(μ_post, σ_post²) | Explores uncertain players; good for returning players or form streaks |

---

## State Persistence (SQLite)

`StateStore` at `~/.ipl-oracle/state.db` (override with `IPL_ORACLE_STATE_DB` env var) persists three tables:

| Table | Purpose | Key operations |
|---|---|---|
| `form` | EWM form score + variance per player | `get_form()`, `update_form_ewm()` |
| `calibration` | (predicted, actual) pairs for Platt scaler | `add_calibration_point()`, `calibration_history()` |
| `runs` | Full snapshot of every oracle invocation | `log_run()`, `record_outcome()` |

Form updates happen in `retro` runs. Calibration points accumulate until 5+ exist, at which point `SimulatorAgent` fits and applies Platt scaling.

---

## Anthropic API Integration

`NarratorAgent` (`agents/narrator.py`) is the only component that calls the Anthropic API.

- **Model:** `claude-opus-4-5` (hardcoded)
- **Max tokens:** ~600
- **System prompt:** Cricket analyst briefing team management — concise, strategic, numbers-first
- **Input:** Compact trace dict (fixture, conditions, forecast, our XI, win prob, toss, strategy decision)
- **Fallback:** Rich-formatted template if `ANTHROPIC_API_KEY` not set or `--no-narrative` flag used

---

## Data Directory Structure

```
ipl_oracle/data/
├── fixtures/
│   └── fixtures.json          {season, fixtures: [{match_id, date, home, away, venue}]}
├── squads/
│   ├── RCB.json               {team_code, name, players: [Player]}
│   ├── MI.json
│   └── ...                    (10 teams total)
├── venues/
│   └── venues.json            {venues: [Venue]}
└── enrichment/
    ├── analyst_insights_2026.json    {league_trends, team_insights: {TEAM: {...}}}
    └── briefings_2026/
        └── RCB.md, MI.md, ...        team strategy markdown
```

Fixture data is bundled for the current IPL season. To refresh from Cricsheet, run `scripts/ingest_cricsheet.py`. See [Data and Refresh](Data-and-Refresh) for the full refresh workflow.
