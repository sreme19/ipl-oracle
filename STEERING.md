# ipl-oracle — Steering Document

> Multi-agent IPL XI optimizer. Ten agents, each typed. You give it a team code;
> it gives you a playing-eleven recommendation, win probability, toss call,
> and a written brief explaining the reasoning.
>
> **This document covers:** architecture decisions, technique rationale, and the
> operational upgrade roadmap (observability, guardrails, evaluation, model routing).

---

## Origin

First project in the oracle series. Built to learn MILP-based combinatorial
optimisation and multi-agent orchestration. Deliberately sequential (no parallel
fan-out) — each agent hands a typed Pydantic contract to the next, making the
data flow explicit and testable.

Introduced: MILP (PuLP/CBC), Bertsimas-Sim robust counterpart, Bayesian Normal-Normal
posterior + Thompson sampling, MDP value iteration, Monte Carlo, Platt scaling.

Later projects (got-oracle, dhurandhar-oracle) introduced game-theoretic and partially-
observable extensions, and replaced the sequential pipeline with LangGraph graphs.

---

## Architecture

### Agent Pipeline (sequential)

```
FixtureAgent
    ↓
ConditionsAgent (venue → α/β formation weights)
    ↓
ScoutAgent (×2, Bayesian form posterior)
    ↓
OpponentSelector (predictive MILP — what XI will they field?)
    ↓
OpponentStrategist (forecast score + bowler profile + threat graph)
    ↓
StrategistAgent (picks: deterministic / robust / Bayesian MILP)
    ↓
SelectorAgent (prescriptive MILP — picks our XI)
    ↓
SimulatorAgent (10k Monte Carlo + MDP value iteration)
    ↓
MC Feedback Loop (swap search: up to 5 rounds, δ = 0.5% threshold)
    ↓
NarratorAgent (Claude — natural-language brief)
    ↓
LinkedInAgent (deterministic templating — LinkedIn post)
```

The only LLM call in the pipeline is `NarratorAgent`. Every other agent is
deterministic: MILP, Bayesian posteriors, Monte Carlo.

### State Persistence (SQLite)

`~/.ipl-oracle/state.db` — three tables:

| Table | Contents |
|---|---|
| `form` | EWM-decayed form score + variance per player, updated each run |
| `calibration` | `(predicted, actual)` pairs for Platt scaler |
| `runs` | Full eval snapshot: win prob, CI width, toss, MC feedback rounds, outcome |

### Two Modes

| Mode | CLI | What it computes |
|---|---|---|
| Pre-match strategy | `run --team RCB` | XI selection + win probability forecast |
| Post-match RCA | `retro --team RCB ...` | Why prediction diverged from result |

---

## Why These Algorithms

**MILP for XI selection:** The selection problem has hard constraints (role minimums,
overseas cap, matchup scores) and a combinatorial solution space. MILP gives the
globally optimal XI subject to constraints, not a greedy heuristic.

**Bertsimas-Sim robust counterpart:** The opponent XI is uncertain (we're predicting
it). Robust optimisation protects against the Γ most-uncertain players in the
opponent's predicted lineup, avoiding brittle selections that only work if we guessed
their XI perfectly.

**Bayesian form posterior:** EWM form scores conflate signal and noise. Normal-Normal
conjugate update separates posterior mean (best estimate of true form) from posterior
variance (how uncertain we are). Thompson sampling adds principled exploration.

**MDP value iteration + Monte Carlo:** MC gives empirical win probability across
10,000 rollouts. MDP value iteration gives exact chase win probability on a
(overs, wickets, runs-needed) grid. Both are needed: MC for batting-first scenarios,
MDP for chase scenarios where you know the target.

**Platt scaling:** Raw MC probabilities are often over- or under-confident. Platt
scaling (logistic regression on `(predicted, actual)` pairs) calibrates them.
Activates automatically after 5 recorded outcomes.

---

## Operational Upgrade Roadmap

### Context: The Protocol Layer Gap

ipl-oracle already has the strongest observability foundation of the three oracle
projects — SQLite `runs` table, Platt calibration, MC feedback loop delta tracking.
The gaps are: LLM-specific observability (narrator tokens/latency), structured
evaluation of narrative quality, and Bedrock migration for model agnosticism.

---

### Phase 1 — LLM Observability (narrator)

**Goal:** See narrator token usage and latency alongside the deterministic agent metrics.

The `runs` table tracks win probability, CI width, MC rounds. It does not track anything
about the narrator call — cost, latency, or whether it hallucinated a player name.

**What to add to `StateStore`:**

```python
# New columns in `runs` table
narrator_model        TEXT     # which model was used
narrator_input_tok    INTEGER
narrator_output_tok   INTEGER
narrator_latency_ms   INTEGER
narrator_grounded     INTEGER  # 1 if all named players are in the XI, 0 otherwise
```

**Narrator grounding check** (simple, high-value):
After the narrator returns, extract all player names from the narrative and verify
each appears in the recommended XI or bench. Any name not in either is a hallucination.
```python
def check_narrator_grounding(narrative: str, xi: list[str], bench: list[str]) -> bool:
    # Returns False if any player name in narrative is not in xi + bench
```

**Bedrock path:** Enable Bedrock model invocation logging → CloudWatch. Query with
CloudWatch Insights across all oracle projects: `filter @message like "narrator_node"`.

---

### Phase 2 — Intelligent Model Routing

**Goal:** Use the right model for the task. Not every brief needs the most capable model.

The narrator has two distinct use cases with different quality requirements:

| Use case | Complexity | Current | Should Use |
|---|---|---|---|
| `--no-narrative` flag | Skip entirely | — | — |
| `--linkedin` post only | Low — templated | Sonnet | Haiku |
| Standard pre-match brief | Medium | Sonnet | Sonnet |
| Post-match RCA | High — causal analysis | Sonnet | Sonnet/Opus |
| `--brief` flag | Low | (not implemented) | Haiku |

**What to change in `narrator.py`:**
- Add `--brief` flag (mirrors dhurandhar-oracle) → uses Haiku
- For LinkedIn-only output (`--linkedin` without `--narrative`): skip Claude, use
  deterministic `LinkedInAgent` template
- Read `NARRATOR_MODEL` env var for override

**Bedrock path:** Bedrock Intelligent Prompt Routing — set a routing profile that sends
brief/LinkedIn requests to Haiku and full briefs to Sonnet automatically.

---

### Phase 3 — Evaluation Pipeline

**Goal:** Treat prediction accuracy as a measurable KPI, not a vibe.

ipl-oracle has the best foundation for this of any project in the series — the
calibration table already records `(predicted, actual)` pairs. The gap is that
there's no dashboard or periodic eval job that surfaces the accuracy metrics.

**What to build:**

1. **`ipl-oracle eval` CLI command** — reads the `calibration` table and outputs:
   ```
   Calibration summary (N=47 matches)
   ─────────────────────────────────────
   Brier score:         0.187
   Log loss:            0.412
   Calibration plot:    exported to ~/.ipl-oracle/calibration.png
   Win rate when >70%:  78.3% (actual vs predicted)
   ```

2. **Platt scaler auto-activation** — currently requires manual tracking. Add a
   startup check: if `calibration` table has ≥5 rows, auto-fit the scaler on init.

3. **Narrator evaluation** — create `tests/golden/narrator/` with 3 fixed
   (team, fixture) pairs. For each, verify the narrative:
   - Names the correct XI players (grounding check)
   - States the correct win probability within 2% of the structured output
   - Does not contradict the toss recommendation

**Bedrock path:** Bedrock Model Evaluation — run weekly evaluation jobs against the
golden dataset. Track hallucination rate as a KPI in the Bedrock console.

---

### Phase 4 — Guardrails

**Goal:** Prevent the narrator from producing harmful or misleading content.

ipl-oracle is lower-stakes than the medical ivf-chatbot, but narratives go into
LinkedIn posts and are read by cricket fans who may act on them.

**What to add:**

1. **Confidence qualifier** — if win probability CI width > 10%, the narrative must
   include a hedging phrase ("uncertain conditions", "wide confidence interval").
   Assert this in the narrator wrapper.

2. **Player name validation** — already described in Phase 1. Make this blocking:
   if a hallucinated player name is detected, regenerate once with an explicit
   instruction to only name players from the provided XI.

3. **Factual constraint** — the narrator receives the structured output as context.
   Add to the system prompt: "Only name players from the XI and bench provided.
   Do not invent statistics. All probabilities must match the structured output."

**Bedrock path:** Bedrock Guardrails with grounding check — verifies that narrative
claims are grounded in the structured output provided as context.

---

### Phase 5 — Bedrock Migration

**Goal:** Route narrator through Bedrock for unified governance across all oracle projects.

**Migration path (narrator.py):**

```python
# Before (Anthropic SDK)
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(model="claude-sonnet-4-6", ...)

# After (Bedrock)
import boto3
client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.converse(
    modelId="anthropic.claude-sonnet-4-6",
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    inferenceConfig={"maxTokens": 1024}
)
```

**What Bedrock adds:**
- **Model agnosticism** — swap narrator to Llama 3 for cost benchmarking
- **Intelligent Prompt Routing** — auto-select Haiku vs Sonnet per call
- **Unified CloudWatch dashboard** — ipl-oracle + got-oracle + dhurandhar-oracle
  narrator calls in one log group
- **Guardrails** — attach grounding check to every narrator call

---

### Priority Order

| Phase | Effort | Value | Do When |
|---|---|---|---|
| 1 — LLM Observability | 4 hours | High | Now — narrator is a black box |
| 2 — Model routing | 2 hours | Medium | After observability baseline |
| 3 — Evaluation pipeline | 1 day | High | Has the best foundation — just surface it |
| 4 — Guardrails | 4 hours | Medium | Before LinkedIn post feature is used publicly |
| 5 — Bedrock migration | 1 day | High | When unifying all oracles under one platform |

---

## Refresh Cycle

- **Fixtures:** Update `ipl_oracle/data/fixtures/fixtures.json` manually from IPL website
  at start of each season. Future: add a scraper for `iplt20.com`.
- **Player data:** Run `scripts/ingest_cricsheet.py` after each match to update form scores.
- **Platt scaler:** Feed every completed match result via `ipl-oracle record-outcome`.
- **Calibration review:** Run `ipl-oracle eval` at season mid-point and end to track drift.

---

## Development Environment

- **Claude Code** + **Windsurf** — used for implementation
- **GitHub Actions** — `pytest` + `ruff check` on every push
- **Plane** — project tracking
