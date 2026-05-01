# Technical Reference

Authoritative specification for every algorithm in `ipl-oracle`. For a plain-language overview see [Modeling Techniques](Modeling-Techniques). For data schemas see [Schema Reference](Schema-Reference).

---

## Algorithm Dependency Chain

The orchestrator runs agents in a fixed sequential order. Each algorithm's output feeds the next:

```
FixtureAgent
  → ConditionsAgent          (venue → α/β weights)
    → OpponentSelector        (predict opponent XI via MILP)
      → OpponentStrategist    (forecast opponent score + threat graph)
        → ScoutAgent          (Bayesian form posteriors for our squad)
          → StrategistAgent   (pick optimization mode)
            → SelectorAgent   (select our XI via MILP)
              → SimulatorAgent (MC + MDP → win probability)
                → [MC Feedback Loop — up to 5 rounds of swaps]
                  → [CI Width Loop — up to 3 refinement attempts]
                    → NarratorAgent (Claude briefing)
```

---

## 1. Mixed-Integer Linear Programming (MILP)

**Question answered:** Given player form scores and role constraints, which 11 players maximise expected match performance?

**Source:** `optimization/milp.py`, consumed by `agents/selector.py` (our XI) and `agents/opponent.py` (opponent XI prediction)

**Why MILP:** XI selection is a combinatorial optimisation problem — binary choices (in/out) with hard constraints (roles, overseas caps). MILP gives the provably optimal solution, unlike heuristic greedy selection. PuLP with the CBC solver handles typical IPL squad sizes (20–25 players) in milliseconds.

### Inputs

| Field | Type | Description |
|---|---|---|
| `players` | `list[Player]` | Full squad with computed per-player weights |
| `must_include` | `list[str]` | Player IDs locked into XI |
| `must_exclude` | `list[str]` | Player IDs locked out |
| `mode` | `OptimizationMode` | `deterministic` \| `robust` \| `bayesian` — determines weight assembly |

### Outputs — `XIOptimization`

| Field | Type | Description |
|---|---|---|
| `selected_xi` | `list[Player]` | 11 selected players (or fewer if relaxed) |
| `objective_value` | `float` | MILP objective at optimum |
| `baseline_value` | `float` | Top-11-by-weight baseline (greedy) |
| `improvement_pct` | `float` | `(objective − baseline) / baseline × 100` |
| `mode` | `OptimizationMode` | Mode used |
| `solve_time_ms` | `float` | Solver wall time |
| `excluded` | `list[Player]` | Bench players |
| `slot_reasons` | `dict[str, str]` | Human-readable reason per selected player |
| `note` | `Optional[str]` | Constraint relaxation note if infeasible |

### Mathematical Formulation

**Decision variables:** xᵢ ∈ {0, 1} for each player i in the squad.

**Objective:**

```
maximise  Σᵢ wᵢ · xᵢ
```

**Constraints:**

```
Σᵢ xᵢ = 11                                    (XI size)
Σᵢ xᵢ · 𝟙[role=WICKET_KEEPER] ≥ 1             (min 1 keeper)
Σᵢ xᵢ · 𝟙[role=WICKET_KEEPER] ≤ 1             (max 1 keeper)
Σᵢ xᵢ · 𝟙[role ∈ {BOWLER, ALL_ROUNDER}] ≥ 4  (min 4 bowling options)
Σᵢ xᵢ · 𝟙[role ∈ {BATSMAN, ALL_ROUNDER, WICKET_KEEPER}] ≥ 5  (min 5 batting options)
Σᵢ xᵢ · 𝟙[overseas] ≤ 4                       (overseas cap)
xⱼ = 1  ∀j ∈ must_include
xⱼ = 0  ∀j ∈ must_exclude
```

If infeasible, constraints relax in order: keeper bound → bowling bound → overseas cap. A `note` is appended to the output.

**Per-player weight assembly** (before mode adjustments):

```
wᵢ = form_score_i × (α × batting_weight_i + β × bowling_weight_i) + matchup_penalty_i
```

where α, β come from `ConditionsVector` and `matchup_penalty` is the negative threat exposure from the opponent's key bowlers.

---

## 2. Bertsimas-Sim Γ-Robust Optimization

**Question answered:** How should the XI change when there is high uncertainty about the opponent's XI?

**Source:** `optimization/robust.py`, consumed by `agents/selector.py`

**Why Robust:** When the opponent XI prediction has high uncertainty (threat edge std > 1.5), a purely deterministic MILP optimised against the expected scenario may perform poorly against alternative opponent lineups. Γ-robustness protects against the Γ most uncertain inputs without being overly conservative across all of them.

### When triggered

`StrategistAgent` selects `mode=robust` when the standard deviation of opponent threat edge weights exceeds 1.5. Γ is scaled between 1.5 and 3.0 proportionally to uncertainty.

### Mathematical Formulation

The robust formulation penalises each player's weight by their worst-case contribution loss when Γ uncertain inputs deviate adversarially:

```
wᵢ' = wᵢ − λᵢ · δᵢ
```

where the optimal penalty allocation solves:

```
maximise  Σᵢ (wᵢ − λᵢ · δᵢ) · xᵢ
subject to  Σᵢ λᵢ ≤ Γ
            λᵢ ∈ [0, 1]  ∀i
            (+ original XI constraints)
```

δᵢ is the uncertainty magnitude for player i (derived from threat edge variance). This is the Bertsimas-Sim (2004) linear reformulation of the robust integer programme — the worst-case adversary selects the Γ players with highest λᵢ · δᵢ to penalise.

**Effect:** Players whose performance depends on knowing the opponent's exact bowling lineup (e.g., aggressive openers vulnerable to specific bowler types) are down-weighted. All-rounders and consistent performers are up-weighted.

---

## 3. Bayesian Normal-Normal Form Posterior + Thompson Sampling

**Question answered:** What is the current best estimate of each player's true form, accounting for both long-run stats and recent match observations?

**Source:** `optimization/bayesian.py`, consumed by `agents/scout.py`

**Why Bayesian:** A player's bundled `form_score` represents long-run expectation (the prior). Recent performance (the observation) should update this. The Normal-Normal conjugate model gives a closed-form posterior — no sampling required for the posterior parameters. Thompson sampling then converts posterior uncertainty into diverse weight draws, which avoids over-fitting to stale data.

### When triggered

`StrategistAgent` selects `mode=bayesian` when any own-player posterior std exceeds 0.20 (high form uncertainty, e.g., a returning player).

### Inputs

| Field | Source | Description |
|---|---|---|
| `μ₀` (prior mean) | `Player.form_score` | Long-run form score |
| `σ₀²` (prior variance) | `Player.form_variance` | Uncertainty in the prior |
| `y` (observation) | `StateStore.get_form()` | EWM-decayed recent form score |
| `σ_y²` (obs noise) | Empirical (0.05²) | Assumed observation noise |

### Mathematical Formulation

**Posterior parameters (conjugate update):**

```
σ_post² = 1 / (1/σ₀² + 1/σ_y²)
μ_post  = σ_post² · (μ₀/σ₀² + y/σ_y²)
```

**Thompson sampling** — draw a weight for the MILP:

```
w_i ~ N(μ_post_i, σ_post_i²)
```

This injects controlled stochasticity: players with high posterior uncertainty explore more, while players with confident posteriors dominate. The MILP is then solved once with these sampled weights.

### Outputs — `Posterior` (dataclass)

| Field | Description |
|---|---|
| `mean` | μ_post — posterior mean form score |
| `variance` | σ_post² |
| `std` | √σ_post² |
| `sample()` | Draw from N(mean, variance) for Thompson sampling |

---

## 4. Markov Decision Process (MDP) Value Iteration

**Question answered:** Given a target score, remaining overs, and current form rates, what is the probability of a successful chase?

**Source:** `optimization/mdp.py`, consumed by `agents/simulator.py`

**Why MDP:** A single Monte Carlo probability estimate does not distinguish between "lots of overs with wickets in hand" and "few overs with runs required" scenarios. Value iteration over the (overs, wickets, runs) state space gives a principled probability that accounts for the full match state, not just averages.

### State Space

```
State = (overs_remaining, wickets_lost, runs_remaining)
overs_remaining ∈ {0, 1, …, 20}
wickets_lost    ∈ {0, 1, …, 10}
runs_remaining  ∈ {0, 1, …, 250}  (discretised)
```

### Inputs

| Parameter | Description |
|---|---|
| `target` | Runs required to win |
| `overs` | Overs remaining in chase |
| `runs_per_ball` | Expected runs per ball (from batting form scores) |
| `dismissal_prob` | P(wicket per ball) (from bowling form scores) |

### Mathematical Formulation

**Per-ball transition:**

```
P(runs=k) = base_dist[k] × (runs_per_ball / base_rpb)    [normalised]
P(wicket)  = dismissal_prob
```

where `_BASE_DIST = [0.40, 0.36, 0.06, 0.005, 0.10, 0.075]` for outcomes `[0, 1, 2, 3, 4, 6]`.

**Bellman recursion** (backward from terminal):

```
V(0, w, r) = 1.0 if r ≤ 0 else 0.0         [terminal: last ball]
V(t, 10, r) = 0.0                            [all out]
V(t, w, r)  = P(wicket) · V(t−1, w+1, r)
            + Σ_k P(runs=k) · V(t−1, w, r−k)
```

Solved by iterating over all states from t=0 upward to t=overs.

**Output:** `V(overs, 0, target)` — probability of a successful chase from the initial state.

**Blending with Monte Carlo:**

```
win_prob = 0.5 × MC_bat_first_win_prob + 0.5 × (1 − MDP_chase_prob)
```

---

## 5. Monte Carlo Ball-by-Ball Simulation

**Question answered:** Across 10,000 simulated matches, what fraction does our team win — and under what toss scenario?

**Source:** `agents/simulator.py`

### Inputs

| Field | Source | Description |
|---|---|---|
| `our_xi` | `SelectorAgent` | Our selected XI with form scores |
| `opp_forecast` | `OpponentStrategist` | Opponent expected score, phase strengths |
| `conditions` | `ConditionsAgent` | Venue factors (par score, dew factor) |
| `sample_size` | CLI `--sample-size` | Default 10,000 rollouts |

**Per-team rates computed from form scores:**

```
runs_per_ball    = mean(expected_runs_per_ball × form_score)  over batters
dismissal_prob   = mean(expected_wickets_per_ball × form_score) over bowlers
```

### Innings Simulation

For each rollout, simulate 120 balls:

```python
for ball in range(120):
    if wickets >= 10: break
    run_outcome ~ Multinomial(base_dist × runs_per_ball adjustment)
    wicket       ~ Bernoulli(dismissal_prob)
    score       += run_outcome
    wickets     += wicket
```

### Toss Scenarios

Both scenarios are simulated per rollout:

- **Batting first:** Simulate our innings → score T; simulate opponent chase of T
- **Bowling first:** Simulate opponent innings → score T; simulate our chase of T

**Win probability:**

```
P(win | bat first)  = fraction of rollouts where our_score > opp_chase_score
P(win | bowl first) = fraction of rollouts where we_chase_successfully
```

**Toss decision:** `bat` if `P(win|bat) > P(win|bowl)`, else `bowl`.

### Outputs — `WinProbability` + `TossDecision`

`WinProbability`:

| Field | Description |
|---|---|
| `win_probability` | Blended MC + MDP estimate |
| `confidence_interval` | 95% Wilson CI |
| `sample_size` | Rollouts used |
| `calibrated` | True if Platt scaler was applied |
| `expected_runs` | Our team's expected first-innings score |
| `expected_runs_ci` | 2.5th / 97.5th percentile |

`TossDecision`:

| Field | Description |
|---|---|
| `decision` | `"bat"` or `"bowl"` |
| `rationale` | e.g. `"bat 82.9% wins vs bowl 66.7%"` |
| `win_prob_batting_first` | Float |
| `win_prob_bowling_first` | Float |

### Wilson Confidence Interval

```
CI = (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)
```

z = 1.96 for 95% coverage.

---

## 6. Platt Scaling (Logistic Calibration)

**Question answered:** Is our raw win probability well-calibrated — and if not, how can historical results correct it?

**Source:** `optimization/calibration.py`, consumed by `agents/simulator.py`

**Why Platt scaling:** MC + MDP models make assumptions (base run distributions, independence of balls) that may produce systematically over- or under-confident probabilities. Platt scaling fits a logistic curve to `(predicted_prob, actual_outcome)` pairs, correcting systematic bias without retraining the underlying model.

### Activation

Minimum 5 calibration points required (stored in `StateStore.calibration` table). Applied automatically when available.

### Mathematical Formulation

Fit logistic regression on historical `(f(x), y)` pairs where f(x) is raw win probability and y ∈ {0, 1} is actual outcome:

```
P(win | f(x)) = sigmoid(A · f(x) + B)   =   1 / (1 + exp(−(A·f(x) + B)))
```

Parameters A, B are fit by maximum likelihood (sklearn `LogisticRegression`).

**Transform:** `calibrated_prob = sigmoid(A × raw_prob + B)`

### Building the calibration dataset

Each `ipl-oracle retro` run (or manual `record-outcome`) logs one calibration point. Over a season (60+ matches), the scaler converges to well-calibrated probabilities.

---

## 7. Exponentially Weighted Moving Average (EWM) Form Decay

**Question answered:** How should yesterday's match performance update a player's form score without completely discarding their long-run record?

**Source:** `io/state.py` → `StateStore.update_form_ewm()`

### Mathematical Formulation

**Score update** (new observation y, decay factor α = 0.4):

```
f_t  = α · y + (1 − α) · f_{t−1}
```

**Variance update** (online estimate):

```
σ_t² = (1 − α) · (σ_{t−1}² + α · (y − f_{t−1})²)
```

Both f and σ² are persisted in SQLite. Initial values seeded from `Player.form_score` and `Player.form_variance` in the squad JSON.

**α = 0.4** means recent observations decay with half-life ≈ 1.3 matches — recent form matters but one bad game doesn't collapse a player's rating.

---

## 8. Bipartite Weighted Threat Matching

**Question answered:** Which of our batters are most exposed to the opponent's key bowlers — and by how much?

**Source:** `agents/opponent.py` → `OpponentStrategist.forecast()`

### Mechanism

Construct a bipartite graph: our batters × opponent bowlers. Edge weight encodes expected runs per ball for that batter-bowler pairing (higher = more exposed to being dismissed or restricted).

**Threat edge:**

```
weight = expected_runs_per_ball_i × dismissal_rate_ij
```

where dismissal_rate_ij is the bowler's historical tendency against this batter type.

### Outputs — `ThreatEdge` per batter-bowler pair

| Field | Description |
|---|---|
| `batter_id` | Our player ID |
| `bowler_id` | Opponent player ID |
| `weight` | Composite threat magnitude |
| `threat_level` | `"low"` / `"medium"` / `"high"` |
| `expected_runs_per_ball` | Expected scoring rate |
| `dismissal_rate` | P(dismissal per ball) by this bowler vs this batter |

**Used by `SelectorAgent`:** Batter's MILP weight is penalised by their maximum threat exposure across the opponent's predicted XI, proportional to the bowler's form score.

---

## 9. MC Feedback Loop

**Question answered:** Can marginal player swaps improve our win probability beyond what the MILP already gives?

**Source:** `orchestrator.py` → `_run_mc_feedback_loop()`

### Mechanism

After the initial MILP selection and full-fidelity simulation:

1. Identify the 3 players in the XI with the lowest MILP weights (not in `must_include`)
2. For each: try swapping with every bench player of the same role
3. Simulate each swap at 2,000 rollouts (fast)
4. If Δwin_prob > 0.5% for any swap → adopt it, update XI
5. Repeat up to 5 rounds

**Complexity:** Up to 5 × 3 × (bench_size) × 2,000 MC calls — typically < 1 second.

**When it triggers meaningful swaps:** When the MILP objective function weights don't perfectly correlate with simulated win probability (e.g., a specialist death-overs bowler has low average weights but high matchup value given this opponent's batting lineup).

---

## 10. CI Width Refinement Loop

**Question answered:** Is the win probability estimate precise enough to make a confident decision?

**Source:** `orchestrator.py` → `_run_ci_width_loop()`

### Mechanism

If the 95% CI width exceeds 0.18 after the initial simulation:

1. Increase Γ (robust parameter) to tighten XI selection toward more consistent players
2. Re-run `SelectorAgent` with updated Γ
3. Re-simulate at full fidelity (10,000 rollouts)
4. Repeat up to 3 attempts

Converges when CI width ≤ 0.18 or maximum attempts reached.

**Rationale:** A CI of (0.45, 0.65) is too wide to recommend a confident toss decision. Tightening the robust parameter reduces XI variance, which in turn reduces simulation variance.

---

## Algorithm Summary

| Algorithm | Stage | Source | Key Output |
|---|---|---|---|
| MILP (deterministic) | Opponent prediction + our XI | `optimization/milp.py` | `XIOptimization` |
| Γ-Robust MILP | Our XI (high uncertainty) | `optimization/robust.py` | Adjusted MILP weights |
| Bayesian Normal-Normal + Thompson | Scout form | `optimization/bayesian.py` | `Posterior` per player |
| MDP Value Iteration | Chase probability | `optimization/mdp.py` | P(chase success) |
| Monte Carlo (10k rollouts) | Win probability | `agents/simulator.py` | `WinProbability`, `TossDecision` |
| Platt Scaling | Calibration | `optimization/calibration.py` | Calibrated win prob |
| EWM Form Decay | Persistent form update | `io/state.py` | Updated form score + variance |
| Bipartite Threat Matching | Matchup weighting | `agents/opponent.py` | `ThreatEdge` list |
| MC Feedback Loop | Post-MILP swap search | `orchestrator.py` | Refined XI + win prob |
| CI Width Loop | Precision control | `orchestrator.py` | Converged CI |
