# Glossary

Cricket, optimisation, and machine-learning terms used throughout `ipl-oracle`.

---

## A

**α (alpha)**
The batting weight in the `ConditionsVector`. Represents how much venue conditions favour run-scoring. Ranges [0, 1]; α + β = 1. Higher at batting-friendly venues (small boundaries, flat pitches).

**All-rounder**
A player who contributes meaningfully with both bat and ball. Counts toward both the batting constraint (≥5) and bowling constraint (≥4) in the MILP.

---

## B

**β (beta)**
The bowling weight in the `ConditionsVector`. Represents how much venue conditions favour bowlers. α + β = 1. Higher at bowling-friendly venues (large boundaries, seaming pitches).

**Bayesian mode**
MILP optimization mode selected when own-squad form uncertainty is high (posterior std > 0.20). Uses Thompson sampling from Normal-Normal posteriors as MILP weights. See [Technical Reference § Bayesian](Technical-Reference).

**Bellman recursion**
The dynamic programming equation for the MDP value function: `V(t, w, r) = max over per-ball outcomes of expected next-state value`. Solved by backward induction from terminal states. See [Technical Reference § MDP](Technical-Reference).

**Bertsimas-Sim robustness**
A tractable linear reformulation of robust integer programming. Protects against the Γ most uncertain inputs. Named after the 2004 paper by Bertsimas and Sim. See [Technical Reference § Robust MILP](Technical-Reference).

**Bipartite threat matching**
A weighted graph between our batters and opponent bowlers. Edge weight = threat magnitude for that specific matchup. High-weight edges reduce the batter's MILP weight. See [Technical Reference § Bipartite Threat Matching](Technical-Reference).

---

## C

**Calibration**
The process of correcting systematic bias in win probability estimates using historical results. `ipl-oracle` uses Platt scaling (logistic regression on predicted vs actual outcomes). See [Calibration](Calibration).

**CBC**
Coin-OR Branch and Cut — the open-source MILP solver used via PuLP. Solves IPL-sized selection problems in milliseconds.

**CI (confidence interval)**
95% Wilson confidence interval around the win probability estimate. Wide CI (> 0.18) triggers the CI Width Loop for refinement.

**CI Width Loop**
Post-selection loop that tightens the robust Γ parameter if the 95% CI width exceeds 0.18, re-selecting and re-simulating until the estimate is precise enough. Up to 3 attempts. See [Multi-Agent System § CI Width Loop](Multi-Agent-System).

**Conjugate prior**
A prior distribution that, when combined with a specific likelihood, yields a posterior of the same family. `ipl-oracle` uses the Normal-Normal conjugate model: Normal prior on form score, Normal likelihood for observations → Normal posterior.

---

## D

**Death overs**
Overs 17–20. High-stakes phase where hitting is maximised. Phase strength for "death" appears in `OpponentForecast.bowling_phase_strengths`.

**Deterministic mode**
Default MILP mode. Uses fixed form-score-based weights without uncertainty adjustments. See [Technical Reference § MILP](Technical-Reference).

**Dew factor**
`Venue.dew_factor` — the expected impact of dew on evening matches. High dew makes chasing easier (ball becomes slippery, harder to grip for bowlers). Encoded in `ConditionsVector.dew_factor` and factored into `WinProbability`.

**Dismissal probability**
`Player.expected_wickets_per_ball` × `form_score` — the per-ball probability of a batsman being dismissed by a particular bowler. Used in both Monte Carlo rollouts and the MDP transition model.

---

## E

**Economy rate**
Runs conceded per over by a bowler. Lower is better. Stored as `Player.bowling_econ`. Used to weight bowling strength in `expected_wickets_per_ball`.

**EWM (Exponentially Weighted Moving Average)**
The form-decay algorithm: `f_t = α·y + (1−α)·f_{t-1}`. α = 0.4 by default. Recent performance decays with a half-life of ~1.3 matches. See [Technical Reference § EWM](Technical-Reference).

**Expected runs per ball**
`Player.expected_runs_per_ball` — a player's expected scoring rate. For batters: batting_sr / 600 (approx). For bowlers: economy / 6. Used as the base rate in Monte Carlo simulations.

---

## F

**Fan-in / fan-out**
LangGraph concurrency terms — not used in `ipl-oracle` (sequential orchestrator). Included here because `got-oracle` and `dhurandhar-oracle` use LangGraph with these patterns.

**Fixture**
A scheduled IPL match. Contains match_id, date, home/away teams, venue, and season. Loaded from `data/fixtures/fixtures.json`.

**Formation bias**
CLI flag `--formation batting|balanced|bowling` that shifts the α/β venue weights ±0.05–0.10 to reflect the team's strategic intent.

**Form score**
`Player.form_score` — a multiplier on expected performance metrics. 1.0 = average form; > 1.0 = above average. Seeded from squad JSON, updated by EWM after retro runs.

---

## Γ (Gamma)

The robustness budget in Bertsimas-Sim optimization. Controls how many uncertain inputs are hedged against. Γ = 0 → deterministic; Γ = N → full minimax robustness. `ipl-oracle` scales Γ between 1.5 and 3.0 based on opponent uncertainty.

---

## H

**Half-life (EWM)**
The number of matches after which a single observation's influence on form score is halved. For α = 0.4, half-life ≈ 1.3 matches — recent form dominates quickly.

---

## M

**MC Feedback Loop**
Post-MILP loop that tries marginal player swaps to further improve simulated win probability. Up to 5 rounds, each adopting a swap if Δwin_prob > 0.5%. See [Multi-Agent System § MC Feedback Loop](Multi-Agent-System).

**MDP (Markov Decision Process)**
The chase probability model. State = (overs remaining, wickets lost, runs remaining). Value iteration over 120 balls. See [Technical Reference § MDP](Technical-Reference).

**MILP (Mixed-Integer Linear Programming)**
The XI selection algorithm. Binary variables (in/out), linear objective (sum of player weights), hard constraints (role counts, overseas cap). See [Technical Reference § MILP](Technical-Reference).

**Must-exclude**
CLI `--must-exclude` flag. Player IDs forced out of the XI (injured, suspended, tactical). Encoded as `xⱼ = 0` constraints in the MILP.

**Must-include**
CLI `--must-include` flag. Player IDs forced into the XI (captain, vice-captain, marquee players). Encoded as `xⱼ = 1` constraints in the MILP.

---

## N

**Normal-Normal conjugate model**
Prior: `form_score ~ N(μ₀, σ₀²)`. Likelihood: `observation ~ N(f, σ_y²)`. Posterior: `N(μ_post, σ_post²)` with closed-form update. Used in Bayesian mode for per-player form estimation. See [Technical Reference § Bayesian](Technical-Reference).

---

## O

**Objective value**
The MILP's optimised objective function value — the weighted sum of selected players. `XIOptimization.objective_value` vs `baseline_value` (greedy top-11) shows how much the constrained MILP adds.

**Overseas cap**
IPL rule limiting overseas players to a maximum of 4 in any XI. Encoded as a MILP constraint: `Σ xᵢ · 𝟙[overseas] ≤ 4`.

---

## P

**Par score**
`Venue.par_first_innings` — the expected first-innings total at a given venue. Used by `ConditionsAgent` to calibrate opponent score forecasts.

**Phase strengths**
`OpponentForecast.bowling_phase_strengths` and `batting_phase_strengths` — multiplicative factors for each match phase (powerplay: overs 1–6, middle: 7–16, death: 17–20). Encodes phase-specific performance patterns.

**Platt scaling**
Logistic calibration of raw win probabilities using historical (predicted, actual) pairs. Corrects systematic over/under-confidence. Requires minimum 5 calibration points. See [Calibration](Calibration) and [Technical Reference § Platt Scaling](Technical-Reference).

**Posterior**
The updated probability distribution over a player's form score after combining prior (squad JSON) and observation (EWM form from StateStore). Represented as the `Posterior` dataclass with `mean`, `variance`, `std`, and `sample()`.

**Powerplay**
Overs 1–6. Fielding restrictions (max 2 fielders outside the 30-yard circle) favour batters and openers.

**PuLP**
Python linear programming library used to formulate and solve the MILP. Uses CBC as the default solver.

---

## R

**Retro command**
Post-match root cause analysis. Compares predictions to actual outcomes, updates form scores via EWM, and logs a Platt calibration point. See [Post-Match RCA](Post-Match-RCA).

**Robust mode**
MILP mode selected when opponent threat uncertainty is high (edge std > 1.5). Uses Bertsimas-Sim Γ-robustness to down-weight players whose performance depends on knowing the exact opponent lineup. See [Technical Reference § Robust MILP](Technical-Reference).

**run_id**
UUID assigned to each oracle invocation logged to `StateStore.runs`. Used with `record-outcome <run_id> --won/--lost` to update calibration.

---

## S

**Sample size**
Number of Monte Carlo rollouts. Default: 10,000 for full fidelity; 2,000 for MC Feedback Loop swap evaluation.

**StateStore**
SQLite-backed persistence layer at `~/.ipl-oracle/state.db`. Stores form scores, calibration history, and run snapshots. See [Multi-Agent System § SQLite State](Multi-Agent-System).

**Strike rate**
Runs scored per 100 balls (`batting_sr`). Used to estimate `expected_runs_per_ball`.

---

## T

**Thompson sampling**
A Bayesian exploration strategy: sample a weight from each player's posterior `N(μ_post, σ_post²)` rather than using the mean directly. Players with high posterior uncertainty explore more, injecting controlled randomness into XI selection. Used in Bayesian mode. See [Technical Reference § Bayesian](Technical-Reference).

**Threat edge**
A batter-bowler matchup risk in the bipartite threat graph. `ThreatEdge.weight` penalises the batter's MILP weight. See [Technical Reference § Bipartite Threat](Technical-Reference).

**Toss decision**
`TossDecision` — whether to choose bat or bowl if we win the toss, based on which scenario yields higher simulated win probability.

---

## V

**Value iteration**
The dynamic programming algorithm for solving the MDP. Iterates the Bellman recursion backward from terminal states (over 0 remaining, target achieved/missed) to compute `V(overs, wickets, runs)` for all reachable states.

**Venue**
Stadium with associated metadata: boundary sizes, par score, historical chase win rate, dew factor, pitch type. Loaded from `data/venues/venues.json`.

---

## W

**Wicket-keeper**
A fielding specialist who also bats. Exactly one must be in every XI. Encoded as `1 ≤ Σ xᵢ · 𝟙[role=WICKET_KEEPER] ≤ 1` in the MILP.

**Wilson confidence interval**
The confidence interval formula used for proportions. More accurate than the normal approximation near 0 and 1. Used for both win probability CI and expected runs CI. See [Technical Reference § Monte Carlo](Technical-Reference).

**Win probability**
`WinProbability.win_probability` — the blended MC + MDP estimate of our team winning the match. Optionally Platt-calibrated. Reported with a 95% confidence interval.
