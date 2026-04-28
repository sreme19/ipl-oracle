# Handoff — running ipl-oracle locally + splitting into its own repo

This guide takes you from cloning the IPL-kiro branch where ipl-oracle was
built to running it locally and (optionally) splitting it into a brand-new
GitHub repository with clean history.

---

## 1. Get the code

The ipl-oracle project lives under the `ipl-oracle/` subdirectory of the
IPL-kiro repository, on branch `claude/plan-multiagent-optimization-1jIF7`.

```bash
git clone https://github.com/sreme19/IPL-kiro.git
cd IPL-kiro
git checkout claude/plan-multiagent-optimization-1jIF7
cd ipl-oracle
```

Everything below assumes you are inside `ipl-oracle/`.

---

## 2. Local dev setup (any IDE)

Requirements: Python 3.10, 3.11, or 3.12. macOS, Linux, or Windows (WSL recommended).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the package in editable mode plus pytest + ruff. The
`ipl-oracle` console script is now on your PATH.

Set the Anthropic key (only needed for the natural-language brief):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

You can put it in a local `.env` file (already gitignored) and source it,
or add it to your shell profile. The CLI also accepts `--no-narrative` to
skip the LLM call entirely.

---

## 3. First run

```bash
ipl-oracle run --team RCB
```

The system will:
1. Look up RCB's next fixture (uses today's date by default).
2. Load both squads.
3. Predict the opponent's XI.
4. Forecast their score and bowling profile.
5. Pick our XI with the chosen optimization mode.
6. Run 10k Monte Carlo + MDP value iteration.
7. Emit a natural-language brief.

If today's date is past the bundled fixtures, anchor the search:

```bash
ipl-oracle run --team RCB --match-date 2026-04-01
```

Other useful invocations:

```bash
# What-if: override opponent / venue / date
ipl-oracle run --team RCB --opponent PBKS --venue "M Chinnaswamy Stadium" --match-date 2026-04-15

# Lock players in or out
ipl-oracle run --team RCB --must-include rcb_kohli --must-exclude rcb_maxwell

# Skip the LLM call (deterministic structured output only)
ipl-oracle run --team RCB --no-narrative

# Emit full JSON trace for downstream tooling
ipl-oracle run --team RCB --json > trace.json

# Inspect data
ipl-oracle fixtures --team RCB
ipl-oracle squad RCB
```

---

## 4. Tests + lint

```bash
pytest -q
ruff check .
```

Nine tests should pass. CI runs the same on push (see
`.github/workflows/ci.yml`).

---

## 5. Refreshing the dataset

The bundled `ipl_oracle/data/` folder is a curated 2026 sample. To pull
real per-ball data from Cricsheet:

```bash
python scripts/ingest_cricsheet.py --output ipl_oracle/data --seasons 2024 2025
```

This downloads `ipl_json.zip` (~150 MB) and rewrites `players/`,
`squads/`, `venues/`, and `h2h/`. The script never touches `fixtures/` —
future fixtures aren't in Cricsheet, so update
`ipl_oracle/data/fixtures/fixtures.json` by hand from the IPL website at
the start of each season.

If you can't reach cricsheet.org from your machine, download
`ipl_json.zip` separately and pass it in:

```bash
python scripts/ingest_cricsheet.py --archive /path/to/ipl_json.zip --output ipl_oracle/data
```

---

## 6. Splitting into a fresh GitHub repo (clean history)

The `ipl-oracle/` directory is fully self-contained — it has its own
`pyproject.toml`, `README.md`, `.github/workflows/ci.yml`, and
`.gitignore`. Lifting it out is a copy + `git init`.

```bash
# From the parent of IPL-kiro
mkdir ipl-oracle-new
cp -R IPL-kiro/ipl-oracle/. ipl-oracle-new/
cd ipl-oracle-new

# Drop the parent-overrides — they were only for living inside IPL-kiro
# (the original IPL-kiro .gitignore blocked data/ and *.toml; in a fresh
# repo there's nothing to override)
# Keep .gitignore as-is; the `!` re-includes are harmless in a clean repo.

git init
git add .
git commit -m "initial commit: ipl-oracle multi-agent IPL XI optimizer"
git branch -M main

# Create the empty repo on GitHub first (no README, no .gitignore — we
# already have them), then:
git remote add origin git@github.com:<your-user>/ipl-oracle.git
git push -u origin main
```

After the first push, GitHub Actions CI (`.github/workflows/ci.yml`) will
run `pytest` and `ruff check` automatically on every push and PR.

If you'd rather preserve the IPL-kiro commit history that touched
`ipl-oracle/`, use `git subtree split` instead:

```bash
cd IPL-kiro
git subtree split --prefix=ipl-oracle -b ipl-oracle-only
cd ..
git clone -b ipl-oracle-only IPL-kiro ipl-oracle
cd ipl-oracle
git remote remove origin
git remote add origin git@github.com:<your-user>/ipl-oracle.git
git push -u origin ipl-oracle-only:main
```

---

## 7. Map of the codebase

```
ipl-oracle/
├── pyproject.toml                          # editable install + console script
├── README.md                               # project overview
├── HANDOFF.md                              # this file
├── .github/workflows/ci.yml                # GitHub Actions: pytest + ruff
├── ipl_oracle/
│   ├── cli.py                              # `ipl-oracle` entrypoint (Typer)
│   ├── orchestrator.py                     # wires every agent end-to-end
│   ├── schemas.py                          # pydantic contracts between agents
│   ├── agents/
│   │   ├── fixture.py                      # next-match lookup
│   │   ├── conditions.py                   # venue → α/β formation weights
│   │   ├── scout.py                        # Bayesian form posterior
│   │   ├── opponent.py                     # OpponentSelector + OpponentStrategist
│   │   ├── strategist.py                   # picks optimization mode
│   │   ├── selector.py                     # MILP for own XI
│   │   ├── simulator.py                    # MC + MDP, win prob + toss
│   │   └── narrator.py                     # Anthropic API → brief
│   ├── optimization/
│   │   ├── milp.py                         # PuLP/CBC binary program
│   │   ├── robust.py                       # Bertsimas-Sim Γ-robust
│   │   ├── bayesian.py                     # Normal-Normal + Thompson
│   │   ├── mdp.py                          # value iteration
│   │   └── calibration.py                  # Platt scaling (sklearn)
│   ├── io/
│   │   ├── loader.py                       # JSON dataset reader
│   │   └── state.py                        # SQLite EWM + calibration log
│   └── data/                               # bundled 2026 sample dataset
│       ├── fixtures/fixtures.json
│       ├── squads/{RCB,PBKS,MI,...}.json
│       └── venues/venues.json
├── scripts/
│   └── ingest_cricsheet.py                 # refresh data from cricsheet.org
└── tests/
    ├── test_milp.py
    ├── test_optimization.py                # bayesian + robust + mdp + platt
    └── test_orchestrator.py                # end-to-end smoke tests
```

---

## 8. Common edits

**Update a player's stats**: edit `ipl_oracle/data/players/<id>.json` (if
you've ingested) or the inline entry in
`ipl_oracle/data/squads/<TEAM>.json` (curated dataset).

**Change role-balance constraints**: `ipl_oracle/optimization/milp.py` →
`MILPConfig` (n_players, min_wicketkeepers, min_bowlers, min_batters,
max_overseas).

**Tune optimization-mode thresholds**:
`ipl_oracle/agents/strategist.py` (`opp_std > 1.5` → robust,
`own_std > 0.20` → bayesian).

**Change MC sample size default**: `ipl_oracle/schemas.py`
(`OracleRequest.sample_size`) or pass `--sample-size N` at the CLI.

**Where state lives**: `~/.ipl-oracle/state.db` by default. Override with
`IPL_ORACLE_STATE_DB=/some/path/state.db`. Delete the file to reset form
and calibration history.

---

## 9. Troubleshooting

**`Missing data file: .../squads/XYZ.json`** — you typed a team code
that isn't in the bundled dataset. Codes are RCB, PBKS, MI, CSK, KKR,
DC, RR, SRH, GT, LSG.

**`ANTHROPIC_API_KEY not set`** — export the key, or pass
`--no-narrative` to skip the LLM call. The structured output (XI,
win-prob, toss) does not require the key.

**MILP infeasible** — happens if `--must-include` over-constrains role
minima or overseas cap. Loosen the locks or check the squad has at
least one wicket-keeper, four bowlers/all-rounders, etc.

**Cricsheet download fails** — the public endpoint occasionally
rate-limits. Use `--archive /path/to/ipl_json.zip` after downloading the
zip in a browser.

**CI fails on a fresh repo** — first push triggers Actions; if it can't
find Python or fails on lint, check `.github/workflows/ci.yml` against
your runner's Python version.

---

## 10. Next steps you might want

- Replace the curated squads with a full Cricsheet pull (`scripts/ingest_cricsheet.py`).
- Add a season-2026 fixtures scraper from `iplt20.com` so
  `ipl_oracle/data/fixtures/fixtures.json` updates itself.
- Log every prediction with the actual outcome via
  `StateStore.add_calibration_point(predicted, actual)` after each match —
  the Platt scaler activates automatically once five points are logged.
- Tighten the MDP scoring distribution (`_BASE_DIST` in
  `ipl_oracle/optimization/mdp.py`) by fitting it to historical
  ball-by-ball data.
