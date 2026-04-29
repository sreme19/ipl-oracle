"""Pydantic contracts that flow between agents."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PlayerRole(str, Enum):
    BATSMAN = "batsman"
    BOWLER = "bowler"
    ALL_ROUNDER = "all_rounder"
    WICKET_KEEPER = "wicket_keeper"


class Player(BaseModel):
    player_id: str
    name: str
    role: PlayerRole
    is_overseas: bool = False
    batting_avg: float = 0.0
    batting_sr: float = 0.0
    bowling_avg: float = 0.0
    bowling_econ: float = 0.0
    expected_runs_per_ball: float = Field(default=0.0, ge=0.0)
    expected_wickets_per_ball: float = Field(default=0.0, ge=0.0)
    form_score: float = Field(default=1.0, ge=0.0)
    form_variance: float = Field(default=0.1, ge=0.0)
    matches: int = 0


class Squad(BaseModel):
    team_code: str
    name: str
    players: list[Player]


class Venue(BaseModel):
    name: str
    city: str
    boundary_straight_m: float = 70.0
    boundary_square_m: float = 65.0
    par_first_innings: float = 170.0
    chasing_win_rate: float = 0.5
    dew_factor: float = 0.0
    pitch_type: Literal["batting", "bowling", "spin", "neutral"] = "neutral"


class Fixture(BaseModel):
    match_id: str
    match_date: date
    home_team: str
    away_team: str
    venue: str
    season: int


class ConditionsVector(BaseModel):
    venue: str
    alpha: float
    beta: float
    size_factor: float
    par_score: float
    dew_factor: float
    chasing_advantage: float
    pitch_type: str
    notes: list[str] = Field(default_factory=list)


class ThreatEdge(BaseModel):
    batter_id: str
    bowler_id: str
    weight: float
    threat_level: Literal["low", "medium", "high"]
    expected_runs_per_ball: float
    dismissal_rate: float


class OptimizationMode(str, Enum):
    DETERMINISTIC = "deterministic"
    ROBUST = "robust"
    BAYESIAN = "bayesian"


class XIOptimization(BaseModel):
    selected_xi: list[Player]
    objective_value: float
    baseline_value: float
    improvement_pct: float
    mode: OptimizationMode
    solve_time_ms: float
    excluded: list[Player] = Field(default_factory=list)
    slot_reasons: dict[str, str] = Field(default_factory=dict)
    note: str | None = None


class OpponentForecast(BaseModel):
    predicted_xi: XIOptimization
    expected_score: float
    expected_score_ci: tuple[float, float]
    bowling_phase_strengths: dict[str, float]
    batting_phase_strengths: dict[str, float]
    key_threats: list[ThreatEdge]


class WinProbability(BaseModel):
    win_probability: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    sample_size: int
    calibrated: bool
    expected_runs: float
    expected_runs_ci: tuple[float, float]


class TossDecision(BaseModel):
    decision: Literal["bat", "bowl"]
    rationale: str
    win_prob_batting_first: float
    win_prob_bowling_first: float


class OracleResult(BaseModel):
    run_id: str = ""
    own_team: str
    opponent_team: str
    fixture: Fixture
    venue: Venue
    conditions: ConditionsVector
    opponent_forecast: OpponentForecast
    own_xi: XIOptimization
    win_probability: WinProbability
    toss: TossDecision
    narrative: str
    decision_trace: dict


class OracleRequest(BaseModel):
    team: str
    opponent: str | None = None
    venue: str | None = None
    match_date: date | None = None
    formation_bias: Literal["batting", "balanced", "bowling"] = "balanced"
    must_include: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)
    sample_size: int = 10000
