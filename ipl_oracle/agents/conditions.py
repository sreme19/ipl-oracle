"""Conditions agent — translates a venue into α/β formation weights and notes."""
from __future__ import annotations

from ..schemas import ConditionsVector, Venue


class ConditionsAgent:
    def evaluate(self, venue: Venue, formation_bias: str = "balanced") -> ConditionsVector:
        size_factor = 70.0 / max(
            (venue.boundary_straight_m + venue.boundary_square_m) / 2.0, 1.0
        )
        size_factor = float(min(max(size_factor, 0.85), 1.20))

        if venue.pitch_type == "batting":
            base_alpha, base_beta = 0.60, 0.40
        elif venue.pitch_type == "bowling":
            base_alpha, base_beta = 0.40, 0.60
        elif venue.pitch_type == "spin":
            base_alpha, base_beta = 0.45, 0.55
        else:
            base_alpha, base_beta = 0.55, 0.45

        if formation_bias == "batting":
            base_alpha += 0.05
            base_beta -= 0.05
        elif formation_bias == "bowling":
            base_alpha -= 0.05
            base_beta += 0.05

        # Dew helps the chasing team (more bowling-friendly first innings).
        if venue.dew_factor > 0.5:
            base_beta = min(0.8, base_beta + 0.03)
            base_alpha = max(0.2, base_alpha - 0.03)

        total = base_alpha + base_beta
        alpha, beta = base_alpha / total, base_beta / total

        notes = []
        if size_factor > 1.05:
            notes.append("short boundaries inflate boundary % — pick power hitters")
        if size_factor < 0.95:
            notes.append("large outfield rewards strike rotation over slogging")
        if venue.dew_factor > 0.5:
            notes.append("dew expected — bowling first carries gripping risk")
        if venue.pitch_type == "spin":
            notes.append("spin-friendly surface — extra spinner pays off")
        if venue.par_first_innings >= 185:
            notes.append(f"par first innings {venue.par_first_innings:.0f} — high-scoring deck")
        elif venue.par_first_innings <= 155:
            notes.append(f"par first innings {venue.par_first_innings:.0f} — low-scoring deck")

        return ConditionsVector(
            venue=venue.name,
            alpha=alpha,
            beta=beta,
            size_factor=size_factor,
            par_score=venue.par_first_innings,
            dew_factor=venue.dew_factor,
            chasing_advantage=venue.chasing_win_rate,
            pitch_type=venue.pitch_type,
            notes=notes,
        )
