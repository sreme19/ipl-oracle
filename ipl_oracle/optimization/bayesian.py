"""Normal-Normal conjugate update for player form posterior + Thompson sampling.

We treat the per-ball expected runs (or wickets) for a player as a normal
random variable with prior mean μ₀ and prior variance σ₀². Each match
contributes a noisy observation y with variance σ_y². The posterior is the
classical Normal-Normal conjugate:

    σ_post² = 1 / (1/σ₀² + 1/σ_y²)
    μ_post  = σ_post² · (μ₀/σ₀² + y/σ_y²)

Thompson sampling draws a candidate weight from N(μ_post, σ_post²) so
the optimizer naturally explores under uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Posterior:
    mean: float
    variance: float

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-12)))

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.normal(self.mean, self.std))


def bayesian_form_posterior(
    prior_mean: float,
    prior_var: float,
    observation: float,
    obs_var: float,
) -> Posterior:
    if prior_var <= 0:
        return Posterior(observation, max(obs_var, 1e-6))
    if obs_var <= 0:
        return Posterior(observation, max(prior_var, 1e-6))
    inv = 1.0 / prior_var + 1.0 / obs_var
    var = 1.0 / inv
    mean = var * (prior_mean / prior_var + observation / obs_var)
    return Posterior(mean=mean, variance=var)


def thompson_weights(
    posteriors: dict[str, Posterior],
    seed: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {pid: post.sample(rng) for pid, post in posteriors.items()}
