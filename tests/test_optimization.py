import numpy as np

from ipl_oracle.optimization.bayesian import bayesian_form_posterior
from ipl_oracle.optimization.calibration import PlattScaler
from ipl_oracle.optimization.mdp import value_iteration_win_prob
from ipl_oracle.optimization.robust import robust_objective_weights


def test_bayesian_posterior_pulls_toward_observation():
    post = bayesian_form_posterior(prior_mean=1.0, prior_var=0.5, observation=1.4, obs_var=0.1)
    assert 1.0 < post.mean < 1.4
    assert post.variance < 0.1  # narrower than observation alone


def test_robust_weights_penalize_uncertain_players():
    nominal = {"a": 10.0, "b": 10.0, "c": 10.0}
    uncertainty = {"a": 0.0, "b": 1.0, "c": 5.0}
    adj = robust_objective_weights(nominal, uncertainty, gamma=2.0)
    assert adj["a"] >= adj["b"] >= adj["c"]
    # Total penalty sums to no more than the budget.
    total_penalty = sum(nominal[k] - adj[k] for k in nominal)
    assert total_penalty <= 2.0 * max(uncertainty.values()) + 1e-6


def test_mdp_chase_prob_in_range():
    p = value_iteration_win_prob(target_runs=160, overs=20.0, runs_per_ball=1.4, dismissal_prob=0.04)
    assert 0.0 <= p <= 1.0
    p_higher_target = value_iteration_win_prob(
        target_runs=220, overs=20.0, runs_per_ball=1.4, dismissal_prob=0.04
    )
    assert p_higher_target <= p


def test_platt_scaler_passes_through_when_unfit():
    scaler = PlattScaler()
    assert abs(scaler.transform(0.6) - 0.6) < 1e-9


def test_platt_scaler_fits_logistic():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 0.9, size=50)
    actual = (raw + rng.normal(0, 0.05, size=50) > 0.5).astype(float)
    scaler = PlattScaler()
    scaler.fit(raw.tolist(), actual.tolist())
    assert scaler.is_fitted
    out = scaler.transform(0.5)
    assert 0.0 < out < 1.0
