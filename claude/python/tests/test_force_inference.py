"""Tests for `murmuration.analysis.force_inference` (Cai et al. 2021, roadmap.md Phase 18 /
design/03_observables_bindings.md §3). Every quantitative check is against a trajectory
*constructed* to exactly follow the model being fit (known weight, known delay, known/zero
weight on the other channels) — the same "exact synthetic construction" discipline every other
analysis module in this project uses, not a real `Simulation` for the numeric assertions
(real flocks don't hand you the ground-truth weights to check recovery against).
"""

import numpy as np

import murmuration as m
from murmuration.analysis import force_inference as fi


def test_alignment_term_matches_a_hand_computed_mean():
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = fi.alignment_term(vel)
    # boid 0's neighbours are 1 and 2 -> mean = (0, 0.5, 0.5)
    assert np.allclose(out[0], [0.0, 0.5, 0.5])


def test_alignment_term_with_no_neighbours_is_zero_not_undefined():
    vel = np.array([[1.0, 0.0, 0.0]])  # a lone boid, no "other" boids at all
    out = fi.alignment_term(vel)
    assert np.allclose(out[0], [0.0, 0.0, 0.0])


def test_repulsion_term_points_away_from_a_close_neighbour():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = fi.repulsion_term(pos, radius=5.0)
    assert np.allclose(out[0], [-1.0, 0.0, 0.0])
    assert np.allclose(out[1], [1.0, 0.0, 0.0])


def test_attraction_term_points_toward_a_far_neighbour_within_radius():
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = fi.attraction_term(pos, radius=5.0)
    assert np.allclose(out[0], [1.0, 0.0, 0.0])


def test_repulsion_and_attraction_ignore_neighbours_outside_the_radius():
    pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    assert np.allclose(fi.repulsion_term(pos, radius=5.0)[0], [0.0, 0.0, 0.0])
    assert np.allclose(fi.attraction_term(pos, radius=5.0)[0], [0.0, 0.0, 0.0])


def _synthetic_alignment_trajectory(n=20, t_steps=200, w_true=0.7, lambda_kin_true=200.0, seed=0):
    """A trajectory constructed to exactly follow `v_t = w_true * alignment(v_{t-1}) + noise`,
    `noise ~ N(0, 1/lambda_kin_true)` -- the ground truth `fit_weights`/`select_delay_time`
    are checked against below. Positions are static (repulsion/attraction aren't exercised by
    this particular synthetic trajectory, only alignment, which depends on velocity alone)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-5.0, 5.0, (n, 3))
    noise_std = (1.0 / lambda_kin_true) ** 0.5
    vel_traj = np.zeros((t_steps, n, 3))
    vel_traj[0] = rng.normal(0.0, 1.0, (n, 3))
    for t in range(1, t_steps):
        align = fi.alignment_term(vel_traj[t - 1])
        vel_traj[t] = w_true * align + rng.normal(0.0, noise_std, (n, 3))
    pos_traj = np.tile(pos, (t_steps, 1, 1))
    return pos_traj, vel_traj


def test_fit_weights_recovers_the_true_alignment_weight_and_lambda_kin():
    w_true, lambda_kin_true = 0.7, 200.0
    pos_traj, vel_traj = _synthetic_alignment_trajectory(w_true=w_true, lambda_kin_true=lambda_kin_true)
    fit = fi.fit_weights(pos_traj, vel_traj, dt=1.0, tau=1, channels=["alignment"])
    assert abs(fit["weights"]["alignment"] - w_true) < 0.05
    assert abs(fit["lambda_kin"] - lambda_kin_true) / lambda_kin_true < 0.1
    assert np.isfinite(fit["log_likelihood"])


def test_fit_weights_at_the_wrong_tau_recovers_a_worse_fit():
    pos_traj, vel_traj = _synthetic_alignment_trajectory()
    right = fi.fit_weights(pos_traj, vel_traj, dt=1.0, tau=1, channels=["alignment"])
    wrong = fi.fit_weights(pos_traj, vel_traj, dt=1.0, tau=3, channels=["alignment"])
    assert right["log_likelihood"] > wrong["log_likelihood"]


def test_select_delay_time_picks_the_true_tau():
    pos_traj, vel_traj = _synthetic_alignment_trajectory()
    selection = fi.select_delay_time(
        pos_traj, vel_traj, dt=1.0, tau_candidates=[1, 2, 3, 4], channels=["alignment"]
    )
    assert selection["best"]["tau"] == 1
    assert len(selection["all"]) == 4


def test_select_delay_time_with_no_valid_tau_returns_none_not_a_crash():
    pos_traj, vel_traj = _synthetic_alignment_trajectory(t_steps=3)
    selection = fi.select_delay_time(
        pos_traj, vel_traj, dt=1.0, tau_candidates=[100], channels=["alignment"]
    )
    assert selection["best"] is None


def test_compare_models_prefers_the_simpler_true_model_via_aic():
    # The synthetic trajectory's true generating process uses ONLY alignment (repulsion/
    # attraction weights are exactly zero). Raw log-likelihood would always weakly favour the
    # bigger "full" model (more free parameters can only help an OLS fit) -- the real point of
    # this test is that AIC, which compare_models actually ranks by, correctly penalises that
    # and prefers "vicsek" instead.
    pos_traj, vel_traj = _synthetic_alignment_trajectory()
    result = fi.compare_models(
        pos_traj,
        vel_traj,
        dt=1.0,
        tau_candidates=[1],
        model_channel_sets={
            "vicsek": ["alignment"],
            "full": ["alignment", "repulsion", "attraction"],
        },
        radii={"repulsion": 2.0, "attraction": 10.0},
    )
    assert result["best_model"] == "vicsek"
    assert result["models"]["full"]["fit"]["log_likelihood"] >= result["models"]["vicsek"]["fit"]["log_likelihood"] - 1e-6
    assert result["models"]["vicsek"]["aic"] < result["models"]["full"]["aic"]


def test_force_inference_report_matches_compare_models_with_its_own_defaults():
    pos_traj, vel_traj = _synthetic_alignment_trajectory()
    report = fi.force_inference_report(pos_traj, vel_traj, dt=1.0, tau_candidates=(1, 2, 3))
    assert report["best_model"] == "vicsek"
    assert "vicsek" in report["models"] and "full" in report["models"]


def _record_trajectory(mode, n=40, warmup=50, steps=60, seed=0, **kwargs):
    sim = m.Simulation(
        boid_count=n,
        mode=mode,
        init_seed=seed,
        **kwargs,
    )
    sim.run_batch(warmup, seed)
    pos_traj, vel_traj = [], []
    for _ in range(steps):
        sim.run_batch(1, seed)
        pos_traj.append(sim.positions())
        vel_traj.append(sim.velocities())
    return np.array(pos_traj), np.array(vel_traj)


def test_force_inference_report_runs_on_a_real_simulation_trajectory():
    """Real end-to-end smoke test against live `Simulation` trajectories from two different
    `FlockingMode`s -- proving this module genuinely operates on *any* trajectory regardless
    of which mode produced it (design/03_observables_bindings.md §3's own requirement), not a
    check of which model AIC prefers for a real (noisy, non-synthetic) flock."""
    for mode in ("vicsek", "pearce"):
        kwargs = dict(phi_p=0.50, phi_a=0.20, steric_enabled=True, steric=0.6) if mode == "pearce" else {}
        pos_traj, vel_traj = _record_trajectory(mode, **kwargs)
        report = fi.force_inference_report(pos_traj, vel_traj, dt=1.0, tau_candidates=(1, 2, 3))
        assert report["best_model"] in ("vicsek", "full")
        for name in ("vicsek", "full"):
            fit = report["models"][name]["fit"]
            assert fit is not None
            assert np.isfinite(fit["log_likelihood"])
            assert np.isfinite(report["models"][name]["aic"])
            for w in fit["weights"].values():
                assert np.isfinite(w)
