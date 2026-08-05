"""Unit tests for murmuration.analysis.direction_fluctuation against synthetic, known-answer
inputs (roadmap.md Phase 15) — same style as test_analysis.py."""

import numpy as np

import murmuration as m
from murmuration.analysis import direction_fluctuation as df


def test_mean_direction_of_aligned_boids_is_a_unit_vector():
    vel = np.tile(np.array([1.0, 0.0, 0.0]), (5, 10, 1))  # (T=5, N=10, 3), all aligned +X
    omega = df.mean_direction(vel)
    assert omega.shape == (5, 3)
    assert np.allclose(omega, [1.0, 0.0, 0.0])


def test_mean_direction_of_opposite_pairs_is_zero():
    vel = np.zeros((1, 4, 3))
    vel[0, 0] = [1.0, 0.0, 0.0]
    vel[0, 1] = [-1.0, 0.0, 0.0]
    vel[0, 2] = [0.0, 1.0, 0.0]
    vel[0, 3] = [0.0, -1.0, 0.0]
    omega = df.mean_direction(vel)
    assert np.allclose(omega, [0.0, 0.0, 0.0], atol=1e-10)


def test_angular_deviation_matches_known_geometry():
    ref = np.array([1.0, 0.0, 0.0])
    omega = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    delta = df.angular_deviation(omega, ref)
    assert np.allclose(delta, [0.0, np.pi / 2, np.pi, np.pi / 2], atol=1e-9)


def test_classify_growth_recovers_a_perfect_diffusive_law():
    t = np.linspace(0.0, 200.0, 300)
    d0 = 0.01
    msd = 2.0 * d0 * t  # exactly linear, matching eq. 14
    result = df.classify_growth(t, msd)
    assert result["favours"] == "diffusive"
    assert abs(result["diffusion_constant"] - d0) < 1e-6
    assert result["r2_diffusive"] > 0.999


def test_classify_growth_recovers_a_perfect_plateau():
    t = np.linspace(0.0, 200.0, 300)
    a, tau = 0.05, 15.0
    msd = a * (1.0 - np.exp(-t / tau))  # exactly the OU relaxation, eq. 12-13
    result = df.classify_growth(t, msd)
    assert result["favours"] == "plateau"
    assert result["r2_plateau"] > 0.999
    assert abs(result["plateau_value"] - a) / a < 0.1


def test_direction_fluctuation_pipeline_runs_on_a_real_simulation_trajectory():
    """Real end-to-end smoke test — not a check of *which* growth law wins.

    Unlike information_transfer's live comparison, this one doesn't need a mid-run command:
    the paper's own methodology (Fig. 1) compares two *independently configured* runs
    (field always-on vs. always-off from t=0), which `step_hooks=["external_field"]` at
    construction already supports. Tried directly (see sci/param_table.md's
    direction_fluctuation.py notes): at N=60/short pytest-scale windows, in this project's
    cohesion-dominant regime, neither the diffusive nor the plateau model fits well for
    either the field-on or field-off case (R^2 near zero or negative for both) — the
    <delta_psi(t)^2> curve itself doesn't cleanly resemble either textbook shape at this
    scale, a different and more basic issue than a favours-mismatch. Longer runs/larger N
    (closer to the reference paper's own N~10^5, L=256) are the natural next thing to try,
    not attempted here.
    """
    sim = m.Simulation(
        boid_count=50,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        steric_enabled=True,
        steric=0.6,
        step_hooks=["external_field"],
        field_x=1.0,
        field_strength=0.3,
    )
    sim.run_batch(100, 0)
    vel_traj = []
    for _ in range(150):
        sim.run_batch(1, 0)
        vel_traj.append(sim.velocities())
    vel_traj = np.array(vel_traj)

    omega = df.mean_direction(vel_traj)
    field_dir = np.array([1.0, 0.0, 0.0])
    delta_psi = df.angular_deviation(omega, field_dir)
    msd = delta_psi**2
    t = np.arange(len(msd), dtype=np.float64)

    result = df.classify_growth(t, msd)
    assert result["favours"] in ("diffusive", "plateau", "insufficient_data")
    if result["favours"] != "insufficient_data":
        assert np.isfinite(result["diffusion_constant"])
