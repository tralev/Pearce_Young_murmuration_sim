"""Unit tests for murmuration.analysis.information_transfer against synthetic, known-answer
inputs (roadmap.md Phase 15) — same style as test_analysis.py. The one live-`Simulation`
integration smoke test is at the bottom of this file (murmur_spin_wave/murmur_external_field
were only just wired into murmur_py's bindings for this — see murmur_py's own module changes).
"""

import numpy as np

import murmuration as m
from murmuration.analysis import information_transfer as it


def test_dominant_motion_plane_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 300)
    # Velocities confined to the XY plane, arbitrary in-plane direction, plus tiny Z jitter.
    vel = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)
    vel = vel[:, None, :] * np.ones((1, 5, 1))  # (T, N=5, 3), same signal per boid
    vel[..., 2] += rng.normal(scale=1e-6, size=vel.shape[:2])
    u, v = it.dominant_motion_plane(vel)
    normal = np.cross(u, v)
    # Normal should be close to +-Z; in-plane means normal has ~zero X/Y component.
    assert abs(abs(normal[2]) - 1.0) < 1e-3
    assert abs(normal[0]) < 1e-2 and abs(normal[1]) < 1e-2


def test_heading_angles_are_unwrapped_and_smooth_for_steady_rotation():
    t = np.linspace(0, 6 * np.pi, 400)
    vel = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)[:, None, :]
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    angles = it.heading_angles(vel, u, v)
    # A steady rotation's unwrapped angle should have no large jumps (unwrap did its job).
    assert np.max(np.abs(np.diff(angles[:, 0]))) < 0.2


def test_pairwise_delays_recovers_a_known_delay_with_correct_sign():
    # Bird 1 turns 10 time units AFTER bird 0 -> bird 1 does NOT turn "before" bird 0, so by
    # Attanasi et al.'s convention (tau_ij > 0 means j turns before i), tau[0, 1] < 0.
    t = np.arange(200, dtype=np.float64)
    angle_0 = 1.0 / (1.0 + np.exp(-(t - 80) / 5.0))
    angle_1 = 1.0 / (1.0 + np.exp(-(t - 90) / 5.0))
    angles = np.stack([angle_0, angle_1], axis=1)
    tau = it.pairwise_delays(angles, dt=1.0)
    assert tau[0, 1] < 0 and abs(tau[0, 1] - (-10.0)) < 1.5
    assert tau[1, 0] > 0 and abs(tau[1, 0] - 10.0) < 1.5
    assert np.allclose(tau + tau.T, 0.0)  # antisymmetric by construction


def test_turning_times_recovers_a_consistent_ordering():
    # A 4-bird chain with known pairwise delays consistent with true times [0, 5, 5, 15].
    true_t = np.array([0.0, 5.0, 5.0, 15.0])
    tau = true_t[:, None] - true_t[None, :]  # tau_ij = t_i - t_j, exactly the model's premise
    t = it.turning_times(tau)
    # Only relative ordering/spacing is guaranteed (t is defined up to a constant).
    t = t - t[0]
    assert np.allclose(t, true_t, atol=1e-8)


def test_dispersion_curve_matches_the_closed_form():
    t = np.array([3.0, 1.0, 2.0])
    t_sorted, x = it.dispersion_curve(t, density_value=1.0)
    assert np.allclose(t_sorted, [1.0, 2.0, 3.0])
    assert np.allclose(x, np.array([1, 2, 3]) ** (1.0 / 3.0))


def test_propagation_speed_recovers_a_perfect_linear_dispersion():
    t = np.linspace(0, 100, 200)
    x = 5.0 * t  # exactly linear, c_s = 5
    result = it.propagation_speed(t, x)
    assert abs(result["c_s"] - 5.0) < 1e-6
    assert result["r2_linear"] > 0.999
    assert result["favours"] == "linear"


def test_propagation_speed_favours_diffusive_for_sqrt_law_dispersion():
    t = np.linspace(1.0, 200.0, 300)  # start > 0 to avoid a degenerate sqrt(0) window edge
    x = 3.0 * np.sqrt(t)
    result = it.propagation_speed(t, x)
    assert result["r2_diffusive"] > result["r2_linear"]
    assert result["favours"] == "diffusive"


def test_speed_vs_polarisation_recovers_eq_10s_linear_relation():
    # Exact eq. 10 with a/sqrt(beta*chi) = 1: c_s = 1/sqrt(1-Phi).
    phis = {0.5: 1.0 / np.sqrt(0.5), 0.8: 1.0 / np.sqrt(0.2), 0.95: 1.0 / np.sqrt(0.05)}
    result = it.speed_vs_polarisation(phis)
    assert abs(result["slope"] - 1.0) < 1e-6
    assert abs(result["intercept"]) < 1e-6
    assert result["r2"] > 0.999


def test_information_transfer_report_end_to_end_on_synthetic_trajectory():
    # A small synthetic "flock" with a spatially-staggered, in-plane collective turn: boid i's
    # heading starts turning at a time proportional to its position along X (mimicking a
    # localized turn origin propagating outward), so the recovered dispersion curve should be
    # close to linear and c_s should be positive and finite.
    rng = np.random.default_rng(7)
    n, steps = 30, 150
    x_pos = np.linspace(0, 20, n)
    pos = np.stack([x_pos, np.zeros(n), np.zeros(n)], axis=-1)
    pos = np.broadcast_to(pos, (steps, n, 3)).copy()
    t = np.arange(steps, dtype=np.float64)
    onset = 20.0 + x_pos * 2.0  # later onset for boids farther along x -> linear propagation
    angle = np.stack(
        [1.0 / (1.0 + np.exp(-(t - onset[i]) / 3.0)) for i in range(n)], axis=1
    )
    vel = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    vel += rng.normal(scale=1e-3, size=vel.shape)

    result = it.information_transfer_report(pos, vel, dt=1.0)
    assert np.isfinite(result["c_s"])
    assert result["density"] > 0
    assert result["favours"] in ("linear", "diffusive")


def _record_trajectory(modifier, n=50, warmup=100, steps=100, seed=0, **kwargs):
    sim = m.Simulation(
        boid_count=n,
        mode="pearce",
        modifier=modifier,
        phi_p=0.50,
        phi_a=0.20,
        steric_enabled=True,
        steric=0.6,
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


def test_information_transfer_report_runs_on_a_real_simulation_trajectory():
    """Real end-to-end smoke test against a live Simulation (small N/short window, matching
    this suite's own fast-test convention) — not a check of *which* propagation law wins.

    Distinguishing linear-vs-diffusive propagation decisively needs a genuine, controlled
    turn stimulus applied mid-run (a sudden field/predator onset, mirroring the real
    predator-triggered turns Attanasi et al. analyze) — checked directly (see
    sci/param_table.md's information_transfer.py notes): under natural fluctuations alone,
    at this small a scale/window, both `spin_wave` and `instant_response` came back
    'diffusive', not the clean split the theory predicts. `murmur_py` doesn't yet expose a
    way to change a running Simulation's parameters (no `SetParam`/command-queue binding to
    Python, only `step`/`run_batch`) — wiring that up is the real prerequisite for a
    decisive live comparison, tracked as follow-up work, not attempted here.
    """
    for modifier in ("instant_response", "spin_wave"):
        pos_traj, vel_traj = _record_trajectory(modifier, coupling=1.0, drive=1.0, chi=2.0)
        result = it.information_transfer_report(pos_traj, vel_traj, dt=1.0)
        assert result["favours"] in ("linear", "diffusive", "insufficient_data")
        if result["favours"] != "insufficient_data":
            assert np.isfinite(result["c_s"])
            assert np.isfinite(result["r2_linear"]) and np.isfinite(result["r2_diffusive"])
        assert result["density"] > 0
