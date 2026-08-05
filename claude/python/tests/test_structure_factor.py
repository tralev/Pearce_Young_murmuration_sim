"""Unit tests for murmuration.analysis.structure_factor against synthetic, known-answer
inputs (roadmap.md Phase 15) — same style as test_analysis.py."""

import numpy as np

import murmuration as m
from murmuration.analysis import structure_factor as sf


def test_structure_factor_of_uncorrelated_points_is_close_to_one():
    # A well-known exact result: for i.i.d. (Poisson-uncorrelated) points, E[S(q)] = 1 for
    # every q well above the box's own finite-size cutoff ~2*pi/L (below that, wavelengths
    # longer than the box itself see boundary artifacts, not a real physical divergence —
    # box half-width is 50 here, i.e. L=100, 2*pi/L~0.063, so q values start above that).
    rng = np.random.default_rng(0)
    pos = rng.uniform(-50, 50, size=(2000, 3))
    q_values = np.array([0.2, 0.4, 0.7, 1.0])
    s_q = sf.structure_factor(pos, q_values, n_directions=100, rng=rng)
    assert np.all(np.abs(s_q - 1.0) < 0.3), f"expected S(q)~1, got {s_q}"


def test_structure_factor_is_nonnegative_and_finite():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(300, 3)) * np.array([5.0, 1.0, 1.0])  # anisotropic cluster
    s_q = sf.structure_factor(pos, np.array([0.1, 1.0, 5.0]), n_directions=32, rng=rng)
    assert np.all(np.isfinite(s_q))
    assert np.all(s_q >= 0.0)


def test_small_q_scaling_recovers_a_known_power_law_divergence():
    q = np.geomspace(0.001, 0.1, 20)
    true_zeta = 0.7
    s_q = 3.0 * q ** (-true_zeta)  # exact power law, no noise
    result = sf.small_q_scaling(q, s_q)
    assert abs(result["zeta"] - true_zeta) < 1e-6
    assert result["diverges"] is True


def test_small_q_scaling_classifies_a_saturating_curve_as_not_diverging():
    q = np.geomspace(0.001, 0.1, 20)
    c_h = 5.0
    zeta_free = 0.7
    s_q = 1.0 / (q**zeta_free + c_h)  # eq. 26's field-suppressed form -> finite as q -> 0
    result = sf.small_q_scaling(q, s_q)
    assert result["diverges"] is False


def test_structure_factor_report_runs_on_a_real_simulation_snapshot():
    sim = m.Simulation(boid_count=200, mode="pearce", phi_p=0.50, phi_a=0.20, steric_enabled=True, steric=0.6)
    sim.run_batch(200, 0)
    pos = sim.positions()
    q_values = np.geomspace(0.05, 2.0, 12)
    s_q = sf.structure_factor(pos, q_values, n_directions=48, rng=np.random.default_rng(2))
    assert np.all(np.isfinite(s_q)) and np.all(s_q >= 0.0)
    result = sf.small_q_scaling(q_values[:4], s_q[:4])
    assert np.isfinite(result["zeta"])
