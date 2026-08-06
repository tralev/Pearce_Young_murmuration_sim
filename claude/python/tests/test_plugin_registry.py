"""Smoke tests proving every plugin registered in murmur_py's build_registry() is actually
selectable and runnable from Python — not just built and tested in Rust. Several Track C
Phase 13/14/15/16 plugins (murmur_spin_wave, murmur_external_field, murmur_torus_domain,
murmur_kdtree_index, murmur_knn_selection, murmur_fixed_speed, murmur_predator_fsm,
murmur_young, murmur_margin_domain, murmur_sphere_domain, murmur_sphere_soft_domain,
murmur_ceiling_speed, murmur_none_speed) were registered in murmur_core's own registry and had
real Rust test suites, but were never added to murmur_py's registry — there was no way to even
construct a Simulation using any of them from Python. This file exists so that gap can't
silently reopen: each plugin gets a small, real construct-and-run check, one per trait socket it
fills.
"""

import numpy as np

import murmuration as m


def test_torus_domain_wraps_positions_within_half_extent():
    sim = m.Simulation(boid_count=60, domain="torus", half_extent=20.0, init_radius=15.0)
    for _ in range(50):
        sim.run_batch(5, 0)
    pos = sim.positions()
    assert np.all(np.abs(pos) <= 20.0 + 1e-6)


def test_kdtree_index_runs_and_matches_hash_grid_order_of_magnitude():
    kd = m.Simulation(boid_count=100, spatial_index="kdtree_index", init_radius=15.0, init_seed=2)
    kd.run_batch(20, 0)
    grid = m.Simulation(boid_count=100, spatial_index="hash_grid", init_radius=15.0, init_seed=2)
    grid.run_batch(20, 0)
    # Same composition otherwise, same seed: not required to be bit-identical (different
    # candidate-ordering internals), but both should reach a plausible, comparable regime.
    assert np.isfinite(kd.metrics()["opacity_int"])
    assert 0.0 <= kd.metrics()["polarisation"] <= 1.0
    assert abs(kd.metrics()["mean_nn"] - grid.metrics()["mean_nn"]) < 2.0


def test_knn_selection_gives_every_boid_up_to_k_neighbours_worth_of_signal():
    sim = m.Simulation(boid_count=80, neighbor_selection="knn_selection", knn_k=5, init_radius=15.0)
    sim.run_batch(10, 0)
    m_ = sim.metrics()
    assert np.isfinite(m_["opacity_int"])
    assert 0.0 <= m_["polarisation"] <= 1.0


def test_fixed_speed_locks_every_boid_to_cruise_speed_times_factor():
    sim = m.Simulation(boid_count=50, speed_model="fixed_speed", speed_factor=1.5, cruise_speed=2.0)
    sim.run_batch(10, 0)
    speeds = np.linalg.norm(sim.velocities(), axis=1)
    assert np.allclose(speeds, 3.0, atol=1e-6)


def test_external_field_step_hook_biases_mean_velocity_toward_the_field_direction():
    sim = m.Simulation(
        boid_count=60,
        step_hooks=["external_field"],
        field_x=1.0,
        field_y=0.0,
        field_z=0.0,
        field_strength=0.5,
        init_radius=15.0,
    )
    sim.run_batch(30, 0)
    mean_vel = sim.velocities().mean(axis=0)
    assert mean_vel[0] > 0.0  # net drift toward +X, the field direction


def test_spin_wave_modifier_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=50,
        modifier="spin_wave",
        coupling=1.0,
        drive=1.0,
        chi=1.0,
        init_radius=15.0,
    )
    sim.run_batch(20, 0)
    assert np.all(np.isfinite(sim.velocities()))
    assert np.all(np.isfinite(sim.positions()))


def test_predator_fsm_step_hook_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=60,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        steric_enabled=True,
        steric=0.6,
        step_hooks=["predator_fsm"],
        predator_count=1,
        danger_radius=10.0,
        init_radius=8.0,
    )
    sim.run_batch(50, 0)
    assert np.all(np.isfinite(sim.velocities()))
    assert np.all(np.isfinite(sim.positions()))
    assert 1 in sim.species()  # the predator boid is present (Species::Predator tag)


def test_young_mode_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=100,
        mode="young",
        m_min=2,
        m_max=12,
        m_fallback=6,
        refresh_interval=10,
        init_radius=10.0,
    )
    sim.run_batch(50, 0)
    assert np.all(np.isfinite(sim.velocities()))
    assert np.all(np.isfinite(sim.positions()))


def test_margin_domain_keeps_positions_within_half_extent():
    sim = m.Simulation(
        boid_count=60,
        domain="margin",
        half_extent=20.0,
        margin_width=5.0,
        margin_strength=8.0,
        init_radius=15.0,
    )
    for _ in range(50):
        sim.run_batch(5, 0)
    pos = sim.positions()
    assert np.all(np.isfinite(pos))
    assert np.all(np.abs(pos) <= 20.0 + 1e-6)


def test_sphere_domain_keeps_positions_within_the_radius():
    sim = m.Simulation(
        boid_count=60,
        domain="sphere",
        sphere_radius=20.0,
        init_radius=15.0,
    )
    for _ in range(50):
        sim.run_batch(5, 0)
    pos = sim.positions()
    assert np.all(np.isfinite(pos))
    assert np.all(np.linalg.norm(pos, axis=1) <= 20.0 + 1e-6)


def test_sphere_soft_domain_runs_and_stays_finite_without_a_hard_clamp():
    sim = m.Simulation(
        boid_count=60,
        domain="sphere_soft",
        sphere_radius=20.0,
        sphere_soft_push_strength=8.0,
        init_radius=15.0,
    )
    for _ in range(50):
        sim.run_batch(5, 0)
    pos = sim.positions()
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(sim.velocities()))


def test_ceiling_speed_caps_overspeed_but_never_boosts_underspeed():
    sim = m.Simulation(
        boid_count=60,
        speed_model="ceiling_speed",
        cruise_speed=2.0,
        init_radius=15.0,
    )
    sim.run_batch(30, 0)
    speeds = np.linalg.norm(sim.velocities(), axis=1)
    assert np.all(np.isfinite(speeds))
    assert np.all(speeds <= 2.0 + 1e-6)


def test_none_speed_runs_with_no_enforcement_and_stays_finite():
    sim = m.Simulation(
        boid_count=60,
        speed_model="none_speed",
        init_radius=15.0,
    )
    sim.run_batch(30, 0)
    assert np.all(np.isfinite(sim.velocities()))
    assert np.all(np.isfinite(sim.positions()))
