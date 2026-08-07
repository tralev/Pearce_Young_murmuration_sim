"""Smoke tests proving every plugin registered in murmur_py's build_registry() is actually
selectable and runnable from Python — not just built and tested in Rust. Several Track C
Phase 13/14/15/16 plugins (murmur_spin_wave, murmur_external_field, murmur_torus_domain,
murmur_kdtree_index, murmur_knn_selection, murmur_fixed_speed, murmur_predator_fsm,
murmur_young, murmur_margin_domain, murmur_sphere_domain, murmur_sphere_soft_domain,
murmur_ceiling_speed, murmur_none_speed, murmur_adaptive_index, murmur_hybrid_selection,
murmur_spatial, murmur_angle, murmur_influencer, murmur_maxent_social, murmur_field,
murmur_boid_state_machine, murmur_ecology, murmur_dynamic_vision_range,
murmur_neighbor_adaptive_speed, murmur_speed_noise, murmur_wander, murmur_ripple,
murmur_obstacles) were registered in murmur_core's own registry and had real Rust test
suites, but were never added to
murmur_py's registry — there was no way to even construct a Simulation using any of them from
Python. This file exists so that gap can't silently reopen: each plugin gets a small, real
construct-and-run check, one per trait socket it fills. The five new `murmur_initializers`
variants (gaussian/grid/blob/tangential/spawn_cube) were added directly to an already-registered
plugin crate, so they don't have this exact gap — but get the same treatment here anyway, for
the same reason: proven reachable from Python, not just built and tested in Rust.
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


def test_gaussian_initializer_produces_finite_state():
    sim = m.Simulation(boid_count=60, init="gaussian", std_dev=5.0)
    sim.run_batch(20, 0)
    assert np.all(np.isfinite(sim.positions()))
    assert np.all(np.isfinite(sim.velocities()))


def test_grid_initializer_places_boids_on_a_lattice_at_step_zero():
    sim = m.Simulation(boid_count=27, init="grid", grid_spacing=2.0)
    pos = sim.positions()
    coords = np.round(pos / 2.0)
    assert np.all(np.abs(pos / 2.0 - coords) < 1e-6)
    sim.run_batch(10, 0)
    assert np.all(np.isfinite(sim.positions()))


def test_blob_initializer_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=100,
        init="blob",
        blob_count=4,
        blob_spread=15.0,
        blob_radius=1.0,
    )
    sim.run_batch(20, 0)
    assert np.all(np.isfinite(sim.positions()))
    assert np.all(np.isfinite(sim.velocities()))


def test_tangential_initializer_starts_with_purely_tangential_velocity():
    sim = m.Simulation(boid_count=60, init="tangential", init_radius=5.0)
    pos = sim.positions()
    vel = sim.velocities()
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    radial_component = np.sum(radial * vel, axis=1)
    assert np.all(np.abs(radial_component) < 1e-6)


def test_spawn_cube_initializer_produces_finite_state():
    sim = m.Simulation(boid_count=60, init="spawn_cube", spawn_size=4.0)
    pos = sim.positions()
    assert np.all(np.abs(pos) <= 4.0 + 1e-6)
    sim.run_batch(20, 0)
    assert np.all(np.isfinite(sim.positions()))


def test_adaptive_index_runs_below_and_at_the_crossover():
    below = m.Simulation(
        boid_count=60,
        spatial_index="adaptive_index",
        adaptive_crossover=5000,
        init_radius=15.0,
    )
    below.run_batch(20, 0)
    assert np.all(np.isfinite(below.positions()))

    at_crossover = m.Simulation(
        boid_count=60,
        spatial_index="adaptive_index",
        adaptive_crossover=60,
        init_radius=15.0,
    )
    at_crossover.run_batch(20, 0)
    assert np.all(np.isfinite(at_crossover.positions()))
    assert np.all(np.isfinite(at_crossover.velocities()))


def test_hybrid_selection_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=80,
        neighbor_selection="hybrid_selection",
        knn_k=5,
        fov_enabled=True,
        fov_half_angle=1.2,
        init_radius=15.0,
    )
    sim.run_batch(20, 0)
    m_ = sim.metrics()
    assert np.isfinite(m_["opacity_int"])
    assert 0.0 <= m_["polarisation"] <= 1.0


def test_spatial_mode_runs_and_produces_finite_state():
    sim = m.Simulation(
        boid_count=80,
        mode="spatial",
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        separation_radius=2.0,
        init_radius=15.0,
    )
    sim.run_batch(30, 0)
    assert np.all(np.isfinite(sim.positions()))
    assert np.all(np.isfinite(sim.velocities()))
    m_ = sim.metrics()
    assert 0.0 <= m_["polarisation"] <= 1.0


def test_angle_mode_runs_and_keeps_speeds_pinned_to_cruise_speed():
    sim = m.Simulation(
        boid_count=80,
        mode="angle",
        max_turn_rate=1.0,
        cruise_speed=1.5,
        init_radius=15.0,
    )
    sim.run_batch(30, 0)
    assert np.all(np.isfinite(sim.positions()))
    speeds = np.linalg.norm(sim.velocities(), axis=1)
    assert np.allclose(speeds, 1.5, atol=1e-3)


def test_influencer_mode_runs_and_a_different_dt_changes_the_trajectory():
    slow = m.Simulation(boid_count=60, mode="influencer", dt=0.01, init_radius=15.0)
    slow.run_batch(60, 0)
    fast = m.Simulation(boid_count=60, mode="influencer", dt=1.0, init_radius=15.0)
    fast.run_batch(60, 0)
    assert np.all(np.isfinite(slow.positions()))
    assert np.all(np.isfinite(fast.positions()))
    assert not np.allclose(slow.positions(), fast.positions())


def test_maxent_social_mode_keeps_the_flock_bounded_under_a_margin_domain():
    sim = m.Simulation(
        boid_count=100,
        mode="maxent_social",
        domain="margin",
        half_extent=20.0,
        margin_width=5.0,
        boundary_weight=3.0,
        boundary_decay=5.0,
        repulsion_radius=2.0,
        attraction_radius=8.0,
        init_radius=15.0,
    )
    sim.run_batch(200, 0)
    assert np.all(np.isfinite(sim.positions()))
    assert np.all(np.isfinite(sim.velocities()))
    r_max = np.linalg.norm(sim.positions(), axis=1).max()
    assert r_max < 40.0, f"flock escaped the bounded domain, R_max={r_max}"


def test_field_mode_runs_and_a_different_dt_changes_the_trajectory():
    slow = m.Simulation(
        boid_count=60, mode="field", dt=0.01, anchor_count=3, anchor_spread=20.0, init_radius=5.0
    )
    slow.run_batch(60, 0)
    fast = m.Simulation(
        boid_count=60, mode="field", dt=1.0, anchor_count=3, anchor_spread=20.0, init_radius=5.0
    )
    fast.run_batch(60, 0)
    assert np.all(np.isfinite(slow.positions()))
    assert np.all(np.isfinite(fast.positions()))
    assert not np.allclose(slow.positions(), fast.positions())


def test_boid_state_machine_step_hook_caps_speed_of_a_densely_packed_flock():
    sim = m.Simulation(
        boid_count=150,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        steric_enabled=True,
        steric=0.6,
        step_hooks=["boid_state_machine"],
        isolated_max=1,
        crowded_min=8,
        crowded_speed_cap=0.5,
        vision_radius=15.0,
        init_radius=8.0,
        cruise_speed=1.0,
    )
    sim.run_batch(60, 0)
    assert np.all(np.isfinite(sim.positions()))
    speeds = np.linalg.norm(sim.velocities(), axis=1)
    assert np.all(np.isfinite(speeds))
    # A tightly-packed flock (init_radius=8 << vision_radius=15) should have at least some
    # boids capped below cruise_speed by the Crowded classification.
    assert speeds.min() < 1.0 - 1e-6


def test_ecology_step_hook_coherence_gate_tightens_flock_spread_when_open():
    def r_max(sim):
        return float(np.linalg.norm(sim.positions(), axis=1).max())

    common = dict(
        boid_count=150,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        step_hooks=["ecology"],
        dusk_hour=18.0,
        season_start_day=0,
        season_end_day=364,
        vision_radius=10.0,
        init_radius=8.0,
        cruise_speed=1.0,
    )
    gate_closed = m.Simulation(hours_per_dt=0.0, coherence_strength=2.0, **common)
    gate_closed.run_batch(80, 4)
    gate_open = m.Simulation(hours_per_dt=1.0, coherence_strength=2.0, **common)
    gate_open.run_batch(80, 4)
    assert np.all(np.isfinite(gate_closed.positions()))
    assert np.all(np.isfinite(gate_open.positions()))
    assert r_max(gate_open) < r_max(gate_closed)


def test_dynamic_vision_range_step_hook_shrinks_radius_for_a_densely_packed_flock():
    sim = m.Simulation(
        boid_count=200,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        step_hooks=["dynamic_vision_range"],
        vision_radius=10.0,
        init_radius=5.0,
        target_neighbor_count=1.0,
        adapt_rate=0.3,
        cruise_speed=1.0,
    )
    initial = sim.describe()["vision_radius"]
    sim.run_batch(30, 3)
    after = sim.describe()["vision_radius"]
    assert np.isfinite(after) and after > 0.0
    assert after < initial


def test_neighbor_adaptive_speed_step_hook_slows_a_densely_packed_flock():
    def mean_speed(sim):
        return float(np.linalg.norm(sim.velocities(), axis=1).mean())

    common = dict(
        boid_count=200,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        step_hooks=["neighbor_adaptive_speed"],
        vision_radius=15.0,
        low_count=1.0,
        high_count=10.0,
        min_speed_factor=0.4,
        max_speed_factor=1.0,
        cruise_speed=1.0,
    )
    tight = m.Simulation(init_radius=8.0, **common)
    tight.run_batch(60, 3)
    loose = m.Simulation(init_radius=80.0, **common)
    loose.run_batch(60, 3)
    assert np.all(np.isfinite(tight.velocities()))
    assert np.all(np.isfinite(loose.velocities()))
    assert mean_speed(tight) < mean_speed(loose) - 0.05


def test_speed_noise_step_hook_is_deterministic_and_lowers_mean_speed():
    def mean_speed(sim):
        return float(np.linalg.norm(sim.velocities(), axis=1).mean())

    common = dict(
        boid_count=150,
        mode="pearce",
        step_hooks=["speed_noise"],
        smoothing=1.0,
        cruise_speed=1.0,
    )
    a = m.Simulation(noise_amplitude=0.5, **common)
    a.run_batch(50, 42)
    b = m.Simulation(noise_amplitude=0.5, **common)
    b.run_batch(50, 42)
    assert a.state_hash() == b.state_hash(), "same base_seed must give a bit-identical run"

    c = m.Simulation(noise_amplitude=0.5, **common)
    c.run_batch(50, 43)
    assert a.state_hash() != c.state_hash(), "a different base_seed must give a different run"

    noisy = m.Simulation(noise_amplitude=0.5, **common)
    noisy.run_batch(60, 3)
    quiet = m.Simulation(noise_amplitude=0.0, **common)
    quiet.run_batch(60, 3)
    assert np.all(np.isfinite(noisy.velocities()))
    assert mean_speed(noisy) < mean_speed(quiet) - 0.02


def test_wander_step_hook_keeps_a_pearce_flock_bounded_around_its_own_moving_target():
    sim = m.Simulation(
        boid_count=150,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        step_hooks=["wander"],
        wander_pull_strength=2.0,
        vision_radius=10.0,
        init_radius=10.0,
        cruise_speed=1.0,
        dt=1.0,
    )
    sim.run_batch(200, 7)
    positions = sim.positions()
    assert np.all(np.isfinite(positions))
    r_max = float(np.linalg.norm(positions, axis=1).max())
    assert r_max < 200.0, f"expected a strong wander pull to keep R_max bounded, got {r_max}"


def test_ripple_step_hook_lowers_mean_speed_when_a_ring_passes_through_the_flock():
    def mean_speed(sim):
        return float(np.linalg.norm(sim.velocities(), axis=1).mean())

    common = dict(
        boid_count=200,
        mode="pearce",
        phi_p=0.03,
        phi_a=0.80,
        step_hooks=["ripple"],
        init_radius=3.0,
        ripple_period=100.0,
        ripple_speed=1.0,
        ripple_width=2.0,
        ripple_min_cap=0.05,
        cruise_speed=1.0,
        dt=1.0,
    )
    rippling = m.Simulation(ripple_amplitude=1.0, **common)
    rippling.step(1.0, 5)
    quiet = m.Simulation(ripple_amplitude=1e-9, **common)
    quiet.step(1.0, 5)
    assert np.all(np.isfinite(rippling.velocities()))
    assert mean_speed(rippling) < mean_speed(quiet) - 0.02


def test_obstacles_step_hook_demonstrably_pushes_a_flock_out_of_a_placed_sphere():
    sim = m.Simulation(
        boid_count=200,
        mode="pearce",
        phi_p=0.03,
        phi_a=0.80,
        step_hooks=["obstacles"],
        init_radius=8.0,
        obstacle_kind=0.0,
        obstacle_radius=6.0,
        obstacle_avoidance_radius=12.0,
        obstacle_push_strength=8.0,
        obstacle_min_gap=0.2,
        vision_radius=10.0,
    )
    initial = np.linalg.norm(sim.positions(), axis=1)
    initially_inside = int(np.sum(initial < 6.0))
    assert initially_inside > 0, "test setup sanity check"

    sim.run_batch(150, 11)

    final = sim.positions()
    assert np.all(np.isfinite(final))
    finally_inside = int(np.sum(np.linalg.norm(final, axis=1) < 6.0))
    assert finally_inside < initially_inside
