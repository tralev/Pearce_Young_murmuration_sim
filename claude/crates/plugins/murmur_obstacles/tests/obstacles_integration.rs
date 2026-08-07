//! End-to-end `murmur_obstacles` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `post_steer` checks. The load-bearing one here is the exact claim
//! `roadmap.md`'s own Phase 18 row already promises: a real flock demonstrably avoids a placed
//! SDF primitive.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_obstacles::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("phi_p", 0.03)
        .with("phi_a", 0.80)
        .with("cell_size", 10.0)
        .with("radius", 8.0) // init sphere straddles the obstacle sphere below
        .with("obstacle_kind", 0.0) // sphere
        .with("obstacle_center_x", 0.0)
        .with("obstacle_center_y", 0.0)
        .with("obstacle_center_z", 0.0)
        .with("obstacle_radius", 6.0)
        .with("obstacle_avoidance_radius", 12.0)
        .with("obstacle_push_strength", 8.0)
        .with("obstacle_min_gap", 0.2);
    let config = SimConfig {
        mode: "pearce".to_string(),
        modifier: "instant_response".to_string(),
        domain: "open".to_string(),
        spatial_index: "hash_grid".to_string(),
        neighbor_selection: "radius_gather".to_string(),
        speed_model: "band".to_string(),
        init: "sphere_volume".to_string(),
        noise: "uniform_sphere".to_string(),
        core_params,
        plugin_params,
        init_seed: seed,
        step_hooks: vec!["obstacles".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 9);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_flock_spawned_straddling_a_placed_sphere_demonstrably_moves_out_of_it() {
    // init_radius=8 with an obstacle sphere of radius=6 at the origin: many boids start inside
    // or very near the obstacle. If avoidance genuinely works, the number of boids still
    // colliding (position inside the sphere) should measurably drop over a real run.
    let mut sim = build_sim(200, 11);

    let initial_positions = sim.positions();
    let obstacle_radius = 6.0_f64;
    let initially_inside = initial_positions
        .iter()
        .filter(|p| p.len() < obstacle_radius)
        .count();
    assert!(
        initially_inside > 0,
        "test setup sanity check: expected at least some boids to start inside the obstacle"
    );

    sim.run_batch(150, 11);

    let final_positions = sim.positions();
    assert!(final_positions.iter().all(|p| p.is_finite()));
    let finally_inside = final_positions
        .iter()
        .filter(|p| p.len() < obstacle_radius)
        .count();

    assert!(
        finally_inside < initially_inside,
        "expected avoidance to measurably reduce the number of boids inside the obstacle: \
         initially_inside={} finally_inside={}",
        initially_inside,
        finally_inside
    );
}

#[test]
fn a_flock_spawned_clear_of_the_obstacle_stays_clear() {
    // A flock spawned well outside the avoidance radius should never be pushed into the
    // obstacle -- a sanity check that the force only ever pushes away, never toward.
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(150)
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("phi_p", 0.03)
        .with("phi_a", 0.80)
        .with("cell_size", 10.0)
        .with("radius", 5.0)
        .with("obstacle_kind", 0.0)
        .with("obstacle_center_x", 100.0)
        .with("obstacle_center_y", 0.0)
        .with("obstacle_center_z", 0.0)
        .with("obstacle_radius", 6.0)
        .with("obstacle_avoidance_radius", 10.0);
    let config = SimConfig {
        mode: "pearce".to_string(),
        modifier: "instant_response".to_string(),
        domain: "open".to_string(),
        spatial_index: "hash_grid".to_string(),
        neighbor_selection: "radius_gather".to_string(),
        speed_model: "band".to_string(),
        init: "sphere_volume".to_string(),
        noise: "uniform_sphere".to_string(),
        core_params,
        plugin_params,
        init_seed: 1,
        step_hooks: vec!["obstacles".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    let (mut sim, _warnings) = Simulation::new(config, &registry).unwrap();
    sim.run_batch(100, 1);
    let positions = sim.positions();
    let obstacle_center = murmur_core::Vec3::new(100.0, 0.0, 0.0);
    assert!(
        positions.iter().all(|p| (*p - obstacle_center).len() > 6.0),
        "no boid should ever end up inside a distant obstacle it never approached"
    );
}

#[test]
fn state_hash_is_identical_across_1_4_and_8_rayon_threads() {
    fn run_with_pool_size(threads: usize) -> u64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sim = build_sim(300, 2024);
            sim.run_batch(50, 2024);
            sim.state_hash()
        })
    }
    let h1 = run_with_pool_size(1);
    let h4 = run_with_pool_size(4);
    let h8 = run_with_pool_size(8);
    assert_eq!(h1, h4, "state_hash differs between 1 and 4 threads");
    assert_eq!(h1, h8, "state_hash differs between 1 and 8 threads");
}
