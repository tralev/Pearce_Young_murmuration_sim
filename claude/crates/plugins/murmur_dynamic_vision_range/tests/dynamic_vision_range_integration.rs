//! End-to-end `murmur_dynamic_vision_range` integration test, run through a real `Simulation`,
//! not just `src/lib.rs`'s unit-level `pre_step` checks. The load-bearing one here is proving
//! `core_params.vision_radius` genuinely mutates over a real run — the first plugin in this
//! project to exercise `SimView::core_params`'s mutable half, not just its read-only `boids`
//! half.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_dynamic_vision_range::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, init_radius: f64, target_neighbor_count: f64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .dt(1.0)
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("phi_p", 0.50)
        .with("phi_a", 0.20)
        .with("cell_size", 10.0)
        .with("radius", init_radius)
        .with("target_neighbor_count", target_neighbor_count)
        .with("adapt_rate", 0.3);
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
        step_hooks: vec!["dynamic_vision_range".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 9, 30.0, 6.5);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
    assert!(sim.composition().core_params.vision_radius.is_finite());
    assert!(sim.composition().core_params.vision_radius > 0.0);
}

#[test]
fn vision_radius_shrinks_for_a_densely_packed_flock_with_a_low_target() {
    // 200 boids packed into a small init_radius (5.0, << the initial vision_radius=10.0) with a
    // deliberately low target_neighbor_count -- the flock starts far denser than the target
    // wants, so the radius should shrink measurably below its initial 10.0 over real steps.
    let mut sim = build_sim(200, 3, 5.0, 1.0);
    let initial = sim.composition().core_params.vision_radius;
    sim.run_batch(30, 3);
    let after = sim.composition().core_params.vision_radius;
    assert!(after.is_finite() && after > 0.0);
    assert!(
        after < initial,
        "expected vision_radius to shrink from a densely-packed flock's low target: initial={} after={}",
        initial, after
    );
}

#[test]
fn vision_radius_grows_for_a_sparse_flock_with_a_high_target() {
    // 40 boids scattered across a large init_radius (200.0, >> the initial vision_radius=10.0)
    // with a deliberately high target_neighbor_count -- the flock starts far sparser than the
    // target wants, so the radius should grow measurably above its initial 10.0.
    let mut sim = build_sim(40, 5, 200.0, 30.0);
    let initial = sim.composition().core_params.vision_radius;
    sim.run_batch(30, 5);
    let after = sim.composition().core_params.vision_radius;
    assert!(after.is_finite() && after > 0.0);
    assert!(
        after > initial,
        "expected vision_radius to grow from a sparse flock's high target: initial={} after={}",
        initial,
        after
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
            let mut sim = build_sim(300, 2024, 20.0, 6.5);
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
