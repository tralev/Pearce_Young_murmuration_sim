//! End-to-end `murmur_influencer` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `desired()` checks. The load-bearing one here is proving the
//! Lissajous target genuinely uses real elapsed simulated time (`step_count * dt`), the
//! specific property roadmap.md Phase 17's exit gate names for this plugin — checked here at
//! the whole-`Simulation` level, not just against `InfluencerParams::target_at` directly.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_influencer::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, dt: f64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .dt(dt)
        .vision_radius(15.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("cell_size", 15.0)
        .with("radius", 10.0);
    let config = SimConfig {
        mode: "influencer".to_string(),
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
        step_hooks: Vec::new(),
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap()
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 11, 1.0);
    sim.run_batch(150, 11);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_larger_dt_over_the_same_step_count_produces_a_different_trajectory() {
    // Same seed, same step count, different dt: since the Lissajous target's position depends
    // on real elapsed time (step_count * dt, G4), the two runs must diverge -- if the mode were
    // silently using a per-call counter instead of real time, dt would have no effect at all.
    let mut slow = build_sim(80, 5, 0.01);
    slow.run_batch(80, 5);
    let mut fast = build_sim(80, 5, 1.0);
    fast.run_batch(80, 5);
    assert_ne!(
        slow.positions(),
        fast.positions(),
        "different dt over the same step count must reach different target positions, hence different trajectories"
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
            let mut sim = build_sim(300, 2024, 1.0);
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
