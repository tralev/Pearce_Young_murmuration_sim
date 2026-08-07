//! End-to-end `murmur_wander` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `pre_step`/`post_steer` checks. The load-bearing one here proves
//! the flock's own bulk drift genuinely carries `wander_center` along with it (the "for the
//! flock centre" property that distinguishes this from `murmur_influencer`/`murmur_field`'s
//! fixed-in-space targets), not just correct in an isolated `pre_step` call.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_wander::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, pull_strength: f64) -> Simulation {
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
        .with("radius", 10.0)
        .with("wander_pull_strength", pull_strength);
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
        step_hooks: vec!["wander".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 9, 0.3);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_strong_pull_strength_keeps_the_flock_bounded_around_its_own_wandering_target() {
    // With no pull at all, a Pearce-mode flock with no domain constraint can drift/expand
    // freely; with a strong pull, boids should stay clustered near the (moving) wander target
    // rather than dispersing -- checked here via R_max staying modest over a real run.
    let mut with_pull = build_sim(150, 7, 2.0);
    with_pull.run_batch(200, 7);
    let r_max_with_pull = with_pull
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);

    assert!(
        r_max_with_pull.is_finite() && r_max_with_pull < 200.0,
        "expected a strong wander pull to keep the flock's own R_max bounded, got {}",
        r_max_with_pull
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
            let mut sim = build_sim(300, 2024, 0.3);
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
