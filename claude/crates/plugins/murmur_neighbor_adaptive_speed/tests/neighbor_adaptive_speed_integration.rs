//! End-to-end `murmur_neighbor_adaptive_speed` integration test, run through a real
//! `Simulation`, not just `src/lib.rs`'s unit-level `post_steer`/`speed_cap_multiplier` checks.
//! The load-bearing one here mirrors `murmur_boid_state_machine`'s own proof, confirming this
//! plugin's continuous multiplier reaches real boids through the same G1->G3 chain: a
//! densely-packed real flock ends up measurably slower than a sparse one.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_neighbor_adaptive_speed::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, init_radius: f64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(15.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("phi_p", 0.50)
        .with("phi_a", 0.20)
        .with("cell_size", 15.0)
        .with("radius", init_radius)
        .with("low_count", 1.0)
        .with("high_count", 10.0)
        .with("min_speed_factor", 0.4)
        .with("max_speed_factor", 1.0);
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
        step_hooks: vec!["neighbor_adaptive_speed".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 9, 15.0);
    sim.run_batch(100, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_densely_packed_flock_ends_up_measurably_slower_than_a_sparse_one() {
    // Same boid count, same seed, same everything except the initial spawn radius: a tight
    // packing puts most boids' neighbour counts up near high_count=10 under vision_radius=15,
    // a loose one keeps them down near low_count=1. If the continuous multiplier genuinely
    // reaches real boids through the same G1->G3 chain boid_state_machine proved, the tight
    // flock's mean speed should end up measurably below the loose flock's.
    let mut tight = build_sim(200, 3, 8.0);
    tight.run_batch(60, 3);
    let tight_mean_speed: f64 =
        tight.velocities().iter().map(|v| v.len()).sum::<f64>() / tight.velocities().len() as f64;

    let mut loose = build_sim(200, 3, 80.0);
    loose.run_batch(60, 3);
    let loose_mean_speed: f64 =
        loose.velocities().iter().map(|v| v.len()).sum::<f64>() / loose.velocities().len() as f64;

    assert!(
        tight_mean_speed < loose_mean_speed - 0.05,
        "expected the densely-packed flock's mean speed ({}) to be measurably below the sparse \
         flock's ({}) -- the continuous speed cap doesn't appear to be reaching real boids",
        tight_mean_speed,
        loose_mean_speed
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
            let mut sim = build_sim(300, 2024, 10.0);
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
