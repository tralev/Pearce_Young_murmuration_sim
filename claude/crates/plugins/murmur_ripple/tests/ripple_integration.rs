//! End-to-end `murmur_ripple` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `pre_step`/`post_steer` checks. The load-bearing one here proves a
//! real, testable consequence of the "wave of deceleration" claim: a boid sitting near the
//! flock's own centroid (where every train's ring passes through at t=0) measurably slows down
//! at the very first step, compared to the same boid under an otherwise-identical run with
//! ripple's own effect strength at zero.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_ripple::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, amplitude: f64) -> Simulation {
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
        .with("phi_p", 0.03)
        .with("phi_a", 0.80)
        .with("cell_size", 10.0)
        .with("radius", 3.0) // small init radius -- most boids start near the centroid, right
        // where every train's own ring sits at t=0
        .with("ripple_period", 100.0)
        .with("ripple_speed", 1.0)
        .with("ripple_width", 2.0)
        .with("ripple_amplitude", amplitude)
        .with("ripple_min_cap", 0.05);
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
        init_seed: 4,
        step_hooks: vec!["ripple".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap()
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 0.5);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_ring_passing_through_the_flock_measurably_lowers_mean_speed_at_the_first_step() {
    let mut rippling = build_sim(200, 1.0);
    rippling.step(1.0, 5);
    let rippling_mean_speed: f64 = rippling.velocities().iter().map(|v| v.len()).sum::<f64>()
        / rippling.velocities().len() as f64;

    let mut quiet = build_sim(200, 1e-9); // effectively no ripple effect (amplitude must be > 0)
    quiet.step(1.0, 5);
    let quiet_mean_speed: f64 =
        quiet.velocities().iter().map(|v| v.len()).sum::<f64>() / quiet.velocities().len() as f64;

    assert!(
        rippling_mean_speed < quiet_mean_speed - 0.02,
        "expected a strong ring at t=0 (right at the centroid, where the flock is packed) to \
         measurably lower mean speed relative to an amplitude~0 control: rippling={} quiet={}",
        rippling_mean_speed,
        quiet_mean_speed
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
            let mut sim = build_sim(300, 0.5);
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
