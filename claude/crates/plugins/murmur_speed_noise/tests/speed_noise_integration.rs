//! End-to-end `murmur_speed_noise` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `post_steer` checks. The load-bearing ones here mirror G8's own
//! `murmur_core::pipeline` proof but through a real plugin composed into a real batch: the same
//! `base_seed` gives bit-identical trajectories, a different `base_seed` gives a genuinely
//! different one, and a strong noise amplitude measurably pulls a flock's mean speed below
//! `cruise_speed`.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_speed_noise::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, noise_amplitude: f64) -> Simulation {
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
        .with("radius", 10.0)
        .with("noise_amplitude", noise_amplitude)
        .with("smoothing", 1.0); // pure per-step noise, no smoothing lag, for a sharp proof
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
        init_seed: 5,
        step_hooks: vec!["speed_noise".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap()
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 0.2);
    sim.run_batch(100, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn the_same_base_seed_gives_a_bit_identical_trajectory() {
    let mut sim_a = build_sim(100, 0.2);
    sim_a.run_batch(50, 42);
    let mut sim_b = build_sim(100, 0.2);
    sim_b.run_batch(50, 42);
    assert_eq!(sim_a.state_hash(), sim_b.state_hash());
}

#[test]
fn a_different_base_seed_gives_a_genuinely_different_trajectory() {
    let mut sim_a = build_sim(100, 0.2);
    sim_a.run_batch(50, 42);
    let mut sim_b = build_sim(100, 0.2);
    sim_b.run_batch(50, 43);
    assert_ne!(sim_a.state_hash(), sim_b.state_hash());
}

#[test]
fn a_strong_noise_amplitude_measurably_pulls_mean_speed_below_cruise_speed() {
    let mut noisy = build_sim(200, 0.5);
    noisy.run_batch(60, 3);
    let noisy_mean_speed: f64 =
        noisy.velocities().iter().map(|v| v.len()).sum::<f64>() / noisy.velocities().len() as f64;

    let mut quiet = build_sim(200, 0.0);
    quiet.run_batch(60, 3);
    let quiet_mean_speed: f64 =
        quiet.velocities().iter().map(|v| v.len()).sum::<f64>() / quiet.velocities().len() as f64;

    assert!(
        noisy_mean_speed < quiet_mean_speed - 0.02,
        "expected a strong downward-only noise amplitude to measurably lower mean speed \
         relative to noise_amplitude=0: noisy={} quiet={}",
        noisy_mean_speed,
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
            let mut sim = build_sim(300, 0.2);
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
