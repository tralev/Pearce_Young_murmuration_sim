//! End-to-end `murmur_angle` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `desired()` checks. The load-bearing one here is the
//! thread-count-determinism proof: `src/lib.rs`'s own module doc argues `Angle`'s per-boid
//! `HashMap` needs no `spin_wave`-style double buffering, since each boid only ever touches its
//! own key. This test checks that argument empirically rather than trusting it — the same
//! discipline that caught a real thread-count-dependent bug in `murmur_spin_wave` (G4) via
//! exactly this class of test, not a hypothetical concern.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_angle::register(&mut reg);
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
        .vision_radius(15.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("cell_size", 15.0)
        .with("radius", 10.0)
        .with("max_turn_rate", 1.0);
    let config = SimConfig {
        mode: "angle".to_string(),
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
    let mut sim = build_sim(200, 7);
    sim.run_batch(200, 7);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn boid_speeds_stay_at_cruise_speed_since_angle_only_changes_direction() {
    let mut sim = build_sim(100, 3);
    sim.run_batch(100, 3);
    for v in sim.velocities() {
        assert!(
            (v.len() - 1.0).abs() < 1e-6,
            "Angle only rotates heading; BandSpeed should keep speed pinned near cruise_speed, got {}",
            v.len()
        );
    }
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
