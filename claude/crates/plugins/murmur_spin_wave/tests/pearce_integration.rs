//! End-to-end proof that `SpinWaveResponse` composes with a real `FlockingMode` (Pearce) inside
//! the actual pipeline (roadmap.md Phase 13 exit gate: "each new plugin passes... a config-swap
//! test"). Critically, this exercises `respond()`'s `RwLock<HashMap<...>>` under genuine
//! concurrent load — `Simulation::step()`'s read phase calls `respond()` for every boid via
//! rayon, across however many threads the pool has — not just the single-threaded unit tests
//! in `src/lib.rs`. If the locking were wrong (e.g. a lock held across an unrelated call,
//! introducing a deadlock), this would hang or panic, not just produce a wrong number.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_spin_wave::register(&mut reg);
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
        .max_force(10.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("cell_size", 10.0)
        .with("radius", 25.0)
        .with("coupling", 0.5)
        .with("drive", 1.0)
        .with("chi", 1.0);
    let config = SimConfig {
        mode: "pearce".to_string(),
        modifier: "spin_wave".to_string(),
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
fn runs_under_real_parallel_read_phase_without_hanging_or_nan() {
    let mut sim = build_sim(300, 11);
    for step in 0..200 {
        sim.step(1.0, 11);
        for p in sim.positions() {
            assert!(p.is_finite(), "position went non-finite at step {step}");
        }
        for v in sim.velocities() {
            assert!(v.is_finite(), "velocity went non-finite at step {step}");
        }
    }
}

#[test]
fn state_hash_is_identical_across_1_4_and_8_rayon_threads() {
    // The determinism gate every other pipeline test enforces (roadmap.md Phase 5) — proves
    // the RwLock-based per-boid state doesn't introduce thread-count-dependent behaviour
    // (e.g. via HashMap iteration order affecting anything observable; it shouldn't, since
    // coupling_sum only ever sums over ctx.neighbors' own fixed order, never iterates the map).
    fn run_with_pool_size(threads: usize) -> u64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sim = build_sim(150, 7);
            sim.run_batch(30, 7);
            sim.state_hash()
        })
    }
    let h1 = run_with_pool_size(1);
    let h4 = run_with_pool_size(4);
    let h8 = run_with_pool_size(8);
    assert_eq!(h1, h4, "state_hash differs between 1 and 4 threads");
    assert_eq!(h1, h8, "state_hash differs between 1 and 8 threads");
}
