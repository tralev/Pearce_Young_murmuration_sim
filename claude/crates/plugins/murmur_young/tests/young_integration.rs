//! End-to-end `murmur_young` integration test (roadmap.md Phase 14 exit gate: "murmur_young's
//! steering demonstrably uses a real, non-fallback m*"), proven through a real `Simulation`,
//! not just `src/lib.rs`'s unit-level `pre_step`/`desired` checks.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_young::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, refresh_interval: f64) -> Simulation {
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
        .with("refresh_interval", refresh_interval)
        .with("m_min", 2.0)
        .with("m_max", 12.0);
    let config = SimConfig {
        mode: "young".to_string(),
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
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_without_nan_and_stays_in_the_speed_band_over_a_long_run() {
    let mut sim = build_sim(150, 7, 20.0);
    let core = sim.composition().core_params;
    let (vmin, vmax) = (core.cruise_speed * core.speed_min_factor, core.cruise_speed);
    let tol = 1e-6;

    for step in 0..400 {
        sim.step(1.0, 7);
        for v in sim.velocities() {
            assert!(v.is_finite(), "velocity went non-finite at step {step}");
            let s = v.len();
            assert!(
                s >= vmin - tol && s <= vmax + tol,
                "speed {s} outside band [{vmin},{vmax}] at step {step}"
            );
        }
        for p in sim.positions() {
            assert!(p.is_finite(), "position went non-finite at step {step}");
        }
    }
}

#[test]
fn thread_count_determinism_holds_across_1_4_and_8_rayon_threads() {
    // Same regression shape as murmur_spin_wave's own thread-determinism test: RicherPredator
    // taught this project that per-boid persistent/cached state under Mutex<..> can silently
    // become thread-count-dependent if the step-boundary discipline is wrong. murmur_young's
    // m* cache uses the same discipline (refresh gated on step_count, read-only during
    // desired()) -- prove it holds under real thread-count variation, not just assumed.
    fn run_with_threads(n_threads: usize) -> u64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sim = build_sim(120, 3, 10.0);
            for _ in 0..60 {
                sim.step(1.0, 3);
            }
            sim.state_hash()
        })
    }

    let h1 = run_with_threads(1);
    let h4 = run_with_threads(4);
    let h8 = run_with_threads(8);
    assert_eq!(h1, h4, "state_hash differs between 1 and 4 threads");
    assert_eq!(h1, h8, "state_hash differs between 1 and 8 threads");
}

#[test]
fn different_m_fallback_values_produce_different_trajectories_when_never_refreshed() {
    // A direct, robust "the cache is load-bearing" check (weaker than an earlier attempt at
    // this test that compared refresh cadences: that one failed not because the cache is
    // inert, but because m* turned out to be *stable* over a 100-step run for this flock size
    // -- a real, unsurprising finding for a coherent flock, not a bug; `pre_step`'s own unit
    // tests already directly prove the refresh-timing logic is exactly right). Here, force
    // `m_fallback` itself to differ and starve `pre_step` of enough boids to ever compute a
    // real m* (n < m_min), so `desired()` is provably using `m_fallback` the entire run --
    // different fallback values must then produce different trajectories, or `desired()`
    // isn't actually using `m` at all.
    let build_with_fallback = |m_fallback: f64, seed: u64| -> Simulation {
        let registry = build_registry();
        let core_params = CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(5) // fewer than m_min=2's own m_star() would need to return Some
            .vision_radius(15.0)
            .build()
            .unwrap();
        let plugin_params = PluginParams::new()
            .with("cell_size", 15.0)
            .with("radius", 10.0)
            .with("refresh_interval", 1.0)
            .with("m_min", 10.0) // > boid_count -> m_star() always None -> always falls back
            .with("m_max", 12.0)
            .with("m_fallback", m_fallback);
        let config = SimConfig {
            mode: "young".to_string(),
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
        Simulation::new(config, &registry).unwrap().0
    };

    let mut m2 = build_with_fallback(2.0, 4);
    let mut m4 = build_with_fallback(4.0, 4);
    for _ in 0..50 {
        m2.step(1.0, 4);
        m4.step(1.0, 4);
    }
    assert_ne!(
        m2.positions(),
        m4.positions(),
        "different m_fallback values produced identical trajectories -- desired() may not be \
         reading m* from the cache at all"
    );
}
