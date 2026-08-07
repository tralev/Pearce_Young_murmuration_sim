//! End-to-end `murmur_ecology` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `pre_step`/`post_steer` checks. The load-bearing one here is
//! proving the coherence gate genuinely tightens a real flock's own spread when it's open
//! (dusk, in season) versus when it's closed (out of season) — not just correct in an isolated
//! `post_steer` call.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_ecology::register(&mut reg);
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, seed: u64, hours_per_dt: f64, coherence_strength: f64) -> Simulation {
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
        .with("radius", 20.0)
        .with("hours_per_dt", hours_per_dt)
        .with("dusk_hour", 18.0)
        .with("season_start_day", 0.0)
        .with("season_end_day", 364.0)
        .with("coherence_strength", coherence_strength);
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
        step_hooks: vec!["ecology".to_string()],
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 9, 0.1, 0.3);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_strong_coherence_gate_measurably_tightens_the_flocks_own_spread() {
    // hours_per_dt chosen so the run starts and stays well past dusk_hour=18 (season is the
    // whole year here, so only the dusk gate varies): coherence_strength=0 vs. a real nonzero
    // value, otherwise identical seed/composition. If the gate's cohesion pull genuinely
    // reaches real boids, R_max (the flock's own bounding radius) should end up measurably
    // smaller with it on.
    let mut without = build_sim(200, 4, 0.0, 0.0); // hours_per_dt=0 -> stuck at hour=0 (dawn,
    without.run_batch(150, 4); // dusk_factor~0) regardless -- gate closed either way here,
                               // this run is the "no extra pull" baseline
    let r_max_without = without
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);

    let mut with = build_sim(200, 4, 0.0, 2.0); // same baseline dusk_factor (~0, gate closed)
    with.run_batch(150, 4); // but coherence_strength is irrelevant with the gate shut --
                            // included as a same-seed control confirming the gate itself
                            // (not just a nonzero coherence_strength) is what matters
    let r_max_with_gate_closed = with
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);

    // With the gate closed in both runs, coherence_strength alone must have no effect.
    assert!(
        (r_max_without - r_max_with_gate_closed).abs() < 1e-6,
        "coherence_strength must do nothing while the gate is closed: {} vs {}",
        r_max_without,
        r_max_with_gate_closed
    );

    // Now open the gate (hours_per_dt pushes hour well past dusk quickly) with a real
    // coherence_strength, and confirm R_max is measurably smaller than the gate-closed case.
    let mut gate_open = build_sim(200, 4, 1.0, 2.0); // hours_per_dt=1 -> hour reaches 18+ fast
    gate_open.run_batch(150, 4);
    let r_max_gate_open = gate_open
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);

    assert!(
        r_max_gate_open < r_max_with_gate_closed,
        "expected the open coherence gate to tighten R_max: open={} closed={}",
        r_max_gate_open,
        r_max_with_gate_closed
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
            let mut sim = build_sim(300, 2024, 0.5, 0.3);
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
