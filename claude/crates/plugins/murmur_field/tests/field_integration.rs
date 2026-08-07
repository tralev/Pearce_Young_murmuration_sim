//! End-to-end `murmur_field` integration test, run through a real `Simulation`, not just
//! `src/lib.rs`'s unit-level `desired()` checks. The load-bearing one here is proving the
//! multi-anchor "field" claim itself: a real flock, given multiple anchors, ends up split
//! across more than one of them -- not all collapsing onto a single point, which would make
//! this indistinguishable from `murmur_influencer`'s single shared target.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_field::register(&mut reg);
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
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("cell_size", 10.0)
        .with("radius", 5.0) // initial sphere-volume spawn radius, near the origin
        .with("anchor_count", 3.0)
        .with("anchor_spread", 30.0)
        .with("amplitude_x", 2.0) // small oscillation relative to anchor_spread, so anchors
        .with("amplitude_y", 2.0) // stay well separated from each other throughout the run
        .with("amplitude_z", 1.0)
        .with("repulsion_radius", 2.0);
    let config = SimConfig {
        mode: "field".to_string(),
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
fn runs_for_real_and_stays_finite() {
    let mut sim = build_sim(150, 13, 1.0);
    sim.run_batch(150, 13);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn a_larger_dt_over_the_same_step_count_produces_a_different_trajectory() {
    let mut slow = build_sim(80, 5, 0.01);
    slow.run_batch(80, 5);
    let mut fast = build_sim(80, 5, 1.0);
    fast.run_batch(80, 5);
    assert_ne!(
        slow.positions(),
        fast.positions(),
        "different dt over the same step count must reach different anchor positions, hence different trajectories"
    );
}

#[test]
fn the_flock_splits_across_more_than_one_anchor_not_all_onto_a_single_point() {
    // 3 anchors, spread 30 apart, boids all spawned near the origin. If this plugin behaved
    // like murmur_influencer (one shared target), every boid would end up near the SAME
    // anchor. The real "field" claim is that they split across at least 2 of the 3 -- assessed
    // here by simple nearest-anchor-centre clustering (anchor motion amplitude, 2.0, is small
    // relative to anchor_spread=30.0, so "nearest anchor centre" is a stable, meaningful label
    // even mid-run).
    let mut sim = build_sim(300, 42, 1.0);
    sim.run_batch(400, 42);

    let anchor_spread = 30.0_f64;
    let centres: Vec<(f64, f64)> = (0..3)
        .map(|k| {
            let angle = k as f64 * (2.0 * std::f64::consts::PI / 3.0);
            (anchor_spread * angle.cos(), anchor_spread * angle.sin())
        })
        .collect();

    let mut counts = [0usize; 3];
    for p in sim.positions() {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, &(cx, cy)) in centres.iter().enumerate() {
            let d = ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        counts[best] += 1;
    }

    let anchors_with_boids = counts.iter().filter(|&&c| c > 0).count();
    assert!(
        anchors_with_boids >= 2,
        "expected the flock split across at least 2 of the 3 anchors, got counts={:?}",
        counts
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
