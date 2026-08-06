//! End-to-end `murmur_maxent_social` integration test, run through a real `Simulation`, not
//! just `src/lib.rs`'s unit-level `desired()` checks. The load-bearing one here is proving the
//! boundary channel (G5, `Domain::boundary_distance()`) actually keeps a real flock roughly
//! within a bounded `Domain` over many steps, end to end — not just correct in an isolated
//! `desired()` call against a stub domain.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_maxent_social::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_margin_domain::register(&mut reg);
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
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("cell_size", 10.0)
        .with("radius", 15.0) // initial sphere-volume spawn radius
        .with("half_extent", 20.0) // Margin domain half-width -- smaller than the spawn radius
        .with("margin_width", 5.0)
        .with("repulsion_radius", 2.0)
        .with("attraction_radius", 8.0)
        .with("boundary_weight", 3.0)
        .with("boundary_decay", 5.0)
        .with("desire_weight", 0.0);
    let config = SimConfig {
        mode: "maxent_social".to_string(),
        modifier: "instant_response".to_string(),
        domain: "margin".to_string(),
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
    let mut sim = build_sim(150, 9);
    sim.run_batch(150, 9);
    let positions = sim.positions();
    let velocities = sim.velocities();
    assert!(positions.iter().all(|v| v.is_finite()));
    assert!(velocities.iter().all(|v| v.is_finite()));
}

#[test]
fn the_boundary_channel_pulls_the_flock_back_toward_a_bounded_domain_over_many_steps() {
    // Spawned within a radius-15 sphere, well outside the margin domain's half_extent=20 cube on
    // the diagonal (a sphere of radius 15 already pokes past a cube of half-width 20 on some
    // axes combinations) -- but the real test is over many steps: with boundary_weight=3.0 and
    // Margin's own hard clamp also active, positions must not drift further and further out;
    // R_max (the flock's own bounding radius) should stay bounded, not grow without limit.
    let mut sim = build_sim(200, 21);
    sim.run_batch(30, 21);
    let r_max_early = sim
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);
    sim.run_batch(300, 21);
    let r_max_late = sim
        .positions()
        .iter()
        .map(|p| p.len())
        .fold(0.0_f64, f64::max);
    // Margin's own half_extent=20 hard clamp already guarantees boundedness on its own -- the
    // real property under test is that R_max doesn't run away, which it would if the boundary
    // channel's direction/sign were wrong (e.g. accidentally pointing outward).
    assert!(
        r_max_late < r_max_early + 15.0,
        "R_max grew from {} to {} -- boundary channel does not appear to be containing the flock",
        r_max_early,
        r_max_late
    );
    assert!(r_max_late.is_finite());
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
