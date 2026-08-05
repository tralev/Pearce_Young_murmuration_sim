//! Vicsek order-disorder under `torus` (roadmap.md Phase 13 exit gate): `order_disorder.rs`'s
//! open-space version needed an artificially dense initial pack (`radius: 4.0` against
//! `vision_radius: 5.0`) purely to keep the neighbour graph connected long enough to reach
//! *global* consensus — open space has no cohesion term and no boundary, so a normal-density
//! sparse pack fragments into locally-ordered, globally-uncorrelated clusters first. A periodic
//! domain removes that need: this is Vicsek et al. 1995's own setup (a periodic box), not an
//! open-space adaptation of it. Same assertions as `order_disorder.rs`, but with a normal
//! placement density (`radius: 15.0` against the same `vision_radius: 5.0` — matching the
//! ratio other slice tests use, e.g. `slice_integration.rs`'s 25.0-vs-10.0), no density tuning.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_vicsek::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_torus_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32, diffusion: f64, seed: u64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(10.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(5.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("couplage", 1.0)
        .with("diffusion", diffusion)
        .with("cell_size", 5.0)
        // Normal placement density (no cohesion-workaround tuning, unlike order_disorder.rs's
        // open-space version) — a periodic box keeps the neighbour graph globally connected on
        // its own via wrap-around, not via artificial initial density.
        .with("radius", 15.0)
        // Box comfortably larger than the initial placement, so periodicity is reachable
        // (boids do wrap during the run) without dominating from step 0.
        .with("half_extent", 20.0);
    let config = SimConfig {
        mode: "vicsek".to_string(),
        modifier: "instant_response".to_string(),
        domain: "torus".to_string(),
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
fn polarisation_rises_as_angular_noise_decreases_without_density_tuning() {
    let steps = 300;
    let n = 400;

    let mut low_noise = build_sim(n, 0.05, 42);
    low_noise.run_batch(steps, 42);
    let alpha_low_noise = low_noise.metrics().polarisation;

    let mut high_noise = build_sim(n, 3.0, 42);
    high_noise.run_batch(steps, 42);
    let alpha_high_noise = high_noise.metrics().polarisation;

    assert!(
        alpha_low_noise > alpha_high_noise,
        "expected low-noise polarisation ({alpha_low_noise}) > high-noise polarisation \
         ({alpha_high_noise})"
    );
    assert!(
        alpha_low_noise > 0.5,
        "low-noise Vicsek under torus should reach a clearly ordered state, got {alpha_low_noise}"
    );
}

#[test]
fn zero_diffusion_settles_to_near_full_order() {
    let mut sim = build_sim(300, 0.0, 7);
    sim.run_batch(300, 7);
    let alpha = sim.metrics().polarisation;
    assert!(
        alpha > 0.9,
        "expected near-total order with zero noise under torus, got {alpha}"
    );
}

#[test]
fn no_nan_over_a_long_run_including_wrap_events() {
    let mut sim = build_sim(200, 1.0, 3);
    for _ in 0..500 {
        sim.step(1.0, 3);
        for p in sim.positions() {
            assert!(p.is_finite());
            // Positions must stay within the box — a wrap that failed to actually clamp back
            // in range would let a boid escape to infinity over enough steps.
            assert!(
                p.x.abs() <= 20.0 + 1e-6 && p.y.abs() <= 20.0 + 1e-6 && p.z.abs() <= 20.0 + 1e-6
            );
        }
        for v in sim.velocities() {
            assert!(v.is_finite());
        }
    }
}
