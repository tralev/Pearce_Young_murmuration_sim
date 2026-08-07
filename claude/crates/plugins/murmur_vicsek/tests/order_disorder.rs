//! Vicsek order-disorder transition (roadmap.md Phase 8 exit gate): polarisation should rise
//! as angular noise (diffusion) decreases — the classic Vicsek 1995 sanity check, run here as
//! an independent proof that the `FlockingMode` seam genuinely generalizes beyond Pearce (not
//! just a copy of Phase 5's gate).

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_vicsek::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
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
        .max_force(10.0) // large: let InstantResponse reach the desired heading in ~1 step,
        // matching Vicsek's own "instantly adopt the blended heading" update rule
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(5.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("couplage", 1.0)
        .with("diffusion", diffusion)
        .with("cell_size", 5.0)
        // Vicsek has no cohesion term (unlike Pearce) — in open, unbounded space the only
        // thing keeping the neighbour graph connected long enough to reach *global* (not just
        // per-cluster) consensus is starting dense enough that vision_radius covers most of
        // the flock; a sparser pack fragments into locally-ordered, globally-uncorrelated
        // clusters before noise-driven convergence can finish.
        .with("radius", 4.0);
    let config = SimConfig {
        mode: "vicsek".to_string(),
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
fn polarisation_rises_as_angular_noise_decreases() {
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
        "low-noise Vicsek should reach a clearly ordered state, got {alpha_low_noise}"
    );
}

#[test]
fn zero_diffusion_settles_to_near_full_order() {
    let mut sim = build_sim(300, 0.0, 7);
    sim.run_batch(300, 7);
    let alpha = sim.metrics().polarisation;
    assert!(
        alpha > 0.9,
        "expected near-total order with zero noise, got {alpha}"
    );
}

#[test]
fn no_nan_over_a_long_run() {
    let mut sim = build_sim(200, 1.0, 3);
    for _ in 0..500 {
        sim.step(1.0, 3);
        for p in sim.positions() {
            assert!(p.is_finite());
        }
        for v in sim.velocities() {
            assert!(v.is_finite());
        }
    }
}
