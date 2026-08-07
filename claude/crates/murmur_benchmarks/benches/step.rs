//! `Simulation::step()` throughput under the real default composition — `PearceProjection` +
//! `InstantResponse` + `OpenSpace` + `HashGrid` + `RadiusGather` + `BandSpeed` +
//! `SphereVolume`/`UniformSphere`, the same registration/param set
//! `murmur_pearce/tests/slice_integration.rs` already proves scientifically correct. This
//! bench only asks "how fast," never "is it right" — correctness stays that integration test's
//! job, not this crate's.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn build_sim(n: u32) -> Simulation {
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
        .with("sigma", 4.0)
        .with("body_radius", 0.5)
        .with("max_candidates", 20.0)
        .with("cell_size", 10.0)
        .with("radius", (n as f64).cbrt() * 5.0); // keep density roughly constant across N
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
        init_seed: 7,
        step_hooks: Vec::new(),
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_step");
    for &n in &[100u32, 1_000, 5_000] {
        let mut sim = build_sim(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sim.step(1.0, 1));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
