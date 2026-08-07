//! `HashGrid` rebuild/query cost at increasing `N` — a real baseline for Tier 4's own "300k-boid
//! headless-batch milestone" concern about incremental rebuild cost, measured here against the
//! current always-full-rebuild implementation before any incremental version exists to compare
//! against.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use murmur_core::{BoidColumns, SpatialIndex, Species, Vec3};
use murmur_hash_grid::HashGrid;

fn boids_with(n: u32, spread: f64) -> BoidColumns {
    let mut b = BoidColumns::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let p = Vec3::new(
            ((t * 12.9898).sin() * 43758.5453).fract() * spread,
            ((t * 78.233).sin() * 12345.6789).fract() * spread,
            ((t * 37.719).sin() * 98765.4321).fract() * spread,
        );
        b.add(p, Vec3::ZERO, Species::Prey, i as u64);
    }
    b
}

fn bench_rebuild(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_grid_rebuild");
    for &n in &[1_000u32, 10_000, 50_000] {
        let spread = (n as f64).cbrt() * 5.0;
        let boids = boids_with(n, spread);
        let mut grid = HashGrid::new(10.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| grid.rebuild(&boids));
        });
    }
    group.finish();
}

fn bench_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_grid_candidates");
    for &n in &[1_000u32, 10_000, 50_000] {
        let spread = (n as f64).cbrt() * 5.0;
        let boids = boids_with(n, spread);
        let mut grid = HashGrid::new(10.0);
        grid.rebuild(&boids);
        let mut out = Vec::new();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                out.clear();
                grid.candidates(Vec3::ZERO, 10.0, &mut out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rebuild, bench_candidates);
criterion_main!(benches);
