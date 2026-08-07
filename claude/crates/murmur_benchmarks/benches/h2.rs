//! `murmur_core::h2`'s dense-solver cost at increasing `N` — the current baseline any future
//! sparse partial eigensolver (Tier 4's own named follow-up, blocking H₂ at 300k boids) would
//! need to beat. `N` stays modest here (the eigensolve is `O(N³)`, `h2.rs`'s own module doc) —
//! large enough to see the growth curve, small enough that the whole suite still runs in a
//! reasonable time.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use murmur_core::{h2_at_m, m_star, Vec3};

fn sample_cloud(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Vec3::new(t.sin() * 5.0, t.cos() * 5.0, (t * 0.7).sin() * 3.0)
        })
        .collect()
}

fn bench_h2_at_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("h2_at_m_fixed_m6");
    for &n in &[100usize, 200, 400] {
        let pos = sample_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| h2_at_m(&pos, 6));
        });
    }
    group.finish();
}

fn bench_m_star_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("m_star_full_sweep_2_to_12");
    for &n in &[100usize, 200, 400] {
        let pos = sample_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| m_star(&pos, 2..=12));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_h2_at_m, bench_m_star_sweep);
criterion_main!(benches);
