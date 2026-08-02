//! Per-boid deterministic RNG (design/01_core.md §10).
//!
//! The sole randomness source is `base_seed` on `step()`; each boid's RNG is
//! `hash(base_seed, boid.seed, step_count)` (SplitMix64), so the parallel read phase is
//! independent of thread count and call order — a pure function of its three inputs.

use crate::math::Vec3;
pub use rand_chacha::ChaCha8Rng as Rng;
use rand_core::{RngCore, SeedableRng};

/// One SplitMix64 step (Vigna's public-domain construction) — deterministic seed mixing.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic per-boid RNG for a given `(base_seed, boid_seed, step_count)` triple. A pure
/// function of its inputs — safe to call from any thread, in any order, inside the parallel
/// read phase.
pub fn for_boid(base_seed: u64, boid_seed: u64, step_count: u64) -> Rng {
    let mut h = splitmix64(base_seed);
    h = splitmix64(h ^ boid_seed);
    h = splitmix64(h ^ step_count);
    Rng::seed_from_u64(h)
}

/// Uniform f64 in `[0, 1)` from the top 53 bits of a `u64` draw (full `f64` mantissa
/// precision, standard technique). Exposed for plugin authors needing a plain uniform draw
/// (e.g. an `Initializer`'s cube-root volume sampling or per-axis box sampling).
#[inline]
pub fn uniform01(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Area-uniform random unit vector on S² (`z = 1 − 2u` avoids pole clustering) — design/01_core.md
/// §12.1. Shared by `UniformSphere` noise, `SphereVolume`/`SphereShell` initializers, and the
/// Band `SpeedModel`'s unstall reseed.
pub fn sample_unit_sphere(rng: &mut Rng) -> Vec3 {
    let z = 1.0 - 2.0 * uniform01(rng);
    let t = std::f64::consts::TAU * uniform01(rng);
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * t.cos(), r * t.sin(), z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw4(mut rng: Rng) -> [u64; 4] {
        [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ]
    }

    #[test]
    fn reproducible_for_same_inputs() {
        let a = draw4(for_boid(42, 7, 100));
        let b = draw4(for_boid(42, 7, 100));
        assert_eq!(a, b);
    }

    #[test]
    fn order_independent() {
        // A pure function of its inputs: interleaving calls for different boids/steps in any
        // order must not perturb any individual boid's sequence (there is no shared mutable
        // RNG state between boids).
        let direct = draw4(for_boid(1, 2, 3));
        let _ = draw4(for_boid(1, 99, 3)); // simulate another boid's draw happening "between"
        let _ = draw4(for_boid(1, 2, 4)); // and another step's draw
        let interleaved = draw4(for_boid(1, 2, 3));
        assert_eq!(direct, interleaved);
    }

    #[test]
    fn sample_unit_sphere_produces_unit_vectors() {
        let mut rng = for_boid(1, 2, 3);
        for _ in 0..200 {
            let v = sample_unit_sphere(&mut rng);
            assert!(v.is_finite());
            assert!(
                (v.len() - 1.0).abs() < 1e-9,
                "expected unit length, got {}",
                v.len()
            );
        }
    }

    #[test]
    fn sample_unit_sphere_is_reproducible_and_varied() {
        let a = sample_unit_sphere(&mut for_boid(9, 9, 9));
        let b = sample_unit_sphere(&mut for_boid(9, 9, 9));
        assert_eq!(a, b);

        let mut rng = for_boid(9, 9, 9);
        let first = sample_unit_sphere(&mut rng);
        let second = sample_unit_sphere(&mut rng);
        assert_ne!(first, second, "successive draws should differ");
    }

    #[test]
    fn distinct_inputs_give_distinct_streams() {
        let by_seed = draw4(for_boid(1, 2, 3));
        let by_boid = draw4(for_boid(1, 3, 3));
        let by_step = draw4(for_boid(1, 2, 4));
        let by_base = draw4(for_boid(2, 2, 3));
        assert_ne!(by_seed, by_boid);
        assert_ne!(by_seed, by_step);
        assert_ne!(by_seed, by_base);
    }
}
