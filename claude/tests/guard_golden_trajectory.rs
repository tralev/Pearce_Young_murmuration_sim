//! Pinned state-hash regression — the "golden trajectory" guard (roadmap.md Phase 2b).
//!
//! Full `Simulation`-level golden-trajectory regression needs `step()`, which lands in Phase
//! 5; that phase extends this test to real simulation trajectories. For now this guard proves
//! the *mechanism* — a deterministic construction hashed and compared against a pinned golden
//! value in `fixtures/golden/` — against `murmur_core`'s already-real deterministic
//! primitives (`rng::for_boid` + `BoidColumns`), so the hashing/pinning plumbing exists and is
//! exercised before Phase 5 has anything richer to hash.

mod support;

use murmur_core::{rng, BoidColumns, Species, Vec3};
use rand_core::RngCore;

/// FNV-1a — simple, dependency-free, good enough for a regression fingerprint (not a
/// cryptographic hash).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn state_hash(boids: &BoidColumns) -> u64 {
    let mut bytes = Vec::new();
    for i in boids.iter_active() {
        for v in [boids.pos[i as usize], boids.vel[i as usize]] {
            bytes.extend_from_slice(&v.x.to_bits().to_le_bytes());
            bytes.extend_from_slice(&v.y.to_bits().to_le_bytes());
            bytes.extend_from_slice(&v.z.to_bits().to_le_bytes());
        }
    }
    fnv1a(&bytes)
}

/// A fixed, deterministic "trivial fixture composition" (roadmap.md Phase 2b's exit-gate
/// wording): 5 boids placed from `rng::for_boid` draws under a fixed base seed. Not a real
/// `Simulation` (no plugins involved) — see module doc for why.
fn build_fixture_state() -> BoidColumns {
    let mut boids = BoidColumns::with_capacity(5);
    for i in 0..5u32 {
        let mut r = rng::for_boid(1234, i as u64, 0);
        let pos = Vec3::new(
            (r.next_u64() % 1000) as f64 / 100.0,
            (r.next_u64() % 1000) as f64 / 100.0,
            (r.next_u64() % 1000) as f64 / 100.0,
        );
        boids.add(pos, Vec3::new(1.0, 0.0, 0.0), Species::Prey, i as u64);
    }
    boids
}

/// Pinned when this guard was written (from `build_fixture_state()` + `state_hash()` as they
/// stand today), stored as plain hex text in `fixtures/golden/`. If this legitimately needs to
/// change — the RNG mixing function changes, the hash function changes — regenerate
/// deliberately and explain why in the commit; an unexplained change here is exactly what this
/// guard exists to catch.
fn golden_path() -> std::path::PathBuf {
    support::workspace_root().join("fixtures/golden/fixture_construction.hash")
}

#[test]
fn fixture_construction_state_hash_matches_golden() {
    let boids = build_fixture_state();
    let hash = state_hash(&boids);

    let path = golden_path();
    let expected_text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read golden fixture {}: {e}", path.display()));
    let expected = u64::from_str_radix(expected_text.trim(), 16)
        .unwrap_or_else(|e| panic!("golden fixture {} is not valid hex: {e}", path.display()));

    assert_eq!(
        hash,
        expected,
        "fixture construction state_hash changed from the pinned golden value in {} — if \
         intentional, regenerate the pin deliberately and explain why in the commit \
         (guard_golden_trajectory)",
        path.display()
    );
}

#[test]
fn state_hash_is_deterministic_across_calls() {
    let a = state_hash(&build_fixture_state());
    let b = state_hash(&build_fixture_state());
    assert_eq!(a, b);
}
