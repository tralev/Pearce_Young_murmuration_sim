//! `BoidColumns` — Struct-of-Arrays boid storage (design/01_core.md §2).
//!
//! Capacity is fixed at construction; `step()` never allocates. Indices are stable: index `i`
//! refers to the same boid until removed, via a bitvec `active` flag + O(1) free-list add/remove.

use crate::math::Vec3;
use bitvec::vec::BitVec;

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Species {
    #[default]
    Prey = 0,
    Predator = 1,
    Custom(u16),
}

pub struct BoidColumns {
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    pub species: Vec<Species>,
    pub seed: Vec<u64>,
    pub active: BitVec,
    // per-frame scratch (reused; never reallocated in step()):
    pub acc: Vec<Vec3>,
    pub theta: Vec<f64>,
    free: Vec<u32>,
}

impl BoidColumns {
    /// Allocates storage for exactly `capacity` boids, all initially inactive.
    pub fn with_capacity(capacity: u32) -> Self {
        let n = capacity as usize;
        BoidColumns {
            pos: vec![Vec3::ZERO; n],
            vel: vec![Vec3::ZERO; n],
            species: vec![Species::Prey; n],
            seed: vec![0; n],
            active: BitVec::repeat(false, n),
            acc: vec![Vec3::ZERO; n],
            theta: vec![0.0; n],
            // pop() yields the lowest free index first, so the first `capacity` adds hand out
            // ascending indices 0, 1, 2, ... — a convenience for tests/determinism, not a
            // correctness requirement.
            free: (0..capacity).rev().collect(),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.pos.len() as u32
    }

    pub fn active_count(&self) -> u32 {
        self.active.count_ones() as u32
    }

    /// Inserts a boid into a free slot, returning its stable index. `None` if at capacity —
    /// capacity is fixed at construction and `add` never grows the columns.
    pub fn add(&mut self, pos: Vec3, vel: Vec3, species: Species, seed: u64) -> Option<u32> {
        let i = self.free.pop()? as usize;
        self.pos[i] = pos;
        self.vel[i] = vel;
        self.species[i] = species;
        self.seed[i] = seed;
        self.acc[i] = Vec3::ZERO;
        self.theta[i] = 0.0;
        self.active.set(i, true);
        Some(i as u32)
    }

    /// Deactivates a boid, returning its slot to the free-list for reuse. Other boids' indices
    /// are unaffected. No-op if `index` is already inactive or out of range.
    pub fn remove(&mut self, index: u32) {
        let i = index as usize;
        if i >= self.active.len() || !self.active[i] {
            return;
        }
        self.active.set(i, false);
        self.free.push(index);
    }

    /// Stable indices of every active boid, ascending.
    pub fn iter_active(&self) -> impl Iterator<Item = u32> + '_ {
        self.active.iter_ones().map(|i| i as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_returns_ascending_indices_and_sets_active() {
        let mut b = BoidColumns::with_capacity(4);
        let i0 = b
            .add(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 10)
            .unwrap();
        let i1 = b
            .add(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 11)
            .unwrap();
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(b.active_count(), 2);
        assert!(b.active[i0 as usize]);
        assert!(b.active[i1 as usize]);
    }

    #[test]
    fn add_beyond_capacity_returns_none() {
        let mut b = BoidColumns::with_capacity(1);
        assert!(b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).is_some());
        assert!(b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 1).is_none());
        assert_eq!(b.capacity(), 1); // add() never grows storage
    }

    #[test]
    fn remove_keeps_other_indices_stable() {
        let mut b = BoidColumns::with_capacity(3);
        let i0 = b
            .add(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0)
            .unwrap();
        let i1 = b
            .add(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1)
            .unwrap();
        let i2 = b
            .add(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 2)
            .unwrap();

        b.remove(i1);
        assert!(!b.active[i1 as usize]);
        assert_eq!(b.active_count(), 2);
        // i0 and i2 untouched — same index, same data.
        assert_eq!(b.pos[i0 as usize], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(b.pos[i2 as usize], Vec3::new(3.0, 0.0, 0.0));
        assert!(b.active[i0 as usize]);
        assert!(b.active[i2 as usize]);
    }

    #[test]
    fn remove_then_add_reuses_the_freed_slot() {
        let mut b = BoidColumns::with_capacity(2);
        let i0 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        let i1 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 1).unwrap();
        b.remove(i0);
        let i2 = b
            .add(Vec3::new(9.0, 9.0, 9.0), Vec3::ZERO, Species::Predator, 42)
            .unwrap();
        assert_eq!(i2, i0); // slot reused
        assert!(b.active[i2 as usize]);
        assert_eq!(b.species[i2 as usize], Species::Predator);
        assert!(b.active[i1 as usize]); // untouched sibling
    }

    #[test]
    fn remove_is_a_noop_on_an_already_inactive_or_out_of_range_index() {
        let mut b = BoidColumns::with_capacity(2);
        let i0 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        b.remove(i0);
        b.remove(i0); // double-remove: no panic, no double-free of the slot
        assert_eq!(b.active_count(), 0);
        b.remove(999); // out of range: no panic
    }

    #[test]
    fn iter_active_yields_only_active_indices_ascending() {
        let mut b = BoidColumns::with_capacity(4);
        let i0 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        let _i1 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 1).unwrap();
        let i2 = b.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 2).unwrap();
        b.remove(1);
        let active: Vec<u32> = b.iter_active().collect();
        assert_eq!(active, vec![i0, i2]);
    }

    #[test]
    fn species_custom_variant_carries_a_distinct_tag() {
        let a = Species::Custom(7);
        let b = Species::Custom(8);
        assert_ne!(a, b);
        assert_ne!(Species::Custom(0), Species::Prey);
    }
}
