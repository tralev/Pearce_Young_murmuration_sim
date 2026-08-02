//! `SpatialIndex` — neighbour-query acceleration structure trait (design/01_core.md §5). The
//! trait is core; `HashGrid` (the first plugin) and later `KDTree`/`adaptive_index` all
//! implement it with equal status.

use crate::boids::BoidColumns;
use crate::math::Vec3;

pub trait SpatialIndex: Send + Sync {
    /// Rebuilds the index from current boid positions. Typically O(N); called once per step
    /// before the read phase.
    fn rebuild(&mut self, boids: &BoidColumns);

    /// Appends candidate boid indices within (approximately) radius `r` of `p` into `out`
    /// (may include boids just outside `r` — the caller applies the exact distance test).
    /// `out` is caller-owned scratch, not cleared by this call.
    fn candidates(&self, p: Vec3, r: f64, out: &mut Vec<u32>);

    /// k nearest, no distance cutoff — for topological/k-NN `NeighborSelection` strategies.
    /// Default falls back to an expanding-radius retry over `candidates()`; concrete indices
    /// should override with something genuinely efficient (an expanding ring search for
    /// `HashGrid`, a real k-NN query for a k-d tree).
    fn candidates_knn(&self, p: Vec3, k: u32, out: &mut Vec<u32>) {
        let mut r = 1.0_f64;
        loop {
            out.clear();
            self.candidates(p, r, out);
            if out.len() as u32 >= k || r > 1e6 {
                break;
            }
            r *= 2.0;
        }
    }

    fn name(&self) -> &'static str;
}
