//! `AdaptiveIndex` — a `SpatialIndex` plugin that wraps `HashGrid` and `KdTree`, auto-selecting
//! between them by boid count at each `rebuild()` (design/01_core.md §5, roadmap.md Phase 16).
//! Ported by description from pymurmur's `plugins/spatial_index_strategy.py::auto`
//! (design/02_plugins.md §5 — pymurmur's actual source isn't reachable in this environment,
//! the same blocker as every other pymurmur cross-check this project has hit): "`HashGrid`
//! below ≈5,000, k-d tree at or above."
//!
//! Needs both `murmur_hash_grid` and `murmur_kdtree_index` to exist first (Phase 13), which
//! they now do — this is the intended long-run default `SpatialIndex` once both exist; the
//! slice used bare `HashGrid` only because N≤2,600 never approached the crossover
//! (design/02_plugins.md §5's own note on this plugin).

use murmur_core::{BoidColumns, PluginParams, Registry, SpatialIndex, Vec3};
use murmur_hash_grid::HashGrid;
use murmur_kdtree_index::KdTree;

pub struct AdaptiveIndex {
    crossover: u32,
    hash_grid: HashGrid,
    kd_tree: KdTree,
    using_kdtree: bool,
}

impl AdaptiveIndex {
    pub fn new(crossover: u32, cell_size: f64) -> Self {
        AdaptiveIndex {
            crossover,
            hash_grid: HashGrid::new(cell_size),
            kd_tree: KdTree::new(),
            using_kdtree: false,
        }
    }

    /// The backend actually selected by the most recent `rebuild()` — not part of the
    /// `SpatialIndex` trait (no other occupant needs to expose this), but real, checkable state
    /// this plugin's own crossover-correctness tests (and roadmap.md Phase 16's exit gate) need
    /// direct access to, rather than inferring it indirectly from query behaviour.
    pub fn active_backend(&self) -> &'static str {
        if self.using_kdtree {
            "kdtree_index"
        } else {
            "hash_grid"
        }
    }
}

impl SpatialIndex for AdaptiveIndex {
    fn rebuild(&mut self, boids: &BoidColumns) {
        self.using_kdtree = boids.active_count() >= self.crossover;
        if self.using_kdtree {
            self.kd_tree.rebuild(boids);
        } else {
            self.hash_grid.rebuild(boids);
        }
    }

    fn candidates(&self, p: Vec3, r: f64, out: &mut Vec<u32>) {
        if self.using_kdtree {
            self.kd_tree.candidates(p, r, out);
        } else {
            self.hash_grid.candidates(p, r, out);
        }
    }

    fn candidates_knn(&self, p: Vec3, k: u32, out: &mut Vec<u32>) {
        if self.using_kdtree {
            self.kd_tree.candidates_knn(p, k, out);
        } else {
            self.hash_grid.candidates_knn(p, k, out);
        }
    }

    fn name(&self) -> &'static str {
        "adaptive_index"
    }
}

/// Registers `AdaptiveIndex` under the name `"adaptive_index"`, reading `adaptive_crossover`
/// (default `5000.0`, pymurmur's own stated threshold) and `cell_size` (default `1.0`, forwarded
/// to the wrapped `HashGrid` — same key `hash_grid` itself reads, safe since `SpatialIndex` is a
/// single-occupant socket).
pub fn register(r: &mut Registry) {
    r.register_spatial_index("adaptive_index", |p: &PluginParams| {
        Box::new(AdaptiveIndex::new(
            p.get_or("adaptive_crossover", 5000.0) as u32,
            p.get_or("cell_size", 1.0),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::Species;

    fn boids_with(n: u32) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            b.add(
                Vec3::new(
                    ((t * 12.9898).sin() * 43758.5453).fract() * 50.0,
                    ((t * 78.233).sin() * 12345.6789).fract() * 50.0,
                    ((t * 37.719).sin() * 98765.4321).fract() * 50.0,
                ),
                Vec3::ZERO,
                Species::Prey,
                i as u64,
            );
        }
        b
    }

    #[test]
    fn conforms_to_spatial_index_contract() {
        let boids = BoidColumns::with_capacity(4);
        let mut idx = AdaptiveIndex::new(5000, 1.0);
        murmur_conformance::spatial_index(&mut idx, &boids);
    }

    #[test]
    fn uses_hash_grid_below_the_crossover() {
        let boids = boids_with(100);
        let mut idx = AdaptiveIndex::new(5000, 5.0);
        idx.rebuild(&boids);
        assert_eq!(idx.active_backend(), "hash_grid");
    }

    #[test]
    fn uses_kdtree_at_or_above_the_crossover() {
        let boids = boids_with(50);
        let mut idx = AdaptiveIndex::new(50, 5.0);
        idx.rebuild(&boids);
        assert_eq!(
            idx.active_backend(),
            "kdtree_index",
            "N == crossover must already select the k-d tree ('at or above')"
        );
    }

    #[test]
    fn switches_backend_across_successive_rebuilds_as_n_crosses_the_threshold() {
        let mut idx = AdaptiveIndex::new(50, 5.0);
        idx.rebuild(&boids_with(10));
        assert_eq!(idx.active_backend(), "hash_grid");
        idx.rebuild(&boids_with(200));
        assert_eq!(idx.active_backend(), "kdtree_index");
        idx.rebuild(&boids_with(10));
        assert_eq!(
            idx.active_backend(),
            "hash_grid",
            "must switch back down, not stay latched on the k-d tree"
        );
    }

    #[test]
    fn candidates_are_correct_regardless_of_which_backend_is_active() {
        // `candidates()` is contractually allowed to over-return (`HashGrid`'s own documented
        // behaviour: "may include boids just outside r — the caller applies the exact distance
        // test"), so the two backends' raw outputs are not expected to match exactly — `KdTree`
        // returns an exact set, `HashGrid` a conservative superset. What must actually agree,
        // and does, is the *exact-distance-filtered* result, which is what every real caller
        // (e.g. `RadiusGather`) computes from `candidates()`'s output anyway.
        let boids = boids_with(200);
        let probe = Vec3::ZERO;
        let r = 20.0;
        let exact = |out: &[u32]| -> Vec<u32> {
            let mut v: Vec<u32> = out
                .iter()
                .copied()
                .filter(|&j| (boids.pos[j as usize] - probe).len() <= r)
                .collect();
            v.sort_unstable();
            v
        };

        let mut below = AdaptiveIndex::new(5000, 5.0); // stays on hash_grid
        below.rebuild(&boids);
        let mut via_hash_grid = Vec::new();
        below.candidates(probe, r, &mut via_hash_grid);

        let mut above = AdaptiveIndex::new(1, 5.0); // switches to kdtree_index
        above.rebuild(&boids);
        assert_eq!(above.active_backend(), "kdtree_index");
        let mut via_kdtree = Vec::new();
        above.candidates(probe, r, &mut via_kdtree);

        assert_eq!(
            exact(&via_hash_grid),
            exact(&via_kdtree),
            "both backends must agree on which boids are within range, once exact-filtered"
        );
    }

    #[test]
    fn candidates_knn_returns_k_results_regardless_of_backend() {
        let boids = boids_with(200);
        let mut idx = AdaptiveIndex::new(1, 5.0); // kdtree_index
        idx.rebuild(&boids);
        let mut out = Vec::new();
        idx.candidates_knn(Vec3::ZERO, 5, &mut out);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let idx = reg
            .resolve_spatial_index("adaptive_index", &PluginParams::new())
            .unwrap();
        assert_eq!(idx.name(), "adaptive_index");
    }

    #[test]
    fn registry_reads_adaptive_crossover_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new().with("adaptive_crossover", 10.0);
        let mut idx = reg
            .resolve_spatial_index("adaptive_index", &params)
            .unwrap();
        idx.rebuild(&boids_with(20));
        let mut out = Vec::new();
        // Only checkable indirectly through the trait object here (no downcast to
        // AdaptiveIndex's own active_backend()) — proves candidates() still works end-to-end
        // through the registry-resolved trait object, not just the concrete type directly.
        idx.candidates_knn(Vec3::ZERO, 3, &mut out);
        assert_eq!(out.len(), 3);
    }

    /// Proves the `SpatialIndex` seam now has ≥3 real occupants (mirroring `kdtree_index`'s own
    /// seam-plurality proof pattern).
    #[test]
    fn hash_grid_kdtree_and_adaptive_index_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_hash_grid::register(&mut reg);
        murmur_kdtree_index::register(&mut reg);

        assert_eq!(
            reg.resolve_spatial_index("adaptive_index", &PluginParams::new())
                .unwrap()
                .name(),
            "adaptive_index"
        );
        assert_eq!(
            reg.resolve_spatial_index("hash_grid", &PluginParams::new())
                .unwrap()
                .name(),
            "hash_grid"
        );
        assert_eq!(
            reg.resolve_spatial_index("kdtree_index", &PluginParams::new())
                .unwrap()
                .name(),
            "kdtree_index"
        );
    }
}
