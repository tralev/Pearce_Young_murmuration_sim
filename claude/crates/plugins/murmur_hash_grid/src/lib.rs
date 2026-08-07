//! `HashGrid` — the first `SpatialIndex` plugin (design/01_core.md §5). Open space has no
//! fixed extent, so the grid is a hash map keyed by integer cell coordinates rather than a
//! dense array. No scientific specificity of its own — a data-structure choice, built first
//! only because it's the simplest thing that unblocks the slice (design/00_overview.md §2).

use std::collections::HashMap;

use murmur_core::{BoidColumns, CoreParams, PluginParams, Registry, SpatialIndex, Vec3, Warning};

pub struct HashGrid {
    cell_size: f64,
    cells: HashMap<[i64; 3], Vec<u32>>,
}

impl HashGrid {
    pub fn new(cell_size: f64) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "cell_size must be finite and > 0"
        );
        HashGrid {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_of(&self, p: Vec3) -> [i64; 3] {
        [
            (p.x / self.cell_size).floor() as i64,
            (p.y / self.cell_size).floor() as i64,
            (p.z / self.cell_size).floor() as i64,
        ]
    }
}

impl SpatialIndex for HashGrid {
    /// Clears each existing bucket in place (keeping its allocation) rather than dropping and
    /// reinserting the whole map, so a flock revisiting the same cells across frames doesn't
    /// keep re-allocating — buckets only grow on a boid's first visit to a brand-new cell.
    fn rebuild(&mut self, boids: &BoidColumns) {
        for bucket in self.cells.values_mut() {
            bucket.clear();
        }
        for i in boids.iter_active() {
            let cell = self.cell_of(boids.pos[i as usize]);
            self.cells.entry(cell).or_default().push(i);
        }
    }

    /// Superset of brute-force within `r`: sweeps every cell whose closest corner could be
    /// within `r` of `p` (a `+1` cell margin absorbs `p`'s own offset within its cell), so it
    /// may include boids just outside `r` — the caller applies the exact distance test
    /// (design/01_core.md §5).
    fn candidates(&self, p: Vec3, r: f64, out: &mut Vec<u32>) {
        out.clear();
        let base = self.cell_of(p);
        let reach = (r / self.cell_size).ceil() as i64 + 1;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
                    if let Some(bucket) = self.cells.get(&cell) {
                        out.extend_from_slice(bucket);
                    }
                }
            }
        }
    }

    /// An expanding ring search seeded from `cell_size` (matching droidmur's
    /// `SpatialGrid::query_knn` pattern, design/01_core.md §5) — overrides the trait's generic
    /// default so the growth starts from a grid-aware initial radius instead of an arbitrary
    /// `1.0`, and so a sparse flock (where one guess would under-return) still converges.
    ///
    /// Bounds growth by `reach` (the cell-cube half-width `candidates()` sweeps), not by an
    /// absolute radius multiplier: `candidates()` costs O(reach³), so a purely multiplicative
    /// radius cap (as in the trait's generic default) can demand an astronomically expensive
    /// sweep long before that cap is ever reached — this bit the conformance harness's
    /// zero-boid fixture directly (it never finds `k`, so naive growth never stops). If the
    /// index genuinely doesn't contain `k` boids within a computationally sane neighbourhood,
    /// this returns whatever was found instead of hanging.
    fn candidates_knn(&self, p: Vec3, k: u32, out: &mut Vec<u32>) {
        const MAX_REACH: i64 = 64; // worst single sweep: (2*64+1)^3 ≈ 2.15M cell lookups
        let mut r = self.cell_size;
        loop {
            self.candidates(p, r, out);
            if out.len() as u32 >= k {
                return;
            }
            let next_reach = (r * 2.0 / self.cell_size).ceil() as i64 + 1;
            if next_reach > MAX_REACH {
                return;
            }
            r *= 2.0;
        }
    }

    fn name(&self) -> &'static str {
        "hash_grid"
    }

    fn resolved_params(&self) -> PluginParams {
        PluginParams::new().with("cell_size", self.cell_size)
    }

    /// design/01_core.md §4.1's own named example: a `cell_size` set independently of
    /// `vision_radius` is a non-critical inconsistency (this index still works at any
    /// `cell_size` — `candidates()` is a correctness-preserving superset sweep regardless, see
    /// its own doc comment — just at a worse than necessary `O((r/cell_size)³)` sweep cost),
    /// so it's silently snapped to match rather than rejected. Exact-equality comparison is
    /// deliberate, not a numerical-tolerance oversight: both values are always literal,
    /// author-supplied doubles (a `PluginParams` override or a builder default), never a
    /// computed quantity that could carry rounding error.
    fn validate_and_fix(
        &mut self,
        core: &CoreParams,
        _others: &[(&str, PluginParams)],
    ) -> Vec<Warning> {
        if self.cell_size != core.vision_radius {
            let old = self.cell_size;
            self.cell_size = core.vision_radius;
            vec![Warning {
                plugin: self.name(),
                message: format!(
                    "cell_size ({old}) disagreed with vision_radius ({}); snapped to match \
                     (design/01_core.md §4.1)",
                    core.vision_radius
                ),
            }]
        } else {
            Vec::new()
        }
    }
}

/// Registers `HashGrid` under the name `"hash_grid"`, reading `cell_size` from `PluginParams`
/// (default `1.0` if unset). A real `Simulation` construction runs `validate_and_fix` above
/// afterward, snapping a mismatched `cell_size` to `vision_radius` (design/01_core.md §4.1).
pub fn register(r: &mut Registry) {
    r.register_spatial_index("hash_grid", |p: &PluginParams| {
        Box::new(HashGrid::new(p.get_or("cell_size", 1.0)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::Species;

    fn brute_force(boids: &[Vec3], p: Vec3, r: f64) -> Vec<usize> {
        boids
            .iter()
            .enumerate()
            .filter(|(_, &q)| (q - p).len() <= r)
            .map(|(i, _)| i)
            .collect()
    }

    fn deterministic_positions(n: usize, spread: f64) -> Vec<Vec3> {
        // Simple deterministic pseudo-random spread (no RNG dependency needed for this crate).
        (0..n)
            .map(|i| {
                let t = i as f64;
                Vec3::new(
                    ((t * 12.9898).sin() * 43758.5453).fract() * spread,
                    ((t * 78.233).sin() * 12345.6789).fract() * spread,
                    ((t * 37.719).sin() * 98765.4321).fract() * spread,
                )
            })
            .collect()
    }

    fn columns_from(positions: &[Vec3]) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(positions.len() as u32);
        for (i, &p) in positions.iter().enumerate() {
            b.add(p, Vec3::ZERO, Species::Prey, i as u64);
        }
        b
    }

    #[test]
    fn candidates_are_a_superset_of_brute_force() {
        let positions = deterministic_positions(200, 50.0);
        let boids = columns_from(&positions);
        let mut grid = HashGrid::new(5.0);
        grid.rebuild(&boids);

        for probe in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(25.0, 25.0, 25.0),
            Vec3::new(10.0, 40.0, 5.0),
        ] {
            for r in [3.0, 8.0, 15.0] {
                let expected: std::collections::HashSet<usize> =
                    brute_force(&positions, probe, r).into_iter().collect();
                let mut out = Vec::new();
                grid.candidates(probe, r, &mut out);
                let got: std::collections::HashSet<usize> =
                    out.iter().map(|&i| i as usize).collect();
                assert!(
                    expected.is_subset(&got),
                    "grid candidates at probe={probe:?} r={r} missed brute-force hits: \
                     expected {expected:?}, got {got:?}"
                );
            }
        }
    }

    #[test]
    fn candidates_knn_returns_exactly_k_even_in_a_sparse_flock() {
        // Sparse: few boids, spread far apart relative to cell_size — a single fixed-radius
        // guess (starting from cell_size=5) would under-return without the expanding search.
        // cell_size is kept in the same order of magnitude as the spread (as it would be in
        // practice, sized to vision_radius) — HashGrid's per-cell sweep is O((r/cell_size)^3),
        // so a cell_size wildly mismatched to the required search radius is a pathological
        // misconfiguration, not something candidates_knn needs to survive gracefully.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0),
            Vec3::new(-100.0, 0.0, 0.0),
        ];
        let boids = columns_from(&positions);
        let mut grid = HashGrid::new(5.0);
        grid.rebuild(&boids);

        let mut out = Vec::new();
        grid.candidates_knn(Vec3::ZERO, 3, &mut out);
        assert!(
            out.len() >= 3,
            "expected at least 3 candidates, got {}",
            out.len()
        );
    }

    #[test]
    fn rebuild_reuses_bucket_allocations_across_frames() {
        let positions = deterministic_positions(100, 30.0);
        let boids = columns_from(&positions);
        let mut grid = HashGrid::new(5.0);

        grid.rebuild(&boids);
        let cell_count_after_first = grid.cells.len();
        grid.rebuild(&boids); // same positions: no new cells should appear
        assert_eq!(grid.cells.len(), cell_count_after_first);
    }

    #[test]
    fn candidates_knn_on_an_empty_index_terminates_without_hanging() {
        let boids = BoidColumns::with_capacity(4); // zero boids added — k can never be satisfied
        let mut grid = HashGrid::new(1.0);
        grid.rebuild(&boids);
        let mut out = Vec::new();
        grid.candidates_knn(Vec3::ZERO, 2, &mut out); // must return promptly, not hang
        assert!(out.is_empty());
    }

    #[test]
    fn conforms_to_spatial_index_contract() {
        let boids = BoidColumns::with_capacity(4);
        let mut grid = HashGrid::new(1.0);
        murmur_conformance::spatial_index(&mut grid, &boids);
    }

    fn core_params_with_vision_radius(vision_radius: f64) -> murmur_core::CoreParams {
        murmur_core::CoreParams::builder()
            .vision_radius(vision_radius)
            .boid_count(1)
            .build()
            .unwrap()
    }

    #[test]
    fn validate_and_fix_snaps_a_mismatched_cell_size_to_vision_radius_and_warns() {
        let mut grid = HashGrid::new(3.0);
        let core = core_params_with_vision_radius(7.0);
        let warnings = grid.validate_and_fix(&core, &[]);
        assert_eq!(grid.cell_size, 7.0, "cell_size should be snapped in place");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].plugin, "hash_grid");
        assert!(warnings[0].message.contains("cell_size"));
    }

    #[test]
    fn validate_and_fix_is_silent_when_cell_size_already_matches_vision_radius() {
        let mut grid = HashGrid::new(7.0);
        let core = core_params_with_vision_radius(7.0);
        let warnings = grid.validate_and_fix(&core, &[]);
        assert_eq!(grid.cell_size, 7.0);
        assert!(warnings.is_empty());
    }

    #[test]
    fn resolved_params_reports_the_live_cell_size() {
        let grid = HashGrid::new(4.5);
        assert_eq!(grid.resolved_params().get("cell_size"), Some(4.5));
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let idx = reg
            .resolve_spatial_index("hash_grid", &PluginParams::new())
            .unwrap();
        assert_eq!(idx.name(), "hash_grid");
    }

    /// Proves the `SpatialIndex` seam is real: a trivial stub implementing the same trait
    /// registers and resolves alongside the real plugin (roadmap.md Phase 3 exit gate).
    #[test]
    fn a_different_spatial_index_plugin_swaps_in_via_the_same_seam() {
        struct StubIndex;
        impl SpatialIndex for StubIndex {
            fn rebuild(&mut self, _boids: &BoidColumns) {}
            fn candidates(&self, _p: Vec3, _r: f64, out: &mut Vec<u32>) {
                out.clear();
            }
            fn name(&self) -> &'static str {
                "stub_index"
            }
        }
        let mut reg = Registry::new();
        register(&mut reg);
        reg.register_spatial_index("stub_index", |_p| Box::new(StubIndex));

        let grid = reg
            .resolve_spatial_index("hash_grid", &PluginParams::new())
            .unwrap();
        let stub = reg
            .resolve_spatial_index("stub_index", &PluginParams::new())
            .unwrap();
        assert_eq!(grid.name(), "hash_grid");
        assert_eq!(stub.name(), "stub_index");
    }
}
