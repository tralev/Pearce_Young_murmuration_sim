//! `KdTree` — a real k-d tree `SpatialIndex` plugin (design/01_core.md §5, roadmap.md
//! Phase 13). Unlike `HashGrid` (a hash-bucketed grid with an expanding-cell-sweep `knn`
//! fallback), this is a genuine k-d tree: `rebuild()` partitions boids into an implicit,
//! balanced binary layout in place (`[T]::select_nth_unstable_by`, O(N log N)); `candidates()`
//! and `candidates_knn()` are real O(log N)-average range/nearest-neighbour searches with
//! branch pruning, not an approximation. No scientific specificity of its own — same status as
//! `HashGrid`, a data-structure choice (design/00_overview.md §2's governing rule).

use murmur_core::{BoidColumns, PluginParams, Registry, SpatialIndex, Vec3};

fn axis_component(v: Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Recursively partitions `points` into an implicit k-d tree layout in place: after this call,
/// `points[len/2]` is the subtree's root (splitting on `axis = depth % 3`), `points[..len/2]`
/// holds the left subtree (axis value ≤ root's), and `points[len/2 + 1..]` holds the right
/// subtree (axis value ≥ root's) — recursively, each with `depth + 1`.
fn build(points: &mut [(Vec3, u32)], depth: usize) {
    if points.len() <= 1 {
        return;
    }
    let axis = depth % 3;
    let mid = points.len() / 2;
    points.select_nth_unstable_by(mid, |a, b| {
        axis_component(a.0, axis)
            .partial_cmp(&axis_component(b.0, axis))
            .expect("boid positions must never be NaN")
    });
    let (left, rest) = points.split_at_mut(mid);
    let (_pivot, right) = rest.split_first_mut().expect("mid < points.len()");
    build(left, depth + 1);
    build(right, depth + 1);
}

/// Exact range search (not an over-return like `HashGrid`'s cell sweep — a real k-d tree can
/// afford the exact distance test inline). Still a valid `candidates()` implementation: exact
/// is a (trivial) subset of "may include boids just outside r."
fn range_search(points: &[(Vec3, u32)], p: Vec3, r: f64, depth: usize, out: &mut Vec<u32>) {
    if points.is_empty() {
        return;
    }
    let axis = depth % 3;
    let mid = points.len() / 2;
    let (pt, idx) = points[mid];
    if (pt - p).len() <= r {
        out.push(idx);
    }
    let diff = axis_component(p, axis) - axis_component(pt, axis);
    // Left holds axis values <= pivot: reachable if the query's lower edge could cross it.
    if diff <= r {
        range_search(&points[..mid], p, r, depth + 1, out);
    }
    // Right holds axis values >= pivot: reachable if the query's upper edge could cross it.
    if diff >= -r {
        range_search(&points[mid + 1..], p, r, depth + 1, out);
    }
}

/// Inserts `(dist_sq, idx)` into `heap` (kept sorted ascending by `dist_sq`, capped at `k`
/// entries) if it's among the `k` closest seen so far. `O(k)` per insert (a plain sorted `Vec`,
/// not a real heap) — appropriate for the small `k` (typically single digits to a few dozen)
/// this is ever called with; a real `BinaryHeap` would need a `NaN`-safe ordering wrapper for
/// no real benefit at this scale.
fn insert_bounded(heap: &mut Vec<(f64, u32)>, k: usize, dist_sq: f64, idx: u32) {
    if heap.len() < k {
        let pos = heap.partition_point(|&(d, _)| d < dist_sq);
        heap.insert(pos, (dist_sq, idx));
    } else if let Some(&(worst, _)) = heap.last() {
        if dist_sq < worst {
            let pos = heap.partition_point(|&(d, _)| d < dist_sq);
            heap.insert(pos, (dist_sq, idx));
            heap.pop();
        }
    }
}

/// Exact k-nearest search with branch pruning: descends the near subtree unconditionally, the
/// far subtree only if it could still contain something closer than the current k-th best.
fn knn_search(points: &[(Vec3, u32)], p: Vec3, k: usize, depth: usize, heap: &mut Vec<(f64, u32)>) {
    if points.is_empty() || k == 0 {
        return;
    }
    let axis = depth % 3;
    let mid = points.len() / 2;
    let (pt, idx) = points[mid];
    insert_bounded(heap, k, (pt - p).len_sq(), idx);

    let diff = axis_component(p, axis) - axis_component(pt, axis);
    let (near, far) = if diff <= 0.0 {
        (&points[..mid], &points[mid + 1..])
    } else {
        (&points[mid + 1..], &points[..mid])
    };
    knn_search(near, p, k, depth + 1, heap);
    let worst = if heap.len() < k {
        f64::INFINITY
    } else {
        heap.last().map(|&(d, _)| d).unwrap_or(f64::INFINITY)
    };
    if diff * diff <= worst {
        knn_search(far, p, k, depth + 1, heap);
    }
}

#[derive(Default)]
pub struct KdTree {
    points: Vec<(Vec3, u32)>,
}

impl KdTree {
    pub fn new() -> Self {
        KdTree::default()
    }
}

impl SpatialIndex for KdTree {
    /// O(N) to collect + O(N log N) to partition — same complexity class as `HashGrid`'s O(N)
    /// rebuild plus this index's better query asymptotics, not a free lunch either way.
    fn rebuild(&mut self, boids: &BoidColumns) {
        self.points.clear();
        for i in boids.iter_active() {
            self.points.push((boids.pos[i as usize], i));
        }
        build(&mut self.points, 0);
    }

    fn candidates(&self, p: Vec3, r: f64, out: &mut Vec<u32>) {
        out.clear(); // matches `HashGrid`'s actual (clear-and-refill) behaviour, not the
                     // trait doc's stale "not cleared" comment — every existing caller
                     // (`RadiusGather`) already relies on the tested precedent, not the doc.
        range_search(&self.points, p, r, 0, out);
    }

    fn candidates_knn(&self, p: Vec3, k: u32, out: &mut Vec<u32>) {
        out.clear();
        let mut heap = Vec::with_capacity(k as usize);
        knn_search(&self.points, p, k as usize, 0, &mut heap);
        out.extend(heap.into_iter().map(|(_, idx)| idx));
    }

    fn name(&self) -> &'static str {
        "kdtree_index"
    }
}

/// Registers `KdTree` under the name `"kdtree_index"` — no plugin params needed (unlike
/// `HashGrid`'s `cell_size`, a k-d tree has no comparable tuning knob at this level of detail).
pub fn register(r: &mut Registry) {
    r.register_spatial_index("kdtree_index", |_p: &PluginParams| Box::new(KdTree::new()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::Species;

    fn deterministic_positions(n: usize, spread: f64) -> Vec<Vec3> {
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

    fn brute_force_within(boids: &[Vec3], p: Vec3, r: f64) -> std::collections::HashSet<usize> {
        boids
            .iter()
            .enumerate()
            .filter(|(_, &q)| (q - p).len() <= r)
            .map(|(i, _)| i)
            .collect()
    }

    fn brute_force_knn(boids: &[Vec3], p: Vec3, k: usize) -> std::collections::HashSet<usize> {
        let mut with_dist: Vec<(f64, usize)> = boids
            .iter()
            .enumerate()
            .map(|(i, &q)| ((q - p).len_sq(), i))
            .collect();
        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        with_dist.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn conforms_to_spatial_index_contract() {
        let boids = BoidColumns::with_capacity(4);
        let mut tree = KdTree::new();
        murmur_conformance::spatial_index(&mut tree, &boids);
    }

    #[test]
    fn candidates_exactly_match_brute_force() {
        let positions = deterministic_positions(200, 50.0);
        let boids = columns_from(&positions);
        let mut tree = KdTree::new();
        tree.rebuild(&boids);

        for probe in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(25.0, 25.0, 25.0),
            Vec3::new(10.0, 40.0, 5.0),
        ] {
            for r in [3.0, 8.0, 15.0] {
                let expected = brute_force_within(&positions, probe, r);
                let mut out = Vec::new();
                tree.candidates(probe, r, &mut out);
                let got: std::collections::HashSet<usize> =
                    out.iter().map(|&i| i as usize).collect();
                assert_eq!(
                    got, expected,
                    "kdtree candidates at probe={probe:?} r={r} diverged from brute force"
                );
            }
        }
    }

    #[test]
    fn candidates_knn_finds_the_true_k_nearest() {
        let positions = deterministic_positions(150, 40.0);
        let boids = columns_from(&positions);
        let mut tree = KdTree::new();
        tree.rebuild(&boids);

        for probe in [Vec3::ZERO, Vec3::new(20.0, 5.0, -10.0)] {
            for k in [1u32, 5, 20] {
                let expected = brute_force_knn(&positions, probe, k as usize);
                let mut out = Vec::new();
                tree.candidates_knn(probe, k, &mut out);
                assert_eq!(out.len(), k as usize);
                let got: std::collections::HashSet<usize> =
                    out.iter().map(|&i| i as usize).collect();
                assert_eq!(
                    got, expected,
                    "kdtree_knn at probe={probe:?} k={k} diverged from brute-force top-k"
                );
            }
        }
    }

    #[test]
    fn empty_index_terminates_without_panicking() {
        let boids = BoidColumns::with_capacity(4);
        let mut tree = KdTree::new();
        tree.rebuild(&boids);
        let mut out = Vec::new();
        tree.candidates(Vec3::ZERO, 10.0, &mut out);
        assert!(out.is_empty());
        tree.candidates_knn(Vec3::ZERO, 5, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn single_boid_index_works() {
        let boids = columns_from(&[Vec3::new(1.0, 2.0, 3.0)]);
        let mut tree = KdTree::new();
        tree.rebuild(&boids);
        let mut out = Vec::new();
        tree.candidates(Vec3::ZERO, 100.0, &mut out);
        assert_eq!(out, vec![0]);
        tree.candidates_knn(Vec3::ZERO, 5, &mut out);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn rebuild_reflects_updated_positions() {
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0);
        boids.add(Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut tree = KdTree::new();
        tree.rebuild(&boids);

        let mut out = Vec::new();
        tree.candidates(Vec3::ZERO, 1.0, &mut out);
        assert_eq!(out, vec![0]);

        boids.pos[1] = Vec3::new(0.5, 0.0, 0.0); // boid 1 moves into range
        tree.rebuild(&boids);
        tree.candidates(Vec3::ZERO, 1.0, &mut out);
        let got: std::collections::HashSet<u32> = out.iter().copied().collect();
        assert_eq!(got, [0, 1].into_iter().collect());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let idx = reg
            .resolve_spatial_index("kdtree_index", &PluginParams::new())
            .unwrap();
        assert_eq!(idx.name(), "kdtree_index");
    }

    /// Proves the `SpatialIndex` seam now has ≥2 real occupants (roadmap.md Phase 13 exit
    /// gate, mirroring Phase 3's/`torus_domain`'s pattern).
    #[test]
    fn hash_grid_and_kdtree_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_hash_grid::register(&mut reg);

        let kdtree = reg
            .resolve_spatial_index("kdtree_index", &PluginParams::new())
            .unwrap();
        let grid = reg
            .resolve_spatial_index("hash_grid", &PluginParams::new())
            .unwrap();
        assert_eq!(kdtree.name(), "kdtree_index");
        assert_eq!(grid.name(), "hash_grid");
    }
}
