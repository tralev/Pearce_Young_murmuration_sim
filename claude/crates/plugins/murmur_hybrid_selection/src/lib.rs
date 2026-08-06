//! `HybridSelection` — a metric + topological `NeighborSelection` plugin with an optional
//! forward-facing FOV cone (design/01_core.md §6.1, roadmap.md Phase 16). Ported by
//! description from pymurmur's `plugins/neighbor_selection.py` — its "richest selector":
//! design/02_plugins.md §5 names it "Metric + topological + optional per-interaction FOV cones
//! (separate cone per separation/alignment/cohesion)". pymurmur's actual source isn't reachable
//! in this environment (the same blocker as every other pymurmur cross-check this project has
//! hit).
//!
//! **A deliberate, documented scope reduction from pymurmur's description.**
//! `NeighborSelection::select()` returns one `Vec<Neighbor>` — the trait has no notion of
//! separate per-force-channel (separation/alignment/cohesion) neighbour sets, and every
//! `FlockingMode`/`SteeringModifier` in this codebase consumes a single shared list. Giving
//! this plugin three independent FOV cones would mean either fabricating per-channel behaviour
//! nothing downstream can actually use, or extending the trait signature itself — real
//! architectural-gap territory (in the spirit of G1–G7, roadmap.md §12), not something to do
//! silently inside one plugin's implementation. Implemented instead: **one shared** optional
//! forward-facing FOV cone, applied uniformly to the combined metric+topological set — genuinely
//! useful (a hybrid selector that also respects a visibility cone), honestly short of pymurmur's
//! fuller per-channel design, and documented as such rather than silently narrowed.
//!
//! **Metric + topological combination.** Returns the *union* of (a) every boid within
//! `vision_radius` (`RadiusGather`'s own criterion — arbitrarily many, unbounded by count) and
//! (b) the `k` topologically nearest boids (`KnnSelection`'s own criterion — a fixed count,
//! regardless of how sparse or dense the neighbourhood is). Each criterion alone has a known
//! failure mode this hybrid avoids: metric-only can return zero neighbours in a sparse patch of
//! flock even one boid-width outside `vision_radius`; topological-only can pull in an
//! arbitrarily distant "nearest" boid when the whole flock is sparse. The union keeps both
//! guarantees simultaneously.

use murmur_core::{
    BoidColumns, CoreParams, Neighbor, NeighborSelection, PluginParams, Registry, SpatialIndex,
    MIN_LEN,
};

pub struct HybridSelection {
    pub k: u32,
    pub fov_enabled: bool,
    pub fov_half_angle: f64,
}

impl HybridSelection {
    pub fn new(k: u32, fov_enabled: bool, fov_half_angle: f64) -> Self {
        HybridSelection {
            k,
            fov_enabled,
            fov_half_angle,
        }
    }
}

impl NeighborSelection for HybridSelection {
    fn select(
        &self,
        index: &dyn SpatialIndex,
        i: u32,
        boids: &BoidColumns,
        params: &CoreParams,
    ) -> Vec<Neighbor> {
        let pos_i = boids.pos[i as usize];

        // Metric: everything within vision_radius (index may over-return; exact test here).
        let mut metric_raw = Vec::new();
        index.candidates(pos_i, params.vision_radius, &mut metric_raw);
        let metric = metric_raw
            .into_iter()
            .filter(|&j| j != i && (boids.pos[j as usize] - pos_i).len() <= params.vision_radius);

        // Topological: exact k nearest (the `+1` absorbs the observer itself, always its own
        // closest "candidate" at distance 0) — same exact-sort-and-truncate approach
        // `KnnSelection` uses, so this stays backend-agnostic regardless of whether the
        // composed `SpatialIndex` over-returns from `candidates_knn`.
        let mut topo_raw = Vec::new();
        index.candidates_knn(pos_i, self.k + 1, &mut topo_raw);
        let mut topo_with_dist: Vec<(f64, u32)> = topo_raw
            .into_iter()
            .filter(|&j| j != i)
            .map(|j| ((boids.pos[j as usize] - pos_i).len(), j))
            .collect();
        topo_with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        topo_with_dist.truncate(self.k as usize);

        let mut union: Vec<u32> = metric
            .chain(topo_with_dist.into_iter().map(|(_, j)| j))
            .collect();
        union.sort_unstable();
        union.dedup();

        // The observer's own forward heading, for the optional FOV cone. A stalled
        // (near-zero-speed) observer has no defined "forward" — treated as omnidirectional
        // rather than excluding everything or picking an arbitrary direction.
        let heading = boids.vel[i as usize];
        let forward = if heading.len() > MIN_LEN {
            Some(heading.normalized())
        } else {
            None
        };
        let cos_half_angle = self.fov_half_angle.cos();

        let mut neighbors = Vec::with_capacity(union.len());
        for j in union {
            let offset = boids.pos[j as usize] - pos_i;
            let distance = offset.len();
            // coincident boids have no well-defined bearing — dropped, not degenerate
            if distance <= MIN_LEN {
                continue;
            }
            let direction = offset / distance;
            if self.fov_enabled {
                if let Some(f) = forward {
                    if direction.dot(f) < cos_half_angle {
                        continue; // outside the forward visibility cone
                    }
                }
            }
            neighbors.push(Neighbor {
                index: j,
                distance,
                direction,
                velocity: boids.vel[j as usize],
            });
        }
        neighbors
    }

    fn name(&self) -> &'static str {
        "hybrid_selection"
    }
}

/// Registers `HybridSelection` under the name `"hybrid_selection"`, reading `k` (default `6.0`,
/// the same key `knn_selection` reads — safe, `NeighborSelection` is a single-occupant socket),
/// `fov_enabled` (default `0.0`/off — plain metric+topological hybrid is the baseline
/// behaviour), and `fov_half_angle` (default `1.2` radians, ≈69°, a plausible forward cone).
pub fn register(r: &mut Registry) {
    r.register_neighbor_selection("hybrid_selection", |p: &PluginParams| {
        Box::new(HybridSelection::new(
            p.get_or("k", 6.0) as u32,
            p.get_or("fov_enabled", 0.0) != 0.0,
            p.get_or("fov_half_angle", 1.2),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{Species, Vec3};
    use murmur_hash_grid::HashGrid;
    use murmur_kdtree_index::KdTree;

    fn params(vision_radius: f64) -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(vision_radius)
            .build()
            .unwrap()
    }

    #[test]
    fn conforms_to_neighbor_selection_contract() {
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0);
        let mut grid = HashGrid::new(2.0);
        grid.rebuild(&boids);
        murmur_conformance::neighbor_selection(
            &HybridSelection::new(3, false, 1.0),
            &grid,
            &boids,
            &params(10.0),
        );
    }

    #[test]
    fn metric_component_includes_boids_beyond_the_topological_k() {
        // k=1, but 3 boids all sit within vision_radius=10 — the union must include all 3, not
        // just the single topologically-nearest one.
        let mut boids = BoidColumns::with_capacity(4);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        boids.add(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        boids.add(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 2);
        boids.add(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 3);

        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors =
            HybridSelection::new(1, false, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(
            neighbors.len(),
            3,
            "metric radius must pull in all 3, beyond k=1"
        );
    }

    #[test]
    fn topological_component_includes_a_boid_beyond_vision_radius() {
        // vision_radius=5 sees nothing; the only 2 boids that exist are farther out (6, 7), but
        // k=2 must still return them both via the topological path.
        let mut boids = BoidColumns::with_capacity(3);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        boids.add(Vec3::new(6.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        boids.add(Vec3::new(7.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 2);

        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors = HybridSelection::new(2, false, 1.0).select(&index, i, &boids, &params(5.0));
        assert_eq!(
            neighbors.len(),
            2,
            "topological k must rescue neighbours beyond an empty metric radius"
        );
    }

    #[test]
    fn union_has_no_duplicate_entries_for_a_boid_in_both_sets() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        boids.add(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1); // in both sets

        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors =
            HybridSelection::new(5, false, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn fov_cone_excludes_a_neighbour_directly_behind_the_observer() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids
            .add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0) // heading +x
            .unwrap();
        boids.add(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1); // directly behind

        let mut index = KdTree::new();
        index.rebuild(&boids);

        let without_fov =
            HybridSelection::new(5, false, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(
            without_fov.len(),
            1,
            "no cone: the boid behind is still a neighbour"
        );

        let with_fov = HybridSelection::new(5, true, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(
            with_fov.len(),
            0,
            "1.0 rad half-angle forward cone must exclude a boid directly behind"
        );
    }

    #[test]
    fn fov_cone_includes_a_neighbour_ahead_of_the_observer() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids
            .add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0)
            .unwrap();
        boids.add(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1); // straight ahead

        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors = HybridSelection::new(5, true, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn a_stalled_observer_with_no_heading_is_treated_as_omnidirectional() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap(); // zero velocity
        boids.add(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);

        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors = HybridSelection::new(5, true, 1.0).select(&index, i, &boids, &params(10.0));
        assert_eq!(
            neighbors.len(),
            1,
            "a stalled observer has no defined forward direction, so the cone must not exclude anything"
        );
    }

    #[test]
    fn works_identically_over_either_spatial_index_backend() {
        let mut boids = BoidColumns::with_capacity(5);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        for k in 1..5 {
            boids.add(
                Vec3::new(k as f64 * 1.5, 0.0, 0.0),
                Vec3::ZERO,
                Species::Prey,
                k as u64,
            );
        }

        let mut kd = KdTree::new();
        kd.rebuild(&boids);
        let mut hash = HashGrid::new(2.0);
        hash.rebuild(&boids);

        let sel = HybridSelection::new(2, false, 1.0);
        let mut via_kd: Vec<u32> = sel
            .select(&kd, i, &boids, &params(3.0))
            .iter()
            .map(|n| n.index)
            .collect();
        let mut via_hash: Vec<u32> = sel
            .select(&hash, i, &boids, &params(3.0))
            .iter()
            .map(|n| n.index)
            .collect();
        via_kd.sort();
        via_hash.sort();
        assert_eq!(via_kd, via_hash);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let sel = reg
            .resolve_neighbor_selection("hybrid_selection", &PluginParams::new())
            .unwrap();
        assert_eq!(sel.name(), "hybrid_selection");
    }

    #[test]
    fn registry_reads_k_and_fov_overrides() {
        let mut reg = Registry::new();
        register(&mut reg);
        let p = PluginParams::new()
            .with("k", 1.0)
            .with("fov_enabled", 1.0)
            .with("fov_half_angle", 1.0);
        let sel = reg
            .resolve_neighbor_selection("hybrid_selection", &p)
            .unwrap();

        let mut boids = BoidColumns::with_capacity(2);
        let i = boids
            .add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0)
            .unwrap();
        boids.add(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut index = KdTree::new();
        index.rebuild(&boids);

        let neighbors = sel.select(&index, i, &boids, &params(10.0));
        assert_eq!(neighbors.len(), 0, "fov_enabled override must take effect");
    }

    /// Proves the `NeighborSelection` seam now has ≥3 real occupants (mirroring
    /// `knn_selection`'s own seam-plurality proof pattern).
    #[test]
    fn radius_gather_knn_and_hybrid_selection_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_radius_gather::register(&mut reg);
        murmur_knn_selection::register(&mut reg);

        assert_eq!(
            reg.resolve_neighbor_selection("hybrid_selection", &PluginParams::new())
                .unwrap()
                .name(),
            "hybrid_selection"
        );
        assert_eq!(
            reg.resolve_neighbor_selection("radius_gather", &PluginParams::new())
                .unwrap()
                .name(),
            "radius_gather"
        );
        assert_eq!(
            reg.resolve_neighbor_selection("knn_selection", &PluginParams::new())
                .unwrap()
                .name(),
            "knn_selection"
        );
    }
}
