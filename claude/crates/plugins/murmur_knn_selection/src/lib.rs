//! `KnnSelection` — a topological (fixed neighbour *count*, not radius) `NeighborSelection`
//! plugin (design/01_core.md §6.1, roadmap.md Phase 13). Queries
//! `index.candidates_knn(pos, k + 1, ...)` (the `+1` absorbs the observer itself, which is
//! always its own closest "candidate" at distance 0), then applies the exact distance test,
//! sorts, and truncates to the `k` nearest — same "index over-returns, selection strategy does
//! the exact work" division of labour `RadiusGather` uses for radius-based gathering. No
//! scientific specificity of its own — matches the k-NN branch of `sci/comparison.md`'s
//! taxonomy (e.g. Young's own m-nearest-neighbour consensus graph), one of several plausible
//! strategies, not privileged over `RadiusGather`.

use murmur_core::{
    BoidColumns, CoreParams, Neighbor, NeighborSelection, PluginParams, Registry, SpatialIndex,
    MIN_LEN,
};

pub struct KnnSelection {
    k: u32,
}

impl KnnSelection {
    pub fn new(k: u32) -> Self {
        KnnSelection { k }
    }
}

impl NeighborSelection for KnnSelection {
    fn select(
        &self,
        index: &dyn SpatialIndex,
        i: u32,
        boids: &BoidColumns,
        _params: &CoreParams,
    ) -> Vec<Neighbor> {
        let pos_i = boids.pos[i as usize];
        let mut candidate_indices = Vec::new();
        index.candidates_knn(pos_i, self.k + 1, &mut candidate_indices);

        let mut with_dist: Vec<(f64, u32)> = candidate_indices
            .iter()
            .copied()
            .filter(|&j| j != i) // never one's own neighbour
            .map(|j| ((boids.pos[j as usize] - pos_i).len(), j))
            // coincident boids have no well-defined bearing — dropped, not degenerate
            .filter(|&(d, _)| d > MIN_LEN)
            .collect();
        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        with_dist.truncate(self.k as usize);

        with_dist
            .into_iter()
            .map(|(distance, j)| {
                let offset = boids.pos[j as usize] - pos_i;
                Neighbor {
                    index: j,
                    distance,
                    direction: offset / distance, // unit BEARING — never the neighbour's heading
                    velocity: boids.vel[j as usize],
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "knn_selection"
    }
}

/// Registers `KnnSelection` under the name `"knn_selection"`, reading `k` from `PluginParams`
/// (default `6.0` — Young's own empirical m* constant, a reasonable topological-neighbour-count
/// default even outside the H₂ context specifically).
pub fn register(r: &mut Registry) {
    r.register_neighbor_selection("knn_selection", |p: &PluginParams| {
        Box::new(KnnSelection::new(p.get_or("k", 6.0) as u32))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{Species, Vec3};
    use murmur_hash_grid::HashGrid;
    use murmur_kdtree_index::KdTree;

    fn params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn boids_in_a_line(n: usize) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(n as u32);
        for i in 0..n {
            b.add(
                Vec3::new(i as f64, 0.0, 0.0),
                Vec3::ZERO,
                Species::Prey,
                i as u64,
            );
        }
        b
    }

    #[test]
    fn conforms_to_neighbor_selection_contract() {
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0);
        let mut index = HashGrid::new(2.0);
        index.rebuild(&boids);
        murmur_conformance::neighbor_selection(&KnnSelection::new(3), &index, &boids, &params());
    }

    #[test]
    fn direction_is_bearing_not_the_neighbors_heading() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids
            .add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0)
            .unwrap();
        boids.add(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Species::Prey,
            1,
        );

        let mut index = KdTree::new();
        index.rebuild(&boids);

        let neighbors = KnnSelection::new(3).select(&index, i, &boids, &params());
        assert_eq!(neighbors.len(), 1);
        let n = &neighbors[0];
        assert!(
            (n.direction - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-9,
            "bearing should point +y"
        );
        assert!(
            (n.velocity - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9,
            "velocity should be the neighbour's own heading, +x"
        );
    }

    /// The real "topological, not radius" proof: with 10 boids strung out along a line, `k=3`
    /// must always return exactly the 3 nearest, regardless of how far the flock as a whole
    /// spans — `vision_radius` (10.0, per `params()`) is deliberately smaller than the line's
    /// full extent, so a radius-based strategy would see a different (and position-dependent)
    /// neighbour count here, not the fixed 3 this plugin guarantees.
    #[test]
    fn always_returns_the_k_nearest_regardless_of_spread() {
        let boids = boids_in_a_line(10); // positions 0..9 on the x-axis
        let mut index = KdTree::new();
        index.rebuild(&boids);

        // Boid 0 (at the line's end): the 3 nearest are 1, 2, 3.
        let neighbors = KnnSelection::new(3).select(&index, 0, &boids, &params());
        let mut got: Vec<u32> = neighbors.iter().map(|n| n.index).collect();
        got.sort();
        assert_eq!(got, vec![1, 2, 3]);

        // Boid 5 (in the middle): the 3 nearest are 4, 6, and either 3 or 7 (tied at distance
        // 2) — check the guaranteed pair plus a total count of exactly 3.
        let neighbors = KnnSelection::new(3).select(&index, 5, &boids, &params());
        assert_eq!(neighbors.len(), 3);
        let got: std::collections::HashSet<u32> = neighbors.iter().map(|n| n.index).collect();
        assert!(got.contains(&4));
        assert!(got.contains(&6));
    }

    #[test]
    fn works_identically_over_either_spatial_index_backend() {
        let boids = boids_in_a_line(10);

        let mut kd = KdTree::new();
        kd.rebuild(&boids);
        let mut hash = HashGrid::new(2.0);
        hash.rebuild(&boids);

        let mut via_kd: Vec<u32> = KnnSelection::new(3)
            .select(&kd, 5, &boids, &params())
            .iter()
            .map(|n| n.index)
            .collect();
        let mut via_hash: Vec<u32> = KnnSelection::new(3)
            .select(&hash, 5, &boids, &params())
            .iter()
            .map(|n| n.index)
            .collect();
        via_kd.sort();
        via_hash.sort();
        assert_eq!(
            via_kd, via_hash,
            "the selection strategy must not depend on which SpatialIndex backend is composed"
        );
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let sel = reg
            .resolve_neighbor_selection("knn_selection", &PluginParams::new())
            .unwrap();
        assert_eq!(sel.name(), "knn_selection");
    }

    #[test]
    fn registry_reads_k_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let p = PluginParams::new().with("k", 2.0);
        let sel = reg.resolve_neighbor_selection("knn_selection", &p).unwrap();

        let boids = boids_in_a_line(10);
        let mut index = KdTree::new();
        index.rebuild(&boids);
        let neighbors = sel.select(&index, 5, &boids, &params());
        assert_eq!(neighbors.len(), 2);
    }

    /// Proves the `NeighborSelection` seam now has ≥2 real occupants (roadmap.md Phase 13 exit
    /// gate, mirroring `torus_domain`'s/`kdtree_index`'s pattern).
    #[test]
    fn radius_gather_and_knn_selection_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_radius_gather::register(&mut reg);

        let knn = reg
            .resolve_neighbor_selection("knn_selection", &PluginParams::new())
            .unwrap();
        let radius = reg
            .resolve_neighbor_selection("radius_gather", &PluginParams::new())
            .unwrap();
        assert_eq!(knn.name(), "knn_selection");
        assert_eq!(radius.name(), "radius_gather");
    }
}
