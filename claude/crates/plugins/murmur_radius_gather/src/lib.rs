//! `RadiusGather` — the first `NeighborSelection` plugin (design/01_core.md §6.1). Queries
//! `index.candidates(pos, vision_radius)`, applies the exact distance test, and fills
//! bearing/velocity for each surviving candidate. Radius-based gathering has no scientific
//! specificity of its own — one of several plausible strategies (`sci/comparison.md`'s
//! taxonomy), built first only because it's the simplest thing that unblocks the slice.
//!
//! **Note on domain-awareness.** The `NeighborSelection::select` trait signature
//! (design/01_core.md §6.1) does not receive a `&dyn Domain` — unlike `BoidCtx`, which does —
//! so this implementation computes displacement as plain `pos[j] − pos[i]`, correct under
//! `OpenSpace` (the only `Domain` this slice composes) but not minimum-image-aware under a
//! future wrapping `Domain` (e.g. `Torus`). That gap is inherited from the trait's designed
//! signature, not introduced here.

use murmur_core::{
    BoidColumns, CoreParams, Neighbor, NeighborSelection, PluginParams, Registry, SpatialIndex,
    MIN_LEN,
};

pub struct RadiusGather;

impl NeighborSelection for RadiusGather {
    fn select(
        &self,
        index: &dyn SpatialIndex,
        i: u32,
        boids: &BoidColumns,
        params: &CoreParams,
    ) -> Vec<Neighbor> {
        let pos_i = boids.pos[i as usize];
        let mut candidate_indices = Vec::new();
        index.candidates(pos_i, params.vision_radius, &mut candidate_indices);

        let mut neighbors = Vec::with_capacity(candidate_indices.len());
        for &j in &candidate_indices {
            if j == i {
                continue; // never one's own neighbour
            }
            let offset = boids.pos[j as usize] - pos_i;
            let distance = offset.len();
            // exact distance test (the index may over-return); coincident boids have no
            // well-defined bearing and are dropped rather than producing a degenerate direction.
            if distance > params.vision_radius || distance <= MIN_LEN {
                continue;
            }
            neighbors.push(Neighbor {
                index: j,
                distance,
                direction: offset / distance, // unit BEARING — never the neighbour's heading
                velocity: boids.vel[j as usize], // the neighbour's own heading × speed
            });
        }
        neighbors
    }

    fn name(&self) -> &'static str {
        "radius_gather"
    }
}

/// Registers `RadiusGather` under the name `"radius_gather"`.
pub fn register(r: &mut Registry) {
    r.register_neighbor_selection("radius_gather", |_p: &PluginParams| Box::new(RadiusGather));
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{Species, Vec3};
    use murmur_hash_grid::HashGrid;

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

    /// The v1 data-model bug regression test: a neighbour's bearing (direction to it) must
    /// never be confused with its heading (its own velocity direction) — here they point in
    /// deliberately different directions so conflating them would be caught immediately.
    #[test]
    fn direction_is_bearing_not_the_neighbors_heading() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids
            .add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0)
            .unwrap();
        // Neighbour sits due +y of the observer, but is heading due +x.
        let _j = boids
            .add(
                Vec3::new(0.0, 5.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Species::Prey,
                1,
            )
            .unwrap();

        let mut grid = HashGrid::new(2.0);
        grid.rebuild(&boids);
        let p = params(10.0);

        let neighbors = RadiusGather.select(&grid, i, &boids, &p);
        assert_eq!(neighbors.len(), 1);
        let n = &neighbors[0];
        assert_ne!(
            n.direction,
            n.velocity.normalized(),
            "bearing must not equal heading here"
        );
        assert!(
            (n.direction - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-9,
            "bearing should point +y"
        );
        assert!(
            (n.velocity - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9,
            "velocity should be the neighbour's own heading, +x"
        );
    }

    #[test]
    fn excludes_self_and_out_of_range_candidates() {
        let mut boids = BoidColumns::with_capacity(3);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        boids
            .add(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1)
            .unwrap(); // in range
        boids
            .add(Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 2)
            .unwrap(); // out of range

        let mut grid = HashGrid::new(5.0);
        grid.rebuild(&boids);
        let p = params(10.0);

        let neighbors = RadiusGather.select(&grid, i, &boids, &p);
        assert_eq!(neighbors.len(), 1);
        assert_ne!(neighbors[0].index, i);
        assert!(neighbors[0].distance <= 10.0);
    }

    #[test]
    fn distance_and_direction_are_consistent() {
        let mut boids = BoidColumns::with_capacity(2);
        let i = boids.add(Vec3::ZERO, Vec3::ZERO, Species::Prey, 0).unwrap();
        boids
            .add(Vec3::new(3.0, 4.0, 0.0), Vec3::ZERO, Species::Prey, 1)
            .unwrap(); // dist = 5

        let mut grid = HashGrid::new(10.0);
        grid.rebuild(&boids);
        let p = params(10.0);

        let neighbors = RadiusGather.select(&grid, i, &boids, &p);
        assert_eq!(neighbors.len(), 1);
        let n = &neighbors[0];
        assert!((n.distance - 5.0).abs() < 1e-9);
        assert!(
            (n.direction.len() - 1.0).abs() < 1e-9,
            "direction must be a unit bearing"
        );
        assert!((n.direction - Vec3::new(0.6, 0.8, 0.0)).len() < 1e-9);
    }

    #[test]
    fn conforms_to_neighbor_selection_contract() {
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0);
        let mut grid = HashGrid::new(2.0);
        grid.rebuild(&boids);
        let p = params(10.0);
        murmur_conformance::neighbor_selection(&RadiusGather, &grid, &boids, &p);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let sel = reg
            .resolve_neighbor_selection("radius_gather", &PluginParams::new())
            .unwrap();
        assert_eq!(sel.name(), "radius_gather");
    }

    /// Proves the `NeighborSelection` seam is real: a trivial stub implementing the same trait
    /// registers and resolves alongside the real plugin (roadmap.md Phase 3 exit gate).
    #[test]
    fn a_different_neighbor_selection_plugin_swaps_in_via_the_same_seam() {
        struct StubSelection;
        impl NeighborSelection for StubSelection {
            fn select(
                &self,
                _index: &dyn SpatialIndex,
                _i: u32,
                _boids: &BoidColumns,
                _params: &CoreParams,
            ) -> Vec<Neighbor> {
                Vec::new()
            }
            fn name(&self) -> &'static str {
                "stub_selection"
            }
        }
        let mut reg = Registry::new();
        register(&mut reg);
        reg.register_neighbor_selection("stub_selection", |_p| Box::new(StubSelection));

        let real = reg
            .resolve_neighbor_selection("radius_gather", &PluginParams::new())
            .unwrap();
        let stub = reg
            .resolve_neighbor_selection("stub_selection", &PluginParams::new())
            .unwrap();
        assert_eq!(real.name(), "radius_gather");
        assert_eq!(stub.name(), "stub_selection");
    }
}
