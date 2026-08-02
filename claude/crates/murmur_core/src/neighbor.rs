//! `Neighbor` record + `NeighborSelection` trait (design/01_core.md §6). The trait is core;
//! `RadiusGather` (the first plugin) has no privileged status over later k-NN/hybrid+FOV
//! selectors.

use crate::boids::BoidColumns;
use crate::math::Vec3;
use crate::params::CoreParams;
use crate::spatial_index::SpatialIndex;

pub struct Neighbor {
    pub index: u32,
    pub distance: f64,
    /// Unit bearing d̂ⱼ from the observer to this neighbour — never a heading.
    pub direction: Vec3,
    /// The neighbour's own heading × speed — never the bearing. (This distinction was the v1
    /// data-model bug; the two fields are kept explicit here.)
    pub velocity: Vec3,
}

pub trait NeighborSelection: Send + Sync {
    fn select(
        &self,
        index: &dyn SpatialIndex,
        i: u32,
        boids: &BoidColumns,
        params: &CoreParams,
    ) -> Vec<Neighbor>;
    fn name(&self) -> &'static str;
}
