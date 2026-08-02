//! Retained kernel-trait toolkit (design/02_plugins.md §3) — reusable building blocks for
//! plugin authors composing a new `FlockingMode`. Not itself an algorithm choice: no plugin is
//! obligated to use it. Convention: `CohesionKernel::target` returns an **offset** from the
//! boid toward the local centre (not an absolute point); `AlignmentKernel::heading` returns a
//! **unit heading** built from neighbour `velocity` (never `direction`).

use crate::math::Vec3;
use crate::modes::BoidCtx;
use crate::params::CoreParams;

pub trait SeparationKernel: Send + Sync {
    fn weight(&self, d: f64, params: &CoreParams) -> f64;
}

pub trait CohesionKernel: Send + Sync {
    fn target(&self, ctx: BoidCtx<'_>) -> Vec3;
}

pub trait AlignmentKernel: Send + Sync {
    fn heading(&self, ctx: BoidCtx<'_>) -> Vec3;
}
