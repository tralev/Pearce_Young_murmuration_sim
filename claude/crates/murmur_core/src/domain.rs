//! `Domain` — topology/boundary trait (design/01_core.md §3). The trait is core; `OpenSpace`
//! and every other concrete topology (`Torus`, `Margin`, `Sphere`, `SphereSoft`) are plugins —
//! no implementation is structurally privileged.

use crate::math::Vec3;

pub trait Domain: Send + Sync {
    /// Shortest displacement a → b under this topology. In open space this is simply `b − a`;
    /// a torus uses the minimum-image convention.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3;
    /// Applies any positional constraint after integration (no-op for open space; wraps for a
    /// torus; clamps/nudges for a bounded domain).
    fn apply(&self, pos: &mut Vec3, vel: &mut Vec3, dt: f64);
    fn name(&self) -> &'static str;
}
