//! `Domain` — topology/boundary trait (design/01_core.md §3). The trait is core; `OpenSpace`
//! and every other concrete topology (`Torus`, `Margin`, `Sphere`, `SphereSoft`) are plugins —
//! no implementation is structurally privileged.

use crate::math::Vec3;
use crate::params::CoreParams;
use crate::registry::{PluginParams, Warning};

pub trait Domain: Send + Sync {
    /// Shortest displacement a → b under this topology. In open space this is simply `b − a`;
    /// a torus uses the minimum-image convention.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3;
    /// Applies any positional constraint after integration (no-op for open space; wraps for a
    /// torus; clamps/nudges for a bounded domain).
    fn apply(&self, pos: &mut Vec3, vel: &mut Vec3, dt: f64);

    /// Distance to, and inward-pointing unit normal at, the nearest boundary point from `pos` —
    /// `None` for a domain with no boundary concept at all (`OpenSpace`, `Torus`'s periodic
    /// wrap). Fixes **G5** (roadmap.md §12): a force-computing seam (`FlockingMode::desired()`,
    /// or a `StepHook`) had no way to ask "how far am I from a wall, and which direction is
    /// it" *during* force computation — `Domain` previously exposed only `delta()` and the
    /// post-integration `apply()`. Default `None` — optional, costs nothing for a domain
    /// without a boundary, matching every other lazily-fixed gap's own precedent (G4/G6/G7,
    /// D22b: fix exactly when a phase's own work needs it, here `murmur_maxent_social`'s
    /// "boundary" wall-falloff channel, Phase 17). The returned distance is signed exactly like
    /// each bounded `Domain`'s own `apply()` reasons about it (positive inside the boundary,
    /// negative once already past it — `Margin`/`Sphere`/`SphereSoft` never clamp `pos` inside
    /// this query, only `apply()` does that, and only for `Margin`/`Sphere`).
    fn boundary_distance(&self, _pos: Vec3) -> Option<(f64, Vec3)> {
        None
    }

    fn name(&self) -> &'static str;

    /// See `FlockingMode::resolved_params` (modes.rs) — same seam, same default.
    fn resolved_params(&self) -> PluginParams {
        PluginParams::new()
    }

    /// See `FlockingMode::validate_and_fix` (modes.rs) — same seam, same default, same timing.
    fn validate_and_fix(
        &mut self,
        _core: &CoreParams,
        _others: &[(&str, PluginParams)],
    ) -> Vec<Warning> {
        Vec::new()
    }
}
