//! `FlockingMode` (desire) and `SteeringModifier` (response) — design/01_core.md §8. Split into
//! two composable traits specifically so a physical-response model (e.g. behavioural inertia)
//! can pair with *any* mode instead of being baked into one (design decision D13).
//!
//! **Implementation note on `BoidCtx`.** `design/01_core.md` §8's illustrative `BoidCtx` bundles
//! a `&mut OcclusionScratch` and a `&mut f64` (`theta_out`) alongside read-only fields, then
//! reuses one `ctx` value across both `mode.desired(ctx, ...)` and `modifier.respond(ctx, ...)`
//! — which does not borrow-check as written (the doc's own §12.6 admits the borrow choreography
//! is elided). This implementation resolves it, as invited by §4's note that "the illustrative
//! snippets get updated when Phase 1–2 write the real code": `BoidCtx` holds only shared
//! references (`Copy`, reusable across both calls); `FlockingMode::desired` takes the mutable
//! occlusion scratch as its own parameter and returns `theta` as part of `SteerIntent` instead
//! of writing through a borrowed output slot.

use crate::boids::Species;
use crate::domain::Domain;
use crate::math::Vec3;
use crate::neighbor::Neighbor;
use crate::occlusion::OcclusionScratch;
use crate::params::CoreParams;
use crate::rng::Rng;

/// Everything a mode or modifier needs, borrowed for the read phase (immutable → parallel-safe,
/// `Copy` so the same value can be passed to both `desired()` and `respond()`).
#[derive(Clone, Copy)]
pub struct BoidCtx<'a> {
    pub index: u32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub species: Species,
    /// Distance-filtered candidates from the active `NeighborSelection`.
    pub neighbors: &'a [Neighbor],
    pub core_params: &'a CoreParams,
    pub domain: &'a dyn Domain,
}

/// `desired_v`: target velocity (direction × speed) — the input to the active
/// `SteeringModifier`. `extra_force`: an always-applied additive force that bypasses the
/// modifier entirely (e.g. Pearce's short-range steric repulsion — a collision-avoidance
/// reflex, not a "desire" a behavioural-inertia response model should be allowed to dampen or
/// delay). `theta`: this boid's internal opacity, if the mode computes one via `occlude()` —
/// `0.0` for a mode that doesn't (e.g. Vicsek).
pub struct SteerIntent {
    pub desired_v: Vec3,
    pub extra_force: Vec3,
    pub theta: f64,
}

pub trait FlockingMode: Send + Sync {
    /// What this boid wants to do — NOT yet clamped to `max_force`. The active
    /// `SteeringModifier` converts this into the boid's actual acceleration.
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        scratch: &mut OcclusionScratch,
        rng: &mut Rng,
    ) -> SteerIntent;
    fn name(&self) -> &'static str;
}

pub trait SteeringModifier: Send + Sync {
    /// Converts a `FlockingMode`'s desired velocity into this boid's actual acceleration.
    fn respond(&self, ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3;
    fn name(&self) -> &'static str;
}
