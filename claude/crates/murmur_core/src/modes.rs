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

use crate::boids::{BoidColumns, Species};
use crate::domain::Domain;
use crate::math::Vec3;
use crate::neighbor::Neighbor;
use crate::occlusion::OcclusionScratch;
use crate::params::CoreParams;
use crate::rng::Rng;
use crate::step_hook::BoidCheckpointFields;

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
    /// The simulation's step counter *as of the start of this step* (fixes **G4**,
    /// roadmap.md §12: originally neither `FlockingMode::desired()` nor
    /// `SteeringModifier::respond()` had any way to know "when" they were being called —
    /// needed for time-varying targets like `murmur_influencer`'s Lissajous pursuit, and, as
    /// discovered building `murmur_spin_wave` (Phase 13), for any modifier holding persistent
    /// cross-boid state: without a step-boundary signal, such a modifier cannot tell "is this
    /// neighbour's stored state from last step, or already updated this step by another
    /// thread" — a real, thread-count-dependent correctness bug, not just a missing feature).
    pub step_count: u64,
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
    /// Runs once per step, sequentially, *before* the parallel read phase — the seam
    /// `desired()` itself doesn't have (**G7**, roadmap.md §12, found and fixed the same day
    /// building `murmur_young`, Phase 14: `desired()`'s `&self` + per-boid `BoidCtx` genuinely
    /// cannot see any *other* boid's state, and no earlier `FlockingMode` needed to — Pearce/
    /// Vicsek both compute everything from one boid's own `ctx.neighbors`). A mode that needs
    /// a real population-level aggregate (e.g. `murmur_young`'s H₂/m* curve, inherently a
    /// whole-flock quantity, not a per-boid one) precomputes it here into its own interior-
    /// mutable cache (a `Mutex`, the same discipline `murmur_spin_wave`/`murmur_predator_fsm`
    /// already use for cross-boid state under a `&self`-only method) and `desired()` reads the
    /// cache. Default no-op — every mode before `murmur_young` computed everything from its own
    /// `ctx.neighbors`, so this costs nothing for them, matching G4/G6's own "fix lazily, when
    /// actually needed" precedent (D22b).
    fn pre_step(&self, _boids: &BoidColumns, _step_count: u64) {}

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

    /// This boid's own published checkpoint fields, if any (design/05_viz_contract.md §2.1) —
    /// e.g. `murmur_spin_wave`'s own `s_z` as `spin`. `SteeringModifier` is a single-occupant
    /// socket (unlike `StepHook`'s `Vec`), so `batch.rs`'s checkpoint capture calls this once
    /// per boid on the one active modifier and merges it with every `StepHook`'s own answer —
    /// same `BoidCheckpointFields`/default-empty shape, same reasoning as `StepHook`'s own
    /// version of this method.
    fn checkpoint_boid_fields(&self, _index: u32) -> BoidCheckpointFields {
        BoidCheckpointFields::default()
    }
}
