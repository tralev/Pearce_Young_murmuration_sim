//! `StepHook` — the single, minimal seam deferred feature-plugins (ecology, obstacles,
//! predator–prey, behaviours) attach to (design/02_plugins.md §5).

use crate::boids::BoidColumns;
use crate::math::Vec3;
use crate::modes::BoidCtx;
use crate::params::CoreParams;

/// View into simulation-level state a `StepHook`'s `pre_step` may read or adjust ahead of a
/// step: the (not-yet-mutated-this-step) boid columns — read-only, for a hook that needs
/// aggregate state (e.g. predator–prey caching prey centre of mass, all predator positions) —
/// plus mutable `core_params` (e.g. ecology's time-of-day update).
pub struct SimView<'a> {
    pub boids: &'a BoidColumns,
    pub core_params: &'a mut CoreParams,
    pub step_count: u64,
}

pub trait StepHook: Send + Sync {
    /// May read `sim.boids` and mutate `sim.core_params` ahead of this step.
    fn pre_step(&mut self, _sim: &mut SimView) {}
    /// An additive force contribution, applied after the active `SteeringModifier`. A hook
    /// that fully owns a species' behaviour (e.g. predator–prey's predator motion) may instead
    /// overwrite `*acc` outright for that species — see `murmur_predator` for the precedent.
    fn post_steer(&self, _ctx: BoidCtx<'_>, _acc: &mut Vec3) {}

    /// A per-boid speed-cap multiplier this hook wants enforced this step, if any — `None`
    /// means "no opinion." Fixes **G3** (roadmap.md §12): `SpeedModel::enforce` previously had
    /// no way to see any hook's own per-boid state at all. Read once per active boid in the
    /// write phase, right after `post_steer` (so a hook that computes its own state during
    /// `post_steer` can act on the same-step value, not last step's) — every hook's answer is
    /// combined via `min` (the most restrictive hook wins) before being passed to
    /// `SpeedModel::enforce`'s own `cap_multiplier` parameter. Default `None` — every `StepHook`
    /// before `murmur_boid_state_machine` has no opinion here, so this costs nothing for them.
    fn speed_cap_multiplier(&self, _index: u32) -> Option<f64> {
        None
    }

    fn name(&self) -> &'static str;

    /// Names of other `StepHook`s this one must run after (validated at construction — an
    /// unmet dependency or a cycle is a construction-time error). Falls back to registration
    /// order among hooks with no declared relationship.
    fn dependencies(&self) -> &[&'static str] {
        &[]
    }
}
