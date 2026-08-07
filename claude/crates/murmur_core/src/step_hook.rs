//! `StepHook` — the single, minimal seam deferred feature-plugins (ecology, obstacles,
//! predator–prey, behaviours) attach to (design/02_plugins.md §5).

use crate::boids::BoidColumns;
use crate::math::Vec3;
use crate::modes::BoidCtx;
use crate::params::CoreParams;
use crate::rng::Rng;

/// Per-boid checkpoint fields a `StepHook` may want to publish (design/05_viz_contract.md
/// §2.1: `state`, `speed_mult`, `threat_proximity`, `panic`, `blackening`, `spin`). All-`None`
/// by default — costs nothing for a hook with no opinion on any of them.
///
/// **Generic by construction, not a per-plugin special case in `murmur_core`.** Every hook
/// implements `StepHook::checkpoint_boid_fields` itself, returning whichever of these fields
/// it actually owns; `murmur_core`'s own checkpoint-capture code (`batch.rs`) just calls it on
/// every composed hook and merges the results — it never references a specific plugin by name
/// to do this, keeping the "infrastructure only" boundary (design/00_overview.md) intact.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoidCheckpointFields {
    /// Normal=0/Isolated=1/Crowded=2/Threatened=3 (`murmur_boid_state_machine`'s own
    /// `BoidState` ordering) — design/05's own literal wording.
    pub state: Option<u8>,
    pub speed_mult: Option<f32>,
    pub threat_proximity: Option<f32>,
    pub panic: Option<f32>,
    pub blackening: Option<f32>,
    pub spin: Option<f32>,
}

impl BoidCheckpointFields {
    /// Combines this hook's answer with another's — each field independently takes whichever
    /// side is `Some` (first-populated wins). A defensive merge, not real multi-hook
    /// arbitration: composition rules mean at most one hook is expected to populate any given
    /// field in practice (e.g. only `boid_state_machine` ever sets `state`).
    pub fn merge(self, other: BoidCheckpointFields) -> BoidCheckpointFields {
        BoidCheckpointFields {
            state: self.state.or(other.state),
            speed_mult: self.speed_mult.or(other.speed_mult),
            threat_proximity: self.threat_proximity.or(other.threat_proximity),
            panic: self.panic.or(other.panic),
            blackening: self.blackening.or(other.blackening),
            spin: self.spin.or(other.spin),
        }
    }
}

/// `murmur_ecology`'s published environment state (design/05_viz_contract.md §2.2
/// `Environment`) — the same 8 fields `murmur_ecology::EnvironmentState` itself computes,
/// mirrored here so `murmur_core` can carry them in a `Checkpoint` without depending on the
/// plugin crate (infrastructure never depends on a plugin, design/00_overview.md).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentSnapshot {
    pub day: u64,
    pub hour: f64,
    pub dusk_factor: f64,
    pub is_roosting_time: bool,
    pub is_murmuration_season: bool,
    pub coherence_factor: f64,
    pub temperature: f64,
    pub predator_active: bool,
}

/// `murmur_wander`'s published state (design/05_viz_contract.md §2.2 `WanderState`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WanderSnapshot {
    pub center: Vec3,
    pub heading: Vec3,
}

/// One of `murmur_ripple`'s three pulse trains — origin (the flock's own live centroid,
/// `murmur_ripple`'s own technique), current ring radius, and phase (fraction of `period`
/// elapsed since that train's most recent emission, in `[0, 1)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RippleTrainSnapshot {
    pub origin: Vec3,
    pub radius: f64,
    pub phase: f64,
}

/// `murmur_ripple`'s published state. **A disclosed reinterpretation of
/// design/05_viz_contract.md §2.2's `RippleState`**: the design names a single scalar
/// `envelope_sum: f32`, but `murmur_ripple`'s own `ripple_envelope_sum` is inherently
/// *per-boid* (it depends on each boid's own distance from the centroid) — there is no single
/// scene-wide number that means the same thing. Populated instead via the per-train
/// `[{origin, phase, radius}]` breakdown the same design paragraph explicitly allows as
/// optional, which *is* a well-defined scene-level quantity (three trains, each a closed-form
/// function of elapsed time). A per-boid `ripple_envelope_sum` is unaffected — it's read via
/// `murmur_ripple::Ripple::envelope_sum_of`, same as before, not part of this schema.
#[derive(Debug, Clone, PartialEq)]
pub struct RippleSnapshot {
    pub trains: Vec<RippleTrainSnapshot>,
}

/// One `murmur_obstacles` primitive, mirrored here so `murmur_core` can carry it in a
/// `Checkpoint` without depending on the plugin crate. Carries `axis` for `Cylinder` (a real
/// enrichment beyond design/05_viz_contract.md's own minimal `Cylinder{center,radius,
/// half_height}` shape — `murmur_obstacles` supports arbitrarily-oriented cylinders, not just
/// axis-aligned ones; growing the schema additively like this is the design doc's own §4
/// "deliberate, versioned struct extension" allowance).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObstaclePrimitiveSnapshot {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Box {
        center: Vec3,
        half_extent: Vec3,
    },
    Cylinder {
        center: Vec3,
        axis: Vec3,
        radius: f64,
        half_height: f64,
    },
}

/// design/05_viz_contract.md §2.2's own CSG op vocabulary — `murmur_obstacles`'s `Solid` maps
/// a base primitive (`Union`, `parent: None`) plus an optional `cut` primitive (`Subtract`,
/// `parent: Some(base's own index)`) onto this flat, parent-indexed node list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOp {
    Union,
    Subtract,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObstacleNodeSnapshot {
    pub primitive: ObstaclePrimitiveSnapshot,
    pub csg_op: CsgOp,
    /// Index into the same `Vec<ObstacleNodeSnapshot>` — `None` for a root-level (unioned)
    /// solid, `Some(i)` for a `cut` primitive subtracted from node `i`.
    pub parent: Option<u32>,
}

/// Scene-level checkpoint fields a `StepHook` may want to publish (design/05_viz_contract.md
/// §2.2: `environment`, `obstacles`, `wander`, `ripple`, `dynamic_vision_range`). All-empty by
/// default — costs nothing for a hook with no opinion on any of them. Same generic,
/// no-plugin-names-in-`murmur_core` construction as `BoidCheckpointFields`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneCheckpointFields {
    pub environment: Option<EnvironmentSnapshot>,
    pub obstacles: Vec<ObstacleNodeSnapshot>,
    pub wander: Option<WanderSnapshot>,
    pub ripple: Option<RippleSnapshot>,
    pub dynamic_vision_range: Option<f32>,
}

impl SceneCheckpointFields {
    /// Combines this hook's answer with another's — same first-populated-wins discipline as
    /// `BoidCheckpointFields::merge`. `obstacles` is a `Vec`, so "populated" means non-empty;
    /// concatenating rather than picking one side would be wrong if two different `obstacles`
    /// instances were ever composed at once (not possible today — `StepHook` composition has
    /// no uniqueness constraint per name, but two `"obstacles"` hooks would be a config
    /// mistake, not a real scene split), so the first non-empty list wins, same as every other
    /// field here.
    pub fn merge(self, other: SceneCheckpointFields) -> SceneCheckpointFields {
        SceneCheckpointFields {
            environment: self.environment.or(other.environment),
            obstacles: if self.obstacles.is_empty() {
                other.obstacles
            } else {
                self.obstacles
            },
            wander: self.wander.or(other.wander),
            ripple: self.ripple.or(other.ripple),
            dynamic_vision_range: self.dynamic_vision_range.or(other.dynamic_vision_range),
        }
    }
}

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
    ///
    /// `rng` fixes **G8** (roadmap.md §12): before this, no `StepHook` had any path to genuine,
    /// `base_seed`-tied randomness at all — `SpeedModel::enforce` was the only per-boid caller
    /// with an `Rng` in hand. Draws from the same single, sequential write-phase `Rng` stream
    /// `enforce`'s own unstall reseed already draws from (`rng::for_boid(base_seed,
    /// WRITE_PHASE_RNG_SALT, step_count)`, advanced once per active boid in fixed index order),
    /// not a fresh per-boid stream — deterministic and thread-count-independent by construction,
    /// since the write phase itself is already sequential, not parallel.
    fn post_steer(&self, _ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut Rng) {}

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

    /// This boid's own published checkpoint fields, if any (design/05_viz_contract.md §2.1) —
    /// read once per active boid when a checkpoint is captured (`batch.rs::capture_checkpoint`,
    /// after `post_steer`/`speed_cap_multiplier` have already run this step, so a hook reports
    /// the same-step value, not last step's). Default: no opinion on anything, the same
    /// zero-cost-for-hooks-that-don't-care shape `speed_cap_multiplier` already established.
    fn checkpoint_boid_fields(&self, _index: u32) -> BoidCheckpointFields {
        BoidCheckpointFields::default()
    }

    /// This hook's own published scene-level checkpoint fields, if any
    /// (design/05_viz_contract.md §2.2) — read once per checkpoint, not once per boid.
    fn checkpoint_scene_fields(&self) -> SceneCheckpointFields {
        SceneCheckpointFields::default()
    }
}
