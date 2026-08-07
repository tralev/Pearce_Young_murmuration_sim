//! The Track B batch/checkpoint/command contract (design/05_viz_contract.md, roadmap.md
//! Phase 10). Supersedes Track A's loop-only `Simulation::run_batch`/`run_batch_with_budget`
//! (pipeline.rs's module doc) with real periodic checkpoints and an atomically-applied command
//! queue, via [`Simulation::run_batch_checked`]/[`Simulation::run_batch_with_budget_checked`].
//!
//! **Scope.** Every field/command design/05_viz_contract.md §2/§3 names is represented here.
//! Core per-boid state, `murmur_predator`'s position/velocity (`predators`), `SetParam` over
//! the live-mutable subset of `CoreParams`, `Reset`, and `SetCheckpointStride` were always
//! real. The per-boid/scene-level fields belonging to `boid_state_machine`/`predator_fsm`/
//! `spin_wave`/`ecology`/`obstacles`/`wander`/`ripple`/`dynamic_vision_range` (`state`,
//! `speed_mult`, `threat_proximity`, `panic`, `blackening`, `spin`, `environment`,
//! `obstacles`, `wander`, `ripple`, `dynamic_vision_range`) are now wired too, generically via
//! `StepHook::checkpoint_boid_fields`/`checkpoint_scene_fields` (step_hook.rs) — `murmur_core`
//! never references a specific plugin by name to do this; each hook opts in on its own.
//! `SetEnvironment`/`AddObstacle`/`RemoveObstacle`'s write directions are now wired too,
//! generically via `StepHook::validate_command`/`apply_command` (`ecology`'s and
//! `murmur_obstacles`'s own implementations are the two real handlers — see `Command`'s own doc
//! for why each needed real design work, not a direct field overwrite). `RequestMetric` (native
//! H₂/`DensityScaling`/`ShapePCA`/`TauRho`) and `consensus_degree`/`h2_result` remain documented
//! no-ops/omitted fields — no native H₂-on-checkpoint path exists yet, a real, disclosed,
//! separately-scoped follow-up, not this pass's job — same "fix lazily" practice as
//! roadmap.md's G1–G8.

use crate::boids::Species;
use crate::math::Vec3;
use crate::metrics::Metrics;
use crate::params::CoreParams;
use crate::pipeline::Simulation;
use crate::step_hook::{
    BoidCheckpointFields, CsgOp, ObstaclePrimitiveSnapshot, SceneCheckpointFields,
};

/// Delivered once per `Simulation`, before the first checkpoint (design/05 §2.0). Everything
/// here is constant for the simulation's lifetime (composition is fixed at construction, D14).
#[derive(Debug, Clone)]
pub struct SessionHeader {
    pub session_id: u64,
    /// (socket, plugin_name) pairs — same shape as `Composition::plugin_names`.
    pub composition: Vec<(&'static str, String)>,
    pub boundary_name: String,
    /// A cheap config fingerprint (FNV-1a over plugin names + `CoreParams`) — not a real
    /// build/git hash (no build-info crate wired in); good enough to notice "these two
    /// checkpoints came from differently-configured simulations" in a test or log.
    pub build_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InterpolationHint {
    pub max_displacement: f64,
    /// Always `false` today — no `StepHook` yet tags "an eventful mutation happened this
    /// stride" (design/05 §2.2's own example is a predator capture, which Phase 8's minimal
    /// predator FSM doesn't model). Real event-tagging is the deferred `EventBuffer` future
    /// enhancement (deepseek_roadmap.md appendix) or Phase 15's richer predator FSM.
    pub state_changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoidSnapshot {
    pub position: Vec3,
    pub velocity: Vec3,
    pub species: Species,
    pub theta: f64,
    /// design/05_viz_contract.md §2.1's `state`/`speed_mult`/`threat_proximity`/`panic`/
    /// `blackening`/`spin` — collected generically from every composed `StepHook`'s own
    /// `checkpoint_boid_fields` (step_hook.rs), never a plugin-name special case here.
    pub checkpoint_fields: BoidCheckpointFields,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredatorSnapshot {
    pub position: Vec3,
    pub velocity: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub session_id: u64,
    pub step_count: u64,
    pub sim_time: f64,
    pub base_seed: u64,
    pub center_of_mass: Vec3,
    pub metrics: Metrics,
    pub interpolation_hint: InterpolationHint,
    pub boids: Vec<BoidSnapshot>,
    /// Empty if `predator` isn't part of this simulation's composition, or none are alive —
    /// same "empty, not absent" rule as every other optional group (design/05 §1).
    pub predators: Vec<PredatorSnapshot>,
    /// design/05_viz_contract.md §2.2's `environment`/`obstacles`/`wander`/`ripple`/
    /// `dynamic_vision_range` — collected generically from every composed `StepHook`'s own
    /// `checkpoint_scene_fields` (step_hook.rs), never a plugin-name special case here.
    pub scene_fields: SceneCheckpointFields,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckpointBuffer {
    pub checkpoints: Vec<Checkpoint>,
}

/// design/05_viz_contract.md §3. Every variant maps to a documented behaviour even when its
/// target plugin isn't part of the current composition — the design's own "no-op, not error"
/// rule. `SetEnvironment`/`AddObstacle`/`RemoveObstacle`'s write directions are all wired
/// (routed generically through `StepHook::validate_command`/`apply_command`, `ecology`'s and
/// `murmur_obstacles`'s own implementations are the two real handlers).
///
/// `SetEnvironment`: `ecology`'s own `EnvironmentState` is otherwise purely *derived* each step
/// from `step_count * dt`, so this works by setting a persistent time offset, not overwriting a
/// field that `pre_step` would just immediately recompute away.
///
/// `AddObstacle`/`RemoveObstacle` needed a real, previously-undisclosed prerequisite first:
/// design/05 §2.2's own `ObstacleNode` checkpoint shape didn't originally assign a *stable* id
/// to a placed obstacle (`parent` was a positional index into that one checkpoint's own flat
/// list, not a durable handle) — `ObstacleNodeSnapshot::id` (step_hook.rs) closes that, and
/// `murmur_obstacles::Obstacle` is the one plugin that actually tracks it stably across
/// checkpoints (a plain `ObstacleScene` built without the `Obstacle` wrapper still only reports
/// positional ids — see both types' own docs). This plugin's own 2-level CSG limit (one `cut`
/// per `Solid`, `murmur_obstacles`'s own module doc) means an `AddObstacle` with
/// `csg_op: Subtract` can only attach to a `parent` that doesn't already have a cut — rejected
/// as malformed otherwise, not silently overwritten.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    AddPredator {
        position: Vec3,
        velocity: Vec3,
    },
    RemovePredator {
        id: u32,
    },
    /// Routed generically to every composed `StepHook`'s own `validate_command`/`apply_command`
    /// — `murmur_obstacles::Obstacle` is the one real handler (see the enum's own doc). A no-op
    /// if `obstacles` isn't composed (design/05 §3's own "target not composed" rule).
    /// `csg_op: Union` adds a new root solid (`parent` must be `None`); `csg_op: Subtract`
    /// attaches `primitive` as an existing solid's `cut` (`parent` must be `Some(id)` of a
    /// solid that doesn't already have one).
    AddObstacle {
        primitive: ObstaclePrimitiveSnapshot,
        csg_op: CsgOp,
        parent: Option<u32>,
    },
    /// Removes a whole solid (its base and, if any, its own cut together) by the `id`
    /// `ObstacleNodeSnapshot::id` reports. A no-op if `obstacles` isn't composed; rejected as
    /// malformed if it is composed but no solid has this `id`.
    RemoveObstacle {
        id: u32,
    },
    SetParam {
        name: String,
        value: f64,
    },
    /// Routed generically to every composed `StepHook`'s own `validate_command`/`apply_command`
    /// — `ecology`'s own implementation is the one real handler (see the enum's own doc). A
    /// no-op if `ecology` isn't composed (design/05 §3's own "target not composed" rule).
    SetEnvironment {
        day: u64,
        hour: f64,
    },
    Reset {
        count: u32,
        seed: Option<u64>,
    },
    SetCheckpointStride {
        stride: u32,
    },
    /// Native H₂ is Track C Phase 14; `DensityScaling`/`ShapePCA`/`TauRho` are Python-only for
    /// v1 per design/05 §3. Always a documented no-op today, not a silent failure.
    RequestMetric,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommandError {
    pub index: usize,
    pub reason: String,
}

/// The `CoreParams` fields `SetParam` can target live — the subset that's a plain field on
/// `Simulation`'s already-stored `core_params` and safe to change between batches without
/// re-resolving any plugin. Plugin-private params (`phi_p`, `body_radius`, ...) are baked into
/// each plugin at construction (design/02_plugins.md §1) and have no live-mutation path yet —
/// a real limitation, not an oversight; `SetParam` targeting one is rejected as an unknown
/// name, same as a genuine typo, rather than silently doing nothing.
const LIVE_CORE_PARAM_NAMES: &[&str] = &[
    "cruise_speed",
    "max_force",
    "speed_min_factor",
    "dt",
    "vision_radius",
];

/// `Command::AddObstacle`'s own basic malformed-value check — every `Vec3`/`f64` field finite,
/// mirroring `AddPredator`'s own `position`/`velocity` finiteness check. Whether `csg_op`/
/// `parent` form a *valid combination*, and whether `parent` actually names a live solid, is
/// necessarily plugin-specific (only `murmur_obstacles::Obstacle` knows its own solids), so
/// that part is left to `StepHook::validate_command`.
fn primitive_is_finite(p: &ObstaclePrimitiveSnapshot) -> bool {
    match p {
        ObstaclePrimitiveSnapshot::Sphere { center, radius } => {
            center.is_finite() && radius.is_finite()
        }
        ObstaclePrimitiveSnapshot::Box {
            center,
            half_extent,
        } => center.is_finite() && half_extent.is_finite(),
        ObstaclePrimitiveSnapshot::Cylinder {
            center,
            axis,
            radius,
            half_height,
        } => {
            center.is_finite() && axis.is_finite() && radius.is_finite() && half_height.is_finite()
        }
    }
}

fn validate_core_param(name: &str, value: f64) -> Result<(), String> {
    if !LIVE_CORE_PARAM_NAMES.contains(&name) {
        return Err(format!(
            "unknown or not-yet-live-mutable param name: `{name}`"
        ));
    }
    if !value.is_finite() {
        return Err(format!("`{name}` value must be finite, got {value}"));
    }
    match name {
        "cruise_speed" | "max_force" | "vision_radius" if value <= 0.0 => {
            Err(format!("`{name}` must be > 0"))
        }
        "speed_min_factor" if !(0.0..=1.0).contains(&value) => {
            Err("`speed_min_factor` must be in [0, 1]".to_string())
        }
        "dt" if !(0.0..=crate::params::DT_MAX).contains(&value) => {
            Err(format!("`dt` must be in [0, {}]", crate::params::DT_MAX))
        }
        _ => Ok(()),
    }
}

fn apply_core_param(params: &mut CoreParams, name: &str, value: f64) {
    match name {
        "cruise_speed" => params.cruise_speed = value,
        "max_force" => params.max_force = value,
        "speed_min_factor" => params.speed_min_factor = value,
        "dt" => params.dt = value,
        "vision_radius" => params.vision_radius = value,
        _ => unreachable!("validate_core_param already rejected any other name"),
    }
}

/// FNV-1a over plugin names + `CoreParams`' fields — same algorithm as
/// `Simulation::state_hash`, just over config rather than boid state.
pub(crate) fn fingerprint(plugin_names: &[(&str, &str)], params: &CoreParams) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    for (socket, name) in plugin_names {
        mix(socket.as_bytes());
        mix(name.as_bytes());
    }
    for v in [
        params.cruise_speed,
        params.max_force,
        params.speed_min_factor,
        params.dt,
        params.vision_radius,
    ] {
        mix(&v.to_bits().to_le_bytes());
    }
    mix(&params.boid_count.to_le_bytes());
    hash
}

impl Simulation {
    /// Delivered once, before the first checkpoint (design/05 §2.0) — cheap to recompute on
    /// demand rather than cached, since composition never changes after construction (D14).
    pub fn session_header(&self) -> SessionHeader {
        SessionHeader {
            session_id: self.session_id,
            composition: self
                .plugin_names()
                .iter()
                .map(|&(socket, name)| (socket, name.to_string()))
                .collect(),
            boundary_name: self.plugin_names()[2].1.to_string(), // "domain" socket
            build_hash: self.build_hash,
        }
    }

    /// Validates a command queue against the current composition/state without mutating
    /// anything — every error found is collected (not just the first), matching design/05
    /// §3's "surfaced to the host as an error list."
    fn validate_commands(&self, commands: &[Command]) -> Vec<CommandError> {
        let mut errors = Vec::new();
        for (index, command) in commands.iter().enumerate() {
            let reason = match command {
                Command::SetParam { name, value } => validate_core_param(name, *value).err(),
                Command::SetCheckpointStride { stride } => {
                    if *stride == 0 {
                        Some("stride must be > 0".to_string())
                    } else {
                        None
                    }
                }
                Command::AddPredator { position, velocity } => {
                    if !position.is_finite() || !velocity.is_finite() {
                        Some("position/velocity must be finite".to_string())
                    } else {
                        None
                    }
                }
                Command::SetEnvironment { hour, .. } => {
                    if !hour.is_finite() {
                        Some("hour must be finite".to_string())
                    } else {
                        self.step_hooks
                            .iter()
                            .find_map(|h| h.validate_command(command))
                    }
                }
                Command::AddObstacle { primitive, .. } => {
                    if !primitive_is_finite(primitive) {
                        Some("primitive fields must be finite".to_string())
                    } else {
                        self.step_hooks
                            .iter()
                            .find_map(|h| h.validate_command(command))
                    }
                }
                Command::RemoveObstacle { .. } => self
                    .step_hooks
                    .iter()
                    .find_map(|h| h.validate_command(command)),
                Command::RemovePredator { .. } | Command::Reset { .. } | Command::RequestMetric => {
                    None
                }
            };
            if let Some(reason) = reason {
                errors.push(CommandError { index, reason });
            }
        }
        errors
    }

    /// Applies an already-validated command queue, in order. Never called with an unvalidated
    /// queue — `run_batch_checked`/`run_batch_with_budget_checked` are the only callers.
    fn apply_commands(&mut self, commands: Vec<Command>) {
        for command in commands {
            match command {
                Command::AddPredator { position, velocity } => {
                    // No-op if `predator` isn't composed (design/05 §3) — checked by name,
                    // since `step_hooks` doesn't otherwise expose which hook is which. Also a
                    // no-op if `BoidColumns` has no free slot — legitimate fixed-capacity
                    // behaviour, not the G6 bug it used to be: `SimConfig::spawn_headroom`
                    // (roadmap.md §12, G6 — fixed) lets a host reserve spawn capacity up front;
                    // a host that didn't request any headroom gets exactly this no-op instead.
                    if self.step_hooks.iter().any(|h| h.name() == "predator") {
                        let seed = self.next_boid_seed;
                        self.next_boid_seed += 1;
                        let _ = self.boids.add(position, velocity, Species::Predator, seed);
                    }
                }
                Command::RemovePredator { id } => {
                    let i = id as usize;
                    if i < self.boids.species.len()
                        && self.boids.active[i]
                        && self.boids.species[i] == Species::Predator
                    {
                        self.boids.remove(id);
                    }
                }
                Command::AddObstacle {
                    primitive,
                    csg_op,
                    parent,
                } => self.dispatch_to_hooks(Command::AddObstacle {
                    primitive,
                    csg_op,
                    parent,
                }),
                Command::RemoveObstacle { id } => {
                    self.dispatch_to_hooks(Command::RemoveObstacle { id })
                }
                Command::SetEnvironment { day, hour } => {
                    self.dispatch_to_hooks(Command::SetEnvironment { day, hour })
                }
                Command::SetParam { name, value } => {
                    apply_core_param(&mut self.core_params, &name, value);
                }
                Command::Reset { count, seed } => {
                    self.reset_flock(count, seed);
                }
                Command::SetCheckpointStride { stride } => {
                    self.checkpoint_stride = stride;
                }
                Command::RequestMetric => {
                    // Native H2/DensityScaling/ShapePCA/TauRho path doesn't exist yet
                    // (Phase 14) — always a documented no-op today (see module doc).
                }
            }
        }
    }

    /// Routes `cmd` to every composed `StepHook`'s own `apply_command` — no plugin-name
    /// special-casing here, a hook that isn't `cmd`'s intended target just does nothing
    /// (design/05 §3's own "target not composed" rule). Shared by `SetEnvironment`/
    /// `AddObstacle`/`RemoveObstacle`, the three commands with a real per-hook handler today.
    fn dispatch_to_hooks(&mut self, cmd: Command) {
        for hook in &mut self.step_hooks {
            hook.apply_command(&cmd, self.step_count, self.core_params.dt);
        }
    }

    /// `Command::Reset` (design/05 §3): reinitializes positions/velocities/count under the
    /// **same fixed composition** — does not touch which `Initializer` (or any other socket)
    /// is selected. `count` is clamped to the `BoidColumns` capacity fixed at construction;
    /// growing beyond it isn't possible without a new `Simulation` (capacity is fixed by
    /// design, not a Phase 10 limitation). Resets to an all-`Prey` flock — reintroducing a
    /// predator afterward is `AddPredator`'s job, since `Reset` takes no `predator_count`
    /// (matching the design's own command list, which has no such field).
    fn reset_flock(&mut self, count: u32, seed: Option<u64>) {
        for i in self.boids.iter_active().collect::<Vec<_>>() {
            self.boids.remove(i);
        }
        let count = count.min(self.boids.capacity());
        let seed = seed.unwrap_or(self.session_id);
        let mut place_rng = crate::rng::for_boid(seed, 0, 0);
        let (positions, velocities) = self.init.place(count, &self.core_params, &mut place_rng);
        for (i, (pos, vel)) in positions.into_iter().zip(velocities).enumerate() {
            self.boids
                .add(pos, vel, Species::Prey, i as u64)
                .expect("count was clamped to capacity above");
        }
        self.next_boid_seed = count as u64;
        self.step_count = 0;
        self.sim_time = 0.0;
        self.accum_max_displacement = 0.0;
        self.metrics = Metrics::collect(&self.boids, 0, self.core_params.cruise_speed);
    }

    fn center_of_mass(&self) -> Vec3 {
        let active: Vec<u32> = self.boids.iter_active().collect();
        if active.is_empty() {
            return Vec3::ZERO;
        }
        let sum = active
            .iter()
            .fold(Vec3::ZERO, |acc, &i| acc + self.boids.pos[i as usize]);
        sum / active.len() as f64
    }

    /// Every composed hook's own `checkpoint_boid_fields(index)`, plus the single active
    /// `SteeringModifier`'s own (e.g. `murmur_spin_wave`'s `spin`), merged in that order
    /// (`BoidCheckpointFields::merge`'s own first-populated-wins rule) — generic collection, no
    /// plugin-name special case. `pub` (not just `capture_checkpoint`'s own internal use):
    /// `murmur_py` reads this directly for its own per-step `Snapshot` extension, since design/
    /// 05_viz_contract.md §4's "Python path keeps the existing per-step... contract" framing
    /// means Python surfaces these fields as live per-step state, not by exposing `Checkpoint`
    /// objects the way the C-ABI batch/checkpoint contract does.
    pub fn checkpoint_boid_fields(&self, index: u32) -> BoidCheckpointFields {
        self.step_hooks
            .iter()
            .fold(BoidCheckpointFields::default(), |acc, hook| {
                acc.merge(hook.checkpoint_boid_fields(index))
            })
            .merge(self.modifier.checkpoint_boid_fields(index))
    }

    /// `checkpoint_boid_fields`, once per active boid, in the same `iter_active()` ascending-
    /// index order every other per-boid accessor (`positions`/`velocities`/`species`/
    /// `opacity`) already uses — lets a caller `zip` this against those directly.
    pub fn checkpoint_boid_fields_all(&self) -> Vec<BoidCheckpointFields> {
        self.boids
            .iter_active()
            .map(|i| self.checkpoint_boid_fields(i))
            .collect()
    }

    /// Every composed hook's own `checkpoint_scene_fields()`, merged in registration order —
    /// same generic collection as `checkpoint_boid_fields`, called once per checkpoint rather
    /// than once per boid. `pub` for the same reason `checkpoint_boid_fields` is.
    pub fn checkpoint_scene_fields(&self) -> SceneCheckpointFields {
        self.step_hooks
            .iter()
            .fold(SceneCheckpointFields::default(), |acc, hook| {
                acc.merge(hook.checkpoint_scene_fields())
            })
    }

    fn capture_checkpoint(&self, base_seed: u64) -> Checkpoint {
        let boids: Vec<BoidSnapshot> = self
            .boids
            .iter_active()
            .map(|i| BoidSnapshot {
                position: self.boids.pos[i as usize],
                velocity: self.boids.vel[i as usize],
                species: self.boids.species[i as usize],
                theta: self.boids.theta[i as usize],
                checkpoint_fields: self.checkpoint_boid_fields(i),
            })
            .collect();
        let predators: Vec<PredatorSnapshot> = self
            .boids
            .iter_active()
            .filter(|&i| self.boids.species[i as usize] == Species::Predator)
            .map(|i| PredatorSnapshot {
                position: self.boids.pos[i as usize],
                velocity: self.boids.vel[i as usize],
            })
            .collect();
        Checkpoint {
            session_id: self.session_id,
            step_count: self.step_count,
            sim_time: self.sim_time,
            base_seed,
            center_of_mass: self.center_of_mass(),
            metrics: self.metrics,
            interpolation_hint: InterpolationHint {
                max_displacement: self.accum_max_displacement,
                state_changed: false,
            },
            boids,
            predators,
            scene_fields: self.checkpoint_scene_fields(),
        }
    }

    /// Runs `steps` steps, capturing a checkpoint every `checkpoint_stride` steps (local to
    /// this call: the 1st, 2nd, ... `stride`-th step of *this batch*, not counted against the
    /// simulation's absolute `step_count` — so behaviour doesn't depend on how many steps a
    /// prior batch happened to run). Exactly `steps / stride` (integer division) checkpoints
    /// per call — "checkpoints captured at exactly `checkpoint_stride` intervals"
    /// (roadmap.md Phase 10 exit gate), not a ceiling/partial-final-checkpoint rule.
    fn run_steps(
        &mut self,
        steps: u32,
        base_seed: u64,
        time_budget_ms: Option<f64>,
    ) -> (CheckpointBuffer, bool) {
        let start = std::time::Instant::now();
        let mut buffer = CheckpointBuffer::default();
        let stride = self.checkpoint_stride.max(1);
        self.accum_max_displacement = 0.0;
        for local_step in 1..=steps {
            if let Some(budget) = time_budget_ms {
                if start.elapsed().as_secs_f64() * 1000.0 > budget {
                    return (buffer, false);
                }
            }
            self.step(self.core_params.dt, base_seed);
            if local_step % stride == 0 {
                buffer.checkpoints.push(self.capture_checkpoint(base_seed));
                self.accum_max_displacement = 0.0;
            }
        }
        (buffer, true)
    }

    /// The real Track B batch entry point (design/05_viz_contract.md, roadmap.md Phase 10) —
    /// distinct from Track A's `run_batch` (pipeline.rs), which stays as the older loop-only
    /// version rather than being removed out from under any existing caller. Validates
    /// `commands` atomically: on any invalid command, returns every error found and the
    /// simulation is left completely untouched (no partial application, no steps run).
    pub fn run_batch_checked(
        &mut self,
        steps: u32,
        base_seed: u64,
        commands: Vec<Command>,
    ) -> Result<CheckpointBuffer, Vec<CommandError>> {
        let errors = self.validate_commands(&commands);
        if !errors.is_empty() {
            return Err(errors);
        }
        self.apply_commands(commands);
        Ok(self.run_steps(steps, base_seed, None).0)
    }

    /// As `run_batch_checked`, but stops early if `time_budget_ms` elapses (checked between
    /// steps, not preemptively mid-step) — returns whatever checkpoints were captured so far
    /// plus a completion flag. State is never corrupted by an early stop: every step that ran
    /// completed in full before the budget check.
    pub fn run_batch_with_budget_checked(
        &mut self,
        steps: u32,
        base_seed: u64,
        commands: Vec<Command>,
        time_budget_ms: f64,
    ) -> Result<(CheckpointBuffer, bool), Vec<CommandError>> {
        let errors = self.validate_commands(&commands);
        if !errors.is_empty() {
            return Err(errors);
        }
        self.apply_commands(commands);
        Ok(self.run_steps(steps, base_seed, Some(time_budget_ms)))
    }
}

#[cfg(test)]
mod tests;
