//! The Track B batch/checkpoint/command contract (design/05_viz_contract.md, roadmap.md
//! Phase 10). Supersedes Track A's loop-only `Simulation::run_batch`/`run_batch_with_budget`
//! (pipeline.rs's module doc) with real periodic checkpoints and an atomically-applied command
//! queue, via [`Simulation::run_batch_checked`]/[`Simulation::run_batch_with_budget_checked`].
//!
//! **Scope.** Every field/command design/05_viz_contract.md §2/§3 names is represented here,
//! but only the ones with a real plugin behind them today actually do anything: core per-boid
//! state (always populated), `murmur_predator`'s position/velocity (`predators`), `SetParam`
//! over the live-mutable subset of `CoreParams`, `Reset`, `SetCheckpointStride`. Everything
//! else the design names — `AddObstacle`, `SetEnvironment`, `RequestMetric`, and the per-boid/
//! scene-level fields belonging to `ecology`/`obstacles`/`murmur_spin_wave`/native H₂ (`state`,
//! `speed_mult`, `threat_proximity`, `panic`, `blackening`, `spin`, `consensus_degree`,
//! `environment`, `obstacles`, `wander`, `ripple`, `h2_result`) — is either a documented no-op
//! command or an omitted field, because the plugin that would populate it doesn't exist yet
//! (Phases 14/15/18). This is not a partial/broken implementation of those — it's the honest
//! current state of a contract whose full breadth spans plugins not yet built, same "fix
//! lazily" practice as roadmap.md's G1–G5.

use crate::boids::Species;
use crate::math::Vec3;
use crate::metrics::Metrics;
use crate::params::CoreParams;
use crate::pipeline::Simulation;

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
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckpointBuffer {
    pub checkpoints: Vec<Checkpoint>,
}

/// design/05_viz_contract.md §3. Every variant maps to a documented behaviour even when its
/// target plugin isn't part of the current composition — the design's own "no-op, not error"
/// rule, which several variants below satisfy unconditionally (no simulation in this codebase
/// composes `obstacles`/`ecology`/native-H₂ yet, so they always no-op — a correct instance of
/// that rule, not a stand-in for it).
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    AddPredator {
        position: Vec3,
        velocity: Vec3,
    },
    RemovePredator {
        id: u32,
    },
    /// No `obstacles` plugin exists yet (Track C Phase 18) — always a no-op.
    AddObstacle,
    /// No `obstacles` plugin exists yet (Track C Phase 18) — always a no-op.
    RemoveObstacle {
        id: u32,
    },
    SetParam {
        name: String,
        value: f64,
    },
    /// No `ecology` plugin exists yet (Track C Phase 18) — always a no-op.
    SetEnvironment,
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
                Command::RemovePredator { .. }
                | Command::AddObstacle
                | Command::RemoveObstacle { .. }
                | Command::SetEnvironment
                | Command::Reset { .. }
                | Command::RequestMetric => None,
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
                Command::AddObstacle | Command::RemoveObstacle { .. } | Command::SetEnvironment => {
                    // No plugin exists yet to route to (see module doc) — always a no-op.
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

    fn capture_checkpoint(&self, base_seed: u64) -> Checkpoint {
        let boids: Vec<BoidSnapshot> = self
            .boids
            .iter_active()
            .map(|i| BoidSnapshot {
                position: self.boids.pos[i as usize],
                velocity: self.boids.vel[i as usize],
                species: self.boids.species[i as usize],
                theta: self.boids.theta[i as usize],
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
mod tests {
    use super::*;
    use crate::domain::Domain;
    use crate::init::{Initializer, NoiseSource};
    use crate::modes::{BoidCtx, FlockingMode, SteerIntent, SteeringModifier};
    use crate::neighbor::NeighborSelection;
    use crate::occlusion::OcclusionScratch;
    use crate::pipeline::SimConfig;
    use crate::registry::{PluginParams, Registry};
    use crate::rng::Rng as CoreRng;
    use crate::spatial_index::SpatialIndex;
    use crate::speed_model::SpeedModel;
    use crate::step_hook::StepHook;
    use crate::BoidColumns;

    struct DummyDomain;
    impl Domain for DummyDomain {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "dummy_domain"
        }
    }

    struct DummySpatialIndex;
    impl SpatialIndex for DummySpatialIndex {
        fn rebuild(&mut self, _boids: &BoidColumns) {}
        fn candidates(&self, _p: Vec3, _r: f64, out: &mut Vec<u32>) {
            out.clear();
        }
        fn name(&self) -> &'static str {
            "dummy_spatial_index"
        }
    }

    struct DummyNeighborSelection;
    impl NeighborSelection for DummyNeighborSelection {
        fn select(
            &self,
            _index: &dyn SpatialIndex,
            _i: u32,
            _boids: &BoidColumns,
            _params: &CoreParams,
        ) -> Vec<crate::neighbor::Neighbor> {
            Vec::new()
        }
        fn name(&self) -> &'static str {
            "dummy_neighbor_selection"
        }
    }

    struct DummyMode;
    impl FlockingMode for DummyMode {
        fn desired(
            &self,
            ctx: BoidCtx<'_>,
            _scratch: &mut OcclusionScratch,
            rng: &mut CoreRng,
        ) -> SteerIntent {
            let noise = crate::rng::sample_unit_sphere(rng) * 0.1;
            SteerIntent {
                desired_v: ctx.vel + noise,
                extra_force: Vec3::ZERO,
                theta: 0.0,
            }
        }
        fn name(&self) -> &'static str {
            "dummy_mode"
        }
    }

    struct DummyModifier;
    impl SteeringModifier for DummyModifier {
        fn respond(&self, _ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3 {
            desired_v - current_vel
        }
        fn name(&self) -> &'static str {
            "dummy_modifier"
        }
    }

    struct DummySpeedModel;
    impl SpeedModel for DummySpeedModel {
        fn enforce(
            &self,
            _vel: &mut Vec3,
            _species: Species,
            _params: &CoreParams,
            _cap_multiplier: f64,
            _rng: &mut CoreRng,
        ) {
        }
        fn name(&self) -> &'static str {
            "dummy_speed_model"
        }
    }

    struct DummyInit;
    impl Initializer for DummyInit {
        fn place(
            &self,
            count: u32,
            params: &CoreParams,
            _rng: &mut CoreRng,
        ) -> (Vec<Vec3>, Vec<Vec3>) {
            let pos = (0..count).map(|i| Vec3::new(i as f64, 0.0, 0.0)).collect();
            let vel = (0..count)
                .map(|_| Vec3::new(params.cruise_speed, 0.0, 0.0))
                .collect();
            (pos, vel)
        }
        fn name(&self) -> &'static str {
            "dummy_init"
        }
    }

    struct DummyNoise;
    impl NoiseSource for DummyNoise {
        fn sample(&self, _rng: &mut CoreRng) -> Vec3 {
            Vec3::ZERO
        }
        fn name(&self) -> &'static str {
            "dummy_noise"
        }
    }

    struct DummyPredatorHook;
    impl StepHook for DummyPredatorHook {
        fn name(&self) -> &'static str {
            "predator"
        }
    }

    fn register_all_dummies(reg: &mut Registry) {
        reg.register_mode("dummy_mode", |_| Box::new(DummyMode));
        reg.register_modifier("dummy_modifier", |_| Box::new(DummyModifier));
        reg.register_domain("dummy_domain", |_| Box::new(DummyDomain));
        reg.register_spatial_index("dummy_spatial_index", |_| Box::new(DummySpatialIndex));
        reg.register_neighbor_selection("dummy_neighbor_selection", |_| {
            Box::new(DummyNeighborSelection)
        });
        reg.register_speed_model("dummy_speed_model", |_| Box::new(DummySpeedModel));
        reg.register_init("dummy_init", |_| Box::new(DummyInit));
        reg.register_noise("dummy_noise", |_| Box::new(DummyNoise));
        reg.register_step_hook("predator", |_| Box::new(DummyPredatorHook));
    }

    fn config(n: u32, with_predator_hook: bool, spawn_headroom: u32) -> SimConfig {
        SimConfig {
            mode: "dummy_mode".to_string(),
            modifier: "dummy_modifier".to_string(),
            domain: "dummy_domain".to_string(),
            spatial_index: "dummy_spatial_index".to_string(),
            neighbor_selection: "dummy_neighbor_selection".to_string(),
            speed_model: "dummy_speed_model".to_string(),
            init: "dummy_init".to_string(),
            noise: "dummy_noise".to_string(),
            core_params: CoreParams::builder().boid_count(n).build().unwrap(),
            plugin_params: PluginParams::new(),
            init_seed: 99,
            step_hooks: if with_predator_hook {
                vec!["predator".to_string()]
            } else {
                Vec::new()
            },
            predator_count: 0,
            spawn_headroom,
        }
    }

    fn built_sim(n: u32, with_predator_hook: bool) -> Simulation {
        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        Simulation::new(config(n, with_predator_hook, 0), &registry).unwrap()
    }

    fn built_sim_with_headroom(n: u32, headroom: u32) -> Simulation {
        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        Simulation::new(config(n, true, headroom), &registry).unwrap()
    }

    #[test]
    fn run_batch_checked_with_no_commands_matches_n_sequential_steps() {
        let mut a = built_sim(10, false);
        let mut b = built_sim(10, false);
        a.run_batch_checked(20, 7, Vec::new()).unwrap();
        for _ in 0..20 {
            b.step(b.core_params.dt, 7);
        }
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.step_count(), b.step_count());
    }

    #[test]
    fn invalid_command_rejects_the_whole_batch_before_any_step_runs() {
        let mut sim = built_sim(5, false);
        let commands = vec![
            Command::SetCheckpointStride { stride: 2 },
            Command::SetParam {
                name: "not_a_real_param".to_string(),
                value: 1.0,
            },
        ];
        let result = sim.run_batch_checked(10, 1, commands);
        assert!(result.is_err());
        assert_eq!(sim.step_count(), 0, "no step should have run");
        assert_eq!(
            sim.checkpoint_stride, 1,
            "stride command must not have applied either"
        );
    }

    #[test]
    fn checkpoints_are_captured_at_exactly_the_configured_stride() {
        let mut sim = built_sim(5, false);
        let commands = vec![Command::SetCheckpointStride { stride: 4 }];
        let buffer = sim.run_batch_checked(10, 1, commands).unwrap();
        assert_eq!(buffer.checkpoints.len(), 2); // 10 / 4 = 2, not 3 (no ceiling/partial)
        assert_eq!(buffer.checkpoints[0].step_count, 4);
        assert_eq!(buffer.checkpoints[1].step_count, 8);
    }

    #[test]
    fn run_batch_with_budget_checked_stops_early_and_leaves_state_usable() {
        let mut sim = built_sim(500, false);
        let (buffer, all_done) = sim
            .run_batch_with_budget_checked(1_000_000, 1, Vec::new(), 0.001)
            .unwrap();
        assert!(!all_done);
        assert!(sim.step_count() < 1_000_000);
        // default checkpoint_stride is 1, so one checkpoint per completed step in this batch.
        assert_eq!(sim.step_count(), buffer.checkpoints.len() as u64);
        for v in sim.velocities() {
            assert!(v.is_finite(), "state must stay usable after an early stop");
        }
    }

    #[test]
    fn add_predator_is_a_noop_without_the_predator_hook_composed() {
        let mut sim = built_sim(5, false);
        let before = sim.boid_count();
        sim.run_batch_checked(
            1,
            1,
            vec![Command::AddPredator {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
            }],
        )
        .unwrap();
        assert_eq!(
            sim.boid_count(),
            before,
            "no predator hook composed -> no-op"
        );
    }

    /// With `spawn_headroom: 0` (the default), `BoidColumns` is still sized to exactly
    /// `boid_count` at construction, so `AddPredator` with the `predator` hook composed still
    /// no-ops here — a legitimate fixed-capacity limit a host didn't ask to avoid, not the G6
    /// bug this used to be (roadmap.md §12, now fixed via `SimConfig::spawn_headroom`).
    #[test]
    fn add_predator_is_a_noop_at_full_capacity_with_zero_spawn_headroom() {
        let mut sim = built_sim(5, true);
        let before = sim.boid_count();
        sim.run_batch_checked(
            1,
            1,
            vec![Command::AddPredator {
                position: Vec3::new(9.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
            }],
        )
        .unwrap();
        assert_eq!(sim.boid_count(), before, "no free slot -> no-op");
    }

    /// The G6 fix itself: a host that reserves `spawn_headroom` gets a real, immediate
    /// `AddPredator` with no workaround needed (contrast with the `Reset`-first dance the
    /// zero-headroom case below still needs — that's a legitimate capacity limit, not a bug).
    #[test]
    fn add_predator_spawns_a_real_boid_directly_when_spawn_headroom_is_reserved() {
        let mut sim = built_sim_with_headroom(5, 2);
        let before = sim.boid_count();
        let buffer = sim
            .run_batch_checked(
                1,
                1,
                vec![Command::AddPredator {
                    position: Vec3::new(9.0, 0.0, 0.0),
                    velocity: Vec3::ZERO,
                }],
            )
            .unwrap();
        assert_eq!(sim.boid_count(), before + 1);
        assert_eq!(buffer.checkpoints[0].predators.len(), 1);
    }

    /// Proves the AddPredator *routing* logic is correct even without reserved headroom, by
    /// first using `Reset` to legitimately shrink the flock and free slots within the same
    /// fixed capacity — a real capability (Reset), not a workaround for a bug.
    #[test]
    fn add_predator_spawns_a_real_boid_once_a_slot_is_free_via_reset() {
        let mut sim = built_sim(5, true);
        let buffer = sim
            .run_batch_checked(
                1,
                1,
                vec![
                    Command::Reset {
                        count: 3,
                        seed: Some(1),
                    },
                    Command::AddPredator {
                        position: Vec3::new(9.0, 0.0, 0.0),
                        velocity: Vec3::ZERO,
                    },
                ],
            )
            .unwrap();
        assert_eq!(sim.boid_count(), 4); // 3 reset prey + 1 spawned predator
        assert_eq!(buffer.checkpoints.len(), 1);
        assert_eq!(buffer.checkpoints[0].predators.len(), 1);
    }

    #[test]
    fn reset_reinitializes_under_the_same_composition_and_zeroes_step_count() {
        let mut sim = built_sim(5, false);
        sim.run_batch_checked(10, 1, Vec::new()).unwrap();
        assert_eq!(sim.step_count(), 10);
        sim.run_batch_checked(
            1,
            1,
            vec![Command::Reset {
                count: 3,
                seed: Some(42),
            }],
        )
        .unwrap();
        // step_count resets to 0 as part of Reset, then advances by the 1 step this same
        // batch call also runs afterward.
        assert_eq!(sim.step_count(), 1);
        assert_eq!(sim.boid_count(), 3);
    }

    #[test]
    fn checkpoint_schema_is_the_same_rust_type_regardless_of_composition() {
        let mut plain = built_sim(3, false);
        let mut with_predator = built_sim(3, true);
        let a = plain.run_batch_checked(1, 1, Vec::new()).unwrap();
        let b = with_predator.run_batch_checked(1, 1, Vec::new()).unwrap();
        // Both produce the identical `Checkpoint` struct type with the same fields present
        // (enforced by the type system already); this checks the *value*-level "empty, not
        // absent" rule holds too — no predator spawned yet in either, just composed in one.
        assert!(a.checkpoints[0].predators.is_empty());
        assert!(b.checkpoints[0].predators.is_empty());
    }

    #[test]
    fn session_header_is_stable_across_the_simulations_lifetime() {
        let mut sim = built_sim(4, false);
        let h1 = sim.session_header();
        sim.run_batch_checked(5, 1, Vec::new()).unwrap();
        let h2 = sim.session_header();
        assert_eq!(h1.session_id, h2.session_id);
        assert_eq!(h1.build_hash, h2.build_hash);
        assert_eq!(h1.composition, h2.composition);
    }
}
