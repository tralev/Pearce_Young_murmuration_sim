//! `Simulation` construction and the `step()`/`run_batch()` pipeline (design/01_core.md §8,
//! §12.6). Plugin composition is fixed at construction and never changes for the
//! `Simulation`'s lifetime (design/00_overview.md "Interaction model").
//!
//! **Scope note (Track A vs. Track B).** `run_batch`/`run_batch_with_budget` here are the
//! *Track A* minimal versions — they just loop `step()`, with no `CheckpointBuffer` or
//! `Command` queue. Kept as-is rather than removed, so no existing caller breaks. The real
//! Track B batch contract (periodic checkpoints, atomically applied commands —
//! design/05_viz_contract.md, roadmap.md Phase 10) is `batch.rs`'s
//! `Simulation::run_batch_checked`/`run_batch_with_budget_checked`.

use rayon::prelude::*;

use crate::boids::{BoidColumns, Species};
use crate::domain::Domain;
use crate::error::ConfigError;
use crate::init::{Initializer, NoiseSource};
use crate::math::Vec3;
use crate::metrics::Metrics;
use crate::modes::{BoidCtx, FlockingMode, SteeringModifier};
use crate::neighbor::NeighborSelection;
use crate::occlusion::OcclusionScratch;
use crate::params::{CoreParams, DT_MAX};
use crate::registry::{PluginParams, Registry};
use crate::rng::{self, sample_unit_sphere};
use crate::spatial_index::SpatialIndex;
use crate::speed_model::SpeedModel;
use crate::step_hook::{SimView, StepHook};

/// A distinct, fixed "boid seed" used only to derive the write-phase RNG (the per-step
/// unstall reseed) — never a real boid's own seed (those are assigned densely from `0`), so
/// there's no risk of accidentally correlating with any boid's read-phase draw.
const WRITE_PHASE_RNG_SALT: u64 = u64::MAX;

/// Which plugin fills each socket, by name, plus the params needed to construct a
/// `Simulation`. Names are owned `String`s, not `&'static str`: composition selection is a
/// runtime value that flows across the Python/FFI boundary (design/02_plugins.md §1 — "Python
/// selects via strings"), so it can't be a compile-time constant here even though `Registry`'s
/// own factory keys are.
pub struct SimConfig {
    pub mode: String,
    pub modifier: String,
    pub domain: String,
    pub spatial_index: String,
    pub neighbor_selection: String,
    pub speed_model: String,
    pub init: String,
    pub noise: String,
    pub core_params: CoreParams,
    /// Shared plugin-config blob passed to every factory (design/02_plugins.md §1). A
    /// per-plugin split (a dotted-path namespace, design/01_core.md §4.2) is a Phase 5+
    /// concern once real plugins have their own private params to fill from it.
    pub plugin_params: PluginParams,
    /// Seeds the one-time RNG draw used to place the initial flock.
    pub init_seed: u64,
    /// `StepHook` plugin names to activate, in registration order — empty by default (none
    /// of the slice's default composition uses one; predator–prey, Phase 8, is the first).
    /// Execution order follows this `Vec`'s order (design/02_plugins.md §5: "falls back to
    /// registration order" — with only ever one real hook so far, no dependency-graph
    /// resolution has been needed yet; `StepHook::dependencies()` is defined for when it is).
    pub step_hooks: Vec<String>,
    /// How many of the `Initializer`-placed boids are `Species::Predator` instead of
    /// `Species::Prey` (the last `predator_count` of them, by placement order) — `0` by
    /// default. A minimal way to get predators into a simulation for Phase 8's proof that the
    /// `StepHook` seam works; dynamically adding a predator mid-run is Track B's `AddPredator`
    /// command (design/05_viz_contract.md §3, roadmap.md Phase 10).
    pub predator_count: u32,
    /// Extra `BoidColumns` capacity beyond `core_params.boid_count`, reserved at construction
    /// for runtime-spawned boids (`Command::AddPredator`, batch.rs). `0` by default — a
    /// simulation that never expects live spawning pays nothing for this. Fixes **G6**
    /// (roadmap.md §12): before this field existed, `BoidColumns` was always sized to exactly
    /// the initial placement count, so `AddPredator` could never succeed even with `predator`
    /// composed — there was never a free slot. Capacity is still fixed for the simulation's
    /// lifetime (design/00_overview.md's storage invariant) — this just lets a host that knows
    /// it wants live spawning reserve room for it up front, rather than growing unboundedly.
    pub spawn_headroom: u32,
}

pub struct Simulation {
    pub(crate) boids: BoidColumns,
    pub(crate) core_params: CoreParams,
    pub(crate) domain: Box<dyn Domain>,
    pub(crate) mode: Box<dyn FlockingMode>,
    pub(crate) modifier: Box<dyn SteeringModifier>,
    pub(crate) speed_model: Box<dyn SpeedModel>,
    pub(crate) spatial_index: Box<dyn SpatialIndex>,
    pub(crate) neighbor_selection: Box<dyn NeighborSelection>,
    #[allow(dead_code)] // held for composition/reproducibility; not yet called (no mode uses
    // the standalone NoiseSource trait object internally — Pearce draws its own noise inline)
    pub(crate) noise: Box<dyn NoiseSource>,
    /// Held (not just consumed at construction) so `Command::Reset` (batch.rs, roadmap.md
    /// Phase 10) can re-place the flock under the same fixed composition without needing the
    /// `Registry` again.
    pub(crate) init: Box<dyn Initializer>,
    pub(crate) step_hooks: Vec<Box<dyn StepHook>>,
    pub(crate) step_count: u64,
    pub(crate) sim_time: f64,
    pub(crate) metrics: Metrics,
    mode_name: String,
    modifier_name: String,
    domain_name: String,
    spatial_index_name: String,
    neighbor_selection_name: String,
    speed_model_name: String,
    init_name: String,
    noise_name: String,
    /// batch.rs state — see that module for why each exists.
    pub(crate) session_id: u64,
    pub(crate) build_hash: u64,
    pub(crate) checkpoint_stride: u32,
    pub(crate) next_boid_seed: u64,
    pub(crate) accum_max_displacement: f64,
}

/// Active plugin names per socket, plus the resolved `CoreParams` — the macro→micro
/// introspection seam (design/04_micro_macro.md §4). A simplified `Composition`: it does not
/// yet carry every plugin's own erased private-params struct (design's full
/// `plugin_params: &[(&str, &dyn ErasedPluginParams)]`) — that's added when a consumer
/// (Phase 6's Python bindings) actually needs per-plugin param introspection, not invented
/// speculatively here.
pub struct Composition<'a> {
    pub plugin_names: [(&'static str, &'a str); 8],
    pub core_params: &'a CoreParams,
}

impl Simulation {
    /// Resolves every socket's plugin by name from `registry`, places the initial flock via
    /// the chosen `Initializer`, and returns the constructed `Simulation` — or the first
    /// `ConfigError` encountered (an unregistered name, including every name against an empty
    /// `Registry`).
    pub fn new(config: SimConfig, registry: &Registry) -> Result<Self, ConfigError> {
        let mode = registry.resolve_mode(&config.mode, &config.plugin_params)?;
        let modifier = registry.resolve_modifier(&config.modifier, &config.plugin_params)?;
        let domain = registry.resolve_domain(&config.domain, &config.plugin_params)?;
        let spatial_index =
            registry.resolve_spatial_index(&config.spatial_index, &config.plugin_params)?;
        let neighbor_selection = registry
            .resolve_neighbor_selection(&config.neighbor_selection, &config.plugin_params)?;
        let speed_model =
            registry.resolve_speed_model(&config.speed_model, &config.plugin_params)?;
        let init = registry.resolve_init(&config.init, &config.plugin_params)?;
        let noise = registry.resolve_noise(&config.noise, &config.plugin_params)?;
        let mut step_hooks = Vec::with_capacity(config.step_hooks.len());
        for name in &config.step_hooks {
            step_hooks.push(registry.resolve_step_hook(name, &config.plugin_params)?);
        }

        let mut boids = BoidColumns::with_capacity(
            config
                .core_params
                .boid_count
                .saturating_add(config.spawn_headroom),
        );
        let mut place_rng = rng::for_boid(config.init_seed, 0, 0);
        let (positions, velocities) = init.place(
            config.core_params.boid_count,
            &config.core_params,
            &mut place_rng,
        );
        let n = config.core_params.boid_count;
        let predator_count = config.predator_count.min(n);
        for (i, (pos, vel)) in positions.into_iter().zip(velocities).enumerate() {
            let species = if n - (i as u32) <= predator_count {
                Species::Predator
            } else {
                Species::Prey
            };
            boids
                .add(pos, vel, species, i as u64)
                .expect("BoidColumns sized to boid_count by construction");
        }

        let metrics = Metrics::collect(&boids, 0, config.core_params.cruise_speed);
        let plugin_names_for_hash = [
            ("mode", config.mode.as_str()),
            ("modifier", config.modifier.as_str()),
            ("domain", config.domain.as_str()),
            ("spatial_index", config.spatial_index.as_str()),
            ("neighbor_selection", config.neighbor_selection.as_str()),
            ("speed_model", config.speed_model.as_str()),
            ("init", config.init.as_str()),
            ("noise", config.noise.as_str()),
        ];
        let build_hash = crate::batch::fingerprint(&plugin_names_for_hash, &config.core_params);

        Ok(Simulation {
            boids,
            core_params: config.core_params,
            domain,
            mode,
            modifier,
            speed_model,
            spatial_index,
            neighbor_selection,
            noise,
            init,
            step_hooks,
            step_count: 0,
            sim_time: 0.0,
            metrics,
            mode_name: config.mode,
            modifier_name: config.modifier,
            domain_name: config.domain,
            spatial_index_name: config.spatial_index,
            neighbor_selection_name: config.neighbor_selection,
            speed_model_name: config.speed_model,
            init_name: config.init,
            noise_name: config.noise,
            session_id: config.init_seed,
            build_hash,
            checkpoint_stride: 1,
            next_boid_seed: n as u64,
            accum_max_displacement: 0.0,
        })
    }

    pub fn boid_count(&self) -> u32 {
        self.boids.active_count()
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Owned snapshots of active boids' state, in ascending stable-index order. A minimal
    /// placeholder for inspection/testing ahead of Phase 6's real output contract (zero-copy
    /// numpy views over the same columns, design/03_observables_bindings.md §2.1).
    pub fn positions(&self) -> Vec<Vec3> {
        self.boids
            .iter_active()
            .map(|i| self.boids.pos[i as usize])
            .collect()
    }
    pub fn velocities(&self) -> Vec<Vec3> {
        self.boids
            .iter_active()
            .map(|i| self.boids.vel[i as usize])
            .collect()
    }
    pub fn opacity(&self) -> Vec<f64> {
        self.boids
            .iter_active()
            .map(|i| self.boids.theta[i as usize])
            .collect()
    }
    pub fn species(&self) -> Vec<Species> {
        self.boids
            .iter_active()
            .map(|i| self.boids.species[i as usize])
            .collect()
    }

    /// Active plugin names per socket — the macro→micro introspection seam (design/04 §4).
    pub fn plugin_names(&self) -> [(&'static str, &str); 8] {
        [
            ("mode", self.mode_name.as_str()),
            ("modifier", self.modifier_name.as_str()),
            ("domain", self.domain_name.as_str()),
            ("spatial_index", self.spatial_index_name.as_str()),
            ("neighbor_selection", self.neighbor_selection_name.as_str()),
            ("speed_model", self.speed_model_name.as_str()),
            ("init", self.init_name.as_str()),
            ("noise", self.noise_name.as_str()),
        ]
    }

    pub fn composition(&self) -> Composition<'_> {
        Composition {
            plugin_names: self.plugin_names(),
            core_params: &self.core_params,
        }
    }

    /// Advances the simulation by one step (design/01_core.md §8's pipeline). `dt` is clamped
    /// to `[0, DT_MAX]`; `base_seed` is the sole external randomness source — each boid's RNG
    /// is a pure function of `(base_seed, boid.seed, step_count)`, so the parallel read phase
    /// is independent of thread count and scheduling order.
    pub fn step(&mut self, dt: f64, base_seed: u64) {
        let dt = dt.clamp(0.0, DT_MAX);
        self.spatial_index.rebuild(&self.boids);
        self.run_pre_step_hooks();
        self.read_phase(base_seed);
        self.write_phase(dt, base_seed);
        self.metrics =
            Metrics::collect(&self.boids, self.step_count, self.core_params.cruise_speed);
        self.step_count += 1;
        self.sim_time += dt;
    }

    /// Runs every `StepHook::pre_step` in registration order, ahead of the read phase — a
    /// no-op loop until Phase 8's predator–prey (the first `StepHook` plugin) populates
    /// `step_hooks`. Each hook gets a `SimView` borrowing the (unmutated-this-step) boid
    /// columns plus mutable `core_params`, so a hook can cache whatever aggregate state it
    /// needs (e.g. predator positions, prey centre of mass) or tune a param ahead of this
    /// step (e.g. ecology's time-of-day update, a later plugin).
    fn run_pre_step_hooks(&mut self) {
        let boids = &self.boids;
        let step_count = self.step_count;
        for hook in &mut self.step_hooks {
            let mut view = SimView {
                boids,
                core_params: &mut self.core_params,
                step_count,
            };
            hook.pre_step(&mut view);
        }
    }

    /// Read phase: parallel per boid (rayon), no mutation of `pos`/`vel`. Writes only into
    /// `acc`/`theta`, which are detached from `self.boids` for the duration (via
    /// `mem::take`) so the closure can hold `&self.boids` (for neighbour gathering) alongside
    /// disjoint local `&mut` slices — the "split_at_mut/index-partition" pattern design/01
    /// §12.6 describes, applied at the field level since `acc`/`theta` are whole separate
    /// columns.
    fn read_phase(&mut self, base_seed: u64) {
        let mut acc = std::mem::take(&mut self.boids.acc);
        let mut theta = std::mem::take(&mut self.boids.theta);

        let boids = &self.boids;
        let spatial_index: &dyn SpatialIndex = &*self.spatial_index;
        let neighbor_selection: &dyn NeighborSelection = &*self.neighbor_selection;
        let mode: &dyn FlockingMode = &*self.mode;
        let modifier: &dyn SteeringModifier = &*self.modifier;
        let domain: &dyn Domain = &*self.domain;
        let core_params = &self.core_params;
        let step_count = self.step_count;

        acc.par_iter_mut()
            .zip(theta.par_iter_mut())
            .enumerate()
            .for_each_init(
                OcclusionScratch::default,
                |scratch, (i, (acc_i, theta_i))| {
                    if !boids.active[i] {
                        *acc_i = Vec3::ZERO;
                        *theta_i = 0.0;
                        return;
                    }
                    let idx = i as u32;
                    let mut rng = rng::for_boid(base_seed, boids.seed[i], step_count);
                    let neighbors =
                        neighbor_selection.select(spatial_index, idx, boids, core_params);
                    let ctx = BoidCtx {
                        index: idx,
                        pos: boids.pos[i],
                        vel: boids.vel[i],
                        species: boids.species[i],
                        neighbors: &neighbors,
                        core_params,
                        domain,
                        step_count,
                    };
                    let intent = mode.desired(ctx, scratch, &mut rng);
                    let response = modifier.respond(ctx, intent.desired_v, boids.vel[i]);
                    *acc_i = response + intent.extra_force;
                    *theta_i = intent.theta;
                },
            );

        self.boids.acc = acc;
        self.boids.theta = theta;
    }

    /// Write phase: sequential, fixed index order (determinism). `StepHook::post_steer` (an
    /// additive force layered on top of `acc` before it's consumed) runs first — a no-op loop
    /// in this phase's default composition (`step_hooks` is empty until Phase 8's
    /// predator–prey), then Band-style speed enforcement, then integration, then
    /// `Domain::apply`, then a finite-position safety net.
    fn write_phase(&mut self, dt: f64, base_seed: u64) {
        let mut write_rng = rng::for_boid(base_seed, WRITE_PHASE_RNG_SALT, self.step_count);
        let vmin = self.core_params.cruise_speed * self.core_params.speed_min_factor;

        for i in self.boids.iter_active().collect::<Vec<_>>() {
            let idx = i as usize;

            if !self.step_hooks.is_empty() {
                let domain: &dyn Domain = &*self.domain;
                let ctx = BoidCtx {
                    index: i,
                    pos: self.boids.pos[idx],
                    vel: self.boids.vel[idx],
                    species: self.boids.species[idx],
                    neighbors: &[], // see StepHook trait doc: full neighbour wiring lands with
                    // the first real StepHook plugin (Phase 8), which doesn't need them anyway
                    // (predator-prey reads global predator state, not local neighbours).
                    core_params: &self.core_params,
                    step_count: self.step_count,
                    domain,
                };
                let mut acc_i = self.boids.acc[idx];
                for hook in &self.step_hooks {
                    hook.post_steer(ctx, &mut acc_i);
                }
                self.boids.acc[idx] = acc_i;
            }

            let pos_before = self.boids.pos[idx];

            self.boids.vel[idx] += self.boids.acc[idx] * dt;
            self.speed_model.enforce(
                &mut self.boids.vel[idx],
                self.boids.species[idx],
                &self.core_params,
                &mut write_rng,
            );
            self.boids.pos[idx] += self.boids.vel[idx] * dt;
            self.domain
                .apply(&mut self.boids.pos[idx], &mut self.boids.vel[idx], dt);

            if !self.boids.pos[idx].is_finite() {
                self.boids.pos[idx] = Vec3::ZERO;
                self.boids.vel[idx] = sample_unit_sphere(&mut write_rng) * vmin;
            }
            self.boids.acc[idx] = Vec3::ZERO;

            // batch.rs's `InterpolationHint::max_displacement` — a running max across every
            // step since the last checkpoint capture, reset there (design/05 §2.2).
            let displacement = (self.boids.pos[idx] - pos_before).len();
            if displacement > self.accum_max_displacement {
                self.accum_max_displacement = displacement;
            }
        }
    }

    /// Track A minimal batch loop — see module doc. Runs `steps` iterations of `step()` back
    /// to back at `core_params.dt`.
    pub fn run_batch(&mut self, steps: u32, base_seed: u64) {
        for _ in 0..steps {
            self.step(self.core_params.dt, base_seed);
        }
    }

    /// Runs steps until either `steps` complete or `time_budget_ms` elapses (checked between
    /// steps, not preemptively mid-step). Returns `(steps_completed, all_requested_steps_ran)`.
    pub fn run_batch_with_budget(
        &mut self,
        steps: u32,
        base_seed: u64,
        time_budget_ms: f64,
    ) -> (u32, bool) {
        let start = std::time::Instant::now();
        for completed in 0..steps {
            if start.elapsed().as_secs_f64() * 1000.0 > time_budget_ms {
                return (completed, false);
            }
            self.step(self.core_params.dt, base_seed);
        }
        (steps, true)
    }

    /// A deterministic fingerprint of the active boids' positions and velocities — used to
    /// prove thread-count-independence (same `state_hash` regardless of the rayon pool size)
    /// and, later, for golden-trajectory regression tests. FNV-1a over f64 bit patterns in
    /// fixed (ascending) index order — not cryptographic, just a cheap regression fingerprint.
    pub fn state_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        for i in self.boids.iter_active() {
            for v in [self.boids.pos[i as usize], self.boids.vel[i as usize]] {
                mix(&v.x.to_bits().to_le_bytes());
                mix(&v.y.to_bits().to_le_bytes());
                mix(&v.z.to_bits().to_le_bytes());
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::Initializer;
    use crate::math::Vec3;
    use crate::modes::{BoidCtx, SteerIntent};
    use crate::occlusion::OcclusionScratch;
    use crate::rng::Rng as CoreRng;

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
            // A small rng-driven wobble on top of the current heading: enough that positions
            // actually change step over step (a meaningful thing for the determinism /
            // no-NaN pipeline tests to check), while staying a trivial stand-in mode — no
            // occlusion, no real flocking rule.
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
    }

    fn dummy_config() -> SimConfig {
        SimConfig {
            mode: "dummy_mode".to_string(),
            modifier: "dummy_modifier".to_string(),
            domain: "dummy_domain".to_string(),
            spatial_index: "dummy_spatial_index".to_string(),
            neighbor_selection: "dummy_neighbor_selection".to_string(),
            speed_model: "dummy_speed_model".to_string(),
            init: "dummy_init".to_string(),
            noise: "dummy_noise".to_string(),
            core_params: CoreParams::builder().boid_count(5).build().unwrap(),
            plugin_params: PluginParams::new(),
            init_seed: 1,
            step_hooks: Vec::new(),
            predator_count: 0,
            spawn_headroom: 0,
        }
    }

    #[test]
    fn empty_registry_makes_new_fail_not_panic() {
        let registry = Registry::new();
        let result = Simulation::new(dummy_config(), &registry);
        assert!(result.is_err());
    }

    #[test]
    fn fully_registered_dummy_composition_constructs_successfully() {
        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        let sim = Simulation::new(dummy_config(), &registry).unwrap();
        assert_eq!(sim.boid_count(), 5);
        assert_eq!(sim.step_count(), 0);
        let names = sim.plugin_names();
        assert!(names.contains(&("mode", "dummy_mode")));
        assert!(names.contains(&("noise", "dummy_noise")));
    }

    #[test]
    fn partially_registered_composition_fails_with_unknown_plugin() {
        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        // Leave "domain" unregistered by asking for a name that was never registered.
        let mut config = dummy_config();
        config.domain = "nonexistent_domain".to_string();
        match Simulation::new(config, &registry) {
            Err(e) => assert_eq!(
                e,
                ConfigError::UnknownPlugin {
                    socket: "Domain",
                    name: "nonexistent_domain".into()
                }
            ),
            Ok(_) => panic!("expected UnknownPlugin"),
        }
    }

    fn built_dummy_sim(n: u32) -> Simulation {
        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        let mut config = dummy_config();
        config.core_params = CoreParams::builder().boid_count(n).build().unwrap();
        Simulation::new(config, &registry).unwrap()
    }

    #[test]
    fn step_runs_without_nan_and_advances_step_count() {
        let mut sim = built_dummy_sim(20);
        for step in 0..50 {
            sim.step(1.0, 42);
            assert_eq!(sim.step_count(), step + 1);
            for i in sim.boids.iter_active() {
                assert!(
                    sim.boids.pos[i as usize].is_finite(),
                    "pos went non-finite at step {step}"
                );
                assert!(
                    sim.boids.vel[i as usize].is_finite(),
                    "vel went non-finite at step {step}"
                );
            }
        }
    }

    #[test]
    fn run_batch_runs_the_requested_number_of_steps() {
        let mut sim = built_dummy_sim(10);
        sim.run_batch(15, 7);
        assert_eq!(sim.step_count(), 15);
    }

    #[test]
    fn run_batch_with_budget_completes_within_a_generous_budget() {
        let mut sim = built_dummy_sim(10);
        let (completed, all_done) = sim.run_batch_with_budget(20, 7, 10_000.0);
        assert_eq!(completed, 20);
        assert!(all_done);
        assert_eq!(sim.step_count(), 20);
    }

    #[test]
    fn run_batch_with_budget_stops_early_under_a_tiny_budget() {
        let mut sim = built_dummy_sim(500);
        let (completed, all_done) = sim.run_batch_with_budget(1_000_000, 7, 0.001);
        assert!(!all_done);
        assert!(completed < 1_000_000);
        assert_eq!(sim.step_count() as u32, completed);
    }

    #[test]
    fn state_hash_is_identical_across_1_4_and_8_rayon_threads() {
        fn run_with_pool_size(threads: usize) -> u64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut sim = built_dummy_sim(200);
                sim.run_batch(30, 12345);
                sim.state_hash()
            })
        }
        let h1 = run_with_pool_size(1);
        let h4 = run_with_pool_size(4);
        let h8 = run_with_pool_size(8);
        assert_eq!(h1, h4, "state_hash differs between 1 and 4 threads");
        assert_eq!(h1, h8, "state_hash differs between 1 and 8 threads");
    }

    #[test]
    fn composition_reports_plugin_names_and_core_params() {
        let sim = built_dummy_sim(3);
        let comp = sim.composition();
        assert!(comp.plugin_names.contains(&("mode", "dummy_mode")));
        assert_eq!(comp.core_params.boid_count, 3);
    }

    /// StepHook ordering contract (roadmap.md Phase 8): with no declared `dependencies()`,
    /// execution order follows registration order (design/02_plugins.md §5). Uses a `static`
    /// log rather than a closure capturing shared state: `Registry`'s factories are plain `fn`
    /// pointers (design/02_plugins.md §1), which can't close over anything.
    static ORDER_LOG: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());

    struct LoggingHook(&'static str);
    impl StepHook for LoggingHook {
        fn post_steer(&self, _ctx: BoidCtx<'_>, _acc: &mut Vec3) {
            ORDER_LOG.lock().unwrap().push(self.0);
        }
        fn name(&self) -> &'static str {
            self.0
        }
    }
    fn make_hook_a(_p: &PluginParams) -> Box<dyn StepHook> {
        Box::new(LoggingHook("hook_a"))
    }
    fn make_hook_b(_p: &PluginParams) -> Box<dyn StepHook> {
        Box::new(LoggingHook("hook_b"))
    }

    #[test]
    fn step_hooks_execute_in_registration_order() {
        ORDER_LOG.lock().unwrap().clear();

        let mut registry = Registry::new();
        register_all_dummies(&mut registry);
        registry.register_step_hook("hook_a", make_hook_a);
        registry.register_step_hook("hook_b", make_hook_b);

        let mut config = dummy_config();
        config.core_params = CoreParams::builder().boid_count(1).build().unwrap();
        config.step_hooks = vec!["hook_a".to_string(), "hook_b".to_string()];
        let mut sim = Simulation::new(config, &registry).unwrap();

        sim.step(1.0, 1);

        assert_eq!(*ORDER_LOG.lock().unwrap(), vec!["hook_a", "hook_b"]);
    }
}
