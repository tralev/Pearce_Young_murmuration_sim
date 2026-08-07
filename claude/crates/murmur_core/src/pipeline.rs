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
use crate::registry::{PluginParams, Registry, Warning};
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
    /// Resolves every socket's plugin by name from `registry`, runs each socket's
    /// `validate_and_fix` (design/01_core.md §4.1 — self-correcting non-critical
    /// inconsistencies like `HashGrid`'s `cell_size` vs `CoreParams.vision_radius`), places
    /// the initial flock via the chosen `Initializer`, and returns the constructed
    /// `Simulation` plus every `Warning` collected along the way — or the first `ConfigError`
    /// encountered (an unregistered name, including every name against an empty `Registry`).
    /// Warnings are never fatal; a hard-invalid combination is still rejected earlier, at a
    /// plugin's own `PluginParams` resolution.
    pub fn new(
        config: SimConfig,
        registry: &Registry,
    ) -> Result<(Self, Vec<Warning>), ConfigError> {
        let mut mode = registry.resolve_mode(&config.mode, &config.plugin_params)?;
        let mut modifier = registry.resolve_modifier(&config.modifier, &config.plugin_params)?;
        let mut domain = registry.resolve_domain(&config.domain, &config.plugin_params)?;
        let mut spatial_index =
            registry.resolve_spatial_index(&config.spatial_index, &config.plugin_params)?;
        let mut neighbor_selection = registry
            .resolve_neighbor_selection(&config.neighbor_selection, &config.plugin_params)?;
        let mut speed_model =
            registry.resolve_speed_model(&config.speed_model, &config.plugin_params)?;
        let mut init = registry.resolve_init(&config.init, &config.plugin_params)?;
        let mut noise = registry.resolve_noise(&config.noise, &config.plugin_params)?;
        let mut step_hooks = Vec::with_capacity(config.step_hooks.len());
        for name in &config.step_hooks {
            step_hooks.push(registry.resolve_step_hook(name, &config.plugin_params)?);
        }

        let mut warnings = Vec::new();
        {
            // A frozen snapshot of every socket's own resolved config, taken before any
            // socket's `validate_and_fix` runs — deliberately not a live view of the other
            // boxed trait objects, so mutating one socket here never needs to alias another.
            let others: [(&str, PluginParams); 8] = [
                (config.mode.as_str(), mode.resolved_params()),
                (config.modifier.as_str(), modifier.resolved_params()),
                (config.domain.as_str(), domain.resolved_params()),
                (
                    config.spatial_index.as_str(),
                    spatial_index.resolved_params(),
                ),
                (
                    config.neighbor_selection.as_str(),
                    neighbor_selection.resolved_params(),
                ),
                (config.speed_model.as_str(), speed_model.resolved_params()),
                (config.init.as_str(), init.resolved_params()),
                (config.noise.as_str(), noise.resolved_params()),
            ];
            warnings.extend(mode.validate_and_fix(&config.core_params, &others));
            warnings.extend(modifier.validate_and_fix(&config.core_params, &others));
            warnings.extend(domain.validate_and_fix(&config.core_params, &others));
            warnings.extend(spatial_index.validate_and_fix(&config.core_params, &others));
            warnings.extend(neighbor_selection.validate_and_fix(&config.core_params, &others));
            warnings.extend(speed_model.validate_and_fix(&config.core_params, &others));
            warnings.extend(init.validate_and_fix(&config.core_params, &others));
            warnings.extend(noise.validate_and_fix(&config.core_params, &others));
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

        let simulation = Simulation {
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
        };
        Ok((simulation, warnings))
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
        self.mode.pre_step(&self.boids, self.step_count);
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

            // G3 (roadmap.md §12): the most restrictive `StepHook::speed_cap_multiplier()`
            // wins (combined via `min`) — `1.0` (no tightening) when no hook has an opinion,
            // so this costs nothing when `step_hooks` is empty or no hook overrides it.
            let mut cap_multiplier = 1.0_f64;

            if !self.step_hooks.is_empty() {
                let domain: &dyn Domain = &*self.domain;
                // G1 (roadmap.md §12): `ctx.neighbors` used to be a hardcoded `&[]` — no hook
                // could see real neighbour data during `post_steer` at all. Fixed by recomputing
                // it here via the same `NeighborSelection` the read phase uses. Two disclosed
                // wrinkles, both acceptable for what needs this (`murmur_boid_state_machine`'s
                // coarse local-density classification, not precision physics): (1) `spatial_index`
                // itself isn't rebuilt mid-write-phase, so cell/bucket membership reflects
                // start-of-step positions; (2) since this loop is sequential and mutates
                // `boids.pos`/`vel` in place as it goes, a lower-indexed boid processed earlier
                // this same step has *already* moved by the time a later boid's neighbour query
                // reads its position — a real, if minor, Gauss-Seidel-style mixing of
                // pre-/post-integration state, not a crash or NaN risk. `FlockingMode::desired()`
                // stays the source of truth for anything needing a consistent, whole-flock-frozen
                // snapshot; this seam is deliberately the cheaper, approximate one.
                let neighbors = self.neighbor_selection.select(
                    &*self.spatial_index,
                    i,
                    &self.boids,
                    &self.core_params,
                );
                let ctx = BoidCtx {
                    index: i,
                    pos: self.boids.pos[idx],
                    vel: self.boids.vel[idx],
                    species: self.boids.species[idx],
                    neighbors: &neighbors,
                    core_params: &self.core_params,
                    step_count: self.step_count,
                    domain,
                };
                let mut acc_i = self.boids.acc[idx];
                for hook in &self.step_hooks {
                    hook.post_steer(ctx, &mut acc_i, &mut write_rng);
                    if let Some(m) = hook.speed_cap_multiplier(i) {
                        cap_multiplier = cap_multiplier.min(m);
                    }
                }
                self.boids.acc[idx] = acc_i;
            }

            let pos_before = self.boids.pos[idx];

            self.boids.vel[idx] += self.boids.acc[idx] * dt;
            self.speed_model.enforce(
                &mut self.boids.vel[idx],
                self.boids.species[idx],
                &self.core_params,
                cap_multiplier,
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
mod tests;
