//! `DynamicVisionRange` — a flock-wide adaptive perception-radius `StepHook` (design/02_plugins.md
//! §5, roadmap.md Phase 18, pymurmur's `physics/extensions/dynamic_vision_range.py`, ported by
//! description — pymurmur's actual source isn't reachable in this environment, the same blocker
//! as every other pymurmur cross-check in this project). Each `pre_step`, it measures the flock's
//! own current average neighbour count (a brute-force O(N²) pairwise sweep over `sim.boids`, the
//! same simplification `murmur_predator_fsm` already established as fine at slice scale) and
//! nudges a multiplier on `core_params.vision_radius` toward whatever value would bring that
//! average back toward `target_neighbor_count` — shrink the radius when the flock is too densely
//! connected, grow it when too sparse.
//!
//! **This is the first plugin to actually exercise `SimView::core_params`'s mutable half** — every
//! prior `StepHook` (including `murmur_ecology`, whose own `pre_step` only reads `sim.boids`) has
//! only ever read it. `step_hook.rs`'s own module doc names "e.g. ecology's time-of-day update" as
//! the motivating example for that mutability, but `murmur_ecology` computes its own internal
//! `EnvironmentState` instead of writing back into `core_params` — this plugin is the first to
//! genuinely close that loop, and the pipeline plumbing needs no changes to support it:
//! `Simulation::step` already re-reads `self.core_params` fresh for every neighbour-selection call
//! after `run_pre_step_hooks()` runs, so a hook's own `core_params.vision_radius` write here is
//! live for the very same step, not the next one.
//!
//! **`target_neighbor_count` defaults to `6.5`** — not an arbitrary guess, but the midpoint of
//! this project's own Y-a hard-gate target (`m* ≈ 6–7`, Young et al. 2013's own robustness-per-
//! neighbour optimum, already load-bearing elsewhere in this codebase — see `sci/param_table.md`
//! and `design/03_observables_bindings.md`'s own acceptance harness). Framing this plugin's
//! feedback target around a figure this project already treats as meaningful is a deliberate
//! choice, not a claim that pymurmur's own default (unreachable, as always) used the same number.
//!
//! **Straight-line Euclidean distance, not `Domain::delta`** — same disclosed simplification
//! `murmur_predator_fsm`'s own O(N²) sweep already established: correct under `Open`/`Margin`/
//! `Sphere`(`Soft`) domains, where straight-line distance *is* the real metric, but would
//! undercount neighbours near a `Torus` domain's own wraparound seam. Not fixed here, for the
//! same reason it wasn't fixed there — a real, disclosed scope limit, not an oversight.
//!
//! **Deliberately out of scope**: `SpatialIndex`'s own cell size (e.g. `HashGrid::cell_size`) is
//! not re-tuned alongside a shrinking/growing `vision_radius` — a performance-only concern (a
//! search radius that outgrows its index's cell size still returns correct results, just sweeps
//! more cells), not a correctness one, so left as a disclosed follow-up rather than solved here.

use murmur_core::{
    BoidColumns, ConfigError, PluginParams, Registry, SceneCheckpointFields, SimView, StepHook,
    Vec3,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicVisionRangeParams {
    pub target_neighbor_count: f64,
    pub adapt_rate: f64,
    pub min_multiplier: f64,
    pub max_multiplier: f64,
}

impl DynamicVisionRangeParams {
    pub fn builder() -> DynamicVisionRangeParamsBuilder {
        DynamicVisionRangeParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicVisionRangeParamsBuilder {
    target_neighbor_count: f64,
    adapt_rate: f64,
    min_multiplier: f64,
    max_multiplier: f64,
}

impl Default for DynamicVisionRangeParamsBuilder {
    fn default() -> Self {
        DynamicVisionRangeParamsBuilder {
            target_neighbor_count: 6.5, // midpoint of this project's own Y-a target, m*~=6-7
            adapt_rate: 0.1,
            min_multiplier: 0.25,
            max_multiplier: 4.0,
        }
    }
}

impl DynamicVisionRangeParamsBuilder {
    pub fn target_neighbor_count(mut self, v: f64) -> Self {
        self.target_neighbor_count = v;
        self
    }
    pub fn adapt_rate(mut self, v: f64) -> Self {
        self.adapt_rate = v;
        self
    }
    pub fn min_multiplier(mut self, v: f64) -> Self {
        self.min_multiplier = v;
        self
    }
    pub fn max_multiplier(mut self, v: f64) -> Self {
        self.max_multiplier = v;
        self
    }

    pub fn build(self) -> Result<DynamicVisionRangeParams, ConfigError> {
        if !(self.target_neighbor_count.is_finite() && self.target_neighbor_count > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "target_neighbor_count",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.adapt_rate.is_finite() && self.adapt_rate >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "adapt_rate",
                reason: "must be finite and >= 0".into(),
            });
        }
        if !(self.min_multiplier.is_finite() && self.min_multiplier > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "min_multiplier",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.max_multiplier.is_finite() && self.max_multiplier >= self.min_multiplier) {
            return Err(ConfigError::InvalidParam {
                field: "max_multiplier",
                reason: "must be finite and >= min_multiplier".into(),
            });
        }
        Ok(DynamicVisionRangeParams {
            target_neighbor_count: self.target_neighbor_count,
            adapt_rate: self.adapt_rate,
            min_multiplier: self.min_multiplier,
            max_multiplier: self.max_multiplier,
        })
    }
}

/// Brute-force average active-boid neighbour count within `radius` — O(N²), the same
/// simplification `murmur_predator_fsm`'s own `pre_step` sweep already established as fine at
/// slice scale. Returns `0.0` for 0 or 1 active boids (no pairs to count).
fn average_neighbor_count(boids: &BoidColumns, radius: f64) -> f64 {
    let positions: Vec<Vec3> = boids.iter_active().map(|i| boids.pos[i as usize]).collect();
    let n = positions.len();
    if n <= 1 {
        return 0.0;
    }
    let r2 = radius * radius;
    let mut total = 0u64;
    for (i, &pi) in positions.iter().enumerate() {
        for (j, &pj) in positions.iter().enumerate() {
            if i != j && (pi - pj).len_sq() <= r2 {
                total += 1;
            }
        }
    }
    total as f64 / n as f64
}

pub struct DynamicVisionRange {
    pub params: DynamicVisionRangeParams,
    base_vision_radius: Option<f64>,
    multiplier: f64,
}

impl DynamicVisionRange {
    pub fn new(params: DynamicVisionRangeParams) -> Self {
        DynamicVisionRange {
            params,
            base_vision_radius: None,
            multiplier: 1.0,
        }
    }

    /// The current radius multiplier — `1.0` before the first `pre_step` call, thereafter
    /// clamped to `[min_multiplier, max_multiplier]`. Exposed for tests and introspection.
    pub fn current_multiplier(&self) -> f64 {
        self.multiplier
    }
}

impl StepHook for DynamicVisionRange {
    fn pre_step(&mut self, sim: &mut SimView) {
        let base = *self
            .base_vision_radius
            .get_or_insert(sim.core_params.vision_radius);
        let avg = average_neighbor_count(sim.boids, sim.core_params.vision_radius);
        let relative_error =
            (avg - self.params.target_neighbor_count) / self.params.target_neighbor_count;
        self.multiplier *= 1.0 - self.params.adapt_rate * relative_error;
        self.multiplier = self
            .multiplier
            .clamp(self.params.min_multiplier, self.params.max_multiplier);
        sim.core_params.vision_radius = base * self.multiplier;
    }

    /// design/05_viz_contract.md §2.2's `dynamic_vision_range` — the same multiplier
    /// `current_multiplier()` already exposes for tests/introspection, now also published
    /// through the generic `StepHook` checkpoint-field seam (roadmap.md's own follow-up on
    /// design/05 alignment).
    fn checkpoint_scene_fields(&self) -> SceneCheckpointFields {
        SceneCheckpointFields {
            dynamic_vision_range: Some(self.multiplier as f32),
            ..Default::default()
        }
    }

    fn name(&self) -> &'static str {
        "dynamic_vision_range"
    }
}

pub fn register(r: &mut Registry) {
    r.register_step_hook("dynamic_vision_range", |p: &PluginParams| {
        let d = DynamicVisionRangeParamsBuilder::default();
        let params = DynamicVisionRangeParams::builder()
            .target_neighbor_count(p.get_or("target_neighbor_count", d.target_neighbor_count))
            .adapt_rate(p.get_or("adapt_rate", d.adapt_rate))
            .min_multiplier(p.get_or("min_multiplier", d.min_multiplier))
            .max_multiplier(p.get_or("max_multiplier", d.max_multiplier))
            .build()
            .unwrap_or_else(|_| {
                DynamicVisionRangeParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(DynamicVisionRange::new(params)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Species};

    fn core_params(vision_radius: f64) -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .dt(1.0)
            .vision_radius(vision_radius)
            .build()
            .unwrap()
    }

    fn boids_at(positions: &[Vec3]) -> BoidColumns {
        let mut cols = BoidColumns::with_capacity(positions.len() as u32);
        for p in positions {
            cols.add(*p, Vec3::ZERO, Species::Prey, 0).unwrap();
        }
        cols
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook =
            DynamicVisionRange::new(DynamicVisionRangeParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn checkpoint_scene_fields_publishes_the_current_multiplier() {
        let hook = DynamicVisionRange::new(DynamicVisionRangeParams::builder().build().unwrap());
        assert_eq!(
            hook.checkpoint_scene_fields().dynamic_vision_range,
            Some(1.0)
        );
    }

    #[test]
    fn average_neighbor_count_matches_a_hand_computed_case() {
        // 3 boids on a line at x=0,1,2: radius=1.5 sees immediate neighbours only.
        // boid 0 sees boid 1 (dist 1) -> 1; boid 1 sees 0 and 2 -> 2; boid 2 sees 1 -> 1.
        // average = (1+2+1)/3 = 4/3.
        let boids = boids_at(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ]);
        let avg = average_neighbor_count(&boids, 1.5);
        assert!((avg - 4.0 / 3.0).abs() < 1e-9, "got {}", avg);
    }

    #[test]
    fn average_neighbor_count_is_zero_for_a_single_boid() {
        let boids = boids_at(&[Vec3::new(0.0, 0.0, 0.0)]);
        assert_eq!(average_neighbor_count(&boids, 100.0), 0.0);
    }

    #[test]
    fn pre_step_caches_the_original_vision_radius_on_the_first_call() {
        let mut hook =
            DynamicVisionRange::new(DynamicVisionRangeParams::builder().build().unwrap());
        assert_eq!(hook.base_vision_radius, None);
        let boids = boids_at(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)]);
        let mut core = core_params(10.0);
        let mut sim = SimView {
            boids: &boids,
            core_params: &mut core,
            step_count: 0,
        };
        hook.pre_step(&mut sim);
        assert_eq!(hook.base_vision_radius, Some(10.0));
    }

    #[test]
    fn radius_shrinks_when_average_neighbor_count_exceeds_target() {
        // 6 boids tightly packed (all mutually visible at radius=10) with a low target=1.0 ->
        // average neighbour count (5, everyone sees everyone else) is far above target ->
        // multiplier should shrink below 1.0.
        let mut hook = DynamicVisionRange::new(
            DynamicVisionRangeParams::builder()
                .target_neighbor_count(1.0)
                .adapt_rate(0.2)
                .build()
                .unwrap(),
        );
        let positions: Vec<Vec3> = (0..6)
            .map(|i| Vec3::new(i as f64 * 0.1, 0.0, 0.0))
            .collect();
        let boids = boids_at(&positions);
        let mut core = core_params(10.0);
        let mut sim = SimView {
            boids: &boids,
            core_params: &mut core,
            step_count: 0,
        };
        hook.pre_step(&mut sim);
        assert!(
            sim.core_params.vision_radius < 10.0,
            "got {}",
            sim.core_params.vision_radius
        );
        assert!(hook.current_multiplier() < 1.0);
    }

    #[test]
    fn radius_grows_when_average_neighbor_count_is_below_target() {
        // 2 boids far apart (radius=1.0 sees no one) with a high target=10.0 -> average
        // neighbour count (0) is far below target -> multiplier should grow above 1.0.
        let mut hook = DynamicVisionRange::new(
            DynamicVisionRangeParams::builder()
                .target_neighbor_count(10.0)
                .adapt_rate(0.2)
                .build()
                .unwrap(),
        );
        let boids = boids_at(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)]);
        let mut core = core_params(1.0);
        let mut sim = SimView {
            boids: &boids,
            core_params: &mut core,
            step_count: 0,
        };
        hook.pre_step(&mut sim);
        assert!(
            sim.core_params.vision_radius > 1.0,
            "got {}",
            sim.core_params.vision_radius
        );
        assert!(hook.current_multiplier() > 1.0);
    }

    #[test]
    fn multiplier_stays_within_configured_bounds_over_many_steps() {
        // A relentlessly-too-crowded scenario (many mutually-visible boids, target=0.01) run for
        // 200 pre_step calls in a row -- multiplier must never leave [min_multiplier,
        // max_multiplier] and vision_radius must stay finite and positive throughout.
        let mut hook = DynamicVisionRange::new(
            DynamicVisionRangeParams::builder()
                .target_neighbor_count(0.01)
                .adapt_rate(0.9)
                .min_multiplier(0.5)
                .max_multiplier(2.0)
                .build()
                .unwrap(),
        );
        let positions: Vec<Vec3> = (0..20)
            .map(|i| Vec3::new(i as f64 * 0.01, 0.0, 0.0))
            .collect();
        let boids = boids_at(&positions);
        let mut core = core_params(50.0);
        for _ in 0..200 {
            let mut sim = SimView {
                boids: &boids,
                core_params: &mut core,
                step_count: 0,
            };
            hook.pre_step(&mut sim);
            assert!(hook.current_multiplier() >= 0.5 - 1e-9);
            assert!(hook.current_multiplier() <= 2.0 + 1e-9);
            assert!(sim.core_params.vision_radius.is_finite());
            assert!(sim.core_params.vision_radius > 0.0);
        }
    }

    #[test]
    fn builder_rejects_a_nonpositive_target_neighbor_count() {
        assert!(DynamicVisionRangeParams::builder()
            .target_neighbor_count(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_negative_adapt_rate() {
        assert!(DynamicVisionRangeParams::builder()
            .adapt_rate(-0.1)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_nonpositive_min_multiplier() {
        assert!(DynamicVisionRangeParams::builder()
            .min_multiplier(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_max_multiplier_below_min_multiplier() {
        assert!(DynamicVisionRangeParams::builder()
            .min_multiplier(2.0)
            .max_multiplier(1.0)
            .build()
            .is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new();
        assert!(reg
            .resolve_step_hook("dynamic_vision_range", &params)
            .is_ok());
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new().with("target_neighbor_count", -5.0);
        assert!(reg
            .resolve_step_hook("dynamic_vision_range", &params)
            .is_ok());
    }
}
