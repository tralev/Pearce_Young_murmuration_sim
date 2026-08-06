//! `NeighborAdaptiveSpeed` — a continuous, per-boid neighbour-count-driven speed-cap `StepHook`
//! (design/02_plugins.md §5, roadmap.md Phase 18, pymurmur's
//! `physics/extensions/neighbor_adaptive_speed.py`, ported by description — pymurmur's actual
//! source isn't reachable in this environment, the same blocker as every other pymurmur
//! cross-check this project has hit; design/02_plugins.md's own one-line description is "minor
//! per-boid speed modulation extensions; low priority," so this is deliberately kept small).
//!
//! **Identical shape to `murmur_boid_state_machine`, gets G1 and G3 for free** — exactly as
//! `roadmap.md` §12/18 anticipated. `post_steer` reads `ctx.neighbors.len()` (real neighbour
//! data, G1's own fix) and caches a per-boid multiplier in a `Mutex<HashMap<u32, f64>>` (the
//! same plugin-owned side-column pattern `murmur_boid_state_machine`/`murmur_angle`/
//! `murmur_spin_wave` already established); `speed_cap_multiplier` reads that cache back for
//! `SpeedModel::enforce` (G3's own channel). No new architectural work needed — this plugin
//! exists specifically to prove that "gets G3 for free now" claim, not just repeat it.
//!
//! **The one real difference from `murmur_boid_state_machine`**: that plugin's Crowded/Normal/
//! Isolated/Threatened classification is a discrete, priority-ordered state machine with a
//! *single* cap value (`crowded_speed_cap`) applied only to the Crowded bucket. This plugin is
//! instead a continuous linear interpolation between `max_speed_factor` (at or below
//! `low_count` neighbours) and `min_speed_factor` (at or above `high_count` neighbours) — a
//! smoothly graded congestion response rather than a hard cutoff, closer in spirit to
//! Greenshields' 1935 linear speed-density relation from macroscopic traffic-flow theory (speed
//! falls off roughly linearly as local density rises toward a "jam" level) than to a discrete
//! state. Disclosed as a deliberate analogy, not a citation: no source in this project ties that
//! specific functional form to starling flocking, and pymurmur's own source (which would settle
//! it either way) isn't reachable here.
//!
//! **The cap only ever narrows, same G3 contract every other `SpeedModel`-influencing hook
//! follows**: `max_speed_factor` and `min_speed_factor` are both constrained to `(0, 1]`, with
//! `max_speed_factor >= min_speed_factor` — there is no way to *boost* a sparse boid's speed
//! above `cruise_speed` through this channel, only ever cap a crowded one below it.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{BoidCtx, ConfigError, PluginParams, Registry, Rng, StepHook, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborAdaptiveSpeedParams {
    pub low_count: f64,
    pub high_count: f64,
    pub min_speed_factor: f64,
    pub max_speed_factor: f64,
}

impl NeighborAdaptiveSpeedParams {
    pub fn builder() -> NeighborAdaptiveSpeedParamsBuilder {
        NeighborAdaptiveSpeedParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NeighborAdaptiveSpeedParamsBuilder {
    low_count: f64,
    high_count: f64,
    min_speed_factor: f64,
    max_speed_factor: f64,
}

impl Default for NeighborAdaptiveSpeedParamsBuilder {
    fn default() -> Self {
        NeighborAdaptiveSpeedParamsBuilder {
            low_count: 2.0,
            high_count: 12.0,
            min_speed_factor: 0.5,
            max_speed_factor: 1.0,
        }
    }
}

impl NeighborAdaptiveSpeedParamsBuilder {
    pub fn low_count(mut self, v: f64) -> Self {
        self.low_count = v;
        self
    }
    pub fn high_count(mut self, v: f64) -> Self {
        self.high_count = v;
        self
    }
    pub fn min_speed_factor(mut self, v: f64) -> Self {
        self.min_speed_factor = v;
        self
    }
    pub fn max_speed_factor(mut self, v: f64) -> Self {
        self.max_speed_factor = v;
        self
    }

    pub fn build(self) -> Result<NeighborAdaptiveSpeedParams, ConfigError> {
        if !(self.low_count.is_finite() && self.low_count >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "low_count",
                reason: "must be finite and >= 0".into(),
            });
        }
        if !(self.high_count.is_finite() && self.high_count > self.low_count) {
            return Err(ConfigError::InvalidParam {
                field: "high_count",
                reason: "must be finite and > low_count".into(),
            });
        }
        if !(self.min_speed_factor.is_finite()
            && self.min_speed_factor > 0.0
            && self.min_speed_factor <= 1.0)
        {
            return Err(ConfigError::InvalidParam {
                field: "min_speed_factor",
                reason: "must be finite and in (0, 1]".into(),
            });
        }
        if !(self.max_speed_factor.is_finite()
            && self.max_speed_factor > 0.0
            && self.max_speed_factor <= 1.0)
        {
            return Err(ConfigError::InvalidParam {
                field: "max_speed_factor",
                reason: "must be finite and in (0, 1]".into(),
            });
        }
        if self.max_speed_factor < self.min_speed_factor {
            return Err(ConfigError::InvalidParam {
                field: "max_speed_factor",
                reason: "must be >= min_speed_factor -- a cap only ever narrows".into(),
            });
        }
        Ok(NeighborAdaptiveSpeedParams {
            low_count: self.low_count,
            high_count: self.high_count,
            min_speed_factor: self.min_speed_factor,
            max_speed_factor: self.max_speed_factor,
        })
    }
}

pub struct NeighborAdaptiveSpeed {
    pub params: NeighborAdaptiveSpeedParams,
    multiplier: Mutex<HashMap<u32, f64>>,
}

impl NeighborAdaptiveSpeed {
    pub fn new(params: NeighborAdaptiveSpeedParams) -> Self {
        NeighborAdaptiveSpeed {
            params,
            multiplier: Mutex::new(HashMap::new()),
        }
    }

    /// This boid's most recently computed speed-cap multiplier, if it's ever been seen by
    /// `post_steer` — not part of the `StepHook` trait, but real, checkable state this plugin's
    /// own tests need direct read access to.
    pub fn multiplier_of(&self, index: u32) -> Option<f64> {
        self.multiplier.lock().unwrap().get(&index).copied()
    }

    fn compute_multiplier(&self, neighbor_count: u32) -> f64 {
        let n = neighbor_count as f64;
        if n <= self.params.low_count {
            self.params.max_speed_factor
        } else if n >= self.params.high_count {
            self.params.min_speed_factor
        } else {
            let t = (n - self.params.low_count) / (self.params.high_count - self.params.low_count);
            self.params.max_speed_factor
                + t * (self.params.min_speed_factor - self.params.max_speed_factor)
        }
    }
}

impl StepHook for NeighborAdaptiveSpeed {
    fn post_steer(&self, ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut Rng) {
        let m = self.compute_multiplier(ctx.neighbors.len() as u32);
        self.multiplier.lock().unwrap().insert(ctx.index, m);
    }

    fn speed_cap_multiplier(&self, index: u32) -> Option<f64> {
        self.multiplier_of(index)
    }

    fn name(&self) -> &'static str {
        "neighbor_adaptive_speed"
    }
}

/// Registers `NeighborAdaptiveSpeed` under the name `"neighbor_adaptive_speed"`, reading each of
/// `NeighborAdaptiveSpeedParams`'s fields from `PluginParams`. The factory type can't return
/// `Result` (design/02_plugins.md §1), so a malformed override falls back to the default rather
/// than panicking — same pattern as every other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("neighbor_adaptive_speed", |p: &PluginParams| {
        let d = NeighborAdaptiveSpeedParamsBuilder::default();
        let params = NeighborAdaptiveSpeedParams::builder()
            .low_count(p.get_or("low_count", d.low_count))
            .high_count(p.get_or("high_count", d.high_count))
            .min_speed_factor(p.get_or("min_speed_factor", d.min_speed_factor))
            .max_speed_factor(p.get_or("max_speed_factor", d.max_speed_factor))
            .build()
            .unwrap_or_else(|_| {
                NeighborAdaptiveSpeedParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(NeighborAdaptiveSpeed::new(params)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Neighbor, Species};

    struct StubDomain;
    impl Domain for StubDomain {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "stub_domain"
        }
    }

    fn core_params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn neighbors(n: usize) -> Vec<Neighbor> {
        (0..n)
            .map(|i| Neighbor {
                index: i as u32 + 1,
                distance: 1.0,
                direction: Vec3::new(1.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
            })
            .collect()
    }

    fn ctx<'a>(
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel: Vec3::new(0.5, 0.0, 0.0),
            species: Species::Prey,
            neighbors,
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook =
            NeighborAdaptiveSpeed::new(NeighborAdaptiveSpeedParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn at_or_below_low_count_the_multiplier_is_max_speed_factor() {
        let hook = NeighborAdaptiveSpeed::new(
            NeighborAdaptiveSpeedParams::builder()
                .low_count(2.0)
                .max_speed_factor(1.0)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(1);
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&n, &params, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert_eq!(hook.multiplier_of(0), Some(1.0));
        assert_eq!(hook.speed_cap_multiplier(0), Some(1.0));
    }

    #[test]
    fn at_or_above_high_count_the_multiplier_is_min_speed_factor() {
        let hook = NeighborAdaptiveSpeed::new(
            NeighborAdaptiveSpeedParams::builder()
                .high_count(10.0)
                .min_speed_factor(0.4)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(20);
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&n, &params, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert_eq!(hook.multiplier_of(0), Some(0.4));
    }

    #[test]
    fn the_midpoint_neighbor_count_gives_the_midpoint_multiplier() {
        let hook = NeighborAdaptiveSpeed::new(
            NeighborAdaptiveSpeedParams::builder()
                .low_count(0.0)
                .high_count(10.0)
                .min_speed_factor(0.4)
                .max_speed_factor(1.0)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(5); // exactly halfway between low_count=0 and high_count=10
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&n, &params, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        let expected = 0.5 * (1.0 + 0.4); // midpoint of max and min
        assert!(
            (hook.multiplier_of(0).unwrap() - expected).abs() < 1e-9,
            "got {:?}",
            hook.multiplier_of(0)
        );
    }

    #[test]
    fn the_multiplier_is_monotone_nonincreasing_in_neighbor_count() {
        let hook =
            NeighborAdaptiveSpeed::new(NeighborAdaptiveSpeedParams::builder().build().unwrap());
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        let mut previous = f64::INFINITY;
        for count in [0usize, 1, 2, 4, 6, 8, 10, 12, 15, 20] {
            let n = neighbors(count);
            hook.post_steer(
                ctx(&n, &params, &domain),
                &mut acc,
                &mut murmur_core::rng::for_boid(1, 2, 3),
            );
            let m = hook.multiplier_of(0).unwrap();
            assert!(
                m <= previous + 1e-12,
                "multiplier should never increase as neighbour count grows: count={} m={} previous={}",
                count,
                m,
                previous
            );
            previous = m;
        }
    }

    #[test]
    fn multiplier_of_an_unseen_boid_is_none() {
        let hook =
            NeighborAdaptiveSpeed::new(NeighborAdaptiveSpeedParams::builder().build().unwrap());
        assert_eq!(hook.multiplier_of(42), None);
        assert_eq!(hook.speed_cap_multiplier(42), None);
    }

    #[test]
    fn builder_rejects_a_high_count_not_greater_than_low_count() {
        assert!(NeighborAdaptiveSpeedParams::builder()
            .low_count(5.0)
            .high_count(5.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_min_speed_factor_above_one() {
        assert!(NeighborAdaptiveSpeedParams::builder()
            .min_speed_factor(1.5)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_max_speed_factor_below_min_speed_factor() {
        assert!(NeighborAdaptiveSpeedParams::builder()
            .min_speed_factor(0.8)
            .max_speed_factor(0.5)
            .build()
            .is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("neighbor_adaptive_speed", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "neighbor_adaptive_speed");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("min_speed_factor", 5.0);
        let hook = reg
            .resolve_step_hook("neighbor_adaptive_speed", &bad)
            .unwrap();
        assert_eq!(hook.name(), "neighbor_adaptive_speed");
    }
}
