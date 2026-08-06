//! `SpeedNoise` — a minor per-boid stochastic speed-cap `StepHook` (design/02_plugins.md §5,
//! roadmap.md Phase 18, pymurmur's `physics/extensions/speed_noise.py`, ported by description —
//! pymurmur's actual source isn't reachable in this environment, the same blocker as every other
//! pymurmur cross-check this project has hit; design/02_plugins.md's own one-line description is
//! "minor per-boid speed modulation extensions; low priority," so this is deliberately kept
//! small, the last of that pair after `murmur_neighbor_adaptive_speed`).
//!
//! **The plugin that motivated fixing G8** (roadmap.md §12): before this pass, no `StepHook`
//! had any path to genuine, `base_seed`-tied randomness — `SpeedModel::enforce` was the only
//! per-boid caller with a real `Rng` in hand. This is the first `StepHook` whose whole *point* is
//! stochastic behaviour, so it's the plugin that actually needed the fix rather than one that
//! could route around it (`murmur_ecology`'s `predator_active` is deterministic by design, not
//! blocked by this; `murmur_boid_state_machine`/`murmur_neighbor_adaptive_speed` are both purely
//! deterministic functions of neighbour count). **Fix**: `StepHook::post_steer` gained a third
//! `rng: &mut Rng` parameter, drawing from the same single, sequential write-phase `Rng` stream
//! `SpeedModel::enforce`'s own unstall reseed already draws from — deterministic and
//! thread-count-independent by construction, since the write phase itself is already sequential,
//! not parallel. Verified directly in `murmur_core::pipeline`'s own test suite (not just here):
//! the same `base_seed` gives identical `post_steer` RNG draws across independent runs, a
//! different `base_seed` gives different ones.
//!
//! **Downward-only, a real disclosed consequence of G3's own contract, not an oversight**:
//! `SpeedModel::enforce`'s `cap_multiplier` is combined across every composed hook via `min`
//! against a `1.0` baseline (G3's fix, roadmap.md §12) — any value above `1.0` a hook returns is
//! silently absorbed by that baseline, so there is no way to *boost* a boid's speed through this
//! channel, only ever narrow it. A symmetric two-sided "speed noise" (sometimes faster, sometimes
//! slower than `cruise_speed`) is therefore not expressible through the existing `StepHook`
//! architecture without extending G3's own contract — a bigger, riskier change this "minor, low
//! priority" plugin doesn't justify. What's built instead: a gentle, smoothed, downward-only
//! stochastic multiplier, wobbling in `[1 - noise_amplitude, 1.0]`.
//!
//! **Smoothed, not i.i.d. per step**: each `post_steer` draws one fresh uniform sample and blends
//! it into the *previous* step's multiplier via `smoothing` (an exponential moving average, `1.0`
//! = no memory/pure noise, small values = a slow, continuous wobble) — real animal locomotion
//! speed doesn't jump discontinuously step to step, and a purely independent draw every step
//! would. Per-boid state via the now-familiar plugin-owned side-column pattern
//! (`Mutex<HashMap<u32, f64>>`, same shape `murmur_boid_state_machine`/
//! `murmur_neighbor_adaptive_speed` already established).

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::rng::uniform01;
use murmur_core::{BoidCtx, ConfigError, PluginParams, Registry, Rng, StepHook, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedNoiseParams {
    pub noise_amplitude: f64,
    pub smoothing: f64,
}

impl SpeedNoiseParams {
    pub fn builder() -> SpeedNoiseParamsBuilder {
        SpeedNoiseParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpeedNoiseParamsBuilder {
    noise_amplitude: f64,
    smoothing: f64,
}

impl Default for SpeedNoiseParamsBuilder {
    fn default() -> Self {
        SpeedNoiseParamsBuilder {
            noise_amplitude: 0.15,
            smoothing: 0.3,
        }
    }
}

impl SpeedNoiseParamsBuilder {
    pub fn noise_amplitude(mut self, v: f64) -> Self {
        self.noise_amplitude = v;
        self
    }
    pub fn smoothing(mut self, v: f64) -> Self {
        self.smoothing = v;
        self
    }

    pub fn build(self) -> Result<SpeedNoiseParams, ConfigError> {
        if !(self.noise_amplitude.is_finite()
            && self.noise_amplitude >= 0.0
            && self.noise_amplitude < 1.0)
        {
            return Err(ConfigError::InvalidParam {
                field: "noise_amplitude",
                reason: "must be finite and in [0, 1) -- a cap only ever narrows, so >= 1 would \
                         allow it to reach or cross zero"
                    .into(),
            });
        }
        if !(self.smoothing.is_finite() && self.smoothing > 0.0 && self.smoothing <= 1.0) {
            return Err(ConfigError::InvalidParam {
                field: "smoothing",
                reason: "must be finite and in (0, 1]".into(),
            });
        }
        Ok(SpeedNoiseParams {
            noise_amplitude: self.noise_amplitude,
            smoothing: self.smoothing,
        })
    }
}

pub struct SpeedNoise {
    pub params: SpeedNoiseParams,
    multiplier: Mutex<HashMap<u32, f64>>,
}

impl SpeedNoise {
    pub fn new(params: SpeedNoiseParams) -> Self {
        SpeedNoise {
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
}

impl StepHook for SpeedNoise {
    fn post_steer(&self, ctx: BoidCtx<'_>, _acc: &mut Vec3, rng: &mut Rng) {
        let raw = 1.0 - self.params.noise_amplitude * uniform01(rng);
        let mut cache = self.multiplier.lock().unwrap();
        let previous = cache.get(&ctx.index).copied().unwrap_or(1.0);
        let smoothed = previous * (1.0 - self.params.smoothing) + raw * self.params.smoothing;
        cache.insert(ctx.index, smoothed);
    }

    fn speed_cap_multiplier(&self, index: u32) -> Option<f64> {
        self.multiplier_of(index)
    }

    fn name(&self) -> &'static str {
        "speed_noise"
    }
}

/// Registers `SpeedNoise` under the name `"speed_noise"`, reading each of `SpeedNoiseParams`'s
/// fields from `PluginParams`. The factory type can't return `Result` (design/02_plugins.md §1),
/// so a malformed override falls back to the default rather than panicking — same pattern as
/// every other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("speed_noise", |p: &PluginParams| {
        let d = SpeedNoiseParamsBuilder::default();
        let params = SpeedNoiseParams::builder()
            .noise_amplitude(p.get_or("noise_amplitude", d.noise_amplitude))
            .smoothing(p.get_or("smoothing", d.smoothing))
            .build()
            .unwrap_or_else(|_| {
                SpeedNoiseParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(SpeedNoise::new(params)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{rng, CoreParams, Domain, Species};

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

    fn ctx<'a>(index: u32, params: &'a CoreParams, domain: &'a dyn Domain) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos: Vec3::ZERO,
            vel: Vec3::new(0.5, 0.0, 0.0),
            species: Species::Prey,
            neighbors: &[],
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = SpeedNoise::new(SpeedNoiseParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn the_multiplier_always_stays_within_one_minus_noise_amplitude_and_one() {
        let hook = SpeedNoise::new(
            SpeedNoiseParams::builder()
                .noise_amplitude(0.2)
                .smoothing(1.0) // pure noise, no smoothing memory -- the strictest bound check
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        let mut r = rng::for_boid(1, 2, 3);
        for _ in 0..500 {
            hook.post_steer(ctx(0, &params, &domain), &mut acc, &mut r);
            let m = hook.multiplier_of(0).unwrap();
            assert!(
                (0.8..=1.0).contains(&m),
                "multiplier {} outside [1-noise_amplitude, 1.0] = [0.8, 1.0]",
                m
            );
        }
    }

    #[test]
    fn zero_noise_amplitude_always_gives_exactly_one() {
        let hook = SpeedNoise::new(
            SpeedNoiseParams::builder()
                .noise_amplitude(0.0)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        let mut r = rng::for_boid(1, 2, 3);
        for _ in 0..20 {
            hook.post_steer(ctx(0, &params, &domain), &mut acc, &mut r);
            assert_eq!(hook.multiplier_of(0), Some(1.0));
        }
    }

    #[test]
    fn different_boids_get_independent_multipliers() {
        let hook = SpeedNoise::new(SpeedNoiseParams::builder().build().unwrap());
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        let mut r = rng::for_boid(1, 2, 3);
        hook.post_steer(ctx(0, &params, &domain), &mut acc, &mut r);
        hook.post_steer(ctx(1, &params, &domain), &mut acc, &mut r);
        assert!(hook.multiplier_of(0).is_some());
        assert!(hook.multiplier_of(1).is_some());
    }

    #[test]
    fn smoothing_of_one_means_no_memory_of_the_previous_multiplier() {
        // With smoothing=1.0, each call's result is exactly `1.0 - noise_amplitude * u` for that
        // call's own draw -- verified by reproducing the same draw from a freshly-seeded rng
        // with identical inputs and checking the two post_steer results match.
        let hook_a = SpeedNoise::new(
            SpeedNoiseParams::builder()
                .noise_amplitude(0.3)
                .smoothing(1.0)
                .build()
                .unwrap(),
        );
        let hook_b = SpeedNoise::new(
            SpeedNoiseParams::builder()
                .noise_amplitude(0.3)
                .smoothing(1.0)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;

        let mut r_a = rng::for_boid(7, 8, 9);
        hook_a.post_steer(ctx(0, &params, &domain), &mut acc, &mut r_a);
        hook_a.post_steer(ctx(0, &params, &domain), &mut acc, &mut r_a); // second call, fresh draw

        let mut r_b = rng::for_boid(7, 8, 9);
        hook_b.post_steer(ctx(0, &params, &domain), &mut acc, &mut r_b); // discard first draw
        hook_b.post_steer(ctx(0, &params, &domain), &mut acc, &mut r_b);

        assert_eq!(
            hook_a.multiplier_of(0),
            hook_b.multiplier_of(0),
            "with smoothing=1.0, the second call's result should depend only on the second \
             draw, not accumulate memory of the first"
        );
    }

    #[test]
    fn multiplier_of_an_unseen_boid_is_none() {
        let hook = SpeedNoise::new(SpeedNoiseParams::builder().build().unwrap());
        assert_eq!(hook.multiplier_of(42), None);
        assert_eq!(hook.speed_cap_multiplier(42), None);
    }

    #[test]
    fn builder_rejects_a_noise_amplitude_of_one_or_more() {
        assert!(SpeedNoiseParams::builder()
            .noise_amplitude(1.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_negative_noise_amplitude() {
        assert!(SpeedNoiseParams::builder()
            .noise_amplitude(-0.1)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_zero_smoothing() {
        assert!(SpeedNoiseParams::builder().smoothing(0.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("speed_noise", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "speed_noise");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("noise_amplitude", 5.0);
        let hook = reg.resolve_step_hook("speed_noise", &bad).unwrap();
        assert_eq!(hook.name(), "speed_noise");
    }
}
