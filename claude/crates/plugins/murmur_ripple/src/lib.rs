//! `Ripple` — a 3-train travelling density-pulse envelope `StepHook` (design/02_plugins.md §5,
//! roadmap.md Phase 18, pymurmur's `physics/extensions/ripple.py`, ported by description —
//! pymurmur's actual source isn't reachable in this environment, the same blocker as every other
//! pymurmur cross-check this project has hit). The design doc's own one-line description: "3-train
//! travelling density-pulse envelopes; publishes `ripple_envelope_sum`."
//!
//! **Interpretation, disclosed.** "3-train" is read as *exactly three pulse trains sharing one
//! curve shape, evenly staggered in time* — the same "one shared shape, N staggered copies"
//! pattern `murmur_field` already established for its own `anchor_count` anchors, applied here to
//! *time* offsets instead of spatial ones. Each train periodically (every `period` simulated time
//! units) emits a new pulse that expands outward from the flock's own *live centroid* (`pre_step`
//! computes it fresh every step, the same technique `murmur_ecology`/`murmur_wander` already use)
//! at a fixed `speed`, as a Gaussian-shaped ring: `envelope_k(d, t) = amplitude *
//! exp(-(d - speed·((t - k·period/3) mod period))² / (2·width²))`, where `d` is a boid's own
//! distance from the centroid. `ripple_envelope_sum` (the design doc's own published name, taken
//! literally) is the sum of all three trains' contributions at that boid's position — published
//! as a real accessor, `pub fn envelope_sum_of(&self, index: u32) -> Option<f64>`, the same
//! pattern `murmur_boid_state_machine`'s `state_of`/`murmur_wander`'s `wander_center` already
//! established.
//!
//! **The actual force: a gentle, downward-only speed-cap wobble as a ring washes over a boid**,
//! via `speed_cap_multiplier` (G3's own channel) — `cap = (1 - min(envelope_sum, 1))`, clamped to
//! `[min_cap, 1.0]`. Downward-only for the same disclosed reason `murmur_neighbor_adaptive_speed`/
//! `murmur_speed_noise` already documented: `SpeedModel::enforce`'s `cap_multiplier` is combined
//! via `min` against a `1.0` baseline, so a hook can only ever narrow the ceiling, never boost it.
//! A visible, testable "wave of deceleration" rippling through the flock as each ring's radius
//! grows — distinct from `murmur_predator_fsm`'s own `wave` mechanism (a lateral steering nudge
//! triggered by a live predator, grounded in Procaccini et al. 2011's real "agitation wave"
//! finding): this plugin is an *ambient*, predator-independent density pulse, not a threat
//! response, so a density/speed modulation (not a directional steering force) is the more
//! natural, honestly-scoped mechanic for it.
//!
//! **Fully deterministic, no G8 dependency** — like `murmur_wander`, unlike `murmur_speed_noise`
//! (the plugin that motivated fixing G8, roadmap.md §12): a closed-form function of
//! `step_count * dt` (G4's own contract). `post_steer`'s own `rng` parameter goes unused.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{BoidCtx, ConfigError, PluginParams, Registry, Rng, SimView, StepHook, Vec3};

const NUM_TRAINS: u32 = 3;
const MIN_ENVELOPE: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RippleParams {
    pub period: f64,
    pub speed: f64,
    pub width: f64,
    pub amplitude: f64,
    pub min_cap: f64,
}

impl RippleParams {
    pub fn builder() -> RippleParamsBuilder {
        RippleParamsBuilder::default()
    }

    /// Train `k`'s (of `NUM_TRAINS`) current ring radius at elapsed time `t`: `speed` times the
    /// time elapsed since that train's most recent emission, wrapping every `period` (and evenly
    /// staggered across trains by `k * period / NUM_TRAINS`, `murmur_field`'s own anchor-stagger
    /// idea applied to time instead of space).
    fn ring_radius(&self, k: u32, t: f64) -> f64 {
        let offset = k as f64 * (self.period / NUM_TRAINS as f64);
        let phase = (t - offset).rem_euclid(self.period);
        self.speed * phase
    }

    /// The sum of all `NUM_TRAINS` trains' Gaussian-ring envelope contributions at distance `d`
    /// from the live flock centroid, at elapsed time `t` — the "`ripple_envelope_sum`" the design
    /// doc's own wording names.
    fn envelope_sum(&self, d: f64, t: f64) -> f64 {
        (0..NUM_TRAINS)
            .map(|k| {
                let r = self.ring_radius(k, t);
                let delta = d - r;
                self.amplitude * (-(delta * delta) / (2.0 * self.width * self.width)).exp()
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RippleParamsBuilder {
    period: f64,
    speed: f64,
    width: f64,
    amplitude: f64,
    min_cap: f64,
}

impl Default for RippleParamsBuilder {
    fn default() -> Self {
        RippleParamsBuilder {
            period: 20.0,
            speed: 2.0,
            width: 3.0,
            amplitude: 0.3,
            min_cap: 0.05,
        }
    }
}

impl RippleParamsBuilder {
    pub fn period(mut self, v: f64) -> Self {
        self.period = v;
        self
    }
    pub fn speed(mut self, v: f64) -> Self {
        self.speed = v;
        self
    }
    pub fn width(mut self, v: f64) -> Self {
        self.width = v;
        self
    }
    pub fn amplitude(mut self, v: f64) -> Self {
        self.amplitude = v;
        self
    }
    pub fn min_cap(mut self, v: f64) -> Self {
        self.min_cap = v;
        self
    }

    pub fn build(self) -> Result<RippleParams, ConfigError> {
        if !(self.period.is_finite() && self.period > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "period",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.speed.is_finite() && self.speed > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "speed",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.width.is_finite() && self.width > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "width",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.amplitude.is_finite() && self.amplitude > 0.0 && self.amplitude <= 1.0) {
            return Err(ConfigError::InvalidParam {
                field: "amplitude",
                reason: "must be finite and in (0, 1]".into(),
            });
        }
        if !(self.min_cap.is_finite() && self.min_cap > 0.0 && self.min_cap < 1.0) {
            return Err(ConfigError::InvalidParam {
                field: "min_cap",
                reason: "must be finite and in (0, 1)".into(),
            });
        }
        Ok(RippleParams {
            period: self.period,
            speed: self.speed,
            width: self.width,
            amplitude: self.amplitude,
            min_cap: self.min_cap,
        })
    }
}

pub struct Ripple {
    pub params: RippleParams,
    // (live flock centroid, elapsed time) -- refreshed once per step in `pre_step`.
    center: Mutex<(Vec3, f64)>,
    // Per-boid `ripple_envelope_sum`, cached in `post_steer` -- the design doc's own published
    // quantity, read back by both `speed_cap_multiplier` and `envelope_sum_of`.
    envelope: Mutex<HashMap<u32, f64>>,
}

impl Ripple {
    pub fn new(params: RippleParams) -> Self {
        Ripple {
            params,
            center: Mutex::new((Vec3::ZERO, 0.0)),
            envelope: Mutex::new(HashMap::new()),
        }
    }

    /// This boid's most recently computed `ripple_envelope_sum`, if it's ever been seen by
    /// `post_steer` — the design doc's own "publishes `ripple_envelope_sum`" requirement, not
    /// part of the `StepHook` trait itself (no other occupant needs it).
    pub fn envelope_sum_of(&self, index: u32) -> Option<f64> {
        self.envelope.lock().unwrap().get(&index).copied()
    }
}

impl StepHook for Ripple {
    fn pre_step(&mut self, sim: &mut SimView) {
        let t = sim.step_count as f64 * sim.core_params.dt;

        let mut sum = Vec3::ZERO;
        let mut count = 0u32;
        for i in sim.boids.iter_active() {
            sum += sim.boids.pos[i as usize];
            count += 1;
        }
        let centroid = if count > 0 {
            sum * (1.0 / count as f64)
        } else {
            Vec3::ZERO
        };

        // `&mut self` here means exclusive access to `self.center` too -- `get_mut()` updates it
        // directly, no lock needed (nothing to contend with during `pre_step`, which
        // `run_pre_step_hooks` calls sequentially, once per hook per step).
        *self.center.get_mut().unwrap() = (centroid, t);
    }

    fn post_steer(&self, ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut Rng) {
        let (centroid, t) = *self.center.lock().unwrap();
        let d = (ctx.pos - centroid).len();
        let sum = self.params.envelope_sum(d, t);
        self.envelope.lock().unwrap().insert(ctx.index, sum);
    }

    fn speed_cap_multiplier(&self, index: u32) -> Option<f64> {
        let sum = self.envelope_sum_of(index)?;
        if sum <= MIN_ENVELOPE {
            return None;
        }
        Some((1.0 - sum.min(1.0)).clamp(self.params.min_cap, 1.0))
    }

    fn name(&self) -> &'static str {
        "ripple"
    }
}

/// Registers `Ripple` under the name `"ripple"`, reading each of `RippleParams`'s fields from
/// `PluginParams`. The factory type can't return `Result` (design/02_plugins.md §1), so a
/// malformed override falls back to the default rather than panicking — same pattern as every
/// other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("ripple", |p: &PluginParams| {
        let d = RippleParamsBuilder::default();
        let params = RippleParams::builder()
            .period(p.get_or("ripple_period", d.period))
            .speed(p.get_or("ripple_speed", d.speed))
            .width(p.get_or("ripple_width", d.width))
            .amplitude(p.get_or("ripple_amplitude", d.amplitude))
            .min_cap(p.get_or("ripple_min_cap", d.min_cap))
            .build()
            .unwrap_or_else(|_| RippleParams::builder().build().expect("defaults are valid"));
        Box::new(Ripple::new(params)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{rng, BoidColumns, CoreParams, Domain, Species};

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

    fn core_params(dt: f64) -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .dt(dt)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn ctx<'a>(
        index: u32,
        pos: Vec3,
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos,
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    fn run_pre_step(
        hook: &mut Ripple,
        boids: &BoidColumns,
        params: &mut CoreParams,
        step_count: u64,
    ) {
        let mut view = SimView {
            boids,
            core_params: params,
            step_count,
        };
        hook.pre_step(&mut view);
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = Ripple::new(RippleParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn ring_radius_grows_linearly_with_elapsed_time_within_one_period() {
        let params = RippleParams::builder()
            .period(100.0)
            .speed(2.0)
            .build()
            .unwrap();
        assert!((params.ring_radius(0, 0.0) - 0.0).abs() < 1e-9);
        assert!((params.ring_radius(0, 10.0) - 20.0).abs() < 1e-9);
        assert!((params.ring_radius(0, 30.0) - 60.0).abs() < 1e-9);
    }

    #[test]
    fn ring_radius_wraps_at_the_period() {
        let params = RippleParams::builder()
            .period(10.0)
            .speed(1.0)
            .build()
            .unwrap();
        // t=12 -> 12 mod 10 = 2 -> radius = speed * 2 = 2, not 12.
        assert!((params.ring_radius(0, 12.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn the_three_trains_are_evenly_staggered_in_time() {
        let params = RippleParams::builder()
            .period(30.0)
            .speed(1.0)
            .build()
            .unwrap();
        // At t=0, train 0 is at phase 0, train 1 at phase 20 (its own emission was 10 time
        // units in the past given a 10-unit stagger... concretely: offset_k = k*(30/3) = k*10,
        // so at t=0, phase_k = (0 - k*10) mod 30 = (30 - k*10) mod 30.
        assert!((params.ring_radius(0, 0.0) - 0.0).abs() < 1e-9);
        assert!((params.ring_radius(1, 0.0) - 20.0).abs() < 1e-9);
        assert!((params.ring_radius(2, 0.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn envelope_sum_peaks_at_the_ring_radius_and_decays_with_distance() {
        let params = RippleParams::builder()
            .period(1000.0) // long enough that only train 0 (phase 0 at t=0) matters near d=0
            .speed(1.0)
            .width(1.0)
            .amplitude(1.0)
            .build()
            .unwrap();
        let at_ring = params.envelope_sum(0.0, 0.0); // train 0's ring is at radius 0 when t=0
        let far_away = params.envelope_sum(50.0, 0.0);
        assert!(at_ring > far_away);
        assert!(far_away >= 0.0);
        assert!(at_ring.is_finite() && far_away.is_finite());
    }

    #[test]
    fn pre_step_caches_the_live_centroid_not_a_fixed_point() {
        let mut hook = Ripple::new(RippleParams::builder().build().unwrap());
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0);
        boids.add(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut params = core_params(1.0);
        run_pre_step(&mut hook, &boids, &mut params, 0);
        assert!((hook.center.lock().unwrap().0 - Vec3::ZERO).len() < 1e-9);

        boids.pos[0] = Vec3::new(90.0, 0.0, 0.0);
        boids.pos[1] = Vec3::new(110.0, 0.0, 0.0);
        run_pre_step(&mut hook, &boids, &mut params, 1);
        assert!((hook.center.lock().unwrap().0 - Vec3::new(100.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn post_steer_publishes_a_real_envelope_sum_per_boid() {
        let mut hook = Ripple::new(
            RippleParams::builder()
                .amplitude(1.0)
                .width(1.0)
                .build()
                .unwrap(),
        );
        let boids = BoidColumns::with_capacity(0);
        let mut params = core_params(1.0);
        run_pre_step(&mut hook, &boids, &mut params, 0); // centroid = ZERO, t = 0

        assert_eq!(hook.envelope_sum_of(0), None, "unseen boid must be None");

        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(0, Vec3::ZERO, &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        let sum = hook
            .envelope_sum_of(0)
            .expect("boid 0 was just seen by post_steer");
        assert!(sum.is_finite() && sum >= 0.0);
    }

    #[test]
    fn speed_cap_multiplier_only_ever_narrows_and_stays_within_min_cap_and_one() {
        let hook = Ripple::new(
            RippleParams::builder()
                .amplitude(1.0)
                .width(1.0)
                .min_cap(0.1)
                .build()
                .unwrap(),
        );
        // Directly populate the envelope cache via post_steer at the ring's own peak (d=0, t=0).
        let boids = BoidColumns::with_capacity(0);
        let _ = boids; // pre_step not needed here; center defaults to (ZERO, 0.0) already
        let params = core_params(1.0);
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(0, Vec3::ZERO, &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        let m = hook
            .speed_cap_multiplier(0)
            .expect("a strong envelope must produce a cap");
        assert!((0.1 - 1e-9..=1.0 + 1e-9).contains(&m), "got {}", m);
    }

    #[test]
    fn speed_cap_multiplier_is_none_far_from_every_ring() {
        let hook = Ripple::new(
            RippleParams::builder()
                .amplitude(1.0)
                .width(0.5)
                .build()
                .unwrap(),
        );
        let params = core_params(1.0);
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        // Far from any train's own ring radius at t=0 (all near 0, 10, or 20 with a default
        // period=20/speed=2 -- 10000 is nowhere near any of them).
        hook.post_steer(
            ctx(0, Vec3::new(10_000.0, 0.0, 0.0), &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        assert_eq!(hook.speed_cap_multiplier(0), None);
    }

    #[test]
    fn builder_rejects_a_nonpositive_period() {
        assert!(RippleParams::builder().period(0.0).build().is_err());
    }

    #[test]
    fn builder_rejects_an_amplitude_above_one() {
        assert!(RippleParams::builder().amplitude(1.5).build().is_err());
    }

    #[test]
    fn builder_rejects_a_min_cap_outside_zero_one() {
        assert!(RippleParams::builder().min_cap(1.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("ripple", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "ripple");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("ripple_amplitude", 5.0);
        let hook = reg.resolve_step_hook("ripple", &bad).unwrap();
        assert_eq!(hook.name(), "ripple");
    }
}
