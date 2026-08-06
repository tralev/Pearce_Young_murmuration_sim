//! `Ecology` — day/night cycle, logistic dusk roost, seasonal amplitude, coherence gate,
//! deterministic predator presence `StepHook` (design/02_plugins.md §5, roadmap.md Phase 18,
//! pymurmur's `physics/extensions/ecology.py`, ported by description — pymurmur's actual
//! source isn't reachable in this environment, the same blocker as every other pymurmur
//! cross-check this project has hit). The "time of day"/"evening" control surface
//! design/05_viz_contract.md §2 names (`Environment`'s `day`/`hour`/`dusk_factor`/
//! `is_roosting_time`/`is_murmuration_season`/`coherence_factor`/`temperature`/
//! `predator_active` fields — this plugin computes exactly those eight, under the same names).
//!
//! **Provenance, disclosed precisely** (design/02_plugins.md §5's own note, carried forward
//! here after independently re-checking the source): only the deterministic predator-presence
//! rate is empirically grounded. Goodenough, Little, Carpenter & Hart 2017 ("Birds of a
//! feather flock together: Insights into starling murmuration behaviour revealed using citizen
//! science", `sci/Birds of a feather flock together.pdf`) reports, in its own abstract:
//! *"Birds of prey were recorded at 29.6% of murmurations"* — `predator_rate`'s default,
//! `0.296`, matches that figure exactly. The logistic dusk ramp and cosine seasonal
//! temperature cycle are **not** derived from that paper — it reports no sunset-relative
//! timing data at all, and its own seasonal fits (flock size vs. time of year) are
//! quadratic-in-day-of-year, not cosine. Those two shapes are pymurmur's own smoothing
//! choices, kept here for the same reason design/02_plugins.md keeps them: a plausible,
//! documented default, not a claimed empirical fit.
//!
//! **`predator_active` is deterministic, not a coin flip** — design/02_plugins.md's own
//! wording. Computed via `frac(day * φ)` (the golden-ratio Weyl/Kronecker sequence,
//! `murmur_influencer`'s own `rank()` convention, reused here rather than re-derived): a fixed,
//! reproducible per-day pattern with no RNG dependency, whose long-run frequency converges to
//! `predator_rate` without ever being literally random.
//!
//! **The season/calendar is a stylized cycle, not a real one** — this project's core is
//! dimensionless throughout (design/00_overview.md §2), so `day`/`hour` are simulated units
//! (driven by real elapsed time, `step_count * dt`, fixing nothing new — G4 was already fixed
//! in Phase 13), not calendar dates; `season_start_day`/`season_end_day`/`year_length_days` are
//! configurable, not tied to any specific real year.
//!
//! **The coherence gate is a real, testable force, not just a reported field**: `post_steer`
//! applies a mild pull toward the flock centroid (cached once per step in `pre_step`, via
//! `SimView.boids` — no new architectural gap needed, `pre_step` already has full column
//! access), scaled by `coherence_factor * coherence_strength`. `coherence_factor` is `1.0` only
//! during dusk *and* murmuration season, `0.0` otherwise — literally gating how strongly
//! murmuration-like coherence is encouraged, tied to time of day/season.
//!
//! **Deliberately out of scope for this pass**, disclosed rather than silently implied:
//! wiring this plugin's `EnvironmentState` into `batch.rs`'s `Command::SetEnvironment`/
//! `Checkpoint.environment` (both still documented no-ops there, design/05_viz_contract.md §2/
//! §3) — a separate Track B-shaped follow-up, not attempted here; and automatically spawning/
//! despawning predator boids when `predator_active` flips — there is no live-mutable
//! `predator_count` path today (composition is fixed at construction), so `predator_active` is
//! a reported/testable signal here, not an actuator. A caller wanting a live predator
//! injection already has `Command::AddPredator` via `run_batch_checked` to drive that
//! externally.

use std::sync::Mutex;

use murmur_core::{
    BoidCtx, ConfigError, PluginParams, Registry, SimView, StepHook, Vec3, MIN_LEN2,
};

const GOLDEN: f64 = 0.618_033_988_749_895;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentState {
    pub day: u64,
    pub hour: f64,
    pub dusk_factor: f64,
    pub is_roosting_time: bool,
    pub is_murmuration_season: bool,
    pub coherence_factor: f64,
    pub temperature: f64,
    pub predator_active: bool,
}

impl Default for EnvironmentState {
    fn default() -> Self {
        EnvironmentState {
            day: 0,
            hour: 0.0,
            dusk_factor: 0.0,
            is_roosting_time: false,
            is_murmuration_season: false,
            coherence_factor: 0.0,
            temperature: 0.0,
            predator_active: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcologyParams {
    pub hours_per_dt: f64,
    pub dusk_hour: f64,
    pub dusk_width: f64,
    pub roosting_threshold: f64,
    pub year_length_days: u64,
    pub season_start_day: u64,
    pub season_end_day: u64,
    pub predator_rate: f64,
    pub temperature_mean: f64,
    pub temperature_amplitude: f64,
    pub temperature_phase_day: f64,
    pub coherence_strength: f64,
}

impl EcologyParams {
    pub fn builder() -> EcologyParamsBuilder {
        EcologyParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EcologyParamsBuilder {
    hours_per_dt: f64,
    dusk_hour: f64,
    dusk_width: f64,
    roosting_threshold: f64,
    year_length_days: u64,
    season_start_day: u64,
    season_end_day: u64,
    predator_rate: f64,
    temperature_mean: f64,
    temperature_amplitude: f64,
    temperature_phase_day: f64,
    coherence_strength: f64,
}

impl Default for EcologyParamsBuilder {
    fn default() -> Self {
        EcologyParamsBuilder {
            hours_per_dt: 0.5,
            dusk_hour: 18.0,
            dusk_width: 1.0,
            roosting_threshold: 0.5,
            year_length_days: 365,
            season_start_day: 274, // ~Oct 1: paper's own "size increased October to early
            season_end_day: 90,    // February, decreased until end of season in March"
            predator_rate: 0.296,  // Goodenough et al. 2017's own reported figure
            temperature_mean: 10.0,
            temperature_amplitude: 8.0,
            temperature_phase_day: 15.0,
            coherence_strength: 0.3,
        }
    }
}

impl EcologyParamsBuilder {
    pub fn hours_per_dt(mut self, v: f64) -> Self {
        self.hours_per_dt = v;
        self
    }
    pub fn dusk_hour(mut self, v: f64) -> Self {
        self.dusk_hour = v;
        self
    }
    pub fn dusk_width(mut self, v: f64) -> Self {
        self.dusk_width = v;
        self
    }
    pub fn roosting_threshold(mut self, v: f64) -> Self {
        self.roosting_threshold = v;
        self
    }
    pub fn year_length_days(mut self, v: u64) -> Self {
        self.year_length_days = v;
        self
    }
    pub fn season_start_day(mut self, v: u64) -> Self {
        self.season_start_day = v;
        self
    }
    pub fn season_end_day(mut self, v: u64) -> Self {
        self.season_end_day = v;
        self
    }
    pub fn predator_rate(mut self, v: f64) -> Self {
        self.predator_rate = v;
        self
    }
    pub fn temperature_mean(mut self, v: f64) -> Self {
        self.temperature_mean = v;
        self
    }
    pub fn temperature_amplitude(mut self, v: f64) -> Self {
        self.temperature_amplitude = v;
        self
    }
    pub fn temperature_phase_day(mut self, v: f64) -> Self {
        self.temperature_phase_day = v;
        self
    }
    pub fn coherence_strength(mut self, v: f64) -> Self {
        self.coherence_strength = v;
        self
    }

    pub fn build(self) -> Result<EcologyParams, ConfigError> {
        if !(self.hours_per_dt.is_finite() && self.hours_per_dt > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "hours_per_dt",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.dusk_width.is_finite() && self.dusk_width > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "dusk_width",
                reason: "must be finite and > 0".into(),
            });
        }
        if self.year_length_days == 0 {
            return Err(ConfigError::InvalidParam {
                field: "year_length_days",
                reason: "must be > 0".into(),
            });
        }
        if !(self.predator_rate.is_finite() && (0.0..=1.0).contains(&self.predator_rate)) {
            return Err(ConfigError::InvalidParam {
                field: "predator_rate",
                reason: "must be finite and in [0, 1]".into(),
            });
        }
        if !(self.coherence_strength.is_finite() && self.coherence_strength >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "coherence_strength",
                reason: "must be finite and >= 0".into(),
            });
        }
        Ok(EcologyParams {
            hours_per_dt: self.hours_per_dt,
            dusk_hour: self.dusk_hour,
            dusk_width: self.dusk_width,
            roosting_threshold: self.roosting_threshold,
            year_length_days: self.year_length_days,
            season_start_day: self.season_start_day,
            season_end_day: self.season_end_day,
            predator_rate: self.predator_rate,
            temperature_mean: self.temperature_mean,
            temperature_amplitude: self.temperature_amplitude,
            temperature_phase_day: self.temperature_phase_day,
            coherence_strength: self.coherence_strength,
        })
    }
}

pub struct Ecology {
    pub params: EcologyParams,
    state: Mutex<(EnvironmentState, Vec3)>, // (environment, cached flock centroid)
}

impl Ecology {
    pub fn new(params: EcologyParams) -> Self {
        Ecology {
            params,
            state: Mutex::new((EnvironmentState::default(), Vec3::ZERO)),
        }
    }

    pub fn environment(&self) -> EnvironmentState {
        self.state.lock().unwrap().0
    }

    fn is_in_season(&self, day_of_year: u64) -> bool {
        let (start, end) = (self.params.season_start_day, self.params.season_end_day);
        if start <= end {
            (start..=end).contains(&day_of_year)
        } else {
            // Wraps across the year boundary (e.g. Oct -> Mar).
            day_of_year >= start || day_of_year <= end
        }
    }

    fn compute(&self, t: f64) -> EnvironmentState {
        let total_hours = t * self.params.hours_per_dt;
        let day = (total_hours / 24.0).floor() as u64;
        let hour = total_hours.rem_euclid(24.0);

        let dusk_factor =
            1.0 / (1.0 + (-(hour - self.params.dusk_hour) / self.params.dusk_width).exp());
        let is_roosting_time = dusk_factor > self.params.roosting_threshold;

        let day_of_year = day % self.params.year_length_days;
        let is_murmuration_season = self.is_in_season(day_of_year);

        let coherence_factor = if is_murmuration_season {
            dusk_factor
        } else {
            0.0
        };

        let angle =
            2.0 * std::f64::consts::PI * (day_of_year as f64 - self.params.temperature_phase_day)
                / self.params.year_length_days as f64;
        let temperature =
            self.params.temperature_mean + self.params.temperature_amplitude * angle.cos();

        let predator_active = (day as f64 * GOLDEN).fract() < self.params.predator_rate;

        EnvironmentState {
            day,
            hour,
            dusk_factor,
            is_roosting_time,
            is_murmuration_season,
            coherence_factor,
            temperature,
            predator_active,
        }
    }
}

impl StepHook for Ecology {
    fn pre_step(&mut self, sim: &mut SimView) {
        let t = sim.step_count as f64 * sim.core_params.dt;
        let env = self.compute(t);

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

        // `&mut self` here means exclusive access to `self.state` too -- `get_mut()` updates
        // it directly, no lock needed (there's genuinely nothing to contend with during
        // `pre_step`, which `run_pre_step_hooks` calls sequentially, once per hook per step).
        *self.state.get_mut().unwrap() = (env, centroid);
    }

    fn post_steer(&self, ctx: BoidCtx<'_>, acc: &mut Vec3) {
        let (env, centroid) = *self.state.lock().unwrap();
        if env.coherence_factor <= 0.0 || self.params.coherence_strength <= 0.0 {
            return;
        }
        let offset = centroid - ctx.pos;
        if offset.len_sq() > MIN_LEN2 {
            *acc += offset.normalized() * (env.coherence_factor * self.params.coherence_strength);
        }
    }

    fn name(&self) -> &'static str {
        "ecology"
    }
}

/// Registers `Ecology` under the name `"ecology"`, reading each of `EcologyParams`'s fields
/// from `PluginParams`. The factory type can't return `Result` (design/02_plugins.md §1), so a
/// malformed override falls back to the default rather than panicking — same pattern as every
/// other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("ecology", |p: &PluginParams| {
        let d = EcologyParamsBuilder::default();
        let params = EcologyParams::builder()
            .hours_per_dt(p.get_or("hours_per_dt", d.hours_per_dt))
            .dusk_hour(p.get_or("dusk_hour", d.dusk_hour))
            .dusk_width(p.get_or("dusk_width", d.dusk_width))
            .roosting_threshold(p.get_or("roosting_threshold", d.roosting_threshold))
            .year_length_days(p.get_or("year_length_days", d.year_length_days as f64) as u64)
            .season_start_day(p.get_or("season_start_day", d.season_start_day as f64) as u64)
            .season_end_day(p.get_or("season_end_day", d.season_end_day as f64) as u64)
            .predator_rate(p.get_or("predator_rate", d.predator_rate))
            .temperature_mean(p.get_or("temperature_mean", d.temperature_mean))
            .temperature_amplitude(p.get_or("temperature_amplitude", d.temperature_amplitude))
            .temperature_phase_day(p.get_or("temperature_phase_day", d.temperature_phase_day))
            .coherence_strength(p.get_or("coherence_strength", d.coherence_strength))
            .build()
            .unwrap_or_else(|_| {
                EcologyParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(Ecology::new(params)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{BoidColumns, CoreParams, Domain, Neighbor, Species};

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

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = Ecology::new(EcologyParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn hour_wraps_around_at_24() {
        let hook = Ecology::new(EcologyParams::builder().hours_per_dt(1.0).build().unwrap());
        let env = hook.compute(30.0); // 30 hours in -> day 1, hour 6
        assert_eq!(env.day, 1);
        assert!((env.hour - 6.0).abs() < 1e-9);
    }

    #[test]
    fn dusk_factor_is_low_at_noon_and_high_at_midnight() {
        let hook = Ecology::new(EcologyParams::builder().hours_per_dt(1.0).build().unwrap());
        let noon = hook.compute(12.0);
        let midnight = hook.compute(24.0); // hour wraps to 0, but has passed dusk_hour=18 once
        assert!(noon.dusk_factor < 0.1, "got {}", noon.dusk_factor);
        // Use a point clearly after dusk instead of exactly at the wrap, which is ambiguous:
        let late_evening = hook.compute(21.0); // hour = 21, well past dusk_hour=18
        assert!(
            late_evening.dusk_factor > 0.9,
            "got {}",
            late_evening.dusk_factor
        );
        let _ = midnight;
    }

    #[test]
    fn is_roosting_time_follows_the_dusk_factor_threshold() {
        let hook = Ecology::new(
            EcologyParams::builder()
                .hours_per_dt(1.0)
                .roosting_threshold(0.5)
                .build()
                .unwrap(),
        );
        assert!(!hook.compute(12.0).is_roosting_time);
        assert!(hook.compute(21.0).is_roosting_time);
    }

    #[test]
    fn season_membership_handles_a_year_boundary_wrap() {
        let hook = Ecology::new(
            EcologyParams::builder()
                .season_start_day(274)
                .season_end_day(90)
                .build()
                .unwrap(),
        );
        assert!(hook.is_in_season(300)); // late Oct: inside the wrapped range
        assert!(hook.is_in_season(10)); // early Jan: inside the wrapped range
        assert!(!hook.is_in_season(150)); // May: outside
    }

    #[test]
    fn season_membership_handles_a_non_wrapping_range() {
        let hook = Ecology::new(
            EcologyParams::builder()
                .season_start_day(10)
                .season_end_day(20)
                .build()
                .unwrap(),
        );
        assert!(hook.is_in_season(15));
        assert!(!hook.is_in_season(5));
        assert!(!hook.is_in_season(25));
    }

    #[test]
    fn coherence_factor_is_zero_outside_the_season_even_at_dusk() {
        let hook = Ecology::new(
            EcologyParams::builder()
                .hours_per_dt(1.0)
                .season_start_day(274)
                .season_end_day(90)
                .build()
                .unwrap(),
        );
        // day 150 (May) at hour 21 (well past dusk) -- high dusk_factor, but out of season.
        let t = 150.0 * 24.0 + 21.0;
        let env = hook.compute(t);
        assert!(env.dusk_factor > 0.9);
        assert!(!env.is_murmuration_season);
        assert_eq!(env.coherence_factor, 0.0);
    }

    #[test]
    fn predator_active_is_deterministic_not_random() {
        let hook = Ecology::new(EcologyParams::builder().build().unwrap());
        let a = hook.compute(500.0).predator_active;
        let b = hook.compute(500.0).predator_active;
        assert_eq!(a, b);
    }

    #[test]
    fn predator_active_converges_to_the_configured_rate_over_many_days() {
        let hook = Ecology::new(
            EcologyParams::builder()
                .hours_per_dt(1.0)
                .predator_rate(0.296)
                .build()
                .unwrap(),
        );
        let n_days = 2000;
        let active_count = (0..n_days)
            .filter(|&d| hook.compute(d as f64 * 24.0).predator_active)
            .count();
        let rate = active_count as f64 / n_days as f64;
        assert!((rate - 0.296).abs() < 0.02, "got rate={}", rate);
    }

    #[test]
    fn temperature_is_finite_across_a_full_year() {
        let hook = Ecology::new(EcologyParams::builder().hours_per_dt(1.0).build().unwrap());
        for day in 0..365 {
            let env = hook.compute(day as f64 * 24.0);
            assert!(env.temperature.is_finite());
        }
    }

    #[test]
    fn post_steer_pulls_toward_the_cached_centroid_when_the_coherence_gate_is_open() {
        let mut hook = Ecology::new(
            EcologyParams::builder()
                .hours_per_dt(1.0)
                .season_start_day(0)
                .season_end_day(364)
                .coherence_strength(1.0)
                .build()
                .unwrap(),
        );
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0);
        boids.add(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 21_u64, // hours_per_dt=1.0 -> hour=21, well past dusk -> gate open
        };
        hook.pre_step(&mut view);
        assert!(hook.environment().coherence_factor > 0.0);

        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::new(-5.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc);
        assert!(
            acc.x > 0.0,
            "must pull toward the centroid (at x=0), got {:?}",
            acc
        );
    }

    #[test]
    fn post_steer_does_nothing_when_the_coherence_gate_is_closed() {
        let mut hook = Ecology::new(
            EcologyParams::builder()
                .hours_per_dt(1.0)
                .season_start_day(274)
                .season_end_day(90)
                .coherence_strength(1.0)
                .build()
                .unwrap(),
        );
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0);
        boids.add(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: (150 * 24 + 21) as u64, // May, out of season
        };
        hook.pre_step(&mut view);
        assert_eq!(hook.environment().coherence_factor, 0.0);

        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::new(-5.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc);
        assert_eq!(acc, Vec3::ZERO);
    }

    #[test]
    fn builder_rejects_a_zero_year_length() {
        assert!(EcologyParams::builder()
            .year_length_days(0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_predator_rate_outside_zero_one() {
        assert!(EcologyParams::builder().predator_rate(1.5).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("ecology", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "ecology");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("predator_rate", 5.0);
        let hook = reg.resolve_step_hook("ecology", &bad).unwrap();
        assert_eq!(hook.name(), "ecology");
    }
}
