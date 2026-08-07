//! `Wander` — a bounded attractor-motion `StepHook` for the flock's own centre (design/02_plugins.md
//! §5, roadmap.md Phase 18, pymurmur's `physics/extensions/wander.py`, ported by description —
//! pymurmur's actual source isn't reachable in this environment, the same blocker as every other
//! pymurmur cross-check this project has hit).
//!
//! **Interpretation, disclosed.** Design/02_plugins.md §5's one-line description is "Bounded
//! 13-term attractor motion for the flock centre; publishes `wander_center`/`wander_heading`."
//! The "13-term" figure isn't independently reconstructable without pymurmur's own unreachable
//! source, and this plugin doesn't force its own parameter count to artificially match it (the
//! same honesty this project already applies to unreachable figures elsewhere, e.g. `m*` not
//! being force-fit to a target range in `sci/param_table.md`). What's built instead reuses
//! `murmur_field`'s own exact 11-term Lissajous-plus-envelope curve formula (`amplitude_{x,y,z}`,
//! `frequency_{x,y,z}`, `phase_{x,y,z}`, `envelope_amplitude`, `envelope_frequency` — the same
//! math, independently re-derived here rather than imported, since `curve_offset` is a private
//! method on `murmur_field`'s own `FieldParams` and these are deliberately independent plugin
//! crates, matching `murmur_predator_fsm`/`murmur_spin_wave`'s own precedent of not
//! cross-depending on plugin internals), plus one addition: `pull_strength` (12 configurable
//! terms total).
//!
//! **The one genuine, distinguishing design choice**: unlike `murmur_influencer`/`murmur_field`,
//! whose curve offsets anchor to a *fixed point in space*, this plugin's curve offsets the
//! flock's own *live centroid* (`pre_step` computes it fresh every step, the same technique
//! `murmur_ecology` already established) — the wander target moves *with* the flock rather than
//! being a distant point it chases across open space. That's the actual meaning of "for the
//! flock centre" in the design doc's own phrasing, and it's what "bounded" describes here: the
//! curve's own amplitude terms already bound the *excursion* (same as `murmur_field`'s), but that
//! excursion is bounded *relative to wherever the flock currently is*, not relative to a fixed
//! origin.
//!
//! **`wander_heading`, published as a real accessor** (`pub fn wander_heading`, matching the
//! design doc's own "publishes wander_center/wander_heading" — not part of the `StepHook` trait
//! itself, since no other occupant needs it, the same pattern `murmur_boid_state_machine`'s
//! `state_of`/`murmur_ecology`'s `environment` already established): the exact closed-form
//! derivative of the curve offset with respect to elapsed time, normalized to a unit direction —
//! not the flock's own bulk translation, a disclosed, deliberate scope choice (the curve's own
//! intrinsic heading, independent of where the flock itself happens to be drifting).
//!
//! **Fully deterministic, no RNG needed** — unlike `murmur_speed_noise` (the plugin that
//! motivated fixing **G8**, roadmap.md §12), this hook's entire behaviour is a closed-form
//! function of `step_count * dt` (G4's own contract, fixed in Phase 13), the same convention
//! `murmur_influencer`/`murmur_field` already use. `post_steer` doesn't touch its own `rng`
//! parameter at all.

use std::sync::Mutex;

use murmur_core::{
    BoidCtx, ConfigError, PluginParams, Registry, Rng, SceneCheckpointFields, SimView, StepHook,
    Vec3, WanderSnapshot, MIN_LEN2,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WanderParams {
    pub amplitude: Vec3,
    pub frequency: Vec3,
    pub phase: Vec3,
    pub envelope_amplitude: f64,
    pub envelope_frequency: f64,
    pub pull_strength: f64,
}

impl WanderParams {
    pub fn builder() -> WanderParamsBuilder {
        WanderParamsBuilder::default()
    }

    fn envelope(&self, t: f64) -> f64 {
        1.0 + self.envelope_amplitude * (self.envelope_frequency * t).sin()
    }

    fn envelope_deriv(&self, t: f64) -> f64 {
        self.envelope_amplitude * self.envelope_frequency * (self.envelope_frequency * t).cos()
    }

    /// The wander curve's own shape at elapsed time `t`, before the live centroid offset is
    /// applied — `murmur_field`'s exact `curve_offset` formula (stagger fixed at `0.0`, since
    /// there is only ever one wander target, not multiple staggered anchors).
    fn curve_offset(&self, t: f64) -> Vec3 {
        Vec3::new(
            self.amplitude.x * (self.frequency.x * t + self.phase.x).sin(),
            self.amplitude.y * (self.frequency.y * t + self.phase.y).sin(),
            self.amplitude.z * (self.frequency.z * t + self.phase.z).sin(),
        ) * self.envelope(t)
    }

    /// The exact closed-form derivative of `curve_offset` with respect to `t` (product rule:
    /// `envelope' * S + envelope * S'`) — a real velocity vector, not a finite-difference
    /// approximation.
    fn curve_velocity(&self, t: f64) -> Vec3 {
        let s = Vec3::new(
            self.amplitude.x * (self.frequency.x * t + self.phase.x).sin(),
            self.amplitude.y * (self.frequency.y * t + self.phase.y).sin(),
            self.amplitude.z * (self.frequency.z * t + self.phase.z).sin(),
        );
        let s_deriv = Vec3::new(
            self.amplitude.x * self.frequency.x * (self.frequency.x * t + self.phase.x).cos(),
            self.amplitude.y * self.frequency.y * (self.frequency.y * t + self.phase.y).cos(),
            self.amplitude.z * self.frequency.z * (self.frequency.z * t + self.phase.z).cos(),
        );
        s * self.envelope_deriv(t) + s_deriv * self.envelope(t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WanderParamsBuilder {
    amplitude: Vec3,
    frequency: Vec3,
    phase: Vec3,
    envelope_amplitude: f64,
    envelope_frequency: f64,
    pull_strength: f64,
}

impl Default for WanderParamsBuilder {
    fn default() -> Self {
        WanderParamsBuilder {
            amplitude: Vec3::new(10.0, 10.0, 5.0),
            frequency: Vec3::new(0.05, 0.07, 0.03),
            phase: Vec3::new(0.0, std::f64::consts::FRAC_PI_2, 0.0),
            envelope_amplitude: 0.3,
            envelope_frequency: 0.01,
            pull_strength: 0.3,
        }
    }
}

impl WanderParamsBuilder {
    pub fn amplitude_x(mut self, v: f64) -> Self {
        self.amplitude.x = v;
        self
    }
    pub fn amplitude_y(mut self, v: f64) -> Self {
        self.amplitude.y = v;
        self
    }
    pub fn amplitude_z(mut self, v: f64) -> Self {
        self.amplitude.z = v;
        self
    }
    pub fn frequency_x(mut self, v: f64) -> Self {
        self.frequency.x = v;
        self
    }
    pub fn frequency_y(mut self, v: f64) -> Self {
        self.frequency.y = v;
        self
    }
    pub fn frequency_z(mut self, v: f64) -> Self {
        self.frequency.z = v;
        self
    }
    pub fn phase_x(mut self, v: f64) -> Self {
        self.phase.x = v;
        self
    }
    pub fn phase_y(mut self, v: f64) -> Self {
        self.phase.y = v;
        self
    }
    pub fn phase_z(mut self, v: f64) -> Self {
        self.phase.z = v;
        self
    }
    pub fn envelope_amplitude(mut self, v: f64) -> Self {
        self.envelope_amplitude = v;
        self
    }
    pub fn envelope_frequency(mut self, v: f64) -> Self {
        self.envelope_frequency = v;
        self
    }
    pub fn pull_strength(mut self, v: f64) -> Self {
        self.pull_strength = v;
        self
    }

    pub fn build(self) -> Result<WanderParams, ConfigError> {
        if !(self.pull_strength.is_finite() && self.pull_strength >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "pull_strength",
                reason: "must be finite and >= 0".into(),
            });
        }
        if !self.envelope_amplitude.is_finite() {
            return Err(ConfigError::InvalidParam {
                field: "envelope_amplitude",
                reason: "must be finite".into(),
            });
        }
        Ok(WanderParams {
            amplitude: self.amplitude,
            frequency: self.frequency,
            phase: self.phase,
            envelope_amplitude: self.envelope_amplitude,
            envelope_frequency: self.envelope_frequency,
            pull_strength: self.pull_strength,
        })
    }
}

pub struct Wander {
    pub params: WanderParams,
    // (wander_center, wander_heading) -- refreshed once per step in `pre_step`, read (possibly
    // many times, once per active boid) in `post_steer`.
    state: Mutex<(Vec3, Vec3)>,
}

impl Wander {
    pub fn new(params: WanderParams) -> Self {
        Wander {
            params,
            state: Mutex::new((Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0))),
        }
    }

    /// The current wander target — this step's flock centroid plus the curve's own offset at
    /// the current elapsed time. Not part of the `StepHook` trait; the design doc's own
    /// "publishes wander_center" requirement.
    pub fn wander_center(&self) -> Vec3 {
        self.state.lock().unwrap().0
    }

    /// The curve's own instantaneous unit heading (its closed-form derivative, normalized) —
    /// the design doc's own "publishes wander_heading" requirement. See the module doc for why
    /// this is the curve's intrinsic heading, not the flock's own bulk translation.
    pub fn wander_heading(&self) -> Vec3 {
        self.state.lock().unwrap().1
    }
}

impl StepHook for Wander {
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

        let center = centroid + self.params.curve_offset(t);
        let velocity = self.params.curve_velocity(t);
        let heading = if velocity.len_sq() > MIN_LEN2 {
            velocity.normalized()
        } else {
            self.state.get_mut().unwrap().1 // hold the previous heading through a momentary stall
        };

        // `&mut self` here means exclusive access to `self.state` too -- `get_mut()` updates it
        // directly, no lock needed (nothing to contend with during `pre_step`, which
        // `run_pre_step_hooks` calls sequentially, once per hook per step).
        *self.state.get_mut().unwrap() = (center, heading);
    }

    fn post_steer(&self, ctx: BoidCtx<'_>, acc: &mut Vec3, _rng: &mut Rng) {
        if self.params.pull_strength <= 0.0 {
            return;
        }
        let center = self.wander_center();
        let offset = center - ctx.pos;
        if offset.len_sq() > MIN_LEN2 {
            *acc += offset.normalized() * self.params.pull_strength;
        }
    }

    /// design/05_viz_contract.md §2.2's `wander` — the same `wander_center`/`wander_heading`
    /// accessors, now also published through the generic `StepHook` checkpoint-field seam.
    fn checkpoint_scene_fields(&self) -> SceneCheckpointFields {
        SceneCheckpointFields {
            wander: Some(WanderSnapshot {
                center: self.wander_center(),
                heading: self.wander_heading(),
            }),
            ..Default::default()
        }
    }

    fn name(&self) -> &'static str {
        "wander"
    }
}

/// Registers `Wander` under the name `"wander"`, reading each of `WanderParams`'s fields from
/// `PluginParams`. The factory type can't return `Result` (design/02_plugins.md §1), so a
/// malformed override falls back to the default rather than panicking — same pattern as every
/// other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("wander", |p: &PluginParams| {
        let d = WanderParamsBuilder::default();
        let params = WanderParams::builder()
            .amplitude_x(p.get_or("wander_amplitude_x", d.amplitude.x))
            .amplitude_y(p.get_or("wander_amplitude_y", d.amplitude.y))
            .amplitude_z(p.get_or("wander_amplitude_z", d.amplitude.z))
            .frequency_x(p.get_or("wander_frequency_x", d.frequency.x))
            .frequency_y(p.get_or("wander_frequency_y", d.frequency.y))
            .frequency_z(p.get_or("wander_frequency_z", d.frequency.z))
            .phase_x(p.get_or("wander_phase_x", d.phase.x))
            .phase_y(p.get_or("wander_phase_y", d.phase.y))
            .phase_z(p.get_or("wander_phase_z", d.phase.z))
            .envelope_amplitude(p.get_or("wander_envelope_amplitude", d.envelope_amplitude))
            .envelope_frequency(p.get_or("wander_envelope_frequency", d.envelope_frequency))
            .pull_strength(p.get_or("wander_pull_strength", d.pull_strength))
            .build()
            .unwrap_or_else(|_| WanderParams::builder().build().expect("defaults are valid"));
        Box::new(Wander::new(params)) as Box<dyn StepHook>
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

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = Wander::new(WanderParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn checkpoint_scene_fields_publishes_the_cached_center_and_heading() {
        let mut hook = Wander::new(WanderParams::builder().build().unwrap());
        let boids = BoidColumns::with_capacity(0);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 5,
        };
        hook.pre_step(&mut view);
        let fields = hook.checkpoint_scene_fields();
        let published = fields.wander.expect("wander must publish after pre_step");
        assert_eq!(published.center, hook.wander_center());
        assert_eq!(published.heading, hook.wander_heading());
    }

    #[test]
    fn wander_center_tracks_the_live_flock_centroid_not_a_fixed_point() {
        let mut hook = Wander::new(
            WanderParams::builder()
                .amplitude_x(0.0)
                .amplitude_y(0.0)
                .amplitude_z(0.0)
                .build()
                .unwrap(), // zero curve amplitude -> wander_center should equal the centroid exactly
        );
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0);
        boids.add(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 1);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);
        assert!((hook.wander_center() - Vec3::ZERO).len() < 1e-9);

        // Move the flock -- wander_center should follow, not stay fixed.
        boids.pos[0] = Vec3::new(90.0, 0.0, 0.0);
        boids.pos[1] = Vec3::new(110.0, 0.0, 0.0);
        let mut view2 = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 1,
        };
        hook.pre_step(&mut view2);
        assert!((hook.wander_center() - Vec3::new(100.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn wander_heading_is_a_real_unit_vector_when_the_curve_is_moving() {
        let mut hook = Wander::new(WanderParams::builder().build().unwrap());
        let boids = BoidColumns::with_capacity(0);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 10,
        };
        hook.pre_step(&mut view);
        let h = hook.wander_heading();
        assert!(h.is_finite());
        assert!(
            (h.len() - 1.0).abs() < 1e-6,
            "expected a unit vector, got len={}",
            h.len()
        );
    }

    #[test]
    fn a_different_dt_changes_wander_center_at_the_same_step_count() {
        // Same G4 real-elapsed-time contract murmur_influencer/murmur_field already use.
        let mut hook_slow = Wander::new(WanderParams::builder().build().unwrap());
        let mut hook_fast = Wander::new(WanderParams::builder().build().unwrap());
        let boids = BoidColumns::with_capacity(0);
        let mut params_slow = core_params(0.01);
        let mut params_fast = core_params(1.0);
        hook_slow.pre_step(&mut SimView {
            boids: &boids,
            core_params: &mut params_slow,
            step_count: 20,
        });
        hook_fast.pre_step(&mut SimView {
            boids: &boids,
            core_params: &mut params_fast,
            step_count: 20,
        });
        assert!(
            (hook_slow.wander_center() - hook_fast.wander_center()).len() > 1e-6,
            "a different dt at the same step_count should reach a different elapsed time, hence \
             a different wander_center"
        );
    }

    #[test]
    fn post_steer_pulls_toward_the_cached_wander_center() {
        let mut hook = Wander::new(
            WanderParams::builder()
                .amplitude_x(0.0)
                .amplitude_y(0.0)
                .amplitude_z(0.0)
                .pull_strength(1.0)
                .build()
                .unwrap(),
        );
        let mut boids = BoidColumns::with_capacity(1);
        boids.add(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey, 0);
        let mut params = core_params(1.0);
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view); // wander_center == centroid == (-5, 0, 0), so no pull on this boid

        let domain = StubDomain;
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::new(-15.0, 0.0, 0.0), // away from the cached centroid at x=-5
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc, &mut rng::for_boid(1, 2, 3));
        assert!(acc.x > 0.0, "must pull toward the centroid, got {:?}", acc);
    }

    #[test]
    fn zero_pull_strength_produces_no_force() {
        let hook = Wander::new(WanderParams::builder().pull_strength(0.0).build().unwrap());
        let domain = StubDomain;
        let params = core_params(1.0);
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::new(-1000.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc, &mut rng::for_boid(1, 2, 3));
        assert_eq!(acc, Vec3::ZERO);
    }

    #[test]
    fn builder_rejects_a_negative_pull_strength() {
        assert!(WanderParams::builder().pull_strength(-1.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("wander", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "wander");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("wander_pull_strength", -5.0);
        let hook = reg.resolve_step_hook("wander", &bad).unwrap();
        assert_eq!(hook.name(), "wander");
    }
}
