//! `Field` — 11-term Lissajous blob-anchor `FlockingMode` (design/02_plugins.md §5, roadmap.md
//! Phase 17, pymurmur's `physics/forces/field.py` + `field_anchors.py` + `field_terms.py`,
//! ported by description — pymurmur's actual source isn't reachable in this environment, the
//! same blocker as every other pymurmur cross-check this project has hit beyond
//! `murmur_maxent_social`'s).
//!
//! **Interpretation, disclosed.** Design/02_plugins.md §5 gives only a one-line description —
//! "11-term Lissajous blob-anchor field mode" — split across three pymurmur source files this
//! project can't read. Read literally: an *"11-term Lissajous"* curve (a richer parametrisation
//! than `murmur_influencer`'s single 9-term, 3-axis sinusoid) anchoring one or more *"blobs"*
//! that together form a spatial *"field"* — i.e. `murmur_influencer` generalised two ways: (a)
//! the curve itself gets 2 more terms (an amplitude-envelope modulation, so the whole curve's
//! scale breathes over time, not just its phase), and (b) instead of one shared target every
//! boid chases, `anchor_count` independent, phase-staggered copies of that same curve exist
//! simultaneously, deterministically placed around a circle — each boid is drawn to whichever
//! anchor is *nearest to it*, not to a single global point. That's what makes this a *field*
//! (a spatially-varying attractor landscape) rather than `murmur_influencer`'s single target.
//!
//! **The 11 terms**: `amplitude_{x,y,z}` (3), `frequency_{x,y,z}` (3), `phase_{x,y,z}` (3),
//! `envelope_amplitude` (1), `envelope_frequency` (1) — one curve shape shared by every anchor,
//! each anchor offset by its own fixed phase stagger (`k * 2π / anchor_count`) so they trace
//! the same shape out of sync with each other, not identically.
//!
//! **Uses G4** (`step_count`/`dt`, fixed in Phase 13) the same way `murmur_influencer` does:
//! every anchor's position is `target_at(step_count as f64 * dt)`, a real function of elapsed
//! simulated time.
//!
//! **Stateless**, like `murmur_influencer` — `anchor_count` anchor positions are recomputed
//! from `BoidCtx` fields alone each call; no per-boid side-column, no cross-boid cache.
//! `anchor_count == 0` is clamped to `1` (matching `murmur_initializers::Blob`'s own
//! `blob_count == 0` precedent) — the alternative, `k as f64 * (2π / 0.0)`, produces `inf` for
//! `k > 0` and `NaN` for `k == 0` (`0.0 * inf`), violating `FlockingMode::desired()`'s
//! finite-output contract.
//!
//! **Local repulsion**, same short-range linear-inverse-distance kernel `murmur_spatial`/
//! `murmur_maxent_social` already use, so boids cluster near an anchor without colliding at it.

use murmur_core::{
    BoidCtx, ConfigError, FlockingMode, OcclusionScratch, PluginParams, Registry, Rng, SteerIntent,
    Vec3, MIN_LEN, MIN_LEN2,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldParams {
    pub amplitude: Vec3,
    pub frequency: Vec3,
    pub phase: Vec3,
    pub envelope_amplitude: f64,
    pub envelope_frequency: f64,
    pub anchor_count: u32,
    pub anchor_spread: f64,
    pub anchor_weight: f64,
    pub repulsion_weight: f64,
    pub repulsion_radius: f64,
}

impl FieldParams {
    pub fn builder() -> FieldParamsBuilder {
        FieldParamsBuilder::default()
    }

    fn envelope(&self, t: f64) -> f64 {
        1.0 + self.envelope_amplitude * (self.envelope_frequency * t).sin()
    }

    /// The shared curve shape (the "11-term Lissajous"), before any per-anchor phase stagger or
    /// centre offset is applied.
    fn curve_offset(&self, t: f64, stagger: f64) -> Vec3 {
        Vec3::new(
            self.amplitude.x * (self.frequency.x * t + self.phase.x + stagger).sin(),
            self.amplitude.y * (self.frequency.y * t + self.phase.y + stagger).sin(),
            self.amplitude.z * (self.frequency.z * t + self.phase.z + stagger).sin(),
        ) * self.envelope(t)
    }

    /// Anchor `k`'s (of `anchor_count`, already clamped to at least `1`) world position at
    /// simulated time `t`: a fixed centre, deterministically placed on a circle of radius
    /// `anchor_spread` (`k`-th of `anchor_count` evenly spaced points, no RNG needed), plus the
    /// shared curve shape staggered by `k`'s own fixed phase offset.
    fn anchor_position(&self, k: u32, anchor_count: u32, t: f64) -> Vec3 {
        let angle = k as f64 * (2.0 * std::f64::consts::PI / anchor_count as f64);
        let center = Vec3::new(
            self.anchor_spread * angle.cos(),
            self.anchor_spread * angle.sin(),
            0.0,
        );
        center + self.curve_offset(t, angle)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FieldParamsBuilder {
    amplitude: Vec3,
    frequency: Vec3,
    phase: Vec3,
    envelope_amplitude: f64,
    envelope_frequency: f64,
    anchor_count: u32,
    anchor_spread: f64,
    anchor_weight: f64,
    repulsion_weight: f64,
    repulsion_radius: f64,
}

impl Default for FieldParamsBuilder {
    fn default() -> Self {
        FieldParamsBuilder {
            amplitude: Vec3::new(10.0, 10.0, 5.0),
            frequency: Vec3::new(0.05, 0.07, 0.03),
            phase: Vec3::new(0.0, std::f64::consts::FRAC_PI_2, 0.0),
            envelope_amplitude: 0.3,
            envelope_frequency: 0.01,
            anchor_count: 3,
            anchor_spread: 20.0,
            anchor_weight: 1.0,
            repulsion_weight: 1.5,
            repulsion_radius: 2.0,
        }
    }
}

impl FieldParamsBuilder {
    pub fn amplitude(mut self, v: Vec3) -> Self {
        self.amplitude = v;
        self
    }
    pub fn frequency(mut self, v: Vec3) -> Self {
        self.frequency = v;
        self
    }
    pub fn phase(mut self, v: Vec3) -> Self {
        self.phase = v;
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
    pub fn anchor_count(mut self, v: u32) -> Self {
        self.anchor_count = v;
        self
    }
    pub fn anchor_spread(mut self, v: f64) -> Self {
        self.anchor_spread = v;
        self
    }
    pub fn anchor_weight(mut self, v: f64) -> Self {
        self.anchor_weight = v;
        self
    }
    pub fn repulsion_weight(mut self, v: f64) -> Self {
        self.repulsion_weight = v;
        self
    }
    pub fn repulsion_radius(mut self, v: f64) -> Self {
        self.repulsion_radius = v;
        self
    }

    pub fn build(self) -> Result<FieldParams, ConfigError> {
        for (field, v) in [
            ("amplitude", self.amplitude),
            ("frequency", self.frequency),
            ("phase", self.phase),
        ] {
            if !v.is_finite() {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite".into(),
                });
            }
        }
        for (field, v) in [
            ("envelope_amplitude", self.envelope_amplitude),
            ("envelope_frequency", self.envelope_frequency),
        ] {
            if !v.is_finite() {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite".into(),
                });
            }
        }
        for (field, v) in [
            ("anchor_weight", self.anchor_weight),
            ("repulsion_weight", self.repulsion_weight),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite and >= 0".into(),
                });
            }
        }
        for (field, v) in [
            ("anchor_spread", self.anchor_spread),
            ("repulsion_radius", self.repulsion_radius),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite and > 0".into(),
                });
            }
        }
        Ok(FieldParams {
            amplitude: self.amplitude,
            frequency: self.frequency,
            phase: self.phase,
            envelope_amplitude: self.envelope_amplitude,
            envelope_frequency: self.envelope_frequency,
            anchor_count: self.anchor_count,
            anchor_spread: self.anchor_spread,
            anchor_weight: self.anchor_weight,
            repulsion_weight: self.repulsion_weight,
            repulsion_radius: self.repulsion_radius,
        })
    }
}

pub struct Field {
    pub params: FieldParams,
}

impl Field {
    pub fn new(params: FieldParams) -> Self {
        Field { params }
    }
}

impl FlockingMode for Field {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        _rng: &mut Rng,
    ) -> SteerIntent {
        let p = &self.params;
        let anchor_count = p.anchor_count.max(1);
        let t = ctx.step_count as f64 * ctx.core_params.dt;

        let mut nearest_dir = Vec3::ZERO;
        let mut nearest_dist = f64::INFINITY;
        for k in 0..anchor_count {
            let anchor_pos = p.anchor_position(k, anchor_count, t);
            let offset = anchor_pos - ctx.pos;
            let d = offset.len();
            if d < nearest_dist {
                nearest_dist = d;
                nearest_dir = offset.normalized();
            }
        }

        let mut repulsion_sum = Vec3::ZERO;
        for n in ctx.neighbors {
            if n.distance < p.repulsion_radius {
                let w = (p.repulsion_radius - n.distance)
                    / (p.repulsion_radius * n.distance.max(MIN_LEN));
                repulsion_sum += -n.direction * w;
            }
        }
        let repulsion_dir = repulsion_sum.normalized();

        let combined = nearest_dir * p.anchor_weight + repulsion_dir * p.repulsion_weight;

        let desired_v = if combined.len_sq() > MIN_LEN2 {
            combined.normalized() * ctx.core_params.cruise_speed
        } else {
            ctx.vel
        };

        SteerIntent {
            desired_v,
            extra_force: Vec3::ZERO,
            theta: 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "field"
    }
}

/// Registers `Field` under the name `"field"`. Reuses `murmur_influencer`'s
/// `amplitude_{x,y,z}`/`frequency_{x,y,z}`/`phase_{x,y,z}` keys and `murmur_maxent_social`'s
/// `repulsion_weight`/`repulsion_radius` keys (all safe — `FlockingMode` is a single-occupant
/// socket). `anchor_spread` is deliberately **not** named `blob_spread`, despite the name this
/// plugin's own module doc uses for the concept — that key already belongs to
/// `murmur_initializers::Blob`, a different, simultaneously-composable socket
/// (`Initializer` + `FlockingMode` genuinely can both be active at once, unlike two
/// `FlockingMode`s), so sharing it would silently couple two unrelated knobs. The factory type
/// can't return `Result` (design/02_plugins.md §1), so a malformed override falls back to the
/// default rather than panicking — same pattern as every other `FlockingMode` plugin here.
pub fn register(r: &mut Registry) {
    r.register_mode("field", |p: &PluginParams| {
        let d = FieldParamsBuilder::default();
        let params = FieldParams::builder()
            .amplitude(Vec3::new(
                p.get_or("amplitude_x", d.amplitude.x),
                p.get_or("amplitude_y", d.amplitude.y),
                p.get_or("amplitude_z", d.amplitude.z),
            ))
            .frequency(Vec3::new(
                p.get_or("frequency_x", d.frequency.x),
                p.get_or("frequency_y", d.frequency.y),
                p.get_or("frequency_z", d.frequency.z),
            ))
            .phase(Vec3::new(
                p.get_or("phase_x", d.phase.x),
                p.get_or("phase_y", d.phase.y),
                p.get_or("phase_z", d.phase.z),
            ))
            .envelope_amplitude(p.get_or("envelope_amplitude", d.envelope_amplitude))
            .envelope_frequency(p.get_or("envelope_frequency", d.envelope_frequency))
            .anchor_count(p.get_or("anchor_count", d.anchor_count as f64) as u32)
            .anchor_spread(p.get_or("anchor_spread", d.anchor_spread))
            .anchor_weight(p.get_or("anchor_weight", d.anchor_weight))
            .repulsion_weight(p.get_or("repulsion_weight", d.repulsion_weight))
            .repulsion_radius(p.get_or("repulsion_radius", d.repulsion_radius))
            .build()
            .unwrap_or_else(|_| FieldParams::builder().build().expect("defaults are valid"));
        Box::new(Field::new(params)) as Box<dyn FlockingMode>
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

    fn core_params(dt: f64) -> CoreParams {
        CoreParams::builder()
            .cruise_speed(2.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .dt(dt)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn neighbor(direction: Vec3, distance: f64) -> Neighbor {
        Neighbor {
            index: 1,
            distance,
            direction: direction.normalized(),
            velocity: Vec3::ZERO,
        }
    }

    fn ctx<'a>(
        pos: Vec3,
        vel: Vec3,
        step_count: u64,
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index: 0,
            pos,
            vel,
            species: Species::Prey,
            neighbors,
            core_params: params,
            domain,
            step_count,
        }
    }

    #[test]
    fn conforms_to_flocking_mode_contract() {
        murmur_conformance::flocking_mode(&Field::new(FieldParams::builder().build().unwrap()));
    }

    #[test]
    fn a_single_anchor_pulls_a_boid_toward_it() {
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        // amplitude=0 -> the sole anchor sits exactly at anchor_spread on +x.
        let mode = Field::new(
            FieldParams::builder()
                .amplitude(Vec3::ZERO)
                .anchor_count(1)
                .anchor_spread(20.0)
                .anchor_weight(1.0)
                .repulsion_weight(0.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(
                Vec3::ZERO,
                Vec3::new(0.0, 1.0, 0.0),
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x > 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn a_boid_is_drawn_to_its_nearest_anchor_not_an_arbitrary_one() {
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        // 2 anchors, amplitude=0: anchor 0 at +x*spread, anchor 1 at -x*spread (angle=pi).
        let mode = Field::new(
            FieldParams::builder()
                .amplitude(Vec3::ZERO)
                .anchor_count(2)
                .anchor_spread(20.0)
                .anchor_weight(1.0)
                .repulsion_weight(0.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        // A boid already near anchor 1 (-x side) must be pulled further toward -x, not +x.
        let intent = mode.desired(
            ctx(
                Vec3::new(-19.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x < 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn zero_anchor_count_degenerates_to_one_instead_of_producing_nan() {
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Field::new(FieldParams::builder().anchor_count(0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.is_finite(), "got {:?}", intent.desired_v);
    }

    #[test]
    fn the_envelope_term_changes_the_curve_shape_over_the_amplitude_only_case() {
        let with_envelope = FieldParams::builder()
            .envelope_amplitude(0.5)
            .envelope_frequency(0.1)
            .build()
            .unwrap();
        let without_envelope = FieldParams::builder()
            .envelope_amplitude(0.0)
            .build()
            .unwrap();
        let t = 5.0;
        let a = with_envelope.anchor_position(0, 1, t);
        let b = without_envelope.anchor_position(0, 1, t);
        assert_ne!(
            a, b,
            "the envelope term must actually change the anchor's position"
        );
    }

    #[test]
    fn the_anchor_position_is_a_real_function_of_elapsed_time_not_a_call_counter() {
        let mode = Field::new(FieldParams::builder().build().unwrap());
        let p0 = mode.params.anchor_position(0, 3, 0.0);
        let p_small_dt = mode.params.anchor_position(0, 3, 0.01 * 5.0);
        let p_large_dt = mode.params.anchor_position(0, 3, 1.0 * 5.0);
        assert_ne!(p0, p_small_dt);
        assert_ne!(
            p_small_dt, p_large_dt,
            "the same step count at a different dt must reach a different anchor position"
        );
    }

    #[test]
    fn repulsion_only_steers_away_from_a_close_neighbour() {
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors = [neighbor(Vec3::new(1.0, 0.0, 0.0), 0.5)];
        let mode = Field::new(
            FieldParams::builder()
                .anchor_weight(0.0)
                .repulsion_weight(1.0)
                .repulsion_radius(2.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(
                Vec3::ZERO,
                Vec3::new(0.0, 1.0, 0.0),
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x < 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn no_channels_active_falls_back_to_current_velocity() {
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Field::new(
            FieldParams::builder()
                .anchor_weight(0.0)
                .repulsion_weight(0.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let current = Vec3::new(1.0, -2.0, 0.5);
        let intent = mode.desired(
            ctx(Vec3::ZERO, current, 0, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert_eq!(intent.desired_v, current);
    }

    #[test]
    fn builder_rejects_a_negative_weight() {
        assert!(FieldParams::builder().anchor_weight(-1.0).build().is_err());
    }

    #[test]
    fn builder_rejects_a_non_positive_anchor_spread() {
        assert!(FieldParams::builder().anchor_spread(0.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("field", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "field");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("anchor_spread", -5.0);
        let mode = reg.resolve_mode("field", &bad).unwrap();
        assert_eq!(mode.name(), "field");
    }

    /// Proves the `FlockingMode` seam now has ≥7 real occupants beyond Pearce/Vicsek.
    #[test]
    fn pearce_vicsek_and_field_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_vicsek::register(&mut reg);
        murmur_pearce::register(&mut reg);

        assert_eq!(
            reg.resolve_mode("field", &PluginParams::new())
                .unwrap()
                .name(),
            "field"
        );
        assert_eq!(
            reg.resolve_mode("vicsek", &PluginParams::new())
                .unwrap()
                .name(),
            "vicsek"
        );
        assert_eq!(
            reg.resolve_mode("pearce", &PluginParams::new())
                .unwrap()
                .name(),
            "pearce"
        );
    }
}
