//! `MaxentSocial` — 5-channel generative `FlockingMode` (design/02_plugins.md §5, roadmap.md
//! Phase 17), after Cai, Zhang, Chen & Wang 2021, "A General Statistical Mechanics Framework
//! for the Collective Motion of Animals" (arXiv:2112.15560, `sci/2112.15560v1.pdf` — read
//! directly for this port, unlike most other ports in this project, where the cited source was
//! unreachable). The paper derives a maximum-entropy velocity distribution whose mean is driven
//! by a force `F_i` (its eq. 5) summing five weighted channels — alignment, repulsion,
//! attraction, boundary, and "desire" (an animal's own will/goal direction) — and shows the
//! Vicsek model and Helbing's social-force model are both special cases of it (its §I.B).
//!
//! **Adapting the paper's raw force into this project's `desired_v` contract.** The paper's
//! `F_i` (eq. 5) combines with a kinetic-inertia term (eq. 14–15) to update a raw velocity
//! directly; `FlockingMode::desired()` instead returns a *target* velocity for the composed
//! `SteeringModifier` to steer toward. Each channel here is reduced to a **unit direction**
//! (alignment/repulsion/attraction: summed then renormalised; boundary: a single
//! proximity-graded direction, not a multi-source sum, so it keeps its own magnitude rather
//! than being flattened to unit length; desire: the configured goal direction, already unit),
//! then blended by each channel's weight and renormalised to `cruise_speed` — the same "blend
//! unit directions, weights control the blend ratio" approach `murmur_spatial`/`murmur_vicsek`
//! already use, chosen for the same reason: it keeps every channel's weight comparable
//! regardless of neighbour count or density, which the paper's own raw-force sum does not
//! guarantee on its own.
//!
//! **Channel definitions, each disclosed against the paper's own eq. 5 text:**
//! - **Alignment** (`f_ali`): mean unit heading of self + every neighbour the composed
//!   `NeighborSelection` returns (the paper allows either metric or topological neighbour sets
//!   per channel — "the definition of neighbors can be either metric or topological in our
//!   framework" — deferred here to whichever `NeighborSelection` is composed, not re-filtered).
//!   Includes the observer's own heading in the mean, deliberately matching `Vicsek`'s own
//!   "classic Vicsek includes self" convention exactly.
//! - **Repulsion** (`f_rep`): neighbours within `repulsion_radius`, summed with a
//!   linear-inverse-distance weight (matching `murmur_spatial`'s own separation kernel), then
//!   renormalised — short-range, strongest as distance → 0.
//! - **Attraction** (`f_att`): neighbours within `attraction_radius`, summed as a mean offset
//!   toward them (unweighted by distance, matching `murmur_spatial`'s cohesion kernel) — the
//!   paper's own worked bird-flocking example describes exactly this as a long-range
//!   "roost"/merge-flocks effect, metric-defined.
//! - **Boundary** (`f_bou`): **G5** (roadmap.md §12, `Domain::boundary_distance()`), fixed
//!   specifically to build this channel. `g_bou(d) = exp(-max(d, 0) / boundary_decay)` —
//!   Helbing's own exponential wall-falloff, the special case §I.B of the paper claims. **Sign
//!   convention, disclosed**: the paper's own text defines `u_bou` pointing *from* the animal
//!   *to* the nearest boundary point (outward) — this implementation instead points the
//!   boundary channel **inward** (`Domain::boundary_distance()`'s own contract), since an
//!   outward-pointing contribution cannot reduce to Helbing's actual *repulsive* wall force
//!   without an unstated sign flip buried in the paper's `g_bou`/`λ_bou` terms. Implementing the
//!   net repulsive effect directly is more useful and less error-prone than replicating a sign
//!   convention the paper itself doesn't fully pin down.
//! - **Desire** (`f_des`): a fixed configured `goal_direction` — the paper's own "will of the
//!   animal," a genuine external input, not derived from neighbours or state.
//!
//! **Reduces to Vicsek** (§I.B's own claimed special case, verified directly in
//! `tests::with_only_alignment_active_it_matches_vicseks_own_mean_heading_rule` against
//! `murmur_vicsek` itself): with only the alignment channel active, this mode's direction is
//! exactly `Vicsek`'s own full-coupling (`couplage=1.0`, no noise) mean-neighbour-heading rule.

use murmur_core::{
    BoidCtx, ConfigError, FlockingMode, OcclusionScratch, PluginParams, Registry, Rng, SteerIntent,
    Vec3, MIN_LEN, MIN_LEN2,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxentSocialParams {
    pub align_weight: f64,
    pub repulsion_weight: f64,
    pub attraction_weight: f64,
    pub boundary_weight: f64,
    pub desire_weight: f64,
    pub repulsion_radius: f64,
    pub attraction_radius: f64,
    pub boundary_decay: f64,
    pub goal_direction: Vec3,
}

impl MaxentSocialParams {
    pub fn builder() -> MaxentSocialParamsBuilder {
        MaxentSocialParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaxentSocialParamsBuilder {
    align_weight: f64,
    repulsion_weight: f64,
    attraction_weight: f64,
    boundary_weight: f64,
    desire_weight: f64,
    repulsion_radius: f64,
    attraction_radius: f64,
    boundary_decay: f64,
    goal_direction: Vec3,
}

impl Default for MaxentSocialParamsBuilder {
    fn default() -> Self {
        MaxentSocialParamsBuilder {
            align_weight: 1.0,
            repulsion_weight: 1.5,
            attraction_weight: 0.5,
            boundary_weight: 1.0,
            desire_weight: 0.0,
            repulsion_radius: 2.0,
            attraction_radius: 10.0,
            boundary_decay: 5.0,
            goal_direction: Vec3::new(1.0, 0.0, 0.0),
        }
    }
}

impl MaxentSocialParamsBuilder {
    pub fn align_weight(mut self, v: f64) -> Self {
        self.align_weight = v;
        self
    }
    pub fn repulsion_weight(mut self, v: f64) -> Self {
        self.repulsion_weight = v;
        self
    }
    pub fn attraction_weight(mut self, v: f64) -> Self {
        self.attraction_weight = v;
        self
    }
    pub fn boundary_weight(mut self, v: f64) -> Self {
        self.boundary_weight = v;
        self
    }
    pub fn desire_weight(mut self, v: f64) -> Self {
        self.desire_weight = v;
        self
    }
    pub fn repulsion_radius(mut self, v: f64) -> Self {
        self.repulsion_radius = v;
        self
    }
    pub fn attraction_radius(mut self, v: f64) -> Self {
        self.attraction_radius = v;
        self
    }
    pub fn boundary_decay(mut self, v: f64) -> Self {
        self.boundary_decay = v;
        self
    }
    pub fn goal_direction(mut self, v: Vec3) -> Self {
        self.goal_direction = v;
        self
    }

    pub fn build(self) -> Result<MaxentSocialParams, ConfigError> {
        for (field, v) in [
            ("align_weight", self.align_weight),
            ("repulsion_weight", self.repulsion_weight),
            ("attraction_weight", self.attraction_weight),
            ("boundary_weight", self.boundary_weight),
            ("desire_weight", self.desire_weight),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite and >= 0".into(),
                });
            }
        }
        for (field, v) in [
            ("repulsion_radius", self.repulsion_radius),
            ("attraction_radius", self.attraction_radius),
            ("boundary_decay", self.boundary_decay),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite and > 0".into(),
                });
            }
        }
        if !self.goal_direction.is_finite() {
            return Err(ConfigError::InvalidParam {
                field: "goal_direction",
                reason: "must be finite".into(),
            });
        }
        Ok(MaxentSocialParams {
            align_weight: self.align_weight,
            repulsion_weight: self.repulsion_weight,
            attraction_weight: self.attraction_weight,
            boundary_weight: self.boundary_weight,
            desire_weight: self.desire_weight,
            repulsion_radius: self.repulsion_radius,
            attraction_radius: self.attraction_radius,
            boundary_decay: self.boundary_decay,
            goal_direction: self.goal_direction,
        })
    }
}

pub struct MaxentSocial {
    pub params: MaxentSocialParams,
}

impl MaxentSocial {
    pub fn new(params: MaxentSocialParams) -> Self {
        MaxentSocial { params }
    }
}

impl FlockingMode for MaxentSocial {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        _rng: &mut Rng,
    ) -> SteerIntent {
        let p = &self.params;

        // Alignment: mean unit heading of self + every composed neighbour -- deliberately
        // matching `Vicsek`'s own "classic Vicsek includes self in the neighbourhood average"
        // convention exactly (not just "similar"), so the align-only reduction below is a
        // genuine equality, not an approximation.
        let heading = ctx.vel.normalized();
        let mut align_sum = heading;
        for n in ctx.neighbors {
            if n.velocity.len_sq() > MIN_LEN2 {
                align_sum += n.velocity.normalized();
            }
        }
        let align_dir = if align_sum.len_sq() > MIN_LEN2 {
            align_sum.normalized()
        } else {
            heading
        };

        // Repulsion: short-range, linear-inverse-distance weighted, summed then renormalised.
        let mut repulsion_sum = Vec3::ZERO;
        for n in ctx.neighbors {
            if n.distance < p.repulsion_radius {
                let w = (p.repulsion_radius - n.distance)
                    / (p.repulsion_radius * n.distance.max(MIN_LEN));
                repulsion_sum += -n.direction * w;
            }
        }
        let repulsion_dir = repulsion_sum.normalized();

        // Attraction: long-range mean offset toward neighbours within attraction_radius.
        let mut attraction_sum = Vec3::ZERO;
        for n in ctx.neighbors {
            if n.distance < p.attraction_radius {
                attraction_sum += n.direction;
            }
        }
        let attraction_dir = attraction_sum.normalized();

        // Boundary: G5. A single proximity-graded direction, not a multi-source sum, so its
        // magnitude (g_bou, already in (0, 1]) is kept rather than renormalised away.
        let boundary_contribution = match ctx.domain.boundary_distance(ctx.pos) {
            Some((distance, inward_normal)) => {
                let g_bou = (-distance.max(0.0) / p.boundary_decay).exp();
                inward_normal * g_bou
            }
            None => Vec3::ZERO,
        };

        let desire_dir = p.goal_direction.normalized();

        let combined = align_dir * p.align_weight
            + repulsion_dir * p.repulsion_weight
            + attraction_dir * p.attraction_weight
            + boundary_contribution * p.boundary_weight
            + desire_dir * p.desire_weight;

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
        "maxent_social"
    }
}

/// Registers `MaxentSocial` under the name `"maxent_social"`, reading each of
/// `MaxentSocialParams`'s fields from `PluginParams` (`goal_direction` as `goal_x`/`goal_y`/
/// `goal_z`). The factory type can't return `Result` (design/02_plugins.md §1), so a malformed
/// override falls back to the default rather than panicking — same pattern as every other
/// `FlockingMode` plugin here.
pub fn register(r: &mut Registry) {
    r.register_mode("maxent_social", |p: &PluginParams| {
        let d = MaxentSocialParamsBuilder::default();
        let params = MaxentSocialParams::builder()
            .align_weight(p.get_or("align_weight", d.align_weight))
            .repulsion_weight(p.get_or("repulsion_weight", d.repulsion_weight))
            .attraction_weight(p.get_or("attraction_weight", d.attraction_weight))
            .boundary_weight(p.get_or("boundary_weight", d.boundary_weight))
            .desire_weight(p.get_or("desire_weight", d.desire_weight))
            .repulsion_radius(p.get_or("repulsion_radius", d.repulsion_radius))
            .attraction_radius(p.get_or("attraction_radius", d.attraction_radius))
            .boundary_decay(p.get_or("boundary_decay", d.boundary_decay))
            .goal_direction(Vec3::new(
                p.get_or("goal_x", d.goal_direction.x),
                p.get_or("goal_y", d.goal_direction.y),
                p.get_or("goal_z", d.goal_direction.z),
            ))
            .build()
            .unwrap_or_else(|_| {
                MaxentSocialParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(MaxentSocial::new(params)) as Box<dyn FlockingMode>
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

    /// A bounded stub whose boundary sits exactly `dist` along +x from `pos`, purely to drive
    /// the boundary channel without depending on a real bounded `Domain` plugin's own geometry.
    struct StubBoundary {
        dist: f64,
    }
    impl Domain for StubBoundary {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn boundary_distance(&self, _pos: Vec3) -> Option<(f64, Vec3)> {
            Some((self.dist, Vec3::new(-1.0, 0.0, 0.0))) // inward = -x
        }
        fn name(&self) -> &'static str {
            "stub_boundary"
        }
    }

    fn core_params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(2.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn neighbor(direction: Vec3, distance: f64, velocity: Vec3) -> Neighbor {
        Neighbor {
            index: 1,
            distance,
            direction: direction.normalized(),
            velocity,
        }
    }

    fn ctx<'a>(
        vel: Vec3,
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel,
            species: Species::Prey,
            neighbors,
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    fn only(weight_field: &str) -> MaxentSocialParamsBuilder {
        let mut b = MaxentSocialParams::builder()
            .align_weight(0.0)
            .repulsion_weight(0.0)
            .attraction_weight(0.0)
            .boundary_weight(0.0)
            .desire_weight(0.0);
        b = match weight_field {
            "align" => b.align_weight(1.0),
            "repulsion" => b.repulsion_weight(1.0),
            "attraction" => b.attraction_weight(1.0),
            "boundary" => b.boundary_weight(1.0),
            "desire" => b.desire_weight(1.0),
            _ => unreachable!(),
        };
        b
    }

    #[test]
    fn conforms_to_flocking_mode_contract() {
        murmur_conformance::flocking_mode(&MaxentSocial::new(
            MaxentSocialParams::builder().build().unwrap(),
        ));
    }

    #[test]
    fn with_only_alignment_active_it_matches_vicseks_own_mean_heading_rule() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors = [
            neighbor(Vec3::new(1.0, 0.0, 0.0), 3.0, Vec3::new(0.0, 1.0, 0.0)),
            neighbor(Vec3::new(0.0, 1.0, 0.0), 4.0, Vec3::new(1.0, 0.0, 0.0)),
        ];
        let current_vel = Vec3::new(0.0, 0.0, 1.0);

        let maxent = MaxentSocial::new(only("align").build().unwrap());
        let vicsek = murmur_vicsek::Vicsek::new(
            murmur_vicsek::VicsekParams::builder()
                .couplage(1.0)
                .diffusion(0.0)
                .build()
                .unwrap(),
        );

        let mut scratch = OcclusionScratch::default();
        let mut rng1 = murmur_core::rng::for_boid(1, 0, 0);
        let mut rng2 = murmur_core::rng::for_boid(1, 0, 0);
        let a = maxent.desired(
            ctx(current_vel, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng1,
        );
        let b = vicsek.desired(
            ctx(current_vel, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng2,
        );

        assert!(
            (a.desired_v - b.desired_v).len() < 1e-9,
            "align-only MaxentSocial should exactly match Vicsek(couplage=1, diffusion=0): got {:?} vs {:?}",
            a.desired_v,
            b.desired_v
        );
    }

    #[test]
    fn repulsion_only_steers_away_from_a_close_neighbour() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors = [neighbor(Vec3::new(1.0, 0.0, 0.0), 0.5, Vec3::ZERO)];
        let mode = MaxentSocial::new(only("repulsion").repulsion_radius(2.0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x < 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn a_neighbour_beyond_the_repulsion_radius_contributes_nothing_to_repulsion() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors = [neighbor(Vec3::new(1.0, 0.0, 0.0), 5.0, Vec3::ZERO)];
        let mode = MaxentSocial::new(only("repulsion").repulsion_radius(2.0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        // No repulsion, and no other channel active -> falls back to current velocity.
        assert_eq!(intent.desired_v, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn attraction_only_steers_toward_a_long_range_neighbour() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors = [neighbor(Vec3::new(1.0, 0.0, 0.0), 8.0, Vec3::ZERO)];
        let mode = MaxentSocial::new(only("attraction").attraction_radius(10.0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x > 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn desire_only_steers_exactly_toward_the_configured_goal_direction() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = MaxentSocial::new(
            only("desire")
                .goal_direction(Vec3::new(0.0, 0.0, 1.0))
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(1.0, 0.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(
            (intent.desired_v.normalized() - Vec3::new(0.0, 0.0, 1.0)).len() < 1e-9,
            "got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn boundary_only_under_an_unbounded_domain_contributes_nothing() {
        let params = core_params();
        let domain = StubDomain; // boundary_distance() default: None
        let neighbors: [Neighbor; 0] = [];
        let mode = MaxentSocial::new(only("boundary").build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let current = Vec3::new(0.0, 3.0, 0.0);
        let intent = mode.desired(
            ctx(current, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert_eq!(
            intent.desired_v, current,
            "no boundary concept -> falls back to current velocity"
        );
    }

    #[test]
    fn boundary_only_points_inward_near_a_wall() {
        let params = core_params();
        let domain = StubBoundary { dist: 0.5 };
        let neighbors: [Neighbor; 0] = [];
        let mode = MaxentSocial::new(only("boundary").boundary_decay(5.0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(
            intent.desired_v.x < 0.0,
            "must push inward (-x), got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn boundary_channel_decays_with_distance_relative_to_a_fixed_desire_baseline() {
        // With boundary + desire both active and desire pointing a different, fixed direction
        // (+y), the blend must lean toward "inward" (-x) when very close to the wall and lean
        // toward the desire baseline (+y) when far from it -- a direct, testable proof that
        // g_bou actually decays with distance, not just that the direction is correct at one
        // fixed distance.
        let params = core_params();
        let neighbors: [Neighbor; 0] = [];
        let mode = MaxentSocial::new(
            MaxentSocialParams::builder()
                .align_weight(0.0)
                .repulsion_weight(0.0)
                .attraction_weight(0.0)
                .boundary_weight(1.0)
                .desire_weight(0.05) // small, fixed baseline -- boundary should dominate near the wall
                .boundary_decay(2.0)
                .goal_direction(Vec3::new(0.0, 1.0, 0.0))
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);

        let near_domain = StubBoundary { dist: 0.01 };
        let near = mode.desired(
            ctx(Vec3::ZERO, &neighbors, &params, &near_domain),
            &mut scratch,
            &mut rng,
        );
        let far_domain = StubBoundary { dist: 100.0 };
        let far = mode.desired(
            ctx(Vec3::ZERO, &neighbors, &params, &far_domain),
            &mut scratch,
            &mut rng,
        );

        assert!(
            near.desired_v.x < -0.9 * params.cruise_speed,
            "near the wall: got {:?}",
            near.desired_v
        );
        assert!(
            far.desired_v.y > 0.9 * params.cruise_speed,
            "far from the wall: got {:?}",
            far.desired_v
        );
    }

    #[test]
    fn no_channels_active_falls_back_to_current_velocity() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = MaxentSocial::new(
            MaxentSocialParams::builder()
                .align_weight(0.0)
                .repulsion_weight(0.0)
                .attraction_weight(0.0)
                .boundary_weight(0.0)
                .desire_weight(0.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let current = Vec3::new(1.0, -2.0, 0.5);
        let intent = mode.desired(
            ctx(current, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert_eq!(intent.desired_v, current);
    }

    #[test]
    fn builder_rejects_a_negative_weight() {
        assert!(MaxentSocialParams::builder()
            .align_weight(-1.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_non_positive_radius() {
        assert!(MaxentSocialParams::builder()
            .repulsion_radius(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg
            .resolve_mode("maxent_social", &PluginParams::new())
            .unwrap();
        assert_eq!(mode.name(), "maxent_social");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("repulsion_radius", -5.0);
        let mode = reg.resolve_mode("maxent_social", &bad).unwrap();
        assert_eq!(mode.name(), "maxent_social");
    }

    /// Proves the `FlockingMode` seam now has ≥6 real occupants beyond Pearce/Vicsek.
    #[test]
    fn pearce_vicsek_and_maxent_social_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_vicsek::register(&mut reg);
        murmur_pearce::register(&mut reg);

        assert_eq!(
            reg.resolve_mode("maxent_social", &PluginParams::new())
                .unwrap()
                .name(),
            "maxent_social"
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
