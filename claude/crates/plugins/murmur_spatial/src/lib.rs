//! `Spatial` — classic topological Reynolds flocking (design/02_plugins.md §5, roadmap.md
//! Phase 17): separation + alignment + cohesion, the textbook boids rules (Reynolds 1987), as
//! distinct from Pearce's visual-projection occlusion model and Vicsek's alignment-only rule.
//! The third `FlockingMode` plugin, and the **first real occupant of `murmur_core::kernels`'s
//! `SeparationKernel`/`AlignmentKernel`/`CohesionKernel` toolkit** (design/02_plugins.md §3) —
//! those traits existed since Phase 1 with zero implementations; this plugin is nearly free
//! precisely because that infrastructure was already there to build against.
//!
//! Each kernel is a concrete, non-trait-object field here (not `Box<dyn ...>`): with exactly
//! one implementation of each and no per-plugin mechanism to select among sub-kernels via
//! `PluginParams`, boxing would add indirection without adding real swappability — the traits
//! themselves are still genuinely implemented and reusable by any *future* `FlockingMode` that
//! wants a different kernel combination, which is what the toolkit is for.
//!
//! **Combination convention.** Each channel is reduced to a **unit direction** before blending
//! (separation's per-neighbour inverse-distance-weighted repulsion summed then renormalised;
//! alignment's mean neighbour heading; cohesion's mean offset normalised toward the local
//! centre), then blended by `separation_weight`/`alignment_weight`/`cohesion_weight` and
//! renormalised to `cruise_speed` — the same "blend unit directions, not raw magnitudes"
//! approach `Vicsek` already uses for its own heading blend, so the three channels' weights are
//! comparable to each other regardless of how many neighbours are in range or how close they
//! are.

use murmur_core::{
    AlignmentKernel, BoidCtx, CohesionKernel, ConfigError, CoreParams, FlockingMode,
    OcclusionScratch, PluginParams, Registry, Rng, SeparationKernel, SteerIntent, Vec3, MIN_LEN,
    MIN_LEN2,
};
use serde::{Deserialize, Serialize};

/// Linear-falloff, inverse-distance repulsion: strongest as `d -> 0`, zero at/beyond `radius`.
/// `d` is clamped away from exactly `0` before dividing (a coincident boid has no well-defined
/// repulsion direction either — `Neighbor`'s own producers already drop those, but this stays
/// safe even if a future `NeighborSelection` doesn't).
pub struct LinearInverseSeparation {
    pub radius: f64,
}
impl SeparationKernel for LinearInverseSeparation {
    fn weight(&self, d: f64, _params: &CoreParams) -> f64 {
        if d < self.radius {
            (self.radius - d) / (self.radius * d.max(MIN_LEN))
        } else {
            0.0
        }
    }
}

/// Mean unit heading of neighbours (never bearing — `Neighbor::velocity`, matching the
/// codebase-wide bearing/heading distinction). Falls back to the observer's own heading if no
/// neighbour has a well-defined one (e.g. an empty neighbourhood, or every neighbour stalled).
pub struct MeanVelocityAlignment;
impl AlignmentKernel for MeanVelocityAlignment {
    fn heading(&self, ctx: BoidCtx<'_>) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for n in ctx.neighbors {
            if n.velocity.len_sq() > MIN_LEN2 {
                sum += n.velocity.normalized();
            }
        }
        if sum.len_sq() > MIN_LEN2 {
            sum.normalized()
        } else {
            ctx.vel.normalized()
        }
    }
}

/// Mean offset toward neighbours' centre (an **offset**, per the kernel toolkit's own
/// convention — `kernels.rs`'s doc comment — not an absolute position).
pub struct MeanOffsetCohesion;
impl CohesionKernel for MeanOffsetCohesion {
    fn target(&self, ctx: BoidCtx<'_>) -> Vec3 {
        if ctx.neighbors.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for n in ctx.neighbors {
            sum += n.direction * n.distance; // bearing * distance = offset to the neighbour
        }
        sum * (1.0 / ctx.neighbors.len() as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialParams {
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
    /// Beyond this distance, `LinearInverseSeparation`'s repulsion is exactly zero.
    pub separation_radius: f64,
}

impl SpatialParams {
    pub fn builder() -> SpatialParamsBuilder {
        SpatialParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialParamsBuilder {
    separation_weight: f64,
    alignment_weight: f64,
    cohesion_weight: f64,
    separation_radius: f64,
}

impl Default for SpatialParamsBuilder {
    fn default() -> Self {
        SpatialParamsBuilder {
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            separation_radius: 2.0,
        }
    }
}

impl SpatialParamsBuilder {
    pub fn separation_weight(mut self, v: f64) -> Self {
        self.separation_weight = v;
        self
    }
    pub fn alignment_weight(mut self, v: f64) -> Self {
        self.alignment_weight = v;
        self
    }
    pub fn cohesion_weight(mut self, v: f64) -> Self {
        self.cohesion_weight = v;
        self
    }
    pub fn separation_radius(mut self, v: f64) -> Self {
        self.separation_radius = v;
        self
    }

    pub fn build(self) -> Result<SpatialParams, ConfigError> {
        for (field, v) in [
            ("separation_weight", self.separation_weight),
            ("alignment_weight", self.alignment_weight),
            ("cohesion_weight", self.cohesion_weight),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite and >= 0".into(),
                });
            }
        }
        if !(self.separation_radius.is_finite() && self.separation_radius > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "separation_radius",
                reason: "must be finite and > 0".into(),
            });
        }
        Ok(SpatialParams {
            separation_weight: self.separation_weight,
            alignment_weight: self.alignment_weight,
            cohesion_weight: self.cohesion_weight,
            separation_radius: self.separation_radius,
        })
    }
}

pub struct Spatial {
    pub params: SpatialParams,
    separation_kernel: LinearInverseSeparation,
    alignment_kernel: MeanVelocityAlignment,
    cohesion_kernel: MeanOffsetCohesion,
}

impl Spatial {
    pub fn new(params: SpatialParams) -> Self {
        Spatial {
            separation_kernel: LinearInverseSeparation {
                radius: params.separation_radius,
            },
            alignment_kernel: MeanVelocityAlignment,
            cohesion_kernel: MeanOffsetCohesion,
            params,
        }
    }
}

impl FlockingMode for Spatial {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        _rng: &mut Rng,
    ) -> SteerIntent {
        let mut separation = Vec3::ZERO;
        for n in ctx.neighbors {
            let w = self.separation_kernel.weight(n.distance, ctx.core_params);
            if w > 0.0 {
                separation += -n.direction * w; // away from the neighbour
            }
        }
        let separation_dir = separation.normalized();

        let alignment_dir = self.alignment_kernel.heading(ctx);

        let cohesion_offset = self.cohesion_kernel.target(ctx);
        let cohesion_dir = cohesion_offset.normalized();

        let combined = separation_dir * self.params.separation_weight
            + alignment_dir * self.params.alignment_weight
            + cohesion_dir * self.params.cohesion_weight;

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
        "spatial"
    }
}

/// Registers `Spatial` under the name `"spatial"`, reading `separation_weight`/
/// `alignment_weight`/`cohesion_weight`/`separation_radius` from `PluginParams` (defaulting to
/// a separation-dominant classic-boids blend). The factory type can't return `Result`
/// (design/02_plugins.md §1), so a malformed override falls back to the default rather than
/// panicking — same pattern as `murmur_pearce`/`murmur_vicsek`.
pub fn register(r: &mut Registry) {
    r.register_mode("spatial", |p: &PluginParams| {
        let params = SpatialParams::builder()
            .separation_weight(p.get_or("separation_weight", 1.5))
            .alignment_weight(p.get_or("alignment_weight", 1.0))
            .cohesion_weight(p.get_or("cohesion_weight", 1.0))
            .separation_radius(p.get_or("separation_radius", 2.0))
            .build()
            .unwrap_or_else(|_| {
                SpatialParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(Spatial::new(params)) as Box<dyn FlockingMode>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{Domain, Neighbor, Species};

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

    #[test]
    fn conforms_to_flocking_mode_contract() {
        murmur_conformance::flocking_mode(&Spatial::new(SpatialParams::builder().build().unwrap()));
    }

    #[test]
    fn a_close_neighbour_produces_repulsion_away_from_it() {
        let params = core_params();
        let domain = StubDomain;
        // Neighbour due +x, well inside separation_radius=2.0, no alignment/cohesion pull to
        // confound the result (isolate separation by zeroing the other two weights).
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 0.5, Vec3::ZERO);
        let neighbors = [n];
        let mode = Spatial::new(
            SpatialParams::builder()
                .separation_weight(1.0)
                .alignment_weight(0.0)
                .cohesion_weight(0.0)
                .separation_radius(2.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(
            intent.desired_v.x < 0.0,
            "must steer away (-x) from a neighbour at +x, got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn a_neighbour_beyond_separation_radius_produces_no_separation_contribution() {
        let params = core_params();
        let domain = StubDomain;
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 5.0, Vec3::ZERO); // beyond radius=2.0
        let neighbors = [n];
        let mode = Spatial::new(
            SpatialParams::builder()
                .separation_weight(1.0)
                .alignment_weight(0.0)
                .cohesion_weight(1.0) // cohesion still pulls toward the neighbour
                .separation_radius(2.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        // With separation silent, only cohesion acts — must move toward +x (the neighbour),
        // not away from it.
        assert!(intent.desired_v.x > 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn alignment_blends_toward_the_mean_neighbour_heading() {
        let params = core_params();
        let domain = StubDomain;
        // Neighbour far enough to have zero separation/cohesion pull along x (place it
        // exactly overhead in z instead) so only alignment's heading (+x) shows through.
        let n = neighbor(Vec3::new(0.0, 0.0, 1.0), 5.0, Vec3::new(1.0, 0.0, 0.0));
        let neighbors = [n];
        let mode = Spatial::new(
            SpatialParams::builder()
                .separation_weight(0.0)
                .alignment_weight(1.0)
                .cohesion_weight(0.0)
                .separation_radius(2.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(
            (intent.desired_v.normalized() - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9,
            "must align exactly to the sole neighbour's heading, got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn cohesion_pulls_toward_the_mean_neighbour_offset() {
        let params = core_params();
        let domain = StubDomain;
        // Two neighbours symmetric about +x: mean offset points along +x.
        let n1 = neighbor(Vec3::new(1.0, 1.0, 0.0), 5.0, Vec3::ZERO);
        let n2 = neighbor(Vec3::new(1.0, -1.0, 0.0), 5.0, Vec3::ZERO);
        let neighbors = [n1, n2];
        let mode = Spatial::new(
            SpatialParams::builder()
                .separation_weight(0.0)
                .alignment_weight(0.0)
                .cohesion_weight(1.0)
                .separation_radius(2.0)
                .build()
                .unwrap(),
        );
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x > 0.0, "got {:?}", intent.desired_v);
        assert!(
            intent.desired_v.y.abs() < 1e-9,
            "the y components should cancel, got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn no_neighbours_falls_back_to_current_velocity() {
        let params = core_params();
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Spatial::new(SpatialParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let current = Vec3::new(0.0, 2.0, 0.0);
        let intent = mode.desired(
            ctx(current, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert_eq!(intent.desired_v, current);
    }

    #[test]
    fn desired_v_is_always_finite() {
        let params = core_params();
        let domain = StubDomain;
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 0.0001, Vec3::ZERO); // nearly coincident
        let neighbors = [n];
        let mode = Spatial::new(SpatialParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.is_finite(), "got {:?}", intent.desired_v);
    }

    #[test]
    fn builder_rejects_a_negative_weight() {
        let err = SpatialParams::builder().separation_weight(-1.0).build();
        assert!(err.is_err());
    }

    #[test]
    fn builder_rejects_a_non_positive_separation_radius() {
        let err = SpatialParams::builder().separation_radius(0.0).build();
        assert!(err.is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("spatial", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "spatial");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("separation_radius", -5.0);
        let mode = reg.resolve_mode("spatial", &bad).unwrap();
        assert_eq!(mode.name(), "spatial"); // must not panic during resolution
    }

    /// Proves the `FlockingMode` seam now has ≥3 real occupants beyond Pearce/Vicsek.
    #[test]
    fn pearce_vicsek_and_spatial_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_vicsek::register(&mut reg);
        murmur_pearce::register(&mut reg);

        assert_eq!(
            reg.resolve_mode("spatial", &PluginParams::new())
                .unwrap()
                .name(),
            "spatial"
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
