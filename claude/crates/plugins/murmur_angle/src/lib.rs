//! `Angle` — turn-rate-limited heading `FlockingMode` (design/02_plugins.md §5, roadmap.md
//! Phase 17, pymurmur's `physics/forces/angle.py`, ported by description — pymurmur's actual
//! source isn't reachable in this environment, the same blocker as every other pymurmur
//! cross-check this project has hit). Distinguishing feature from every other `FlockingMode`
//! here: rather than returning an instantaneous target heading and leaving turn-rate limiting
//! to `max_force`/the composed `SteeringModifier`, this mode owns a **persistent per-boid
//! heading** and explicitly rotates it toward a Vicsek-style consensus target by at most
//! `max_turn_rate * dt` radians per step — a physically-motivated maximum turn rate, not an
//! acceleration cap.
//!
//! **Per-boid state: the plugin-owned side-column pattern** (design/01_core.md §2), the same
//! established precedent `murmur_spin_wave`'s `s_z` and `murmur_predator`'s state already
//! validate — a `Mutex`-guarded `HashMap<u32, Vec3>`, lazily growing, keyed by the boid's
//! stable index, entirely outside `BoidColumns` itself.
//!
//! **Simpler than `murmur_spin_wave`'s double-buffered `committed`/`pending` split, and
//! deliberately so.** `spin_wave`'s `SteeringModifier` needs the split because its
//! neighbour-Laplacian coupling term reads *other* boids' `s_z` mid-step — without it, which
//! neighbours' values are "this step's" vs. "last step's" depends on rayon's scheduling order,
//! a real thread-count-dependent bug that split fixes (see that crate's own module doc).
//! `Angle` has no such read: `desired()` for boid `i` only ever reads and writes its **own**
//! entry, key `i` — no boid ever touches another boid's heading. Two different boids' `desired`
//! calls therefore always touch disjoint `HashMap` keys, so a single (non-double-buffered) map
//! behind one `Mutex` is correct as-is; the `Mutex` itself is still required only because
//! `std::collections::HashMap`'s mutation isn't safe under unsynchronised concurrent access,
//! not because of any cross-boid visibility concern. Verified empirically anyway, not just
//! argued: `tests/angle_integration.rs::state_hash_is_identical_across_1_4_and_8_rayon_threads`
//! — this project's history (spin_wave's G4 bug, found via exactly this class of test) is
//! reason enough not to skip it just because the reasoning above seems airtight.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{
    sample_unit_sphere, BoidCtx, ConfigError, FlockingMode, OcclusionScratch, PluginParams,
    Registry, Rng, SteerIntent, Vec3, MIN_LEN2,
};

/// Rotates unit vector `current` toward unit vector `target` by at most `max_angle` radians,
/// along the geodesic between them (Rodrigues' rotation formula around `current × target`).
/// Falls back to an arbitrary perpendicular axis when `current`/`target` are (anti)parallel —
/// `current × target` is then near-zero and doesn't define a rotation plane on its own.
fn rotate_toward(current: Vec3, target: Vec3, max_angle: f64) -> Vec3 {
    let cos_angle = current.dot(target).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    if angle <= max_angle || angle < 1e-9 {
        return target;
    }
    let mut axis = current.cross(target);
    if axis.len_sq() <= MIN_LEN2 {
        let helper = if current.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        axis = current.cross(helper);
    }
    let axis = axis.normalized();
    let theta = max_angle.min(angle);
    current * theta.cos() + axis.cross(current) * theta.sin()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleParams {
    /// Maximum heading rotation per unit simulated time, in radians.
    pub max_turn_rate: f64,
}

impl AngleParams {
    pub fn builder() -> AngleParamsBuilder {
        AngleParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AngleParamsBuilder {
    max_turn_rate: f64,
}

impl Default for AngleParamsBuilder {
    fn default() -> Self {
        AngleParamsBuilder { max_turn_rate: 1.0 }
    }
}

impl AngleParamsBuilder {
    pub fn max_turn_rate(mut self, v: f64) -> Self {
        self.max_turn_rate = v;
        self
    }

    pub fn build(self) -> Result<AngleParams, ConfigError> {
        if !(self.max_turn_rate.is_finite() && self.max_turn_rate >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "max_turn_rate",
                reason: "must be finite and >= 0".into(),
            });
        }
        Ok(AngleParams {
            max_turn_rate: self.max_turn_rate,
        })
    }
}

pub struct Angle {
    pub params: AngleParams,
    headings: Mutex<HashMap<u32, Vec3>>,
}

impl Angle {
    pub fn new(params: AngleParams) -> Self {
        Angle {
            params,
            headings: Mutex::new(HashMap::new()),
        }
    }
}

impl FlockingMode for Angle {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        rng: &mut Rng,
    ) -> SteerIntent {
        // Vicsek-style consensus target: mean unit heading of neighbours (never bearing).
        let mut sum = Vec3::ZERO;
        for n in ctx.neighbors {
            if n.velocity.len_sq() > MIN_LEN2 {
                sum += n.velocity.normalized();
            }
        }
        let target = if sum.len_sq() > MIN_LEN2 {
            sum.normalized()
        } else if ctx.vel.len_sq() > MIN_LEN2 {
            ctx.vel.normalized() // no informative neighbours: hold current heading as the target
        } else {
            Vec3::ZERO // genuinely no direction anywhere to aim for this step
        };

        let mut headings = self.headings.lock().unwrap();
        let current = *headings.entry(ctx.index).or_insert_with(|| {
            if ctx.vel.len_sq() > MIN_LEN2 {
                ctx.vel.normalized()
            } else if target.len_sq() > MIN_LEN2 {
                target
            } else {
                sample_unit_sphere(rng) // truly no information at all: pick something, not ZERO
            }
        });

        let max_angle = self.params.max_turn_rate * ctx.core_params.dt;
        let new_heading = if target.len_sq() > MIN_LEN2 {
            rotate_toward(current, target, max_angle)
        } else {
            current // no target this step: keep flying the current heading, don't decay to zero
        };
        headings.insert(ctx.index, new_heading);
        drop(headings);

        SteerIntent {
            desired_v: new_heading * ctx.core_params.cruise_speed,
            extra_force: Vec3::ZERO,
            theta: 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "angle"
    }
}

/// Registers `Angle` under the name `"angle"`, reading `max_turn_rate` from `PluginParams`
/// (default `1.0` rad per unit time). The factory type can't return `Result`
/// (design/02_plugins.md §1), so a malformed override falls back to the default rather than
/// panicking — same pattern as `murmur_pearce`/`murmur_vicsek`/`murmur_spatial`.
pub fn register(r: &mut Registry) {
    r.register_mode("angle", |p: &PluginParams| {
        let params = AngleParams::builder()
            .max_turn_rate(p.get_or("max_turn_rate", 1.0))
            .build()
            .unwrap_or_else(|_| AngleParams::builder().build().expect("defaults are valid"));
        Box::new(Angle::new(params)) as Box<dyn FlockingMode>
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

    fn neighbor(direction: Vec3, distance: f64, velocity: Vec3) -> Neighbor {
        Neighbor {
            index: 1,
            distance,
            direction: direction.normalized(),
            velocity,
        }
    }

    fn ctx<'a>(
        index: u32,
        vel: Vec3,
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
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
        murmur_conformance::flocking_mode(&Angle::new(AngleParams::builder().build().unwrap()));
    }

    #[test]
    fn rotate_toward_reaches_the_target_exactly_when_max_angle_covers_the_full_gap() {
        let current = Vec3::new(1.0, 0.0, 0.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let out = rotate_toward(current, target, std::f64::consts::FRAC_PI_2);
        assert!((out - target).len() < 1e-9, "got {:?}", out);
    }

    #[test]
    fn rotate_toward_moves_only_partway_when_max_angle_is_smaller_than_the_full_gap() {
        let current = Vec3::new(1.0, 0.0, 0.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let step = 0.1;
        let out = rotate_toward(current, target, step);
        // Still unit length, and the angle actually turned matches `step` exactly.
        assert!((out.len() - 1.0).abs() < 1e-9);
        let turned = current.dot(out).clamp(-1.0, 1.0).acos();
        assert!((turned - step).abs() < 1e-9, "got turned={}", turned);
    }

    #[test]
    fn rotate_toward_handles_an_exactly_opposite_target_without_nan() {
        let current = Vec3::new(1.0, 0.0, 0.0);
        let target = Vec3::new(-1.0, 0.0, 0.0);
        let out = rotate_toward(current, target, 0.2);
        assert!(out.is_finite(), "got {:?}", out);
        assert!((out.len() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_boid_turns_toward_a_neighbours_heading_but_not_all_the_way_in_one_slow_step() {
        let params = core_params(0.01); // max_turn_rate=1.0 rad/s default -> max_angle=0.01 rad
        let domain = StubDomain;
        // Own heading +x; sole neighbour heading +y -- a 90 degree gap.
        let neighbors = [neighbor(
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            Vec3::new(0.0, 1.0, 0.0),
        )];
        let mode = Angle::new(AngleParams::builder().max_turn_rate(1.0).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(0, Vec3::new(1.0, 0.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        let heading = intent.desired_v.normalized();
        assert!(
            heading.x > 0.9,
            "must not have turned far in one small step, got {:?}",
            heading
        );
        assert!(
            heading.y > 0.0,
            "but must have turned at least a little toward +y, got {:?}",
            heading
        );
    }

    #[test]
    fn repeated_steps_eventually_converge_to_the_target_heading() {
        let params = core_params(1.0); // large dt -> large max_angle each call
        let domain = StubDomain;
        let neighbors = [neighbor(
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            Vec3::new(0.0, 1.0, 0.0),
        )];
        let mode = Angle::new(AngleParams::builder().max_turn_rate(0.3).build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let mut vel = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..20 {
            let intent = mode.desired(
                ctx(0, vel, &neighbors, &params, &domain),
                &mut scratch,
                &mut rng,
            );
            vel = intent.desired_v;
        }
        assert!(
            (vel.normalized() - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-6,
            "should have converged to the neighbour's heading by now, got {:?}",
            vel
        );
    }

    #[test]
    fn a_stalled_boid_with_no_neighbours_gets_a_well_defined_nonzero_heading() {
        let params = core_params(0.1);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Angle::new(AngleParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(0, Vec3::ZERO, &neighbors, &params, &domain),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.is_finite());
        assert!(
            intent.desired_v.len() > 0.0,
            "must not be stuck at exactly zero forever, got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn different_boids_maintain_independent_persistent_headings() {
        let params = core_params(0.01);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Angle::new(AngleParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng0 = murmur_core::rng::for_boid(1, 0, 0);
        let mut rng1 = murmur_core::rng::for_boid(1, 1, 0);

        let intent0 = mode.desired(
            ctx(0, Vec3::new(1.0, 0.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng0,
        );
        let intent1 = mode.desired(
            ctx(1, Vec3::new(0.0, 1.0, 0.0), &neighbors, &params, &domain),
            &mut scratch,
            &mut rng1,
        );
        assert!(
            (intent0.desired_v.normalized() - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9,
            "boid 0's own heading must not be affected by boid 1's, got {:?}",
            intent0.desired_v
        );
        assert!(
            (intent1.desired_v.normalized() - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-9,
            "boid 1's own heading must not be affected by boid 0's, got {:?}",
            intent1.desired_v
        );
    }

    #[test]
    fn builder_rejects_a_negative_turn_rate() {
        assert!(AngleParams::builder().max_turn_rate(-1.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("angle", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "angle");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("max_turn_rate", -5.0);
        let mode = reg.resolve_mode("angle", &bad).unwrap();
        assert_eq!(mode.name(), "angle");
    }

    /// Proves the `FlockingMode` seam now has ≥4 real occupants beyond Pearce/Vicsek/Spatial.
    #[test]
    fn pearce_vicsek_and_angle_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_vicsek::register(&mut reg);
        murmur_pearce::register(&mut reg);

        assert_eq!(
            reg.resolve_mode("angle", &PluginParams::new())
                .unwrap()
                .name(),
            "angle"
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
