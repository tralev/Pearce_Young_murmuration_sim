//! Generic per-trait plugin-conformance harness (design/00_overview.md decision D15).
//!
//! Each function asserts the invariants every implementation of that trait must satisfy —
//! finite output, respects `[0,1]`/clamp bounds, deterministic given the same seed — so a new
//! plugin author gets a ready-made correctness checklist instead of reinventing basic sanity
//! tests per plugin. Call these from the plugin crate's own `#[cfg(test)]` module against a
//! real instance. A dev-dependency every plugin crate uses; `murmur_core` itself does not
//! depend on this at runtime (design/00 D15, roadmap.md workspace layout).

use murmur_core::{
    rng, BoidColumns, BoidCtx, CoreParams, Domain, FlockingMode, Initializer, Neighbor,
    NeighborSelection, NoiseSource, OcclusionScratch, Rng, SimView, SpatialIndex, Species,
    SpeedModel, SteeringModifier, StepHook, Vec3,
};

/// A minimal, always-correct `Domain` used only to build a `BoidCtx` fixture for conformance
/// checks on *other* traits (e.g. `flocking_mode`) — never itself the thing under test.
pub struct FixtureDomain;
impl Domain for FixtureDomain {
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }
    fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
    fn name(&self) -> &'static str {
        "fixture_domain"
    }
}

pub fn fixture_core_params() -> CoreParams {
    CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(8)
        .vision_radius(5.0)
        .build()
        .expect("fixture params are valid by construction")
}

pub fn fixture_neighbors() -> Vec<Neighbor> {
    vec![
        Neighbor {
            index: 1,
            distance: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
            velocity: Vec3::new(0.0, 1.0, 0.0),
        },
        Neighbor {
            index: 2,
            distance: 2.0,
            direction: Vec3::new(0.0, 1.0, 0.0),
            velocity: Vec3::new(1.0, 0.0, 0.0),
        },
    ]
}

pub fn fixture_rng(seed: u64) -> Rng {
    rng::for_boid(1, seed, 0)
}

fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
    (a - b).len() < 1e-9
}

pub fn domain(d: &dyn Domain) {
    assert!(!d.name().is_empty(), "Domain::name() must be non-empty");

    let a = Vec3::new(1.0, 2.0, 3.0);
    let delta_self = d.delta(a, a);
    assert!(delta_self.is_finite(), "Domain::delta must be finite");
    assert_eq!(
        delta_self,
        Vec3::ZERO,
        "Domain::delta(a, a) must be zero — shortest displacement from a point to itself"
    );

    let b = Vec3::new(4.0, -1.0, 2.5);
    assert!(
        d.delta(a, b).is_finite(),
        "Domain::delta must be finite for distinct points"
    );

    let mut pos = Vec3::new(10.0, 10.0, 10.0);
    let mut vel = Vec3::new(1.0, 0.0, 0.0);
    d.apply(&mut pos, &mut vel, 1.0);
    assert!(
        pos.is_finite() && vel.is_finite(),
        "Domain::apply must leave pos/vel finite"
    );
}

pub fn spatial_index(index: &mut dyn SpatialIndex, boids: &BoidColumns) {
    assert!(
        !index.name().is_empty(),
        "SpatialIndex::name() must be non-empty"
    );
    index.rebuild(boids);

    let mut out = Vec::new();
    index.candidates(Vec3::ZERO, 10.0, &mut out); // must not panic

    let mut knn_out = Vec::new();
    index.candidates_knn(Vec3::ZERO, 2, &mut knn_out); // must terminate (reaching here proves it did)
    let _ = knn_out.len();
}

pub fn neighbor_selection(
    sel: &dyn NeighborSelection,
    index: &dyn SpatialIndex,
    boids: &BoidColumns,
    params: &CoreParams,
) {
    assert!(
        !sel.name().is_empty(),
        "NeighborSelection::name() must be non-empty"
    );
    for i in boids.iter_active() {
        let neighbors = sel.select(index, i, boids, params);
        for n in &neighbors {
            assert!(
                n.direction.is_finite(),
                "Neighbor::direction must be finite"
            );
            assert!(n.velocity.is_finite(), "Neighbor::velocity must be finite");
            assert!(
                n.distance.is_finite() && n.distance >= 0.0,
                "Neighbor::distance must be finite and non-negative"
            );
            if n.direction.len_sq() > 0.0 {
                let l = n.direction.len();
                assert!(
                    (l - 1.0).abs() < 1e-6,
                    "Neighbor::direction must be a unit bearing, got len={l}"
                );
            }
        }
    }
}

pub fn flocking_mode(mode: &dyn FlockingMode) {
    assert!(
        !mode.name().is_empty(),
        "FlockingMode::name() must be non-empty"
    );
    let domain = FixtureDomain;
    let params = fixture_core_params();
    let neighbors = fixture_neighbors();
    let ctx = BoidCtx {
        index: 0,
        pos: Vec3::ZERO,
        vel: Vec3::new(1.0, 0.0, 0.0),
        species: Species::Prey,
        neighbors: &neighbors,
        core_params: &params,
        domain: &domain,
    };
    let mut scratch = OcclusionScratch::default();

    let mut rng1 = fixture_rng(7);
    let intent1 = mode.desired(ctx, &mut scratch, &mut rng1);
    assert!(
        intent1.desired_v.is_finite(),
        "FlockingMode::desired must return a finite desired_v"
    );
    assert!(
        intent1.extra_force.is_finite(),
        "FlockingMode::desired must return a finite extra_force"
    );
    assert!(
        intent1.theta.is_finite() && (0.0..=1.0).contains(&intent1.theta),
        "FlockingMode::desired's theta must be finite and in [0, 1], got {}",
        intent1.theta
    );

    let mut rng2 = fixture_rng(7);
    let intent2 = mode.desired(ctx, &mut scratch, &mut rng2);
    assert!(
        vec3_approx_eq(intent1.desired_v, intent2.desired_v),
        "FlockingMode::desired must be deterministic for the same rng seed"
    );
}

pub fn steering_modifier(modifier: &dyn SteeringModifier) {
    assert!(
        !modifier.name().is_empty(),
        "SteeringModifier::name() must be non-empty"
    );
    let domain = FixtureDomain;
    let params = fixture_core_params();
    let neighbors = fixture_neighbors();
    let ctx = BoidCtx {
        index: 0,
        pos: Vec3::ZERO,
        vel: Vec3::new(1.0, 0.0, 0.0),
        species: Species::Prey,
        neighbors: &neighbors,
        core_params: &params,
        domain: &domain,
    };
    let acc = modifier.respond(ctx, Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    assert!(
        acc.is_finite(),
        "SteeringModifier::respond must return a finite acceleration"
    );
}

pub fn speed_model(model: &dyn SpeedModel) {
    assert!(
        !model.name().is_empty(),
        "SpeedModel::name() must be non-empty"
    );
    let params = fixture_core_params();
    let mut rng = fixture_rng(3);

    for species in [Species::Prey, Species::Predator, Species::Custom(7)] {
        let mut stalled = Vec3::ZERO;
        model.enforce(&mut stalled, species, &params, &mut rng);
        assert!(
            stalled.is_finite(),
            "SpeedModel::enforce must leave velocity finite even from a stall"
        );

        let mut overspeed = Vec3::new(1000.0, 0.0, 0.0);
        model.enforce(&mut overspeed, species, &params, &mut rng);
        assert!(
            overspeed.is_finite(),
            "SpeedModel::enforce must leave velocity finite for an overspeed boid"
        );
    }
}

pub fn initializer(init: &dyn Initializer) {
    assert!(
        !init.name().is_empty(),
        "Initializer::name() must be non-empty"
    );
    let params = fixture_core_params();
    let mut rng = fixture_rng(5);
    let (pos, vel) = init.place(params.boid_count, &params, &mut rng);
    assert_eq!(
        pos.len(),
        params.boid_count as usize,
        "Initializer::place must return exactly `count` positions"
    );
    assert_eq!(
        vel.len(),
        params.boid_count as usize,
        "Initializer::place must return exactly `count` velocities"
    );
    assert!(
        pos.iter().all(|p| p.is_finite()),
        "Initializer::place positions must be finite"
    );
    assert!(
        vel.iter().all(|v| v.is_finite()),
        "Initializer::place velocities must be finite"
    );
}

pub fn noise_source(noise: &dyn NoiseSource) {
    assert!(
        !noise.name().is_empty(),
        "NoiseSource::name() must be non-empty"
    );
    let mut rng = fixture_rng(9);
    for _ in 0..16 {
        assert!(
            noise.sample(&mut rng).is_finite(),
            "NoiseSource::sample must be finite"
        );
    }
}

pub fn step_hook(hook: &mut dyn StepHook) {
    assert!(
        !hook.name().is_empty(),
        "StepHook::name() must be non-empty"
    );
    assert!(
        !hook.dependencies().contains(&hook.name()),
        "StepHook must not declare itself as a dependency"
    );

    let mut params = fixture_core_params();
    let mut boids = BoidColumns::with_capacity(2);
    boids.add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0);
    boids.add(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Species::Predator,
        1,
    );
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);

    let domain = FixtureDomain;
    let ctx_params = fixture_core_params();
    let neighbors = fixture_neighbors();
    let ctx = BoidCtx {
        index: 0,
        pos: Vec3::ZERO,
        vel: Vec3::new(1.0, 0.0, 0.0),
        species: Species::Prey,
        neighbors: &neighbors,
        core_params: &ctx_params,
        domain: &domain,
    };
    let mut acc = Vec3::ZERO;
    hook.post_steer(ctx, &mut acc);
    assert!(
        acc.is_finite(),
        "StepHook::post_steer must leave acc finite"
    );
}

#[cfg(test)]
mod self_test {
    //! Proves the harness itself has teeth: a conforming dummy passes every check, and a
    //! deliberately broken one is caught (not just trivially green).
    use super::*;
    use murmur_core::SteerIntent;

    struct GoodMode;
    impl FlockingMode for GoodMode {
        fn desired(
            &self,
            ctx: BoidCtx<'_>,
            _s: &mut OcclusionScratch,
            _r: &mut Rng,
        ) -> SteerIntent {
            SteerIntent {
                desired_v: ctx.vel,
                extra_force: Vec3::ZERO,
                theta: 0.25,
            }
        }
        fn name(&self) -> &'static str {
            "good_mode"
        }
    }

    struct NanMode;
    impl FlockingMode for NanMode {
        fn desired(
            &self,
            _ctx: BoidCtx<'_>,
            _s: &mut OcclusionScratch,
            _r: &mut Rng,
        ) -> SteerIntent {
            SteerIntent {
                desired_v: Vec3::new(f64::NAN, 0.0, 0.0),
                extra_force: Vec3::ZERO,
                theta: 0.0,
            }
        }
        fn name(&self) -> &'static str {
            "nan_mode"
        }
    }

    #[test]
    fn good_flocking_mode_passes() {
        flocking_mode(&GoodMode);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn nan_flocking_mode_is_caught() {
        flocking_mode(&NanMode);
    }

    struct GoodDomain;
    impl Domain for GoodDomain {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "good_domain"
        }
    }

    struct BrokenDomain;
    impl Domain for BrokenDomain {
        fn delta(&self, _a: Vec3, _b: Vec3) -> Vec3 {
            Vec3::new(1.0, 0.0, 0.0) // wrong: delta(a, a) should be ZERO, this ignores `a`/`b`
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "broken_domain"
        }
    }

    #[test]
    fn good_domain_passes() {
        domain(&GoodDomain);
    }

    #[test]
    #[should_panic(expected = "delta(a, a)")]
    fn broken_domain_is_caught() {
        domain(&BrokenDomain);
    }

    struct GoodInit;
    impl Initializer for GoodInit {
        fn place(&self, count: u32, params: &CoreParams, _rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
            let pos = (0..count).map(|i| Vec3::new(i as f64, 0.0, 0.0)).collect();
            let vel = (0..count)
                .map(|_| Vec3::new(params.cruise_speed, 0.0, 0.0))
                .collect();
            (pos, vel)
        }
        fn name(&self) -> &'static str {
            "good_init"
        }
    }

    struct WrongCountInit;
    impl Initializer for WrongCountInit {
        fn place(
            &self,
            count: u32,
            _params: &CoreParams,
            _rng: &mut Rng,
        ) -> (Vec<Vec3>, Vec<Vec3>) {
            let short = count.saturating_sub(1);
            (
                vec![Vec3::ZERO; short as usize],
                vec![Vec3::ZERO; short as usize],
            )
        }
        fn name(&self) -> &'static str {
            "wrong_count_init"
        }
    }

    #[test]
    fn good_initializer_passes() {
        initializer(&GoodInit);
    }

    #[test]
    #[should_panic(expected = "exactly `count`")]
    fn wrong_count_initializer_is_caught() {
        initializer(&WrongCountInit);
    }

    struct GoodNoise;
    impl NoiseSource for GoodNoise {
        fn sample(&self, _rng: &mut Rng) -> Vec3 {
            Vec3::new(0.1, 0.2, 0.3)
        }
        fn name(&self) -> &'static str {
            "good_noise"
        }
    }

    #[test]
    fn good_noise_source_passes() {
        noise_source(&GoodNoise);
    }

    struct GoodSpeedModel;
    impl SpeedModel for GoodSpeedModel {
        fn enforce(&self, vel: &mut Vec3, _species: Species, params: &CoreParams, rng: &mut Rng) {
            let s = vel.len();
            let (vmax, vmin) = (
                params.cruise_speed,
                params.cruise_speed * params.speed_min_factor,
            );
            *vel = if s > vmax {
                *vel * (vmax / s)
            } else if s < vmin {
                if s > murmur_core::MIN_LEN {
                    *vel * (vmin / s)
                } else {
                    sample_unit(rng) * vmin
                }
            } else {
                *vel
            };
        }
        fn name(&self) -> &'static str {
            "good_speed_model"
        }
    }
    fn sample_unit(rng: &mut Rng) -> Vec3 {
        use rand_core::RngCore;
        let bits = rng.next_u64();
        let x = ((bits & 0xFFFF) as f64 / 65535.0) * 2.0 - 1.0;
        Vec3::new(x, 0.0, (1.0 - x * x).max(0.0).sqrt())
    }

    #[test]
    fn good_speed_model_passes() {
        speed_model(&GoodSpeedModel);
    }

    struct GoodModifier;
    impl SteeringModifier for GoodModifier {
        fn respond(&self, _ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3 {
            desired_v - current_vel
        }
        fn name(&self) -> &'static str {
            "good_modifier"
        }
    }

    #[test]
    fn good_steering_modifier_passes() {
        steering_modifier(&GoodModifier);
    }

    struct GoodSpatialIndex {
        seen: Vec<Vec3>,
    }
    impl SpatialIndex for GoodSpatialIndex {
        fn rebuild(&mut self, boids: &BoidColumns) {
            self.seen = boids.iter_active().map(|i| boids.pos[i as usize]).collect();
        }
        fn candidates(&self, _p: Vec3, _r: f64, out: &mut Vec<u32>) {
            out.clear();
            out.extend(0..self.seen.len() as u32);
        }
        fn name(&self) -> &'static str {
            "good_spatial_index"
        }
    }

    #[test]
    fn good_spatial_index_passes() {
        let boids = BoidColumns::with_capacity(4);
        let mut idx = GoodSpatialIndex { seen: Vec::new() };
        spatial_index(&mut idx, &boids);
    }

    struct GoodNeighborSelection;
    impl NeighborSelection for GoodNeighborSelection {
        fn select(
            &self,
            _index: &dyn SpatialIndex,
            _i: u32,
            _boids: &BoidColumns,
            _params: &CoreParams,
        ) -> Vec<Neighbor> {
            fixture_neighbors()
        }
        fn name(&self) -> &'static str {
            "good_neighbor_selection"
        }
    }

    #[test]
    fn good_neighbor_selection_passes() {
        let mut boids = BoidColumns::with_capacity(2);
        boids.add(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Species::Prey, 0);
        let idx = GoodSpatialIndex { seen: Vec::new() };
        let params = fixture_core_params();
        neighbor_selection(&GoodNeighborSelection, &idx, &boids, &params);
    }

    struct GoodStepHook;
    impl StepHook for GoodStepHook {
        fn name(&self) -> &'static str {
            "good_step_hook"
        }
    }

    #[test]
    fn good_step_hook_passes() {
        step_hook(&mut GoodStepHook);
    }
}
