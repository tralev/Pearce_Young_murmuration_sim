//! `pipeline.rs`'s own `#[cfg(test)]` module, split into a sibling file purely for navigability
//! (a whole-codebase audit flagged `pipeline.rs` as one of the project's largest files) — no
//! behaviour change, `Simulation`'s public API is untouched.

use super::*;
use crate::init::Initializer;
use crate::math::Vec3;
use crate::modes::{BoidCtx, SteerIntent};
use crate::occlusion::OcclusionScratch;
use crate::rng::Rng as CoreRng;

struct DummyDomain;
impl Domain for DummyDomain {
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }
    fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
    fn name(&self) -> &'static str {
        "dummy_domain"
    }
}

struct DummySpatialIndex;
impl SpatialIndex for DummySpatialIndex {
    fn rebuild(&mut self, _boids: &BoidColumns) {}
    fn candidates(&self, _p: Vec3, _r: f64, out: &mut Vec<u32>) {
        out.clear();
    }
    fn name(&self) -> &'static str {
        "dummy_spatial_index"
    }
}

struct DummyNeighborSelection;
impl NeighborSelection for DummyNeighborSelection {
    fn select(
        &self,
        _index: &dyn SpatialIndex,
        _i: u32,
        _boids: &BoidColumns,
        _params: &CoreParams,
    ) -> Vec<crate::neighbor::Neighbor> {
        Vec::new()
    }
    fn name(&self) -> &'static str {
        "dummy_neighbor_selection"
    }
}

struct DummyMode;
impl FlockingMode for DummyMode {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        rng: &mut CoreRng,
    ) -> SteerIntent {
        // A small rng-driven wobble on top of the current heading: enough that positions
        // actually change step over step (a meaningful thing for the determinism /
        // no-NaN pipeline tests to check), while staying a trivial stand-in mode — no
        // occlusion, no real flocking rule.
        let noise = crate::rng::sample_unit_sphere(rng) * 0.1;
        SteerIntent {
            desired_v: ctx.vel + noise,
            extra_force: Vec3::ZERO,
            theta: 0.0,
        }
    }
    fn name(&self) -> &'static str {
        "dummy_mode"
    }
}

struct DummyModifier;
impl SteeringModifier for DummyModifier {
    fn respond(&self, _ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3 {
        desired_v - current_vel
    }
    fn name(&self) -> &'static str {
        "dummy_modifier"
    }
}

struct DummySpeedModel;
impl SpeedModel for DummySpeedModel {
    fn enforce(
        &self,
        _vel: &mut Vec3,
        _species: Species,
        _params: &CoreParams,
        _cap_multiplier: f64,
        _rng: &mut CoreRng,
    ) {
    }
    fn name(&self) -> &'static str {
        "dummy_speed_model"
    }
}

struct DummyInit;
impl Initializer for DummyInit {
    fn place(&self, count: u32, params: &CoreParams, _rng: &mut CoreRng) -> (Vec<Vec3>, Vec<Vec3>) {
        let pos = (0..count).map(|i| Vec3::new(i as f64, 0.0, 0.0)).collect();
        let vel = (0..count)
            .map(|_| Vec3::new(params.cruise_speed, 0.0, 0.0))
            .collect();
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "dummy_init"
    }
}

/// A real (not empty-returning) brute-force `NeighborSelection`, self-contained here since
/// `murmur_core` can't dev-depend on any real plugin crate (they all depend on it — that
/// would be a cycle). Only used by the G1 regression test below.
struct BruteForceNeighbors;
impl NeighborSelection for BruteForceNeighbors {
    fn select(
        &self,
        _index: &dyn SpatialIndex,
        i: u32,
        boids: &BoidColumns,
        params: &CoreParams,
    ) -> Vec<crate::neighbor::Neighbor> {
        let pos_i = boids.pos[i as usize];
        boids
            .iter_active()
            .filter(|&j| j != i)
            .filter_map(|j| {
                let offset = boids.pos[j as usize] - pos_i;
                let d = offset.len();
                (d <= params.vision_radius && d > 1e-9).then(|| crate::neighbor::Neighbor {
                    index: j,
                    distance: d,
                    direction: offset / d,
                    velocity: boids.vel[j as usize],
                })
            })
            .collect()
    }
    fn name(&self) -> &'static str {
        "brute_force_neighbors"
    }
}

struct DummyNoise;
impl NoiseSource for DummyNoise {
    fn sample(&self, _rng: &mut CoreRng) -> Vec3 {
        Vec3::ZERO
    }
    fn name(&self) -> &'static str {
        "dummy_noise"
    }
}

fn register_all_dummies(reg: &mut Registry) {
    reg.register_mode("dummy_mode", |_| Box::new(DummyMode));
    reg.register_modifier("dummy_modifier", |_| Box::new(DummyModifier));
    reg.register_domain("dummy_domain", |_| Box::new(DummyDomain));
    reg.register_spatial_index("dummy_spatial_index", |_| Box::new(DummySpatialIndex));
    reg.register_neighbor_selection("dummy_neighbor_selection", |_| {
        Box::new(DummyNeighborSelection)
    });
    reg.register_speed_model("dummy_speed_model", |_| Box::new(DummySpeedModel));
    reg.register_init("dummy_init", |_| Box::new(DummyInit));
    reg.register_noise("dummy_noise", |_| Box::new(DummyNoise));
}

fn dummy_config() -> SimConfig {
    SimConfig {
        mode: "dummy_mode".to_string(),
        modifier: "dummy_modifier".to_string(),
        domain: "dummy_domain".to_string(),
        spatial_index: "dummy_spatial_index".to_string(),
        neighbor_selection: "dummy_neighbor_selection".to_string(),
        speed_model: "dummy_speed_model".to_string(),
        init: "dummy_init".to_string(),
        noise: "dummy_noise".to_string(),
        core_params: CoreParams::builder().boid_count(5).build().unwrap(),
        plugin_params: PluginParams::new(),
        init_seed: 1,
        step_hooks: Vec::new(),
        predator_count: 0,
        spawn_headroom: 0,
    }
}

#[test]
fn empty_registry_makes_new_fail_not_panic() {
    let registry = Registry::new();
    let result = Simulation::new(dummy_config(), &registry);
    assert!(result.is_err());
}

#[test]
fn fully_registered_dummy_composition_constructs_successfully() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    let (sim, _warnings) = Simulation::new(dummy_config(), &registry).unwrap();
    assert_eq!(sim.boid_count(), 5);
    assert_eq!(sim.step_count(), 0);
    let names = sim.plugin_names();
    assert!(names.contains(&("mode", "dummy_mode")));
    assert!(names.contains(&("noise", "dummy_noise")));
}

#[test]
fn partially_registered_composition_fails_with_unknown_plugin() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    // Leave "domain" unregistered by asking for a name that was never registered.
    let mut config = dummy_config();
    config.domain = "nonexistent_domain".to_string();
    match Simulation::new(config, &registry) {
        Err(e) => assert_eq!(
            e,
            ConfigError::UnknownPlugin {
                socket: "Domain",
                name: "nonexistent_domain".into()
            }
        ),
        Ok(_) => panic!("expected UnknownPlugin"),
    }
}

fn built_dummy_sim(n: u32) -> Simulation {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    let mut config = dummy_config();
    config.core_params = CoreParams::builder().boid_count(n).build().unwrap();
    Simulation::new(config, &registry).unwrap().0
}

#[test]
fn step_runs_without_nan_and_advances_step_count() {
    let mut sim = built_dummy_sim(20);
    for step in 0..50 {
        sim.step(1.0, 42);
        assert_eq!(sim.step_count(), step + 1);
        for i in sim.boids.iter_active() {
            assert!(
                sim.boids.pos[i as usize].is_finite(),
                "pos went non-finite at step {step}"
            );
            assert!(
                sim.boids.vel[i as usize].is_finite(),
                "vel went non-finite at step {step}"
            );
        }
    }
}

#[test]
fn run_batch_runs_the_requested_number_of_steps() {
    let mut sim = built_dummy_sim(10);
    sim.run_batch(15, 7);
    assert_eq!(sim.step_count(), 15);
}

#[test]
fn run_batch_with_budget_completes_within_a_generous_budget() {
    let mut sim = built_dummy_sim(10);
    let (completed, all_done) = sim.run_batch_with_budget(20, 7, 10_000.0);
    assert_eq!(completed, 20);
    assert!(all_done);
    assert_eq!(sim.step_count(), 20);
}

#[test]
fn run_batch_with_budget_stops_early_under_a_tiny_budget() {
    let mut sim = built_dummy_sim(500);
    let (completed, all_done) = sim.run_batch_with_budget(1_000_000, 7, 0.001);
    assert!(!all_done);
    assert!(completed < 1_000_000);
    assert_eq!(sim.step_count() as u32, completed);
}

#[test]
fn state_hash_is_identical_across_1_4_and_8_rayon_threads() {
    fn run_with_pool_size(threads: usize) -> u64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sim = built_dummy_sim(200);
            sim.run_batch(30, 12345);
            sim.state_hash()
        })
    }
    let h1 = run_with_pool_size(1);
    let h4 = run_with_pool_size(4);
    let h8 = run_with_pool_size(8);
    assert_eq!(h1, h4, "state_hash differs between 1 and 4 threads");
    assert_eq!(h1, h8, "state_hash differs between 1 and 8 threads");
}

#[test]
fn composition_reports_plugin_names_and_core_params() {
    let sim = built_dummy_sim(3);
    let comp = sim.composition();
    assert!(comp.plugin_names.contains(&("mode", "dummy_mode")));
    assert_eq!(comp.core_params.boid_count, 3);
}

/// StepHook ordering contract (roadmap.md Phase 8): with no declared `dependencies()`,
/// execution order follows registration order (design/02_plugins.md §5). Uses a `static`
/// log rather than a closure capturing shared state: `Registry`'s factories are plain `fn`
/// pointers (design/02_plugins.md §1), which can't close over anything.
static ORDER_LOG: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());

struct LoggingHook(&'static str);
impl StepHook for LoggingHook {
    fn post_steer(&self, _ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut crate::rng::Rng) {
        ORDER_LOG.lock().unwrap().push(self.0);
    }
    fn name(&self) -> &'static str {
        self.0
    }
}
fn make_hook_a(_p: &PluginParams) -> Box<dyn StepHook> {
    Box::new(LoggingHook("hook_a"))
}
fn make_hook_b(_p: &PluginParams) -> Box<dyn StepHook> {
    Box::new(LoggingHook("hook_b"))
}

#[test]
fn step_hooks_execute_in_registration_order() {
    ORDER_LOG.lock().unwrap().clear();

    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    registry.register_step_hook("hook_a", make_hook_a);
    registry.register_step_hook("hook_b", make_hook_b);

    let mut config = dummy_config();
    config.core_params = CoreParams::builder().boid_count(1).build().unwrap();
    config.step_hooks = vec!["hook_a".to_string(), "hook_b".to_string()];
    let (mut sim, _warnings) = Simulation::new(config, &registry).unwrap();

    sim.step(1.0, 1);

    assert_eq!(*ORDER_LOG.lock().unwrap(), vec!["hook_a", "hook_b"]);
}

struct CapHook(f64);
impl StepHook for CapHook {
    fn speed_cap_multiplier(&self, _index: u32) -> Option<f64> {
        Some(self.0)
    }
    fn name(&self) -> &'static str {
        "cap_hook"
    }
}

/// G3 (roadmap.md §12): the whole point of the channel — a `StepHook`'s
/// `speed_cap_multiplier` must actually reach `SpeedModel::enforce` and be enforced, not
/// just compile. Uses the real `band` `SpeedModel` (not a dummy), so this is a genuine
/// end-to-end proof of the plumbing, not just that the new trait method exists.
#[test]
fn a_step_hooks_speed_cap_multiplier_is_actually_enforced_by_the_speed_model() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    crate::speed_model::register(&mut registry);
    registry.register_step_hook("cap_hook", |_p: &PluginParams| {
        Box::new(CapHook(0.1)) as Box<dyn StepHook>
    });

    let mut config = dummy_config();
    config.speed_model = "band".to_string();
    config.core_params = CoreParams::builder()
        .boid_count(1)
        .cruise_speed(10.0)
        .build()
        .unwrap();
    config.step_hooks = vec!["cap_hook".to_string()];
    let (mut sim, _warnings) = Simulation::new(config, &registry).unwrap();

    sim.step(1.0, 1); // DummyInit starts every boid at speed = cruise_speed = 10.0

    let capped_speed = sim.boids.vel[0].len();
    assert!(
        (capped_speed - 1.0).abs() < 1e-6,
        "expected speed capped at cruise_speed(10.0) * cap_multiplier(0.1) = 1.0, got {}",
        capped_speed
    );
}

/// The combine rule is `min` across hooks — the *most* restrictive one wins, not the last
/// registered or an average.
#[test]
fn multiple_hooks_speed_cap_multipliers_combine_via_min() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    crate::speed_model::register(&mut registry);
    registry.register_step_hook("loose_cap", |_p: &PluginParams| {
        Box::new(CapHook(0.8)) as Box<dyn StepHook>
    });
    registry.register_step_hook("tight_cap", |_p: &PluginParams| {
        Box::new(CapHook(0.2)) as Box<dyn StepHook>
    });

    let mut config = dummy_config();
    config.speed_model = "band".to_string();
    config.core_params = CoreParams::builder()
        .boid_count(1)
        .cruise_speed(10.0)
        .build()
        .unwrap();
    config.step_hooks = vec!["loose_cap".to_string(), "tight_cap".to_string()];
    let (mut sim, _warnings) = Simulation::new(config, &registry).unwrap();

    sim.step(1.0, 1);

    let capped_speed = sim.boids.vel[0].len();
    assert!(
        (capped_speed - 2.0).abs() < 1e-6,
        "expected the tighter 0.2 cap to win (10.0 * 0.2 = 2.0), got {}",
        capped_speed
    );
}

static NEIGHBOR_COUNT_LOG: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

struct NeighborCountingHook;
impl StepHook for NeighborCountingHook {
    fn post_steer(&self, ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut crate::rng::Rng) {
        if ctx.index == 1 {
            NEIGHBOR_COUNT_LOG.lock().unwrap().push(ctx.neighbors.len());
        }
    }
    fn name(&self) -> &'static str {
        "neighbor_counting_hook"
    }
}

/// G1 (roadmap.md §12): `post_steer`'s `ctx.neighbors` was a hardcoded `&[]` — no `StepHook`
/// could see real neighbour data at all. This is the actual end-to-end proof, not just that
/// the code compiles: 3 boids at x=0,1,2 (`DummyInit`'s own spacing), `vision_radius` wide
/// enough to see both others — the middle boid (index 1) must see exactly 2 real neighbours
/// during `post_steer`, not 0.
#[test]
fn post_steers_ctx_neighbors_is_real_not_a_hardcoded_empty_placeholder() {
    NEIGHBOR_COUNT_LOG.lock().unwrap().clear();

    let mut registry = Registry::new();
    registry.register_mode("dummy_mode", |_| Box::new(DummyMode));
    registry.register_modifier("dummy_modifier", |_| Box::new(DummyModifier));
    registry.register_domain("dummy_domain", |_| Box::new(DummyDomain));
    registry.register_spatial_index("dummy_spatial_index", |_| Box::new(DummySpatialIndex));
    registry
        .register_neighbor_selection("brute_force_neighbors", |_| Box::new(BruteForceNeighbors));
    registry.register_speed_model("dummy_speed_model", |_| Box::new(DummySpeedModel));
    registry.register_init("dummy_init", |_| Box::new(DummyInit));
    registry.register_noise("dummy_noise", |_| Box::new(DummyNoise));
    registry.register_step_hook("neighbor_counting_hook", |_p: &PluginParams| {
        Box::new(NeighborCountingHook) as Box<dyn StepHook>
    });

    let mut config = dummy_config();
    config.neighbor_selection = "brute_force_neighbors".to_string();
    config.core_params = CoreParams::builder()
        .boid_count(3)
        .vision_radius(5.0) // wide enough to see both other boids at spacing 1
        .build()
        .unwrap();
    config.step_hooks = vec!["neighbor_counting_hook".to_string()];
    let (mut sim, _warnings) = Simulation::new(config, &registry).unwrap();

    sim.step(1.0, 1);

    assert_eq!(
        *NEIGHBOR_COUNT_LOG.lock().unwrap(),
        vec![2],
        "the middle boid must see both other real boids as neighbours, not an empty placeholder"
    );
}

static RNG_DRAW_LOG: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

struct RngDrawingHook;
impl StepHook for RngDrawingHook {
    fn post_steer(&self, _ctx: BoidCtx<'_>, _acc: &mut Vec3, rng: &mut crate::rng::Rng) {
        use rand_core::RngCore;
        RNG_DRAW_LOG.lock().unwrap().push(rng.next_u64());
    }
    fn name(&self) -> &'static str {
        "rng_drawing_hook"
    }
}

fn rng_drawing_config() -> SimConfig {
    let mut config = dummy_config();
    config.core_params = CoreParams::builder().boid_count(2).build().unwrap();
    config.step_hooks = vec!["rng_drawing_hook".to_string()];
    config
}

/// G8 (roadmap.md §12): before this fix, `StepHook::post_steer` had no path to genuine,
/// `base_seed`-tied randomness at all -- `SpeedModel::enforce` was the only per-boid caller
/// with an `Rng` in hand. Proves the fix two ways: the same `base_seed` gives identical
/// draws across independent runs (determinism), and a different `base_seed` gives different
/// draws (the values are genuinely tied to it, not some fixed internal source).
#[test]
fn post_steers_rng_argument_is_real_and_tied_to_base_seed() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    registry.register_step_hook("rng_drawing_hook", |_p: &PluginParams| {
        Box::new(RngDrawingHook) as Box<dyn StepHook>
    });

    RNG_DRAW_LOG.lock().unwrap().clear();
    let (mut sim_a, _warnings) = Simulation::new(rng_drawing_config(), &registry).unwrap();
    sim_a.step(1.0, 42);
    let draws_a = RNG_DRAW_LOG.lock().unwrap().clone();

    RNG_DRAW_LOG.lock().unwrap().clear();
    let (mut sim_a_repeat, _warnings) = Simulation::new(rng_drawing_config(), &registry).unwrap();
    sim_a_repeat.step(1.0, 42);
    let draws_a_repeat = RNG_DRAW_LOG.lock().unwrap().clone();
    assert_eq!(
        draws_a, draws_a_repeat,
        "the same base_seed must give identical post_steer RNG draws"
    );

    RNG_DRAW_LOG.lock().unwrap().clear();
    let (mut sim_b, _warnings) = Simulation::new(rng_drawing_config(), &registry).unwrap();
    sim_b.step(1.0, 99);
    let draws_b = RNG_DRAW_LOG.lock().unwrap().clone();
    assert_ne!(
        draws_a, draws_b,
        "a different base_seed must give different post_steer RNG draws"
    );
}
