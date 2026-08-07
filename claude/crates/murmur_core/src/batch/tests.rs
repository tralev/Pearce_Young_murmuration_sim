//! `batch.rs`'s own `#[cfg(test)]` module, split into a sibling file purely for navigability
//! (a whole-codebase audit flagged `batch.rs` as one of the project's largest files) — no
//! behaviour change, `Simulation`'s public API is untouched.

use super::*;
use crate::domain::Domain;
use crate::init::{Initializer, NoiseSource};
use crate::modes::{BoidCtx, FlockingMode, SteerIntent, SteeringModifier};
use crate::neighbor::NeighborSelection;
use crate::occlusion::OcclusionScratch;
use crate::pipeline::SimConfig;
use crate::registry::{PluginParams, Registry};
use crate::rng::Rng as CoreRng;
use crate::spatial_index::SpatialIndex;
use crate::speed_model::SpeedModel;
use crate::step_hook::StepHook;
use crate::BoidColumns;

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

struct DummyNoise;
impl NoiseSource for DummyNoise {
    fn sample(&self, _rng: &mut CoreRng) -> Vec3 {
        Vec3::ZERO
    }
    fn name(&self) -> &'static str {
        "dummy_noise"
    }
}

struct DummyPredatorHook;
impl StepHook for DummyPredatorHook {
    fn name(&self) -> &'static str {
        "predator"
    }
}

/// Publishes a fixed `state`/`speed_mult` for every boid and a fixed `dynamic_vision_range`
/// scene-level field — proves `capture_checkpoint` actually collects `StepHook`'s own
/// checkpoint fields generically, without `murmur_core` referencing this (or any) plugin's
/// name to do it.
struct DummyCheckpointHook;
impl StepHook for DummyCheckpointHook {
    fn checkpoint_boid_fields(&self, _index: u32) -> BoidCheckpointFields {
        BoidCheckpointFields {
            state: Some(2),
            speed_mult: Some(0.5),
            ..Default::default()
        }
    }
    fn checkpoint_scene_fields(&self) -> SceneCheckpointFields {
        SceneCheckpointFields {
            dynamic_vision_range: Some(1.25),
            ..Default::default()
        }
    }
    fn name(&self) -> &'static str {
        "dummy_checkpoint_hook"
    }
}

/// A second hook publishing a *different* field than `DummyCheckpointHook` — proves
/// `merge` actually combines independent hooks' answers rather than the last hook winning
/// outright.
struct DummySpinHook;
impl StepHook for DummySpinHook {
    fn checkpoint_boid_fields(&self, _index: u32) -> BoidCheckpointFields {
        BoidCheckpointFields {
            spin: Some(0.75),
            ..Default::default()
        }
    }
    fn name(&self) -> &'static str {
        "dummy_spin_hook"
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
    reg.register_step_hook("predator", |_| Box::new(DummyPredatorHook));
}

fn config(n: u32, with_predator_hook: bool, spawn_headroom: u32) -> SimConfig {
    SimConfig {
        mode: "dummy_mode".to_string(),
        modifier: "dummy_modifier".to_string(),
        domain: "dummy_domain".to_string(),
        spatial_index: "dummy_spatial_index".to_string(),
        neighbor_selection: "dummy_neighbor_selection".to_string(),
        speed_model: "dummy_speed_model".to_string(),
        init: "dummy_init".to_string(),
        noise: "dummy_noise".to_string(),
        core_params: CoreParams::builder().boid_count(n).build().unwrap(),
        plugin_params: PluginParams::new(),
        init_seed: 99,
        step_hooks: if with_predator_hook {
            vec!["predator".to_string()]
        } else {
            Vec::new()
        },
        predator_count: 0,
        spawn_headroom,
    }
}

fn built_sim(n: u32, with_predator_hook: bool) -> Simulation {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    Simulation::new(config(n, with_predator_hook, 0), &registry).unwrap()
}

fn built_sim_with_headroom(n: u32, headroom: u32) -> Simulation {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    Simulation::new(config(n, true, headroom), &registry).unwrap()
}

#[test]
fn run_batch_checked_with_no_commands_matches_n_sequential_steps() {
    let mut a = built_sim(10, false);
    let mut b = built_sim(10, false);
    a.run_batch_checked(20, 7, Vec::new()).unwrap();
    for _ in 0..20 {
        b.step(b.core_params.dt, 7);
    }
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.step_count(), b.step_count());
}

#[test]
fn invalid_command_rejects_the_whole_batch_before_any_step_runs() {
    let mut sim = built_sim(5, false);
    let commands = vec![
        Command::SetCheckpointStride { stride: 2 },
        Command::SetParam {
            name: "not_a_real_param".to_string(),
            value: 1.0,
        },
    ];
    let result = sim.run_batch_checked(10, 1, commands);
    assert!(result.is_err());
    assert_eq!(sim.step_count(), 0, "no step should have run");
    assert_eq!(
        sim.checkpoint_stride, 1,
        "stride command must not have applied either"
    );
}

#[test]
fn checkpoints_are_captured_at_exactly_the_configured_stride() {
    let mut sim = built_sim(5, false);
    let commands = vec![Command::SetCheckpointStride { stride: 4 }];
    let buffer = sim.run_batch_checked(10, 1, commands).unwrap();
    assert_eq!(buffer.checkpoints.len(), 2); // 10 / 4 = 2, not 3 (no ceiling/partial)
    assert_eq!(buffer.checkpoints[0].step_count, 4);
    assert_eq!(buffer.checkpoints[1].step_count, 8);
}

#[test]
fn run_batch_with_budget_checked_stops_early_and_leaves_state_usable() {
    let mut sim = built_sim(500, false);
    let (buffer, all_done) = sim
        .run_batch_with_budget_checked(1_000_000, 1, Vec::new(), 0.001)
        .unwrap();
    assert!(!all_done);
    assert!(sim.step_count() < 1_000_000);
    // default checkpoint_stride is 1, so one checkpoint per completed step in this batch.
    assert_eq!(sim.step_count(), buffer.checkpoints.len() as u64);
    for v in sim.velocities() {
        assert!(v.is_finite(), "state must stay usable after an early stop");
    }
}

#[test]
fn add_predator_is_a_noop_without_the_predator_hook_composed() {
    let mut sim = built_sim(5, false);
    let before = sim.boid_count();
    sim.run_batch_checked(
        1,
        1,
        vec![Command::AddPredator {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
        }],
    )
    .unwrap();
    assert_eq!(
        sim.boid_count(),
        before,
        "no predator hook composed -> no-op"
    );
}

/// With `spawn_headroom: 0` (the default), `BoidColumns` is still sized to exactly
/// `boid_count` at construction, so `AddPredator` with the `predator` hook composed still
/// no-ops here — a legitimate fixed-capacity limit a host didn't ask to avoid, not the G6
/// bug this used to be (roadmap.md §12, now fixed via `SimConfig::spawn_headroom`).
#[test]
fn add_predator_is_a_noop_at_full_capacity_with_zero_spawn_headroom() {
    let mut sim = built_sim(5, true);
    let before = sim.boid_count();
    sim.run_batch_checked(
        1,
        1,
        vec![Command::AddPredator {
            position: Vec3::new(9.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
        }],
    )
    .unwrap();
    assert_eq!(sim.boid_count(), before, "no free slot -> no-op");
}

/// The G6 fix itself: a host that reserves `spawn_headroom` gets a real, immediate
/// `AddPredator` with no workaround needed (contrast with the `Reset`-first dance the
/// zero-headroom case below still needs — that's a legitimate capacity limit, not a bug).
#[test]
fn add_predator_spawns_a_real_boid_directly_when_spawn_headroom_is_reserved() {
    let mut sim = built_sim_with_headroom(5, 2);
    let before = sim.boid_count();
    let buffer = sim
        .run_batch_checked(
            1,
            1,
            vec![Command::AddPredator {
                position: Vec3::new(9.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
            }],
        )
        .unwrap();
    assert_eq!(sim.boid_count(), before + 1);
    assert_eq!(buffer.checkpoints[0].predators.len(), 1);
}

/// Proves the AddPredator *routing* logic is correct even without reserved headroom, by
/// first using `Reset` to legitimately shrink the flock and free slots within the same
/// fixed capacity — a real capability (Reset), not a workaround for a bug.
#[test]
fn add_predator_spawns_a_real_boid_once_a_slot_is_free_via_reset() {
    let mut sim = built_sim(5, true);
    let buffer = sim
        .run_batch_checked(
            1,
            1,
            vec![
                Command::Reset {
                    count: 3,
                    seed: Some(1),
                },
                Command::AddPredator {
                    position: Vec3::new(9.0, 0.0, 0.0),
                    velocity: Vec3::ZERO,
                },
            ],
        )
        .unwrap();
    assert_eq!(sim.boid_count(), 4); // 3 reset prey + 1 spawned predator
    assert_eq!(buffer.checkpoints.len(), 1);
    assert_eq!(buffer.checkpoints[0].predators.len(), 1);
}

#[test]
fn reset_reinitializes_under_the_same_composition_and_zeroes_step_count() {
    let mut sim = built_sim(5, false);
    sim.run_batch_checked(10, 1, Vec::new()).unwrap();
    assert_eq!(sim.step_count(), 10);
    sim.run_batch_checked(
        1,
        1,
        vec![Command::Reset {
            count: 3,
            seed: Some(42),
        }],
    )
    .unwrap();
    // step_count resets to 0 as part of Reset, then advances by the 1 step this same
    // batch call also runs afterward.
    assert_eq!(sim.step_count(), 1);
    assert_eq!(sim.boid_count(), 3);
}

#[test]
fn checkpoint_schema_is_the_same_rust_type_regardless_of_composition() {
    let mut plain = built_sim(3, false);
    let mut with_predator = built_sim(3, true);
    let a = plain.run_batch_checked(1, 1, Vec::new()).unwrap();
    let b = with_predator.run_batch_checked(1, 1, Vec::new()).unwrap();
    // Both produce the identical `Checkpoint` struct type with the same fields present
    // (enforced by the type system already); this checks the *value*-level "empty, not
    // absent" rule holds too — no predator spawned yet in either, just composed in one.
    assert!(a.checkpoints[0].predators.is_empty());
    assert!(b.checkpoints[0].predators.is_empty());
}

/// The load-bearing proof for the whole checkpoint-field extension: `capture_checkpoint`
/// actually collects `StepHook::checkpoint_boid_fields`/`checkpoint_scene_fields` from
/// every composed hook and merges independent hooks' answers together, without
/// `murmur_core` ever special-casing a hook by name to do it. A `Simulation` composed with
/// *no* checkpoint-publishing hook gets all-`None`/all-empty fields (design/05's own
/// "empty, not absent" rule), proving these fields cost nothing when unused.
#[test]
fn capture_checkpoint_generically_collects_and_merges_every_hooks_own_fields() {
    let mut registry = Registry::new();
    register_all_dummies(&mut registry);
    registry.register_step_hook("dummy_checkpoint_hook", |_| Box::new(DummyCheckpointHook));
    registry.register_step_hook("dummy_spin_hook", |_| Box::new(DummySpinHook));

    let mut with_hooks = config(3, false, 0);
    with_hooks.step_hooks = vec![
        "dummy_checkpoint_hook".to_string(),
        "dummy_spin_hook".to_string(),
    ];
    let mut sim = Simulation::new(with_hooks, &registry).unwrap();
    let buffer = sim.run_batch_checked(1, 1, Vec::new()).unwrap();
    let checkpoint = &buffer.checkpoints[0];

    assert_eq!(checkpoint.scene_fields.dynamic_vision_range, Some(1.25));
    for boid in &checkpoint.boids {
        // Merged from two independent hooks: state/speed_mult from DummyCheckpointHook,
        // spin from DummySpinHook -- neither hook alone reports both.
        assert_eq!(boid.checkpoint_fields.state, Some(2));
        assert_eq!(boid.checkpoint_fields.speed_mult, Some(0.5));
        assert_eq!(boid.checkpoint_fields.spin, Some(0.75));
        assert_eq!(boid.checkpoint_fields.threat_proximity, None);
    }

    let mut without_hooks = Simulation::new(config(3, false, 0), &registry).unwrap();
    let buffer2 = without_hooks.run_batch_checked(1, 1, Vec::new()).unwrap();
    let checkpoint2 = &buffer2.checkpoints[0];
    assert_eq!(checkpoint2.scene_fields.dynamic_vision_range, None);
    assert!(checkpoint2.scene_fields.obstacles.is_empty());
    for boid in &checkpoint2.boids {
        assert_eq!(boid.checkpoint_fields, BoidCheckpointFields::default());
    }
}

#[test]
fn session_header_is_stable_across_the_simulations_lifetime() {
    let mut sim = built_sim(4, false);
    let h1 = sim.session_header();
    sim.run_batch_checked(5, 1, Vec::new()).unwrap();
    let h2 = sim.session_header();
    assert_eq!(h1.session_id, h2.session_id);
    assert_eq!(h1.build_hash, h2.build_hash);
    assert_eq!(h1.composition, h2.composition);
}
