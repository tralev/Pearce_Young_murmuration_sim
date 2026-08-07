//! `lib.rs`'s own `#[cfg(test)]` module, split into a sibling file purely for navigability (a
//! whole-codebase audit flagged this crate as having the largest single-plugin source file) —
//! no behaviour change, `RicherPredator`'s public API is untouched.

use super::*;
use murmur_core::{BoidColumns, CoreParams, Domain};

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
        .cruise_speed(1.0)
        .max_force(20.0)
        .speed_min_factor(0.3)
        .boid_count(8)
        .vision_radius(10.0)
        .build()
        .unwrap()
}

fn default_hook() -> RicherPredator {
    RicherPredator::new(
        0.4,  // predator_accel
        20.0, // danger_radius
        50.0, // awareness_radius
        30.0, // wave_trigger_radius
        10.0, // wave_relay_radius
        3.0,  // strike_distance
        5,    // approach_max_steps
        3,    // egress_steps
        2.5,  // flight_strength
        4.0,  // push_strength
        1.5,  // wake_strength
        10.0, // wake_corridor_radius
        0.3,  // blackening_strength
        6,    // blackening_neighbors
        1.0,  // split_strength
        0.5,  // split_trigger
        1.0,  // wave_strength
        0.85, // wave_decay
        0.9,  // wave_relay_gain
    )
}

fn boids_with(entries: &[(Vec3, Vec3, Species)]) -> BoidColumns {
    let mut b = BoidColumns::with_capacity(entries.len() as u32);
    for &(p, v, s) in entries {
        b.add(p, v, s, 0);
    }
    b
}

fn ctx_for<'a>(
    index: u32,
    pos: Vec3,
    vel: Vec3,
    species: Species,
    p: &'a CoreParams,
    domain: &'a StubDomain,
) -> BoidCtx<'a> {
    BoidCtx {
        index,
        pos,
        vel,
        species,
        neighbors: &[],
        core_params: p,
        domain,
        step_count: 0,
    }
}

#[test]
fn predator_starts_in_approach_and_switches_to_egress_within_strike_distance() {
    let mut hook = default_hook();
    // Predator right on top of a single prey -> distance 0 < strike_distance.
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);
    let fsm = hook.fsm.lock().unwrap();
    assert_eq!(fsm.get(&0).unwrap().phase, PredatorPhase::Egress);
}

#[test]
fn predator_returns_to_approach_after_egress_steps() {
    let mut hook = default_hook();
    // Prey kept well beyond strike_distance=3.0 throughout, so once back in Approach it
    // doesn't immediately re-trigger Egress on the very next call (a real predator would
    // only re-approach once it's actually put some distance behind it).
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    {
        let mut fsm = hook.fsm.lock().unwrap();
        fsm.insert(
            0,
            PredatorFsm {
                phase: PredatorPhase::Egress,
                phase_timer: 0,
            },
        );
    }
    let mut params = core_params();
    // egress_steps=3: after 3 more calls the timer reaches 3 and flips back to Approach.
    for step in 0..3 {
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: step,
        };
        hook.pre_step(&mut view);
    }
    let fsm = hook.fsm.lock().unwrap();
    assert_eq!(fsm.get(&0).unwrap().phase, PredatorPhase::Approach);
}

#[test]
fn egress_predator_moves_away_from_prey_not_toward() {
    let mut hook = default_hook();
    let boids = boids_with(&[
        (Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    {
        let mut fsm = hook.fsm.lock().unwrap();
        fsm.insert(
            0,
            PredatorFsm {
                phase: PredatorPhase::Egress,
                phase_timer: 0,
            },
        );
    }
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);

    let domain = StubDomain;
    let p = core_params();
    let ctx = ctx_for(
        0,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Predator,
        &p,
        &domain,
    );
    let mut acc = Vec3::ZERO;
    hook.post_steer(ctx, &mut acc, &mut murmur_core::rng::for_boid(1, 2, 3));
    assert!(
        acc.x < 0.0,
        "egress predator should move further away (-x), got {acc:?}"
    );
}

#[test]
fn push_force_exceeds_baseline_flee_and_only_applies_under_an_approaching_predator() {
    let mut hook = default_hook();
    // Prey at distance 10: inside danger_radius=20 (push/flee should apply) but outside
    // strike_distance=3.0, so a fresh Approach-phase FSM stays Approach for this whole
    // pre_step call instead of immediately flipping to Egress before force computation.
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);
    let domain = StubDomain;
    let p = core_params();
    let ctx = ctx_for(
        1,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc_approach = Vec3::ZERO;
    hook.post_steer(
        ctx,
        &mut acc_approach,
        &mut murmur_core::rng::for_boid(1, 2, 3),
    );

    // Egress case: same geometry, predator already egressing -> no push/flee at all.
    let mut hook2 = default_hook();
    let boids2 = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    {
        let mut fsm = hook2.fsm.lock().unwrap();
        fsm.insert(
            0,
            PredatorFsm {
                phase: PredatorPhase::Egress,
                phase_timer: 0,
            },
        );
    }
    let mut params2 = core_params();
    let mut view2 = SimView {
        boids: &boids2,
        core_params: &mut params2,
        step_count: 0,
    };
    hook2.pre_step(&mut view2);
    let ctx2 = ctx_for(
        1,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc_egress = Vec3::ZERO;
    hook2.post_steer(
        ctx2,
        &mut acc_egress,
        &mut murmur_core::rng::for_boid(1, 2, 3),
    );

    assert!(
        acc_approach.x > 0.0,
        "should push away (+x): {acc_approach:?}"
    );
    assert!(
        acc_egress.len() < 1e-9,
        "an egressing predator should trigger no push/flee at all: {acc_egress:?}"
    );
}

#[test]
fn wake_force_pushes_prey_out_of_the_corridor_behind_an_approaching_predator() {
    let hook = default_hook();
    {
        let mut fsm = hook.fsm.lock().unwrap();
        fsm.insert(
            0,
            PredatorFsm {
                phase: PredatorPhase::Approach,
                phase_timer: 0,
            },
        );
    }
    let boids = boids_with(&[
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Species::Predator,
        ), // flying +x
        (Vec3::new(-5.0, 1.0, 0.0), Vec3::ZERO, Species::Prey), // behind, slightly offset
    ]);
    let mut params = core_params();
    let mut hook_mut = default_hook();
    {
        let mut fsm = hook_mut.fsm.lock().unwrap();
        fsm.insert(
            0,
            PredatorFsm {
                phase: PredatorPhase::Approach,
                phase_timer: 0,
            },
        );
    }
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook_mut.pre_step(&mut view);

    let domain = StubDomain;
    let p = core_params();
    let ctx = ctx_for(
        1,
        Vec3::new(-5.0, 1.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc = Vec3::ZERO;
    hook_mut.post_steer(ctx, &mut acc, &mut murmur_core::rng::for_boid(1, 2, 3));
    // Predator flies +x; prey is behind (-x) and offset +y -> wake force should push
    // further +y (out of the corridor), not toward -y.
    assert!(
        acc.y > 0.0,
        "expected lateral push out of the wake corridor, got {acc:?}"
    );
    let _ = hook; // silence unused (kept for symmetry with other tests' setup style)
}

#[test]
fn blackening_applies_within_awareness_radius_but_not_far_beyond_it() {
    let mut hook = default_hook();
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(30.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // within awareness_radius=50, outside danger_radius=20
        (Vec3::new(32.0, 5.0, 0.0), Vec3::ZERO, Species::Prey), // a neighbour to cohere toward
        (Vec3::new(1000.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // far outside awareness_radius
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);

    let domain = StubDomain;
    let p = core_params();
    let ctx_near = ctx_for(
        1,
        Vec3::new(30.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc_near = Vec3::ZERO;
    hook.post_steer(
        ctx_near,
        &mut acc_near,
        &mut murmur_core::rng::for_boid(1, 2, 3),
    );
    assert!(
        acc_near.len() > 1e-6,
        "expected a nonzero blackening cohesion force: {acc_near:?}"
    );

    let ctx_far = ctx_for(
        3,
        Vec3::new(1000.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc_far = Vec3::ZERO;
    hook.post_steer(
        ctx_far,
        &mut acc_far,
        &mut murmur_core::rng::for_boid(1, 2, 3),
    );
    assert!(
        acc_far.len() < 1e-9,
        "expected no blackening force far outside awareness_radius: {acc_far:?}"
    );
}

#[test]
fn wave_triggers_immediately_near_an_approaching_predator_and_relays_to_a_neighbour_next_step() {
    let mut hook = default_hook();
    // Boid 1 is within wave_trigger_radius=30 of the predator; boid 2 is farther (outside
    // trigger range) but within wave_relay_radius=10 of boid 1.
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (
            Vec3::new(25.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Species::Prey,
        ),
        (Vec3::new(33.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);
    {
        let wave = hook.wave.lock().unwrap();
        assert!(
            wave.get(&1).unwrap().zig_phase > 0.99,
            "boid 1 should trigger immediately"
        );
        assert!(
            wave.get(&2).map(|w| w.zig_phase).unwrap_or(0.0) < 1e-9,
            "boid 2 should not have picked up the wave on the same step it started"
        );
    }
    // Second step: boid 2 should now have relayed a decayed copy of boid 1's zig_phase.
    let mut view2 = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 1,
    };
    hook.pre_step(&mut view2);
    let wave = hook.wave.lock().unwrap();
    assert!(
        wave.get(&2).unwrap().zig_phase > 0.0,
        "boid 2 should have relayed the wave on the following step"
    );
}

#[test]
fn split_pulls_a_panicking_boid_toward_other_panicking_boids_not_calm_ones() {
    let mut hook = default_hook();
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(25.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // triggers wave -> panicking
        (Vec3::new(30.0, 5.0, 0.0), Vec3::ZERO, Species::Prey), // also within trigger radius -> panicking, ahead in +y
        (Vec3::new(-40.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // far away, calm, opposite direction
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);

    let domain = StubDomain;
    let p = core_params();
    let ctx = ctx_for(
        1,
        Vec3::new(25.0, 0.0, 0.0),
        Vec3::ZERO,
        Species::Prey,
        &p,
        &domain,
    );
    let mut acc = Vec3::ZERO;
    hook.post_steer(ctx, &mut acc, &mut murmur_core::rng::for_boid(1, 2, 3));
    // The only other panicking boid (index 2) is in the +y direction relative to boid 1;
    // the calm, much-closer-in-x boid (index 3, far -x) must not dominate the split pull.
    assert!(
        acc.y > 0.0,
        "split should pull toward the other panicking boid (+y): {acc:?}"
    );
}

#[test]
fn conforms_to_step_hook_contract() {
    murmur_conformance::step_hook(&mut default_hook());
}

#[test]
fn checkpoint_boid_fields_publishes_real_threat_panic_and_blackening_diagnostics() {
    let mut hook = default_hook();
    // Approach predator at the origin; two prey close enough for push/wave/blackening to
    // all fire together in one `pre_step` call (danger_radius=20, awareness_radius=50,
    // wave_trigger_radius=30, blackening_neighbors=6 -- one nearby prey neighbour is
    // enough to satisfy `take > 0`).
    let boids = boids_with(&[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
        (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        (Vec3::new(12.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
    ]);
    let mut params = core_params();
    let mut view = SimView {
        boids: &boids,
        core_params: &mut params,
        step_count: 0,
    };
    hook.pre_step(&mut view);

    let fields = hook.checkpoint_boid_fields(1); // prey at (10,0,0)
    let threat_proximity = fields
        .threat_proximity
        .expect("threat_proximity must be populated for a boid pre_step has seen");
    // nearest_approach_dist=10, awareness_radius=50 -> 1 - 10/50 = 0.8.
    assert!(
        (threat_proximity - 0.8).abs() < 1e-6,
        "got {}",
        threat_proximity
    );
    assert_eq!(
        fields.panic,
        Some(1.0),
        "wave should trigger immediately within wave_trigger_radius"
    );
    assert_eq!(
        fields.blackening,
        Some(0.3),
        "blackening should apply with a nearby prey neighbour"
    );

    // A predator boid itself carries no prey diagnostics.
    assert_eq!(
        hook.checkpoint_boid_fields(0),
        BoidCheckpointFields::default()
    );
}

#[test]
fn checkpoint_boid_fields_of_a_boid_pre_step_never_saw_is_all_none() {
    let hook = default_hook();
    assert_eq!(
        hook.checkpoint_boid_fields(99),
        BoidCheckpointFields::default()
    );
}

#[test]
fn registered_name_resolves_via_the_registry() {
    let mut reg = Registry::new();
    register(&mut reg);
    let hook = reg
        .resolve_step_hook("predator_fsm", &PluginParams::new())
        .unwrap();
    assert_eq!(hook.name(), "predator_fsm");
}
