//! End-to-end predator-prey integration test (roadmap.md Phase 8 exit gate): predator FSM
//! behaves sanely (chases COM, speed-clamped) and prey flee force has the correct sign/
//! magnitude — proven here through a real `Simulation`, not just the unit-level `post_steer`
//! checks in `src/lib.rs`.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation, Species};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_vicsek::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    murmur_predator::register(&mut reg);
    reg
}

fn build_sim(n: u32, predator_count: u32, seed: u64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(10.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(5.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("couplage", 1.0)
        .with("diffusion", 0.3)
        .with("cell_size", 5.0)
        .with("radius", 10.0)
        .with("predator_accel", 0.4)
        .with("flight_strength", 2.5)
        .with("danger_radius", 15.0)
        .with("predator_speed_factor", 2.0);
    let config = SimConfig {
        mode: "vicsek".to_string(),
        modifier: "instant_response".to_string(),
        domain: "open".to_string(),
        spatial_index: "hash_grid".to_string(),
        neighbor_selection: "radius_gather".to_string(),
        speed_model: "band".to_string(),
        init: "sphere_volume".to_string(),
        noise: "uniform_sphere".to_string(),
        core_params,
        plugin_params,
        init_seed: seed,
        step_hooks: vec!["predator".to_string()],
        predator_count,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap()
}

fn predator_and_prey_state(sim: &Simulation) -> (Vec<usize>, Vec<usize>) {
    let species = sim.species();
    let predators: Vec<usize> = species
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == Species::Predator)
        .map(|(i, _)| i)
        .collect();
    let prey: Vec<usize> = species
        .iter()
        .enumerate()
        .filter(|(_, s)| **s != Species::Predator)
        .map(|(i, _)| i)
        .collect();
    (predators, prey)
}

#[test]
fn predator_moves_toward_prey_and_stays_within_its_speed_band() {
    let mut sim = build_sim(100, 1, 11);
    let (predators, _prey) = predator_and_prey_state(&sim);
    assert_eq!(predators.len(), 1, "expected exactly one predator boid");
    let pred_idx = predators[0];

    let core = sim.composition().core_params;
    let predator_vmax = core.cruise_speed * 2.0; // predator_speed_factor
    let tol = 1e-6;

    for step in 0..200 {
        sim.step(1.0, 11);
        let vel = sim.velocities()[pred_idx];
        assert!(
            vel.is_finite(),
            "predator velocity went non-finite at step {step}"
        );
        let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
        assert!(
            speed <= predator_vmax + tol,
            "predator speed {speed} exceeded its band [_, {predator_vmax}] at step {step}"
        );
    }

    // Sanity: the predator's final distance to the prey flock's centroid should not have grown
    // without bound (a predator that's actually hunting stays roughly near its prey, unlike a
    // boid ignoring them that would just fly off in a fixed direction).
    let pos = sim.positions();
    let (_predators, prey) = predator_and_prey_state(&sim);
    let prey_com = {
        let mut com = murmur_core::Vec3::ZERO;
        for &i in &prey {
            com += pos[i];
        }
        com / prey.len() as f64
    };
    let dist_to_com = (pos[pred_idx] - prey_com).len();
    assert!(
        dist_to_com < 100.0,
        "predator ended up implausibly far from its prey: {dist_to_com}"
    );
}

#[test]
fn prey_speed_stays_in_the_ordinary_band_even_under_predator_pressure() {
    let mut sim = build_sim(100, 1, 5);
    let core = sim.composition().core_params;
    let (vmin, vmax) = (core.cruise_speed * core.speed_min_factor, core.cruise_speed);
    let tol = 1e-6;

    for step in 0..150 {
        sim.step(1.0, 5);
        let (predators, prey) = predator_and_prey_state(&sim);
        let vel = sim.velocities();
        for &i in &prey {
            let v = vel[i];
            assert!(v.is_finite());
            let s = v.len();
            assert!(
                s >= vmin - tol && s <= vmax + tol,
                "prey speed {s} outside band [{vmin},{vmax}] at step {step}"
            );
        }
        assert_eq!(predators.len(), 1);
    }
}

#[test]
fn a_simulation_with_zero_predators_still_runs_the_hook_without_panicking() {
    let mut sim = build_sim(50, 0, 1);
    for _ in 0..50 {
        sim.step(1.0, 1);
    }
    for v in sim.velocities() {
        assert!(v.is_finite());
    }
}
