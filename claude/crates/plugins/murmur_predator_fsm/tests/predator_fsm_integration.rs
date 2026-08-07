//! End-to-end predator/prey FSM integration test (roadmap.md Phase 15 exit gate), mirroring
//! `murmur_predator/tests/predator_integration.rs`'s pattern: proven through a real
//! `Simulation`, not just `src/lib.rs`'s unit-level `post_steer` checks.

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation, Species};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    murmur_predator_fsm::register(&mut reg);
    reg
}

fn build_sim(n: u32, predator_count: u32, seed: u64, init_radius: f64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(10.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(10.0)
        .build()
        .unwrap();
    let plugin_params = PluginParams::new()
        .with("phi_p", 0.50)
        .with("phi_a", 0.20)
        .with("sigma", 4.0)
        .with("cell_size", 10.0)
        .with("radius", init_radius)
        .with("predator_accel", 0.4)
        .with("flight_strength", 2.5)
        .with("danger_radius", 10.0)
        .with("awareness_radius", 25.0)
        .with("wave_trigger_radius", 15.0)
        .with("wave_relay_radius", 6.0)
        .with("strike_distance", 1.5)
        .with("approach_max_steps", 60.0)
        .with("egress_steps", 20.0)
        .with("predator_speed_factor", 2.0);
    let config = SimConfig {
        mode: "pearce".to_string(),
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
        step_hooks: vec!["predator_fsm".to_string()],
        predator_count,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap().0
}

fn species_split(sim: &Simulation) -> (Vec<usize>, Vec<usize>) {
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
fn predator_stays_speed_banded_and_prey_stay_finite_over_a_long_run() {
    let mut sim = build_sim(100, 1, 11, 8.0);
    let core = sim.composition().core_params;
    let predator_vmax = core.cruise_speed * 2.0;
    let (vmin, vmax) = (core.cruise_speed * core.speed_min_factor, core.cruise_speed);
    let tol = 1e-6;

    for step in 0..300 {
        sim.step(1.0, 11);
        let (predators, prey) = species_split(&sim);
        let vel = sim.velocities();
        for &i in &predators {
            let s = vel[i].len();
            assert!(vel[i].is_finite());
            assert!(
                s <= predator_vmax + tol,
                "predator speed {s} exceeded band at step {step}"
            );
        }
        for &i in &prey {
            let v = vel[i];
            assert!(
                v.is_finite(),
                "prey velocity went non-finite at step {step}"
            );
            let s = v.len();
            assert!(
                s >= vmin - tol && s <= vmax + tol,
                "prey speed {s} outside band [{vmin},{vmax}] at step {step}"
            );
        }
    }
}

#[test]
fn predator_fsm_actually_oscillates_between_approach_and_egress_over_a_long_run() {
    // Indirect proof (no direct FSM-state accessor on Simulation): an Approach-phase predator
    // pursues the prey centroid, an Egress-phase predator flees it -- so if the FSM never
    // transitioned, the predator's distance to the prey centroid would either monotonically
    // shrink (stuck in Approach) or monotonically grow (stuck in Egress) forever. Real
    // oscillation should show both expansions and contractions of that distance over a long
    // run.
    let mut sim = build_sim(100, 1, 3, 8.0);
    let (predators, _prey) = species_split(&sim);
    let pred_idx = predators[0];

    let mut distances = Vec::new();
    for _ in 0..400 {
        sim.step(1.0, 3);
        let pos = sim.positions();
        let (_predators, prey) = species_split(&sim);
        let mut com = murmur_core::Vec3::ZERO;
        for &i in &prey {
            com += pos[i];
        }
        com /= prey.len() as f64;
        distances.push((pos[pred_idx] - com).len());
    }
    let increases = distances.windows(2).filter(|w| w[1] > w[0] + 1e-9).count();
    let decreases = distances.windows(2).filter(|w| w[1] < w[0] - 1e-9).count();
    assert!(
        increases > 5 && decreases > 5,
        "expected both approach (distance shrinking) and egress (distance growing) phases over \
         400 steps, got {increases} increases and {decreases} decreases -- FSM may not be \
         oscillating"
    );
}

#[test]
fn a_simulation_with_zero_predators_still_runs_the_hook_without_panicking() {
    let mut sim = build_sim(50, 0, 1, 8.0);
    for _ in 0..80 {
        sim.step(1.0, 1);
    }
    for v in sim.velocities() {
        assert!(v.is_finite());
    }
}

#[test]
fn sustained_predator_pressure_measurably_increases_flock_dispersion_vs_no_predator() {
    // Not a claim that it *always* splits into disconnected components (that depends on
    // geometry/timing this test doesn't control precisely) -- a weaker, still-real claim that
    // sustained predator pressure (push/wave/split all engaging repeatedly over many
    // approach/egress cycles) visibly disperses the flock more than an unharried one, which a
    // richer FSM should show and the Phase 8 minimal plugin's own simpler flee-only force
    // wasn't designed to emphasize as strongly.
    let mut with_pred = build_sim(150, 1, 42, 8.0);
    let mut without_pred = build_sim(150, 0, 42, 8.0);

    for _ in 0..500 {
        with_pred.step(1.0, 42);
        without_pred.step(1.0, 42);
    }

    let dispersion = |sim: &Simulation, exclude_predators: bool| -> f64 {
        let pos = sim.positions();
        let species = sim.species();
        let mut com = murmur_core::Vec3::ZERO;
        let mut n = 0usize;
        for (i, &p) in pos.iter().enumerate() {
            if exclude_predators && species[i] == Species::Predator {
                continue;
            }
            com += p;
            n += 1;
        }
        com /= n as f64;
        let mut var = 0.0;
        for (i, &p) in pos.iter().enumerate() {
            if exclude_predators && species[i] == Species::Predator {
                continue;
            }
            var += (p - com).len_sq();
        }
        (var / n as f64).sqrt()
    };

    let d_with = dispersion(&with_pred, true);
    let d_without = dispersion(&without_pred, false);
    assert!(
        d_with > d_without,
        "expected sustained predator pressure to increase prey dispersion: with={d_with}, \
         without={d_without}"
    );
}
