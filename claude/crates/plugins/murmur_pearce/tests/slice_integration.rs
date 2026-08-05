//! End-to-end slice integration test (roadmap.md Phase 5 exit gate): the default composition
//! — `PearceProjection` + `InstantResponse` + `OpenSpace` + `HashGrid` + `RadiusGather` +
//! `BandSpeed` + `SphereVolume`/`UniformSphere` — run for real, not just unit-tested in
//! isolation. Checks: 1000 steps @ N=1000 with no NaN, speeds stay within the Band, α rises
//! from ~0 within 500 steps at the murmuration phenotype, Θ̄ stays bounded/plausible and
//! doesn't diverge, and `state_hash()` is identical across 1/4/8 rayon threads for the *real*
//! composition (not just the dummy one already proven generically in `murmur_core`).

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation};

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    reg
}

fn murmuration_plugin_params() -> PluginParams {
    PluginParams::new()
        .with("phi_p", 0.03)
        .with("phi_a", 0.80)
        .with("sigma", 4.0)
        .with("body_radius", 0.5)
        .with("max_candidates", 20.0)
        .with("cell_size", 10.0)
        .with("radius", 25.0) // initial sphere-volume placement radius
}

fn build_sim(n: u32, seed: u64) -> Simulation {
    let registry = build_registry();
    let core_params = CoreParams::builder()
        .cruise_speed(1.0)
        .max_force(1.0)
        .speed_min_factor(0.3)
        .boid_count(n)
        .vision_radius(10.0)
        .build()
        .unwrap();
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
        plugin_params: murmuration_plugin_params(),
        init_seed: seed,
        step_hooks: Vec::new(),
        predator_count: 0,
        spawn_headroom: 0,
    };
    Simulation::new(config, &registry).unwrap()
}

#[test]
fn thousand_steps_at_n_1000_no_nan_and_speeds_stay_in_band() {
    let mut sim = build_sim(1000, 42);
    let core = sim.composition().core_params;
    let (vmin, vmax) = (core.cruise_speed * core.speed_min_factor, core.cruise_speed);
    let tol = 1e-6;

    for step in 0..1000u32 {
        sim.step(1.0, 42);
        if step % 100 == 0 || step == 999 {
            for p in sim.positions() {
                assert!(p.is_finite(), "position went non-finite at step {step}");
            }
            for v in sim.velocities() {
                assert!(v.is_finite(), "velocity went non-finite at step {step}");
                let s = v.len();
                assert!(
                    s >= vmin - tol && s <= vmax + tol,
                    "speed {s} outside band [{vmin}, {vmax}] at step {step}"
                );
            }
        }
    }
}

/// Lightweight sanity check, not a calibrated acceptance gate: "does polarisation rise at all
/// from a near-zero, random-heading start". A precise target range for α under the
/// murmuration phenotype needs the canonical parameter table + equilibration protocol that
/// roadmap.md's pre-coding prep and Phase 7 own (numpy-oracle-validated body_radius/
/// vision_radius/density combination) — not re-derived here by guesswork. This test's
/// `PluginParams` (§ `murmuration_plugin_params`) are illustrative, not the validated set.
#[test]
fn polarisation_rises_from_near_zero_within_500_steps_at_murmuration_phenotype() {
    let mut sim = build_sim(500, 7);
    let initial_alpha = sim.metrics().polarisation;

    let mut alphas = Vec::with_capacity(500);
    for _ in 0..500 {
        sim.step(1.0, 7);
        alphas.push(sim.metrics().polarisation);
    }

    let max_alpha_seen = alphas.iter().cloned().fold(initial_alpha, f64::max);
    let late_window_mean: f64 = alphas[400..].iter().sum::<f64>() / 100.0;
    assert!(
        initial_alpha < 0.3,
        "expected a low initial alpha, got {initial_alpha}"
    );
    assert!(
        max_alpha_seen > 2.0 * initial_alpha.max(0.02),
        "expected polarisation to at least double from its initial value within 500 steps: \
         initial={initial_alpha}, max seen={max_alpha_seen}"
    );
    assert!(
        late_window_mean > initial_alpha,
        "expected polarisation in the last 100 steps ({late_window_mean}) to exceed the \
         initial value ({initial_alpha}) — order should persist, not just spike transiently"
    );
}

#[test]
fn internal_opacity_settles_to_a_plausible_stable_value() {
    let mut sim = build_sim(800, 99);

    // Warm up, then sample a window and check it neither collapses to ~0 (fully dispersed,
    // occlusion never triggers) nor saturates to ~1 (collapsed into an opaque ball), and
    // doesn't trend/diverge across the sampled window — a lightweight sanity check, not the
    // formal P-a hard gate (that's Phase 7, with a proper equilibration protocol).
    sim.run_batch(300, 99);

    let mut samples = Vec::new();
    for _ in 0..20 {
        sim.run_batch(10, 99);
        samples.push(sim.metrics().opacity_int);
    }

    for &theta in &samples {
        assert!(theta.is_finite(), "Theta-bar went non-finite");
        assert!(
            (0.0..=1.0).contains(&theta),
            "Theta-bar {theta} outside [0,1]"
        );
    }
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(
        mean > 0.01 && mean < 0.99,
        "Theta-bar {mean} looks collapsed or fully dispersed, not plausible"
    );

    // No monotonic runaway across the sampled window (a crude divergence check): the last
    // sample shouldn't be wildly larger than the mean of the first half.
    let first_half: f64 = samples[..10].iter().sum::<f64>() / 10.0;
    let last = *samples.last().unwrap();
    assert!(
        (last - first_half).abs() < 0.5,
        "Theta-bar drifted from {first_half} to {last} across the sampled window — looks like it's diverging"
    );
}

#[test]
fn state_hash_is_identical_across_1_4_and_8_rayon_threads_for_the_real_composition() {
    fn run_with_pool_size(threads: usize) -> u64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sim = build_sim(300, 2024);
            sim.run_batch(50, 2024);
            sim.state_hash()
        })
    }
    let h1 = run_with_pool_size(1);
    let h4 = run_with_pool_size(4);
    let h8 = run_with_pool_size(8);
    assert_eq!(
        h1, h4,
        "state_hash differs between 1 and 4 threads (real composition)"
    );
    assert_eq!(
        h1, h8,
        "state_hash differs between 1 and 8 threads (real composition)"
    );
}

#[test]
fn composition_and_describe_report_the_expected_default_plugin_selection() {
    let sim = build_sim(10, 1);
    let names = sim.composition().plugin_names;
    assert!(names.contains(&("mode", "pearce")));
    assert!(names.contains(&("modifier", "instant_response")));
    assert!(names.contains(&("domain", "open")));
    assert!(names.contains(&("spatial_index", "hash_grid")));
    assert!(names.contains(&("neighbor_selection", "radius_gather")));
    assert!(names.contains(&("speed_model", "band")));
}
