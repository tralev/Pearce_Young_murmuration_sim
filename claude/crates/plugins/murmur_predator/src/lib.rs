//! Predator–prey — the slice's minimal `StepHook` plugin (design/01_core.md §11,
//! `sci/sim_new.md`). Not architecturally core (predator–prey is a `StepHook` plugin like any
//! other), built early because it's part of the validated scientific scope. This is the
//! *minimal* `sci/sim_new.md` version; pymurmur's richer FSM + force-bundle superset is a
//! later, separate plugin (design/02_plugins.md §5).
//!
//! **Implementation notes on fitting predator–prey into the `StepHook` seam.**
//! - The predator's own motion (`v_pred += PREDATOR_ACCEL·â`, then clamp to `PREDATOR_SPEED`)
//!   needs the prey centre of mass — global aggregate state no single boid's `BoidCtx` carries.
//!   `pre_step` computes it once (and the list of predator positions, for prey's flee force)
//!   from `SimView::boids`, caching both in a `Mutex` (`StepHook` requires `Sync`; the cache is
//!   written once in `pre_step` and only read in `post_steer`, never concurrently in this
//!   pipeline's sequential write phase, but a `Mutex` is the simplest `Sync`-legal way to
//!   express "computed ahead, read many times").
//! - `post_steer` computes the acceleration a `SteeringModifier` would have needed to reproduce
//!   `v_pred += PREDATOR_ACCEL·â` at `dt=1` (this project's locked integration convention,
//!   design/01_core.md §8) directly from the *current* velocity, then *overwrites* `*acc` (not
//!   `+=`) for predator boids — the paper's predator model is a standalone behaviour, not a
//!   layer on top of whatever `FlockingMode` (e.g. Pearce) would otherwise have produced for a
//!   boid that happens to be a predator.
//! - `PREDATOR_SPEED = 2·v0` is enforced by `BandSpeed`'s species-aware band (a Phase 8
//!   addition to `murmur_core::speed_model`), not by this hook — `post_steer` runs before
//!   integration/speed-enforcement, so clamping the acceleration here would just get
//!   overwritten by the write phase's subsequent speed enforcement anyway.
//! - Prey's flee force *is* additive (`+=`), matching the trait's documented default use.

use std::sync::Mutex;

use murmur_core::{clamp_len, BoidCtx, PluginParams, Registry, SimView, Species, StepHook, Vec3};

struct Cache {
    prey_com: Vec3,
    predator_positions: Vec<Vec3>,
}

pub struct PredatorPrey {
    pub predator_accel: f64,
    pub danger_radius: f64,
    pub flight_strength: f64,
    cache: Mutex<Cache>,
}

impl PredatorPrey {
    pub fn new(predator_accel: f64, danger_radius: f64, flight_strength: f64) -> Self {
        PredatorPrey {
            predator_accel,
            danger_radius,
            flight_strength,
            cache: Mutex::new(Cache {
                prey_com: Vec3::ZERO,
                predator_positions: Vec::new(),
            }),
        }
    }
}

impl StepHook for PredatorPrey {
    fn pre_step(&mut self, sim: &mut SimView) {
        let mut com = Vec3::ZERO;
        let mut prey_count = 0u32;
        let mut predator_positions = Vec::new();
        for i in sim.boids.iter_active() {
            let idx = i as usize;
            match sim.boids.species[idx] {
                Species::Predator => predator_positions.push(sim.boids.pos[idx]),
                Species::Prey | Species::Custom(_) => {
                    com += sim.boids.pos[idx];
                    prey_count += 1;
                }
            }
        }
        let prey_com = if prey_count > 0 {
            com / prey_count as f64
        } else {
            Vec3::ZERO
        };
        let mut cache = self.cache.lock().expect("predator cache mutex poisoned");
        cache.prey_com = prey_com;
        cache.predator_positions = predator_positions;
    }

    fn post_steer(&self, ctx: BoidCtx<'_>, acc: &mut Vec3) {
        let cache = self.cache.lock().expect("predator cache mutex poisoned");
        match ctx.species {
            Species::Predator => {
                let to_prey = cache.prey_com - ctx.pos;
                // Overwrite, not add: predator motion is a standalone behaviour, not a layer
                // on top of whatever FlockingMode produced (see module doc).
                *acc = if to_prey.len_sq() > 1e-18 {
                    let desired_vel = ctx.vel + to_prey.normalized() * self.predator_accel;
                    desired_vel - ctx.vel
                } else {
                    Vec3::ZERO
                };
            }
            Species::Prey | Species::Custom(_) => {
                let mut flee = Vec3::ZERO;
                for &pred_pos in &cache.predator_positions {
                    let away = ctx.pos - pred_pos;
                    let d = away.len();
                    if d < 1e-9 || d >= self.danger_radius {
                        continue; // sci/sim_new.md: zero flee force beyond danger_radius
                    }
                    flee +=
                        away.normalized() * (self.flight_strength * (1.0 - d / self.danger_radius));
                }
                *acc += clamp_len(flee, ctx.core_params.max_force);
            }
        }
    }

    fn name(&self) -> &'static str {
        "predator"
    }
}

/// Registers `PredatorPrey` under the name `"predator"`. Defaults match `sci/sim_new.md`'s
/// starling/raptor scenario: `PREDATOR_ACCEL = 0.4`, `FLIGHT_STRENGTH = 2.5`,
/// `DANGER_RADIUS = 160·(b/9)` where `b` is read from `PluginParams`' `"body_radius"` (default
/// `9.0`, giving the paper's own `DANGER_RADIUS = 160` baseline). `PREDATOR_SPEED = 2·v0` is
/// enforced by `BandSpeed`, not here — see module doc.
pub fn register(r: &mut Registry) {
    r.register_step_hook("predator", |p: &PluginParams| {
        let body_radius = p.get_or("body_radius", 9.0);
        let danger_radius = p.get_or("danger_radius", 160.0 * (body_radius / 9.0));
        Box::new(PredatorPrey::new(
            p.get_or("predator_accel", 0.4),
            danger_radius,
            p.get_or("flight_strength", 2.5),
        ))
    });
}

#[cfg(test)]
mod tests {
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
            .max_force(10.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn boids_with(entries: &[(Vec3, Vec3, Species)]) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(entries.len() as u32);
        for &(p, v, s) in entries {
            b.add(p, v, s, 0);
        }
        b
    }

    #[test]
    fn predator_accelerates_toward_prey_centre_of_mass() {
        let mut hook = PredatorPrey::new(0.4, 50.0, 2.5);
        let boids = boids_with(&[
            (Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        // prey COM = (5, 0, 0); predator at (-10,0,0) -> should accelerate in +x.
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);

        let domain = StubDomain;
        let p = core_params();
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::new(-10.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Predator,
            neighbors: &[],
            core_params: &p,
            domain: &domain,
        };
        let mut acc = Vec3::new(999.0, 999.0, 999.0); // pre-existing value must be overwritten
        hook.post_steer(ctx, &mut acc);
        assert!(
            acc.x > 0.0,
            "predator should accelerate toward +x (prey COM), got {acc:?}"
        );
        assert!((acc.y).abs() < 1e-9 && (acc.z).abs() < 1e-9);
        assert!(
            (acc.len() - 0.4).abs() < 1e-9,
            "acc magnitude should equal predator_accel"
        );
    }

    #[test]
    fn prey_flee_force_points_away_from_a_nearby_predator_and_is_zero_beyond_danger_radius() {
        let mut hook = PredatorPrey::new(0.4, 20.0, 2.5);
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // within danger_radius=20
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

        // Close prey: flee force should point away from the predator (+x) and be nonzero.
        let ctx_close = BoidCtx {
            index: 1,
            pos: Vec3::new(5.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: &p,
            domain: &domain,
        };
        let mut acc_close = Vec3::ZERO;
        hook.post_steer(ctx_close, &mut acc_close);
        assert!(
            acc_close.x > 0.0,
            "flee force should point away (+x), got {acc_close:?}"
        );

        // Far prey (beyond danger_radius): flee force must be exactly zero.
        let ctx_far = BoidCtx {
            index: 2,
            pos: Vec3::new(500.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: &p,
            domain: &domain,
        };
        let mut acc_far = Vec3::ZERO;
        hook.post_steer(ctx_far, &mut acc_far);
        assert_eq!(
            acc_far,
            Vec3::ZERO,
            "flee force must be zero beyond danger_radius"
        );
    }

    #[test]
    fn flee_force_magnitude_ramps_down_with_distance() {
        let hook = PredatorPrey::new(0.4, 20.0, 2.5);
        *hook.cache.lock().unwrap() = Cache {
            prey_com: Vec3::ZERO,
            predator_positions: vec![Vec3::ZERO],
        };

        let domain = StubDomain;
        let p = core_params();
        let near_ctx = BoidCtx {
            index: 1,
            pos: Vec3::new(2.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: &p,
            domain: &domain,
        };
        let far_ctx = BoidCtx {
            pos: Vec3::new(18.0, 0.0, 0.0),
            ..near_ctx
        };

        let mut acc_near = Vec3::ZERO;
        hook.post_steer(near_ctx, &mut acc_near);
        let mut acc_far = Vec3::ZERO;
        hook.post_steer(far_ctx, &mut acc_far);
        assert!(
            acc_near.len() > acc_far.len(),
            "closer prey should feel a stronger flee force: near={acc_near:?}, far={acc_far:?}"
        );
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        murmur_conformance::step_hook(&mut PredatorPrey::new(0.4, 20.0, 2.5));
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("predator", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "predator");
    }
}
