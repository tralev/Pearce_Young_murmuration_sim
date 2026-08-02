//! `SpeedModel` — the trait is core; `BandSpeed` is the one core *implementation* exception
//! (paper-specified, not a strategy choice — design/01_core.md §11/§12.4). `Fixed`, `Ceiling`,
//! and `None` are later plugins against this same trait.
//!
//! **Implementation note on species.** `enforce` takes `species`, added in Phase 8 alongside
//! the predator–prey `StepHook`: the paper's predator model needs a *different* speed band
//! (`[0, predator_speed_factor·v0]`, no lower bound) than prey's `[speed_min_factor·v0, v0]`,
//! and `SpeedModel` — applied uniformly to every boid in the write phase — is the only seam
//! that runs after integration and can enforce it. Not predator-specific coupling: any
//! `SpeedModel` is free to ignore `species` (as `Custom` does here, falling back to the prey
//! band), this just makes per-species bands *possible* generically.

use crate::boids::Species;
use crate::math::{Vec3, MIN_LEN};
use crate::params::CoreParams;
use crate::registry::{PluginParams, Registry};
use crate::rng::{sample_unit_sphere, Rng};

pub trait SpeedModel: Send + Sync {
    /// Enforces this model's speed constraint on one boid's velocity in place, called during
    /// the sequential write phase after acceleration integration (design/01_core.md §8
    /// pipeline step 3). May reseed a stalled (near-zero-speed) boid via `rng`.
    fn enforce(&self, vel: &mut Vec3, species: Species, params: &CoreParams, rng: &mut Rng);
    fn name(&self) -> &'static str;
}

/// Exact renormalisation to `[speed_min_factor·v0, v0]` for prey, `[0, predator_speed_factor·v0]`
/// for predators (design/01_core.md §11: the paper's predator model only clamps a maximum, no
/// unstall floor). A stalled (near-zero-speed) prey boid is unstalled by reseeding to a random
/// heading at `vmin` rather than left undefined (design/01_core.md §12.4). The one `SpeedModel`
/// implementation that lives in `murmur_core` itself — paper-specified, not a strategy choice
/// (design/00_overview.md §2).
pub struct BandSpeed {
    pub predator_speed_factor: f64,
}

impl Default for BandSpeed {
    /// `PREDATOR_SPEED = 2·v0` (sci/sim_new.md default).
    fn default() -> Self {
        BandSpeed {
            predator_speed_factor: 2.0,
        }
    }
}

impl SpeedModel for BandSpeed {
    fn enforce(&self, vel: &mut Vec3, species: Species, params: &CoreParams, rng: &mut Rng) {
        let (vmax, vmin) = match species {
            Species::Predator => (params.cruise_speed * self.predator_speed_factor, 0.0),
            Species::Prey | Species::Custom(_) => (
                params.cruise_speed,
                params.cruise_speed * params.speed_min_factor,
            ),
        };
        let s = vel.len();
        *vel = if s > vmax {
            *vel * (vmax / s)
        } else if s < vmin {
            if s > MIN_LEN {
                *vel * (vmin / s)
            } else {
                sample_unit_sphere(rng) * vmin
            }
        } else {
            *vel
        };
    }

    fn name(&self) -> &'static str {
        "band"
    }
}

/// Registers `BandSpeed` under the name `"band"`, reading `predator_speed_factor` from
/// `PluginParams` (default `2.0`). Not a plugin `register()` (no plugin crate owns this — it's
/// the core exception), but the same registration mechanism, so a host's setup code composes
/// it identically to every other socket (design/02_plugins.md §1).
pub fn register(r: &mut Registry) {
    r.register_speed_model("band", |p: &PluginParams| {
        Box::new(BandSpeed {
            predator_speed_factor: p.get_or("predator_speed_factor", 2.0),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::for_boid;

    fn params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(10.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(1)
            .vision_radius(1.0)
            .build()
            .unwrap()
    }

    #[test]
    fn clamps_overspeed_down_to_vmax() {
        let p = params();
        let mut vel = Vec3::new(100.0, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        BandSpeed::default().enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - p.cruise_speed).abs() < 1e-9);
        assert!(vel.x > 0.0, "direction preserved");
    }

    #[test]
    fn boosts_underspeed_up_to_vmin() {
        let p = params();
        let vmin = p.cruise_speed * p.speed_min_factor;
        let mut vel = Vec3::new(0.5, 0.0, 0.0); // below vmin=3.0, but not stalled
        let mut rng = for_boid(1, 1, 1);
        BandSpeed::default().enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - vmin).abs() < 1e-9);
        assert!(vel.x > 0.0, "direction preserved");
    }

    #[test]
    fn leaves_in_band_speed_untouched() {
        let p = params();
        let mut vel = Vec3::new(5.0, 0.0, 0.0); // within [3, 10]
        let mut rng = for_boid(1, 1, 1);
        let before = vel;
        BandSpeed::default().enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(vel, before);
    }

    #[test]
    fn unstalls_a_zero_velocity_prey_boid_to_vmin_speed() {
        let p = params();
        let vmin = p.cruise_speed * p.speed_min_factor;
        let mut vel = Vec3::ZERO;
        let mut rng = for_boid(1, 1, 1);
        BandSpeed::default().enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!(vel.is_finite());
        assert!((vel.len() - vmin).abs() < 1e-9);
    }

    #[test]
    fn predator_speed_band_has_no_floor_and_a_higher_ceiling() {
        let p = params();
        let band = BandSpeed {
            predator_speed_factor: 2.0,
        };
        let mut rng = for_boid(1, 1, 1);

        // Overspeed predator clamps to 2*v0, not v0.
        let mut fast = Vec3::new(1000.0, 0.0, 0.0);
        band.enforce(&mut fast, Species::Predator, &p, &mut rng);
        assert!((fast.len() - 2.0 * p.cruise_speed).abs() < 1e-9);

        // A slow (but not literally zero) predator is left alone — no unstall floor.
        let mut slow = Vec3::new(0.01, 0.0, 0.0);
        band.enforce(&mut slow, Species::Predator, &p, &mut rng);
        assert_eq!(slow, Vec3::new(0.01, 0.0, 0.0));
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let model = reg
            .resolve_speed_model("band", &PluginParams::new())
            .unwrap();
        assert_eq!(model.name(), "band");
    }

    #[test]
    fn registry_reads_predator_speed_factor_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params_blob = PluginParams::new().with("predator_speed_factor", 3.5);
        let model = reg.resolve_speed_model("band", &params_blob).unwrap();
        let p = params();
        let mut rng = for_boid(1, 1, 1);
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        model.enforce(&mut vel, Species::Predator, &p, &mut rng);
        assert!((vel.len() - 3.5 * p.cruise_speed).abs() < 1e-9);
    }
}
