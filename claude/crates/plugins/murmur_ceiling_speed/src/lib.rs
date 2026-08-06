//! `CeilingSpeed` — cap-only `SpeedModel` plugin (design/01_core.md §11, roadmap.md Phase 16).
//! Ported (by description — pymurmur's `plugins/speed_model.py` source isn't reachable in this
//! environment) from design/02_plugins.md §5's "`fixed_speed`, `ceiling_speed`, `none_speed` —
//! exact renormalisation / cap-only / no enforcement — alternatives to the core Band clamp."
//!
//! Unlike `BandSpeed`'s `[vmin, vmax]` range (which *boosts* an underspeed boid back up to
//! `vmin`), `CeilingSpeed` only ever clamps *down*: a boid above `vmax` is rescaled to exactly
//! `vmax`, direction preserved; anything at or below `vmax` — including a stalled, zero-speed
//! boid — is left completely untouched, no unstall reseed. Keeps `BandSpeed`'s species-specific
//! ceiling (`predator_speed_factor` for predators, plain `cruise_speed` for prey/custom) so it's
//! a genuine drop-in "same ceiling, no floor" alternative rather than a different policy
//! entirely.

use murmur_core::{CoreParams, PluginParams, Registry, Rng, Species, SpeedModel, Vec3};

pub struct CeilingSpeed {
    pub predator_speed_factor: f64,
}

impl SpeedModel for CeilingSpeed {
    fn enforce(&self, vel: &mut Vec3, species: Species, params: &CoreParams, _rng: &mut Rng) {
        let vmax = match species {
            Species::Predator => params.cruise_speed * self.predator_speed_factor,
            Species::Prey | Species::Custom(_) => params.cruise_speed,
        };
        let s = vel.len();
        if s > vmax {
            *vel *= vmax / s;
        }
    }

    fn name(&self) -> &'static str {
        "ceiling_speed"
    }
}

/// Registers `CeilingSpeed` under the name `"ceiling_speed"`, reading `predator_speed_factor`
/// from `PluginParams` (default `2.0`, matching `BandSpeed`'s own default).
pub fn register(r: &mut Registry) {
    r.register_speed_model("ceiling_speed", |p: &PluginParams| {
        Box::new(CeilingSpeed {
            predator_speed_factor: p.get_or("predator_speed_factor", 2.0),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::rng::for_boid;

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
    fn conforms_to_speed_model_contract() {
        murmur_conformance::speed_model(&CeilingSpeed {
            predator_speed_factor: 2.0,
        });
    }

    #[test]
    fn clamps_an_overspeed_prey_boid_down_to_cruise_speed() {
        let p = params();
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        CeilingSpeed {
            predator_speed_factor: 2.0,
        }
        .enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - p.cruise_speed).abs() < 1e-9);
        assert!(vel.x > 0.0, "direction preserved");
    }

    #[test]
    fn never_boosts_an_underspeed_boid() {
        let p = params();
        let mut vel = Vec3::new(0.001, 0.0, 0.0);
        let before = vel;
        let mut rng = for_boid(1, 1, 1);
        CeilingSpeed {
            predator_speed_factor: 2.0,
        }
        .enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(vel, before, "cap-only must never touch an underspeed boid");
    }

    #[test]
    fn leaves_a_stalled_zero_velocity_boid_untouched_no_unstall_reseed() {
        let p = params();
        let mut vel = Vec3::ZERO;
        let mut rng = for_boid(1, 1, 1);
        CeilingSpeed {
            predator_speed_factor: 2.0,
        }
        .enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(
            vel,
            Vec3::ZERO,
            "no floor means no reseed, unlike BandSpeed/FixedSpeed"
        );
    }

    #[test]
    fn predator_ceiling_is_higher_than_preys() {
        let p = params();
        let mut prey = Vec3::new(1000.0, 0.0, 0.0);
        let mut predator = Vec3::new(1000.0, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        let model = CeilingSpeed {
            predator_speed_factor: 2.0,
        };
        model.enforce(&mut prey, Species::Prey, &p, &mut rng);
        model.enforce(&mut predator, Species::Predator, &p, &mut rng);
        assert!((prey.len() - p.cruise_speed).abs() < 1e-9);
        assert!((predator.len() - 2.0 * p.cruise_speed).abs() < 1e-9);
    }

    #[test]
    fn a_speed_already_within_the_ceiling_is_left_exactly_alone() {
        let p = params();
        let mut vel = Vec3::new(5.0, 0.0, 0.0); // below cruise_speed=10.0
        let before = vel;
        let mut rng = for_boid(1, 1, 1);
        CeilingSpeed {
            predator_speed_factor: 2.0,
        }
        .enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(vel, before);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let model = reg
            .resolve_speed_model("ceiling_speed", &PluginParams::new())
            .unwrap();
        assert_eq!(model.name(), "ceiling_speed");
    }

    #[test]
    fn registry_reads_predator_speed_factor_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params_blob = PluginParams::new().with("predator_speed_factor", 3.0);
        let model = reg
            .resolve_speed_model("ceiling_speed", &params_blob)
            .unwrap();
        let p = params();
        let mut rng = for_boid(1, 1, 1);
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        model.enforce(&mut vel, Species::Predator, &p, &mut rng);
        assert!((vel.len() - 3.0 * p.cruise_speed).abs() < 1e-9);
    }
}
