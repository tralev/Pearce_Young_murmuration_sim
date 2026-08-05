//! `FixedSpeed` — a constant-speed `SpeedModel` plugin (design/01_core.md §11, roadmap.md
//! Phase 13). Every boid's speed is locked to exactly `cruise_speed * speed_factor` — direction
//! preserved, magnitude replaced outright — the classic Vicsek/Reynolds constant-speed
//! convention, distinct from `BandSpeed`'s range-based clamp (`[vmin, vmax]`, direction
//! preserved only when already in-range). Species-agnostic by design: `BandSpeed`'s per-species
//! band exists because the paper's predator model specifically needs one (Phase 8), not
//! because every `SpeedModel` must have one — this plugin is the "no, really, one constant
//! speed for everyone" alternative, proving the seam is real beyond the one core-exception
//! implementation (design/00_overview.md §2's governing rule: `BandSpeed` staying in
//! `murmur_core` is paper-specified, not a claim that it's structurally privileged).

use murmur_core::{
    sample_unit_sphere, CoreParams, PluginParams, Registry, Rng, Species, SpeedModel, Vec3, MIN_LEN,
};

pub struct FixedSpeed {
    pub speed_factor: f64,
}

impl SpeedModel for FixedSpeed {
    fn enforce(&self, vel: &mut Vec3, _species: Species, params: &CoreParams, rng: &mut Rng) {
        let target = params.cruise_speed * self.speed_factor;
        let s = vel.len();
        *vel = if s > MIN_LEN {
            *vel * (target / s)
        } else {
            // Undefined direction from a stall — reseed rather than leave it at ZERO
            // (matching `BandSpeed`'s unstall precedent, design/01_core.md §12.4).
            sample_unit_sphere(rng) * target
        };
    }

    fn name(&self) -> &'static str {
        "fixed_speed"
    }
}

/// Registers `FixedSpeed` under the name `"fixed_speed"`, reading `speed_factor` from
/// `PluginParams` (default `1.0` — exactly `v0`).
pub fn register(r: &mut Registry) {
    r.register_speed_model("fixed_speed", |p: &PluginParams| {
        Box::new(FixedSpeed {
            speed_factor: p.get_or("speed_factor", 1.0),
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
        murmur_conformance::speed_model(&FixedSpeed { speed_factor: 1.0 });
    }

    #[test]
    fn locks_an_overspeed_boid_down_to_the_target_speed() {
        let p = params();
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        FixedSpeed { speed_factor: 1.0 }.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - p.cruise_speed).abs() < 1e-9);
        assert!(vel.x > 0.0, "direction preserved");
    }

    #[test]
    fn boosts_an_underspeed_boid_up_to_the_target_speed() {
        let p = params();
        let mut vel = Vec3::new(0.001, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        FixedSpeed { speed_factor: 1.0 }.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - p.cruise_speed).abs() < 1e-9);
        assert!(vel.x > 0.0, "direction preserved");
    }

    /// The real "fixed, not a range" proof: unlike `BandSpeed`, an *in-range* speed is still
    /// overwritten — there is no "leave it alone" zone.
    #[test]
    fn even_a_speed_already_at_the_target_gets_rewritten_not_left_alone() {
        let p = params();
        let mut vel = Vec3::new(0.0, p.cruise_speed, 0.0); // already exactly at target
        let before = vel;
        let mut rng = for_boid(1, 1, 1);
        FixedSpeed { speed_factor: 1.0 }.enforce(&mut vel, Species::Prey, &p, &mut rng);
        // Same value in this exact case (already correct), but reached via the "set," not
        // "leave alone," code path — checked properly by the next test's mid-range case.
        assert_eq!(vel, before);
    }

    #[test]
    fn a_speed_strictly_between_bandspeeds_min_and_max_is_still_overwritten() {
        let p = params(); // vmin=3.0, vmax=10.0 under BandSpeed's convention
        let mut vel = Vec3::new(5.0, 0.0, 0.0); // BandSpeed would leave this untouched
        let mut rng = for_boid(1, 1, 1);
        FixedSpeed { speed_factor: 1.0 }.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!(
            (vel.len() - p.cruise_speed).abs() < 1e-9,
            "FixedSpeed has no untouched zone — everything snaps to the target"
        );
    }

    #[test]
    fn unstalls_a_zero_velocity_boid_to_the_target_speed() {
        let p = params();
        let mut vel = Vec3::ZERO;
        let mut rng = for_boid(1, 1, 1);
        FixedSpeed { speed_factor: 1.0 }.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!(vel.is_finite());
        assert!((vel.len() - p.cruise_speed).abs() < 1e-9);
    }

    #[test]
    fn is_species_agnostic_unlike_bandspeed() {
        let p = params();
        let mut prey = Vec3::new(1000.0, 0.0, 0.0);
        let mut predator = Vec3::new(1000.0, 0.0, 0.0);
        let mut rng = for_boid(1, 1, 1);
        let model = FixedSpeed { speed_factor: 1.0 };
        model.enforce(&mut prey, Species::Prey, &p, &mut rng);
        model.enforce(&mut predator, Species::Predator, &p, &mut rng);
        assert!(
            (prey.len() - predator.len()).abs() < 1e-9,
            "prey and predator get the same fixed speed"
        );
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let model = reg
            .resolve_speed_model("fixed_speed", &PluginParams::new())
            .unwrap();
        assert_eq!(model.name(), "fixed_speed");
    }

    #[test]
    fn registry_reads_speed_factor_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params_blob = PluginParams::new().with("speed_factor", 0.5);
        let model = reg
            .resolve_speed_model("fixed_speed", &params_blob)
            .unwrap();
        let p = params();
        let mut rng = for_boid(1, 1, 1);
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        model.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert!((vel.len() - 0.5 * p.cruise_speed).abs() < 1e-9);
    }

    /// Proves the `SpeedModel` seam now has ≥2 real occupants (roadmap.md Phase 13 exit gate,
    /// mirroring the other Phase 13 plugins' pattern) — `BandSpeed` lives in `murmur_core`
    /// itself (the one core-exception implementation), registered via
    /// `murmur_core::speed_model::register`, same mechanism every other plugin uses.
    #[test]
    fn band_and_fixed_speed_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_core::speed_model::register(&mut reg);

        let fixed = reg
            .resolve_speed_model("fixed_speed", &PluginParams::new())
            .unwrap();
        let band = reg
            .resolve_speed_model("band", &PluginParams::new())
            .unwrap();
        assert_eq!(fixed.name(), "fixed_speed");
        assert_eq!(band.name(), "band");
    }
}
