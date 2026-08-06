//! `NoneSpeed` — no-enforcement `SpeedModel` plugin (design/01_core.md §11, roadmap.md Phase
//! 16). Ported (by description — pymurmur's `plugins/speed_model.py` source isn't reachable in
//! this environment) from design/02_plugins.md §5's "`fixed_speed`, `ceiling_speed`,
//! `none_speed` — exact renormalisation / cap-only / no enforcement — alternatives to the core
//! Band clamp."
//!
//! A true no-op: `enforce` never touches `vel`. Proves the `SpeedModel` seam is a real,
//! optional socket, not a load-bearing part of the integration pipeline every composition must
//! fill with *some* clamp — a caller building e.g. a pure force-integration demo (no speed
//! constraint at all) can select this and get exactly that. Note this means velocity magnitude
//! is genuinely unbounded step over step under this model — that's the intended behaviour, not
//! a gap; any caller wanting bounded speed should pick `band`/`fixed_speed`/`ceiling_speed`
//! instead.

use murmur_core::{CoreParams, PluginParams, Registry, Rng, Species, SpeedModel, Vec3};

pub struct NoneSpeed;

impl SpeedModel for NoneSpeed {
    fn enforce(&self, _vel: &mut Vec3, _species: Species, _params: &CoreParams, _rng: &mut Rng) {}

    fn name(&self) -> &'static str {
        "none_speed"
    }
}

/// Registers `NoneSpeed` under the name `"none_speed"`. Takes no parameters.
pub fn register(r: &mut Registry) {
    r.register_speed_model("none_speed", |_p: &PluginParams| Box::new(NoneSpeed));
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
        murmur_conformance::speed_model(&NoneSpeed);
    }

    #[test]
    fn leaves_an_overspeed_boid_completely_untouched() {
        let p = params();
        let mut vel = Vec3::new(1000.0, 0.0, 0.0);
        let before = vel;
        let mut rng = for_boid(1, 1, 1);
        NoneSpeed.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(vel, before);
    }

    #[test]
    fn leaves_a_stalled_zero_velocity_boid_untouched() {
        let p = params();
        let mut vel = Vec3::ZERO;
        let mut rng = for_boid(1, 1, 1);
        NoneSpeed.enforce(&mut vel, Species::Prey, &p, &mut rng);
        assert_eq!(vel, Vec3::ZERO);
    }

    #[test]
    fn is_species_agnostic_since_it_does_nothing_at_all() {
        let p = params();
        let mut prey = Vec3::new(37.0, -4.0, 2.0);
        let mut predator = prey;
        let mut rng = for_boid(1, 1, 1);
        NoneSpeed.enforce(&mut prey, Species::Prey, &p, &mut rng);
        NoneSpeed.enforce(&mut predator, Species::Predator, &p, &mut rng);
        assert_eq!(prey, predator);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let model = reg
            .resolve_speed_model("none_speed", &PluginParams::new())
            .unwrap();
        assert_eq!(model.name(), "none_speed");
    }

    /// Proves the `SpeedModel` seam now has ≥4 real occupants (mirroring Phase 13's
    /// seam-plurality proof pattern) — `BandSpeed` (core exception) plus `fixed_speed`,
    /// `ceiling_speed`, and this plugin all resolve via the same registry mechanism.
    #[test]
    fn band_fixed_ceiling_and_none_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_core::speed_model::register(&mut reg);
        murmur_fixed_speed::register(&mut reg);
        murmur_ceiling_speed::register(&mut reg);

        assert_eq!(
            reg.resolve_speed_model("none_speed", &PluginParams::new())
                .unwrap()
                .name(),
            "none_speed"
        );
        assert_eq!(
            reg.resolve_speed_model("band", &PluginParams::new())
                .unwrap()
                .name(),
            "band"
        );
        assert_eq!(
            reg.resolve_speed_model("fixed_speed", &PluginParams::new())
                .unwrap()
                .name(),
            "fixed_speed"
        );
        assert_eq!(
            reg.resolve_speed_model("ceiling_speed", &PluginParams::new())
                .unwrap()
                .name(),
            "ceiling_speed"
        );
    }
}
