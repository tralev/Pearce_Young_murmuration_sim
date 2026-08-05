//! `InstantResponse` — the first `SteeringModifier` plugin (design/01_core.md §8). Memoryless,
//! instant response: exactly what every `FlockingMode` used to do internally before this trait
//! existed. Selected as the slice default; carries no privileged status — `murmur_spin_wave`
//! (behavioural inertia, a later plugin) composes with any `FlockingMode` the same way.

use murmur_core::{clamp_len, BoidCtx, PluginParams, Registry, SteeringModifier, Vec3};

pub struct InstantResponse;

impl SteeringModifier for InstantResponse {
    fn respond(&self, ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3 {
        clamp_len(desired_v - current_vel, ctx.core_params.max_force)
    }

    fn name(&self) -> &'static str {
        "instant_response"
    }
}

/// Registers `InstantResponse` under the name `"instant_response"`.
pub fn register(r: &mut Registry) {
    r.register_modifier("instant_response", |_p: &PluginParams| {
        Box::new(InstantResponse)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Species};

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

    fn ctx<'a>(core_params: &'a CoreParams, domain: &'a StubDomain) -> BoidCtx<'a> {
        BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel: Vec3::new(1.0, 0.0, 0.0),
            species: Species::Prey,
            neighbors: &[],
            core_params,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn clamps_to_max_force_when_the_desired_change_is_large() {
        let p = CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(2.0)
            .speed_min_factor(0.3)
            .boid_count(1)
            .vision_radius(1.0)
            .build()
            .unwrap();
        let d = StubDomain;
        let acc = InstantResponse.respond(ctx(&p, &d), Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO);
        assert!((acc.len() - 2.0).abs() < 1e-9);
        assert!(acc.x > 0.0, "direction preserved");
    }

    #[test]
    fn leaves_small_desired_changes_untouched() {
        let p = CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(10.0)
            .speed_min_factor(0.3)
            .boid_count(1)
            .vision_radius(1.0)
            .build()
            .unwrap();
        let d = StubDomain;
        let acc = InstantResponse.respond(ctx(&p, &d), Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        assert_eq!(acc, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn conforms_to_steering_modifier_contract() {
        murmur_conformance::steering_modifier(&InstantResponse);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let m = reg
            .resolve_modifier("instant_response", &PluginParams::new())
            .unwrap();
        assert_eq!(m.name(), "instant_response");
    }

    /// Proves the `SteeringModifier` seam is real: a trivial stub implementing the same trait
    /// registers and resolves alongside the real plugin (roadmap.md Phase 3-pattern exit gate,
    /// applied here to the trait `SteeringModifier` gained in Phase 5).
    #[test]
    fn a_different_steering_modifier_plugin_swaps_in_via_the_same_seam() {
        struct StubModifier;
        impl SteeringModifier for StubModifier {
            fn respond(&self, _ctx: BoidCtx<'_>, _desired_v: Vec3, _current_vel: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn name(&self) -> &'static str {
                "stub_modifier"
            }
        }
        let mut reg = Registry::new();
        register(&mut reg);
        reg.register_modifier("stub_modifier", |_p| Box::new(StubModifier));

        let real = reg
            .resolve_modifier("instant_response", &PluginParams::new())
            .unwrap();
        let stub = reg
            .resolve_modifier("stub_modifier", &PluginParams::new())
            .unwrap();
        assert_eq!(real.name(), "instant_response");
        assert_eq!(stub.name(), "stub_modifier");
    }
}
