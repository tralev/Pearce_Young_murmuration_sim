//! `ExternalField` — a constant directional-bias `StepHook` plugin (design/02_plugins.md §5,
//! roadmap.md Phase 13). Adds a fixed `direction * strength` acceleration to every boid, every
//! step — the simplest possible seam occupant, deliberately picked as the second `StepHook`
//! (after `predator`, a comparatively complex, very domain-specific hook) to prove the seam
//! generalizes to something tiny too. Also has real downstream value: comparing a flock's
//! spontaneous order (no field) against its directed order (field on) is the standard
//! spontaneous-vs-directed-flocking analysis (`sci/`'s `structure_factor.py`/
//! `direction_fluctuation.py`, Phase 15) — this plugin is what "field on" means operationally.
//!
//! No scientific specificity of its own: a uniform bias field is one of several plausible ways
//! to study directed vs. spontaneous order, not privileged over any other `StepHook`.

use murmur_core::{BoidCtx, PluginParams, Registry, Rng, StepHook, Vec3};

pub struct ExternalField {
    /// Unit direction — normalized at construction, `Vec3::ZERO` (never NaN) if the configured
    /// raw direction was itself zero, per `Vec3::normalized()`'s invariant.
    direction: Vec3,
    strength: f64,
}

impl StepHook for ExternalField {
    fn post_steer(&self, _ctx: BoidCtx<'_>, acc: &mut Vec3, _rng: &mut Rng) {
        *acc += self.direction * self.strength;
    }

    fn name(&self) -> &'static str {
        "external_field"
    }
}

/// Registers `ExternalField` under the name `"external_field"`, reading `field_x`/`field_y`/
/// `field_z` (raw direction, default `(1, 0, 0)`, normalized here at construction) and
/// `field_strength` (default `0.1`) from `PluginParams`.
pub fn register(r: &mut Registry) {
    r.register_step_hook("external_field", |p: &PluginParams| {
        let raw = Vec3::new(
            p.get_or("field_x", 1.0),
            p.get_or("field_y", 0.0),
            p.get_or("field_z", 0.0),
        );
        Box::new(ExternalField {
            direction: raw.normalized(),
            strength: p.get_or("field_strength", 0.1),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Species};

    struct DummyDomain;
    impl Domain for DummyDomain {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "dummy_domain"
        }
    }

    fn ctx<'a>(core_params: &'a CoreParams, domain: &'a dyn Domain) -> BoidCtx<'a> {
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

    fn params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(1)
            .vision_radius(1.0)
            .build()
            .unwrap()
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = ExternalField {
            direction: Vec3::new(1.0, 0.0, 0.0),
            strength: 0.1,
        };
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn adds_exactly_direction_times_strength_to_acc() {
        let hook = ExternalField {
            direction: Vec3::new(0.0, 1.0, 0.0),
            strength: 2.5,
        };
        let p = params();
        let domain = DummyDomain;
        let mut acc = Vec3::new(1.0, 1.0, 1.0);
        hook.post_steer(
            ctx(&p, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert!((acc - Vec3::new(1.0, 1.0 + 2.5, 1.0)).len() < 1e-9);
    }

    #[test]
    fn applies_every_step_additively_not_just_once() {
        let hook = ExternalField {
            direction: Vec3::new(1.0, 0.0, 0.0),
            strength: 1.0,
        };
        let p = params();
        let domain = DummyDomain;
        let mut acc = Vec3::ZERO;
        for _ in 0..5 {
            hook.post_steer(
                ctx(&p, &domain),
                &mut acc,
                &mut murmur_core::rng::for_boid(1, 2, 3),
            );
        }
        assert!((acc.x - 5.0).abs() < 1e-9, "should accumulate across calls");
    }

    #[test]
    fn registered_direction_is_normalized_regardless_of_input_magnitude() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params_blob = PluginParams::new()
            .with("field_x", 100.0)
            .with("field_y", 0.0)
            .with("field_z", 0.0)
            .with("field_strength", 1.0);
        let hook = reg
            .resolve_step_hook("external_field", &params_blob)
            .unwrap();
        let p = params();
        let domain = DummyDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&p, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert!(
            (acc.len() - 1.0).abs() < 1e-9,
            "a raw field_x=100 direction should still normalize to unit length before scaling \
             by field_strength=1.0, got |acc|={}",
            acc.len()
        );
    }

    #[test]
    fn zero_direction_input_produces_zero_contribution_not_nan() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params_blob = PluginParams::new()
            .with("field_x", 0.0)
            .with("field_y", 0.0)
            .with("field_z", 0.0)
            .with("field_strength", 5.0);
        let hook = reg
            .resolve_step_hook("external_field", &params_blob)
            .unwrap();
        let p = params();
        let domain = DummyDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&p, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert_eq!(acc, Vec3::ZERO);
        assert!(acc.is_finite());
    }

    #[test]
    fn default_field_strength_is_small_not_dominant() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("external_field", &PluginParams::new())
            .unwrap();
        let p = params();
        let domain = DummyDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(&p, &domain),
            &mut acc,
            &mut murmur_core::rng::for_boid(1, 2, 3),
        );
        assert!((acc.len() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("external_field", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "external_field");
    }

    /// Proves the `StepHook` seam now has ≥2 real occupants (roadmap.md Phase 13 exit gate,
    /// mirroring the other Phase 13 plugins' pattern) — against `predator`, the seam's first
    /// (and considerably more complex) occupant.
    #[test]
    fn predator_and_external_field_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_predator::register(&mut reg);

        let field = reg
            .resolve_step_hook("external_field", &PluginParams::new())
            .unwrap();
        let predator = reg
            .resolve_step_hook("predator", &PluginParams::new())
            .unwrap();
        assert_eq!(field.name(), "external_field");
        assert_eq!(predator.name(), "predator");
    }
}
