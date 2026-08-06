//! `SphereSoft` — asymptotic 1/gap inward-push `Domain` plugin, no hard clamp (design/01_core.md
//! §3, roadmap.md Phase 16). Ported from pymurmur's
//! `plugins/boundary/strategies.py::SphereSoftBoundary` (design/02_plugins.md §5 — this
//! strategy does not exist in droidmur, which only has `margin`; pymurmur is the sole source).
//!
//! Unlike `Sphere`'s hard projection, `SphereSoft` never clamps position: instead it applies an
//! inward velocity push that grows as `1 / gap`, `gap` being the (signed) distance from the
//! nominal `radius` — negligible far inside, strong near the boundary, and — because a literal
//! `1/0` singularity would immediately produce an infinite, non-finite velocity and defeat the
//! entire point of a *soft*, never-diverging boundary — floored at a small constant so the push
//! stays large-but-finite even exactly at or beyond `radius` (`Domain::apply` is contractually
//! required to leave `pos`/`vel` finite, design/01_core.md §3, `murmur_conformance::domain`).

use murmur_core::{Domain, PluginParams, Registry, Vec3, MIN_LEN};

/// Floor on `gap` before it's used as a divisor — keeps the push finite instead of literally
/// infinite right at (or beyond) `radius`.
const MIN_GAP: f64 = 1e-3;

pub struct SphereSoft {
    radius: f64,
    push_strength: f64,
}

impl SphereSoft {
    pub fn new(radius: f64, push_strength: f64) -> Self {
        SphereSoft {
            radius,
            push_strength,
        }
    }
}

impl Domain for SphereSoft {
    /// Not periodic — same direct displacement as `OpenSpace`.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }

    fn apply(&self, pos: &mut Vec3, vel: &mut Vec3, dt: f64) {
        let r = pos.len();
        if r <= 0.0 {
            return; // no defined inward direction at the exact centre
        }
        let normal = *pos / r;
        let gap = (self.radius - r).max(MIN_GAP);
        let accel = self.push_strength / gap;
        *vel -= normal * accel * dt;
    }

    /// G5 (roadmap.md §12): the same geometric boundary `Sphere` queries — distance to the
    /// nominal `radius` surface (positive inside, negative once past it) and the inward unit
    /// normal — even though `apply()` itself never hard-clamps here. The *query* is still
    /// well-defined regardless of how (or whether) `apply()` enforces it; a `StepHook`/
    /// `FlockingMode` asking "how close am I to the wall" wants the same honest answer under
    /// either sphere variant. Same centre-of-domain fallback as `Sphere`: "inward" has no
    /// natural direction at the exact centre, so an arbitrary fixed axis is returned instead of
    /// a non-unit or undefined vector.
    fn boundary_distance(&self, pos: Vec3) -> Option<(f64, Vec3)> {
        let r = pos.len();
        let normal = if r > MIN_LEN {
            -(pos / r)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        Some((self.radius - r, normal))
    }

    fn name(&self) -> &'static str {
        "sphere_soft"
    }
}

/// Registers `SphereSoft` under the name `"sphere_soft"` (design/02_plugins.md §1). Reads
/// `sphere_radius` (shared with `murmur_sphere_domain` — the two never both fill the
/// single-occupant `Domain` socket at once) and `sphere_soft_push_strength` (default `5.0`,
/// deliberately not named `push_strength` since that key is already used by
/// `murmur_predator_fsm`'s `StepHook`, which — unlike another `Domain` plugin — genuinely can
/// be active at the same time as this one).
pub fn register(r: &mut Registry) {
    r.register_domain("sphere_soft", |p: &PluginParams| {
        Box::new(SphereSoft::new(
            p.get_or("sphere_radius", 50.0),
            p.get_or("sphere_soft_push_strength", 5.0),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforms_to_domain_contract() {
        murmur_conformance::domain(&SphereSoft::new(50.0, 5.0));
    }

    #[test]
    fn delta_is_b_minus_a() {
        let d = SphereSoft::new(50.0, 5.0);
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(d.delta(a, b), b - a);
    }

    #[test]
    fn far_from_the_boundary_the_push_is_negligible() {
        let d = SphereSoft::new(50.0, 5.0);
        let mut pos = Vec3::new(1.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!(
            vel.len() < 0.2,
            "expected a small push far from the wall, got {}",
            vel.len()
        );
    }

    #[test]
    fn near_the_boundary_the_push_is_much_stronger_and_points_inward() {
        let d = SphereSoft::new(50.0, 5.0);
        let mut far = Vec3::new(1.0, 0.0, 0.0);
        let mut far_vel = Vec3::ZERO;
        d.apply(&mut far, &mut far_vel, 1.0);

        let mut near = Vec3::new(49.9, 0.0, 0.0);
        let mut near_vel = Vec3::ZERO;
        d.apply(&mut near, &mut near_vel, 1.0);

        assert!(near_vel.len() > far_vel.len() * 10.0);
        assert!(
            near_vel.x < 0.0,
            "push must point inward (toward the centre)"
        );
    }

    #[test]
    fn never_hard_clamps_position_even_far_outside_the_radius() {
        let d = SphereSoft::new(10.0, 5.0);
        let mut pos = Vec3::new(100.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 100.0, "SphereSoft must never move position directly");
        assert!(vel.x < 0.0, "but must still apply an inward push");
    }

    #[test]
    fn stays_finite_exactly_at_the_radius_despite_the_1_over_gap_shape() {
        let d = SphereSoft::new(10.0, 5.0);
        let mut pos = Vec3::new(10.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!(
            vel.is_finite(),
            "must stay finite at the singularity gap=0, got {:?}",
            vel
        );
    }

    #[test]
    fn boundary_distance_is_positive_inside_and_points_inward() {
        let d = SphereSoft::new(10.0, 5.0);
        let (distance, normal) = d.boundary_distance(Vec3::new(6.0, 0.0, 0.0)).unwrap();
        assert!((distance - 4.0).abs() < 1e-9, "got {}", distance);
        assert_eq!(normal, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn boundary_distance_at_the_exact_centre_returns_a_valid_unit_normal() {
        let d = SphereSoft::new(10.0, 5.0);
        let (distance, normal) = d.boundary_distance(Vec3::ZERO).unwrap();
        assert!((distance - 10.0).abs() < 1e-9);
        assert!((normal.len() - 1.0).abs() < 1e-9, "got {:?}", normal);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let d = reg
            .resolve_domain("sphere_soft", &PluginParams::new())
            .unwrap();
        assert_eq!(d.name(), "sphere_soft");
    }

    #[test]
    fn registry_reads_parameter_overrides() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new()
            .with("sphere_radius", 5.0)
            .with("sphere_soft_push_strength", 1000.0);
        let d = reg.resolve_domain("sphere_soft", &params).unwrap();
        let mut pos = Vec3::new(4.9, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!(
            vel.x < -5.0,
            "a much larger push_strength must produce a much larger kick, got {}",
            vel.x
        );
    }
}
