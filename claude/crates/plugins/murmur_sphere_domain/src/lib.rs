//! `Sphere` — hard-projection-to-surface `Domain` plugin (design/01_core.md §3, roadmap.md
//! Phase 16). Ported from pymurmur's `plugins/boundary/strategies.py::SphereBoundary`
//! (design/02_plugins.md §5).
//!
//! A ball of radius `radius` centred on the origin. Any boid that ends up outside it is
//! projected straight back onto the surface (hard clamp, unlike `SphereSoft`'s asymptotic
//! push), and any outward-pointing component of its velocity is removed — the "inward velocity
//! correction" design/02_plugins.md §5 names, distinct from a reflecting bounce.

use murmur_core::{Domain, PluginParams, Registry, Vec3};

pub struct Sphere {
    radius: f64,
}

impl Sphere {
    pub fn new(radius: f64) -> Self {
        Sphere { radius }
    }
}

impl Domain for Sphere {
    /// Not periodic — same direct displacement as `OpenSpace`.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }

    fn apply(&self, pos: &mut Vec3, vel: &mut Vec3, _dt: f64) {
        let r = pos.len();
        if r <= self.radius || r <= 0.0 {
            return;
        }
        let normal = *pos / r;
        *pos = normal * self.radius;
        let radial = vel.dot(normal);
        if radial > 0.0 {
            *vel -= normal * radial;
        }
    }

    fn name(&self) -> &'static str {
        "sphere"
    }
}

/// Registers `Sphere` under the name `"sphere"` (design/02_plugins.md §1). Reads `sphere_radius`
/// (default `50.0`, matching `Torus`'s default scale) — a distinct key from the initializer's
/// own `radius` (`murmur_initializers`'s `SphereShell`/`SphereVolume`) since both can be active
/// at once (an `Initializer` and a `Domain` fill different sockets) and must be independently
/// tunable.
pub fn register(r: &mut Registry) {
    r.register_domain("sphere", |p: &PluginParams| {
        Box::new(Sphere::new(p.get_or("sphere_radius", 50.0)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforms_to_domain_contract() {
        murmur_conformance::domain(&Sphere::new(50.0));
    }

    #[test]
    fn delta_is_b_minus_a() {
        let d = Sphere::new(50.0);
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(d.delta(a, b), b - a);
    }

    #[test]
    fn position_well_inside_the_ball_is_untouched() {
        let d = Sphere::new(50.0);
        let mut pos = Vec3::new(1.0, 2.0, 3.0);
        let mut vel = Vec3::new(1.0, -1.0, 0.5);
        let (pos0, vel0) = (pos, vel);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos, pos0);
        assert_eq!(vel, vel0);
    }

    #[test]
    fn position_beyond_the_radius_is_projected_onto_the_surface() {
        let d = Sphere::new(10.0);
        let mut pos = Vec3::new(20.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!((pos.len() - 10.0).abs() < 1e-9, "got len {}", pos.len());
        assert!((pos.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn outward_velocity_component_is_removed_at_the_surface() {
        let d = Sphere::new(10.0);
        let mut pos = Vec3::new(15.0, 0.0, 0.0);
        let mut vel = Vec3::new(3.0, 2.0, 0.0); // outward (+x) plus tangential (+y)
        d.apply(&mut pos, &mut vel, 1.0);
        assert!(
            vel.x <= 1e-9,
            "outward radial component must be removed, got {}",
            vel.x
        );
        assert_eq!(vel.y, 2.0, "tangential component must be preserved");
    }

    #[test]
    fn inward_velocity_at_the_surface_is_left_alone() {
        let d = Sphere::new(10.0);
        let mut pos = Vec3::new(15.0, 0.0, 0.0);
        let mut vel = Vec3::new(-4.0, 0.0, 0.0); // already heading inward
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(vel.x, -4.0);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let d = reg.resolve_domain("sphere", &PluginParams::new()).unwrap();
        assert_eq!(d.name(), "sphere");
    }

    #[test]
    fn registry_reads_sphere_radius_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new().with("sphere_radius", 5.0);
        let d = reg.resolve_domain("sphere", &params).unwrap();
        let mut pos = Vec3::new(50.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!((pos.len() - 5.0).abs() < 1e-9);
    }
}
