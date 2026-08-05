//! `Torus` — periodic-box `Domain` plugin (design/01_core.md §3, roadmap.md Phase 13). A cubic
//! box of side `2 * half_extent` centred on the origin, wrapping on all three axes.
//!
//! Second `Domain` occupant (after `OpenSpace`), proving the seam is real — but also directly
//! useful on its own terms: Phase 8's Vicsek order-disorder test needed an artificially dense
//! initial pack to keep the neighbour graph connected long enough to reach *global* consensus,
//! because open space has no cohesion term and no boundary to keep a sparse flock's neighbour
//! graph from fragmenting. A periodic domain removes that need — the literature's own Vicsek
//! setup (Vicsek et al. 1995) is a periodic box for exactly this reason, not open space.

use murmur_core::{Domain, PluginParams, Registry, Vec3};

pub struct Torus {
    half_extent: f64,
}

impl Torus {
    pub fn new(half_extent: f64) -> Self {
        Torus { half_extent }
    }

    fn wrap_component(&self, v: f64) -> f64 {
        let full = 2.0 * self.half_extent;
        // rem_euclid keeps this correct (no drift, no while-loop cost) for any magnitude of
        // `v`, not just values already near the box — a boid that somehow ends up far outside
        // (e.g. a future StepHook bug) still wraps back into range in one step, not silently
        // stays lost.
        (v + self.half_extent).rem_euclid(full) - self.half_extent
    }
}

impl Domain for Torus {
    /// Minimum-image convention: the shortest of the direct and wrapped-around displacement on
    /// each axis independently.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        let full = 2.0 * self.half_extent;
        let raw = b - a;
        Vec3::new(
            raw.x - full * (raw.x / full).round(),
            raw.y - full * (raw.y / full).round(),
            raw.z - full * (raw.z / full).round(),
        )
    }

    fn apply(&self, pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {
        // Periodic wrap — velocity is untouched (unlike a reflecting/clamping boundary, a
        // torus never changes heading, only position).
        pos.x = self.wrap_component(pos.x);
        pos.y = self.wrap_component(pos.y);
        pos.z = self.wrap_component(pos.z);
    }

    fn name(&self) -> &'static str {
        "torus"
    }
}

/// Registers `Torus` under the name `"torus"` (design/02_plugins.md §1). `half_extent`
/// defaults to `50.0` — large enough to comfortably hold a few hundred boids at the slice's
/// usual densities without every-step wrapping dominating behaviour, small enough that the
/// periodic boundary is actually reachable/exercised over a few hundred steps.
pub fn register(r: &mut Registry) {
    r.register_domain("torus", |p: &PluginParams| {
        Box::new(Torus::new(p.get_or("half_extent", 50.0)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforms_to_domain_contract() {
        murmur_conformance::domain(&Torus::new(50.0));
    }

    #[test]
    fn delta_is_direct_difference_when_well_within_the_box() {
        let d = Torus::new(50.0);
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(d.delta(a, b), b - a);
    }

    #[test]
    fn delta_wraps_around_when_the_direct_path_is_the_long_way() {
        // A 20-wide box (half_extent=10): points near opposite edges are actually close via
        // the wrap, not far via the direct difference.
        let d = Torus::new(10.0);
        let a = Vec3::new(-9.0, 0.0, 0.0);
        let b = Vec3::new(9.0, 0.0, 0.0);
        let delta = d.delta(a, b);
        // Direct b-a = 18 (the "long way"); the wrapped shortest path is -2 (going the other
        // way around: -9 -> -10/10 -> 9).
        assert!(
            (delta.x + 2.0).abs() < 1e-9,
            "expected minimum-image delta.x ~= -2.0, got {}",
            delta.x
        );
        assert!(
            delta.len() < (b - a).len(),
            "wrapped delta must be shorter than the raw one"
        );
    }

    #[test]
    fn delta_between_a_point_and_itself_is_exactly_zero() {
        let d = Torus::new(10.0);
        let a = Vec3::new(3.0, -4.0, 5.0);
        assert_eq!(d.delta(a, a), Vec3::ZERO);
    }

    #[test]
    fn apply_wraps_a_position_just_outside_the_box_back_in() {
        let d = Torus::new(10.0);
        let mut pos = Vec3::new(11.0, 0.0, -12.0);
        let mut vel = Vec3::new(1.0, 2.0, 3.0);
        let vel0 = vel;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!((pos.x - (-9.0)).abs() < 1e-9, "got {}", pos.x);
        assert!((pos.z - 8.0).abs() < 1e-9, "got {}", pos.z);
        assert_eq!(vel, vel0, "a periodic wrap must not touch velocity");
    }

    #[test]
    fn apply_wraps_a_position_arbitrarily_far_outside_the_box() {
        let d = Torus::new(10.0);
        let mut pos = Vec3::new(1005.0, 0.0, 0.0); // 50 box-widths away
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!(pos.x.is_finite());
        assert!((-10.0..10.0).contains(&pos.x), "got {}", pos.x);
    }

    #[test]
    fn apply_leaves_an_in_bounds_position_unchanged() {
        let d = Torus::new(10.0);
        let mut pos = Vec3::new(3.0, -4.0, 5.0);
        let mut vel = Vec3::ZERO;
        let pos0 = pos;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!((pos - pos0).len() < 1e-9);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let d = reg.resolve_domain("torus", &PluginParams::new()).unwrap();
        assert_eq!(d.name(), "torus");
    }

    #[test]
    fn registry_reads_half_extent_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new().with("half_extent", 5.0);
        let d = reg.resolve_domain("torus", &params).unwrap();
        let mut pos = Vec3::new(6.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert!((-5.0..5.0).contains(&pos.x), "got {}", pos.x);
    }

    /// Proves the `Domain` seam now has ≥2 real occupants (roadmap.md Phase 13 exit gate,
    /// mirroring Phase 3's pattern for the seam's first proof) — `murmur_open_domain` as a
    /// dev-dependency, same pattern already used elsewhere for cross-plugin integration tests
    /// (e.g. `murmur_pearce`'s dev-dependencies).
    #[test]
    fn open_and_torus_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_open_domain::register(&mut reg);

        let torus = reg.resolve_domain("torus", &PluginParams::new()).unwrap();
        let open = reg.resolve_domain("open", &PluginParams::new()).unwrap();
        assert_eq!(torus.name(), "torus");
        assert_eq!(open.name(), "open");
    }
}
