//! `Margin` — soft-nudge + hard-clamp cube `Domain` plugin (design/01_core.md §3, roadmap.md
//! Phase 16). Ported from pymurmur's `plugins/boundary/strategies.py::MarginBoundary`
//! (design/02_plugins.md §5's provenance table — corrected there from an earlier, wrong
//! attribution to droidmur, which has its own from-scratch `margin` rather than being the port
//! source).
//!
//! An axis-aligned cube of half-width `half_extent`, same shape family as `Torus`. Two zones
//! per axis: a soft margin band starting `margin_width` inside the edge, where velocity gets a
//! gradually-strengthening inward nudge (proportional to how deep into the band a boid has
//! gone), and a hard boundary at the edge itself, where position is clamped and any outward
//! velocity component on that axis is zeroed (absorbed, not bounced — pymurmur's
//! `MarginBoundary` is a wall, not a reflector).

use murmur_core::{Domain, PluginParams, Registry, Vec3};

pub struct Margin {
    half_extent: f64,
    margin_width: f64,
    margin_strength: f64,
}

impl Margin {
    pub fn new(half_extent: f64, margin_width: f64, margin_strength: f64) -> Self {
        Margin {
            half_extent,
            margin_width,
            margin_strength,
        }
    }

    /// One axis of the soft nudge + hard clamp. `margin_width <= 0` degenerates to a pure hard
    /// clamp (no soft zone) rather than panicking or dividing by zero.
    fn apply_axis(&self, p: &mut f64, v: &mut f64, dt: f64) {
        let margin_start = self.half_extent - self.margin_width;
        let depth = p.abs() - margin_start;
        // Only the soft zone (margin_start..=half_extent) gets the nudge; beyond half_extent
        // is the hard-clamp regime below, handled separately so the two don't double-apply to
        // the same boid on the same step.
        if depth > 0.0 && depth <= self.margin_width && self.margin_width > 0.0 {
            let frac = depth / self.margin_width;
            let inward = -p.signum();
            *v += inward * self.margin_strength * frac * dt;
        }
        if *p > self.half_extent {
            *p = self.half_extent;
            if *v > 0.0 {
                *v = 0.0;
            }
        } else if *p < -self.half_extent {
            *p = -self.half_extent;
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
}

impl Domain for Margin {
    /// Not periodic — same direct displacement as `OpenSpace`, since there's no wrap-around
    /// shortcut to take inside a bounded-but-not-toroidal cube.
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }

    fn apply(&self, pos: &mut Vec3, vel: &mut Vec3, dt: f64) {
        self.apply_axis(&mut pos.x, &mut vel.x, dt);
        self.apply_axis(&mut pos.y, &mut vel.y, dt);
        self.apply_axis(&mut pos.z, &mut vel.z, dt);
    }

    /// G5 (roadmap.md §12): the nearest of the cube's 6 faces — whichever axis has the smallest
    /// remaining margin (`half_extent - |pos[axis]|`) — with an inward unit normal on that axis
    /// alone (`f64::signum` is `±1.0` even at exactly `0.0`, never `0.0` itself, so this is
    /// always genuinely unit length, never the zero vector).
    fn boundary_distance(&self, pos: Vec3) -> Option<(f64, Vec3)> {
        let margins = [
            self.half_extent - pos.x.abs(),
            self.half_extent - pos.y.abs(),
            self.half_extent - pos.z.abs(),
        ];
        let axis = if margins[0] <= margins[1] && margins[0] <= margins[2] {
            0
        } else if margins[1] <= margins[2] {
            1
        } else {
            2
        };
        let normal = match axis {
            0 => Vec3::new(-pos.x.signum(), 0.0, 0.0),
            1 => Vec3::new(0.0, -pos.y.signum(), 0.0),
            _ => Vec3::new(0.0, 0.0, -pos.z.signum()),
        };
        Some((margins[axis], normal))
    }

    fn name(&self) -> &'static str {
        "margin"
    }
}

/// Registers `Margin` under the name `"margin"` (design/02_plugins.md §1). Reuses `Torus`'s
/// `half_extent` key (same box-half-width semantics, and the two plugins never both fill the
/// single-occupant `Domain` socket at once, so there's no collision) plus two new keys,
/// `margin_width` (default `10.0`) and `margin_strength` (default `5.0`).
pub fn register(r: &mut Registry) {
    r.register_domain("margin", |p: &PluginParams| {
        Box::new(Margin::new(
            p.get_or("half_extent", 50.0),
            p.get_or("margin_width", 10.0),
            p.get_or("margin_strength", 5.0),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforms_to_domain_contract() {
        murmur_conformance::domain(&Margin::new(50.0, 10.0, 5.0));
    }

    #[test]
    fn delta_is_b_minus_a() {
        let d = Margin::new(50.0, 10.0, 5.0);
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(d.delta(a, b), b - a);
    }

    #[test]
    fn deep_interior_position_is_untouched() {
        let d = Margin::new(50.0, 10.0, 5.0);
        let mut pos = Vec3::new(1.0, -2.0, 3.0);
        let mut vel = Vec3::new(1.0, 0.0, 0.0);
        let (pos0, vel0) = (pos, vel);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos, pos0);
        assert_eq!(vel, vel0);
    }

    #[test]
    fn inside_the_margin_band_gets_an_inward_velocity_nudge_but_no_position_change() {
        // half_extent=50, margin_width=10 -> band starts at |p|=40. x=45 is 5 deep into it.
        let d = Margin::new(50.0, 10.0, 5.0);
        let mut pos = Vec3::new(45.0, 0.0, 0.0);
        let mut vel = Vec3::new(1.0, 0.0, 0.0);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 45.0, "the soft zone must not move position");
        assert!(
            vel.x < 1.0,
            "an inward (negative-x) nudge must reduce vel.x, got {}",
            vel.x
        );
    }

    #[test]
    fn beyond_the_edge_position_is_hard_clamped_and_outward_velocity_zeroed() {
        let d = Margin::new(50.0, 10.0, 5.0);
        let mut pos = Vec3::new(60.0, 0.0, 0.0);
        let mut vel = Vec3::new(3.0, 0.0, 0.0);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 50.0);
        assert_eq!(
            vel.x, 0.0,
            "outward velocity at a hard wall must be absorbed, not bounced"
        );
    }

    #[test]
    fn beyond_the_edge_moving_inward_keeps_its_inward_velocity() {
        // A boid already past the edge but heading back in should not have that motion erased.
        let d = Margin::new(50.0, 10.0, 5.0);
        let mut pos = Vec3::new(60.0, 0.0, 0.0);
        let mut vel = Vec3::new(-2.0, 0.0, 0.0);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 50.0);
        assert_eq!(vel.x, -2.0);
    }

    #[test]
    fn each_axis_is_independent() {
        let d = Margin::new(50.0, 10.0, 5.0);
        let mut pos = Vec3::new(55.0, 1.0, -60.0);
        let mut vel = Vec3::new(1.0, 1.0, -1.0);
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 50.0);
        assert_eq!(pos.y, 1.0, "y was well within bounds and must be untouched");
        assert_eq!(pos.z, -50.0);
    }

    #[test]
    fn boundary_distance_finds_the_nearest_face_and_points_inward() {
        let d = Margin::new(50.0, 10.0, 5.0);
        // Well inside on y/z, close to the +x face (margin 5.0) and further from +y (margin 45).
        let (distance, normal) = d.boundary_distance(Vec3::new(45.0, 0.0, 0.0)).unwrap();
        assert!((distance - 5.0).abs() < 1e-9, "got {}", distance);
        assert_eq!(
            normal,
            Vec3::new(-1.0, 0.0, 0.0),
            "must point inward, away from +x"
        );
    }

    #[test]
    fn boundary_distance_is_negative_once_past_the_edge() {
        let d = Margin::new(50.0, 10.0, 5.0);
        let (distance, _) = d.boundary_distance(Vec3::new(60.0, 0.0, 0.0)).unwrap();
        assert!(distance < 0.0, "got {}", distance);
    }

    #[test]
    fn boundary_distance_at_the_exact_centre_returns_a_valid_unit_normal() {
        // pos=(0,0,0): all three margins tie at half_extent -- must not produce a zero vector.
        let d = Margin::new(50.0, 10.0, 5.0);
        let (distance, normal) = d.boundary_distance(Vec3::ZERO).unwrap();
        assert!((distance - 50.0).abs() < 1e-9);
        assert!((normal.len() - 1.0).abs() < 1e-9, "got {:?}", normal);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let d = reg.resolve_domain("margin", &PluginParams::new()).unwrap();
        assert_eq!(d.name(), "margin");
    }

    #[test]
    fn registry_reads_parameter_overrides() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new()
            .with("half_extent", 5.0)
            .with("margin_width", 1.0)
            .with("margin_strength", 100.0);
        let d = reg.resolve_domain("margin", &params).unwrap();
        let mut pos = Vec3::new(6.0, 0.0, 0.0);
        let mut vel = Vec3::ZERO;
        d.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos.x, 5.0);
    }
}
