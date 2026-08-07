//! `Obstacle` — a full SDF+CSG obstacle-avoidance `StepHook` (design/02_plugins.md §5,
//! roadmap.md Phase 18, pymurmur's `physics/obstacles.py`, ported by description — pymurmur's
//! actual source isn't reachable in this environment, the same blocker as every other pymurmur
//! cross-check this project has hit). The design doc's own target: "Full SDF+CSG: sphere/box/
//! cylinder primitives, `union`/`subtract`, numerical gradient, collision detection, kinematic
//! surface correction. **droidmur implements spheres only**, a deliberate v1 scope-cut
//! documented in its own source — useful as a minimal-viable reference, but pymurmur's full
//! system is the actual port target." The last plugin in Phase 18's own catalogue.
//!
//! **Checked both architectural questions this row's own earlier entries flagged, rather than
//! assuming either way**:
//! - **G1's own "parallel-seam half"** (roadmap.md §12: `write_phase`'s `post_steer` loop is
//!   sequential, not rayon-parallel — a *separate* half of G1's original candidate direction
//!   from the `ctx.neighbors` fix `murmur_boid_state_machine` already triggered) — **not
//!   needed**. Evaluating an `ObstacleScene` (a handful of primitives) per boid per step is
//!   `O(N × primitive_count)`, cheaper than the `O(N²)` brute-force sweeps `murmur_predator_fsm`/
//!   `murmur_dynamic_vision_range` already accept as fine at this project's own slice scale.
//! - **G2** (no post-integration position-correction seam) — **not needed either**, and for a
//!   more fundamental reason than "not yet built": G2 is scoped specifically to *pairwise*
//!   corrections needing *other boids'* post-integration state (roadmap.md §12's own wording:
//!   "same-species symmetric, predator–prey asymmetric"). Obstacle avoidance needs no other
//!   boid's state at all — just a boid's own position against a static scene, exactly the shape
//!   `Domain::apply` already handles for domain boundaries. `StepHook` itself has no
//!   post-integration seam of *any* kind (only `pre_step`, before the read phase, and
//!   `post_steer`, contributing to `acc` before integration) — building one purely for a
//!   per-boid-only case would be new, unneeded architecture when a soft, force-based approach
//!   (below) already does the job, the same choice this project already made for
//!   `murmur_sphere_soft_domain` over a hard-clamping `Sphere`.
//!
//! **"Kinematic surface correction," built as a soft avoidance force, not a hard positional
//! clamp** — reusing `murmur_sphere_soft_domain`'s own inverse-distance push formula
//! (`accel = push_strength / gap`, `gap` floored at `min_gap` to avoid a divide-by-zero
//! blow-up), applied as an additive `post_steer` acceleration along the scene's own outward
//! gradient rather than as a direct position/velocity mutation. A fast-enough boid can still
//! nudge slightly into a solid for one step before the growing push force corrects it — the same
//! honestly-disclosed trade-off `SphereSoft` already accepted over `Sphere`'s hard clamp.
//!
//! **SDF primitives and CSG are standard, well-known computational-geometry techniques** (Inigo
//! Quilez's widely-cited "distance functions" reference, `iquilezles.org` — generic, verifiable
//! geometry, not derived from pymurmur's own unreachable source, and not original to this
//! project either): `Primitive::Sphere`/`Box`/`Cylinder` each implement a closed-form signed
//! distance function (negative inside, positive outside, zero on the surface); `Solid` combines
//! one base primitive with an optional `cut` primitive via the standard CSG subtraction formula
//! `max(base, -cut)`; `ObstacleScene` unions any number of `Solid`s via `min`. The outward push
//! direction is `ObstacleScene::gradient`, a **numerical** (central finite-difference) gradient
//! of the combined scene SDF — the design doc's own literal wording, not an analytical
//! shortcut, even though these particular primitives do have simple closed-form gradients.
//!
//! **Collision detection, published as a real accessor**: `pub fn is_colliding(&self, index:
//! u32) -> Option<bool>` (`sdf < 0.0`, cached in `post_steer`, the same pattern
//! `murmur_ripple`'s `envelope_sum_of`/`murmur_wander`'s `wander_center` already established).
//!
//! **Scope, disclosed rather than hidden**: the *engine* (`Primitive`/`Solid`/`ObstacleScene`,
//! all `pub`, all independently constructible and tested) is the full sphere/box/cylinder +
//! union/subtract system the design doc names as pymurmur's own actual target — reachable by
//! any Rust caller. The `register()` factory that builds a `Simulation`-composable `Obstacle`
//! from `PluginParams`' flat key–value blob (design/02_plugins.md §1 — no nested structures)
//! exposes exactly **one** obstacle at a time, of a caller-selected primitive kind, with no
//! `union`/`subtract` — the same "spheres only, a deliberate v1 scope-cut" precedent the design
//! doc itself already names for `droidmur`, just widened to "any one primitive kind" rather than
//! sphere-only. A caller building a `Simulation` directly in Rust (not through the flat
//! `PluginParams` interface) can compose an arbitrarily rich `ObstacleScene` of its own.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{BoidCtx, ConfigError, PluginParams, Registry, Rng, StepHook, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Box {
        center: Vec3,
        half_extent: Vec3,
    },
    Cylinder {
        center: Vec3,
        axis: Vec3,
        radius: f64,
        half_height: f64,
    },
}

impl Primitive {
    /// The signed distance from `p` to this primitive's own surface — negative inside, positive
    /// outside, zero exactly on the surface.
    pub fn sdf(&self, p: Vec3) -> f64 {
        match *self {
            Primitive::Sphere { center, radius } => (p - center).len() - radius,
            Primitive::Box {
                center,
                half_extent,
            } => {
                let d = p - center;
                let qx = d.x.abs() - half_extent.x;
                let qy = d.y.abs() - half_extent.y;
                let qz = d.z.abs() - half_extent.z;
                let outside = Vec3::new(qx.max(0.0), qy.max(0.0), qz.max(0.0)).len();
                let inside = qx.max(qy).max(qz).min(0.0);
                outside + inside
            }
            Primitive::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                let a = axis.normalized();
                let d = p - center;
                let h = d.dot(a);
                let perp = d - a * h;
                (perp.len() - radius).max(h.abs() - half_height)
            }
        }
    }
}

/// A `base` primitive, optionally with a `cut` primitive subtracted out of it — the standard CSG
/// subtraction formula, `max(base_sdf, -cut_sdf)`. Two-level CSG (per-solid subtract, whole-scene
/// union in `ObstacleScene`), a deliberate, disclosed scope choice rather than a fully general
/// nested boolean tree — matching design/02_plugins.md's own named operations, "union/subtract",
/// literally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solid {
    pub base: Primitive,
    pub cut: Option<Primitive>,
}

impl Solid {
    pub fn new(base: Primitive) -> Self {
        Solid { base, cut: None }
    }

    pub fn subtract(mut self, cut: Primitive) -> Self {
        self.cut = Some(cut);
        self
    }

    pub fn sdf(&self, p: Vec3) -> f64 {
        let base = self.base.sdf(p);
        match self.cut {
            Some(cut) => base.max(-cut.sdf(p)),
            None => base,
        }
    }
}

/// The union (via `min`) of any number of `Solid`s — an empty scene's own `sdf` is `+inf`
/// (never triggers avoidance; `gradient` guards against differentiating that explicitly, since
/// `inf - inf` would otherwise be `NaN`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObstacleScene {
    pub solids: Vec<Solid>,
}

impl ObstacleScene {
    pub fn new() -> Self {
        ObstacleScene { solids: Vec::new() }
    }

    pub fn with_solid(mut self, solid: Solid) -> Self {
        self.solids.push(solid);
        self
    }

    pub fn sdf(&self, p: Vec3) -> f64 {
        self.solids
            .iter()
            .map(|s| s.sdf(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// The scene's own outward-pointing unit gradient at `p`, via central finite differences
    /// with step `epsilon` — a numerical gradient, the design doc's own literal wording (not an
    /// analytical shortcut, even though these primitives do have simple closed-form gradients).
    /// `Vec3::ZERO` for an empty scene (no solids to push away from) or a degenerate
    /// (near-zero-gradient) point, matching `Vec3::normalized()`'s own never-NaN contract.
    pub fn gradient(&self, p: Vec3, epsilon: f64) -> Vec3 {
        if self.solids.is_empty() {
            return Vec3::ZERO;
        }
        let dx = Vec3::new(epsilon, 0.0, 0.0);
        let dy = Vec3::new(0.0, epsilon, 0.0);
        let dz = Vec3::new(0.0, 0.0, epsilon);
        let gx = (self.sdf(p + dx) - self.sdf(p - dx)) / (2.0 * epsilon);
        let gy = (self.sdf(p + dy) - self.sdf(p - dy)) / (2.0 * epsilon);
        let gz = (self.sdf(p + dz) - self.sdf(p - dz)) / (2.0 * epsilon);
        Vec3::new(gx, gy, gz).normalized()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObstacleParams {
    pub avoidance_radius: f64,
    pub push_strength: f64,
    pub min_gap: f64,
    pub gradient_epsilon: f64,
}

impl ObstacleParams {
    pub fn builder() -> ObstacleParamsBuilder {
        ObstacleParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ObstacleParamsBuilder {
    avoidance_radius: f64,
    push_strength: f64,
    min_gap: f64,
    gradient_epsilon: f64,
}

impl Default for ObstacleParamsBuilder {
    fn default() -> Self {
        ObstacleParamsBuilder {
            avoidance_radius: 5.0,
            push_strength: 2.0,
            min_gap: 0.1,
            gradient_epsilon: 1e-3,
        }
    }
}

impl ObstacleParamsBuilder {
    pub fn avoidance_radius(mut self, v: f64) -> Self {
        self.avoidance_radius = v;
        self
    }
    pub fn push_strength(mut self, v: f64) -> Self {
        self.push_strength = v;
        self
    }
    pub fn min_gap(mut self, v: f64) -> Self {
        self.min_gap = v;
        self
    }
    pub fn gradient_epsilon(mut self, v: f64) -> Self {
        self.gradient_epsilon = v;
        self
    }

    pub fn build(self) -> Result<ObstacleParams, ConfigError> {
        if !(self.avoidance_radius.is_finite() && self.avoidance_radius > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "avoidance_radius",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.push_strength.is_finite() && self.push_strength >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "push_strength",
                reason: "must be finite and >= 0".into(),
            });
        }
        if !(self.min_gap.is_finite() && self.min_gap > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "min_gap",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.gradient_epsilon.is_finite() && self.gradient_epsilon > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "gradient_epsilon",
                reason: "must be finite and > 0".into(),
            });
        }
        Ok(ObstacleParams {
            avoidance_radius: self.avoidance_radius,
            push_strength: self.push_strength,
            min_gap: self.min_gap,
            gradient_epsilon: self.gradient_epsilon,
        })
    }
}

pub struct Obstacle {
    pub params: ObstacleParams,
    pub scene: ObstacleScene,
    colliding: Mutex<HashMap<u32, bool>>,
}

impl Obstacle {
    pub fn new(params: ObstacleParams, scene: ObstacleScene) -> Self {
        Obstacle {
            params,
            scene,
            colliding: Mutex::new(HashMap::new()),
        }
    }

    /// Whether this boid's own position was inside the obstacle scene (`sdf < 0`) the last time
    /// `post_steer` saw it — collision detection, published as a real accessor, not part of the
    /// `StepHook` trait itself (no other occupant needs it).
    pub fn is_colliding(&self, index: u32) -> Option<bool> {
        self.colliding.lock().unwrap().get(&index).copied()
    }
}

impl StepHook for Obstacle {
    fn post_steer(&self, ctx: BoidCtx<'_>, acc: &mut Vec3, _rng: &mut Rng) {
        let d = self.scene.sdf(ctx.pos);
        self.colliding.lock().unwrap().insert(ctx.index, d < 0.0);

        if d < self.params.avoidance_radius {
            let direction = self.scene.gradient(ctx.pos, self.params.gradient_epsilon);
            let denom = d.max(self.params.min_gap);
            let magnitude = self.params.push_strength / denom;
            *acc += direction * magnitude;
        }
    }

    fn name(&self) -> &'static str {
        "obstacles"
    }
}

/// Builds one `Primitive` from `PluginParams`' flat key–value blob, selected by
/// `obstacle_kind` (`0.0` = sphere, `1.0` = box, `2.0` = cylinder, rounded to the nearest
/// integer; anything else falls back to sphere).
fn primitive_from_params(p: &PluginParams) -> Primitive {
    let center = Vec3::new(
        p.get_or("obstacle_center_x", 0.0),
        p.get_or("obstacle_center_y", 0.0),
        p.get_or("obstacle_center_z", 0.0),
    );
    let kind = p.get_or("obstacle_kind", 0.0).round() as i64;
    match kind {
        1 => Primitive::Box {
            center,
            half_extent: Vec3::new(
                p.get_or("obstacle_half_extent_x", 5.0),
                p.get_or("obstacle_half_extent_y", 5.0),
                p.get_or("obstacle_half_extent_z", 5.0),
            ),
        },
        2 => Primitive::Cylinder {
            center,
            axis: Vec3::new(
                p.get_or("obstacle_axis_x", 0.0),
                p.get_or("obstacle_axis_y", 0.0),
                p.get_or("obstacle_axis_z", 1.0),
            ),
            radius: p.get_or("obstacle_radius", 5.0),
            half_height: p.get_or("obstacle_half_height", 5.0),
        },
        _ => Primitive::Sphere {
            center,
            radius: p.get_or("obstacle_radius", 5.0),
        },
    }
}

/// Registers `Obstacle` under the name `"obstacles"` — a single primitive (sphere, box, or
/// cylinder, selected by `obstacle_kind`), no `union`/`subtract` (see the module doc's own scope
/// note). A malformed `ObstacleParams` override falls back to the default rather than panicking
/// — the factory type can't return `Result` (design/02_plugins.md §1), same pattern as every
/// other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("obstacles", |p: &PluginParams| {
        let d = ObstacleParamsBuilder::default();
        let params = ObstacleParams::builder()
            .avoidance_radius(p.get_or("obstacle_avoidance_radius", d.avoidance_radius))
            .push_strength(p.get_or("obstacle_push_strength", d.push_strength))
            .min_gap(p.get_or("obstacle_min_gap", d.min_gap))
            .gradient_epsilon(p.get_or("obstacle_gradient_epsilon", d.gradient_epsilon))
            .build()
            .unwrap_or_else(|_| {
                ObstacleParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        let scene = ObstacleScene::new().with_solid(Solid::new(primitive_from_params(p)));
        Box::new(Obstacle::new(params, scene)) as Box<dyn StepHook>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{rng, CoreParams, Domain, Species};

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

    fn core_params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn ctx<'a>(
        index: u32,
        pos: Vec3,
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos,
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &[],
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let scene = ObstacleScene::new().with_solid(Solid::new(Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 3.0,
        }));
        let mut hook = Obstacle::new(ObstacleParams::builder().build().unwrap(), scene);
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn sphere_sdf_matches_the_closed_form() {
        let s = Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        assert!((s.sdf(Vec3::new(10.0, 0.0, 0.0)) - 5.0).abs() < 1e-9);
        assert!((s.sdf(Vec3::ZERO) - (-5.0)).abs() < 1e-9);
        assert!(s.sdf(Vec3::new(5.0, 0.0, 0.0)).abs() < 1e-9);
    }

    #[test]
    fn box_sdf_is_zero_exactly_on_a_face_negative_inside_positive_outside() {
        let b = Primitive::Box {
            center: Vec3::ZERO,
            half_extent: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(
            b.sdf(Vec3::new(1.0, 0.0, 0.0)).abs() < 1e-9,
            "on the +x face"
        );
        assert!(b.sdf(Vec3::ZERO) < 0.0, "centre must be inside");
        assert!(b.sdf(Vec3::new(2.0, 0.0, 0.0)) > 0.0, "clearly outside");
        assert!((b.sdf(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cylinder_sdf_matches_the_closed_form() {
        let c = Primitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::new(0.0, 0.0, 1.0),
            radius: 2.0,
            half_height: 3.0,
        };
        assert!(
            (c.sdf(Vec3::ZERO) - (-2.0)).abs() < 1e-9,
            "on-axis, well within height"
        );
        assert!(
            (c.sdf(Vec3::new(5.0, 0.0, 0.0)) - 3.0).abs() < 1e-9,
            "radially outside, within height -> radial excess only"
        );
        assert!(
            (c.sdf(Vec3::new(0.0, 0.0, 10.0)) - 7.0).abs() < 1e-9,
            "on-axis but far past the cap -> axial excess only"
        );
    }

    #[test]
    fn union_of_two_spheres_is_the_nearer_ones_distance() {
        let scene = ObstacleScene::new()
            .with_solid(Solid::new(Primitive::Sphere {
                center: Vec3::new(-100.0, 0.0, 0.0),
                radius: 1.0,
            }))
            .with_solid(Solid::new(Primitive::Sphere {
                center: Vec3::new(10.0, 0.0, 0.0),
                radius: 1.0,
            }));
        let d = scene.sdf(Vec3::ZERO);
        // Nearer to the sphere at (10,0,0): distance to its surface is 10 - 1 = 9.
        assert!((d - 9.0).abs() < 1e-9, "got {}", d);
    }

    #[test]
    fn subtract_removes_a_bite_from_a_solid() {
        let base = Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        let cut = Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 3.0,
        };
        let solid = Solid::new(base).subtract(cut);
        // A point at radius 2 (inside both base and cut) must now be OUTSIDE the resulting
        // shell -- the cut hollowed it out.
        let p = Vec3::new(2.0, 0.0, 0.0);
        assert!(base.sdf(p) < 0.0, "sanity: inside the undrilled base");
        assert!(
            solid.sdf(p) > 0.0,
            "must be outside once the cut hollows it out: got {}",
            solid.sdf(p)
        );
        // A point at radius 4 (inside base, outside cut) is still part of the shell.
        let q = Vec3::new(4.0, 0.0, 0.0);
        assert!(
            solid.sdf(q) < 0.0,
            "the shell itself must remain solid: got {}",
            solid.sdf(q)
        );
    }

    #[test]
    fn gradient_points_away_from_a_spheres_surface() {
        let scene = ObstacleScene::new().with_solid(Solid::new(Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }));
        let g = scene.gradient(Vec3::new(10.0, 0.0, 0.0), 1e-3);
        assert!(g.x > 0.9, "expected a unit +x gradient, got {:?}", g);
        assert!(g.y.abs() < 1e-3 && g.z.abs() < 1e-3);
    }

    #[test]
    fn gradient_of_an_empty_scene_is_zero() {
        let scene = ObstacleScene::new();
        assert_eq!(scene.gradient(Vec3::new(1.0, 2.0, 3.0), 1e-3), Vec3::ZERO);
    }

    #[test]
    fn post_steer_pushes_away_from_a_nearby_sphere() {
        let scene = ObstacleScene::new().with_solid(Solid::new(Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }));
        let hook = Obstacle::new(
            ObstacleParams::builder()
                .avoidance_radius(10.0)
                .push_strength(1.0)
                .build()
                .unwrap(),
            scene,
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(0, Vec3::new(6.0, 0.0, 0.0), &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        assert!(acc.x > 0.0, "must push away from the sphere, got {:?}", acc);
    }

    #[test]
    fn post_steer_does_nothing_beyond_the_avoidance_radius() {
        let scene = ObstacleScene::new().with_solid(Solid::new(Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }));
        let hook = Obstacle::new(
            ObstacleParams::builder()
                .avoidance_radius(2.0)
                .build()
                .unwrap(),
            scene,
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(0, Vec3::new(1000.0, 0.0, 0.0), &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        assert_eq!(acc, Vec3::ZERO);
    }

    #[test]
    fn is_colliding_reflects_whether_the_boid_is_inside_the_solid() {
        let scene = ObstacleScene::new().with_solid(Solid::new(Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }));
        let hook = Obstacle::new(ObstacleParams::builder().build().unwrap(), scene);
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;

        assert_eq!(hook.is_colliding(0), None, "unseen boid must be None");

        hook.post_steer(
            ctx(0, Vec3::ZERO, &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        assert_eq!(
            hook.is_colliding(0),
            Some(true),
            "centre is inside the sphere"
        );

        hook.post_steer(
            ctx(0, Vec3::new(1000.0, 0.0, 0.0), &params, &domain),
            &mut acc,
            &mut rng::for_boid(1, 2, 3),
        );
        assert_eq!(hook.is_colliding(0), Some(false), "far away is outside");
    }

    #[test]
    fn builder_rejects_a_nonpositive_avoidance_radius() {
        assert!(ObstacleParams::builder()
            .avoidance_radius(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_negative_push_strength() {
        assert!(ObstacleParams::builder()
            .push_strength(-1.0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_nonpositive_min_gap() {
        assert!(ObstacleParams::builder().min_gap(0.0).build().is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("obstacles", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "obstacles");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("obstacle_min_gap", -5.0);
        let hook = reg.resolve_step_hook("obstacles", &bad).unwrap();
        assert_eq!(hook.name(), "obstacles");
    }

    #[test]
    fn register_reaches_all_three_primitive_kinds_via_the_flat_kind_selector() {
        assert!(matches!(
            primitive_from_params(&PluginParams::new().with("obstacle_kind", 0.0)),
            Primitive::Sphere { .. }
        ));
        assert!(matches!(
            primitive_from_params(&PluginParams::new().with("obstacle_kind", 1.0)),
            Primitive::Box { .. }
        ));
        assert!(matches!(
            primitive_from_params(&PluginParams::new().with("obstacle_kind", 2.0)),
            Primitive::Cylinder { .. }
        ));
    }
}
