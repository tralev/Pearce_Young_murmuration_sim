//! Default `Initializer`/`NoiseSource` implementations (design/01_core.md §12.5): `SphereShell`,
//! `SphereVolume`, `UniformBox`, `Gaussian`, `Grid`, `Blob`, `Tangential`, `SpawnCube`
//! (`Initializer`), and `UniformSphere` (`NoiseSource`). The traits are core; these concrete
//! impls are plugins, same status as any other socket's default (design/00_overview.md §2) —
//! bundled in one crate since each is only a few lines. The last five close out
//! design/02_plugins.md §5's `Initializer` gap (roadmap.md Phase 16): pymurmur's own catalogue
//! names 6 position + 7 velocity modes, ported here by description (pymurmur's actual source
//! isn't reachable in this environment, the same blocker as every other pymurmur cross-check
//! this project has hit) rather than by exact line-for-line translation.

use std::f64::consts::PI;

use murmur_core::{
    sample_unit_sphere, uniform01, CoreParams, Initializer, NoiseSource, PluginParams, Registry,
    Rng, Vec3,
};

/// Standard-normal sample via the Box-Muller transform. `u1` is floored away from exactly `0`
/// so `ln(u1)` never produces `-inf` (a real, if astronomically unlikely, edge case with a
/// uniform `[0,1)` source).
fn standard_normal(rng: &mut Rng) -> f64 {
    let u1 = uniform01(rng).max(1e-12);
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn gaussian_vec3(rng: &mut Rng, std_dev: f64) -> Vec3 {
    Vec3::new(
        standard_normal(rng) * std_dev,
        standard_normal(rng) * std_dev,
        standard_normal(rng) * std_dev,
    )
}

/// Uniform on a sphere surface of radius `radius`; velocities = random unit × cruise_speed.
pub struct SphereShell {
    pub radius: f64,
}
impl Initializer for SphereShell {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            pos.push(sample_unit_sphere(rng) * self.radius);
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "sphere_shell"
    }
}

/// Uniform in a sphere VOLUME: `dir · cbrt(u) · R` — the cube-root removes centre clustering.
pub struct SphereVolume {
    pub radius: f64,
}
impl Initializer for SphereVolume {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let dir = sample_unit_sphere(rng);
            let r = self.radius * uniform01(rng).cbrt();
            pos.push(dir * r);
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "sphere_volume"
    }
}

/// Uniform in an axis-aligned box (for open-space migration setups).
pub struct UniformBox {
    pub half_extent: Vec3,
}
impl Initializer for UniformBox {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let x = (uniform01(rng) * 2.0 - 1.0) * self.half_extent.x;
            let y = (uniform01(rng) * 2.0 - 1.0) * self.half_extent.y;
            let z = (uniform01(rng) * 2.0 - 1.0) * self.half_extent.z;
            pos.push(Vec3::new(x, y, z));
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "uniform_box"
    }
}

/// Isotropic 3D Gaussian cloud, standard deviation `std_dev` on each axis.
pub struct Gaussian {
    pub std_dev: f64,
}
impl Initializer for Gaussian {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            pos.push(gaussian_vec3(rng, self.std_dev));
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "gaussian"
    }
}

/// A deterministic cubic lattice, `grid_spacing` apart, centred on the origin — the one
/// position-deterministic `Initializer` in this catalogue, useful for regression fixtures where
/// a reproducible starting layout matters more than a physically representative one. Velocities
/// are still randomised (`sample_unit_sphere`), matching every other `Initializer` here. `n`
/// need not be a perfect cube: the lattice side is `ceil(cbrt(n))`, and only the first `n` cells
/// (row-major) are used — the remainder of that cube is simply left unfilled.
pub struct Grid {
    pub grid_spacing: f64,
}
impl Initializer for Grid {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        let side = ((n as f64).cbrt().ceil() as usize).max(1);
        let center = (side as f64 - 1.0) / 2.0;
        for i in 0..n as usize {
            let ix = i % side;
            let iy = (i / side) % side;
            let iz = i / (side * side);
            pos.push(Vec3::new(
                (ix as f64 - center) * self.grid_spacing,
                (iy as f64 - center) * self.grid_spacing,
                (iz as f64 - center) * self.grid_spacing,
            ));
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "grid"
    }
}

/// `blob_count` Gaussian sub-clusters (each `blob_radius` wide), their centres placed
/// volume-uniformly within `blob_spread` of the origin (the same `dir · cbrt(u) · R` law as
/// `SphereVolume`) — a multi-flock-fragment starting condition, distinct from `Gaussian`'s
/// single cloud. Boids are assigned to blobs round-robin (`i % blob_count`), not randomly, so
/// blob population stays balanced regardless of `n`. `blob_count == 0` would be a
/// divide-by-zero in that assignment — clamped to `1` instead (a single blob, degenerating
/// gracefully rather than panicking).
pub struct Blob {
    pub blob_count: u32,
    pub blob_spread: f64,
    pub blob_radius: f64,
}
impl Initializer for Blob {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let blob_count = self.blob_count.max(1);
        let centers: Vec<Vec3> = (0..blob_count)
            .map(|_| {
                let dir = sample_unit_sphere(rng);
                let r = self.blob_spread * uniform01(rng).cbrt();
                dir * r
            })
            .collect();

        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for i in 0..n {
            let center = centers[(i % blob_count) as usize];
            pos.push(center + gaussian_vec3(rng, self.blob_radius));
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "blob"
    }
}

/// Positions uniform on a sphere shell of radius `radius` (same law as `SphereShell`), but
/// velocities are constrained *tangential* to that sphere at each point — an initial "orbiting
/// shell" condition, rather than `SphereShell`'s isotropic-random departure directions. The
/// tangent direction is built from an explicit orthonormal basis of the tangent plane (rather
/// than rejection-sampling a random vector and projecting out its radial component), so it's
/// robust at every point on the sphere with no near-zero-vector renormalisation risk.
pub struct Tangential {
    pub radius: f64,
}
impl Initializer for Tangential {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let dir = sample_unit_sphere(rng);
            pos.push(dir * self.radius);

            // An arbitrary vector guaranteed not (anti)parallel to `dir`, to build an
            // orthonormal tangent-plane basis from.
            let helper = if dir.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            let t1 = dir.cross(helper).normalized();
            let t2 = dir.cross(t1); // already unit length: dir, t1 orthonormal

            let theta = uniform01(rng) * 2.0 * PI;
            let tangent_dir = t1 * theta.cos() + t2 * theta.sin();
            vel.push(tangent_dir * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "tangential"
    }
}

/// A cube (equal half-extent on all three axes) — `UniformBox`'s single-parameter special case,
/// under pymurmur's own name for it.
pub struct SpawnCube {
    pub spawn_size: f64,
}
impl Initializer for SpawnCube {
    fn place(&self, n: u32, p: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos = Vec::with_capacity(n as usize);
        let mut vel = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let x = (uniform01(rng) * 2.0 - 1.0) * self.spawn_size;
            let y = (uniform01(rng) * 2.0 - 1.0) * self.spawn_size;
            let z = (uniform01(rng) * 2.0 - 1.0) * self.spawn_size;
            pos.push(Vec3::new(x, y, z));
            vel.push(sample_unit_sphere(rng) * p.cruise_speed);
        }
        (pos, vel)
    }
    fn name(&self) -> &'static str {
        "spawn_cube"
    }
}

/// Pearce uses `UniformSphere` (area-uniform on S²) for η̂.
pub struct UniformSphere;
impl NoiseSource for UniformSphere {
    fn sample(&self, rng: &mut Rng) -> Vec3 {
        sample_unit_sphere(rng)
    }
    fn name(&self) -> &'static str {
        "uniform_sphere"
    }
}

/// Registers all nine default plugins: `"sphere_shell"`, `"sphere_volume"`, `"uniform_box"`,
/// `"gaussian"`, `"grid"`, `"blob"`, `"tangential"`, `"spawn_cube"` (each reading its own
/// `PluginParams` key — `radius`/`half_extent_*`/`std_dev`/`grid_spacing`/
/// `blob_count`/`blob_spread`/`blob_radius`/`spawn_size`, all defaulting to a plausible slice
/// scale), and `"uniform_sphere"`. `radius` is shared between `sphere_shell`, `sphere_volume`,
/// and `tangential` — safe, since `Initializer` is a single-occupant socket, only one of them
/// is ever active at once.
pub fn register(r: &mut Registry) {
    r.register_init("sphere_shell", |p: &PluginParams| {
        Box::new(SphereShell {
            radius: p.get_or("radius", 1.0),
        })
    });
    r.register_init("sphere_volume", |p: &PluginParams| {
        Box::new(SphereVolume {
            radius: p.get_or("radius", 1.0),
        })
    });
    r.register_init("uniform_box", |p: &PluginParams| {
        Box::new(UniformBox {
            half_extent: Vec3::new(
                p.get_or("half_extent_x", 1.0),
                p.get_or("half_extent_y", 1.0),
                p.get_or("half_extent_z", 1.0),
            ),
        })
    });
    r.register_init("gaussian", |p: &PluginParams| {
        Box::new(Gaussian {
            std_dev: p.get_or("std_dev", 1.0),
        })
    });
    r.register_init("grid", |p: &PluginParams| {
        Box::new(Grid {
            grid_spacing: p.get_or("grid_spacing", 2.0),
        })
    });
    r.register_init("blob", |p: &PluginParams| {
        Box::new(Blob {
            blob_count: p.get_or("blob_count", 4.0) as u32,
            blob_spread: p.get_or("blob_spread", 10.0),
            blob_radius: p.get_or("blob_radius", 1.5),
        })
    });
    r.register_init("tangential", |p: &PluginParams| {
        Box::new(Tangential {
            radius: p.get_or("radius", 1.0),
        })
    });
    r.register_init("spawn_cube", |p: &PluginParams| {
        Box::new(SpawnCube {
            spawn_size: p.get_or("spawn_size", 1.0),
        })
    });
    r.register_noise("uniform_sphere", |_p: &PluginParams| {
        Box::new(UniformSphere)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(2.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(50)
            .vision_radius(1.0)
            .build()
            .unwrap()
    }

    #[test]
    fn sphere_shell_places_positions_on_the_surface() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, vel) = SphereShell { radius: 5.0 }.place(50, &p, &mut rng);
        for x in &pos {
            assert!((x.len() - 5.0).abs() < 1e-9);
        }
        for v in &vel {
            assert!((v.len() - p.cruise_speed).abs() < 1e-9);
        }
    }

    #[test]
    fn sphere_volume_places_positions_within_the_radius() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, _vel) = SphereVolume { radius: 5.0 }.place(50, &p, &mut rng);
        assert!(pos.iter().all(|x| x.len() <= 5.0 + 1e-9));
        // not all clustered at the centre (cube-root law) — at least one point past half-radius
        assert!(pos.iter().any(|x| x.len() > 2.5));
    }

    #[test]
    fn uniform_box_places_positions_within_the_box() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let half = Vec3::new(3.0, 4.0, 5.0);
        let (pos, _vel) = UniformBox { half_extent: half }.place(50, &p, &mut rng);
        for x in &pos {
            assert!(x.x.abs() <= half.x + 1e-9);
            assert!(x.y.abs() <= half.y + 1e-9);
            assert!(x.z.abs() <= half.z + 1e-9);
        }
    }

    #[test]
    fn gaussian_places_positions_around_the_origin_with_the_requested_spread() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, vel) = Gaussian { std_dev: 5.0 }.place(2000, &p, &mut rng);
        let mean = pos.iter().fold(Vec3::ZERO, |a, b| a + *b) * (1.0 / pos.len() as f64);
        assert!(
            mean.len() < 1.0,
            "large-N mean should sit near the origin, got {:?}",
            mean
        );
        let var = pos.iter().map(|x| x.len_sq()).sum::<f64>() / pos.len() as f64;
        // E[|X|^2] for an isotropic 3D Gaussian with per-axis std_dev sigma is 3*sigma^2.
        let expected = 3.0 * 5.0 * 5.0;
        assert!(
            (var - expected).abs() / expected < 0.15,
            "expected mean-square radius near {}, got {}",
            expected,
            var
        );
        for v in &vel {
            assert!((v.len() - p.cruise_speed).abs() < 1e-9);
        }
    }

    #[test]
    fn grid_places_positions_on_a_regular_lattice_with_the_requested_spacing() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        // A perfect cube (27 = 3^3) so every lattice point is used, with a simple exact check.
        let (pos, _vel) = Grid { grid_spacing: 2.0 }.place(27, &p, &mut rng);
        assert_eq!(pos.len(), 27);
        for x in &pos {
            for c in [x.x, x.y, x.z] {
                let steps = c / 2.0;
                assert!(
                    (steps - steps.round()).abs() < 1e-9,
                    "coordinate {} is not an integer multiple of grid_spacing",
                    c
                );
            }
        }
        // No two boids share a position — a real lattice, not everything collapsed to one cell.
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                assert!((pos[i] - pos[j]).len() > 1e-9);
            }
        }
    }

    #[test]
    fn grid_handles_a_non_perfect_cube_count_without_panicking() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, vel) = Grid { grid_spacing: 1.0 }.place(17, &p, &mut rng);
        assert_eq!(pos.len(), 17);
        assert_eq!(vel.len(), 17);
        assert!(pos.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn blob_places_positions_in_distinct_clusters() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let init = Blob {
            blob_count: 4,
            blob_spread: 50.0,
            blob_radius: 0.5,
        };
        let (pos, _vel) = init.place(400, &p, &mut rng);
        // Each blob is tight (radius 0.5) and centres are spread over up to 50 units, so
        // boid 0 (blob 0) and boid 1 (blob 1, round-robin assignment) should usually be much
        // farther apart than the within-blob spread.
        let mut any_far_pair = false;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if (pos[i] - pos[j]).len() > 5.0 {
                    any_far_pair = true;
                }
            }
        }
        assert!(
            any_far_pair,
            "expected at least one pair of blob centres to be well separated"
        );
    }

    #[test]
    fn blob_count_zero_degenerates_to_one_blob_instead_of_panicking() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let init = Blob {
            blob_count: 0,
            blob_spread: 10.0,
            blob_radius: 1.0,
        };
        let (pos, vel) = init.place(10, &p, &mut rng);
        assert_eq!(pos.len(), 10);
        assert!(pos.iter().all(|x| x.is_finite()));
        assert!(vel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tangential_velocities_are_perpendicular_to_their_own_radial_direction() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, vel) = Tangential { radius: 5.0 }.place(50, &p, &mut rng);
        for (x, v) in pos.iter().zip(vel.iter()) {
            assert!((x.len() - 5.0).abs() < 1e-9);
            assert!((v.len() - p.cruise_speed).abs() < 1e-9);
            let radial_dir = *x * (1.0 / x.len());
            assert!(
                v.dot(radial_dir).abs() < 1e-9,
                "velocity must have zero radial component, got dot={}",
                v.dot(radial_dir)
            );
        }
    }

    #[test]
    fn spawn_cube_places_positions_within_an_equal_extent_cube() {
        let p = params();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let (pos, _vel) = SpawnCube { spawn_size: 4.0 }.place(50, &p, &mut rng);
        for x in &pos {
            assert!(x.x.abs() <= 4.0 + 1e-9);
            assert!(x.y.abs() <= 4.0 + 1e-9);
            assert!(x.z.abs() <= 4.0 + 1e-9);
        }
    }

    #[test]
    fn every_new_variant_produces_finite_output_with_no_panic_at_n_zero() {
        let p = params();
        for (pos, vel) in [
            Gaussian { std_dev: 1.0 }.place(0, &p, &mut murmur_core::rng::for_boid(1, 2, 3)),
            Grid { grid_spacing: 1.0 }.place(0, &p, &mut murmur_core::rng::for_boid(1, 2, 3)),
            Blob {
                blob_count: 4,
                blob_spread: 10.0,
                blob_radius: 1.0,
            }
            .place(0, &p, &mut murmur_core::rng::for_boid(1, 2, 3)),
            Tangential { radius: 1.0 }.place(0, &p, &mut murmur_core::rng::for_boid(1, 2, 3)),
            SpawnCube { spawn_size: 1.0 }.place(0, &p, &mut murmur_core::rng::for_boid(1, 2, 3)),
        ] {
            assert_eq!(pos.len(), 0);
            assert_eq!(vel.len(), 0);
        }
    }

    #[test]
    fn conforms_to_initializer_contract() {
        murmur_conformance::initializer(&SphereVolume { radius: 1.0 });
        murmur_conformance::initializer(&SphereShell { radius: 1.0 });
        murmur_conformance::initializer(&UniformBox {
            half_extent: Vec3::new(1.0, 1.0, 1.0),
        });
        murmur_conformance::initializer(&Gaussian { std_dev: 1.0 });
        murmur_conformance::initializer(&Grid { grid_spacing: 1.0 });
        murmur_conformance::initializer(&Blob {
            blob_count: 4,
            blob_spread: 10.0,
            blob_radius: 1.0,
        });
        murmur_conformance::initializer(&Tangential { radius: 1.0 });
        murmur_conformance::initializer(&SpawnCube { spawn_size: 1.0 });
    }

    #[test]
    fn conforms_to_noise_source_contract() {
        murmur_conformance::noise_source(&UniformSphere);
    }

    #[test]
    fn registered_names_resolve_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert_eq!(
            reg.resolve_init("sphere_shell", &PluginParams::new())
                .unwrap()
                .name(),
            "sphere_shell"
        );
        assert_eq!(
            reg.resolve_init("sphere_volume", &PluginParams::new())
                .unwrap()
                .name(),
            "sphere_volume"
        );
        assert_eq!(
            reg.resolve_init("uniform_box", &PluginParams::new())
                .unwrap()
                .name(),
            "uniform_box"
        );
        assert_eq!(
            reg.resolve_init("gaussian", &PluginParams::new())
                .unwrap()
                .name(),
            "gaussian"
        );
        assert_eq!(
            reg.resolve_init("grid", &PluginParams::new())
                .unwrap()
                .name(),
            "grid"
        );
        assert_eq!(
            reg.resolve_init("blob", &PluginParams::new())
                .unwrap()
                .name(),
            "blob"
        );
        assert_eq!(
            reg.resolve_init("tangential", &PluginParams::new())
                .unwrap()
                .name(),
            "tangential"
        );
        assert_eq!(
            reg.resolve_init("spawn_cube", &PluginParams::new())
                .unwrap()
                .name(),
            "spawn_cube"
        );
        assert_eq!(
            reg.resolve_noise("uniform_sphere", &PluginParams::new())
                .unwrap()
                .name(),
            "uniform_sphere"
        );
    }
}
