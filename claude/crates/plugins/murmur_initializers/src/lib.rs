//! Default `Initializer`/`NoiseSource` implementations (design/01_core.md §12.5): `SphereShell`,
//! `SphereVolume`, `UniformBox` (`Initializer`), and `UniformSphere` (`NoiseSource`). The
//! traits are core; these concrete impls are plugins, same status as any other socket's
//! default (design/00_overview.md §2) — bundled in one crate since each is only a few lines.

use murmur_core::{
    sample_unit_sphere, uniform01, CoreParams, Initializer, NoiseSource, PluginParams, Registry,
    Rng, Vec3,
};

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

/// Registers all four default plugins: `"sphere_shell"`, `"sphere_volume"`, `"uniform_box"`
/// (each reading a `radius`/`half_extent_*` from `PluginParams`, default `1.0`), and
/// `"uniform_sphere"`.
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
    fn conforms_to_initializer_contract() {
        murmur_conformance::initializer(&SphereVolume { radius: 1.0 });
        murmur_conformance::initializer(&SphereShell { radius: 1.0 });
        murmur_conformance::initializer(&UniformBox {
            half_extent: Vec3::new(1.0, 1.0, 1.0),
        });
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
            reg.resolve_noise("uniform_sphere", &PluginParams::new())
                .unwrap()
                .name(),
            "uniform_sphere"
        );
    }
}
