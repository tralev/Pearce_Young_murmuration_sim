//! `Initializer` (initial positions + velocities) and `NoiseSource` (stochastic perturbation)
//! traits — core; concrete implementations (`SphereShell`, `SphereVolume`, `UniformBox`,
//! `UniformSphere`, ...) are plugins, same status as any other socket's default.

use crate::math::Vec3;
use crate::params::CoreParams;
use crate::rng::Rng;

pub trait Initializer: Send + Sync {
    fn place(&self, count: u32, params: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>);
    fn name(&self) -> &'static str;
}

/// Pearce uses `UniformSphere` (area-uniform on S²) for η̂.
pub trait NoiseSource: Send + Sync {
    fn sample(&self, rng: &mut Rng) -> Vec3;
    fn name(&self) -> &'static str;
}
