//! `Initializer` (initial positions + velocities) and `NoiseSource` (stochastic perturbation)
//! traits — core; concrete implementations (`SphereShell`, `SphereVolume`, `UniformBox`,
//! `UniformSphere`, ...) are plugins, same status as any other socket's default.

use crate::math::Vec3;
use crate::params::CoreParams;
use crate::registry::{PluginParams, Warning};
use crate::rng::Rng;

pub trait Initializer: Send + Sync {
    fn place(&self, count: u32, params: &CoreParams, rng: &mut Rng) -> (Vec<Vec3>, Vec<Vec3>);
    fn name(&self) -> &'static str;

    /// See `FlockingMode::resolved_params` (modes.rs) — same seam, same default.
    fn resolved_params(&self) -> PluginParams {
        PluginParams::new()
    }

    /// See `FlockingMode::validate_and_fix` (modes.rs) — same seam, same default, same timing.
    fn validate_and_fix(
        &mut self,
        _core: &CoreParams,
        _others: &[(&str, PluginParams)],
    ) -> Vec<Warning> {
        Vec::new()
    }
}

/// Pearce uses `UniformSphere` (area-uniform on S²) for η̂.
pub trait NoiseSource: Send + Sync {
    fn sample(&self, rng: &mut Rng) -> Vec3;
    fn name(&self) -> &'static str;

    /// See `FlockingMode::resolved_params` (modes.rs) — same seam, same default.
    fn resolved_params(&self) -> PluginParams {
        PluginParams::new()
    }

    /// See `FlockingMode::validate_and_fix` (modes.rs) — same seam, same default, same timing.
    fn validate_and_fix(
        &mut self,
        _core: &CoreParams,
        _others: &[(&str, PluginParams)],
    ) -> Vec<Warning> {
        Vec::new()
    }
}
