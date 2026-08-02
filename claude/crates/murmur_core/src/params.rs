//! Parameters, split three ways by who needs to agree on the value (design/01_core.md §4):
//! `CoreParams` (universal), `OcclusionParams` (occlusion-toolkit-level), and per-plugin
//! private params (owned by each plugin crate, not here). Both structs here are validated by
//! their own builder; the hot path does zero checks.

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};

/// `dt = 1.0` by default and for all validated science (design/01_core.md §8's integration-
/// convention note); `DT_MAX` is the hard ceiling `step()` clamps `dt` to.
pub const DT_MAX: f64 = 1.0;

/// Universal — read by the pipeline itself or by more than one independently-selected plugin
/// (e.g. `vision_radius` must agree between `NeighborSelection` and `SpatialIndex`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoreParams {
    pub cruise_speed: f64,
    pub max_force: f64,
    pub speed_min_factor: f64,
    pub boid_count: u32,
    pub dt: f64,
    pub vision_radius: f64,
}

impl CoreParams {
    pub fn builder() -> CoreParamsBuilder {
        CoreParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoreParamsBuilder {
    cruise_speed: f64,
    max_force: f64,
    speed_min_factor: f64,
    boid_count: u32,
    dt: f64,
    vision_radius: f64,
}

impl Default for CoreParamsBuilder {
    fn default() -> Self {
        CoreParamsBuilder {
            cruise_speed: 1.0,
            max_force: 1.0,
            speed_min_factor: 0.3,
            boid_count: 0,
            dt: DT_MAX,
            vision_radius: 1.0,
        }
    }
}

impl CoreParamsBuilder {
    pub fn cruise_speed(mut self, v: f64) -> Self {
        self.cruise_speed = v;
        self
    }
    pub fn max_force(mut self, v: f64) -> Self {
        self.max_force = v;
        self
    }
    pub fn speed_min_factor(mut self, v: f64) -> Self {
        self.speed_min_factor = v;
        self
    }
    pub fn boid_count(mut self, v: u32) -> Self {
        self.boid_count = v;
        self
    }
    pub fn dt(mut self, v: f64) -> Self {
        self.dt = v;
        self
    }
    pub fn vision_radius(mut self, v: f64) -> Self {
        self.vision_radius = v;
        self
    }

    pub fn build(self) -> Result<CoreParams, ConfigError> {
        fn reject(field: &'static str, reason: impl Into<String>) -> ConfigError {
            ConfigError::InvalidParam {
                field,
                reason: reason.into(),
            }
        }
        if !(self.cruise_speed.is_finite() && self.cruise_speed > 0.0) {
            return Err(reject("cruise_speed", "must be finite and > 0"));
        }
        if !(self.max_force.is_finite() && self.max_force > 0.0) {
            return Err(reject("max_force", "must be finite and > 0"));
        }
        if !(0.0..=1.0).contains(&self.speed_min_factor) {
            return Err(reject("speed_min_factor", "must be in [0, 1]"));
        }
        if self.boid_count == 0 {
            return Err(reject("boid_count", "must be > 0"));
        }
        if !(0.0..=DT_MAX).contains(&self.dt) {
            return Err(reject("dt", format!("must be in [0, {DT_MAX}]")));
        }
        if !(self.vision_radius.is_finite() && self.vision_radius > 0.0) {
            return Err(reject("vision_radius", "must be finite and > 0"));
        }
        Ok(CoreParams {
            cruise_speed: self.cruise_speed,
            max_force: self.max_force,
            speed_min_factor: self.speed_min_factor,
            boid_count: self.boid_count,
            dt: self.dt,
            vision_radius: self.vision_radius,
        })
    }
}

/// Toolkit-level — read by the `occlude()` toolkit function and any `FlockingMode` that calls
/// it. Not pipeline-universal (a non-occlusion mode like Vicsek never touches it), but not
/// plugin-private either.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OcclusionParams {
    pub body_radius: f64,
    pub anisotropy: f64,
    pub blind_cone_half_angle: f64,
    pub blind_cone_enabled: bool,
    pub anisotropic_enabled: bool,
    pub max_candidates: u32,
    pub steric_enabled: bool,
    pub steric: f64,
    pub steric_radius_factor: f64,
}

impl OcclusionParams {
    pub fn builder() -> OcclusionParamsBuilder {
        OcclusionParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OcclusionParamsBuilder {
    body_radius: f64,
    anisotropy: f64,
    blind_cone_half_angle: f64,
    blind_cone_enabled: bool,
    anisotropic_enabled: bool,
    max_candidates: u32,
    steric_enabled: bool,
    steric: f64,
    steric_radius_factor: f64,
}

impl Default for OcclusionParamsBuilder {
    fn default() -> Self {
        OcclusionParamsBuilder {
            body_radius: 1.0,
            anisotropy: 1.0,
            blind_cone_half_angle: 0.524, // ~60 deg full rear cone, Pearce SI
            blind_cone_enabled: true,
            anisotropic_enabled: false,
            max_candidates: 20,
            steric_enabled: false,
            steric: 0.6,
            steric_radius_factor: 4.0,
        }
    }
}

impl OcclusionParamsBuilder {
    pub fn body_radius(mut self, v: f64) -> Self {
        self.body_radius = v;
        self
    }
    pub fn anisotropy(mut self, v: f64) -> Self {
        self.anisotropy = v;
        self
    }
    pub fn blind_cone_half_angle(mut self, v: f64) -> Self {
        self.blind_cone_half_angle = v;
        self
    }
    pub fn blind_cone_enabled(mut self, v: bool) -> Self {
        self.blind_cone_enabled = v;
        self
    }
    pub fn anisotropic_enabled(mut self, v: bool) -> Self {
        self.anisotropic_enabled = v;
        self
    }
    pub fn max_candidates(mut self, v: u32) -> Self {
        self.max_candidates = v;
        self
    }
    pub fn steric_enabled(mut self, v: bool) -> Self {
        self.steric_enabled = v;
        self
    }
    pub fn steric(mut self, v: f64) -> Self {
        self.steric = v;
        self
    }
    pub fn steric_radius_factor(mut self, v: f64) -> Self {
        self.steric_radius_factor = v;
        self
    }

    pub fn build(self) -> Result<OcclusionParams, ConfigError> {
        fn reject(field: &'static str, reason: impl Into<String>) -> ConfigError {
            ConfigError::InvalidParam {
                field,
                reason: reason.into(),
            }
        }
        if !(self.body_radius.is_finite() && self.body_radius > 0.0) {
            return Err(reject("body_radius", "must be finite and > 0"));
        }
        if !(self.anisotropy.is_finite() && self.anisotropy > 0.0) {
            return Err(reject("anisotropy", "must be finite and > 0"));
        }
        if !(0.0..=std::f64::consts::PI).contains(&self.blind_cone_half_angle) {
            return Err(reject("blind_cone_half_angle", "must be in [0, pi]"));
        }
        if self.max_candidates == 0 {
            return Err(reject("max_candidates", "must be > 0"));
        }
        if !(self.steric.is_finite() && self.steric >= 0.0) {
            return Err(reject("steric", "must be finite and >= 0"));
        }
        if !(self.steric_radius_factor.is_finite() && self.steric_radius_factor > 0.0) {
            return Err(reject("steric_radius_factor", "must be finite and > 0"));
        }
        Ok(OcclusionParams {
            body_radius: self.body_radius,
            anisotropy: self.anisotropy,
            blind_cone_half_angle: self.blind_cone_half_angle,
            blind_cone_enabled: self.blind_cone_enabled,
            anisotropic_enabled: self.anisotropic_enabled,
            max_candidates: self.max_candidates,
            steric_enabled: self.steric_enabled,
            steric: self.steric,
            steric_radius_factor: self.steric_radius_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_params_builder_accepts_sane_values() {
        let p = CoreParams::builder()
            .cruise_speed(8.94)
            .max_force(40.0)
            .speed_min_factor(0.3)
            .boid_count(400)
            .dt(0.5)
            .vision_radius(30.0)
            .build()
            .unwrap();
        // Every field round-trips through the builder unchanged (also the workspace's
        // config-field-usage drift scan, guard_config_field_usage, that every field is
        // actually read somewhere).
        assert_eq!(p.cruise_speed, 8.94);
        assert_eq!(p.max_force, 40.0);
        assert_eq!(p.speed_min_factor, 0.3);
        assert_eq!(p.boid_count, 400);
        assert_eq!(p.dt, 0.5);
        assert_eq!(p.vision_radius, 30.0);
    }

    #[test]
    fn core_params_builder_default_dt_is_dt_max() {
        let p = CoreParams::builder().boid_count(1).build().unwrap();
        assert_eq!(p.dt, DT_MAX);
    }

    #[test]
    fn core_params_builder_rejects_invalid_values() {
        assert!(CoreParams::builder()
            .cruise_speed(0.0)
            .boid_count(1)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .cruise_speed(-1.0)
            .boid_count(1)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .max_force(0.0)
            .boid_count(1)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .speed_min_factor(1.5)
            .boid_count(1)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .speed_min_factor(-0.1)
            .boid_count(1)
            .build()
            .is_err());
        assert!(CoreParams::builder().boid_count(0).build().is_err());
        assert!(CoreParams::builder()
            .boid_count(1)
            .dt(DT_MAX + 1.0)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .boid_count(1)
            .dt(-0.1)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .boid_count(1)
            .vision_radius(0.0)
            .build()
            .is_err());
        assert!(CoreParams::builder()
            .boid_count(1)
            .cruise_speed(f64::NAN)
            .build()
            .is_err());
    }

    #[test]
    fn occlusion_params_builder_accepts_sane_values() {
        let p = OcclusionParams::builder()
            .body_radius(0.09)
            .anisotropy(1.4)
            .blind_cone_half_angle(0.6)
            .blind_cone_enabled(false)
            .anisotropic_enabled(true)
            .max_candidates(20)
            .steric_enabled(true)
            .steric(0.6)
            .steric_radius_factor(4.0)
            .build()
            .unwrap();
        // Every field round-trips through the builder unchanged (also the workspace's
        // config-field-usage drift scan, guard_config_field_usage, that every field is
        // actually read somewhere).
        assert_eq!(p.body_radius, 0.09);
        assert_eq!(p.anisotropy, 1.4);
        assert_eq!(p.blind_cone_half_angle, 0.6);
        assert!(!p.blind_cone_enabled);
        assert!(p.anisotropic_enabled);
        assert_eq!(p.max_candidates, 20);
        assert!(p.steric_enabled);
        assert_eq!(p.steric, 0.6);
        assert_eq!(p.steric_radius_factor, 4.0);
    }

    #[test]
    fn occlusion_params_builder_rejects_invalid_values() {
        assert!(OcclusionParams::builder().body_radius(0.0).build().is_err());
        assert!(OcclusionParams::builder().anisotropy(-1.0).build().is_err());
        assert!(OcclusionParams::builder()
            .blind_cone_half_angle(-0.1)
            .build()
            .is_err());
        assert!(OcclusionParams::builder()
            .max_candidates(0)
            .build()
            .is_err());
        assert!(OcclusionParams::builder().steric(-1.0).build().is_err());
        assert!(OcclusionParams::builder()
            .steric_radius_factor(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn params_survive_a_serde_roundtrip() {
        let core = CoreParams::builder()
            .boid_count(10)
            .vision_radius(12.5)
            .build()
            .unwrap();
        let json = serde_json::to_string(&core).unwrap();
        let back: CoreParams = serde_json::from_str(&json).unwrap();
        assert_eq!(core, back);

        let occ = OcclusionParams::builder()
            .body_radius(0.09)
            .build()
            .unwrap();
        let json = serde_json::to_string(&occ).unwrap();
        let back: OcclusionParams = serde_json::from_str(&json).unwrap();
        assert_eq!(occ, back);
    }
}
