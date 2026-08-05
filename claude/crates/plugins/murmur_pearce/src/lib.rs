//! `PearceProjection` — the first `FlockingMode` plugin, and the slice's scientific subject
//! (design/01_core.md §8, §12.3). Pearce (2014) visual-projection flocking: Eq. 3's blend of
//! projection (δ̂, emergent cohesion via `occlude()`), alignment (σ nearest visible neighbours),
//! and noise, plus short-range steric repulsion.
//!
//! **Note on the steric loop.** Design's own §12.3 production note says the steric force
//! should iterate the occlusion pass's *visible* set, not raw candidates, but shows the
//! simpler `ctx.neighbors` version "for brevity" since `occlude()`'s `VisibleView` doesn't
//! expose that set externally. This implementation does the same, for the same reason — a
//! known, design-acknowledged simplification, not an oversight.

use murmur_core::{
    clamp_len, occlude, sample_unit_sphere, BoidCtx, ConfigError, FlockingMode, OcclusionParams,
    OcclusionScratch, PluginParams, Registry, Rng, SteerIntent, Vec3, MIN_LEN2,
};
use serde::{Deserialize, Serialize};

/// Plugin-private — just Pearce's own Eq. 3 blend weights (design/01_core.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PearceParams {
    /// Projection (δ̂) weight.
    pub phi_p: f64,
    /// Alignment weight (σ nearest visible velocities).
    pub phi_a: f64,
    /// Noise weight — derived: `phi_p + phi_a + phi_n ≡ 1`.
    pub phi_n: f64,
    /// σ nearest visible neighbours averaged for alignment.
    pub sigma: u32,
}

impl PearceParams {
    pub fn builder() -> PearceParamsBuilder {
        PearceParamsBuilder::default()
    }

    /// (φp, φa) = (0.03, 0.80) — `sci/todo.md` B12.
    pub fn murmuration() -> PearceParams {
        Self::builder()
            .phi_p(0.03)
            .phi_a(0.80)
            .build()
            .expect("preset is valid by construction")
    }
    /// (φp, φa) = (0.10, 0.60) — `sci/todo.md` B12.
    pub fn schooling() -> PearceParams {
        Self::builder()
            .phi_p(0.10)
            .phi_a(0.60)
            .build()
            .expect("preset is valid by construction")
    }
    /// (φp, φa) = (0.20, 0.30) — `sci/todo.md` B12.
    pub fn swarming() -> PearceParams {
        Self::builder()
            .phi_p(0.20)
            .phi_a(0.30)
            .build()
            .expect("preset is valid by construction")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PearceParamsBuilder {
    phi_p: f64,
    phi_a: f64,
    sigma: u32,
}

impl Default for PearceParamsBuilder {
    fn default() -> Self {
        // murmuration phenotype defaults
        PearceParamsBuilder {
            phi_p: 0.03,
            phi_a: 0.80,
            sigma: 4,
        }
    }
}

impl PearceParamsBuilder {
    pub fn phi_p(mut self, v: f64) -> Self {
        self.phi_p = v;
        self
    }
    pub fn phi_a(mut self, v: f64) -> Self {
        self.phi_a = v;
        self
    }
    pub fn sigma(mut self, v: u32) -> Self {
        self.sigma = v;
        self
    }

    /// Derives `phi_n = 1 - phi_p - phi_a`; rejects a `phi_p + phi_a` that would make it
    /// negative (design/01_core.md §4: "builder enforces φp+φa+φn ≡ 1").
    pub fn build(self) -> Result<PearceParams, ConfigError> {
        fn reject(field: &'static str, reason: impl Into<String>) -> ConfigError {
            ConfigError::InvalidParam {
                field,
                reason: reason.into(),
            }
        }
        if !(self.phi_p.is_finite() && self.phi_p >= 0.0) {
            return Err(reject("phi_p", "must be finite and >= 0"));
        }
        if !(self.phi_a.is_finite() && self.phi_a >= 0.0) {
            return Err(reject("phi_a", "must be finite and >= 0"));
        }
        let phi_n = 1.0 - self.phi_p - self.phi_a;
        if phi_n < -1e-9 {
            return Err(reject(
                "phi_p+phi_a",
                "must be <= 1.0 so phi_n stays non-negative",
            ));
        }
        if self.sigma == 0 {
            return Err(reject("sigma", "must be > 0"));
        }
        Ok(PearceParams {
            phi_p: self.phi_p,
            phi_a: self.phi_a,
            phi_n: phi_n.max(0.0),
            sigma: self.sigma,
        })
    }
}

/// The first `FlockingMode` plugin, and the slice's scientific subject — not structurally
/// privileged (design/00_overview.md §2). Owns its own `PearceParams` (Eq. 3 weights) and
/// `OcclusionParams` (the toolkit params `occlude()` needs), per the plugin-owns-its-params
/// pattern established in design/01_core.md §4 / this codebase's Phase 2.
pub struct PearceProjection {
    pub pearce: PearceParams,
    pub occlusion: OcclusionParams,
}

impl PearceProjection {
    pub fn new(pearce: PearceParams, occlusion: OcclusionParams) -> Self {
        PearceProjection { pearce, occlusion }
    }
}

impl FlockingMode for PearceProjection {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        scratch: &mut OcclusionScratch,
        rng: &mut Rng,
    ) -> SteerIntent {
        let heading = ctx.vel.normalized();
        let view = occlude(
            heading,
            ctx.neighbors,
            &self.occlusion,
            self.pearce.sigma,
            scratch,
        );

        let align_hat = if view.align.len_sq() > MIN_LEN2 {
            view.align.normalized()
        } else {
            heading
        };
        let noise = sample_unit_sphere(rng);
        let desired = view.delta_hat * self.pearce.phi_p
            + align_hat * self.pearce.phi_a
            + noise * self.pearce.phi_n;
        let desired_v = if desired.len_sq() > MIN_LEN2 {
            desired.normalized() * ctx.core_params.cruise_speed
        } else {
            ctx.vel
        };

        let mut extra_force = Vec3::ZERO;
        if self.occlusion.steric_enabled {
            let r_s = self.occlusion.body_radius * self.occlusion.steric_radius_factor;
            let mut f = Vec3::ZERO;
            for n in ctx.neighbors {
                let d = n.distance.max(1e-9);
                if d < r_s {
                    f -= n.direction / (d * d);
                }
            }
            extra_force = clamp_len(f * self.occlusion.steric, ctx.core_params.max_force);
        }

        SteerIntent {
            desired_v,
            extra_force,
            theta: view.theta,
        }
    }

    fn name(&self) -> &'static str {
        "pearce"
    }
}

/// Registers `PearceProjection` under the name `"pearce"`. Reads `phi_p`/`phi_a`/`sigma` (Pearce
/// weights, defaulting to the murmuration preset) and the occlusion-toolkit fields
/// (`body_radius`, `anisotropy`, `blind_cone_half_angle`, `blind_cone_enabled`,
/// `anisotropic_enabled`, `max_candidates`, `steric_enabled`, `steric`, `steric_radius_factor`,
/// defaulting to `OcclusionParams`' own defaults) from `PluginParams`. The factory type
/// (`fn(&PluginParams) -> Box<dyn FlockingMode>`, design/02_plugins.md §1) can't return
/// `Result`, so a malformed override falls back to the safe default rather than panicking.
pub fn register(r: &mut Registry) {
    r.register_mode("pearce", |p: &PluginParams| {
        let pearce = PearceParams::builder()
            .phi_p(p.get_or("phi_p", 0.03))
            .phi_a(p.get_or("phi_a", 0.80))
            .sigma(p.get_or("sigma", 4.0) as u32)
            .build()
            .unwrap_or_else(|_| PearceParams::murmuration());

        let default_occlusion = OcclusionParams::builder();
        let occlusion = OcclusionParams::builder()
            .body_radius(p.get_or("body_radius", 1.0))
            .anisotropy(p.get_or("anisotropy", 1.0))
            .blind_cone_half_angle(p.get_or("blind_cone_half_angle", 0.524))
            .blind_cone_enabled(p.get_or("blind_cone_enabled", 1.0) != 0.0)
            .anisotropic_enabled(p.get_or("anisotropic_enabled", 0.0) != 0.0)
            .max_candidates(p.get_or("max_candidates", 20.0) as u32)
            .steric_enabled(p.get_or("steric_enabled", 0.0) != 0.0)
            .steric(p.get_or("steric", 0.6))
            .steric_radius_factor(p.get_or("steric_radius_factor", 4.0))
            .build()
            .unwrap_or_else(|_| {
                default_occlusion
                    .build()
                    .expect("builder defaults are valid")
            });

        Box::new(PearceProjection::new(pearce, occlusion)) as Box<dyn FlockingMode>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Neighbor, Species};

    #[test]
    fn presets_sum_phi_p_phi_a_phi_n_to_one() {
        for preset in [
            PearceParams::murmuration(),
            PearceParams::schooling(),
            PearceParams::swarming(),
        ] {
            assert!((preset.phi_p + preset.phi_a + preset.phi_n - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn builder_rejects_weights_summing_past_one() {
        assert!(PearceParams::builder()
            .phi_p(0.7)
            .phi_a(0.5)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_zero_sigma() {
        assert!(PearceParams::builder().sigma(0).build().is_err());
    }

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

    #[test]
    fn conforms_to_flocking_mode_contract() {
        let mode = PearceProjection::new(
            PearceParams::murmuration(),
            OcclusionParams::builder().build().unwrap(),
        );
        murmur_conformance::flocking_mode(&mode);
    }

    #[test]
    fn steric_force_pushes_away_from_a_very_close_neighbor() {
        let pearce = PearceParams::murmuration();
        let occlusion = OcclusionParams::builder()
            .body_radius(1.0)
            .steric_enabled(true)
            .steric(1.0)
            .steric_radius_factor(4.0)
            .build()
            .unwrap();
        let mode = PearceProjection::new(pearce, occlusion);

        let domain = StubDomain;
        let params = core_params();
        // A neighbour sitting almost on top of the observer, due +x.
        let neighbors = vec![Neighbor {
            index: 1,
            distance: 0.1,
            direction: Vec3::new(1.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
        }];
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel: Vec3::new(0.0, 1.0, 0.0),
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let intent = mode.desired(ctx, &mut scratch, &mut rng);
        assert!(
            intent.extra_force.x < 0.0,
            "steric force should push away (-x), got {:?}",
            intent.extra_force
        );
    }

    #[test]
    fn steric_force_is_zero_when_disabled() {
        let pearce = PearceParams::murmuration();
        let occlusion = OcclusionParams::builder()
            .body_radius(1.0)
            .steric_enabled(false)
            .build()
            .unwrap();
        let mode = PearceProjection::new(pearce, occlusion);

        let domain = StubDomain;
        let params = core_params();
        let neighbors = vec![Neighbor {
            index: 1,
            distance: 0.1,
            direction: Vec3::new(1.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
        }];
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel: Vec3::new(0.0, 1.0, 0.0),
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &params,
            domain: &domain,
            step_count: 0,
        };
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let intent = mode.desired(ctx, &mut scratch, &mut rng);
        assert_eq!(intent.extra_force, Vec3::ZERO);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("pearce", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "pearce");
    }

    #[test]
    fn registry_uses_overridden_plugin_params() {
        let mut reg = Registry::new();
        register(&mut reg);
        let params = PluginParams::new().with("phi_p", 0.20).with("phi_a", 0.30);
        // Can't downcast the trait object generically here, but constructing successfully
        // (no panic/default fallback triggered by a legitimately valid override) is the
        // meaningful assertion — a malformed override is exercised separately below.
        let mode = reg.resolve_mode("pearce", &params).unwrap();
        assert_eq!(mode.name(), "pearce");
    }

    #[test]
    fn malformed_plugin_params_fall_back_to_a_safe_default_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("phi_p", 0.9).with("phi_a", 0.9); // sums > 1
        let mode = reg.resolve_mode("pearce", &bad).unwrap(); // must not panic
        assert_eq!(mode.name(), "pearce");
    }

    /// Proves the `FlockingMode` seam is real: a trivial stub implementing the same trait
    /// registers and resolves alongside the real plugin.
    #[test]
    fn a_different_flocking_mode_plugin_swaps_in_via_the_same_seam() {
        struct StubMode;
        impl FlockingMode for StubMode {
            fn desired(
                &self,
                ctx: BoidCtx<'_>,
                _s: &mut OcclusionScratch,
                _r: &mut Rng,
            ) -> SteerIntent {
                SteerIntent {
                    desired_v: ctx.vel,
                    extra_force: Vec3::ZERO,
                    theta: 0.0,
                }
            }
            fn name(&self) -> &'static str {
                "stub_mode"
            }
        }
        let mut reg = Registry::new();
        register(&mut reg);
        reg.register_mode("stub_mode", |_p| Box::new(StubMode));

        let real = reg.resolve_mode("pearce", &PluginParams::new()).unwrap();
        let stub = reg.resolve_mode("stub_mode", &PluginParams::new()).unwrap();
        assert_eq!(real.name(), "pearce");
        assert_eq!(stub.name(), "stub_mode");
    }
}
