//! `Vicsek` — minimal alignment + angular noise, the classic order-disorder baseline
//! (design/02_plugins.md §5, pymurmur's `physics/forces/vicsek.py`). The second `FlockingMode`
//! plugin built, proving the `FlockingMode` seam genuinely supports more than one occupant
//! (roadmap.md Phase 8) — no occlusion, no projection, just neighbour-heading alignment.
//!
//! Update rule per boid: blend the current heading toward the mean heading of `self +
//! neighbours` by `couplage` (the alignment coupling strength, `[0,1]`; `1.0` reproduces the
//! textbook Vicsek 1995 rule of full alignment to the neighbourhood average), then perturb by
//! isotropic noise scaled by `diffusion`. Cruising speed is constant (`core_params.cruise_speed`)
//! — Vicsek has no speed dynamics of its own.

use murmur_core::{
    BoidCtx, ConfigError, FlockingMode, OcclusionScratch, PluginParams, Registry, Rng, SteerIntent,
    Vec3, MIN_LEN2,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VicsekParams {
    /// Alignment coupling strength toward the neighbourhood mean heading, `[0, 1]`.
    pub couplage: f64,
    /// Angular noise magnitude added after alignment.
    pub diffusion: f64,
}

impl VicsekParams {
    pub fn builder() -> VicsekParamsBuilder {
        VicsekParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VicsekParamsBuilder {
    couplage: f64,
    diffusion: f64,
}

impl Default for VicsekParamsBuilder {
    fn default() -> Self {
        VicsekParamsBuilder {
            couplage: 1.0,
            diffusion: 0.3,
        }
    }
}

impl VicsekParamsBuilder {
    pub fn couplage(mut self, v: f64) -> Self {
        self.couplage = v;
        self
    }
    pub fn diffusion(mut self, v: f64) -> Self {
        self.diffusion = v;
        self
    }

    pub fn build(self) -> Result<VicsekParams, ConfigError> {
        if !(self.couplage.is_finite() && (0.0..=1.0).contains(&self.couplage)) {
            return Err(ConfigError::InvalidParam {
                field: "couplage",
                reason: "must be finite and in [0, 1]".into(),
            });
        }
        if !(self.diffusion.is_finite() && self.diffusion >= 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "diffusion",
                reason: "must be finite and >= 0".into(),
            });
        }
        Ok(VicsekParams {
            couplage: self.couplage,
            diffusion: self.diffusion,
        })
    }
}

pub struct Vicsek {
    pub params: VicsekParams,
}

impl Vicsek {
    pub fn new(params: VicsekParams) -> Self {
        Vicsek { params }
    }
}

impl FlockingMode for Vicsek {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        rng: &mut Rng,
    ) -> SteerIntent {
        let heading = ctx.vel.normalized();

        let mut sum = heading; // classic Vicsek includes self in the neighbourhood average
        for n in ctx.neighbors {
            if n.velocity.len_sq() > MIN_LEN2 {
                sum += n.velocity.normalized();
            }
        }
        let mean_heading = if sum.len_sq() > MIN_LEN2 {
            sum.normalized()
        } else {
            heading
        };

        let blended = heading * (1.0 - self.params.couplage) + mean_heading * self.params.couplage;
        let blended = if blended.len_sq() > MIN_LEN2 {
            blended.normalized()
        } else {
            heading
        };

        let noise = murmur_core::sample_unit_sphere(rng) * self.params.diffusion;
        let desired = blended + noise;
        let desired_v = if desired.len_sq() > MIN_LEN2 {
            desired.normalized() * ctx.core_params.cruise_speed
        } else {
            ctx.vel
        };

        SteerIntent {
            desired_v,
            extra_force: Vec3::ZERO,
            theta: 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "vicsek"
    }
}

/// Registers `Vicsek` under the name `"vicsek"`, reading `couplage`/`diffusion` from
/// `PluginParams` (defaulting to the textbook full-alignment values). The factory type can't
/// return `Result` (design/02_plugins.md §1), so a malformed override falls back to the
/// default rather than panicking — same pattern as `murmur_pearce::register`.
pub fn register(r: &mut Registry) {
    r.register_mode("vicsek", |p: &PluginParams| {
        let params = VicsekParams::builder()
            .couplage(p.get_or("couplage", 1.0))
            .diffusion(p.get_or("diffusion", 0.3))
            .build()
            .unwrap_or_else(|_| VicsekParams::builder().build().expect("defaults are valid"));
        Box::new(Vicsek::new(params)) as Box<dyn FlockingMode>
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Neighbor, Species};

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
    fn builder_rejects_out_of_range_couplage() {
        assert!(VicsekParams::builder().couplage(1.5).build().is_err());
        assert!(VicsekParams::builder().couplage(-0.1).build().is_err());
    }

    #[test]
    fn builder_rejects_negative_diffusion() {
        assert!(VicsekParams::builder().diffusion(-1.0).build().is_err());
    }

    #[test]
    fn zero_diffusion_and_full_couplage_aligns_exactly_to_neighbor_mean() {
        let mode = Vicsek::new(VicsekParams {
            couplage: 1.0,
            diffusion: 0.0,
        });
        let domain = StubDomain;
        let params = core_params();
        let neighbors = vec![
            Neighbor {
                index: 1,
                distance: 1.0,
                direction: Vec3::new(1.0, 0.0, 0.0),
                velocity: Vec3::new(0.0, 1.0, 0.0),
            },
            Neighbor {
                index: 2,
                distance: 1.0,
                direction: Vec3::new(0.0, 1.0, 0.0),
                velocity: Vec3::new(0.0, 1.0, 0.0),
            },
        ];
        let ctx = BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel: Vec3::new(1.0, 0.0, 0.0), // self heading +x
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &params,
            domain: &domain,
        };
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 2, 3);
        let intent = mode.desired(ctx, &mut scratch, &mut rng);
        // mean of (+x, +y, +y) normalized -> mostly +y; exact value: sum=(1,2,0)/norm.
        let expected = Vec3::new(1.0, 2.0, 0.0).normalized() * params.cruise_speed;
        assert!(
            (intent.desired_v - expected).len() < 1e-9,
            "got {:?}",
            intent.desired_v
        );
    }

    #[test]
    fn conforms_to_flocking_mode_contract() {
        let mode = Vicsek::new(VicsekParams::builder().build().unwrap());
        murmur_conformance::flocking_mode(&mode);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("vicsek", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "vicsek");
    }

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
        let real = reg.resolve_mode("vicsek", &PluginParams::new()).unwrap();
        let stub = reg.resolve_mode("stub_mode", &PluginParams::new()).unwrap();
        assert_eq!(real.name(), "vicsek");
        assert_eq!(stub.name(), "stub_mode");
    }
}
