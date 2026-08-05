//! `YoungConsensus` — the `FlockingMode` design/02_plugins.md §5 names for `murmur_young`:
//! "Consensus-alignment + cohesion + steric using m\* neighbours." Closes out Phase 14
//! (roadmap.md §6.2): the H₂ Rust-native path (`murmur_core::h2`, promoted from
//! `claude/spikes/rust_h2/` the same day) finally gets a real consumer, so Y-a/Y-c stop being
//! "the flock happens to score some m\*" and start being "the flock's own steering *uses* the
//! m\* its own sensing-graph robustness curve says is optimal."
//!
//! **m\* source**: droidmur's own on-device pattern (`young_mode.{h,cpp}`, using Spectra/C++)
//! is prior art that computing m\* live is workable at all — not a Rust dependency to
//! replicate. This plugin uses `murmur_core::h2` (`nalgebra`'s dense `SymmetricEigen`)
//! instead, this project's own Rust-native path.
//!
//! **Found and fixed G7 building this** (roadmap.md §12): `FlockingMode::desired()` — `&self`,
//! one boid's own `BoidCtx` — has no way to see any *other* boid's state, and no earlier mode
//! needed to (Pearce/Vicsek both work entirely from `ctx.neighbors`). H₂/m\* is inherently a
//! whole-flock quantity, not a per-boid one — computing it needs every active boid's position
//! at once. Fixed by adding `FlockingMode::pre_step(&self, boids: &BoidColumns, step_count:
//! u64)` (default no-op, so every earlier mode is unaffected), called once per step ahead of
//! the parallel read phase (`pipeline.rs`), mirroring `StepHook::pre_step`'s shape but for
//! `&self` (interior mutability via `Mutex`, the same discipline `murmur_spin_wave`/
//! `murmur_predator_fsm` already use for cross-boid state under a shared-reference method).
//!
//! **Throttled, not recomputed every step.** A full `h2_at_m` sweep is O(N³) per candidate
//! `m` (dense eigendecomposition) — recomputing it every single step would make even
//! moderate-N testing impractically slow. `m*` is refreshed every `refresh_interval` steps
//! (default 20) and cached in between; `desired()` always reads the cached value, never
//! blocks on a fresh eigensolve. This is a genuine, documented performance tradeoff, not the
//! "fixed fallback" alternative design/02_plugins.md's own phrasing names as the other
//! option — the cached value *is* a real eigensolve result, just not from the current exact
//! step; `m_fallback` only applies when there are too few boids for any candidate `m` to be
//! meaningful (`m_star` returns `None`), e.g. right after a `Reset` to a tiny flock.
//!
//! **Steering**: for the cached `m*` nearest neighbours (from whichever `NeighborSelection`
//! is composed, sorted here by distance and truncated to `m*` — this mode doesn't require its
//! own dedicated `NeighborSelection`, any composed one that returns at least `m*` candidates
//! within range works), blends alignment (mean neighbour velocity), cohesion (mean unit
//! bearing toward neighbours — the same `direction` field `murmur_pearce`'s own δ̂ uses, just
//! unweighted by occlusion here) and noise; steric short-range repulsion reuses `murmur_pearce`
//! §12.3's own documented `ctx.neighbors`-based simplification (not the design's ideal
//! visible-set-only version — same known, acknowledged simplification, not a new one).

use std::sync::Mutex;

use murmur_core::{
    clamp_len, sample_unit_sphere, BoidColumns, BoidCtx, ConfigError, FlockingMode, Neighbor,
    OcclusionScratch, PluginParams, Registry, Rng, Species, SteerIntent, Vec3, MIN_LEN2,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YoungParams {
    pub m_min: usize,
    pub m_max: usize,
    pub m_fallback: usize,
    pub refresh_interval: u64,
    pub align_weight: f64,
    pub cohesion_weight: f64,
    pub noise_weight: f64,
    pub steric_enabled: bool,
    pub steric: f64,
    pub steric_radius: f64,
}

impl YoungParams {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.m_min == 0 || self.m_max < self.m_min {
            return Err(ConfigError::InvalidParam {
                field: "m_min/m_max",
                reason: "must satisfy 0 < m_min <= m_max".to_string(),
            });
        }
        if self.refresh_interval == 0 {
            return Err(ConfigError::InvalidParam {
                field: "refresh_interval",
                reason: "must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Default)]
struct Cache {
    m_star: Option<usize>,
    last_computed_step: Option<u64>,
}

pub struct YoungConsensus {
    params: YoungParams,
    cache: Mutex<Cache>,
}

impl YoungConsensus {
    pub fn new(params: YoungParams) -> Result<Self, ConfigError> {
        params.validate()?;
        Ok(YoungConsensus {
            params,
            cache: Mutex::new(Cache::default()),
        })
    }
}

impl FlockingMode for YoungConsensus {
    fn pre_step(&self, boids: &BoidColumns, step_count: u64) {
        let mut cache = self.cache.lock().expect("young cache mutex poisoned");
        let due = match cache.last_computed_step {
            None => true,
            Some(last) => step_count >= last + self.params.refresh_interval,
        };
        if !due {
            return;
        }
        let prey_pos: Vec<Vec3> = boids
            .iter_active()
            .filter(|&i| boids.species[i as usize] != Species::Predator)
            .map(|i| boids.pos[i as usize])
            .collect();
        let m = murmur_core::m_star(&prey_pos, self.params.m_min..=self.params.m_max)
            .unwrap_or(self.params.m_fallback);
        cache.m_star = Some(m);
        cache.last_computed_step = Some(step_count);
    }

    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        rng: &mut Rng,
    ) -> SteerIntent {
        let m = {
            let cache = self.cache.lock().expect("young cache mutex poisoned");
            cache.m_star.unwrap_or(self.params.m_fallback).max(1)
        };

        let mut neighbors: Vec<&Neighbor> = ctx.neighbors.iter().collect();
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let take = m.min(neighbors.len());
        let chosen = &neighbors[..take];

        let heading = if ctx.vel.len_sq() > MIN_LEN2 {
            ctx.vel.normalized()
        } else {
            Vec3::ZERO
        };

        let align_dir = if take > 0 {
            let mut sum = Vec3::ZERO;
            for n in chosen {
                sum += n.velocity;
            }
            if sum.len_sq() > MIN_LEN2 {
                sum.normalized()
            } else {
                heading
            }
        } else {
            heading
        };

        let cohesion_dir = if take > 0 {
            let mut sum = Vec3::ZERO;
            for n in chosen {
                sum += n.direction;
            }
            if sum.len_sq() > MIN_LEN2 {
                sum.normalized()
            } else {
                Vec3::ZERO
            }
        } else {
            Vec3::ZERO
        };

        let noise = sample_unit_sphere(rng);

        let desired = align_dir * self.params.align_weight
            + cohesion_dir * self.params.cohesion_weight
            + noise * self.params.noise_weight;
        let desired_v = if desired.len_sq() > MIN_LEN2 {
            desired.normalized() * ctx.core_params.cruise_speed
        } else {
            ctx.vel
        };

        let mut extra_force = Vec3::ZERO;
        if self.params.steric_enabled {
            let mut f = Vec3::ZERO;
            for n in chosen {
                let d = n.distance.max(1e-9);
                if d < self.params.steric_radius {
                    f -= n.direction / (d * d);
                }
            }
            extra_force = clamp_len(f * self.params.steric, ctx.core_params.max_force);
        }

        SteerIntent {
            desired_v,
            extra_force,
            theta: 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "young"
    }
}

/// Registers `YoungConsensus` under the name `"young"`. Plugin params: `m_min`/`m_max`
/// (default `2`/`12`, matching `sci/todo.md`'s own m* sweep range), `m_fallback` (default `6`,
/// Young 2013's own empirical `m* ~ 6-7`), `refresh_interval` (default `20` steps),
/// `align_weight`/`cohesion_weight`/`noise_weight` (default `0.5`/`0.3`/`0.2`),
/// `steric_enabled` (default off, matching `murmur_pearce`'s own default), `steric` (default
/// `0.6`), `steric_radius_factor` × `body_radius` (defaults `4.0`/`1.0`, same convention as
/// `murmur_pearce`/`murmur_predator`).
pub fn register(r: &mut Registry) {
    r.register_mode("young", |p: &PluginParams| {
        let body_radius = p.get_or("body_radius", 1.0);
        let steric_radius = body_radius * p.get_or("steric_radius_factor", 4.0);
        let params = YoungParams {
            m_min: p.get_or("m_min", 2.0) as usize,
            m_max: p.get_or("m_max", 12.0) as usize,
            m_fallback: p.get_or("m_fallback", 6.0) as usize,
            refresh_interval: p.get_or("refresh_interval", 20.0) as u64,
            align_weight: p.get_or("align_weight", 0.5),
            cohesion_weight: p.get_or("cohesion_weight", 0.3),
            noise_weight: p.get_or("noise_weight", 0.2),
            steric_enabled: p.get_or("steric_enabled", 0.0) != 0.0,
            steric: p.get_or("steric", 0.6),
            steric_radius,
        };
        Box::new(YoungConsensus::new(params).expect("validated by builder default constants"))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain};

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

    fn default_params() -> YoungParams {
        YoungParams {
            m_min: 2,
            m_max: 12,
            m_fallback: 6,
            refresh_interval: 20,
            align_weight: 0.5,
            cohesion_weight: 0.3,
            noise_weight: 0.0, // deterministic for unit tests
            steric_enabled: false,
            steric: 0.6,
            steric_radius: 2.0,
        }
    }

    fn core_params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(10.0)
            .speed_min_factor(0.3)
            .boid_count(20)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn boids_with(entries: &[(Vec3, Vec3, Species)]) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(entries.len() as u32);
        for &(p, v, s) in entries {
            b.add(p, v, s, 0);
        }
        b
    }

    fn neighbor_from(observer: Vec3, other_pos: Vec3, other_vel: Vec3, index: u32) -> Neighbor {
        let offset = other_pos - observer;
        Neighbor {
            index,
            distance: offset.len(),
            direction: offset.normalized(),
            velocity: other_vel,
        }
    }

    #[test]
    fn validate_rejects_zero_m_min_and_inverted_m_range() {
        let mut bad = default_params();
        bad.m_min = 0;
        assert!(YoungConsensus::new(bad).is_err());

        let mut bad2 = default_params();
        bad2.m_min = 10;
        bad2.m_max = 5;
        assert!(YoungConsensus::new(bad2).is_err());
    }

    #[test]
    fn pre_step_computes_a_real_m_star_from_positions() {
        let hook = YoungConsensus::new(default_params()).unwrap();
        // A moderately sized cloud with real structure -- enough boids for m_star() to return
        // Some(_), not fall back.
        let entries: Vec<(Vec3, Vec3, Species)> = (0..20)
            .map(|i| {
                let t = i as f64;
                (
                    Vec3::new(t.sin() * 5.0, t.cos() * 5.0, (t * 0.7).sin() * 3.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Species::Prey,
                )
            })
            .collect();
        let boids = boids_with(&entries);
        hook.pre_step(&boids, 0);
        let cache = hook.cache.lock().unwrap();
        let m = cache.m_star.expect("expected a real computed m*, not None");
        assert!(
            (2..=12).contains(&m),
            "m*={m} outside the configured [2,12] range"
        );
    }

    #[test]
    fn pre_step_is_throttled_by_refresh_interval() {
        let hook = YoungConsensus::new(default_params()).unwrap();
        let entries: Vec<(Vec3, Vec3, Species)> = (0..20)
            .map(|i| {
                let t = i as f64;
                (
                    Vec3::new(t.sin() * 5.0, t.cos() * 5.0, 0.0),
                    Vec3::ZERO,
                    Species::Prey,
                )
            })
            .collect();
        let boids = boids_with(&entries);
        hook.pre_step(&boids, 0);
        let computed_at_0 = hook.cache.lock().unwrap().last_computed_step;
        assert_eq!(computed_at_0, Some(0));

        hook.pre_step(&boids, 5); // well within refresh_interval=20 -> should NOT recompute
        assert_eq!(hook.cache.lock().unwrap().last_computed_step, Some(0));

        hook.pre_step(&boids, 20); // exactly at the threshold -> should recompute
        assert_eq!(hook.cache.lock().unwrap().last_computed_step, Some(20));
    }

    #[test]
    fn predator_positions_are_excluded_from_the_m_star_computation() {
        let hook = YoungConsensus::new(default_params()).unwrap();
        let mut entries: Vec<(Vec3, Vec3, Species)> = (0..20)
            .map(|i| {
                let t = i as f64;
                (
                    Vec3::new(t.sin() * 5.0, t.cos() * 5.0, 0.0),
                    Vec3::ZERO,
                    Species::Prey,
                )
            })
            .collect();
        // A predator very far away must not perturb the prey-only H2 computation.
        entries.push((Vec3::new(10000.0, 0.0, 0.0), Vec3::ZERO, Species::Predator));
        let boids_with_predator = boids_with(&entries);
        hook.pre_step(&boids_with_predator, 0);
        let m_with_predator = hook.cache.lock().unwrap().m_star;

        let hook2 = YoungConsensus::new(default_params()).unwrap();
        entries.pop();
        let boids_without_predator = boids_with(&entries);
        hook2.pre_step(&boids_without_predator, 0);
        let m_without_predator = hook2.cache.lock().unwrap().m_star;

        assert_eq!(m_with_predator, m_without_predator);
    }

    #[test]
    fn desired_v_points_toward_the_blend_of_alignment_and_cohesion() {
        let hook = YoungConsensus::new(default_params()).unwrap();
        {
            let mut cache = hook.cache.lock().unwrap();
            cache.m_star = Some(2);
            cache.last_computed_step = Some(0);
        }
        let observer = Vec3::ZERO;
        // Two neighbours, both to the +x side and both moving +x -> alignment and cohesion
        // should both point roughly +x.
        let neighbors = vec![
            neighbor_from(
                observer,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                1,
            ),
            neighbor_from(
                observer,
                Vec3::new(2.0, 0.5, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                2,
            ),
        ];
        let domain = StubDomain;
        let p = core_params();
        let ctx = BoidCtx {
            index: 0,
            pos: observer,
            vel: Vec3::new(0.0, 1.0, 0.0), // currently moving +y, unrelated to the blend
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &p,
            domain: &domain,
            step_count: 0,
        };
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(0, 0, 0);
        let intent = hook.desired(ctx, &mut scratch, &mut rng);
        assert!(
            intent.desired_v.x > 0.0,
            "expected +x-dominant desired_v, got {:?}",
            intent.desired_v
        );
        assert!((intent.desired_v.len() - p.cruise_speed).abs() < 1e-9);
    }

    #[test]
    fn steric_repels_from_a_very_close_neighbour() {
        let mut params = default_params();
        params.steric_enabled = true;
        params.steric_radius = 5.0;
        let hook = YoungConsensus::new(params).unwrap();
        {
            let mut cache = hook.cache.lock().unwrap();
            cache.m_star = Some(1);
            cache.last_computed_step = Some(0);
        }
        let observer = Vec3::ZERO;
        let neighbors = vec![neighbor_from(
            observer,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            1,
        )];
        let domain = StubDomain;
        let p = core_params();
        let ctx = BoidCtx {
            index: 0,
            pos: observer,
            vel: Vec3::ZERO,
            species: Species::Prey,
            neighbors: &neighbors,
            core_params: &p,
            domain: &domain,
            step_count: 0,
        };
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(0, 0, 0);
        let intent = hook.desired(ctx, &mut scratch, &mut rng);
        assert!(
            intent.extra_force.x < 0.0,
            "expected steric repulsion away from the +x neighbour, got {:?}",
            intent.extra_force
        );
    }

    #[test]
    fn conforms_to_flocking_mode_contract() {
        murmur_conformance::flocking_mode(&YoungConsensus::new(default_params()).unwrap());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg.resolve_mode("young", &PluginParams::new()).unwrap();
        assert_eq!(mode.name(), "young");
    }
}
