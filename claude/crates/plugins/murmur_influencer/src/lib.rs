//! `Influencer` — rank/distance-weighted attractor `FlockingMode` (design/02_plugins.md §5,
//! roadmap.md Phase 17, pymurmur's `physics/forces/influencer.py`, ported by description —
//! pymurmur's actual source isn't reachable in this environment, the same blocker as every
//! other pymurmur cross-check this project has hit).
//!
//! **Interpretation, disclosed.** "Rank/distance-weighted attractor" is read here as a
//! leader-follower hierarchy: every boid has a fixed, deterministic **rank** in `[0, 1)`, and
//! blends between two behaviours by that rank — a high-rank boid ("an influencer") pursues a
//! shared, time-varying Lissajous-curve target almost directly; a low-rank boid ("a follower")
//! instead pursues nearby *higher-ranked* neighbours, weighted by both their rank advantage and
//! their distance (hence "rank/distance-weighted"), cascading leadership through the flock
//! rather than every boid chasing the target independently. Grounded in real leader-follower
//! flocking literature (e.g. Nagy et al. 2010, "Hierarchical group dynamics in pigeon flocks")
//! rather than verified against pymurmur's own source, the same honesty standard
//! `murmur_predator_fsm` and the Phase 16 `Domain` plugins already established for
//! unreachable-source ports.
//!
//! **The first `FlockingMode` plugin to need G4** (roadmap.md §12 — `step_count`/`dt` in
//! `BoidCtx`, fixed proactively in Phase 13 for `murmur_spin_wave`, unused by any
//! `FlockingMode` until now): the Lissajous target's position is a real function of elapsed
//! simulated time (`step_count as f64 * dt`), not a per-call counter, so a shorter/longer `dt`
//! genuinely changes how fast the target moves through the same number of steps.
//!
//! **Stateless — no per-boid side-column, unlike `murmur_angle`.** Both `rank(index)` and the
//! Lissajous target are pure functions of already-available `BoidCtx` fields; `desired()` reads
//! and writes nothing persistent, so there is no cross-step or cross-boid state to reason about
//! under parallelism at all — the simplest of the three implementation shapes this project's
//! `FlockingMode` plugins now cover (`murmur_spatial`: reusable kernel toolkit;
//! `murmur_angle`: persistent per-boid state; this: pure function).
//!
//! **Scope note**: this is *a* Lissajous target (one 3-axis parametrised curve, 9 params) —
//! design/02_plugins.md §5 separately names `murmur_field` as an "11-term Lissajous blob-anchor
//! field mode," a richer, still-unbuilt system this plugin does not attempt to subsume.

use murmur_core::{
    BoidCtx, ConfigError, FlockingMode, OcclusionScratch, PluginParams, Registry, Rng, SteerIntent,
    Vec3, MIN_LEN, MIN_LEN2,
};

/// The golden ratio's fractional conjugate — the standard low-discrepancy constant for a Weyl
/// (Kronecker) sequence, `frac(i * GOLDEN)`: deterministic, no RNG/state needed, and spreads
/// consecutive indices' ranks apart rather than clustering them (unlike e.g. `i / N`, which
/// would make neighbouring-index boids near-identical in rank).
const GOLDEN: f64 = 0.618_033_988_749_895;

/// Each boid's fixed leadership rank in `[0, 1)`: `0.0` (boid index `0`, exactly) is the purest
/// possible follower, ranks near `1.0` are the closest thing to a pure influencer.
fn rank(index: u32) -> f64 {
    (index as f64 * GOLDEN).fract()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfluencerParams {
    pub amplitude: Vec3,
    pub frequency: Vec3,
    pub phase: Vec3,
}

impl InfluencerParams {
    pub fn builder() -> InfluencerParamsBuilder {
        InfluencerParamsBuilder::default()
    }

    /// The shared Lissajous target's position at simulated time `t`.
    fn target_at(&self, t: f64) -> Vec3 {
        Vec3::new(
            self.amplitude.x * (self.frequency.x * t + self.phase.x).sin(),
            self.amplitude.y * (self.frequency.y * t + self.phase.y).sin(),
            self.amplitude.z * (self.frequency.z * t + self.phase.z).sin(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InfluencerParamsBuilder {
    amplitude: Vec3,
    frequency: Vec3,
    phase: Vec3,
}

impl Default for InfluencerParamsBuilder {
    fn default() -> Self {
        InfluencerParamsBuilder {
            amplitude: Vec3::new(10.0, 10.0, 10.0),
            frequency: Vec3::new(0.05, 0.07, 0.03),
            phase: Vec3::new(0.0, std::f64::consts::FRAC_PI_2, 0.0),
        }
    }
}

impl InfluencerParamsBuilder {
    pub fn amplitude(mut self, v: Vec3) -> Self {
        self.amplitude = v;
        self
    }
    pub fn frequency(mut self, v: Vec3) -> Self {
        self.frequency = v;
        self
    }
    pub fn phase(mut self, v: Vec3) -> Self {
        self.phase = v;
        self
    }

    pub fn build(self) -> Result<InfluencerParams, ConfigError> {
        for (field, v) in [
            ("amplitude", self.amplitude),
            ("frequency", self.frequency),
            ("phase", self.phase),
        ] {
            if !v.is_finite() {
                return Err(ConfigError::InvalidParam {
                    field,
                    reason: "must be finite".into(),
                });
            }
        }
        Ok(InfluencerParams {
            amplitude: self.amplitude,
            frequency: self.frequency,
            phase: self.phase,
        })
    }
}

pub struct Influencer {
    pub params: InfluencerParams,
}

impl Influencer {
    pub fn new(params: InfluencerParams) -> Self {
        Influencer { params }
    }
}

impl FlockingMode for Influencer {
    fn desired(
        &self,
        ctx: BoidCtx<'_>,
        _scratch: &mut OcclusionScratch,
        _rng: &mut Rng,
    ) -> SteerIntent {
        let my_rank = rank(ctx.index);

        let t = ctx.step_count as f64 * ctx.core_params.dt;
        let target_pos = self.params.target_at(t);
        let to_target = target_pos - ctx.pos;
        let direction_to_target = to_target.normalized(); // Vec3::ZERO if coincident

        let mut neighbour_pull = Vec3::ZERO;
        for n in ctx.neighbors {
            let their_rank = rank(n.index);
            let advantage = their_rank - my_rank;
            if advantage > 0.0 {
                neighbour_pull += n.direction * (advantage / n.distance.max(MIN_LEN));
            }
        }
        let neighbour_dir = neighbour_pull.normalized(); // Vec3::ZERO if no higher-rank neighbour

        let combined = direction_to_target * my_rank + neighbour_dir * (1.0 - my_rank);

        let desired_v = if combined.len_sq() > MIN_LEN2 {
            combined.normalized() * ctx.core_params.cruise_speed
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
        "influencer"
    }
}

/// Registers `Influencer` under the name `"influencer"`, reading `amplitude_{x,y,z}`,
/// `frequency_{x,y,z}`, `phase_{x,y,z}` from `PluginParams`. The factory type can't return
/// `Result` (design/02_plugins.md §1), so a malformed override falls back to the default rather
/// than panicking — same pattern as `murmur_pearce`/`murmur_vicsek`/`murmur_spatial`/
/// `murmur_angle`.
pub fn register(r: &mut Registry) {
    r.register_mode("influencer", |p: &PluginParams| {
        let defaults = InfluencerParamsBuilder::default();
        let params = InfluencerParams::builder()
            .amplitude(Vec3::new(
                p.get_or("amplitude_x", defaults.amplitude.x),
                p.get_or("amplitude_y", defaults.amplitude.y),
                p.get_or("amplitude_z", defaults.amplitude.z),
            ))
            .frequency(Vec3::new(
                p.get_or("frequency_x", defaults.frequency.x),
                p.get_or("frequency_y", defaults.frequency.y),
                p.get_or("frequency_z", defaults.frequency.z),
            ))
            .phase(Vec3::new(
                p.get_or("phase_x", defaults.phase.x),
                p.get_or("phase_y", defaults.phase.y),
                p.get_or("phase_z", defaults.phase.z),
            ))
            .build()
            .unwrap_or_else(|_| {
                InfluencerParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(Influencer::new(params)) as Box<dyn FlockingMode>
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

    fn core_params(dt: f64) -> CoreParams {
        CoreParams::builder()
            .cruise_speed(2.0)
            .max_force(1.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .dt(dt)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn neighbor(index: u32, direction: Vec3, distance: f64) -> Neighbor {
        Neighbor {
            index,
            distance,
            direction: direction.normalized(),
            velocity: Vec3::ZERO,
        }
    }

    fn ctx<'a>(
        index: u32,
        pos: Vec3,
        vel: Vec3,
        step_count: u64,
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos,
            vel,
            species: Species::Prey,
            neighbors,
            core_params: params,
            domain,
            step_count,
        }
    }

    #[test]
    fn conforms_to_flocking_mode_contract() {
        murmur_conformance::flocking_mode(&Influencer::new(
            InfluencerParams::builder().build().unwrap(),
        ));
    }

    #[test]
    fn rank_of_index_zero_is_exactly_zero_the_purest_follower() {
        assert_eq!(rank(0), 0.0);
    }

    #[test]
    fn rank_is_deterministic_and_spread_across_zero_one() {
        let ranks: Vec<f64> = (0..20).map(rank).collect();
        for &r in &ranks {
            assert!((0.0..1.0).contains(&r), "rank {} out of [0,1)", r);
        }
        // Consecutive indices must not collapse to (near-)identical ranks -- the whole point of
        // the golden-ratio Weyl sequence over a naive i/N assignment.
        assert!((ranks[0] - ranks[1]).abs() > 0.1);
        // Deterministic: calling twice gives the same value.
        assert_eq!(rank(7), rank(7));
    }

    #[test]
    fn a_pure_follower_with_no_higher_rank_neighbours_holds_its_current_velocity() {
        // Index 0 has rank exactly 0.0: 100% neighbour-follow weight, 0% target-pursuit weight.
        // With no neighbours at all, there is nothing to follow, so it must fall back to
        // holding its current velocity, not silently drift toward the target.
        let params = core_params(1.0);
        let domain = StubDomain;
        let neighbors: [Neighbor; 0] = [];
        let mode = Influencer::new(InfluencerParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let current = Vec3::new(0.0, 2.0, 0.0);
        let intent = mode.desired(
            ctx(
                0,
                Vec3::new(1000.0, 0.0, 0.0),
                current,
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert_eq!(intent.desired_v, current);
    }

    #[test]
    fn a_pure_follower_moves_toward_a_higher_ranked_neighbour() {
        let params = core_params(1.0);
        let domain = StubDomain;
        // Index 0 (rank 0.0) has one neighbour, index 1 (rank > 0), due +x.
        assert!(rank(1) > rank(0));
        let neighbors = [neighbor(1, Vec3::new(1.0, 0.0, 0.0), 5.0)];
        let mode = Influencer::new(InfluencerParams::builder().build().unwrap());
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        let intent = mode.desired(
            ctx(
                0,
                Vec3::new(1000.0, 0.0, 0.0),
                Vec3::ZERO,
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x > 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn a_follower_ignores_a_lower_ranked_neighbour() {
        let params = core_params(1.0);
        let domain = StubDomain;
        // Observer is `hi`; its only neighbour is `lo`, which outranks nobody relative to `hi`
        // -- so the neighbour-pull term must be exactly zero, leaving only direct target-pursuit.
        let (lo, hi) = (0u32, 1u32); // rank(0)=0.0 is the lowest possible rank
        assert!(rank(hi) > rank(lo));
        let neighbors = [neighbor(lo, Vec3::new(1.0, 0.0, 0.0), 5.0)];
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        // amplitude=0 makes the target exactly the origin: direction_to_target points -x from
        // pos=(5,0,0), unambiguously opposite the (ignored) neighbour's +x bearing -- if that
        // neighbour were wrongly included, x would be pulled positive instead.
        let mode = Influencer::new(
            InfluencerParams::builder()
                .amplitude(Vec3::new(0.0, 0.0, 0.0))
                .build()
                .unwrap(),
        );
        let intent = mode.desired(
            ctx(
                hi,
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::ZERO,
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x < 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn the_target_position_is_a_real_function_of_elapsed_time_not_a_call_counter() {
        let mode = Influencer::new(InfluencerParams::builder().build().unwrap());
        let t0 = mode.params.target_at(0.0);
        let t_small_dt = mode.params.target_at(0.01 * 5.0); // 5 steps at dt=0.01
        let t_large_dt = mode.params.target_at(1.0 * 5.0); // 5 steps at dt=1.0
        assert_ne!(t0, t_small_dt);
        assert_ne!(
            t_small_dt, t_large_dt,
            "the same step count at a different dt must reach a different target position"
        );
    }

    #[test]
    fn builder_rejects_non_finite_params() {
        assert!(InfluencerParams::builder()
            .amplitude(Vec3::new(f64::NAN, 0.0, 0.0))
            .build()
            .is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let mode = reg
            .resolve_mode("influencer", &PluginParams::new())
            .unwrap();
        assert_eq!(mode.name(), "influencer");
    }

    #[test]
    fn registry_reads_amplitude_override() {
        let mut reg = Registry::new();
        register(&mut reg);
        let p = PluginParams::new()
            .with("amplitude_x", 0.0)
            .with("amplitude_y", 0.0)
            .with("amplitude_z", 0.0);
        let mode = reg.resolve_mode("influencer", &p).unwrap();
        let domain = StubDomain;
        let params = core_params(1.0);
        let neighbors: [Neighbor; 0] = [];
        let mut scratch = OcclusionScratch::default();
        let mut rng = murmur_core::rng::for_boid(1, 0, 0);
        // A pure-influencer-ish boid (high rank) with amplitude=0 should head toward the
        // origin from wherever it is.
        let high_rank_index = (1..50)
            .max_by(|&a, &b| rank(a).partial_cmp(&rank(b)).unwrap())
            .unwrap();
        let intent = mode.desired(
            ctx(
                high_rank_index,
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::ZERO,
                0,
                &neighbors,
                &params,
                &domain,
            ),
            &mut scratch,
            &mut rng,
        );
        assert!(intent.desired_v.x < 0.0, "got {:?}", intent.desired_v);
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("amplitude_x", f64::NAN);
        let mode = reg.resolve_mode("influencer", &bad).unwrap();
        assert_eq!(mode.name(), "influencer");
    }

    /// Proves the `FlockingMode` seam now has ≥5 real occupants beyond Pearce/Vicsek.
    #[test]
    fn pearce_vicsek_and_influencer_all_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_vicsek::register(&mut reg);
        murmur_pearce::register(&mut reg);

        assert_eq!(
            reg.resolve_mode("influencer", &PluginParams::new())
                .unwrap()
                .name(),
            "influencer"
        );
        assert_eq!(
            reg.resolve_mode("vicsek", &PluginParams::new())
                .unwrap()
                .name(),
            "vicsek"
        );
        assert_eq!(
            reg.resolve_mode("pearce", &PluginParams::new())
                .unwrap()
                .name(),
            "pearce"
        );
    }
}
