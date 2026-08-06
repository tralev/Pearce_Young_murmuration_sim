//! `BoidStateMachine` — per-boid Normal/Isolated/Crowded/Threatened state + speed-cap
//! multiplier `StepHook` (design/02_plugins.md §5, roadmap.md Phase 18, pymurmur's
//! `physics/extensions/boid_state_machine.py`, ported by description — pymurmur's actual
//! source isn't reachable in this environment, the same blocker as every other pymurmur
//! cross-check this project has hit).
//!
//! **The plugin that motivated fixing two architectural gaps, both now genuinely exercised
//! rather than just plumbed through:**
//! - **G1** (roadmap.md §12): `StepHook::post_steer`'s `ctx.neighbors` used to be a hardcoded
//!   `&[]` — classifying Isolated/Crowded needs a real local-density signal, which needs real
//!   neighbour data. Fixed in `murmur_core::pipeline`'s `write_phase` by recomputing neighbours
//!   there via the composed `NeighborSelection`, with two disclosed limitations (stale spatial
//!   index cell membership, a same-step position-mixing effect from the loop's own sequential
//!   mutation) documented on that fix directly — acceptable for this hook's coarse density
//!   classification, not for anything needing exact physics.
//! - **G3** (roadmap.md §12): `SpeedModel::enforce` had no channel for any hook to influence
//!   per-boid speed enforcement at all. Fixed by adding `StepHook::speed_cap_multiplier()`
//!   (read once per boid, right after `post_steer`, combined across hooks via `min`) and a
//!   `cap_multiplier` parameter to `SpeedModel::enforce` itself.
//!
//! **Classification** (checked in priority order, per boid, each step):
//! 1. **Threatened**: own speed exceeds `threat_speed_factor * cruise_speed` — a boid moving
//!    unusually fast (a flight response), the readable-from-`BoidCtx`-alone proxy used here
//!    since this hook, standing alone, has no direct predator-position channel the way
//!    `murmur_predator_fsm` does (composing both is what "boid state" actually tracking a real
//!    threat would look like; this hook's Threatened state is the density-*independent*
//!    read of that, always computable).
//! 2. **Isolated**: at most `isolated_max` neighbours (cut off from the flock).
//! 3. **Crowded**: at least `crowded_min` neighbours (an overcrowded local region).
//! 4. **Normal**: none of the above.
//!
//! **The speed-cap multiplier only ever narrows for `Crowded`** (`crowded_speed_cap`, default
//! `0.7`) — `Normal`/`Isolated`/`Threatened` all return `None` (no opinion), deliberately: this
//! hook's own scope is crowding/congestion easing (pymurmur's own `boid_state_speed_mult`
//! naming), not threat response — `SpeedModel::enforce`'s `cap_multiplier` contract only ever
//! *narrows* a ceiling (roadmap.md §12's G3 fix), so there's no way for this hook to *boost* a
//! threatened boid's speed through this channel even if that were desired; that kind of
//! response already belongs to `murmur_predator_fsm`'s own force bundle and `BandSpeed`'s own
//! `predator_speed_factor`, not duplicated here.
//!
//! **Per-boid state: the plugin-owned side-column pattern** (design/01_core.md §2), same
//! `Mutex`-guarded `HashMap<u32, BoidState>` shape `murmur_angle`/`murmur_spin_wave` already
//! established — `Mutex`, not a plain `HashMap`, purely to satisfy `StepHook: Send + Sync`;
//! `write_phase` itself is fully sequential, so there's no genuine concurrent access to guard
//! against, unlike `murmur_spin_wave`'s real parallel-read-phase need.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{BoidCtx, ConfigError, PluginParams, Registry, StepHook, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoidState {
    Normal,
    Isolated,
    Crowded,
    Threatened,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoidStateMachineParams {
    pub isolated_max: u32,
    pub crowded_min: u32,
    pub threat_speed_factor: f64,
    pub crowded_speed_cap: f64,
}

impl BoidStateMachineParams {
    pub fn builder() -> BoidStateMachineParamsBuilder {
        BoidStateMachineParamsBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoidStateMachineParamsBuilder {
    isolated_max: u32,
    crowded_min: u32,
    threat_speed_factor: f64,
    crowded_speed_cap: f64,
}

impl Default for BoidStateMachineParamsBuilder {
    fn default() -> Self {
        BoidStateMachineParamsBuilder {
            isolated_max: 1,
            crowded_min: 12,
            threat_speed_factor: 1.5,
            crowded_speed_cap: 0.7,
        }
    }
}

impl BoidStateMachineParamsBuilder {
    pub fn isolated_max(mut self, v: u32) -> Self {
        self.isolated_max = v;
        self
    }
    pub fn crowded_min(mut self, v: u32) -> Self {
        self.crowded_min = v;
        self
    }
    pub fn threat_speed_factor(mut self, v: f64) -> Self {
        self.threat_speed_factor = v;
        self
    }
    pub fn crowded_speed_cap(mut self, v: f64) -> Self {
        self.crowded_speed_cap = v;
        self
    }

    pub fn build(self) -> Result<BoidStateMachineParams, ConfigError> {
        if self.crowded_min <= self.isolated_max {
            return Err(ConfigError::InvalidParam {
                field: "crowded_min",
                reason: "must be greater than isolated_max".into(),
            });
        }
        if !(self.threat_speed_factor.is_finite() && self.threat_speed_factor > 0.0) {
            return Err(ConfigError::InvalidParam {
                field: "threat_speed_factor",
                reason: "must be finite and > 0".into(),
            });
        }
        if !(self.crowded_speed_cap.is_finite()
            && self.crowded_speed_cap > 0.0
            && self.crowded_speed_cap <= 1.0)
        {
            return Err(ConfigError::InvalidParam {
                field: "crowded_speed_cap",
                reason: "must be finite and in (0, 1] -- a cap only ever narrows".into(),
            });
        }
        Ok(BoidStateMachineParams {
            isolated_max: self.isolated_max,
            crowded_min: self.crowded_min,
            threat_speed_factor: self.threat_speed_factor,
            crowded_speed_cap: self.crowded_speed_cap,
        })
    }
}

pub struct BoidStateMachine {
    pub params: BoidStateMachineParams,
    state: Mutex<HashMap<u32, BoidState>>,
}

impl BoidStateMachine {
    pub fn new(params: BoidStateMachineParams) -> Self {
        BoidStateMachine {
            params,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// This boid's most recently classified state, if it's ever been seen by `post_steer` —
    /// not part of the `StepHook` trait (no other occupant needs it), but real, checkable state
    /// this plugin's own tests (and any future viz consumer) need direct read access to.
    pub fn state_of(&self, index: u32) -> Option<BoidState> {
        self.state.lock().unwrap().get(&index).copied()
    }

    fn classify(&self, ctx: &BoidCtx<'_>) -> BoidState {
        let threat_speed = ctx.core_params.cruise_speed * self.params.threat_speed_factor;
        if ctx.vel.len() > threat_speed {
            return BoidState::Threatened;
        }
        let neighbor_count = ctx.neighbors.len() as u32;
        if neighbor_count <= self.params.isolated_max {
            BoidState::Isolated
        } else if neighbor_count >= self.params.crowded_min {
            BoidState::Crowded
        } else {
            BoidState::Normal
        }
    }
}

impl StepHook for BoidStateMachine {
    fn post_steer(&self, ctx: BoidCtx<'_>, _acc: &mut Vec3, _rng: &mut murmur_core::Rng) {
        let state = self.classify(&ctx);
        self.state.lock().unwrap().insert(ctx.index, state);
    }

    fn speed_cap_multiplier(&self, index: u32) -> Option<f64> {
        match self.state_of(index) {
            Some(BoidState::Crowded) => Some(self.params.crowded_speed_cap),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        "boid_state_machine"
    }
}

/// Registers `BoidStateMachine` under the name `"boid_state_machine"`, reading each of
/// `BoidStateMachineParams`'s fields from `PluginParams`. The factory type can't return
/// `Result` (design/02_plugins.md §1), so a malformed override falls back to the default rather
/// than panicking — same pattern as every other plugin here.
pub fn register(r: &mut Registry) {
    r.register_step_hook("boid_state_machine", |p: &PluginParams| {
        let d = BoidStateMachineParamsBuilder::default();
        let params = BoidStateMachineParams::builder()
            .isolated_max(p.get_or("isolated_max", d.isolated_max as f64) as u32)
            .crowded_min(p.get_or("crowded_min", d.crowded_min as f64) as u32)
            .threat_speed_factor(p.get_or("threat_speed_factor", d.threat_speed_factor))
            .crowded_speed_cap(p.get_or("crowded_speed_cap", d.crowded_speed_cap))
            .build()
            .unwrap_or_else(|_| {
                BoidStateMachineParams::builder()
                    .build()
                    .expect("defaults are valid")
            });
        Box::new(BoidStateMachine::new(params)) as Box<dyn StepHook>
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

    fn neighbors(n: usize) -> Vec<Neighbor> {
        (0..n)
            .map(|i| Neighbor {
                index: i as u32 + 1,
                distance: 1.0,
                direction: Vec3::new(1.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
            })
            .collect()
    }

    fn test_rng() -> murmur_core::Rng {
        murmur_core::rng::for_boid(1, 2, 3)
    }

    fn ctx<'a>(
        vel: Vec3,
        neighbors: &'a [Neighbor],
        params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index: 0,
            pos: Vec3::ZERO,
            vel,
            species: Species::Prey,
            neighbors,
            core_params: params,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        let mut hook = BoidStateMachine::new(BoidStateMachineParams::builder().build().unwrap());
        murmur_conformance::step_hook(&mut hook);
    }

    #[test]
    fn a_boid_with_no_neighbours_is_classified_isolated() {
        let hook = BoidStateMachine::new(BoidStateMachineParams::builder().build().unwrap());
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(0);
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(Vec3::new(0.5, 0.0, 0.0), &n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        );
        assert_eq!(hook.state_of(0), Some(BoidState::Isolated));
    }

    #[test]
    fn a_boid_with_many_neighbours_is_classified_crowded() {
        let hook = BoidStateMachine::new(
            BoidStateMachineParams::builder()
                .isolated_max(1)
                .crowded_min(5)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(10);
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(Vec3::new(0.5, 0.0, 0.0), &n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        );
        assert_eq!(hook.state_of(0), Some(BoidState::Crowded));
    }

    #[test]
    fn a_boid_with_a_middling_neighbour_count_is_classified_normal() {
        let hook = BoidStateMachine::new(
            BoidStateMachineParams::builder()
                .isolated_max(1)
                .crowded_min(10)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let n = neighbors(5);
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(Vec3::new(0.5, 0.0, 0.0), &n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        );
        assert_eq!(hook.state_of(0), Some(BoidState::Normal));
    }

    #[test]
    fn a_fast_moving_boid_is_classified_threatened_regardless_of_neighbour_count() {
        let hook = BoidStateMachine::new(
            BoidStateMachineParams::builder()
                .threat_speed_factor(1.5)
                .build()
                .unwrap(),
        );
        let params = core_params(); // cruise_speed = 1.0
        let domain = StubDomain;
        let n = neighbors(5); // would otherwise be Normal
        let mut acc = Vec3::ZERO;
        hook.post_steer(
            ctx(Vec3::new(2.0, 0.0, 0.0), &n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        ); // speed=2.0 > 1.5
        assert_eq!(hook.state_of(0), Some(BoidState::Threatened));
    }

    #[test]
    fn only_crowded_produces_a_speed_cap_multiplier() {
        let hook = BoidStateMachine::new(
            BoidStateMachineParams::builder()
                .isolated_max(1)
                .crowded_min(5)
                .crowded_speed_cap(0.6)
                .build()
                .unwrap(),
        );
        let params = core_params();
        let domain = StubDomain;
        let mut acc = Vec3::ZERO;

        let isolated_n = neighbors(0);
        hook.post_steer(
            ctx(Vec3::new(0.5, 0.0, 0.0), &isolated_n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        );
        assert_eq!(
            hook.speed_cap_multiplier(0),
            None,
            "Isolated must not tighten"
        );

        let crowded_n = neighbors(10);
        hook.post_steer(
            ctx(Vec3::new(0.5, 0.0, 0.0), &crowded_n, &params, &domain),
            &mut acc,
            &mut test_rng(),
        );
        assert_eq!(
            hook.speed_cap_multiplier(0),
            Some(0.6),
            "Crowded must tighten to crowded_speed_cap"
        );
    }

    #[test]
    fn state_of_an_unseen_boid_is_none() {
        let hook = BoidStateMachine::new(BoidStateMachineParams::builder().build().unwrap());
        assert_eq!(hook.state_of(42), None);
        assert_eq!(hook.speed_cap_multiplier(42), None);
    }

    #[test]
    fn builder_rejects_crowded_min_not_greater_than_isolated_max() {
        assert!(BoidStateMachineParams::builder()
            .isolated_max(5)
            .crowded_min(5)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_a_crowded_speed_cap_above_one() {
        assert!(BoidStateMachineParams::builder()
            .crowded_speed_cap(1.5)
            .build()
            .is_err());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("boid_state_machine", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "boid_state_machine");
    }

    #[test]
    fn a_malformed_override_falls_back_to_defaults_instead_of_panicking() {
        let mut reg = Registry::new();
        register(&mut reg);
        let bad = PluginParams::new().with("crowded_speed_cap", 5.0);
        let hook = reg.resolve_step_hook("boid_state_machine", &bad).unwrap();
        assert_eq!(hook.name(), "boid_state_machine");
    }
}
