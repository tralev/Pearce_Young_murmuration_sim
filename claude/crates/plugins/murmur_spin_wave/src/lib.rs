//! `SpinWaveResponse` — Attanasi et al. 2014's "behavioural inertia" `SteeringModifier`
//! (design/02_plugins.md §5, roadmap.md Phase 13). **2D-only**, per the design's own
//! pre-approved fallback.
//!
//! **The gating check (2026-08-05).** The design doc requires checking whether the paper's SI
//! Appendix G (which "reportedly addresses 3D") is obtainable before scheduling a 3D
//! implementation. The local copy, `sci/emss-59192.pdf`, is the *Europe PMC author-manuscript*
//! version — its own text says "Refer to Web version on PubMed Central for supplementary
//! material" and separately references "SI-Appendix A" through "SI-Appendix G" as external
//! material not included in this 14-page copy. This session has no web-fetch tool, so SI
//! Appendix G is confirmed **not obtainable in this environment** — not assumed, checked.
//! Per the design's own documented contingency: ship the 2D version, defer full 3D
//! indefinitely rather than block the plugin on it.
//!
//! **The physics (from the paper's main text, eq. 1–10).** Alignment is normally instant
//! (`v_i(t+1) = v_i(t) + JΣv_j(t)`) — what `InstantResponse` does. The paper shows this implies
//! *diffusive* (`x ~ √t`) information transport, contradicting real flocks' measured *linear,
//! undamped* turn propagation (`x = c_s·t`, `c_s ∈ [20,40] m/s`). Adding the neighbours'
//! conjugate-momentum "spin" `s_z` (behavioural inertia) as a genuine dynamical variable fixes
//! this: the heading phase φ and spin `s_z` obey a canonical wave-equation pair (eq. 6),
//! `∂φ/∂t = s_z/χ`, `∂s_z/∂t = J∇²φ` (continuum) — discretely, `∂s_z_i/∂t = J·Σ_{j∈N(i)}(φ_j−φ_i)`
//! (the pre-continuum-limit form of eq. 3/6's Hamiltonian, summed over neighbours, not
//! averaged — matching the paper's own un-normalized sum, not an arbitrary choice here).
//!
//! **Generalizing beyond the paper's own alignment-only toy Hamiltonian.** The paper's `H`
//! only contains the neighbour-coupling term because its point is to explain propagation
//! under *pure* alignment. To compose with *any* `FlockingMode` (Pearce, Vicsek, ...) as
//! design/00_overview.md requires, this implementation adds a second driving term pulling φ
//! toward the active mode's own `desired_v` angle — the mode already did the "what do my
//! neighbours/environment want" computation; this modifier's job is only to make the
//! *response* to that computation properly inertial (wave-propagating) instead of instant.
//! `ds_z_i/dt = coupling·Σ_{j∈N(i)}(φ_j−φ_i) + drive·(φ_target,i − φ_i)`. Setting `drive` to
//! `0.0` recovers the paper's original alignment-only Hamiltonian exactly (given an
//! alignment-only `FlockingMode` upstream).
//!
//! **Honest simplifications, beyond "2D only":**
//! - **Fixed reference plane, not a PCA-computed dominant motion plane.** The design doc's
//!   fallback description says "e.g. the flock's dominant motion plane, computed from PCA" —
//!   an example, not a requirement. Computing that needs global per-step aggregate access
//!   (all boids' positions) that `SteeringModifier::respond()` — called per-boid, in the
//!   parallel read phase — doesn't have (no `pre_step`-equivalent on this trait, unlike
//!   `StepHook`). A fixed, plugin-param-configurable plane (default the XY plane) is the
//!   honestly-scoped alternative; only the in-plane velocity component is corrected, any
//!   out-of-plane component is left untouched by this modifier.
//! - **Per-boid state lives in a lazily-growing, double-buffered `Mutex<Inner>`
//!   (`committed`/`pending` `HashMap`s), not a `Vec` sized at construction.**
//!   `SteeringModifier`'s registry factory signature (design/02 §1) receives only
//!   `&PluginParams`, never `boid_count`/`CoreParams` — there is no seam to size a
//!   fixed-capacity per-boid column at construction time (a real, if narrow, architectural gap
//!   this plugin ran into). A lazy map sidesteps it entirely, and as a side effect handles
//!   `Command::Reset`/`AddPredator` changing the live boid count mid-run (batch.rs, roadmap.md
//!   Phase 10) correctly — a `Vec` sized once at construction would not.
//!   **The double-buffering is load-bearing, not decorative**: an earlier single-`RwLock<HashMap>`
//!   version (read your own/neighbours' entries, then immediately write your own back into the
//!   *same* map) was thread-count-*dependent* — caught by
//!   `tests/pearce_integration.rs::state_hash_is_identical_across_1_4_and_8_rayon_threads`,
//!   which hung nothing and paniced on nothing, just silently produced different physics
//!   depending on rayon's scheduling, because which neighbours' values were "this step's
//!   already-updated" vs. "last step's" depended on thread interleaving. `ctx.step_count`
//!   (G4's fix, added specifically because this bug needed it) marks step boundaries: the
//!   first `respond()` call to observe a new `step_count` commits `pending` into `committed`
//!   (a frozen snapshot every boid in that step reads from, regardless of execution order);
//!   every write goes into `pending`, invisible to sibling calls until the *next* step's
//!   commit. One `Mutex` around both maps, not a finer-grained scheme — correctness over
//!   parallel throughput is the accepted tradeoff for this Track C seam-diversity plugin (no
//!   performance target applies to it; that's Phase 20's concern, for a different set of
//!   plugins).
//! - **Not validated against `c_s ∈ [20,40] m/s` here.** That's an empirical claim about real
//!   starling flocks at specific physical scales — reproducing it meaningfully needs Phase
//!   14's calibration pass, and is explicitly `information_transfer.py`'s job in Phase 15 (the
//!   design doc names that analysis module as this plugin's actual acceptance test). What's
//!   tested here is mechanism correctness: neighbour coupling actually propagates phase
//!   differences, the drive term actually pulls toward the target, state persists and composes
//!   safely, and the output stays finite.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{clamp_len, BoidCtx, PluginParams, Registry, SteeringModifier, Vec3, MIN_LEN};

#[derive(Clone, Copy)]
struct SpinState {
    /// Heading phase within the configured plane (radians).
    phi: f64,
    /// Spin / curvature — the momentum canonically conjugate to `phi` (Attanasi et al.'s `s_z`).
    s_z: f64,
}

/// A stable orthonormal in-plane basis (u, v) for `normal` — same construction as
/// `murmur_core::metrics::external_opacity`'s viewing-plane basis (not imported: a small,
/// self-contained utility, not worth exposing from core for one caller).
fn basis_from_normal(normal: Vec3) -> (Vec3, Vec3) {
    let axis = normal.normalized();
    let axis = if axis == Vec3::ZERO {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        axis
    };
    let seed = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(seed).normalized();
    let v = axis.cross(u);
    (u, v)
}

pub struct SpinWaveResponse {
    plane_u: Vec3,
    plane_v: Vec3,
    /// J — neighbour-coupling stiffness (eq. 5/6).
    coupling: f64,
    /// Pulls φ toward the active `FlockingMode`'s own `desired_v` angle — this crate's
    /// generalization beyond the paper's alignment-only Hamiltonian (see module doc). `0.0`
    /// recovers the paper's original model exactly.
    drive: f64,
    /// χ — behavioural inertia/resistance (eq. 5/6); guarded away from zero (division).
    chi: f64,
    state: Mutex<Inner>,
}

/// Double-buffered per-boid state, committed on a step boundary (detected via
/// `BoidCtx::step_count`, G4's fix — see module doc's "honest simplifications" section for why
/// a plain single map is *not* correct here). `committed` is the frozen snapshot every boid
/// reads from during the current step; `pending` accumulates this step's writes and becomes
/// the *next* step's `committed` map on the first `respond()` call that observes a new
/// `step_count`. Everything lives behind one `Mutex`, not a finer-grained scheme — correctness
/// over parallel throughput is the accepted tradeoff for this Track C seam-diversity plugin
/// (no performance target applies to it; see module doc).
struct Inner {
    current_step: Option<u64>,
    committed: HashMap<u32, SpinState>,
    pending: HashMap<u32, SpinState>,
}

impl Inner {
    fn empty() -> Self {
        Inner {
            current_step: None,
            committed: HashMap::new(),
            pending: HashMap::new(),
        }
    }
}

impl SpinWaveResponse {
    fn phi_of(&self, v: Vec3) -> f64 {
        v.dot(self.plane_v).atan2(v.dot(self.plane_u))
    }
}

impl SteeringModifier for SpinWaveResponse {
    fn respond(&self, ctx: BoidCtx<'_>, desired_v: Vec3, current_vel: Vec3) -> Vec3 {
        let dt = ctx.core_params.dt;
        let target_phi = self.phi_of(desired_v);

        let mut inner = self.state.lock().unwrap();
        // First call observing a new step: commit last step's writes into this step's
        // read-only snapshot. Whichever boid (from whichever thread) happens to acquire the
        // lock first for this step does the swap; every other boid this step sees
        // `current_step == Some(ctx.step_count)` already and skips straight to reading —
        // so every boid within one step reads the *same* frozen snapshot, regardless of
        // execution order (the property the earlier single-map version lacked).
        if inner.current_step != Some(ctx.step_count) {
            inner.committed = std::mem::take(&mut inner.pending);
            inner.current_step = Some(ctx.step_count);
        }

        let self_state = inner
            .committed
            .get(&ctx.index)
            .copied()
            .unwrap_or(SpinState {
                // First time this boid is seen (the simulation's very first step, or a boid just
                // spawned via `Command::AddPredator`) — start from its current heading, zero spin.
                phi: self.phi_of(current_vel),
                s_z: 0.0,
            });

        let coupling_sum: f64 = ctx
            .neighbors
            .iter()
            .filter_map(|n| inner.committed.get(&n.index))
            .map(|neighbor| neighbor.phi - self_state.phi)
            .sum();

        // ∂s_z/∂t = J·Σ_neighbours(φ_j − φ_i) + drive·(φ_target − φ_i) — see module doc.
        let ds_z = self.coupling * coupling_sum + self.drive * (target_phi - self_state.phi);
        let new_s_z = self_state.s_z + dt * ds_z;
        // ∂φ/∂t = s_z/χ, semi-implicit (uses the just-updated s_z — more stable for an
        // oscillatory system than a fully explicit Euler step).
        let new_phi = self_state.phi + dt * new_s_z / self.chi;

        inner.pending.insert(
            ctx.index,
            SpinState {
                phi: new_phi,
                s_z: new_s_z,
            },
        );
        drop(inner);

        let speed = current_vel.len().max(desired_v.len());
        let target_inplane = (self.plane_u * new_phi.cos() + self.plane_v * new_phi.sin()) * speed;
        let current_inplane = self.plane_u * current_vel.dot(self.plane_u)
            + self.plane_v * current_vel.dot(self.plane_v);
        // Only the in-plane component is corrected — any out-of-plane component of
        // current_vel is left alone (module doc's "2D only" scope, honestly bounded).
        clamp_len(target_inplane - current_inplane, ctx.core_params.max_force)
    }

    fn name(&self) -> &'static str {
        "spin_wave"
    }
}

/// Registers `SpinWaveResponse` under the name `"spin_wave"`. Plugin params: `plane_normal_x`/
/// `_y`/`_z` (default `(0,0,1)`, the XY plane), `coupling` (J, default `1.0`), `drive` (default
/// `1.0`), `chi` (χ, default `1.0`, floored at `MIN_LEN` to guard the `s_z/χ` division).
pub fn register(r: &mut Registry) {
    r.register_modifier("spin_wave", |p: &PluginParams| {
        let normal = Vec3::new(
            p.get_or("plane_normal_x", 0.0),
            p.get_or("plane_normal_y", 0.0),
            p.get_or("plane_normal_z", 1.0),
        );
        let (plane_u, plane_v) = basis_from_normal(normal);
        Box::new(SpinWaveResponse {
            plane_u,
            plane_v,
            coupling: p.get_or("coupling", 1.0),
            drive: p.get_or("drive", 1.0),
            chi: p.get_or("chi", 1.0).max(MIN_LEN),
            state: Mutex::new(Inner::empty()),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{CoreParams, Domain, Neighbor, Species};

    struct DummyDomain;
    impl Domain for DummyDomain {
        fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
            b - a
        }
        fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
        fn name(&self) -> &'static str {
            "dummy_domain"
        }
    }

    fn params() -> CoreParams {
        CoreParams::builder()
            .cruise_speed(1.0)
            .max_force(10.0)
            .speed_min_factor(0.3)
            .boid_count(4)
            .vision_radius(10.0)
            .dt(1.0)
            .build()
            .unwrap()
    }

    fn ctx<'a>(
        index: u32,
        vel: Vec3,
        neighbors: &'a [Neighbor],
        core_params: &'a CoreParams,
        domain: &'a dyn Domain,
    ) -> BoidCtx<'a> {
        ctx_at_step(index, vel, neighbors, core_params, domain, 0)
    }

    fn ctx_at_step<'a>(
        index: u32,
        vel: Vec3,
        neighbors: &'a [Neighbor],
        core_params: &'a CoreParams,
        domain: &'a dyn Domain,
        step_count: u64,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos: Vec3::ZERO,
            vel,
            species: Species::Prey,
            neighbors,
            core_params,
            domain,
            step_count,
        }
    }

    fn plugin(coupling: f64, drive: f64, chi: f64) -> SpinWaveResponse {
        SpinWaveResponse {
            plane_u: Vec3::new(1.0, 0.0, 0.0),
            plane_v: Vec3::new(0.0, 1.0, 0.0),
            coupling,
            drive,
            chi,
            state: Mutex::new(Inner::empty()),
        }
    }

    #[test]
    fn conforms_to_steering_modifier_contract() {
        murmur_conformance::steering_modifier(&plugin(1.0, 1.0, 1.0));
    }

    #[test]
    fn a_first_step_boid_with_no_neighbours_drifts_toward_the_desired_heading() {
        let m = plugin(0.0, 1.0, 1.0); // drive only, no coupling
        let p = params();
        let domain = DummyDomain;
        let vel = Vec3::new(1.0, 0.0, 0.0);
        let desired = Vec3::new(0.0, 1.0, 0.0); // 90 degrees away
        let acc = m.respond(ctx(0, vel, &[], &p, &domain), desired, vel);
        assert!(acc.is_finite());
        // Some correction toward +y should be produced (a positive y-component), not zero —
        // proves the drive term actually engages from a cold start.
        assert!(
            acc.y > 0.0,
            "expected some turn toward the desired heading, got {acc:?}"
        );
    }

    /// Two calls at the *same* `step_count` must be fully deterministic (both read the same
    /// frozen `committed` snapshot) — this is the actual correctness property the commit-swap
    /// design exists for (a bug here previously showed up as thread-count-dependent
    /// `state_hash` in `tests/pearce_integration.rs`, not as a same-thread repeat-call
    /// difference, but determinism-under-repetition is the same property at a smaller scale).
    #[test]
    fn two_calls_at_the_same_step_are_fully_deterministic() {
        let m = plugin(0.0, 1.0, 1.0);
        let p = params();
        let domain = DummyDomain;
        let vel = Vec3::new(1.0, 0.0, 0.0);
        let desired = Vec3::new(0.0, 1.0, 0.0);

        let acc1 = m.respond(ctx_at_step(0, vel, &[], &p, &domain, 0), desired, vel);
        let acc2 = m.respond(ctx_at_step(0, vel, &[], &p, &domain, 0), desired, vel);
        assert!(acc1.is_finite() && acc2.is_finite());
        assert_eq!(
            acc1, acc2,
            "two calls at the same step_count must read the same frozen snapshot and agree exactly"
        );
    }

    /// Real persistence: a call at a *later* `step_count` must build on what the previous
    /// step's call computed (the commit-swap actually carrying state forward across a step
    /// boundary), not silently reset.
    #[test]
    fn state_carries_forward_across_a_real_step_boundary() {
        let m = plugin(0.0, 1.0, 1.0);
        let p = params();
        let domain = DummyDomain;
        let vel = Vec3::new(1.0, 0.0, 0.0);
        let desired = Vec3::new(0.0, 1.0, 0.0);

        let acc_step0 = m.respond(ctx_at_step(0, vel, &[], &p, &domain, 0), desired, vel);
        let acc_step1 = m.respond(ctx_at_step(0, vel, &[], &p, &domain, 1), desired, vel);
        assert!(acc_step0.is_finite() && acc_step1.is_finite());
        assert_ne!(
            acc_step0, acc_step1,
            "a call at the next step_count should build on the committed state from the \
             previous step, not restart from scratch"
        );
    }

    #[test]
    fn neighbour_coupling_pulls_phase_toward_a_neighbours_phase() {
        // Seed boid 1's state (a "previous step") with a phase 90 degrees away from boid 0's
        // current heading, then run boid 0's respond() with boid 1 as its only neighbour and
        // zero drive (isolating the coupling term). Boid 0's spin should pick up a nonzero
        // value pulling it toward boid 1's phase.
        let m = plugin(1.0, 0.0, 1.0); // coupling only, no drive
                                       // Seed via `pending`, not `committed` directly: the first `respond()` call below (for
                                       // a never-before-seen step_count) triggers the commit swap (`committed =
                                       // take(pending)`), which is what makes this seeded value actually visible as boid 1's
                                       // "previous step" state — matching how a real previous step's writes would arrive.
        m.state.lock().unwrap().pending.insert(
            1,
            SpinState {
                phi: std::f64::consts::FRAC_PI_2, // +y direction
                s_z: 0.0,
            },
        );
        let p = params();
        let domain = DummyDomain;
        let vel = Vec3::new(1.0, 0.0, 0.0); // boid 0 currently faces +x
        let neighbors = [Neighbor {
            index: 1,
            distance: 1.0,
            direction: Vec3::new(0.0, 1.0, 0.0),
            velocity: Vec3::new(0.0, 1.0, 0.0),
        }];
        let acc = m.respond(ctx(0, vel, &neighbors, &p, &domain), vel, vel);
        assert!(acc.is_finite());
        assert!(
            acc.y > 0.0,
            "coupling to a neighbour facing +y should start pulling boid 0 toward +y, got {acc:?}"
        );
    }

    #[test]
    fn zero_length_desired_and_current_velocity_produce_no_nan() {
        let m = plugin(1.0, 1.0, 1.0);
        let p = params();
        let domain = DummyDomain;
        let acc = m.respond(ctx(0, Vec3::ZERO, &[], &p, &domain), Vec3::ZERO, Vec3::ZERO);
        assert!(acc.is_finite());
    }

    #[test]
    fn only_the_in_plane_velocity_component_is_corrected() {
        let m = plugin(0.0, 1.0, 1.0);
        let p = params();
        let domain = DummyDomain;
        // A boid moving partly out of the XY plane (nonzero z).
        let vel = Vec3::new(1.0, 0.0, 5.0);
        let desired = Vec3::new(0.0, 1.0, 0.0);
        let acc = m.respond(ctx(0, vel, &[], &p, &domain), desired, vel);
        assert!(acc.is_finite());
        assert_eq!(
            acc.z, 0.0,
            "the out-of-plane component must be left untouched"
        );
    }

    #[test]
    fn an_unseen_boid_index_does_not_panic_and_starts_from_its_current_heading() {
        let m = plugin(1.0, 1.0, 1.0);
        let p = params();
        let domain = DummyDomain;
        // A brand-new index, e.g. one just spawned via Command::AddPredator, never seen
        // before by this modifier's state map.
        let acc = m.respond(
            ctx(999, Vec3::new(1.0, 0.0, 0.0), &[], &p, &domain),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        assert!(acc.is_finite());
    }

    #[test]
    fn drive_zero_recovers_the_papers_alignment_only_behaviour() {
        // With drive=0, the desired_v argument should have zero influence on the outcome —
        // only neighbour coupling drives the response, matching the paper's own minimal model.
        let m = plugin(0.0, 0.0, 1.0); // both zero: nothing should move the heading at all
        let p = params();
        let domain = DummyDomain;
        let vel = Vec3::new(1.0, 0.0, 0.0);
        let acc = m.respond(ctx(0, vel, &[], &p, &domain), Vec3::new(0.0, 1.0, 0.0), vel);
        assert!(
            acc.len() < 1e-9,
            "zero coupling and zero drive should produce no correction, got {acc:?}"
        );
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let m = reg
            .resolve_modifier("spin_wave", &PluginParams::new())
            .unwrap();
        assert_eq!(m.name(), "spin_wave");
    }

    /// Proves the `SteeringModifier` seam now has ≥2 real occupants (roadmap.md Phase 13 exit
    /// gate, mirroring the other Phase 13 plugins' pattern).
    #[test]
    fn instant_response_and_spin_wave_both_resolve_via_the_same_seam() {
        let mut reg = Registry::new();
        register(&mut reg);
        murmur_instant_response::register(&mut reg);

        let spin = reg
            .resolve_modifier("spin_wave", &PluginParams::new())
            .unwrap();
        let instant = reg
            .resolve_modifier("instant_response", &PluginParams::new())
            .unwrap();
        assert_eq!(spin.name(), "spin_wave");
        assert_eq!(instant.name(), "instant_response");
    }
}
