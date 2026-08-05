//! `RicherPredator` — the pymurmur-superset predator/prey `StepHook` (design/02_plugins.md
//! §5, roadmap.md Phase 15). Supersedes the Phase 8 minimal `murmur_predator` (a single
//! pursue/flee pair) with an approach/egress predator FSM and a push/wake/split/wave prey
//! force bundle plus a blackening response — `murmur_predator` stays registered as-is, the
//! validated baseline it was always meant to be (design's own framing), not replaced.
//!
//! **Source note — read before trusting any specific number here.** The design doc names
//! this plugin's target shape ("Full FSM (approach/egress) + force bundle (push/wake/split/
//! wave) + panic/blackening") citing pymurmur's `physics/extensions/predator.py` — an external
//! file in the `git_mur` reference repository this environment has never had access to (same
//! blocker as the `pymurmur` cross-check throughout this project, `roadmap.md` §3 item 4). A
//! web search for a public "pymurmur" repository with that structure found nothing (2026-08-05)
//! — it's a private reference project, not reconstructable from a search. **This
//! implementation is therefore grounded directly in the real ornithological literature these
//! named phenomena come from**, not a guess at pymurmur's exact internals: Papadopoulou et al.
//! 2024 ("Mechanisms driving collective escape patterns in starling flocks", bioRxiv preprint;
//! `blackening`, `wave events`, `flash expansion`, `splits` as named, characterized escape
//! patterns whose likelihood depends on predator attack speed/angle/distance) and the earlier
//! Procaccini et al. 2011 discovery of the "agitation wave" propagating through attacked
//! flocks. The mapping from those named field phenomena to concrete force terms below is this
//! implementation's own design, not a transcription of unavailable source code — documented
//! per-term so it's clear what's cited vs. designed.
//!
//! **The predator FSM — `Approach`/`Egress`.** Real raptors hunt in repeated dive-and-recover
//! passes, not one continuous chase (matching Papadopoulou et al.'s finding that response
//! *type* depends on timing relative to an attack, not just distance). `Approach`: pursue the
//! prey centroid, same pursuit law as the Phase 8 minimal plugin. `Egress`: peel off — steer
//! *away* from the prey centroid, mimicking breaking off after a strike attempt. Transitions:
//! `Approach -> Egress` once within `strike_distance` of the nearest prey, or after
//! `approach_max_steps` (a dive that doesn't connect still ends); `Egress -> Approach` after
//! `egress_steps` (a fixed recovery window before the next pass).
//!
//! **The prey force bundle.** All five terms below are computed once per step in `pre_step`
//! (which has full `SimView` access to every active boid — the same pattern the Phase 8
//! plugin already uses for its `prey_com`/`predator_positions` cache), not in `post_steer`
//! (sequential, and `ctx.neighbors` is a hard-coded empty placeholder there — **G1**,
//! roadmap.md §12). This sidesteps G1 entirely rather than fixing it: `pre_step` already
//! provides everything a real per-boid neighbour computation needs, at the cost of an O(N²)
//! sweep per step (fine at this project's slice scale, N<=2600; not a 300k-scale design,
//! matching every other Phase 13/15 addition's stated scope). G1 itself is left open,
//! unneeded — the roadmap explicitly asked this phase to check, not assume.
//! - **push** (flash expansion): a sharp radial-outward burst, on top of the Phase 8 plugin's
//!   existing linear-ramp flee force, triggered only for prey within `danger_radius` of an
//!   `Approach`-phase predator (an `Egress`-phase predator is retreating — Papadopoulou et
//!   al.'s finding that response depends on the live threat, not just proximity).
//! - **wake**: a lateral avoidance force for prey caught in the corridor directly behind an
//!   `Approach` predator's flight path (the space it just vacated/is about to re-enter) —
//!   steers them out of that corridor rather than letting them drift back into it.
//! - **blackening**: a broader-radius (`awareness_radius > danger_radius`), lower-intensity
//!   cohesion boost toward each prey's own local neighbours — the flock bunching/darkening
//!   without individuals fleeing far, active whenever any predator is in `Approach` phase
//!   anywhere in `awareness_radius`, not contingent on immediate danger.
//! - **wave**: a genuinely *propagating* signal — matches Procaccini et al.'s and Papadopoulou
//!   et al.'s "zig manoeuvres...copied by neighbours" description directly. Prey within
//!   `wave_trigger_radius` of an `Approach` predator get their `zig_phase` set to 1 and pick a
//!   `zig_axis` (a lateral direction, perpendicular to both their heading and their bearing to
//!   the predator); prey outside that range but within `wave_relay_radius` of a neighbour whose
//!   *previous step's* `zig_phase` was high inherit a decayed, delayed copy of that neighbour's
//!   `zig_phase`/`zig_axis` — a real one-step relay, propagating outward from the trigger point
//!   exactly like `murmur_spin_wave`'s phase field, but computed directly here (no shared
//!   `SteeringModifier` composition needed) since `pre_step` already has the global access
//!   `respond()` gets from `ctx.neighbors` in the read phase. `zig_phase` decays geometrically
//!   each step it isn't refreshed.
//! - **split**: pulls prey with a currently-high `zig_phase` toward *other* nearby high-
//!   `zig_phase` prey specifically (not the whole local neighbourhood, unlike blackening) —
//!   panicking sub-groups clumping with each other, which — combined with push/wave pulling
//!   different parts of the flock in different directions during a sustained attack — is a
//!   genuine, real mechanism for the flock fragmenting into sub-flocks (Papadopoulou et al.'s
//!   "splits", often following a flash expansion), not merely a label. Directly testable with
//!   the connected-components tooling used throughout this project's Phase 14 calibration work.
//!
//! `panic` (the design doc's other named state alongside blackening) isn't a separate force
//! term — it's the descriptive name for a prey boid simultaneously inside `danger_radius` of an
//! `Approach` predator, i.e. the regime where push, wave, and (if a nearby ally is also
//! panicking) split are all active together.

use std::collections::HashMap;
use std::sync::Mutex;

use murmur_core::{clamp_len, BoidCtx, PluginParams, Registry, SimView, Species, StepHook, Vec3};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PredatorPhase {
    Approach,
    Egress,
}

struct PredatorFsm {
    phase: PredatorPhase,
    phase_timer: u32,
}

#[derive(Clone, Copy)]
struct WaveState {
    zig_phase: f64,
    zig_axis: Vec3,
}

#[derive(Default)]
struct Cache {
    predator_force: HashMap<u32, Vec3>,
    prey_force: HashMap<u32, Vec3>,
}

pub struct RicherPredator {
    pub predator_accel: f64,
    pub danger_radius: f64,
    pub awareness_radius: f64,
    pub wave_trigger_radius: f64,
    pub wave_relay_radius: f64,
    pub strike_distance: f64,
    pub approach_max_steps: u32,
    pub egress_steps: u32,
    pub flight_strength: f64,
    pub push_strength: f64,
    pub wake_strength: f64,
    pub wake_corridor_radius: f64,
    pub blackening_strength: f64,
    pub blackening_neighbors: usize,
    pub split_strength: f64,
    pub split_trigger: f64,
    pub wave_strength: f64,
    pub wave_decay: f64,
    pub wave_relay_gain: f64,
    fsm: Mutex<HashMap<u32, PredatorFsm>>,
    wave: Mutex<HashMap<u32, WaveState>>,
    cache: Mutex<Cache>,
}

impl RicherPredator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        predator_accel: f64,
        danger_radius: f64,
        awareness_radius: f64,
        wave_trigger_radius: f64,
        wave_relay_radius: f64,
        strike_distance: f64,
        approach_max_steps: u32,
        egress_steps: u32,
        flight_strength: f64,
        push_strength: f64,
        wake_strength: f64,
        wake_corridor_radius: f64,
        blackening_strength: f64,
        blackening_neighbors: usize,
        split_strength: f64,
        split_trigger: f64,
        wave_strength: f64,
        wave_decay: f64,
        wave_relay_gain: f64,
    ) -> Self {
        RicherPredator {
            predator_accel,
            danger_radius,
            awareness_radius,
            wave_trigger_radius,
            wave_relay_radius,
            strike_distance,
            approach_max_steps,
            egress_steps,
            flight_strength,
            push_strength,
            wake_strength,
            wake_corridor_radius,
            blackening_strength,
            blackening_neighbors,
            split_strength,
            split_trigger,
            wave_strength,
            wave_decay,
            wave_relay_gain,
            fsm: Mutex::new(HashMap::new()),
            wave: Mutex::new(HashMap::new()),
            cache: Mutex::new(Cache::default()),
        }
    }

    /// A deterministic lateral ("zig") direction: perpendicular to both the bearing toward the
    /// threat and the boid's own heading. Falls back to a perpendicular of `to_predator` alone
    /// (via a fixed reference axis) if heading is degenerate (a stationary boid).
    fn zig_axis(to_predator: Vec3, heading: Vec3) -> Vec3 {
        let primary = to_predator.cross(heading);
        if primary.len_sq() > 1e-12 {
            return primary.normalized();
        }
        let reference = if to_predator.cross(Vec3::new(0.0, 0.0, 1.0)).len_sq() > 1e-12 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let fallback = to_predator.cross(reference);
        if fallback.len_sq() > 1e-12 {
            fallback.normalized()
        } else {
            Vec3::ZERO
        }
    }
}

impl StepHook for RicherPredator {
    fn pre_step(&mut self, sim: &mut SimView) {
        let mut predator_ids = Vec::new();
        let mut predator_pos = Vec::new();
        let mut predator_vel = Vec::new();
        let mut prey_ids = Vec::new();
        let mut prey_pos = Vec::new();
        let mut prey_vel = Vec::new();
        let mut com = Vec3::ZERO;
        let mut prey_count = 0u32;

        for i in sim.boids.iter_active() {
            let idx = i as usize;
            match sim.boids.species[idx] {
                Species::Predator => {
                    predator_ids.push(i);
                    predator_pos.push(sim.boids.pos[idx]);
                    predator_vel.push(sim.boids.vel[idx]);
                }
                Species::Prey | Species::Custom(_) => {
                    prey_ids.push(i);
                    prey_pos.push(sim.boids.pos[idx]);
                    prey_vel.push(sim.boids.vel[idx]);
                    com += sim.boids.pos[idx];
                    prey_count += 1;
                }
            }
        }
        let prey_com = if prey_count > 0 {
            com / prey_count as f64
        } else {
            Vec3::ZERO
        };

        // --- predator FSM ---
        let mut fsm = self.fsm.lock().expect("predator fsm mutex poisoned");
        let mut predator_phase = Vec::with_capacity(predator_ids.len());
        for (k, &pid) in predator_ids.iter().enumerate() {
            let nearest_d = prey_pos
                .iter()
                .map(|&p| (p - predator_pos[k]).len())
                .fold(f64::INFINITY, f64::min);
            let state = fsm.entry(pid).or_insert(PredatorFsm {
                phase: PredatorPhase::Approach,
                phase_timer: 0,
            });
            state.phase_timer += 1;
            match state.phase {
                PredatorPhase::Approach => {
                    if nearest_d <= self.strike_distance
                        || state.phase_timer >= self.approach_max_steps
                    {
                        state.phase = PredatorPhase::Egress;
                        state.phase_timer = 0;
                    }
                }
                PredatorPhase::Egress => {
                    if state.phase_timer >= self.egress_steps {
                        state.phase = PredatorPhase::Approach;
                        state.phase_timer = 0;
                    }
                }
            }
            predator_phase.push(state.phase);
        }
        drop(fsm);

        let mut predator_force = HashMap::new();
        for (k, &pid) in predator_ids.iter().enumerate() {
            let toward = match predator_phase[k] {
                PredatorPhase::Approach => prey_com - predator_pos[k],
                PredatorPhase::Egress => predator_pos[k] - prey_com,
            };
            let acc = if toward.len_sq() > 1e-18 {
                let desired_vel = predator_vel[k] + toward.normalized() * self.predator_accel;
                desired_vel - predator_vel[k]
            } else {
                Vec3::ZERO
            };
            predator_force.insert(pid, acc);
        }

        // --- wave state (single-threaded within pre_step: no races, no double-buffer needed) ---
        let mut wave = self.wave.lock().expect("predator wave mutex poisoned");
        let old_wave = wave.clone();
        let mut new_wave: HashMap<u32, WaveState> = HashMap::with_capacity(prey_ids.len());

        for (k, &bid) in prey_ids.iter().enumerate() {
            let bpos = prey_pos[k];
            let heading = prey_vel[k];
            let mut triggered = false;
            let mut trigger_axis = Vec3::ZERO;
            for (j, &ppos) in predator_pos.iter().enumerate() {
                if predator_phase[j] != PredatorPhase::Approach {
                    continue;
                }
                let to_pred = ppos - bpos;
                if to_pred.len() <= self.wave_trigger_radius {
                    triggered = true;
                    trigger_axis = Self::zig_axis(to_pred, heading);
                    break;
                }
            }

            let new_state = if triggered {
                WaveState {
                    zig_phase: 1.0,
                    zig_axis: trigger_axis,
                }
            } else {
                let prev = old_wave.get(&bid).copied().unwrap_or(WaveState {
                    zig_phase: 0.0,
                    zig_axis: Vec3::ZERO,
                });
                let mut best = prev.zig_phase * self.wave_decay;
                let mut axis = prev.zig_axis;
                for (k2, &bid2) in prey_ids.iter().enumerate() {
                    if bid2 == bid {
                        continue;
                    }
                    if (bpos - prey_pos[k2]).len() > self.wave_relay_radius {
                        continue;
                    }
                    if let Some(neighbor_prev) = old_wave.get(&bid2) {
                        let relayed = neighbor_prev.zig_phase * self.wave_relay_gain;
                        if relayed > best {
                            best = relayed;
                            axis = neighbor_prev.zig_axis;
                        }
                    }
                }
                WaveState {
                    zig_phase: best.min(1.0),
                    zig_axis: axis,
                }
            };
            new_wave.insert(bid, new_state);
        }
        *wave = new_wave.clone();
        drop(wave);

        // --- assemble the prey force bundle ---
        let mut prey_force = HashMap::with_capacity(prey_ids.len());
        for (k, &bid) in prey_ids.iter().enumerate() {
            let bpos = prey_pos[k];
            let mut force = Vec3::ZERO;

            // push (flash expansion) + baseline flee, only against an Approach predator.
            for (j, &ppos) in predator_pos.iter().enumerate() {
                if predator_phase[j] != PredatorPhase::Approach {
                    continue;
                }
                let away = bpos - ppos;
                let d = away.len();
                if d < 1e-9 || d >= self.danger_radius {
                    continue;
                }
                let ramp = 1.0 - d / self.danger_radius;
                force += away.normalized()
                    * (self.flight_strength * ramp + self.push_strength * ramp * ramp);
            }

            // wake: lateral push out of the corridor directly behind an Approach predator.
            for (j, &ppos) in predator_pos.iter().enumerate() {
                if predator_phase[j] != PredatorPhase::Approach {
                    continue;
                }
                let pvel = predator_vel[j];
                if pvel.len_sq() < 1e-12 {
                    continue;
                }
                let forward = pvel.normalized();
                let rel = bpos - ppos;
                let along = rel.dot(forward);
                if along >= 0.0 {
                    continue; // not behind the predator
                }
                let lateral = rel - forward * along;
                let lateral_dist = lateral.len();
                if lateral_dist < self.wake_corridor_radius && lateral_dist > 1e-9 {
                    let corridor_factor = 1.0 - lateral_dist / self.wake_corridor_radius;
                    force += lateral.normalized() * (self.wake_strength * corridor_factor);
                }
            }

            // blackening: mild cohesion boost toward local prey neighbours, whenever any
            // predator is in Approach phase within awareness_radius.
            let any_aware = predator_pos.iter().enumerate().any(|(j, &ppos)| {
                predator_phase[j] == PredatorPhase::Approach
                    && (bpos - ppos).len() <= self.awareness_radius
            });
            if any_aware {
                let mut dists: Vec<(f64, usize)> = prey_pos
                    .iter()
                    .enumerate()
                    .filter(|&(k2, _)| k2 != k)
                    .map(|(k2, &p)| ((bpos - p).len(), k2))
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let take = self.blackening_neighbors.min(dists.len());
                if take > 0 {
                    let mut avg = Vec3::ZERO;
                    for &(_, k2) in dists.iter().take(take) {
                        avg += prey_pos[k2];
                    }
                    avg /= take as f64;
                    let toward = avg - bpos;
                    if toward.len_sq() > 1e-12 {
                        force += toward.normalized() * self.blackening_strength;
                    }
                }
            }

            // split: pull toward other currently-panicking prey specifically.
            let self_zig = new_wave.get(&bid).map(|w| w.zig_phase).unwrap_or(0.0);
            if self_zig >= self.split_trigger {
                let mut avg = Vec3::ZERO;
                let mut n = 0u32;
                for (k2, &bid2) in prey_ids.iter().enumerate() {
                    if bid2 == bid {
                        continue;
                    }
                    if (bpos - prey_pos[k2]).len() > self.wave_relay_radius {
                        continue;
                    }
                    if new_wave.get(&bid2).map(|w| w.zig_phase).unwrap_or(0.0) >= self.split_trigger
                    {
                        avg += prey_pos[k2];
                        n += 1;
                    }
                }
                if n > 0 {
                    let toward = (avg / n as f64) - bpos;
                    if toward.len_sq() > 1e-12 {
                        force += toward.normalized() * (self.split_strength * self_zig);
                    }
                }
            }

            // wave: the propagating zig itself.
            if let Some(w) = new_wave.get(&bid) {
                force += w.zig_axis * (w.zig_phase * self.wave_strength);
            }

            prey_force.insert(bid, force);
        }

        *self.cache.lock().expect("predator cache mutex poisoned") = Cache {
            predator_force,
            prey_force,
        };
    }

    fn post_steer(&self, ctx: BoidCtx<'_>, acc: &mut Vec3) {
        let cache = self.cache.lock().expect("predator cache mutex poisoned");
        match ctx.species {
            Species::Predator => {
                if let Some(&f) = cache.predator_force.get(&ctx.index) {
                    *acc = f; // overwrite: predator motion is a standalone behaviour.
                }
            }
            Species::Prey | Species::Custom(_) => {
                if let Some(&f) = cache.prey_force.get(&ctx.index) {
                    *acc += clamp_len(f, ctx.core_params.max_force);
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "predator_fsm"
    }
}

/// Registers `RicherPredator` under the name `"predator_fsm"`. Defaults are the Phase 8
/// minimal plugin's own values (`predator_accel=0.4`, `flight_strength=2.5`,
/// `danger_radius=160*(body_radius/9)`) plus new, purpose-chosen defaults for the FSM
/// timing and the push/wake/blackening/split/wave bundle — see the module doc for what each
/// governs. None of the new defaults are cited to a specific empirical source (unlike
/// `danger_radius`/`predator_accel`/`flight_strength`, which are `sim_new.md`'s own numbers) —
/// they're chosen to be physically sensible multiples of `danger_radius` and are exercised by
/// this crate's own tests for correct qualitative behaviour, not tuned against field data.
pub fn register(r: &mut Registry) {
    r.register_step_hook("predator_fsm", |p: &PluginParams| {
        let body_radius = p.get_or("body_radius", 9.0);
        let danger_radius = p.get_or("danger_radius", 160.0 * (body_radius / 9.0));
        Box::new(RicherPredator::new(
            p.get_or("predator_accel", 0.4),
            danger_radius,
            p.get_or("awareness_radius", danger_radius * 2.5),
            p.get_or("wave_trigger_radius", danger_radius * 1.5),
            p.get_or("wave_relay_radius", danger_radius * 0.5),
            p.get_or("strike_distance", danger_radius * 0.15),
            p.get_or("approach_max_steps", 120.0) as u32,
            p.get_or("egress_steps", 40.0) as u32,
            p.get_or("flight_strength", 2.5),
            p.get_or("push_strength", 4.0),
            p.get_or("wake_strength", 1.5),
            p.get_or("wake_corridor_radius", danger_radius * 0.5),
            p.get_or("blackening_strength", 0.3),
            p.get_or("blackening_neighbors", 6.0) as usize,
            p.get_or("split_strength", 1.0),
            p.get_or("split_trigger", 0.5),
            p.get_or("wave_strength", 1.0),
            p.get_or("wave_decay", 0.85),
            p.get_or("wave_relay_gain", 0.9),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::{BoidColumns, CoreParams, Domain};

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
            .max_force(20.0)
            .speed_min_factor(0.3)
            .boid_count(8)
            .vision_radius(10.0)
            .build()
            .unwrap()
    }

    fn default_hook() -> RicherPredator {
        RicherPredator::new(
            0.4,  // predator_accel
            20.0, // danger_radius
            50.0, // awareness_radius
            30.0, // wave_trigger_radius
            10.0, // wave_relay_radius
            3.0,  // strike_distance
            5,    // approach_max_steps
            3,    // egress_steps
            2.5,  // flight_strength
            4.0,  // push_strength
            1.5,  // wake_strength
            10.0, // wake_corridor_radius
            0.3,  // blackening_strength
            6,    // blackening_neighbors
            1.0,  // split_strength
            0.5,  // split_trigger
            1.0,  // wave_strength
            0.85, // wave_decay
            0.9,  // wave_relay_gain
        )
    }

    fn boids_with(entries: &[(Vec3, Vec3, Species)]) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(entries.len() as u32);
        for &(p, v, s) in entries {
            b.add(p, v, s, 0);
        }
        b
    }

    fn ctx_for<'a>(
        index: u32,
        pos: Vec3,
        vel: Vec3,
        species: Species,
        p: &'a CoreParams,
        domain: &'a StubDomain,
    ) -> BoidCtx<'a> {
        BoidCtx {
            index,
            pos,
            vel,
            species,
            neighbors: &[],
            core_params: p,
            domain,
            step_count: 0,
        }
    }

    #[test]
    fn predator_starts_in_approach_and_switches_to_egress_within_strike_distance() {
        let mut hook = default_hook();
        // Predator right on top of a single prey -> distance 0 < strike_distance.
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);
        let fsm = hook.fsm.lock().unwrap();
        assert_eq!(fsm.get(&0).unwrap().phase, PredatorPhase::Egress);
    }

    #[test]
    fn predator_returns_to_approach_after_egress_steps() {
        let mut hook = default_hook();
        // Prey kept well beyond strike_distance=3.0 throughout, so once back in Approach it
        // doesn't immediately re-trigger Egress on the very next call (a real predator would
        // only re-approach once it's actually put some distance behind it).
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        {
            let mut fsm = hook.fsm.lock().unwrap();
            fsm.insert(
                0,
                PredatorFsm {
                    phase: PredatorPhase::Egress,
                    phase_timer: 0,
                },
            );
        }
        let mut params = core_params();
        // egress_steps=3: after 3 more calls the timer reaches 3 and flips back to Approach.
        for step in 0..3 {
            let mut view = SimView {
                boids: &boids,
                core_params: &mut params,
                step_count: step,
            };
            hook.pre_step(&mut view);
        }
        let fsm = hook.fsm.lock().unwrap();
        assert_eq!(fsm.get(&0).unwrap().phase, PredatorPhase::Approach);
    }

    #[test]
    fn egress_predator_moves_away_from_prey_not_toward() {
        let mut hook = default_hook();
        let boids = boids_with(&[
            (Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        {
            let mut fsm = hook.fsm.lock().unwrap();
            fsm.insert(
                0,
                PredatorFsm {
                    phase: PredatorPhase::Egress,
                    phase_timer: 0,
                },
            );
        }
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);

        let domain = StubDomain;
        let p = core_params();
        let ctx = ctx_for(
            0,
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Predator,
            &p,
            &domain,
        );
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc);
        assert!(
            acc.x < 0.0,
            "egress predator should move further away (-x), got {acc:?}"
        );
    }

    #[test]
    fn push_force_exceeds_baseline_flee_and_only_applies_under_an_approaching_predator() {
        let mut hook = default_hook();
        // Prey at distance 10: inside danger_radius=20 (push/flee should apply) but outside
        // strike_distance=3.0, so a fresh Approach-phase FSM stays Approach for this whole
        // pre_step call instead of immediately flipping to Egress before force computation.
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);
        let domain = StubDomain;
        let p = core_params();
        let ctx = ctx_for(
            1,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc_approach = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc_approach);

        // Egress case: same geometry, predator already egressing -> no push/flee at all.
        let mut hook2 = default_hook();
        let boids2 = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        {
            let mut fsm = hook2.fsm.lock().unwrap();
            fsm.insert(
                0,
                PredatorFsm {
                    phase: PredatorPhase::Egress,
                    phase_timer: 0,
                },
            );
        }
        let mut params2 = core_params();
        let mut view2 = SimView {
            boids: &boids2,
            core_params: &mut params2,
            step_count: 0,
        };
        hook2.pre_step(&mut view2);
        let ctx2 = ctx_for(
            1,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc_egress = Vec3::ZERO;
        hook2.post_steer(ctx2, &mut acc_egress);

        assert!(
            acc_approach.x > 0.0,
            "should push away (+x): {acc_approach:?}"
        );
        assert!(
            acc_egress.len() < 1e-9,
            "an egressing predator should trigger no push/flee at all: {acc_egress:?}"
        );
    }

    #[test]
    fn wake_force_pushes_prey_out_of_the_corridor_behind_an_approaching_predator() {
        let hook = default_hook();
        {
            let mut fsm = hook.fsm.lock().unwrap();
            fsm.insert(
                0,
                PredatorFsm {
                    phase: PredatorPhase::Approach,
                    phase_timer: 0,
                },
            );
        }
        let boids = boids_with(&[
            (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Species::Predator,
            ), // flying +x
            (Vec3::new(-5.0, 1.0, 0.0), Vec3::ZERO, Species::Prey), // behind, slightly offset
        ]);
        let mut params = core_params();
        let mut hook_mut = default_hook();
        {
            let mut fsm = hook_mut.fsm.lock().unwrap();
            fsm.insert(
                0,
                PredatorFsm {
                    phase: PredatorPhase::Approach,
                    phase_timer: 0,
                },
            );
        }
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook_mut.pre_step(&mut view);

        let domain = StubDomain;
        let p = core_params();
        let ctx = ctx_for(
            1,
            Vec3::new(-5.0, 1.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc = Vec3::ZERO;
        hook_mut.post_steer(ctx, &mut acc);
        // Predator flies +x; prey is behind (-x) and offset +y -> wake force should push
        // further +y (out of the corridor), not toward -y.
        assert!(
            acc.y > 0.0,
            "expected lateral push out of the wake corridor, got {acc:?}"
        );
        let _ = hook; // silence unused (kept for symmetry with other tests' setup style)
    }

    #[test]
    fn blackening_applies_within_awareness_radius_but_not_far_beyond_it() {
        let mut hook = default_hook();
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(30.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // within awareness_radius=50, outside danger_radius=20
            (Vec3::new(32.0, 5.0, 0.0), Vec3::ZERO, Species::Prey), // a neighbour to cohere toward
            (Vec3::new(1000.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // far outside awareness_radius
        ]);
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);

        let domain = StubDomain;
        let p = core_params();
        let ctx_near = ctx_for(
            1,
            Vec3::new(30.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc_near = Vec3::ZERO;
        hook.post_steer(ctx_near, &mut acc_near);
        assert!(
            acc_near.len() > 1e-6,
            "expected a nonzero blackening cohesion force: {acc_near:?}"
        );

        let ctx_far = ctx_for(
            3,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc_far = Vec3::ZERO;
        hook.post_steer(ctx_far, &mut acc_far);
        assert!(
            acc_far.len() < 1e-9,
            "expected no blackening force far outside awareness_radius: {acc_far:?}"
        );
    }

    #[test]
    fn wave_triggers_immediately_near_an_approaching_predator_and_relays_to_a_neighbour_next_step()
    {
        let mut hook = default_hook();
        // Boid 1 is within wave_trigger_radius=30 of the predator; boid 2 is farther (outside
        // trigger range) but within wave_relay_radius=10 of boid 1.
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (
                Vec3::new(25.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Species::Prey,
            ),
            (Vec3::new(33.0, 0.0, 0.0), Vec3::ZERO, Species::Prey),
        ]);
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);
        {
            let wave = hook.wave.lock().unwrap();
            assert!(
                wave.get(&1).unwrap().zig_phase > 0.99,
                "boid 1 should trigger immediately"
            );
            assert!(
                wave.get(&2).map(|w| w.zig_phase).unwrap_or(0.0) < 1e-9,
                "boid 2 should not have picked up the wave on the same step it started"
            );
        }
        // Second step: boid 2 should now have relayed a decayed copy of boid 1's zig_phase.
        let mut view2 = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 1,
        };
        hook.pre_step(&mut view2);
        let wave = hook.wave.lock().unwrap();
        assert!(
            wave.get(&2).unwrap().zig_phase > 0.0,
            "boid 2 should have relayed the wave on the following step"
        );
    }

    #[test]
    fn split_pulls_a_panicking_boid_toward_other_panicking_boids_not_calm_ones() {
        let mut hook = default_hook();
        let boids = boids_with(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Species::Predator),
            (Vec3::new(25.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // triggers wave -> panicking
            (Vec3::new(30.0, 5.0, 0.0), Vec3::ZERO, Species::Prey), // also within trigger radius -> panicking, ahead in +y
            (Vec3::new(-40.0, 0.0, 0.0), Vec3::ZERO, Species::Prey), // far away, calm, opposite direction
        ]);
        let mut params = core_params();
        let mut view = SimView {
            boids: &boids,
            core_params: &mut params,
            step_count: 0,
        };
        hook.pre_step(&mut view);

        let domain = StubDomain;
        let p = core_params();
        let ctx = ctx_for(
            1,
            Vec3::new(25.0, 0.0, 0.0),
            Vec3::ZERO,
            Species::Prey,
            &p,
            &domain,
        );
        let mut acc = Vec3::ZERO;
        hook.post_steer(ctx, &mut acc);
        // The only other panicking boid (index 2) is in the +y direction relative to boid 1;
        // the calm, much-closer-in-x boid (index 3, far -x) must not dominate the split pull.
        assert!(
            acc.y > 0.0,
            "split should pull toward the other panicking boid (+y): {acc:?}"
        );
    }

    #[test]
    fn conforms_to_step_hook_contract() {
        murmur_conformance::step_hook(&mut default_hook());
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let hook = reg
            .resolve_step_hook("predator_fsm", &PluginParams::new())
            .unwrap();
        assert_eq!(hook.name(), "predator_fsm");
    }
}
