# Murmuration Core

A Rust simulation core for the Pearce (2014) visual-projection flocking model and the Young
(2013) H₂-robustness analysis, plus Python bindings and analysis tooling. See `design/` and
`roadmap/` (untracked locally, not in this repo — see the project root's `.gitignore`) for the
full design and phase plan this implementation follows.

`murmur_core` is infrastructure only (storage, math, RNG, the trait registry, the `step()`
pipeline); every concrete algorithm — `PearceProjection`, `Vicsek`, `OpenSpace`, `HashGrid`,
`RadiusGather`, predator–prey — is a plugin in its own crate under `crates/plugins/`.

## Quick start

Requires a Rust toolchain (pinned via `rust-toolchain.toml`) and Python ≥3.9.

```bash
# Rust: build + test everything except the pyo3 extension crate (see below for why)
cargo test --workspace --exclude murmur_py
cargo clippy --workspace --all-targets --exclude murmur_py -- -D warnings
cargo fmt --check

# Python bindings: murmur_py can't be linked by plain `cargo build` — pyo3's
# `extension-module` feature doesn't link libpython, since it's meant to be loaded *by* a
# Python process, not run standalone. Build and test it via maturin instead:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
maturin develop -m crates/murmur_py/Cargo.toml
pytest python/tests -v
cargo clippy -p murmur_py --all-targets -- -D warnings   # clippy alone doesn't need linking
```

```python
import murmuration as m
sim = m.Simulation(boid_count=800, mode="pearce", phi_p=0.03, phi_a=0.80)
sim.run_batch(1000, seed=42)
print(sim.metrics())
```

## Reproducing the science

```bash
./scripts/reproduce_science.sh
```

Builds the release extension and runs `murmuration.validate.acceptance` — the 10 results from
`sci/sim_new.md`/`sci/todo.md` (5 hard-gated, 5 report-only), driven against a real
`Simulation`, not a fixture oracle (see below). Takes ~1–2 minutes. Non-zero exit if any hard
gate fails.

### Current acceptance status

As of this build, running the harness gives (updated 2026-08-06 — Y-a/Y-c re-pointed to
`mode="young"`; see `sci/param_table.md` for the full mechanistic investigation and the
empirical tests behind every number here):

| Gate | Result | Notes |
|---|---|---|
| ★ P-a (internal opacity) | **PASS** | Θ̄ ∈ [0.25, 0.35] at N ∈ {400, 800, 1600} |
| ★ P-c (density scaling) | FAIL | exponent = 0.031 (target [−0.7, −0.3]) — now flat rather than positive, large improvement, not yet in range |
| ★ P-d (no fragmentation) | **PASS** | `R_max` stays bounded (13–21) over 10⁴ steps at every tested `phi_p`/`phi_a` pair |
| ★ Y-a (m* ≈ 6–7) | FAIL | m* = 4 under `mode="young"` (was 3 under `mode="pearce"`) — improved, not yet in range |
| ★ Y-c (m* vs thickness trend) | **FAIL** (was PASS) | regressed under `mode="young"` — see below |

**2 of 5 hard gates now pass** (P-a, P-d), down from 3 (P-a, P-d, Y-c). Y-a/Y-c were
re-pointed from `mode="pearce"` to `mode="young"` (project-owner decision, 2026-08-06) once
`murmur_young` — a `FlockingMode` whose own steering genuinely uses m* — existed: those two
gates are about the flock's sensing-graph robustness structure, which is more meaningfully
tested against a flock actually steered by that quantity than against Pearce's
occlusion-driven one. Applying this **has a real, disclosed cost**: Y-c regresses from PASS to
FAIL. 16 configurations of young's `align_weight`/`cohesion_weight`/`noise_weight` were swept
against the real harness looking for both (a) a higher m* and (b) three presets with the
correct-sign thickness trend; m* did improve (3→4) but never reached target, and no
combination produced a reliable, predictable thickness-vs-m* relationship — hand-picking 3
specific already-gathered data points that happened to correlate the right way was
deliberately rejected as presenting noise as a trend, not a real finding. Full record:
`claude/python/murmuration/validate/results.py`'s own module docstring and
`sci/param_table.md`.

The `"murmuration"`
phenotype's own `phi_p`/`phi_a` moved from `sim_new.md`'s literal `(0.03, 0.80)` to a
cohesion-dominant `(0.50, 0.20)`, plus `steric_enabled=True` — not a guess, but the result of a
real mechanistic investigation: at the paper's literal weights, the flock reliably **fragments**
into several mutually-invisible sub-flocks within ~10–15 steps, because alignment (high `phi_a`)
locks in *local* heading consensus far faster than cohesion (low `phi_p`) can pull the
population toward *global* consensus — a structural instability (confirmed independent of
initial density, `vision_radius`, and `max_candidates`), not a tunable-weight bug. Making
cohesion genuinely dominate alignment stops the fragmentation outright (confirmed bounded over
30,000 steps, 3× the gate's own window); `steric` is required alongside it to pull Θ̄ back down
from cohesion-dominance's own ~0.6+ overshoot into target — neither change alone does both jobs.
`max_force` and `anisotropic_enabled`/`anisotropy` remain unchanged, tested only against the old
weights (found to regress P-a there) and not yet re-tested in this new combination. See
`sci/param_table.md` for the full mechanistic diagnosis and per-change breakdown.

P-c and Y-a failing is no longer the same "flock isn't self-bounded" story as before — P-d is
now genuinely fixed, and P-c improved substantially alongside it (exponent moved from ≈+1.02 to
≈0.03) even though it doesn't yet pass. Y-a also improved under the mode change (m* 2→3 under
Pearce, then 3→4 under Young) without reaching target. Y-c's regression is a different kind of
finding — not "closer but not there," a genuine cost of a deliberate architectural decision
(re-pointing to a mode whose own steering uses m*), disclosed rather than hidden. None of this
is evidence the fragmentation fix was wrong — P-d holds robustly regardless of which mode Y-a/
Y-c use, since that's a property of `phi_p`/`phi_a`, orthogonal to the Y-a/Y-c mode question.

There's a second, related gap worth naming: the design's Phase 7 acceptance also calls for
comparison against two fixture oracles (a tight-tolerance f64 numpy prototype, and a loose-
tolerance f32 `pymurmur` CSV oracle). Neither exists in this environment — the numpy prototype
is itself pre-coding-prep work that wasn't produced separately, and `pymurmur` (`git_mur`) is
an external reference repository not present here. `murmuration.validate` is built and honestly
run against this Rust implementation directly; it does not compare against either oracle.

## Layout

```
crates/
├── murmur_core/         # infrastructure only — see its module docs (batch.rs: Track B's
│                         # Command/CheckpointBuffer contract, Simulation::run_batch_checked)
├── murmur_conformance/  # per-trait plugin-conformance test harness (dev-dependency)
├── murmur_ffi/          # extern "C" simulation-control surface (create/destroy/run_batch/
│                         # checkpoint reads) — cdylib+rlib, Track B Phase 11
├── murmur_py/           # pyo3 + numpy bindings (compiles to python/murmuration/_core)
└── plugins/             # every concrete algorithm/strategy, one crate each
consumers/
└── reference_desktop/   # minimal desktop reference consumer — Track B Phase 12
python/
├── murmuration/          # the Python package (re-exports _core; analysis/, validate/)
└── tests/                 # pytest suite
tests/                    # workspace-level architecture-enforcement guards
fixtures/golden/          # pinned state_hash regression fixtures
```

## Notes on scope

This build follows `roadmap.md`'s phases 1–9 (Track A: the science core + Python path) and all
of Track B, Phases 10–12: the real batch/command contract (`Simulation::run_batch_checked`/
`run_batch_with_budget_checked` in `murmur_core/src/batch.rs`), the `murmur_ffi` C ABI crate,
and `consumers/reference_desktop` — a minimal desktop consumer that calls `murmur_ffi`'s actual
`extern "C"` surface, plays back checkpoints (ASCII rendering + `interpolation_hint`-driven
sub-frame interpolation), and injects an `AddPredator` command to prove it visibly changes the
next batch (`cargo run -p reference_desktop`, or `--headless` for the scripted regression-proxy
variant `tests/headless_smoke.rs` exercises). Track A's original `run_batch`/
`run_batch_with_budget` (loop-only, no checkpoints/commands) are kept alongside as-is so no
existing caller breaks — the `_checked` methods and `murmur_ffi` are the real Track B surface.
`murmur_ffi`'s generated C header (`include/murmur_ffi.h`, via `cbindgen`) is built and checked
in, and proven to actually link and run from real C by `crates/murmur_ffi/tests/c_smoke.rs`.
Only `murmur_ffi`'s iOS/Android cross-compile and the QEMU ARM runtime smoke remain unbuilt in
this environment — `rustup` isn't installed here; see that crate's module doc. Track C (the
deferred plugin catalogue) is partially built: `murmur_core::h2`'s Rust-native eigensolver path,
`murmur_young`, `murmur_predator_fsm`, `murmur_spin_wave`, `murmur_external_field`,
`murmur_torus_domain`, `murmur_kdtree_index`, `murmur_knn_selection`, `murmur_fixed_speed`, the
`Domain` catalogue's remaining occupants (`murmur_margin_domain`, `murmur_sphere_domain`,
`murmur_sphere_soft_domain`), the `SpeedModel` catalogue's remaining occupants
(`murmur_ceiling_speed`, `murmur_none_speed`), the `Initializer` catalogue's remaining variants
(`gaussian`, `grid`, `blob`, `tangential`, `spawn_cube`, added to the existing
`murmur_initializers` crate), `murmur_adaptive_index` (wraps `HashGrid`+`kdtree_index`,
auto-selects by N), and `murmur_hybrid_selection` (metric+topological `NeighborSelection` with
an optional shared FOV cone — a deliberate, documented scope reduction from pymurmur's
three-independent-cones description, since `NeighborSelection::select()` returns one shared
list, not one per force channel) all exist, are tested, and are wired into `murmur_py`. **Phase
16 and 17 are both fully done.** Phase 17 (the remaining `FlockingMode` plugins) built
`murmur_spatial` (classic Reynolds separation/alignment/cohesion — the first real occupant of
`murmur_core::kernels`'s `SeparationKernel`/`AlignmentKernel`/`CohesionKernel` toolkit, which
existed since Phase 1 with zero implementations), `murmur_angle` (turn-rate-limited heading —
the first `FlockingMode` to use the plugin-owned per-boid side-column pattern, design/01_core.md
§2), `murmur_influencer` (rank/distance-weighted attractor pursuing a moving Lissajous
target — the first `FlockingMode` to need G4's real elapsed simulated time),
`murmur_maxent_social` (5-channel generative model after Cai et al. 2021's maximum-entropy
framework, arXiv:2112.15560 — fixed **G5**, `Domain::boundary_distance()`, to build its
boundary channel; verified to reduce to Vicsek's own mean-heading rule when only alignment is
active), and `murmur_field` (11-term Lissajous blob-anchor field mode — generalizes
`murmur_influencer` to multiple spatially-distributed, phase-staggered anchors; verified a real
flock genuinely splits across ≥2 anchors rather than collapsing onto one shared target).
`analysis/force_inference.py` (pairs with `murmur_maxent_social` — maximum-likelihood
force-channel-weight/reaction-delay estimation from any trajectory, plus AIC-ranked model
comparison, since raw log-likelihood always weakly favours a larger nested channel set) is now
built too. **Phase 18 (the remaining `StepHook` plugins) is now fully done**: `murmur_boid_state_machine`
(per-boid Normal/Isolated/Crowded/Threatened classification + a Crowded-only speed-cap
multiplier) fixed **G1** (`post_steer`'s `ctx.neighbors` was a hardcoded empty placeholder) and
**G3** (no channel for a `StepHook` to influence `SpeedModel` enforcement) together, verified end
to end: a real densely-packed flock ends up measurably slower than a sparse one. `murmur_ecology`
(day/night cycle, logistic dusk roost, seasonal amplitude, deterministic predator presence, and a
real coherence-gate pull toward the flock centroid) — only its `predator_rate=0.296` default is
empirically grounded (Goodenough et al. 2017's own reported figure, read directly); the
dusk/seasonal curve shapes are disclosed as pymurmur's own smoothing choices, not claimed fits.
`murmur_dynamic_vision_range` (a flock-wide adaptive perception-radius feedback loop, target
defaulting to `6.5` — the midpoint of this project's own Y-a hard-gate figure, Young et al. 2013)
was the first plugin to actually mutate `SimView::core_params` rather than only read it.
`murmur_neighbor_adaptive_speed` reused the same G1/G3-powered shape `murmur_boid_state_machine`
established, but as a continuous linear interpolation instead of a discrete cutoff.
`murmur_speed_noise` — a smoothed, downward-only stochastic per-boid speed-cap wobble — was the
plugin that motivated fixing **G8**: `StepHook::post_steer` previously had no path to genuine,
`base_seed`-tied randomness at all (every prior hook happened to be deterministic), verified two
ways: the same `base_seed` reproduces a bit-identical run, and a strong noise amplitude
measurably lowers a real flock's mean speed. `murmur_wander` — a bounded attractor pull toward
the flock's own live centroid (not a fixed point, unlike `murmur_influencer`/`murmur_field`) —
reused `murmur_field`'s own 11-term Lissajous+envelope curve formula plus a closed-form analytic
heading. `murmur_ripple` — three evenly time-staggered travelling Gaussian pulse rings expanding
from the flock's own live centroid, publishing each boid's own `ripple_envelope_sum` and using it
as a downward-only speed-cap wobble via G3. `murmur_obstacles` — the last item, full SDF+CSG
(`Primitive::Sphere`/`Box`/`Cylinder`, `union`/`subtract`, a numerical central-finite-difference
gradient, collision detection published via `is_colliding`) — checked and found neither G1's own
parallel-seam half nor G2 needed; "kinematic surface correction" is a soft avoidance force
(reusing `murmur_sphere_soft_domain`'s own inverse-distance push formula) rather than a hard
positional clamp, verified end to end: a flock spawned straddling a placed sphere demonstrably
moves out of it. G2 (a post-integration position-correction seam for *pairwise* boid–boid
collision) remains the one open architectural gap — no named plugin in the catalogue needs it.

**Post-Phase-18 audit pass (2026-08-07).** A full-repository audit (fresh build/test/clippy/fmt,
every plugin crate's own wiring, a `design/*.md` alignment check) found the codebase otherwise
clean, plus three real, worth-fixing items: `murmur_young::register()` was the one plugin in the
whole catalogue still panicking on a malformed override instead of falling back to defaults
(fixed); `batch.rs`'s own module doc and `Command::AddObstacle`/`SetEnvironment` doc comments
still said "no `obstacles`/`ecology` plugin exists yet" — true when written, false since Phase 18
(fixed); and `design/05_viz_contract.md`'s own checkpoint schema (`state`/`speed_mult`/
`threat_proximity`/`panic`/`blackening`/`spin`/`environment`/`obstacles`/`wander`/`ripple`/
`dynamic_vision_range`) was never actually wired into `Checkpoint`, even though every producing
plugin now exists. Fixed generically: `StepHook`/`SteeringModifier` both gained
`checkpoint_boid_fields`/`checkpoint_scene_fields` default methods (`murmur_core` never
references a specific plugin by name to collect them — each hook opts in on its own), `Checkpoint`
now carries the merged result, `murmur_ffi`'s C ABI and `murmur_py`'s `Snapshot` were both
extended to surface it (`CBoidSnapshot`/`CCheckpoint` gained `has_x`/`x` field pairs and new
fixed-size structs for `CEnvironment`/`CWanderState`/`CRippleState`/`CObstacleNode`; `Snapshot`
gained `boid_state`/`speed_mult`/`threat_proximity`/`panic`/`blackening`/`spin` NaN-sentinel
arrays and a `scene` dict), and `murmur_ffi`'s own `full_registry()` — previously missing 8 of
the plugins whose state it now carries — was brought up to date so the C ABI can actually compose
them. `pipeline.rs`/`batch.rs`/`murmur_predator_fsm`'s own `#[cfg(test)]` modules (the project's
three largest files) were split into sibling `tests.rs` files for navigability — no public API
change.

**`design/01_core.md` §4.1's cross-plugin `validate()` (2026-08-07, same day).** The one item the
audit above left deliberately open, built as its own follow-up. The doc's two named examples
(`phi_p + phi_a > 1.0` rejected, `HashGrid`'s `cell_size` silently snapped to `vision_radius`)
turned out to both be single-plugin self-checks, not true plugin-vs-plugin comparisons — the
`phi_p + phi_a` one was already handled by `PearceParamsBuilder::build()`'s own existing
rejection, so the only real gap was the `cell_size` clamp. Built generically anyway, ready for a
real cross-plugin pair when one is proposed: all 8 socket traits (`FlockingMode`,
`SteeringModifier`, `Domain`, `SpatialIndex`, `NeighborSelection`, `SpeedModel`, `Initializer`,
`NoiseSource`) gained default `resolved_params(&self) -> PluginParams` (reusing the existing
`PluginParams(HashMap<String, f64>)` as the erasure, rather than inventing the doc's
named-but-never-defined `ErasedPluginParams`) and `validate_and_fix(&mut self, core: &CoreParams,
others: &[(&str, PluginParams)]) -> Vec<Warning>` methods. `Simulation::new()` now snapshots every
socket's `resolved_params()` before any correction runs (so mutating one socket never needs to
alias another's live trait object), then calls each `validate_and_fix` in turn and returns
`Result<(Self, Vec<Warning>), ConfigError>` — a signature change that touched every call site in
the workspace. `HashGrid` is the one live implementer: a `cell_size` that disagrees with
`vision_radius` is corrected in place and reported, never rejected. Surfaced through both
consumer surfaces: `murmur_ffi` gained `murmur_warning_count`/`murmur_warning_message` (mirroring
the existing `murmur_last_error_message` convention), and `murmur_py`'s `Simulation` gained a
`warnings()` method. Verified end-to-end (not just unit-level) with a real `Simulation`
composition, the C smoke test, and a Python test, each deliberately setting a mismatched
`cell_size` and checking the correction + `Warning` both land.

569 Rust tests (up from 565), 108 pytest tests (up from 106). `design/01_core.md` §4.1 is now
fully closed. What's still open: `Command::AddObstacle`/`RemoveObstacle`/`SetEnvironment`'s live
write-routing, native H₂/`RequestMetric` routing (`consensus_degree`/`h2_result`), and G2
(pairwise boid–boid collision) — no named plugin needs it yet.
