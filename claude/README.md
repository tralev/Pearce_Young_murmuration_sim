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

As of this build, running the harness gives (updated 2026-08-05, Phase 14's second calibration
pass — see `sci/param_table.md` for the full mechanistic investigation and the empirical tests
behind it):

| Gate | Result | Notes |
|---|---|---|
| ★ P-a (internal opacity) | **PASS** | Θ̄ ∈ [0.25, 0.35] at N ∈ {400, 800, 1600} |
| ★ P-c (density scaling) | FAIL | exponent = 0.031 (target [−0.7, −0.3]) — now flat rather than positive, large improvement, not yet in range |
| ★ P-d (no fragmentation) | **PASS** | `R_max` stays bounded (13–21) over 10⁴ steps at every tested `phi_p`/`phi_a` pair |
| ★ Y-a (m* ≈ 6–7) | FAIL | m* = 3 (was 2) — improved, not yet in range |
| ★ Y-c (m* vs thickness trend) | **PASS** | correct-sign trend, holding across all three phenotypes |

**3 of 5 hard gates now pass** (P-a, P-d, Y-c), up from 2 (P-a, Y-c). The `"murmuration"`
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
now genuinely fixed, and both remaining gates improved substantially alongside it (P-c's
exponent moved from ≈+1.02 to ≈0.03; Y-a's m* moved from 2 to 3, with the sensing graph now
confirmed fully connected at every tested `m`). Both are report-confirmed as *closer*, not
newly caused by anything this pass changed — closing them fully is separate, ongoing tuning
work, not evidence the fragmentation fix was wrong.

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
this environment — `rustup` isn't installed here; see that crate's module doc. Track C
(the deferred plugin catalogue — obstacles, ecology, H₂'s Rust-native eigensolver path, etc.)
is not built.
