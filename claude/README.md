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

As of this build, running the harness gives:

| Gate | Result | Notes |
|---|---|---|
| ★ P-a (internal opacity) | **PASS** | Θ̄ ∈ [0.25, 0.35] at N ∈ {400, 800, 1600} |
| ★ P-c (density scaling) | FAIL | exponent comes out positive (flock growing denser with N), not ≈ −0.5 |
| ★ P-d (no fragmentation) | FAIL | `R_max` grows steadily rather than saturating over 10⁴ steps |
| ★ Y-a (m* ≈ 6–7) | FAIL | comes out as 2 |
| ★ Y-c (m* vs thickness trend) | FAIL | downstream of Y-a's issue |

P-c/P-d/Y-a/Y-c failing is coherent, not random: `R_max` growing steadily (P-d) means the flock
isn't staying self-bounded at long time scales under this build's default `body_radius`/
`vision_radius`/density combination, which is exactly what drags density scaling (P-c) and the
H₂ graph structure (Y-a/Y-c) off target too. This is a **parameter calibration gap, not a
mechanism bug** — `occlude()` itself is unit-tested against all its documented invariants
(§7 of `design/01_core.md`), and P-a passing shows the internal-opacity feedback loop works
correctly at the time scale it was checked at. Closing this gap is the job the design's
pre-coding numpy-prototype de-risking step (`roadmap.md` §3 item 1) was meant to do before any
Rust was written; that step wasn't produced separately during this build, so the gap is
inherited here rather than hidden. Tightening `body_radius`/`vision_radius`/initial density
against that kind of calibration pass — or against `pymurmur` fixtures, once available — is the
next real step, not a re-run of this harness with different guesses.

There's a second, related gap worth naming: the design's Phase 7 acceptance also calls for
comparison against two fixture oracles (a tight-tolerance f64 numpy prototype, and a loose-
tolerance f32 `pymurmur` CSV oracle). Neither exists in this environment — the numpy prototype
is itself pre-coding-prep work that wasn't produced separately, and `pymurmur` (`git_mur`) is
an external reference repository not present here. `murmuration.validate` is built and honestly
run against this Rust implementation directly; it does not compare against either oracle.

## Layout

```
crates/
├── murmur_core/         # infrastructure only — see its module docs
├── murmur_conformance/  # per-trait plugin-conformance test harness (dev-dependency)
├── murmur_py/           # pyo3 + numpy bindings (compiles to python/murmuration/_core)
└── plugins/             # every concrete algorithm/strategy, one crate each
python/
├── murmuration/          # the Python package (re-exports _core; analysis/, validate/)
└── tests/                 # pytest suite
tests/                    # workspace-level architecture-enforcement guards
fixtures/golden/          # pinned state_hash regression fixtures
```

## Notes on scope

This build follows `roadmap.md`'s phases 1–9 (Track A: the science core + Python path).
Track B (the native/C-ABI batch+checkpoint contract, `run_batch`'s full `CheckpointBuffer`/
`Command` machinery) and Track C (the deferred plugin catalogue — obstacles, ecology, H₂'s
Rust-native eigensolver path, etc.) are not built. `run_batch`/`run_batch_with_budget` here are
Track A's minimal loop-only versions.
