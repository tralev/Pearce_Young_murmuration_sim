#!/usr/bin/env bash
# Reproduces the Pearce/Young validation run: builds the Rust extension, then runs the 10
# results + 5 hard gates against a real Simulation (design/03_observables_bindings.md §4).
#
# Usage: claude/scripts/reproduce_science.sh
#
# Takes ~1-2 minutes (Phase 7/9's acceptance run includes a 10,000-step-per-phi_p P-d sweep).
# Exits non-zero if any of the 5 hard gates fail — see README.md "Current acceptance status"
# for this build's honestly-reported current pass/fail state before assuming a red exit here
# means something is broken versus already a known, documented calibration gap.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements-dev.txt
maturin develop --release -m crates/murmur_py/Cargo.toml
python -m murmuration.validate.acceptance
