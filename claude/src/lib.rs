//! No library code — this crate exists solely so the workspace root can own `tests/`, the
//! workspace-level architecture-enforcement guards (roadmap.md Phase 2b) that aren't owned by
//! any one crate (e.g. "does `murmur_core`'s `Cargo.toml` stay free of `murmur_*` plugin
//! dependencies"). Cargo requires a package to have at least one target; this is that target.
