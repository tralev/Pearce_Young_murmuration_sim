//! Shared text-scanning helpers for the workspace-level architecture-enforcement guards
//! (roadmap.md Phase 2b). These are deliberately simple text scans, not a real Rust parser —
//! good enough for the guards' purpose (catching drift/regressions), not a substitute for
//! `cargo metadata`/`syn`-based tooling if a guard ever needs to get stricter.
//!
//! Each `tests/*.rs` file compiles this module into its own separate test binary, and no
//! single guard uses every helper here — hence the blanket allow rather than per-binary
//! dead-code noise.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The workspace root — `CARGO_MANIFEST_DIR` for this package IS the workspace root, since
/// this package exists solely to own `tests/` (see `src/lib.rs`).
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn crates_dir() -> PathBuf {
    workspace_root().join("crates")
}

/// Every `.rs` file under `crates/`, recursively, excluding `target/`.
pub fn all_rust_source_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(&crates_dir(), &mut out);
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("target") {
                continue;
            }
            walk(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

pub fn read_all(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

/// An approximation of "test code only": for a file under any `tests/` directory, the whole
/// file; for a file with an inline `#[cfg(test)] mod ...`, everything from that marker onward.
/// Text-based, not scope-aware — good enough to bias a drift scan toward test code without a
/// full parser.
pub fn test_only_text(paths: &[PathBuf]) -> String {
    let mut out = String::new();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if path.components().any(|c| c.as_os_str() == "tests") {
            out.push_str(&text);
            out.push('\n');
        } else if let Some(idx) = text.find("#[cfg(test)]") {
            out.push_str(&text[idx..]);
            out.push('\n');
        }
    }
    out
}
