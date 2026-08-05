//! Phase 11 exit gate (roadmap.md §5): "Rust-side test harness linking the compiled cdylib
//! round-trips correctly." `murmur_ffi`'s own `#[cfg(test)]` suite (src/lib.rs) already proves
//! the `extern "C"` functions work when called from Rust — but that never proves the
//! cbindgen-generated header (`include/murmur_ffi.h`) itself is valid, linkable C. This test
//! closes that gap for real: it compiles and runs an actual C program
//! (`tests/c_smoke/main.c`) against the generated header and the compiled `cdylib`.
//!
//! Requires `cc` (present via Xcode Command Line Tools on this platform) and that
//! `include/murmur_ffi.h` has already been generated (`cbindgen --config cbindgen.toml
//! --output include/murmur_ffi.h`, run from this crate's directory) — not regenerated
//! automatically here, since this crate has no build-time dependency on `cbindgen` itself
//! (keeping a code-generation tool out of the normal build graph, matching design/05 §4's
//! "hand-written extern C to start" decision, D7).

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_target_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(Path::parent) // claude/ (workspace root)
        .expect("murmur_ffi should be two directories below the workspace root");
    workspace_root.join("target")
}

fn find_dylib(target_dir: &Path) -> PathBuf {
    for profile in ["debug", "release"] {
        let candidate = target_dir.join(profile).join("libmurmur_ffi.dylib");
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "couldn't find libmurmur_ffi.dylib under {}/{{debug,release}} — run `cargo build -p murmur_ffi` first",
        target_dir.display()
    );
}

#[test]
fn header_and_dylib_link_and_run_from_real_c() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let header_dir = manifest_dir.join("include");
    let header_file = header_dir.join("murmur_ffi.h");
    assert!(
        header_file.exists(),
        "{} doesn't exist — generate it first: cd crates/murmur_ffi && \
         <cbindgen> --config cbindgen.toml --output include/murmur_ffi.h",
        header_file.display()
    );
    let c_source = manifest_dir.join("tests/c_smoke/main.c");
    let target_dir = workspace_target_dir();
    let dylib = find_dylib(&target_dir);
    let dylib_dir = dylib.parent().unwrap();

    let out_binary = target_dir.join("c_smoke_test_bin");

    let compile = Command::new("cc")
        .arg(&c_source)
        .arg("-I")
        .arg(&header_dir)
        .arg("-L")
        .arg(dylib_dir)
        .arg("-lmurmur_ffi")
        .arg("-o")
        .arg(&out_binary)
        .output()
        .expect("failed to invoke cc");
    assert!(
        compile.status.success(),
        "cc failed to compile/link the C smoke test:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&out_binary)
        .env("DYLD_LIBRARY_PATH", dylib_dir)
        .output()
        .expect("failed to run the compiled C smoke test binary");
    assert!(
        run.status.success(),
        "C smoke test binary failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(stdout.contains("C_SMOKE_OK"), "unexpected output: {stdout}");
}
