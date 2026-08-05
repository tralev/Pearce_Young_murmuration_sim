//! Phase 12 exit gate (roadmap.md §5): "a scripted headless variant exists as an automatable
//! regression proxy." This is that proxy — runs the compiled `reference_desktop` binary and
//! asserts it exits successfully with a `PASS` summary line, in both its headless (no
//! human-readable rendering needed to judge pass/fail) and normal (rendering + interpolation +
//! command-injection output) modes.

use std::process::Command;

#[test]
fn headless_run_exits_successfully_and_reports_pass() {
    let output = Command::new(env!("CARGO_BIN_EXE_reference_desktop"))
        .args([
            "--headless",
            "--boid-count",
            "60",
            "--steps",
            "20",
            "--stride",
            "5",
        ])
        .output()
        .expect("failed to run reference_desktop binary");
    assert!(
        output.status.success(),
        "reference_desktop exited with failure.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim_end(),
        "PASS",
        "headless mode should print nothing but the PASS/FAIL summary, got: {stdout}"
    );
}

#[test]
fn non_headless_run_also_succeeds_and_renders_frames() {
    let output = Command::new(env!("CARGO_BIN_EXE_reference_desktop"))
        .args(["--boid-count", "40", "--steps", "10", "--stride", "5"])
        .output()
        .expect("failed to run reference_desktop binary");
    assert!(
        output.status.success(),
        "reference_desktop exited with failure.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("step "),
        "expected rendered checkpoint lines, got: {stdout}"
    );
    assert!(stdout.contains("interpolating steps"));
    assert!(stdout.contains("injected AddPredator"));
    assert!(stdout.trim_end().ends_with("PASS"));
}

#[test]
fn default_run_with_no_args_also_succeeds() {
    let output = Command::new(env!("CARGO_BIN_EXE_reference_desktop"))
        .output()
        .expect("failed to run reference_desktop binary");
    assert!(
        output.status.success(),
        "default-args run exited with failure.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
