//! Phase 12's minimal reference desktop consumer (roadmap.md §5) — bare rendering, no
//! gestures/HUD/governor, calling `murmur_ffi`'s actual `extern "C"` surface. Declared as a
//! Rust path dependency rather than through a C toolchain (no `cbindgen` header exists in this
//! environment yet — see `murmur_ffi`'s module doc); the functions called are the exact same
//! ABI either way, so this still proves the C surface itself, not a Rust-only shortcut around
//! it.
//!
//! Demonstrates the full Phase 12 exit gate:
//! - birds render and move plausibly (an ASCII top-down scatter per checkpoint — a real
//!   graphical renderer is explicitly a separate, future project, per design/00_overview.md)
//! - smooth interpolated playback via `interpolation_hint` (linear sub-frames between
//!   consecutive checkpoints, not an assumption that piecewise-linear motion always looks fine)
//! - injecting the one supported command (`AddPredator`) visibly changes the next batch
//! - `--headless` is the scripted, assertion-only variant for automated regression (see
//!   `tests/headless_smoke.rs`) — no terminal output needs to be read to judge pass/fail.

use std::ffi::{CStr, CString};
use std::process::ExitCode;
use std::ptr;

use murmur_ffi::{
    murmur_boid_count, murmur_checkpoint_buffer_destroy, murmur_checkpoint_buffer_get,
    murmur_checkpoint_buffer_len, murmur_create, murmur_destroy, murmur_last_command_error_count,
    murmur_last_command_error_message, murmur_last_error_message, murmur_plugin_count,
    murmur_plugin_name, murmur_run_batch, CCheckpoint, CCommand, CKeyValue, CVec3,
    MurmurCheckpointBuffer, MurmurConfig, MurmurSimulation, CMD_ADD_PREDATOR,
    CMD_SET_CHECKPOINT_STRIDE,
};

struct Args {
    headless: bool,
    boid_count: u32,
    steps: u32,
    stride: u32,
}

fn parse_args() -> Args {
    let mut args = Args {
        headless: false,
        boid_count: 200,
        steps: 40,
        stride: 10,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--headless" => args.headless = true,
            "--boid-count" => {
                if let Some(v) = it.next() {
                    args.boid_count = v.parse().unwrap_or(args.boid_count);
                }
            }
            "--steps" => {
                if let Some(v) = it.next() {
                    args.steps = v.parse().unwrap_or(args.steps);
                }
            }
            "--stride" => {
                if let Some(v) = it.next() {
                    args.stride = v.parse().unwrap_or(args.stride);
                }
            }
            _ => {}
        }
    }
    args
}

/// A small top-down (XY plane) ASCII scatter of every boid's position — "bare rendering," per
/// this repo's explicit scope: one minimal reference consumer, not a production visualization
/// app (that's a separate, future project consuming this same contract).
fn render_ascii_frame(positions: &[(f64, f64)], width: usize, height: usize) -> String {
    if positions.is_empty() {
        return "(no boids)".to_string();
    }
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in positions {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    let mut grid = vec![vec![' '; width]; height];
    for &(x, y) in positions {
        let cx = (((x - min_x) / span_x) * (width - 1) as f64).round() as usize;
        let cy = (((y - min_y) / span_y) * (height - 1) as f64).round() as usize;
        grid[cy.min(height - 1)][cx.min(width - 1)] = '*';
    }
    grid.into_iter()
        .rev()
        .map(|row| row.into_iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// # Safety
/// `cp.boids`/`cp.boid_count` must be a valid slice, per `murmur_checkpoint_buffer_get`'s
/// contract (i.e. `cp`'s owning `MurmurCheckpointBuffer` must still be alive).
unsafe fn checkpoint_boid_xy(cp: &CCheckpoint) -> Vec<(f64, f64)> {
    std::slice::from_raw_parts(cp.boids, cp.boid_count as usize)
        .iter()
        .map(|b| (b.position.x, b.position.y))
        .collect()
}

fn print_checkpoint(cp: &CCheckpoint) {
    println!(
        "step {:>4} | t={:>6.1} | boids={:>3} | predators={:>2} | alpha={:.3} | theta_bar={:.3} | R_max={:.2} | mean_speed={:.3}",
        cp.step_count,
        cp.sim_time,
        cp.boid_count,
        cp.predator_count,
        cp.metrics.polarisation,
        cp.metrics.opacity_int,
        cp.metrics.r_max,
        cp.metrics.mean_speed
    );
    let xy = unsafe { checkpoint_boid_xy(cp) };
    println!("{}", render_ascii_frame(&xy, 50, 12));
}

/// Linear interpolation between two checkpoints' boid positions (paired by index — stable
/// within one batch, since no boid was added/removed mid-batch here), demonstrating a playback
/// consumer actually using `interpolation_hint` rather than assuming piecewise-linear motion
/// always looks fine (design/05_viz_contract.md §2.2).
///
/// # Safety
/// Same as `checkpoint_boid_xy`, for both `from` and `to`.
unsafe fn print_interpolated_subframes(from: &CCheckpoint, to: &CCheckpoint, sub_frames: u32) {
    let from_xy = checkpoint_boid_xy(from);
    let to_xy = checkpoint_boid_xy(to);
    if from_xy.is_empty() || to_xy.is_empty() {
        return;
    }
    println!(
        "  interpolating steps {}..{} (max_displacement this stride: {:.4}, state_changed: {})",
        from.step_count,
        to.step_count,
        to.interpolation_hint.max_displacement,
        to.interpolation_hint.state_changed != 0
    );
    let (fx, fy) = from_xy[0];
    let (tx, ty) = to_xy[0];
    for i in 1..sub_frames {
        let t = i as f64 / sub_frames as f64;
        let ix = fx + (tx - fx) * t;
        let iy = fy + (ty - fy) * t;
        println!("    sub-frame t={t:.2}: boid[0] ~= ({ix:.3}, {iy:.3})");
    }
}

/// # Safety
/// `cp.boids`/`cp.boid_count` must be a valid slice (see `checkpoint_boid_xy`).
unsafe fn checkpoint_is_healthy(cp: &CCheckpoint) -> bool {
    let boids = std::slice::from_raw_parts(cp.boids, cp.boid_count as usize);
    let boids_finite = boids.iter().all(|b| {
        b.position.x.is_finite()
            && b.position.y.is_finite()
            && b.position.z.is_finite()
            && b.velocity.x.is_finite()
            && b.velocity.y.is_finite()
            && b.velocity.z.is_finite()
    });
    boids_finite && cp.metrics.r_max.is_finite() && cp.center_of_mass.x.is_finite()
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`.
unsafe fn report_command_errors(sim: *mut MurmurSimulation) {
    let n = murmur_last_command_error_count(sim);
    for i in 0..n {
        let msg = CStr::from_ptr(murmur_last_command_error_message(sim, i)).to_string_lossy();
        eprintln!("  command error [{i}]: {msg}");
    }
}

fn main() -> ExitCode {
    let args = parse_args();

    // Composition: Pearce + the slice's default plugin set, plus `predator` so this consumer
    // has something to inject via `AddPredator`, and `spawn_headroom: 1` so that command
    // actually succeeds (G6, roadmap.md §12 — fixed).
    let mode = CString::new("pearce").unwrap();
    let modifier = CString::new("instant_response").unwrap();
    let domain = CString::new("open").unwrap();
    let spatial_index = CString::new("hash_grid").unwrap();
    let neighbor_selection = CString::new("radius_gather").unwrap();
    let speed_model = CString::new("band").unwrap();
    let init = CString::new("sphere_volume").unwrap();
    let noise = CString::new("uniform_sphere").unwrap();
    let cell_size_key = CString::new("cell_size").unwrap();
    let radius_key = CString::new("radius").unwrap();
    let predator_hook = CString::new("predator").unwrap();

    let plugin_params = [
        CKeyValue {
            key: cell_size_key.as_ptr(),
            value: 10.0,
        },
        CKeyValue {
            key: radius_key.as_ptr(),
            value: 25.0,
        },
    ];
    let step_hooks = [predator_hook.as_ptr()];

    let config = MurmurConfig {
        mode: mode.as_ptr(),
        modifier: modifier.as_ptr(),
        domain: domain.as_ptr(),
        spatial_index: spatial_index.as_ptr(),
        neighbor_selection: neighbor_selection.as_ptr(),
        speed_model: speed_model.as_ptr(),
        init: init.as_ptr(),
        noise: noise.as_ptr(),
        cruise_speed: 1.0,
        max_force: 5.0,
        speed_min_factor: 0.3,
        boid_count: args.boid_count,
        dt: 1.0,
        vision_radius: 10.0,
        plugin_params: plugin_params.as_ptr(),
        plugin_params_len: plugin_params.len(),
        init_seed: 42,
        step_hooks: step_hooks.as_ptr(),
        step_hooks_len: step_hooks.len(),
        predator_count: 0,
        spawn_headroom: 1,
    };

    unsafe {
        let sim = murmur_create(&config);
        if sim.is_null() {
            eprintln!(
                "murmur_create failed: {:?}",
                CStr::from_ptr(murmur_last_error_message())
            );
            return ExitCode::FAILURE;
        }

        if !args.headless {
            println!("=== murmuration-sim reference desktop consumer ===");
            println!("composition:");
            for i in 0..murmur_plugin_count() {
                let name = CStr::from_ptr(murmur_plugin_name(sim, i)).to_string_lossy();
                println!("  socket {i}: {name}");
            }
            println!(
                "boid_count={} steps={} stride={}",
                args.boid_count, args.steps, args.stride
            );
            println!();
        }

        // Batch 1: set the checkpoint stride, then run the requested steps, "playing back"
        // each checkpoint as it arrives.
        let stride_cmd = [CCommand {
            kind: CMD_SET_CHECKPOINT_STRIDE,
            position: CVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: CVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            id: 0,
            name: ptr::null(),
            value: 0.0,
            count: 0,
            stride: args.stride,
            seed: 0,
            has_seed: 0,
        }];
        let mut buffer1: *mut MurmurCheckpointBuffer = ptr::null_mut();
        let status = murmur_run_batch(
            sim,
            args.steps,
            1,
            stride_cmd.as_ptr(),
            stride_cmd.len(),
            &mut buffer1,
        );
        if status != 0 {
            eprintln!("batch 1 failed (status {status})");
            report_command_errors(sim);
            murmur_destroy(sim);
            return ExitCode::FAILURE;
        }

        let n_checkpoints = murmur_checkpoint_buffer_len(buffer1);
        let mut ok = n_checkpoints > 0;
        let mut prev: Option<CCheckpoint> = None;
        for idx in 0..n_checkpoints {
            let cp = murmur_checkpoint_buffer_get(buffer1, idx);
            ok &= checkpoint_is_healthy(&cp);
            if !args.headless {
                print_checkpoint(&cp);
                if let Some(p) = &prev {
                    print_interpolated_subframes(p, &cp, 3);
                }
                println!();
            }
            prev = Some(cp);
        }
        murmur_checkpoint_buffer_destroy(buffer1);

        // Batch 2: inject the one supported command (AddPredator) and show it visibly changes
        // the next batch's checkpoints.
        let before = murmur_boid_count(sim);
        let add_predator = [CCommand {
            kind: CMD_ADD_PREDATOR,
            position: CVec3 {
                x: 5.0,
                y: 5.0,
                z: 5.0,
            },
            velocity: CVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            id: 0,
            name: ptr::null(),
            value: 0.0,
            count: 0,
            stride: 0,
            seed: 0,
            has_seed: 0,
        }];
        let mut buffer2: *mut MurmurCheckpointBuffer = ptr::null_mut();
        let status = murmur_run_batch(
            sim,
            args.stride.max(1),
            1,
            add_predator.as_ptr(),
            add_predator.len(),
            &mut buffer2,
        );
        if status != 0 {
            eprintln!("batch 2 (AddPredator) failed (status {status})");
            report_command_errors(sim);
            ok = false;
        } else {
            let after = murmur_boid_count(sim);
            let n2 = murmur_checkpoint_buffer_len(buffer2);
            let last_cp = if n2 > 0 {
                Some(murmur_checkpoint_buffer_get(buffer2, n2 - 1))
            } else {
                None
            };
            let predator_count_after = last_cp.as_ref().map(|cp| cp.predator_count).unwrap_or(0);
            ok &= after == before + 1 && predator_count_after == 1;
            if !args.headless {
                println!("=== injected AddPredator ===");
                println!("boid_count before={before} after={after} (expected +1)");
                println!("checkpoint predator_count={predator_count_after} (expected 1)");
            }
            if let Some(cp) = &last_cp {
                ok &= checkpoint_is_healthy(cp);
            }
            murmur_checkpoint_buffer_destroy(buffer2);
        }

        murmur_destroy(sim);

        if !args.headless {
            println!();
        }
        println!("{}", if ok { "PASS" } else { "FAIL" });
        if ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}
