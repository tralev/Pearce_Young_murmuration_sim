//! The extern "C" simulation-control surface (design/05_viz_contract.md §4, roadmap.md
//! Phase 11) — create/destroy, `run_batch`/`run_batch_with_budget`, checkpoint reads,
//! `describe`/`composition`. **Simulation-control only** (D19): no screen-to-world math, no
//! capability-probe function, no config-dump helper beyond what `composition()`-equivalents
//! already provide — those belong in a consumer, not here (design/05 §4).
//!
//! **Environment scope note.** `rustup` isn't installed in this environment, so
//! `aarch64-apple-ios`/`aarch64-linux-android` cross-compile targets aren't available, and the
//! QEMU ARM runtime smoke test can't run without them — genuinely blocked on tooling, not
//! silently skipped, same "document, don't fake" practice as the pymurmur-checkout gap
//! (roadmap.md §3 item 4). `cbindgen`, however, **is** installed (routed to
//! `/Volumes/.../.tooling/cargo-install` rather than the default `~/.cargo/bin`, to keep a
//! near-full boot volume out of it) and `include/murmur_ffi.h` is generated and checked in.
//! `tests/c_smoke.rs` compiles and runs a real C program (`tests/c_smoke/main.c`) against that
//! header and the built `cdylib` — proving the header is genuinely valid, linkable C, not just
//! plausible-looking output — closing that part of Phase 11's exit gate for real. Regenerate
//! the header after any change to this crate's public surface: `cd crates/murmur_ffi &&
//! <path-to-cbindgen> --config cbindgen.toml --output include/murmur_ffi.h`. There's no CI/
//! build-time check that the checked-in header is still in sync with the Rust source (that
//! would need `cbindgen` itself as a build dependency, a bigger change than this pass made) —
//! a real, if minor, staleness risk worth knowing about, not a solved problem.
//!
//! **Command encoding.** `Command`'s Rust-level `String`/`Option` fields aren't FFI-safe, so
//! `CCommand` is a flat, always-fully-sized struct with unused fields for whichever variant
//! it isn't (a standard, simple C ABI pattern for a control-plane API, not a hot path — no
//! need for a real C tagged union). `name` is the sole borrowed pointer among them (for
//! `SetParam`); it's read and copied into an owned `String` during the `murmur_run_batch*`
//! call and never retained past that call returning.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

use murmur_core::batch::{Checkpoint, Command};
use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation, Species, Vec3};

fn full_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_vicsek::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    murmur_predator::register(&mut reg);
    reg
}

thread_local! {
    /// Set by any FFI function that can fail before a `MurmurSimulation` exists (today, just
    /// `murmur_create`). Valid until the next call to a function that can set it, on the same
    /// thread — the standard C "last error" convention.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl AsRef<str>) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg.as_ref()).ok();
    });
}

/// Returns the most recent `murmur_create` failure's message on this thread, or null if there
/// hasn't been one (or it wasn't valid UTF-8/contained an interior NUL, which can't happen for
/// error messages we construct ourselves). Valid until the next call to `murmur_create` on
/// this thread.
#[no_mangle]
pub extern "C" fn murmur_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// # Safety
/// `ptr` must be null or a valid pointer to a NUL-terminated, UTF-8 C string, live for the
/// duration of this call.
unsafe fn cstr_to_string(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("unexpected null string pointer".to_string());
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(str::to_string)
        .map_err(|_| "string is not valid UTF-8".to_string())
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<CVec3> for Vec3 {
    fn from(v: CVec3) -> Vec3 {
        Vec3::new(v.x, v.y, v.z)
    }
}
impl From<Vec3> for CVec3 {
    fn from(v: Vec3) -> CVec3 {
        CVec3 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

#[repr(C)]
pub struct CKeyValue {
    pub key: *const c_char,
    pub value: f64,
}

/// The construction-time configuration (design/02_plugins.md §1's registry-resolution shape,
/// flattened for C). Every `*const c_char` here is read and copied during `murmur_create` only
/// — none are retained past that call returning.
#[repr(C)]
pub struct MurmurConfig {
    pub mode: *const c_char,
    pub modifier: *const c_char,
    pub domain: *const c_char,
    pub spatial_index: *const c_char,
    pub neighbor_selection: *const c_char,
    pub speed_model: *const c_char,
    pub init: *const c_char,
    pub noise: *const c_char,
    pub cruise_speed: f64,
    pub max_force: f64,
    pub speed_min_factor: f64,
    pub boid_count: u32,
    pub dt: f64,
    pub vision_radius: f64,
    pub plugin_params: *const CKeyValue,
    pub plugin_params_len: usize,
    pub init_seed: u64,
    pub step_hooks: *const *const c_char,
    pub step_hooks_len: usize,
    pub predator_count: u32,
    /// Extra `BoidColumns` capacity reserved for runtime-spawned boids
    /// (`CMD_ADD_PREDATOR`) — `SimConfig::spawn_headroom` (roadmap.md §12, G6). `0` means no
    /// live spawning will ever succeed for this simulation, same as omitting it.
    pub spawn_headroom: u32,
}

/// The 8 trait sockets, in the fixed order `Simulation::plugin_names()` returns them —
/// `murmur_plugin_name`'s `socket_index` indexes into this.
pub const MURMUR_SOCKET_COUNT: u32 = 8;

pub struct MurmurSimulation {
    inner: Simulation,
    /// Cached at construction — composition never changes for a simulation's lifetime (D14),
    /// so this is computed once, not per-call. Order matches `MURMUR_SOCKET_COUNT`'s sockets.
    plugin_name_cache: Vec<CString>,
    /// Populated by the most recent `murmur_run_batch`/`murmur_run_batch_with_budget` call
    /// that returned a nonzero (validation-error) status; cleared on the next successful call.
    last_command_errors: Vec<CString>,
}

/// # Safety
/// `config` must be null or point to a valid, fully-initialized `MurmurConfig` whose string/
/// array pointers are each valid for the duration of this call (see `MurmurConfig`'s doc).
/// Returns null on any failure — check `murmur_last_error_message()`.
#[no_mangle]
pub unsafe extern "C" fn murmur_create(config: *const MurmurConfig) -> *mut MurmurSimulation {
    if config.is_null() {
        set_last_error("murmur_create: config pointer is null");
        return ptr::null_mut();
    }
    let cfg = &*config;

    macro_rules! try_str {
        ($field:expr, $label:literal) => {
            match cstr_to_string($field) {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("murmur_create: {} {}", $label, e));
                    return ptr::null_mut();
                }
            }
        };
    }
    let mode = try_str!(cfg.mode, "mode:");
    let modifier = try_str!(cfg.modifier, "modifier:");
    let domain = try_str!(cfg.domain, "domain:");
    let spatial_index = try_str!(cfg.spatial_index, "spatial_index:");
    let neighbor_selection = try_str!(cfg.neighbor_selection, "neighbor_selection:");
    let speed_model = try_str!(cfg.speed_model, "speed_model:");
    let init = try_str!(cfg.init, "init:");
    let noise = try_str!(cfg.noise, "noise:");

    let mut plugin_params = PluginParams::new();
    if cfg.plugin_params_len > 0 {
        if cfg.plugin_params.is_null() {
            set_last_error("murmur_create: plugin_params is null but plugin_params_len > 0");
            return ptr::null_mut();
        }
        let kvs = std::slice::from_raw_parts(cfg.plugin_params, cfg.plugin_params_len);
        for kv in kvs {
            let key = match cstr_to_string(kv.key) {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("murmur_create: plugin_params key: {e}"));
                    return ptr::null_mut();
                }
            };
            plugin_params = plugin_params.with(key, kv.value);
        }
    }

    let mut step_hooks = Vec::with_capacity(cfg.step_hooks_len);
    if cfg.step_hooks_len > 0 {
        if cfg.step_hooks.is_null() {
            set_last_error("murmur_create: step_hooks is null but step_hooks_len > 0");
            return ptr::null_mut();
        }
        let ptrs = std::slice::from_raw_parts(cfg.step_hooks, cfg.step_hooks_len);
        for &p in ptrs {
            match cstr_to_string(p) {
                Ok(s) => step_hooks.push(s),
                Err(e) => {
                    set_last_error(format!("murmur_create: step_hooks entry: {e}"));
                    return ptr::null_mut();
                }
            }
        }
    }

    let core_params = match CoreParams::builder()
        .cruise_speed(cfg.cruise_speed)
        .max_force(cfg.max_force)
        .speed_min_factor(cfg.speed_min_factor)
        .boid_count(cfg.boid_count)
        .dt(cfg.dt)
        .vision_radius(cfg.vision_radius)
        .build()
    {
        Ok(p) => p,
        Err(e) => {
            set_last_error(format!("murmur_create: {e}"));
            return ptr::null_mut();
        }
    };

    let sim_config = SimConfig {
        mode,
        modifier,
        domain,
        spatial_index,
        neighbor_selection,
        speed_model,
        init,
        noise,
        core_params,
        plugin_params,
        init_seed: cfg.init_seed,
        step_hooks,
        predator_count: cfg.predator_count,
        spawn_headroom: cfg.spawn_headroom,
    };

    let registry = full_registry();
    match Simulation::new(sim_config, &registry) {
        Ok(inner) => {
            let plugin_name_cache = inner
                .plugin_names()
                .iter()
                .map(|&(_, name)| {
                    CString::new(name).unwrap_or_else(|_| CString::new("<invalid>").unwrap())
                })
                .collect();
            Box::into_raw(Box::new(MurmurSimulation {
                inner,
                plugin_name_cache,
                last_command_errors: Vec::new(),
            }))
        }
        Err(e) => {
            set_last_error(format!("murmur_create: {e}"));
            ptr::null_mut()
        }
    }
}

/// # Safety
/// `sim` must be either null or a pointer previously returned by `murmur_create` and not yet
/// passed to `murmur_destroy`.
#[no_mangle]
pub unsafe extern "C" fn murmur_destroy(sim: *mut MurmurSimulation) {
    if !sim.is_null() {
        drop(Box::from_raw(sim));
    }
}

#[no_mangle]
pub extern "C" fn murmur_plugin_count() -> u32 {
    MURMUR_SOCKET_COUNT
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`. Returns null if `socket_index` is out of
/// range. The returned pointer is valid until `murmur_destroy(sim)`.
#[no_mangle]
pub unsafe extern "C" fn murmur_plugin_name(
    sim: *const MurmurSimulation,
    socket_index: u32,
) -> *const c_char {
    if sim.is_null() {
        return ptr::null();
    }
    let sim = &*sim;
    match sim.plugin_name_cache.get(socket_index as usize) {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`.
#[no_mangle]
pub unsafe extern "C" fn murmur_session_id(sim: *const MurmurSimulation) -> u64 {
    if sim.is_null() {
        return 0;
    }
    (&*sim).inner.session_header().session_id
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`.
#[no_mangle]
pub unsafe extern "C" fn murmur_build_hash(sim: *const MurmurSimulation) -> u64 {
    if sim.is_null() {
        return 0;
    }
    (&*sim).inner.session_header().build_hash
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`.
#[no_mangle]
pub unsafe extern "C" fn murmur_boid_count(sim: *const MurmurSimulation) -> u32 {
    if sim.is_null() {
        return 0;
    }
    (&*sim).inner.boid_count()
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`.
#[no_mangle]
pub unsafe extern "C" fn murmur_step_count(sim: *const MurmurSimulation) -> u64 {
    if sim.is_null() {
        return 0;
    }
    (&*sim).inner.step_count()
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`. Returns null if `index` is out of range
/// for the error list left by the most recent failed `murmur_run_batch*` call. The pointer is
/// valid until the next `murmur_run_batch*` call on this `sim` or `murmur_destroy(sim)`.
#[no_mangle]
pub unsafe extern "C" fn murmur_last_command_error_count(sim: *const MurmurSimulation) -> usize {
    if sim.is_null() {
        return 0;
    }
    (&*sim).last_command_errors.len()
}

/// # Safety
/// See `murmur_last_command_error_count`.
#[no_mangle]
pub unsafe extern "C" fn murmur_last_command_error_message(
    sim: *const MurmurSimulation,
    index: usize,
) -> *const c_char {
    if sim.is_null() {
        return ptr::null();
    }
    match (&*sim).last_command_errors.get(index) {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

// `CCommand::kind` tag values. Public — any real consumer (this crate's own reference desktop
// consumer included, roadmap.md Phase 12) needs these to construct a valid `CCommand`; they
// were private until Phase 12 tried to use them from outside the crate and couldn't.
pub const CMD_ADD_PREDATOR: u8 = 0;
pub const CMD_REMOVE_PREDATOR: u8 = 1;
pub const CMD_ADD_OBSTACLE: u8 = 2;
pub const CMD_REMOVE_OBSTACLE: u8 = 3;
pub const CMD_SET_PARAM: u8 = 4;
pub const CMD_SET_ENVIRONMENT: u8 = 5;
pub const CMD_RESET: u8 = 6;
pub const CMD_SET_CHECKPOINT_STRIDE: u8 = 7;
pub const CMD_REQUEST_METRIC: u8 = 8;

/// A flat, always-fully-sized encoding of `murmur_core::batch::Command` — see module doc
/// "Command encoding." `kind` selects one of the `CMD_*` constants; only the fields that
/// variant actually uses are meaningful.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CCommand {
    pub kind: u8,
    pub position: CVec3,
    pub velocity: CVec3,
    pub id: u32,
    /// `SetParam` only — a borrowed, NUL-terminated UTF-8 string, read during the
    /// `murmur_run_batch*` call and not retained past it. Null for every other `kind`.
    pub name: *const c_char,
    pub value: f64,
    pub count: u32,
    pub stride: u32,
    pub seed: u64,
    pub has_seed: u8,
}

/// # Safety
/// `ptr` must be null (with `len == 0`) or point to `len` valid, initialized `CCommand`s; any
/// `name` pointers within them must be valid for the duration of this call.
unsafe fn decode_commands(ptr: *const CCommand, len: usize) -> Result<Vec<Command>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err("commands pointer is null but commands_len > 0".to_string());
    }
    let raw = std::slice::from_raw_parts(ptr, len);
    let mut out = Vec::with_capacity(len);
    for (i, c) in raw.iter().enumerate() {
        let cmd = match c.kind {
            CMD_ADD_PREDATOR => Command::AddPredator {
                position: c.position.into(),
                velocity: c.velocity.into(),
            },
            CMD_REMOVE_PREDATOR => Command::RemovePredator { id: c.id },
            CMD_ADD_OBSTACLE => Command::AddObstacle,
            CMD_REMOVE_OBSTACLE => Command::RemoveObstacle { id: c.id },
            CMD_SET_PARAM => Command::SetParam {
                name: cstr_to_string(c.name).map_err(|e| format!("command {i}: {e}"))?,
                value: c.value,
            },
            CMD_SET_ENVIRONMENT => Command::SetEnvironment,
            CMD_RESET => Command::Reset {
                count: c.count,
                seed: if c.has_seed != 0 { Some(c.seed) } else { None },
            },
            CMD_SET_CHECKPOINT_STRIDE => Command::SetCheckpointStride { stride: c.stride },
            CMD_REQUEST_METRIC => Command::RequestMetric,
            other => return Err(format!("command {i}: unknown kind {other}")),
        };
        out.push(cmd);
    }
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CBoidSnapshot {
    pub position: CVec3,
    pub velocity: CVec3,
    /// Not a direct cast of `Species` (a data-carrying `#[repr(u16)]` enum isn't guaranteed a
    /// simple C-compatible layout) — an explicit, documented encoding instead: `0` = Prey,
    /// `1` = Predator, `1000 + tag` = `Custom(tag)`.
    pub species_code: u16,
    pub theta: f64,
}

fn species_code(s: Species) -> u16 {
    match s {
        Species::Prey => 0,
        Species::Predator => 1,
        Species::Custom(tag) => 1000u16.saturating_add(tag),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPredatorSnapshot {
    pub position: CVec3,
    pub velocity: CVec3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CMetrics {
    pub polarisation: f64,
    pub nematic_order: f64,
    pub opacity_int: f64,
    pub opacity_ext: f64,
    pub r_max: f64,
    pub dispersion: f64,
    pub mean_nn: f64,
    pub mean_speed: f64,
    pub angular_momentum: f64,
    pub count: u32,
    pub step: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CInterpolationHint {
    pub max_displacement: f64,
    pub state_changed: u8,
}

/// One checkpoint's data, C-ABI shape (design/05 §2's per-boid/scene-level fields). `boids`/
/// `predators` point into arrays owned by the `MurmurCheckpointBuffer` this came from — valid
/// until `murmur_checkpoint_buffer_destroy` is called on that buffer, not just for this call.
#[repr(C)]
pub struct CCheckpoint {
    pub session_id: u64,
    pub step_count: u64,
    pub sim_time: f64,
    pub base_seed: u64,
    pub center_of_mass: CVec3,
    pub metrics: CMetrics,
    pub interpolation_hint: CInterpolationHint,
    pub boid_count: u32,
    pub boids: *const CBoidSnapshot,
    pub predator_count: u32,
    pub predators: *const CPredatorSnapshot,
}

/// Owns the repr(C)-converted per-checkpoint arrays so pointers handed out by
/// `murmur_checkpoint_buffer_get` stay valid for the buffer's lifetime.
pub struct MurmurCheckpointBuffer {
    checkpoints: Vec<Checkpoint>,
    c_boids: Vec<Vec<CBoidSnapshot>>,
    c_predators: Vec<Vec<CPredatorSnapshot>>,
}

impl MurmurCheckpointBuffer {
    fn from_checkpoints(checkpoints: Vec<Checkpoint>) -> Self {
        let c_boids = checkpoints
            .iter()
            .map(|cp| {
                cp.boids
                    .iter()
                    .map(|b| CBoidSnapshot {
                        position: b.position.into(),
                        velocity: b.velocity.into(),
                        species_code: species_code(b.species),
                        theta: b.theta,
                    })
                    .collect()
            })
            .collect();
        let c_predators = checkpoints
            .iter()
            .map(|cp| {
                cp.predators
                    .iter()
                    .map(|p| CPredatorSnapshot {
                        position: p.position.into(),
                        velocity: p.velocity.into(),
                    })
                    .collect()
            })
            .collect();
        MurmurCheckpointBuffer {
            checkpoints,
            c_boids,
            c_predators,
        }
    }
}

/// # Safety
/// `buf` must be a live pointer from a `murmur_run_batch*` call, not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn murmur_checkpoint_buffer_len(buf: *const MurmurCheckpointBuffer) -> usize {
    if buf.is_null() {
        return 0;
    }
    (&*buf).checkpoints.len()
}

/// # Safety
/// `buf` must be a live pointer from a `murmur_run_batch*` call. `index` must be `<
/// murmur_checkpoint_buffer_len(buf)` — out-of-range returns a zeroed `CCheckpoint` with null
/// array pointers rather than reading out of bounds.
#[no_mangle]
pub unsafe extern "C" fn murmur_checkpoint_buffer_get(
    buf: *const MurmurCheckpointBuffer,
    index: usize,
) -> CCheckpoint {
    let zeroed = CCheckpoint {
        session_id: 0,
        step_count: 0,
        sim_time: 0.0,
        base_seed: 0,
        center_of_mass: CVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        metrics: CMetrics {
            polarisation: 0.0,
            nematic_order: 0.0,
            opacity_int: 0.0,
            opacity_ext: 0.0,
            r_max: 0.0,
            dispersion: 0.0,
            mean_nn: 0.0,
            mean_speed: 0.0,
            angular_momentum: 0.0,
            count: 0,
            step: 0,
        },
        interpolation_hint: CInterpolationHint {
            max_displacement: 0.0,
            state_changed: 0,
        },
        boid_count: 0,
        boids: ptr::null(),
        predator_count: 0,
        predators: ptr::null(),
    };
    if buf.is_null() {
        return zeroed;
    }
    let buf = &*buf;
    let Some(cp) = buf.checkpoints.get(index) else {
        return zeroed;
    };
    let c_boids = &buf.c_boids[index];
    let c_predators = &buf.c_predators[index];
    CCheckpoint {
        session_id: cp.session_id,
        step_count: cp.step_count,
        sim_time: cp.sim_time,
        base_seed: cp.base_seed,
        center_of_mass: cp.center_of_mass.into(),
        metrics: CMetrics {
            polarisation: cp.metrics.polarisation,
            nematic_order: cp.metrics.nematic_order,
            opacity_int: cp.metrics.opacity_int,
            opacity_ext: cp.metrics.opacity_ext,
            r_max: cp.metrics.r_max,
            dispersion: cp.metrics.dispersion,
            mean_nn: cp.metrics.mean_nn,
            mean_speed: cp.metrics.mean_speed,
            angular_momentum: cp.metrics.angular_momentum,
            count: cp.metrics.count,
            step: cp.metrics.step,
        },
        interpolation_hint: CInterpolationHint {
            max_displacement: cp.interpolation_hint.max_displacement,
            state_changed: cp.interpolation_hint.state_changed as u8,
        },
        boid_count: c_boids.len() as u32,
        boids: c_boids.as_ptr(),
        predator_count: c_predators.len() as u32,
        predators: c_predators.as_ptr(),
    }
}

/// # Safety
/// `buf` must be null or a pointer previously returned by a `murmur_run_batch*` call and not
/// yet destroyed. Every `CCheckpoint` obtained from `buf` becomes dangling after this call.
#[no_mangle]
pub unsafe extern "C" fn murmur_checkpoint_buffer_destroy(buf: *mut MurmurCheckpointBuffer) {
    if !buf.is_null() {
        drop(Box::from_raw(buf));
    }
}

/// # Safety
/// `sim` must be a live pointer from `murmur_create`. `commands`/`commands_len` per
/// `decode_commands`'s contract. `out_buffer` must be a valid, writable `*mut
/// *mut MurmurCheckpointBuffer`.
///
/// Returns `0` on success (`*out_buffer` is set to a live buffer the caller must eventually
/// pass to `murmur_checkpoint_buffer_destroy`). Returns `1` if any command failed validation
/// (`*out_buffer` is set to null; the simulation is left completely untouched — no steps ran;
/// inspect `murmur_last_command_error_count`/`_message`). Returns `-1` for a malformed call
/// (null `sim`/`out_buffer`, or an unparseable command array) — check
/// `murmur_last_error_message`.
#[no_mangle]
pub unsafe extern "C" fn murmur_run_batch(
    sim: *mut MurmurSimulation,
    steps: u32,
    base_seed: u64,
    commands: *const CCommand,
    commands_len: usize,
    out_buffer: *mut *mut MurmurCheckpointBuffer,
) -> i32 {
    if sim.is_null() || out_buffer.is_null() {
        set_last_error("murmur_run_batch: null sim or out_buffer pointer");
        return -1;
    }
    let sim = &mut *sim;
    let cmds = match decode_commands(commands, commands_len) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(format!("murmur_run_batch: {e}"));
            *out_buffer = ptr::null_mut();
            return -1;
        }
    };
    match sim.inner.run_batch_checked(steps, base_seed, cmds) {
        Ok(buffer) => {
            sim.last_command_errors.clear();
            *out_buffer = Box::into_raw(Box::new(MurmurCheckpointBuffer::from_checkpoints(
                buffer.checkpoints,
            )));
            0
        }
        Err(errors) => {
            sim.last_command_errors = errors
                .into_iter()
                .map(|e| {
                    CString::new(format!("[{}] {}", e.index, e.reason))
                        .unwrap_or_else(|_| CString::new("<invalid error message>").unwrap())
                })
                .collect();
            *out_buffer = ptr::null_mut();
            1
        }
    }
}

/// # Safety
/// As `murmur_run_batch`, plus `out_all_done` must be a valid, writable `*mut u8`. On success,
/// `*out_all_done` is `1` if every requested step ran before `time_budget_ms` elapsed, `0` if
/// the batch stopped early (the returned buffer still holds every checkpoint captured before
/// the stop, and `sim`'s state is fully valid — no step is ever left partially applied).
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn murmur_run_batch_with_budget(
    sim: *mut MurmurSimulation,
    steps: u32,
    base_seed: u64,
    commands: *const CCommand,
    commands_len: usize,
    time_budget_ms: f64,
    out_buffer: *mut *mut MurmurCheckpointBuffer,
    out_all_done: *mut u8,
) -> i32 {
    if sim.is_null() || out_buffer.is_null() || out_all_done.is_null() {
        set_last_error("murmur_run_batch_with_budget: null pointer argument");
        return -1;
    }
    let sim = &mut *sim;
    let cmds = match decode_commands(commands, commands_len) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(format!("murmur_run_batch_with_budget: {e}"));
            *out_buffer = ptr::null_mut();
            return -1;
        }
    };
    match sim
        .inner
        .run_batch_with_budget_checked(steps, base_seed, cmds, time_budget_ms)
    {
        Ok((buffer, all_done)) => {
            sim.last_command_errors.clear();
            *out_all_done = all_done as u8;
            *out_buffer = Box::into_raw(Box::new(MurmurCheckpointBuffer::from_checkpoints(
                buffer.checkpoints,
            )));
            0
        }
        Err(errors) => {
            sim.last_command_errors = errors
                .into_iter()
                .map(|e| {
                    CString::new(format!("[{}] {}", e.index, e.reason))
                        .unwrap_or_else(|_| CString::new("<invalid error message>").unwrap())
                })
                .collect();
            *out_buffer = ptr::null_mut();
            *out_all_done = 0;
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::mem::{align_of, size_of};

    fn c_str(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    /// Every string field lives at least as long as `murmur_create`'s call, which is all the
    /// contract requires — holding the owning `CString`s alongside the `MurmurConfig` that
    /// borrows them is the caller's job in real C too.
    struct ConfigStrings {
        mode: CString,
        modifier: CString,
        domain: CString,
        spatial_index: CString,
        neighbor_selection: CString,
        speed_model: CString,
        init: CString,
        noise: CString,
    }

    fn default_config_strings() -> ConfigStrings {
        ConfigStrings {
            mode: c_str("pearce"),
            modifier: c_str("instant_response"),
            domain: c_str("open"),
            spatial_index: c_str("hash_grid"),
            neighbor_selection: c_str("radius_gather"),
            speed_model: c_str("band"),
            init: c_str("sphere_volume"),
            noise: c_str("uniform_sphere"),
        }
    }

    fn default_config(s: &ConfigStrings, n: u32) -> MurmurConfig {
        MurmurConfig {
            mode: s.mode.as_ptr(),
            modifier: s.modifier.as_ptr(),
            domain: s.domain.as_ptr(),
            spatial_index: s.spatial_index.as_ptr(),
            neighbor_selection: s.neighbor_selection.as_ptr(),
            speed_model: s.speed_model.as_ptr(),
            init: s.init.as_ptr(),
            noise: s.noise.as_ptr(),
            cruise_speed: 1.0,
            max_force: 10.0,
            speed_min_factor: 0.3,
            boid_count: n,
            dt: 1.0,
            vision_radius: 5.0,
            plugin_params: ptr::null(),
            plugin_params_len: 0,
            init_seed: 7,
            step_hooks: ptr::null(),
            step_hooks_len: 0,
            predator_count: 0,
            spawn_headroom: 0,
        }
    }

    #[test]
    fn create_run_batch_read_checkpoint_destroy_round_trips_correctly() {
        unsafe {
            let strings = default_config_strings();
            let config = default_config(&strings, 20);
            let sim = murmur_create(&config);
            assert!(
                !sim.is_null(),
                "{:?}",
                CStr::from_ptr(murmur_last_error_message())
            );

            assert_eq!(murmur_plugin_count(), 8);
            assert_eq!(murmur_boid_count(sim), 20);
            assert_eq!(murmur_step_count(sim), 0);
            let mode_name = CStr::from_ptr(murmur_plugin_name(sim, 0)).to_str().unwrap();
            assert_eq!(mode_name, "pearce");

            let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
            let commands = [CCommand {
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
                stride: 5,
                seed: 0,
                has_seed: 0,
            }];
            let status = murmur_run_batch(
                sim,
                10,
                1,
                commands.as_ptr(),
                commands.len(),
                &mut out_buffer,
            );
            assert_eq!(status, 0);
            assert!(!out_buffer.is_null());
            assert_eq!(murmur_checkpoint_buffer_len(out_buffer), 2); // 10 / 5

            let cp0 = murmur_checkpoint_buffer_get(out_buffer, 0);
            assert_eq!(cp0.step_count, 5);
            assert_eq!(cp0.boid_count, 20);
            assert!(!cp0.boids.is_null());
            let boid0 = &*cp0.boids;
            assert!(boid0.position.x.is_finite());

            murmur_checkpoint_buffer_destroy(out_buffer);
            murmur_destroy(sim);
        }
    }

    #[test]
    fn invalid_command_leaves_sim_untouched_and_reports_errors() {
        unsafe {
            let strings = default_config_strings();
            let config = default_config(&strings, 5);
            let sim = murmur_create(&config);
            assert!(!sim.is_null());

            let bad_name = c_str("not_a_real_param");
            let commands = [CCommand {
                kind: CMD_SET_PARAM,
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
                name: bad_name.as_ptr(),
                value: 1.0,
                count: 0,
                stride: 0,
                seed: 0,
                has_seed: 0,
            }];
            let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
            let status = murmur_run_batch(
                sim,
                10,
                1,
                commands.as_ptr(),
                commands.len(),
                &mut out_buffer,
            );
            assert_eq!(status, 1);
            assert!(out_buffer.is_null());
            assert_eq!(murmur_step_count(sim), 0, "no step should have run");
            assert_eq!(murmur_last_command_error_count(sim), 1);
            let msg = CStr::from_ptr(murmur_last_command_error_message(sim, 0))
                .to_str()
                .unwrap();
            assert!(msg.contains("not_a_real_param"));

            murmur_destroy(sim);
        }
    }

    #[test]
    fn null_config_fails_cleanly_with_a_last_error_message() {
        unsafe {
            let sim = murmur_create(ptr::null());
            assert!(sim.is_null());
            let msg = CStr::from_ptr(murmur_last_error_message())
                .to_str()
                .unwrap();
            assert!(msg.contains("null"));
        }
    }

    #[test]
    fn repeated_create_destroy_cycles_do_not_crash_or_leak_obviously() {
        unsafe {
            let strings = default_config_strings();
            for _ in 0..200 {
                let config = default_config(&strings, 30);
                let sim = murmur_create(&config);
                assert!(!sim.is_null());
                let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
                let status = murmur_run_batch(sim, 3, 1, ptr::null(), 0, &mut out_buffer);
                assert_eq!(status, 0);
                murmur_checkpoint_buffer_destroy(out_buffer);
                murmur_destroy(sim);
            }
        }
    }

    #[test]
    fn repr_c_struct_sizes_and_alignments_are_stable_on_this_platform() {
        assert_eq!(size_of::<CVec3>(), 24);
        assert_eq!(align_of::<CVec3>(), 8);
        assert_eq!(size_of::<CBoidSnapshot>(), size_of::<CVec3>() * 2 + 8 + 8); // padding-aware
        assert_eq!(align_of::<CBoidSnapshot>(), 8);
        assert_eq!(align_of::<CCheckpoint>(), 8);
        assert_eq!(align_of::<CMetrics>(), 8);
    }

    #[test]
    fn run_batch_with_budget_stops_early_and_reports_all_done_false() {
        unsafe {
            let strings = default_config_strings();
            let config = default_config(&strings, 500);
            let sim = murmur_create(&config);
            assert!(!sim.is_null());

            let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
            let mut all_done: u8 = 1;
            let status = murmur_run_batch_with_budget(
                sim,
                1_000_000,
                1,
                ptr::null(),
                0,
                0.001,
                &mut out_buffer,
                &mut all_done,
            );
            assert_eq!(status, 0);
            assert_eq!(all_done, 0);
            assert!(!out_buffer.is_null());
            assert!(murmur_step_count(sim) < 1_000_000);

            murmur_checkpoint_buffer_destroy(out_buffer);
            murmur_destroy(sim);
        }
    }

    /// G6 (roadmap.md §12) is fixed, but a simulation with `spawn_headroom: 0` (the default,
    /// used by `default_config`) still legitimately has no free slot — so this test still
    /// frees one via `Reset` first, same as before, now correctly understood as exercising a
    /// real capability rather than working around a bug. See
    /// `add_predator_succeeds_directly_via_spawn_headroom_through_the_c_encoding` below for
    /// the direct fix path through the C API.
    #[test]
    fn add_predator_command_round_trips_through_the_c_encoding_via_reset() {
        unsafe {
            let strings = default_config_strings();
            let mut config = default_config(&strings, 5);
            let hook = c_str("predator");
            let hooks = [hook.as_ptr()];
            config.step_hooks = hooks.as_ptr();
            config.step_hooks_len = 1;
            let sim = murmur_create(&config);
            assert!(
                !sim.is_null(),
                "{:?}",
                CStr::from_ptr(murmur_last_error_message())
            );

            // Free a slot first via Reset, then AddPredator in the same batch.
            let commands = [
                CCommand {
                    kind: CMD_RESET,
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
                    count: 3,
                    stride: 0,
                    seed: 42,
                    has_seed: 1,
                },
                CCommand {
                    kind: CMD_ADD_PREDATOR,
                    position: CVec3 {
                        x: 9.0,
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
                    stride: 0,
                    seed: 0,
                    has_seed: 0,
                },
            ];
            let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
            let status = murmur_run_batch(
                sim,
                1,
                1,
                commands.as_ptr(),
                commands.len(),
                &mut out_buffer,
            );
            assert_eq!(status, 0);
            assert_eq!(murmur_boid_count(sim), 4); // 3 reset prey + 1 spawned predator

            let cp0 = murmur_checkpoint_buffer_get(out_buffer, 0);
            assert_eq!(cp0.predator_count, 1);

            murmur_checkpoint_buffer_destroy(out_buffer);
            murmur_destroy(sim);
        }
    }

    /// The G6 fix itself, through the C API: `MurmurConfig::spawn_headroom` reserved at
    /// `murmur_create` time makes `AddPredator` succeed directly, no `Reset` workaround needed.
    #[test]
    fn add_predator_succeeds_directly_via_spawn_headroom_through_the_c_encoding() {
        unsafe {
            let strings = default_config_strings();
            let mut config = default_config(&strings, 5);
            let hook = c_str("predator");
            let hooks = [hook.as_ptr()];
            config.step_hooks = hooks.as_ptr();
            config.step_hooks_len = 1;
            config.spawn_headroom = 2;
            let sim = murmur_create(&config);
            assert!(
                !sim.is_null(),
                "{:?}",
                CStr::from_ptr(murmur_last_error_message())
            );
            let before = murmur_boid_count(sim);

            let commands = [CCommand {
                kind: CMD_ADD_PREDATOR,
                position: CVec3 {
                    x: 9.0,
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
                stride: 0,
                seed: 0,
                has_seed: 0,
            }];
            let mut out_buffer: *mut MurmurCheckpointBuffer = ptr::null_mut();
            let status = murmur_run_batch(
                sim,
                1,
                1,
                commands.as_ptr(),
                commands.len(),
                &mut out_buffer,
            );
            assert_eq!(status, 0);
            assert_eq!(murmur_boid_count(sim), before + 1);

            let cp0 = murmur_checkpoint_buffer_get(out_buffer, 0);
            assert_eq!(cp0.predator_count, 1);

            murmur_checkpoint_buffer_destroy(out_buffer);
            murmur_destroy(sim);
        }
    }
}
