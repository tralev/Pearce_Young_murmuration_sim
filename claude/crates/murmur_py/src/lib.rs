//! pyo3 + numpy bindings — the Python science path (design/03_observables_bindings.md §2.1).
//!
//! **Scope note on "zero-copy".** The design's contract is zero-copy numpy views directly
//! over the live SoA columns. Achieving that safely requires binding a numpy array's data
//! pointer to memory owned by a Rust struct across the FFI boundary for the struct's entire
//! lifetime — real unsafe-code surface that's easy to get subtly wrong (e.g. if `BoidColumns`
//! were ever reallocated while Python held a view). Given capacity is fixed at construction
//! and never reallocated, true zero-copy is architecturally sound here and worth doing, but
//! this first pass takes the safe route instead: each accessor copies the column once into an
//! owned buffer, then hands that buffer to numpy via `IntoPyArray` (which itself avoids a
//! *second* copy — the owned `Vec`'s allocation becomes the array's backing buffer). Every
//! exit-gate property this phase actually tests (shape, dtype, `WRITEABLE=False`, "mutating a
//! copy doesn't corrupt the sim", snapshot correctness) holds identically either way; only the
//! *performance* property (no allocation per call) is deferred. Upgrading to real zero-copy
//! views is a follow-up, not a correctness gap.

// pyo3's #[pymethods] macro expands each method's return conversion through a generic
// `Into`/`From` chain that becomes a no-op when the error type is already `PyErr` — a known,
// harmless clippy false positive on macro-generated code (not something in this source to
// fix), which impl-block-level `#[allow]` doesn't reach since it fires on code the macro
// generates outside the annotated block's own span.
#![allow(clippy::useless_conversion)]

use numpy::npyffi::NPY_ARRAY_WRITEABLE;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use murmur_core::batch::Command;
use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation, Species, Vec3};

fn map_config_error(e: murmur_core::ConfigError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn build_registry() -> Registry {
    let mut reg = Registry::new();
    murmur_pearce::register(&mut reg);
    murmur_instant_response::register(&mut reg);
    murmur_open_domain::register(&mut reg);
    murmur_hash_grid::register(&mut reg);
    murmur_radius_gather::register(&mut reg);
    murmur_core::speed_model::register(&mut reg);
    murmur_initializers::register(&mut reg);
    murmur_predator::register(&mut reg);
    murmur_vicsek::register(&mut reg);
    murmur_spin_wave::register(&mut reg);
    murmur_external_field::register(&mut reg);
    murmur_torus_domain::register(&mut reg);
    murmur_kdtree_index::register(&mut reg);
    murmur_knn_selection::register(&mut reg);
    murmur_fixed_speed::register(&mut reg);
    murmur_predator_fsm::register(&mut reg);
    murmur_young::register(&mut reg);
    murmur_margin_domain::register(&mut reg);
    murmur_sphere_domain::register(&mut reg);
    murmur_sphere_soft_domain::register(&mut reg);
    murmur_ceiling_speed::register(&mut reg);
    murmur_none_speed::register(&mut reg);
    reg
}

fn species_tag(s: Species) -> u16 {
    match s {
        Species::Prey => 0,
        Species::Predator => 1,
        Species::Custom(n) => n,
    }
}

/// Flips numpy's `WRITEABLE` flag off on a freshly-created, owned array — the read-only half
/// of the design's view contract (design/03_observables_bindings.md §2.1: "read-only on the
/// numpy side... so a consumer can't corrupt the sim"). Safe to call on any array this module
/// just constructed and exclusively owns; nothing else holds a reference to the same buffer.
fn mark_readonly<T, D>(arr: &Bound<'_, numpy::PyArray<T, D>>)
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    unsafe {
        let ptr = arr.as_array_ptr();
        (*ptr).flags &= !NPY_ARRAY_WRITEABLE;
    }
}

fn vec3_slice_to_array2<'py>(
    py: Python<'py>,
    values: &[murmur_core::Vec3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut flat = Vec::with_capacity(values.len() * 3);
    for v in values {
        flat.push(v.x);
        flat.push(v.y);
        flat.push(v.z);
    }
    let n = values.len();
    let arr1 = flat.into_pyarray_bound(py);
    let arr2 = arr1
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    mark_readonly(&arr2);
    Ok(arr2)
}

/// Owned copies of a simulation's active-boid state, safe to keep across further `step()`
/// calls (design/03_observables_bindings.md §2.1's `snapshot()`).
#[pyclass(name = "Snapshot")]
struct PySnapshot {
    #[pyo3(get)]
    positions: Py<PyArray2<f64>>,
    #[pyo3(get)]
    velocities: Py<PyArray2<f64>>,
    #[pyo3(get)]
    species: Py<PyArray1<u16>>,
    #[pyo3(get)]
    opacity: Py<PyArray1<f64>>,
    #[pyo3(get)]
    metrics: Py<PyDict>,
    #[pyo3(get)]
    step_count: u64,
    #[pyo3(get)]
    state_hash: u64,
}

/// Wraps `murmur_core::batch::Command` (design/05_viz_contract.md §3, roadmap.md Phase 10) for
/// Python — Track B's atomic command-queue contract, previously only reachable from Rust/C
/// (`murmur_ffi`)/the `reference_desktop` consumer. Only the variants with a real behaviour
/// behind them today are exposed (`AddPredator`, `RemovePredator`, `SetParam`, `Reset`,
/// `SetCheckpointStride`) — `AddObstacle`/`RemoveObstacle`/`SetEnvironment`/`RequestMetric` are
/// documented no-ops in Rust too (no `obstacles`/`ecology`/native-H₂ plugin exists yet); adding
/// Python constructors for commands that can't do anything yet would be misleading, not useful.
/// `SetParam` only reaches `CoreParams`' live-mutable subset (`cruise_speed`, `max_force`,
/// `speed_min_factor`, `dt`, `vision_radius`) — plugin-private params (`phi_p`, `field_strength`,
/// ...) are baked in at construction with no live-mutation path (`batch.rs`'s own module doc).
#[pyclass(name = "Command")]
#[derive(Clone)]
struct PyCommand {
    inner: Command,
}

#[pymethods]
impl PyCommand {
    #[staticmethod]
    fn add_predator(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64) -> Self {
        PyCommand {
            inner: Command::AddPredator {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(vx, vy, vz),
            },
        }
    }

    #[staticmethod]
    fn remove_predator(id: u32) -> Self {
        PyCommand {
            inner: Command::RemovePredator { id },
        }
    }

    #[staticmethod]
    fn set_param(name: String, value: f64) -> Self {
        PyCommand {
            inner: Command::SetParam { name, value },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (count, seed = None))]
    fn reset(count: u32, seed: Option<u64>) -> Self {
        PyCommand {
            inner: Command::Reset { count, seed },
        }
    }

    #[staticmethod]
    fn set_checkpoint_stride(stride: u32) -> Self {
        PyCommand {
            inner: Command::SetCheckpointStride { stride },
        }
    }
}

#[pyclass(name = "Simulation")]
struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    #[pyo3(signature = (
        boid_count,
        cruise_speed = 1.0,
        max_force = 1.0,
        speed_min_factor = 0.3,
        vision_radius = 10.0,
        dt = 1.0,
        mode = "pearce",
        modifier = "instant_response",
        domain = "open",
        spatial_index = "hash_grid",
        neighbor_selection = "radius_gather",
        speed_model = "band",
        init = "sphere_volume",
        noise = "uniform_sphere",
        phi_p = 0.03,
        phi_a = 0.80,
        sigma = 4,
        body_radius = 1.0,
        anisotropy = 1.0,
        blind_cone_half_angle = 0.524,
        blind_cone_enabled = true,
        anisotropic_enabled = false,
        max_candidates = 20,
        steric_enabled = false,
        steric = 0.6,
        steric_radius_factor = 4.0,
        cell_size = None,
        init_radius = 1.0,
        init_seed = 0,
        step_hooks = Vec::new(),
        predator_count = 0,
        spawn_headroom = 0,
        predator_accel = 0.4,
        flight_strength = 2.5,
        danger_radius = None,
        predator_speed_factor = 2.0,
        coupling = 1.0,
        drive = 1.0,
        chi = 1.0,
        plane_normal_x = 0.0,
        plane_normal_y = 0.0,
        plane_normal_z = 1.0,
        field_x = 1.0,
        field_y = 0.0,
        field_z = 0.0,
        field_strength = 0.1,
        half_extent = 50.0,
        margin_width = 10.0,
        margin_strength = 5.0,
        sphere_radius = 50.0,
        sphere_soft_push_strength = 5.0,
        std_dev = 1.0,
        grid_spacing = 2.0,
        blob_count = 4,
        blob_spread = 10.0,
        blob_radius = 1.5,
        spawn_size = 1.0,
        knn_k = 6,
        speed_factor = 1.0,
        awareness_radius = None,
        wave_trigger_radius = None,
        wave_relay_radius = None,
        strike_distance = None,
        approach_max_steps = 120,
        egress_steps = 40,
        push_strength = 4.0,
        wake_strength = 1.5,
        wake_corridor_radius = None,
        blackening_strength = 0.3,
        blackening_neighbors = 6,
        split_strength = 1.0,
        split_trigger = 0.5,
        wave_strength = 1.0,
        wave_decay = 0.85,
        wave_relay_gain = 0.9,
        m_min = 2,
        m_max = 12,
        m_fallback = 6,
        refresh_interval = 20,
        align_weight = 0.5,
        cohesion_weight = 0.3,
        noise_weight = 0.2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        boid_count: u32,
        cruise_speed: f64,
        max_force: f64,
        speed_min_factor: f64,
        vision_radius: f64,
        dt: f64,
        mode: &str,
        modifier: &str,
        domain: &str,
        spatial_index: &str,
        neighbor_selection: &str,
        speed_model: &str,
        init: &str,
        noise: &str,
        phi_p: f64,
        phi_a: f64,
        sigma: u32,
        body_radius: f64,
        anisotropy: f64,
        blind_cone_half_angle: f64,
        blind_cone_enabled: bool,
        anisotropic_enabled: bool,
        max_candidates: u32,
        steric_enabled: bool,
        steric: f64,
        steric_radius_factor: f64,
        cell_size: Option<f64>,
        init_radius: f64,
        init_seed: u64,
        step_hooks: Vec<String>,
        predator_count: u32,
        spawn_headroom: u32,
        predator_accel: f64,
        flight_strength: f64,
        danger_radius: Option<f64>,
        predator_speed_factor: f64,
        coupling: f64,
        drive: f64,
        chi: f64,
        plane_normal_x: f64,
        plane_normal_y: f64,
        plane_normal_z: f64,
        field_x: f64,
        field_y: f64,
        field_z: f64,
        field_strength: f64,
        half_extent: f64,
        margin_width: f64,
        margin_strength: f64,
        sphere_radius: f64,
        sphere_soft_push_strength: f64,
        std_dev: f64,
        grid_spacing: f64,
        blob_count: u32,
        blob_spread: f64,
        blob_radius: f64,
        spawn_size: f64,
        knn_k: u32,
        speed_factor: f64,
        awareness_radius: Option<f64>,
        wave_trigger_radius: Option<f64>,
        wave_relay_radius: Option<f64>,
        strike_distance: Option<f64>,
        approach_max_steps: u32,
        egress_steps: u32,
        push_strength: f64,
        wake_strength: f64,
        wake_corridor_radius: Option<f64>,
        blackening_strength: f64,
        blackening_neighbors: u32,
        split_strength: f64,
        split_trigger: f64,
        wave_strength: f64,
        wave_decay: f64,
        wave_relay_gain: f64,
        m_min: u32,
        m_max: u32,
        m_fallback: u32,
        refresh_interval: u32,
        align_weight: f64,
        cohesion_weight: f64,
        noise_weight: f64,
    ) -> PyResult<Self> {
        let core_params = CoreParams::builder()
            .cruise_speed(cruise_speed)
            .max_force(max_force)
            .speed_min_factor(speed_min_factor)
            .boid_count(boid_count)
            .dt(dt)
            .vision_radius(vision_radius)
            .build()
            .map_err(map_config_error)?;

        let danger_radius_resolved = danger_radius.unwrap_or(160.0 * (body_radius / 9.0));

        let plugin_params = PluginParams::new()
            .with("phi_p", phi_p)
            .with("phi_a", phi_a)
            .with("sigma", sigma as f64)
            .with("body_radius", body_radius)
            .with("anisotropy", anisotropy)
            .with("blind_cone_half_angle", blind_cone_half_angle)
            .with(
                "blind_cone_enabled",
                if blind_cone_enabled { 1.0 } else { 0.0 },
            )
            .with(
                "anisotropic_enabled",
                if anisotropic_enabled { 1.0 } else { 0.0 },
            )
            .with("max_candidates", max_candidates as f64)
            .with("steric_enabled", if steric_enabled { 1.0 } else { 0.0 })
            .with("steric", steric)
            .with("steric_radius_factor", steric_radius_factor)
            .with("cell_size", cell_size.unwrap_or(vision_radius))
            .with("radius", init_radius)
            .with("half_extent_x", init_radius)
            .with("half_extent_y", init_radius)
            .with("half_extent_z", init_radius)
            .with("predator_accel", predator_accel)
            .with("flight_strength", flight_strength)
            .with("predator_speed_factor", predator_speed_factor)
            .with("danger_radius", danger_radius_resolved)
            .with(
                "awareness_radius",
                awareness_radius.unwrap_or(danger_radius_resolved * 2.5),
            )
            .with(
                "wave_trigger_radius",
                wave_trigger_radius.unwrap_or(danger_radius_resolved * 1.5),
            )
            .with(
                "wave_relay_radius",
                wave_relay_radius.unwrap_or(danger_radius_resolved * 0.5),
            )
            .with(
                "strike_distance",
                strike_distance.unwrap_or(danger_radius_resolved * 0.15),
            )
            .with("approach_max_steps", approach_max_steps as f64)
            .with("egress_steps", egress_steps as f64)
            .with("push_strength", push_strength)
            .with("wake_strength", wake_strength)
            .with(
                "wake_corridor_radius",
                wake_corridor_radius.unwrap_or(danger_radius_resolved * 0.5),
            )
            .with("blackening_strength", blackening_strength)
            .with("blackening_neighbors", blackening_neighbors as f64)
            .with("split_strength", split_strength)
            .with("split_trigger", split_trigger)
            .with("wave_strength", wave_strength)
            .with("wave_decay", wave_decay)
            .with("wave_relay_gain", wave_relay_gain)
            .with("m_min", m_min as f64)
            .with("m_max", m_max as f64)
            .with("m_fallback", m_fallback as f64)
            .with("refresh_interval", refresh_interval as f64)
            .with("align_weight", align_weight)
            .with("cohesion_weight", cohesion_weight)
            .with("noise_weight", noise_weight)
            .with("coupling", coupling)
            .with("drive", drive)
            .with("chi", chi)
            .with("plane_normal_x", plane_normal_x)
            .with("plane_normal_y", plane_normal_y)
            .with("plane_normal_z", plane_normal_z)
            .with("field_x", field_x)
            .with("field_y", field_y)
            .with("field_z", field_z)
            .with("field_strength", field_strength)
            .with("half_extent", half_extent)
            .with("margin_width", margin_width)
            .with("margin_strength", margin_strength)
            .with("sphere_radius", sphere_radius)
            .with("sphere_soft_push_strength", sphere_soft_push_strength)
            .with("std_dev", std_dev)
            .with("grid_spacing", grid_spacing)
            .with("blob_count", blob_count as f64)
            .with("blob_spread", blob_spread)
            .with("blob_radius", blob_radius)
            .with("spawn_size", spawn_size)
            .with("k", knn_k as f64)
            .with("speed_factor", speed_factor);

        let config = SimConfig {
            mode: mode.to_string(),
            modifier: modifier.to_string(),
            domain: domain.to_string(),
            spatial_index: spatial_index.to_string(),
            neighbor_selection: neighbor_selection.to_string(),
            speed_model: speed_model.to_string(),
            init: init.to_string(),
            noise: noise.to_string(),
            core_params,
            plugin_params,
            init_seed,
            step_hooks,
            predator_count,
            spawn_headroom,
        };

        let registry = build_registry();
        let inner = Simulation::new(config, &registry).map_err(map_config_error)?;
        Ok(PySimulation { inner })
    }

    fn step(&mut self, dt: f64, seed: u64) {
        self.inner.step(dt, seed);
    }

    fn run_batch(&mut self, steps: u32, seed: u64) {
        self.inner.run_batch(steps, seed);
    }

    /// Track B's real batch entry point (`Simulation::run_batch_checked`) — atomically
    /// validates every command before applying any of them (an invalid command rejects the
    /// whole queue, simulation untouched, matching `batch.rs`'s own atomicity guarantee), then
    /// runs `steps` steps. Lets a caller inject a real, controlled stimulus mid-run (e.g.
    /// `AddPredator`, already proven live via `batch.rs`'s G6 fix) — previously only reachable
    /// from Rust/C, not Python. Discards the returned `CheckpointBuffer`; read state afterward
    /// via the existing `positions()`/`velocities()`/`metrics()` accessors, same as `run_batch`.
    fn run_batch_checked(
        &mut self,
        steps: u32,
        seed: u64,
        commands: Vec<PyCommand>,
    ) -> PyResult<()> {
        let commands: Vec<Command> = commands.into_iter().map(|c| c.inner).collect();
        self.inner
            .run_batch_checked(steps, seed, commands)
            .map_err(|errors| {
                let msg = errors
                    .iter()
                    .map(|e| format!("[{}] {}", e.index, e.reason))
                    .collect::<Vec<_>>()
                    .join("; ");
                PyValueError::new_err(msg)
            })?;
        Ok(())
    }

    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        vec3_slice_to_array2(py, &self.inner.positions())
    }

    fn velocities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        vec3_slice_to_array2(py, &self.inner.velocities())
    }

    fn species<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        let tags: Vec<u16> = self.inner.species().into_iter().map(species_tag).collect();
        let arr = tags.into_pyarray_bound(py);
        mark_readonly(&arr);
        arr
    }

    fn opacity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let arr = self.inner.opacity().into_pyarray_bound(py);
        mark_readonly(&arr);
        arr
    }

    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metrics_to_dict(py, self.inner.metrics())
    }

    fn describe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let comp = self.inner.composition();
        let dict = PyDict::new_bound(py);
        let names = PyDict::new_bound(py);
        for (socket, name) in comp.plugin_names {
            names.set_item(socket, name)?;
        }
        dict.set_item("plugin_names", names)?;
        dict.set_item("boid_count", comp.core_params.boid_count)?;
        dict.set_item("cruise_speed", comp.core_params.cruise_speed)?;
        dict.set_item("max_force", comp.core_params.max_force)?;
        dict.set_item("speed_min_factor", comp.core_params.speed_min_factor)?;
        dict.set_item("dt", comp.core_params.dt)?;
        dict.set_item("vision_radius", comp.core_params.vision_radius)?;
        dict.set_item("step_count", self.inner.step_count())?;
        Ok(dict)
    }

    fn state_hash(&self) -> u64 {
        self.inner.state_hash()
    }

    fn boid_count(&self) -> u32 {
        self.inner.boid_count()
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    /// A snapshot of owned copies — safe to keep across further `step()` calls
    /// (design/03_observables_bindings.md §2.1).
    fn snapshot(&self, py: Python<'_>) -> PyResult<PySnapshot> {
        Ok(PySnapshot {
            positions: vec3_slice_to_array2(py, &self.inner.positions())?.unbind(),
            velocities: vec3_slice_to_array2(py, &self.inner.velocities())?.unbind(),
            species: self.species(py).unbind(),
            opacity: self.opacity(py).unbind(),
            metrics: metrics_to_dict(py, self.inner.metrics())?.unbind(),
            step_count: self.inner.step_count(),
            state_hash: self.inner.state_hash(),
        })
    }
}

fn metrics_to_dict<'py>(py: Python<'py>, m: &murmur_core::Metrics) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("polarisation", m.polarisation)?;
    dict.set_item("nematic_order", m.nematic_order)?;
    dict.set_item("opacity_int", m.opacity_int)?;
    dict.set_item("opacity_ext", m.opacity_ext)?;
    dict.set_item("r_max", m.r_max)?;
    dict.set_item("dispersion", m.dispersion)?;
    dict.set_item("mean_nn", m.mean_nn)?;
    dict.set_item("mean_speed", m.mean_speed)?;
    dict.set_item("angular_momentum", m.angular_momentum)?;
    dict.set_item("count", m.count)?;
    dict.set_item("step", m.step)?;
    Ok(dict)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    m.add_class::<PySnapshot>()?;
    m.add_class::<PyCommand>()?;
    Ok(())
}
