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

use murmur_core::{CoreParams, PluginParams, Registry, SimConfig, Simulation, Species};

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
            .with(
                "danger_radius",
                danger_radius.unwrap_or(160.0 * (body_radius / 9.0)),
            )
            .with("coupling", coupling)
            .with("drive", drive)
            .with("chi", chi)
            .with("plane_normal_x", plane_normal_x)
            .with("plane_normal_y", plane_normal_y)
            .with("plane_normal_z", plane_normal_z)
            .with("field_x", field_x)
            .with("field_y", field_y)
            .with("field_z", field_z)
            .with("field_strength", field_strength);

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
    Ok(())
}
