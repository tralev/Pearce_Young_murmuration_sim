"""Type stubs for the compiled `_core` extension (design/03_observables_bindings.md §2.1)."""

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

class Snapshot:
    """Owned copies of a simulation's active-boid state — safe to keep across further
    `Simulation.step()` calls."""

    positions: npt.NDArray[np.float64]  # (N, 3), read-only
    velocities: npt.NDArray[np.float64]  # (N, 3), read-only
    species: npt.NDArray[np.uint16]  # (N,), read-only
    opacity: npt.NDArray[np.float64]  # (N,), read-only
    metrics: Dict[str, float]
    step_count: int
    state_hash: int

class Command:
    """Track B's atomic command-queue contract (design/05_viz_contract.md §3), for use with
    `Simulation.run_batch_checked`. Only the variants with a real behaviour behind them today
    are exposed — see `murmur_py`'s own module doc."""

    @staticmethod
    def add_predator(x: float, y: float, z: float, vx: float, vy: float, vz: float) -> "Command": ...
    @staticmethod
    def remove_predator(id: int) -> "Command": ...
    @staticmethod
    def set_param(name: str, value: float) -> "Command": ...
    @staticmethod
    def reset(count: int, seed: Optional[int] = None) -> "Command": ...
    @staticmethod
    def set_checkpoint_stride(stride: int) -> "Command": ...

class Simulation:
    """A constructed, running simulation. Plugin composition is fixed at construction and
    never changes for its lifetime (design/00_overview.md "Interaction model")."""

    def __init__(
        self,
        boid_count: int,
        cruise_speed: float = 1.0,
        max_force: float = 1.0,
        speed_min_factor: float = 0.3,
        vision_radius: float = 10.0,
        dt: float = 1.0,
        mode: str = "pearce",
        modifier: str = "instant_response",
        domain: str = "open",
        spatial_index: str = "hash_grid",
        neighbor_selection: str = "radius_gather",
        speed_model: str = "band",
        init: str = "sphere_volume",
        noise: str = "uniform_sphere",
        phi_p: float = 0.03,
        phi_a: float = 0.80,
        sigma: int = 4,
        body_radius: float = 1.0,
        anisotropy: float = 1.0,
        blind_cone_half_angle: float = 0.524,
        blind_cone_enabled: bool = True,
        anisotropic_enabled: bool = False,
        max_candidates: int = 20,
        steric_enabled: bool = False,
        steric: float = 0.6,
        steric_radius_factor: float = 4.0,
        cell_size: Optional[float] = None,
        init_radius: float = 1.0,
        init_seed: int = 0,
        step_hooks: List[str] = [],
        predator_count: int = 0,
        predator_accel: float = 0.4,
        flight_strength: float = 2.5,
        danger_radius: Optional[float] = None,
        predator_speed_factor: float = 2.0,
        coupling: float = 1.0,
        drive: float = 1.0,
        chi: float = 1.0,
        plane_normal_x: float = 0.0,
        plane_normal_y: float = 0.0,
        plane_normal_z: float = 1.0,
        field_x: float = 1.0,
        field_y: float = 0.0,
        field_z: float = 0.0,
        field_strength: float = 0.1,
        half_extent: float = 50.0,
        margin_width: float = 10.0,
        margin_strength: float = 5.0,
        sphere_radius: float = 50.0,
        sphere_soft_push_strength: float = 5.0,
        std_dev: float = 1.0,
        grid_spacing: float = 2.0,
        blob_count: int = 4,
        blob_spread: float = 10.0,
        blob_radius: float = 1.5,
        spawn_size: float = 1.0,
        adaptive_crossover: int = 5000,
        fov_enabled: bool = False,
        fov_half_angle: float = 1.2,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        separation_radius: float = 2.0,
        max_turn_rate: float = 1.0,
        amplitude_x: float = 10.0,
        amplitude_y: float = 10.0,
        amplitude_z: float = 10.0,
        frequency_x: float = 0.05,
        frequency_y: float = 0.07,
        frequency_z: float = 0.03,
        phase_x: float = 0.0,
        phase_y: float = 1.5707963267948966,
        phase_z: float = 0.0,
        repulsion_weight: float = 1.5,
        attraction_weight: float = 0.5,
        boundary_weight: float = 1.0,
        desire_weight: float = 0.0,
        repulsion_radius: float = 2.0,
        attraction_radius: float = 10.0,
        boundary_decay: float = 5.0,
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        goal_z: float = 0.0,
        envelope_amplitude: float = 0.3,
        envelope_frequency: float = 0.01,
        anchor_count: int = 3,
        anchor_spread: float = 20.0,
        anchor_weight: float = 1.0,
        isolated_max: int = 1,
        crowded_min: int = 12,
        threat_speed_factor: float = 1.5,
        crowded_speed_cap: float = 0.7,
        knn_k: int = 6,
        speed_factor: float = 1.0,
        awareness_radius: Optional[float] = None,
        wave_trigger_radius: Optional[float] = None,
        wave_relay_radius: Optional[float] = None,
        strike_distance: Optional[float] = None,
        approach_max_steps: int = 120,
        egress_steps: int = 40,
        push_strength: float = 4.0,
        wake_strength: float = 1.5,
        wake_corridor_radius: Optional[float] = None,
        blackening_strength: float = 0.3,
        blackening_neighbors: int = 6,
        split_strength: float = 1.0,
        split_trigger: float = 0.5,
        wave_strength: float = 1.0,
        wave_decay: float = 0.85,
        wave_relay_gain: float = 0.9,
        m_min: int = 2,
        m_max: int = 12,
        m_fallback: int = 6,
        refresh_interval: int = 20,
        align_weight: float = 0.5,
        cohesion_weight: float = 0.3,
        noise_weight: float = 0.2,
        hours_per_dt: float = 0.5,
        dusk_hour: float = 18.0,
        dusk_width: float = 1.0,
        roosting_threshold: float = 0.5,
        year_length_days: int = 365,
        season_start_day: int = 274,
        season_end_day: int = 90,
        predator_rate: float = 0.296,
        temperature_mean: float = 10.0,
        temperature_amplitude: float = 8.0,
        temperature_phase_day: float = 15.0,
        coherence_strength: float = 0.3,
    ) -> None: ...
    def step(self, dt: float, seed: int) -> None: ...
    def run_batch(self, steps: int, seed: int) -> None: ...
    def run_batch_checked(self, steps: int, seed: int, commands: List[Command]) -> None: ...
    def positions(self) -> npt.NDArray[np.float64]: ...  # (N, 3), read-only
    def velocities(self) -> npt.NDArray[np.float64]: ...  # (N, 3), read-only
    def species(self) -> npt.NDArray[np.uint16]: ...  # (N,), read-only
    def opacity(self) -> npt.NDArray[np.float64]: ...  # (N,), read-only
    def metrics(self) -> Dict[str, float]: ...
    def describe(self) -> Dict[str, object]: ...
    def state_hash(self) -> int: ...
    def boid_count(self) -> int: ...
    def step_count(self) -> int: ...
    def snapshot(self) -> Snapshot: ...
