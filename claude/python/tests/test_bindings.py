"""pytest suite for the murmur_py bindings (roadmap.md Phase 6 exit gate):
views (N,3)/(N,) f64/uint16, WRITEABLE=False; mutating a copy doesn't corrupt the sim;
snapshot() survives further step() calls; Rust state_hash == hash of the Python snapshot.
"""

import struct

import numpy as np
import pytest

import murmuration as m


def make_sim(n=40, seed=1):
    return m.Simulation(
        boid_count=n,
        vision_radius=10.0,
        body_radius=0.5,
        init_radius=15.0,
        init_seed=seed,
    )


def test_positions_and_velocities_are_n_by_3_float64():
    sim = make_sim(40)
    pos = sim.positions()
    vel = sim.velocities()
    assert pos.shape == (40, 3)
    assert vel.shape == (40, 3)
    assert pos.dtype == np.float64
    assert vel.dtype == np.float64


def test_species_and_opacity_are_length_n():
    sim = make_sim(40)
    species = sim.species()
    opacity = sim.opacity()
    assert species.shape == (40,)
    assert opacity.shape == (40,)
    assert species.dtype == np.uint16
    assert opacity.dtype == np.float64


@pytest.mark.parametrize("accessor", ["positions", "velocities", "species", "opacity"])
def test_views_are_not_writeable(accessor):
    sim = make_sim(20)
    arr = getattr(sim, accessor)()
    assert arr.flags.writeable is False
    with pytest.raises(ValueError):
        arr[0] = arr[0] * 0 + 1


def test_mutating_a_copy_does_not_corrupt_the_simulation():
    sim = make_sim(20)
    before = sim.positions().copy()
    mutated = sim.positions().copy()
    mutated[0, 0] = 999999.0
    # The copy is independent; re-reading from the sim must be unaffected.
    after = sim.positions()
    assert np.array_equal(before, after)
    assert after[0, 0] != 999999.0


def test_snapshot_survives_further_step_calls():
    sim = make_sim(30, seed=7)
    sim.run_batch(5, 7)
    snap = sim.snapshot()
    frozen_positions = snap.positions.copy()
    frozen_step_count = snap.step_count

    sim.run_batch(20, 7)  # advance the live sim well past the snapshot

    assert sim.step_count() != frozen_step_count
    assert not np.array_equal(sim.positions(), frozen_positions)
    # The snapshot's own arrays are untouched by the sim's continued evolution.
    assert np.array_equal(snap.positions, frozen_positions)
    assert snap.step_count == frozen_step_count


def test_snapshot_checkpoint_fields_are_all_nan_or_empty_with_no_step_hooks_composed():
    # design/05_viz_contract.md §2.1/§2.2's "empty, not absent" rule: a composition with no
    # publishing StepHook/SteeringModifier still has every field, just unpopulated.
    sim = make_sim(10, seed=3)
    sim.run_batch(2, 3)
    snap = sim.snapshot()
    for field in ("boid_state", "speed_mult", "threat_proximity", "panic", "blackening", "spin"):
        arr = getattr(snap, field)
        assert arr.shape == (10,)
        assert np.all(np.isnan(arr)), f"{field} should be all-NaN with no publishing hook"
    assert snap.scene["environment"] is None
    assert snap.scene["wander"] is None
    assert snap.scene["ripple_trains"] is None
    assert snap.scene["dynamic_vision_range"] is None
    assert snap.scene["obstacles"] == []


def test_snapshot_publishes_real_boid_state_machine_and_ecology_checkpoint_fields():
    sim = m.Simulation(
        boid_count=60,
        mode="pearce",
        phi_p=0.50,
        phi_a=0.20,
        step_hooks=["boid_state_machine", "ecology"],
        isolated_max=1,
        crowded_min=5,
        vision_radius=15.0,
        init_radius=8.0,
        init_seed=5,
    )
    sim.run_batch(3, 5)
    snap = sim.snapshot()

    # boid_state_machine classifies every active boid every step -> never NaN.
    assert not np.any(np.isnan(snap.boid_state))
    assert np.all(np.isin(snap.boid_state, [0.0, 1.0, 2.0, 3.0]))
    # threat_proximity/panic/blackening belong to predator_fsm, not boid_state_machine -- still
    # all-NaN here, proving fields are attributed per-hook, not blanket-populated once any hook
    # is present.
    assert np.all(np.isnan(snap.threat_proximity))

    # ecology always publishes a real environment every step.
    env = snap.scene["environment"]
    assert env is not None
    assert isinstance(env["day"], int)
    assert isinstance(env["hour"], float)
    assert isinstance(env["predator_active"], bool)


def test_snapshot_publishes_real_wander_and_obstacles_checkpoint_fields():
    sim = m.Simulation(
        boid_count=40,
        mode="pearce",
        phi_p=0.03,
        phi_a=0.80,
        step_hooks=["wander", "obstacles"],
        wander_pull_strength=0.5,
        obstacle_kind=0.0,
        obstacle_radius=5.0,
        init_radius=20.0,
        init_seed=9,
    )
    sim.run_batch(2, 9)
    snap = sim.snapshot()

    wander = snap.scene["wander"]
    assert wander is not None
    assert len(wander["center"]) == 3
    assert len(wander["heading"]) == 3

    obstacles = snap.scene["obstacles"]
    assert len(obstacles) == 1
    assert obstacles[0]["kind"] == "sphere"
    assert obstacles[0]["radius"] == 5.0
    assert obstacles[0]["csg_op"] == "union"
    assert obstacles[0]["parent"] is None


def _fnv1a(data: bytes) -> int:
    """Matches murmur_core::Simulation::state_hash's FNV-1a exactly (pipeline.rs)."""
    h = 0xCBF29CE484222325
    mask = (1 << 64) - 1
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & mask
    return h


def _python_state_hash(positions: np.ndarray, velocities: np.ndarray) -> int:
    buf = bytearray()
    for pos_row, vel_row in zip(positions, velocities):
        for row in (pos_row, vel_row):
            for value in row:
                buf += struct.pack("<d", float(value))  # IEEE754 LE bytes, matches to_bits().to_le_bytes()
    return _fnv1a(bytes(buf))


def test_rust_state_hash_matches_hash_of_the_python_snapshot():
    sim = make_sim(25, seed=3)
    sim.run_batch(10, 3)
    snap = sim.snapshot()

    computed = _python_state_hash(snap.positions, snap.velocities)
    assert computed == snap.state_hash
    assert snap.state_hash == sim.state_hash()


def test_metrics_dict_has_the_expected_keys():
    sim = make_sim(20)
    metrics = sim.metrics()
    expected = {
        "polarisation", "nematic_order", "opacity_int", "opacity_ext", "r_max",
        "dispersion", "mean_nn", "mean_speed", "angular_momentum", "count", "step",
    }
    assert expected.issubset(metrics.keys())
    assert metrics["count"] == 20


def test_describe_reports_the_default_plugin_selection():
    sim = make_sim(10)
    desc = sim.describe()
    assert desc["plugin_names"]["mode"] == "pearce"
    assert desc["plugin_names"]["modifier"] == "instant_response"
    assert desc["plugin_names"]["domain"] == "open"
    assert desc["boid_count"] == 10


def test_invalid_params_raise_value_error_not_panic():
    with pytest.raises(ValueError):
        m.Simulation(boid_count=10, cruise_speed=-1.0)


def test_invalid_plugin_name_raises_value_error():
    with pytest.raises(ValueError):
        m.Simulation(boid_count=10, mode="not_a_real_mode")


def test_step_advances_state_deterministically_given_the_same_seed():
    sim_a = make_sim(15, seed=99)
    sim_b = make_sim(15, seed=99)
    sim_a.run_batch(10, 123)
    sim_b.run_batch(10, 123)
    assert sim_a.state_hash() == sim_b.state_hash()
    assert np.array_equal(sim_a.positions(), sim_b.positions())
