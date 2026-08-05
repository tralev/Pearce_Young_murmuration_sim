"""pytest suite for Simulation.run_batch_checked / Command — Track B's atomic command-queue
contract (design/05_viz_contract.md §3, roadmap.md Phase 10), previously only reachable from
Rust/C (murmur_ffi)/reference_desktop, never from Python. Same style as test_bindings.py.
"""

import numpy as np
import pytest

import murmuration as m


def test_add_predator_is_a_noop_without_the_predator_step_hook():
    sim = m.Simulation(boid_count=30, spawn_headroom=5, init_radius=15.0)
    sim.run_batch(1, 0)
    before = sim.metrics()["count"]
    sim.run_batch_checked(1, 0, [m.Command.add_predator(0.0, 0.0, 20.0, 0.0, 0.0, -1.0)])
    assert sim.metrics()["count"] == before  # documented no-op, not a silent bug


def test_add_predator_spawns_a_real_boid_when_predator_hook_and_headroom_present():
    sim = m.Simulation(boid_count=30, spawn_headroom=5, init_radius=15.0, step_hooks=["predator"])
    sim.run_batch(1, 0)
    before = sim.metrics()["count"]
    sim.run_batch_checked(1, 0, [m.Command.add_predator(0.0, 0.0, 20.0, 0.0, 0.0, -1.0)])
    assert sim.metrics()["count"] == before + 1
    assert 1 in sim.species()  # Species::Predator tag


def test_remove_predator_removes_the_spawned_boid():
    sim = m.Simulation(boid_count=30, spawn_headroom=5, init_radius=15.0, step_hooks=["predator"])
    sim.run_batch_checked(1, 0, [m.Command.add_predator(0.0, 0.0, 20.0, 0.0, 0.0, -1.0)])
    n_after_add = sim.metrics()["count"]
    predator_id = int(np.nonzero(sim.species() == 1)[0][0])
    sim.run_batch_checked(1, 0, [m.Command.remove_predator(predator_id)])
    assert sim.metrics()["count"] == n_after_add - 1


def test_set_param_changes_a_live_core_param():
    sim = m.Simulation(boid_count=30, init_radius=15.0, max_force=1.0)
    sim.run_batch_checked(1, 0, [m.Command.set_param("max_force", 0.25)])
    # No direct getter for CoreParams from Python -- verify indirectly: a much smaller
    # max_force should visibly slow how fast velocities can change between two batches.
    v0 = sim.velocities().copy()
    sim.run_batch(1, 0)
    v1 = sim.velocities()
    max_delta = np.max(np.linalg.norm(v1 - v0, axis=1))
    assert max_delta <= 0.25 + 1e-6


def test_invalid_command_rejects_the_whole_queue_and_leaves_state_untouched():
    sim = m.Simulation(boid_count=30, init_radius=15.0, step_hooks=["predator"], spawn_headroom=5)
    sim.run_batch(1, 0)
    before_pos = sim.positions().copy()
    before_step = sim.metrics()["step"]
    with pytest.raises(ValueError):
        sim.run_batch_checked(
            5,
            0,
            [
                m.Command.add_predator(0.0, 0.0, 20.0, 0.0, 0.0, -1.0),  # valid
                m.Command.set_param("phi_p", 0.5),  # invalid -- not live-mutable
            ],
        )
    # Atomic: the valid AddPredator must NOT have been applied, and no steps run.
    assert sim.metrics()["step"] == before_step
    assert np.array_equal(sim.positions(), before_pos)
    assert 1 not in sim.species()


def test_reset_command_reinitializes_the_flock():
    sim = m.Simulation(boid_count=30, init_radius=15.0)
    sim.run_batch(5, 0)
    sim.run_batch_checked(0, 0, [m.Command.reset(20, seed=7)])
    assert sim.metrics()["count"] == 20
    assert sim.metrics()["step"] == 0


def test_run_batch_checked_with_no_commands_behaves_like_run_batch():
    sim_a = m.Simulation(boid_count=30, init_radius=15.0, init_seed=1)
    sim_b = m.Simulation(boid_count=30, init_radius=15.0, init_seed=1)
    sim_a.run_batch(10, 0)
    sim_b.run_batch_checked(10, 0, [])
    assert np.array_equal(sim_a.positions(), sim_b.positions())
