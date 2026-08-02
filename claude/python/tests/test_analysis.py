"""Unit tests for murmuration.analysis against synthetic, known-answer inputs (roadmap.md
Phase 7). These don't depend on the simulation — see test_validate.py for that."""

import numpy as np

from murmuration.analysis import density, h2, shape_pca, tau_rho


def test_shape_pca_isotropic_cloud_has_aspect_near_one():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(2000, 3))  # isotropic Gaussian
    shape = shape_pca.flock_shape(pos)
    assert shape["aspect"] < 1.3, f"expected near-isotropic, got aspect={shape['aspect']}"
    assert shape["thickness"] > 0.7


def test_shape_pca_elongated_cloud_has_high_aspect_low_thickness():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(2000, 3)) * np.array([10.0, 1.0, 1.0])  # stretched along x
    shape = shape_pca.flock_shape(pos)
    assert shape["aspect"] > 3.0
    assert shape["thickness"] < 0.3


def test_shape_pca_eigs_are_descending():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(500, 3)) * np.array([5.0, 2.0, 1.0])
    eigs = shape_pca.flock_shape(pos)["eigs"]
    assert eigs[0] >= eigs[1] >= eigs[2]


def test_m_star_from_shape_is_monotone_decreasing_in_aspect():
    low = shape_pca.m_star_from_shape(1.0)
    mid = shape_pca.m_star_from_shape(2.0)
    high = shape_pca.m_star_from_shape(3.0)
    assert low > mid > high
    assert np.isclose(low, 9.78)
    assert np.isclose(high, 6.05)


def test_h2_curve_on_a_dense_cluster_is_finite_and_connected():
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(200, 3))  # dense single cluster: always connected
    curve = h2.h2_curve(pos, m_values=range(1, 10))
    for m in range(5, 10):
        assert h2.is_connected_at(curve, m), f"expected connected at m={m}"
    m = h2.m_star(curve)
    assert 1 <= m <= 9


def test_h2_curve_on_disjoint_clusters_is_disconnected_at_low_m():
    rng = np.random.default_rng(3)
    # Two well-separated clusters: a small m-NN graph won't bridge them.
    cluster_a = rng.normal(size=(50, 3)) + np.array([0.0, 0.0, 0.0])
    cluster_b = rng.normal(size=(50, 3)) + np.array([1000.0, 0.0, 0.0])
    pos = np.vstack([cluster_a, cluster_b])
    curve = h2.h2_curve(pos, m_values=[1, 2])
    assert not h2.is_connected_at(curve, 1)


def _sample_uniform_sphere(n, radius, rng):
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = radius * rng.uniform(size=n) ** (1 / 3)  # cube-root law -> uniform in volume
    return dirs * r[:, None]


def test_density_is_consistent_across_scale_for_the_same_true_density():
    # density() estimates via a gyration radius over the inner 85% of points, which is
    # systematically smaller than the sphere's actual bounding radius (Rg = sqrt(3/5)*R for a
    # uniform ball) — so it does NOT equal N/((4/3)*pi*R^3) for the *full* radius R. What it
    # should do is agree across two spheres that share the same true point density, at
    # different absolute scale.
    rng = np.random.default_rng(4)
    n1, r1 = 6000, 10.0
    true_density = n1 / r1**3
    r2 = 20.0
    n2 = int(true_density * r2**3)

    rho1 = density.density(_sample_uniform_sphere(n1, r1, rng))
    rho2 = density.density(_sample_uniform_sphere(n2, r2, rng))
    assert abs(rho1 - rho2) / rho1 < 0.15, f"rho1={rho1}, rho2={rho2}"


def test_scaling_exponent_recovers_a_known_power_law():
    ns = np.array([400, 800, 1600, 2600])
    true_b = -0.5
    rhos = 100.0 * ns**true_b  # exact power law, no noise
    b = density.scaling_exponent(ns, rhos)
    assert abs(b - true_b) < 1e-6


def test_tau_rho_is_larger_for_slowly_varying_series_than_white_noise():
    t = np.arange(500)
    slow = np.sin(2 * np.pi * t / 200.0)  # long-period oscillation
    rng = np.random.default_rng(5)
    noise = rng.normal(size=500)

    tau_slow = tau_rho.tau_rho(slow, dt_sample=1.0)
    tau_noise = tau_rho.tau_rho(noise, dt_sample=1.0)
    assert tau_slow > tau_noise
