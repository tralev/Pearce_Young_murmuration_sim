"""The 10 results (design/03_observables_bindings.md §4) as callables -> numbers, driven
against a real `murmuration.Simulation`.

Warmup/sampling windows here are moderate, chosen for a runnable acceptance pass in this
environment — not the full Thompson-2024-style 20k-warmup/80k-sample protocol the design docs
cite as an independent corroborating precedent. Tightening the equilibration protocol (and the
`vision_radius`/`body_radius`/density combination) against a real calibration pass is future
tuning work (roadmap.md's pre-coding numpy-prototype step, not redone here).
"""

import numpy as np

from murmuration import Simulation
from murmuration.analysis import density, h2, shape_pca, tau_rho

PHENOTYPES = {
    "murmuration": dict(phi_p=0.03, phi_a=0.80),
    "schooling": dict(phi_p=0.10, phi_a=0.60),
    "swarming": dict(phi_p=0.20, phi_a=0.30),
}

DEFAULT_SIM_KWARGS = dict(
    vision_radius=10.0,
    body_radius=0.5,
    cell_size=10.0,
    max_candidates=20,
    cruise_speed=1.0,
    max_force=1.0,
)


def init_radius_for(n: int, target_density: float = 0.02) -> float:
    """A density-matched initial placement radius, so equilibration doesn't have to travel
    far — the eventual density is still whatever the occlusion feedback loop self-regulates
    to, this just picks a reasonable starting point."""
    return float((n / ((4.0 / 3.0) * np.pi * target_density)) ** (1.0 / 3.0))


def make_sim(n: int, phi_p: float, phi_a: float, seed: int = 0, **overrides) -> Simulation:
    kwargs = dict(DEFAULT_SIM_KWARGS)
    kwargs.update(overrides)
    return Simulation(
        boid_count=n,
        phi_p=phi_p,
        phi_a=phi_a,
        init_radius=init_radius_for(n),
        init_seed=seed,
        **kwargs,
    )


def sample_metric(sim: Simulation, key: str, n_samples: int, stride: int, seed: int) -> np.ndarray:
    out = np.empty(n_samples)
    for i in range(n_samples):
        sim.run_batch(stride, seed)
        out[i] = sim.metrics()[key]
    return out


# --- Pearce (2014) --------------------------------------------------------------------


def p_a_internal_opacity(ns=(400, 800, 1600), warmup=800, samples=20, stride=10, seed=0) -> dict:
    """Star: mean internal opacity Theta-bar clusters near mu ~ 0.30 at the murmuration
    phenotype. Returns {N: mean_theta}."""
    out = {}
    for n in ns:
        sim = make_sim(n, **PHENOTYPES["murmuration"], seed=seed)
        sim.run_batch(warmup, seed)
        thetas = sample_metric(sim, "opacity_int", samples, stride, seed)
        out[n] = float(np.mean(thetas))
    return out


def p_b_theta_vs_inverse_n(theta_by_n: dict) -> dict:
    """Report: Theta vs 1/N should be linear (R^2 ~ 0.99). Reuses P-a's data."""
    ns = np.array(sorted(theta_by_n))
    thetas = np.array([theta_by_n[n] for n in ns])
    inv_n = 1.0 / ns
    slope, intercept = np.polyfit(inv_n, thetas, 1)
    pred = slope * inv_n + intercept
    ss_res = np.sum((thetas - pred) ** 2)
    ss_tot = np.sum((thetas - thetas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def p_c_density_scaling(ns=(400, 800, 1600, 2600), warmup=800, seed=0) -> dict:
    """Star: density scaling rho(N) ~ N^(-1/2) (open boundary). Returns
    {"exponent": b, "rho_by_n": {...}}."""
    rho_by_n = {}
    for n in ns:
        sim = make_sim(n, **PHENOTYPES["murmuration"], seed=seed)
        sim.run_batch(warmup, seed)
        rho_by_n[n] = density.density(sim.positions())
    ns_arr = np.array(sorted(rho_by_n))
    rhos = np.array([rho_by_n[n] for n in ns_arr])
    b = density.scaling_exponent(ns_arr, rhos)
    return {"exponent": b, "rho_by_n": rho_by_n}


def p_d_no_fragmentation(phi_ps=(0.03, 0.1, 0.2), n=400, steps=10_000, check_every=500, seed=0) -> dict:
    """Star: no fragmentation for any phi_p > 0 — R_max stays bounded (no monotonic
    divergence) over a long run. Returns {phi_p: [r_max samples over time]}."""
    out = {}
    for phi_p in phi_ps:
        # phi_a=0.80 stays valid (phi_p + phi_a <= 1) across all three phi_p values tested.
        sim = make_sim(n, phi_p=phi_p, phi_a=0.80, seed=seed)
        trace = []
        remaining = steps
        while remaining > 0:
            step = min(check_every, remaining)
            sim.run_batch(step, seed)
            remaining -= step
            trace.append(sim.metrics()["r_max"])
        out[phi_p] = trace
    return out


def p_e_tau_rho_vs_phi_p(phi_ps=(0.03, 0.10, 0.20), n=400, warmup=500, samples=200, stride=2, seed=0) -> dict:
    """Report: density correlation time tau_rho decreases as phi_p increases."""
    out = {}
    for phi_p in phi_ps:
        # Hold the noise fraction fixed at murmuration's own 0.17 across all phi_p values,
        # so phi_p is the only thing varying between runs (phi_p=0.03 recovers exactly the
        # murmuration preset's phi_a=0.80).
        phi_a = max(0.0, 1.0 - phi_p - 0.17)
        sim = make_sim(n, phi_p=phi_p, phi_a=phi_a, seed=seed)
        sim.run_batch(warmup, seed)
        rho_series = []
        for _ in range(samples):
            sim.run_batch(stride, seed)
            rho_series.append(density.density(sim.positions()))
        out[phi_p] = tau_rho.tau_rho(rho_series, dt_sample=float(stride))
    return out


def p_f_phenotypes_distinct(n=400, warmup=800, seed=0) -> dict:
    """Report: the three phenotypes give distinct (alpha, Theta-bar) regimes."""
    out = {}
    for name, weights in PHENOTYPES.items():
        sim = make_sim(n, **weights, seed=seed)
        sim.run_batch(warmup, seed)
        m = sim.metrics()
        out[name] = {"alpha": m["polarisation"], "theta": m["opacity_int"]}
    return out


# --- Young (2013) ----------------------------------------------------------------------


def y_a_m_star(n=800, warmup=800, seed=0) -> dict:
    """Star: robustness-per-neighbour peaks at m* ~ 6-7."""
    sim = make_sim(n, **PHENOTYPES["murmuration"], seed=seed)
    sim.run_batch(warmup, seed)
    curve = h2.h2_curve(sim.positions())
    return {"m_star": h2.m_star(curve), "curve": curve}


def y_b_m_star_vs_n(ns=(400, 800, 1600), warmup=800, seed=0) -> dict:
    """Report: m* is independent of flock size N."""
    out = {}
    for n in ns:
        sim = make_sim(n, **PHENOTYPES["murmuration"], seed=seed)
        sim.run_batch(warmup, seed)
        curve = h2.h2_curve(sim.positions())
        out[n] = h2.m_star(curve)
    return out


def y_c_m_star_vs_thickness(n=800, warmup=800, seed=0) -> dict:
    """Star: m* vs flock thickness (PCA lambda_3/lambda_1) trend, sign should be negative
    (thinner/more elongated flocks need fewer neighbours for the same robustness). Spans
    thickness by varying phenotype (murmuration/schooling/swarming produce different order
    and, empirically, different shapes)."""
    out = {}
    for name, weights in PHENOTYPES.items():
        sim = make_sim(n, **weights, seed=seed)
        sim.run_batch(warmup, seed)
        pos = sim.positions()
        shape = shape_pca.flock_shape(pos)
        curve = h2.h2_curve(pos)
        out[name] = {"thickness": shape["thickness"], "m_star": h2.m_star(curve)}
    return out


def y_d_connectivity(n=800, warmup=800, ms=(5, 6, 7, 8), seed=0) -> dict:
    """Report: the sensing graph is connected for all m >= 5."""
    sim = make_sim(n, **PHENOTYPES["murmuration"], seed=seed)
    sim.run_batch(warmup, seed)
    curve = h2.h2_curve(sim.positions(), m_values=ms)
    return {m: h2.is_connected_at(curve, m) for m in ms}
