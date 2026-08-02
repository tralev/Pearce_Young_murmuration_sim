"""Density scaling (Pearce 2014) — design/03_observables_bindings.md §3.3.

Mean-field derivation (todo.md B5-B6): a random ray through a homogeneous isotropic 3D flock
hits sky with probability Psky ~ exp(-rho*b^2*R). Marginal opacity sets Psky = 1/2, so
rho*b^2*R ~ ln2. Since N ~ rho*R^3, eliminating R gives the scaling law:

    rho(N) ~ N^(-1/2)   and   L(N) ~ N^(+1/2)      (d = 3)

Measured per run (open boundary, self-sized flock):
    centre  = per-axis median of positions              (robust to stragglers)
    r_i     = |pos_i - centre|; keep fastest 85% (drop farthest 15%)
    Rg      = sqrt(mean(r_i^2))                           gyration radius
    rho_N   = N_kept / ((4/3) pi Rg^3)                     number density
"""

import numpy as np


def density(pos: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    c = np.median(pos, axis=0)
    r = np.linalg.norm(pos - c, axis=1)
    keep = r <= np.quantile(r, 0.85)
    rg = np.sqrt(np.mean(r[keep] ** 2))
    if rg <= 0:
        return np.inf
    return float(keep.sum() / ((4.0 / 3.0) * np.pi * rg**3))


def scaling_exponent(ns, rhos) -> float:
    """P-c: fitted log-log slope b of rho vs N; ideal b = -0.5."""
    b, _c = np.polyfit(np.log(ns), np.log(rhos), 1)
    return float(b)
