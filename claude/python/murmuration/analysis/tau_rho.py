"""Density correlation time (Pearce Fig. 2f) — design/03_observables_bindings.md §3.4.

rho(t)   = N / Vol(convex hull of positions at t)
C(dt)    = <rho(t) rho(t+dt)> - <rho>^2                autocovariance
tau_rho  = sum_{lag>=0} C(lag)/C(0) * dt_sample          integrate to first zero-crossing

Buffer ~500 snapshots, sample every ~10 frames; cap the integration window at 25% of the
buffer if no zero-crossing is found.
"""

import numpy as np


def tau_rho(rho_series, dt_sample: float) -> float:
    x = np.asarray(rho_series, dtype=np.float64) - np.mean(rho_series)
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    if ac[0] == 0:
        return 0.0
    ac = ac / ac[0]
    zc = int(np.argmax(ac < 0)) if np.any(ac < 0) else len(ac) // 4
    return float(np.sum(ac[:zc]) * dt_sample)
