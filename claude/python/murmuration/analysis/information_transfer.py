"""Information transfer and behavioural inertia (Attanasi et al. 2014, `sci/emss-59192.pdf`)
— design/03_observables_bindings.md §3. Doubles as `murmur_spin_wave`'s acceptance test
(design/02_plugins.md §5, roadmap.md Phase 15): does a simulated flock reproduce the paper's
headline finding (turning information propagates *linearly and undamped*, `x(t) = c_s*t`,
not diffusively, `x(t) ~ sqrt(t)`, the prediction of a memoryless/`InstantResponse` model), and
does `c_s` scale with polarisation the way eq. 10 predicts?

Pipeline, mirroring the paper's own methodology (main text, page 2 and eq. 9-10):
  1. `dominant_motion_plane` — PCA of velocity fluctuations over a trajectory window, giving
     the flock's own (u, v) motion plane (the paper works in the plane the turn actually lies
     in; `murmur_spin_wave` itself uses a *fixed*, configurable plane instead — see that
     plugin's module doc for why — so this is computed independently here in analysis, not
     read back from the plugin's internal state).
  2. `heading_angles` — project velocities onto that plane, per-boid unwrapped angle phi_i(t).
  3. `pairwise_delays` — for every pair (i, j), the mutual turning delay tau_ij (paper's own
     sign convention: tau_ij > 0 means j turns *before* i), from cross-correlating each pair's
     angular-velocity (dphi/dt) signal, i.e. when their *turns* line up, not their raw heading.
  4. `turning_times` — a single self-consistent turning time t_i per boid from the pairwise
     delay matrix (least-squares/row-mean).
  5. `dispersion_curve` — rank-order the t_i and compute x(t) = (rank/density)^(1/3), the
     radius of the sphere containing the first `rank` birds to turn (paper's own x(t)
     definition, assuming a spatially localized turn origin in 3D).
  6. `propagation_speed` — fit the early/intermediate regime of x(t) (paper's own "before
     border effects kick in" caveat) to both a linear (ballistic, spin-wave prediction) and a
     diffusive (sqrt, memoryless-response prediction) model; `c_s` is the linear fit's slope.
  7. `speed_vs_polarisation` — eq. 10, c_s = a/sqrt(beta*chi) * 1/sqrt(1-Phi): given (Phi, c_s)
     pairs from several runs, checks c_s scales linearly with 1/sqrt(1-Phi). Phi is exactly
     this project's own `polarisation` metric (paper's eq. 9's own remark).

**What this does and doesn't claim.** The paper's `c_s in [20,40] m/s` is an empirical range
for real starlings at their own physical scale (m, s) — this project's core is dimensionless
throughout (design/00_overview.md §2), so there is no calibrated conversion to compare that
literal range against. What *is* dimensionless and directly testable is the paper's headline
*structural* claim: linear/ballistic propagation, not diffusive — `propagation_speed`'s
`r2_linear` vs `r2_diffusive` comparison operationalizes exactly that, regardless of units.
"""

import numpy as np

from murmuration.analysis import density as density_mod


def dominant_motion_plane(vel_traj: np.ndarray) -> tuple:
    """vel_traj: (T, N, 3). Returns an orthonormal in-plane basis (u, v) spanning the flock's
    dominant motion plane — the two largest-variance directions of the velocity trajectory
    (PCA), matching `murmur_spin_wave`'s own module-doc description of the "computed from
    PCA" fallback. Computed here in analysis, not read back from the plugin, which lacks the
    cross-boid access needed to do this itself at `respond()` time (see that plugin's doc)."""
    vel_traj = np.asarray(vel_traj, dtype=np.float64)
    flat = vel_traj.reshape(-1, 3)
    flat = flat - flat.mean(axis=0)
    cov = np.cov(flat.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    u = eigvecs[:, order[0]]
    v = eigvecs[:, order[1]]
    u = u / np.linalg.norm(u)
    v = v - (v @ u) * u  # re-orthogonalize defensively against numerical drift
    v = v / np.linalg.norm(v)
    return u, v


def heading_angles(vel_traj: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Projects (T, N, 3) velocities onto the (u, v) plane; returns unwrapped per-boid
    heading angle phi_i(t), shape (T, N). Absolute angle values are basis-dependent (PCA
    fixes a plane, not an in-plane reference direction) — only *relative* timing between
    boids, which is all the delay/dispersion pipeline below uses, is meaningful."""
    vel_traj = np.asarray(vel_traj, dtype=np.float64)
    x = vel_traj @ u
    y = vel_traj @ v
    phi = np.arctan2(y, x)
    return np.unwrap(phi, axis=0)


def _cross_correlation_lag(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Lag (time units) at which `b` best matches `a`: positive means b's pattern appears
    `lag` time units after a's (b lags a). Full cross-correlation (`numpy.correlate`) plus
    parabolic interpolation around the discrete peak for sub-step precision."""
    a = a - a.mean()
    b = b - b.mean()
    n = len(a)
    corr = np.correlate(a, b, mode="full")  # lags run from -(n-1) to (n-1)
    lags = np.arange(-(n - 1), n)
    k = int(np.argmax(corr))
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    # numpy.correlate(a, b, "full")'s peak position gives the shift that aligns b *backward*
    # onto a (i.e. how far a leads b), the opposite sign of this function's own stated
    # convention ("positive means b lags a") — negate to match, pinned by
    # test_pairwise_delays_recovers_a_known_delay_with_correct_sign.
    return -(lags[k] + delta) * dt


def pairwise_delays(angles: np.ndarray, dt: float) -> np.ndarray:
    """angles: (T, N) unwrapped heading angles. Returns the (N, N) antisymmetric delay
    matrix tau: tau[i, j] is the amount of time by which bird j turns before bird i
    (tau > 0) or after (tau < 0) — Attanasi et al. 2014's own sign convention (main text,
    page 2). Computed from cross-correlating each pair's angular-velocity signal (dphi/dt),
    matching the paper's use of the *turning event* itself, not raw heading."""
    omega = np.gradient(angles, dt, axis=0)  # (T, N)
    n = angles.shape[1]
    tau = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # lag > 0: omega[:, j] best matches omega[:, i] delayed by `lag`, i.e. j's turn
            # happens `lag` after i's, i.e. i turns before j -> tau_ij (Attanasi's "j turns
            # before i" convention) is the negative of that.
            lag = _cross_correlation_lag(omega[:, i], omega[:, j], dt)
            tau[i, j] = -lag
            tau[j, i] = lag
    return tau


def turning_times(tau: np.ndarray) -> np.ndarray:
    """Least-squares-consistent per-boid turning time t_i from the pairwise delay matrix:
    minimizes sum_ij (t_i - t_j - tau_ij)^2 subject to tau_ij = t_i - t_j, whose normal-
    equations solution (for an all-pairs, antisymmetric design with zero-mean noise) is the
    row mean of tau. t_i is defined only up to an additive constant (no absolute time
    origin) — only relative ordering/ranking and inter-boid spacing matter downstream."""
    return tau.mean(axis=1)


def dispersion_curve(t: np.ndarray, density_value: float) -> tuple:
    """Rank-ordered dispersion curve x(t): sorted turning times t_(1) <= ... <= t_(N), and
    x_(k) = (k / density)^(1/3) — the radius of the sphere containing the first k birds to
    turn, assuming a spatially localized turn origin in a homogeneous 3D flock (Attanasi et
    al.'s own x(t) definition, main text page 2). Uses this project's own
    `analysis.density.density()` for rho, for consistency with P-c. Returns (t_sorted, x)."""
    t_sorted = np.sort(t)
    ranks = np.arange(1, len(t) + 1)
    x = (ranks / density_value) ** (1.0 / 3.0)
    return t_sorted, x


def propagation_speed(t_sorted: np.ndarray, x: np.ndarray, fit_quantiles=(0.1, 0.9)) -> dict:
    """Fits the dispersion curve's early/intermediate regime (excludes both ends, matching
    the paper's own "clear linear regime for early and intermediate times, before border
    effects kick in" caveat) to two competing models: linear (`x = c_s*t + b`, the ballistic/
    behavioural-inertia prediction, eq. 8) and diffusive (`x = k*sqrt(t)`, the memoryless/
    `InstantResponse` prediction, eq. 4). Both models are fit with exactly one free
    parameter, forced through the dispersion curve's own true physical origin — the first
    bird to turn (`t=0 -> x=0`, before which no information has traveled) — *not* re-zeroed
    to the fit window's own start: doing that would erase exactly the curvature that
    distinguishes the two hypotheses (any smooth curve looks locally near-linear once
    re-zeroed to a nonzero point well inside it). The `fit_quantiles` window only *selects*
    which points participate, matching the paper's own "before border effects kick in"
    caveat about the curve's tail — it doesn't move the origin. An unequal parameter count
    (e.g. an unconstrained intercept on one model but not the other) would separately bias
    the R^2 comparison regardless of which model actually describes the data — this is why
    both are single-parameter fits. Returns c_s (the linear fit's slope) and each model's
    R^2, plus which one the data favours — this comparison, not `c_s`'s absolute value, is
    the dimensionless, structural form of the paper's headline result."""
    t0, x0 = t_sorted[0], x[0]
    t_full = t_sorted - t0
    x_full = x - x0
    lo, hi = np.quantile(t_full, fit_quantiles)
    mask = (t_full >= lo) & (t_full <= hi)
    if mask.sum() < 3:
        return {
            "c_s": float("nan"),
            "r2_linear": 0.0,
            "r2_diffusive": 0.0,
            "favours": "insufficient_data",
        }
    t_fit = t_full[mask]
    x_fit = x_full[mask]
    ss_tot = np.sum((x_fit - x_fit.mean()) ** 2)

    denom_lin = np.sum(t_fit**2)
    c_s = np.sum(t_fit * x_fit) / denom_lin if denom_lin > 0 else 0.0
    pred_lin = c_s * t_fit
    r2_linear = 1.0 - np.sum((x_fit - pred_lin) ** 2) / ss_tot if ss_tot > 0 else 1.0

    sqrt_t = np.sqrt(np.clip(t_fit, 0.0, None))
    denom_diff = np.sum(sqrt_t**2)
    k = np.sum(sqrt_t * x_fit) / denom_diff if denom_diff > 0 else 0.0
    pred_diff = k * sqrt_t
    r2_diffusive = 1.0 - np.sum((x_fit - pred_diff) ** 2) / ss_tot if ss_tot > 0 else 1.0

    favours = "linear" if r2_linear >= r2_diffusive else "diffusive"
    return {
        "c_s": float(c_s),
        "r2_linear": float(r2_linear),
        "r2_diffusive": float(r2_diffusive),
        "favours": favours,
    }


def speed_vs_polarisation(c_s_by_phi: dict) -> dict:
    """Eq. 10: c_s = [a/sqrt(beta*chi)] * 1/sqrt(1-Phi) — c_s should scale linearly with
    1/sqrt(1-Phi), slope = a/sqrt(beta*chi) (flock-independent if beta/chi are). Given
    {Phi: c_s} pairs from several runs at different noise/coupling settings, fits that
    relation and reports slope/intercept/R^2 as a structural check of eq. 10 (dimensionless
    — the slope's absolute value isn't compared to anything, only the linearity is)."""
    phis = np.array(sorted(c_s_by_phi))
    c_s_vals = np.array([c_s_by_phi[p] for p in phis])
    x = 1.0 / np.sqrt(np.clip(1.0 - phis, 1e-12, None))
    slope, intercept = np.polyfit(x, c_s_vals, 1)
    pred = slope * x + intercept
    ss_tot = np.sum((c_s_vals - c_s_vals.mean()) ** 2)
    r2 = 1.0 - np.sum((c_s_vals - pred) ** 2) / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def information_transfer_report(pos_traj: np.ndarray, vel_traj: np.ndarray, dt: float) -> dict:
    """Top-level orchestrator: pos_traj/vel_traj (T, N, 3) -> the full pipeline above.
    density is measured from the trajectory's final frame (the flock's state once the
    analyzed turn has propagated through it)."""
    density_value = density_mod.density(np.asarray(pos_traj)[-1])
    u, v = dominant_motion_plane(vel_traj)
    angles = heading_angles(vel_traj, u, v)
    tau = pairwise_delays(angles, dt)
    t = turning_times(tau)
    t_sorted, x = dispersion_curve(t, density_value)
    result = propagation_speed(t_sorted, x)
    result["density"] = density_value
    return result
