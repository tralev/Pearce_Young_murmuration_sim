"""Mean-direction fluctuations (Brambati/Fava/Ginelli 2023, `sci/2205.07724v3.pdf`) —
design/03_observables_bindings.md §3, roadmap.md Phase 15. The "dynamical approach" half of
that paper's two complementary directed-vs-spontaneous-flocking signatures (the other half
is `structure_factor.py`). Needs a run with the `external_field` `StepHook` active for the
directed-motion comparison case.

The flock's mean direction (paper's eq. 6-7, generalized here from a 2D angle to the 3D
`Omega(t) = (1/N) sum_i v_hat_i(t)` this project's own `polarisation` metric already takes
the magnitude of): `delta_psi(t)`, the geodesic angle between `Omega_hat(t)` and a
reference direction, is predicted to grow **diffusively and unboundedly** in the absence of
an external field (free rotational diffusion of the Nambu-Goldstone mode, eq. 14:
`<delta_psi(t)^2> ~ 2*D0*t`), but to **plateau** at a finite value under even a small,
constant external field (eq. 12-13, an Ornstein-Uhlenbeck relaxation toward the field
direction, stiffness proportional to the field amplitude `h`): does `<delta_psi(t)^2>` grow
without bound (spontaneous) or saturate (directed)?
"""

import numpy as np


def mean_direction(vel_traj: np.ndarray) -> np.ndarray:
    """vel_traj: (T, N, 3). Returns Omega(t), shape (T, 3): the flock's mean heading vector
    at each frame (eq. 6's `Omega(t) = (1/N) sum s_i`, generalized to 3D) — exactly the
    vector this project's own `polarisation` metric takes the magnitude of,
    `|Omega(t)| = alpha(t) in [0,1]`. Zero-speed boids contribute the zero vector, never
    NaN."""
    vel_traj = np.asarray(vel_traj, dtype=np.float64)
    speed = np.linalg.norm(vel_traj, axis=-1, keepdims=True)
    heading = np.divide(vel_traj, speed, out=np.zeros_like(vel_traj), where=speed > 1e-9)
    return heading.mean(axis=1)  # (T, 3)


def angular_deviation(omega: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """omega: (T, 3) (not necessarily unit length — `|omega(t)|` is the polarisation
    `alpha(t)`). reference: (3,), a fixed unit vector — the field direction for a directed
    run, or `omega[0]` itself for a spontaneous run (paper's `theta_h`, eq. 7, generalized:
    for h=0 there's no field direction, so the run's own initial mean direction is the only
    available reference). Returns delta_psi(t), the geodesic angle (radians) between
    `Omega_hat(t)` and `reference` — the 3D generalization of the paper's scalar
    `delta_psi = psi(t) - theta_h`."""
    omega = np.asarray(omega, dtype=np.float64)
    norm = np.linalg.norm(omega, axis=-1)
    safe = np.where(norm > 1e-9, norm, 1.0)
    omega_hat = omega / safe[:, None]
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref / np.linalg.norm(ref)
    cos_angle = np.clip(omega_hat @ ref, -1.0, 1.0)
    return np.arccos(cos_angle)


def classify_growth(t: np.ndarray, msd: np.ndarray, fit_quantiles=(0.2, 0.9)) -> dict:
    """t: (T,) elapsed time (t[0] should be 0). msd: (T,) `delta_psi(t)^2` (`msd[0]` should
    be 0 — `delta_psi(0)=0` by construction if `reference` was chosen as the run's own
    initial/field direction). Fits the given time window to two competing, single-free-
    parameter models anchored at the true origin `(0, 0)` (matching
    `information_transfer.propagation_speed`'s reasoning: an unequal parameter count would
    bias the R^2 comparison): linear (`msd = 2*D0*t`, the diffusive/spontaneous prediction,
    eq. 14) and a saturating exponential (`msd = a*(1 - exp(-t/tau))`, the Ornstein-
    Uhlenbeck/directed prediction, eq. 12-13 — `tau` is fit via a coarse log-spaced search
    since it enters nonlinearly, `a` by linear least squares at each candidate `tau`).
    Returns each model's fit and R^2, plus which the data favours."""
    lo, hi = np.quantile(t, fit_quantiles)
    mask = (t >= lo) & (t <= hi)
    if mask.sum() < 3:
        return {
            "diffusion_constant": float("nan"),
            "r2_diffusive": 0.0,
            "r2_plateau": 0.0,
            "favours": "insufficient_data",
        }
    t_fit = t[mask]
    msd_fit = msd[mask]
    ss_tot = np.sum((msd_fit - msd_fit.mean()) ** 2)

    denom = np.sum(t_fit**2)
    two_d0 = np.sum(t_fit * msd_fit) / denom if denom > 0 else 0.0
    pred_diff = two_d0 * t_fit
    r2_diffusive = 1.0 - np.sum((msd_fit - pred_diff) ** 2) / ss_tot if ss_tot > 0 else 1.0

    def best_fit_over(tau_candidates):
        best = (-np.inf, 0.0, tau_candidates[0])
        for tau in tau_candidates:
            basis = 1.0 - np.exp(-t_fit / tau)
            denom_b = np.sum(basis**2)
            a = np.sum(basis * msd_fit) / denom_b if denom_b > 0 else 0.0
            pred = a * basis
            r2 = 1.0 - np.sum((msd_fit - pred) ** 2) / ss_tot if ss_tot > 0 else 1.0
            if r2 > best[0]:
                best = (r2, a, tau)
        return best

    # Coarse log-spaced search, then a finer refinement bracketing the coarse optimum —
    # tau enters the model nonlinearly, so a single coarse grid alone can land noticeably
    # off the true optimum even when the data is an exact match to the model.
    t_span = max(t_fit[-1] - t_fit[0], 1e-9)
    coarse = np.geomspace(t_span / 50.0, t_span * 5.0, 40)
    best_r2_plateau, best_a, best_tau = best_fit_over(coarse)
    fine = np.geomspace(best_tau / 3.0, best_tau * 3.0, 60)
    best_r2_plateau, best_a, best_tau = best_fit_over(fine)

    favours = "diffusive" if r2_diffusive >= best_r2_plateau else "plateau"
    return {
        "diffusion_constant": float(two_d0 / 2.0),
        "r2_diffusive": float(r2_diffusive),
        "r2_plateau": float(best_r2_plateau),
        "plateau_value": float(best_a),
        "plateau_tau": float(best_tau),
        "favours": favours,
    }
