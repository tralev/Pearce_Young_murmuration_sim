"""Static structure factor (Brambati/Fava/Ginelli 2023, `sci/2205.07724v3.pdf`) —
design/03_observables_bindings.md §3, roadmap.md Phase 15. The "static approach" half of
that paper's two complementary directed-vs-spontaneous-flocking signatures (the other half
is `direction_fluctuation.py`).

The static density structure factor (paper's eq. 22-24):
    S(q) = (1/N) < |sum_n exp(i q . r_n)|^2 >,   S(q) = <S(q)>_{|q|=q}  (isotropic average)

Toner & Tu theory predicts the free (spontaneous, no external field) flock's isotropically-
averaged structure factor **diverges** algebraically at small wavenumber, `S(q) ~ q^-zeta`
(`zeta > 0`, eq. 25) — the density-fluctuation signature of the long-ranged velocity
correlations a Nambu-Goldstone mode produces. A small, homogeneous external field
*suppresses* this divergence at a field-dependent cutoff wavenumber, `S(q,h) ~ 1/(q^zeta +
C*h)` (eq. 26), which is finite as `q -> 0`: `S(0,h) ~ 1/h` (eq. 39). So: does the
isotropically-averaged `S(q)` at small `q` diverge (spontaneous) or saturate to a finite
value (directed)?

**Dimensionality note.** The source paper works in 2D (a Vicsek-in-a-plane system); this
project is 3D throughout (design/00_overview.md §2's strict-3D invariant). `S(q)`'s
isotropic average here is over random directions on `S^2`, not the circle — the physics
(the actual value of `zeta`) may differ in 3D, but the *qualitative* divergence-vs-
saturation test this module implements is dimension-agnostic (a direct consequence of
`S(q,h)`'s functional form, eq. 26, not of any particular dimension).
"""

import numpy as np


def structure_factor(
    pos: np.ndarray, q_values: np.ndarray, n_directions: int = 64, rng=None
) -> np.ndarray:
    """pos: (N, 3) positions. q_values: (K,) wavevector magnitudes |q|. Returns S(q) for
    each entry — the isotropic average (eq. 22-23) of `S(q) = (1/N)|sum_n exp(i q.r_n)|^2`
    over `n_directions` random directions on S^2 (area-uniform, same formula as
    `murmur_core::rng::sample_unit_sphere`)."""
    if rng is None:
        rng = np.random.default_rng()
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]

    z = 1.0 - 2.0 * rng.random(n_directions)
    theta = 2.0 * np.pi * rng.random(n_directions)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    dirs = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)  # (n_directions, 3)

    s_q = np.empty(len(q_values))
    for k, q_mag in enumerate(q_values):
        q_vecs = dirs * q_mag  # (n_directions, 3)
        # macOS's Accelerate BLAS backend (numpy's default there) emits spurious "divide by
        # zero"/"overflow"/"invalid value" RuntimeWarnings on some matmul shapes even when
        # every input and output is genuinely finite and correct (verified directly: not
        # observed with other BLAS backends, and `phase` itself never actually contains
        # NaN/Inf here) — a known Accelerate quirk, not a real numerical problem.
        with np.errstate(all="ignore"):
            phase = pos @ q_vecs.T  # (N, n_directions)
        rho_q = np.sum(np.exp(1j * phase), axis=0)  # (n_directions,)
        s_q[k] = np.mean(np.abs(rho_q) ** 2) / n
    return s_q


def small_q_scaling(q_values: np.ndarray, s_q: np.ndarray, diverging_threshold: float = 0.1) -> dict:
    """Fits log S(q) vs log q (should be evaluated over a small-q range) to a power law.
    Returns the fitted exponent `zeta` (`S(q) ~ q^-zeta`) and whether the data is
    classified as diverging: the free-flock prediction is `zeta > 0` (eq. 25); the field-
    suppressed prediction approaches a finite constant as q -> 0 (eq. 39), i.e. `zeta ~ 0`
    or negative over a small-q window that's already past the field's cutoff wavenumber.
    `diverging_threshold` separates a real divergence from fit noise around zero."""
    log_q = np.log(q_values)
    log_s = np.log(s_q)
    slope, _intercept = np.polyfit(log_q, log_s, 1)
    zeta = -slope
    return {"zeta": float(zeta), "diverges": bool(zeta > diverging_threshold)}
