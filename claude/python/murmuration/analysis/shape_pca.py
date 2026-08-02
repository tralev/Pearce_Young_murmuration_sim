"""Flock shape PCA (Young 2013) — design/03_observables_bindings.md §3.2.

C = (1/N) sum_i (r_i - r_bar)(r_i - r_bar)^T          3x3 covariance
eig(C): lambda_1 >= lambda_2 >= lambda_3               variance along principal axes
aspect_ratio  = sqrt(lambda_1 / lambda_3)   (>= 1)
thickness     = sqrt(lambda_3 / lambda_1)   (in (0, 1])
"""

import numpy as np


def flock_shape(pos: np.ndarray) -> dict:
    """pos: (N, 3). Returns eigs (descending), aspect, thickness."""
    pos = np.asarray(pos, dtype=np.float64)
    c = np.cov((pos - pos.mean(axis=0)).T)
    lam = np.sort(np.linalg.eigvalsh(c))[::-1]  # lambda_1 >= lambda_2 >= lambda_3
    lam = np.clip(lam, 0.0, None)  # guard tiny negative numerical noise
    aspect = np.sqrt(lam[0] / lam[2]) if lam[2] > 0 else np.inf
    thickness = np.sqrt(lam[2] / lam[0]) if lam[0] > 0 else 0.0
    return {"eigs": lam, "aspect": float(aspect), "thickness": float(thickness)}


def m_star_from_shape(aspect: float) -> float:
    """Empirical m*-from-shape mapping (sci/todo.md): linear interpolation between
    (aspect=1 -> m*=9.78) and (aspect=3 -> m*=6.05), clamped to that range."""
    t = np.clip((aspect - 1.0) / (3.0 - 1.0), 0.0, 1.0)
    return 9.78 + t * (6.05 - 9.78)
