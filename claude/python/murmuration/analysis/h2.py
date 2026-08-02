"""H2 robustness (Young 2013) — design/03_observables_bindings.md §3.1.

Heading agreement is noisy linear consensus on the m-nearest-neighbour graph, uniform weights
a_ij = 1/m (Young Fig. S1: uniform is both simplest and most robust). Stacked: dx/dt = -Lx + xi,
Laplacian L = D - A.

H2 = sqrt(Trace(Sigma)) = sqrt((1/2N) * sum_{i>=2} 1/lambda_i)   lambda_2..lambda_N of L
R_nodal = 1 / (H2 / sqrt(N))          per-individual robustness (size-normalised, inverted)
R_per_m = R_nodal / m                  robustness per neighbour (sensing cost)
m*      = argmax_m R_per_m             finite optimum => m* ~ 6-7

This is the reference implementation the (deferred) Rust-native H2 path is validated against
(design/03 §3.1) — this project ships the Python path only for now; see roadmap.md decision
D18's documented fallback ("Python-only H2 for v1... revisit").
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

# A connected graph's Laplacian has an algebraic-multiplicity-1 zero eigenvalue (one per
# connected component); more than one eigenvalue at/near 0 means the graph is disconnected.
_ZERO_EIGENVALUE_TOL = 1e-9


def h2_curve(pos: np.ndarray, m_values=range(1, 13)) -> dict:
    """pos: (N,3) f64. Returns dict m -> (H2, R_nodal, R_per_m, near_zero_eigenvalue_count).

    `near_zero_eigenvalue_count` is the number of connected components seen among the
    eigenvalues actually computed (capped at `k = min(N-1, 200)`, so on a very large N this
    is a lower bound on the true component count, not necessarily exact — irrelevant at this
    project's slice scale, N<=2600 with k=200 comfortably covering the low end of the
    spectrum where any disconnection shows up).
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    tree = cKDTree(pos)
    out = {}
    for m in m_values:
        if m >= n:
            out[m] = (np.inf, 0.0, 0.0, n)
            continue
        _, idx = tree.query(pos, k=m + 1)  # +1: self is nearest
        rows = np.repeat(np.arange(n), m)
        cols = idx[:, 1:].ravel()
        a = csr_matrix((np.full(rows.size, 1.0 / m), (rows, cols)), shape=(n, n))
        a = a.maximum(a.T)  # undirected symmetrisation
        laplacian = csr_matrix(np.diag(np.asarray(a.sum(1)).ravel())) - a
        k = min(n - 1, 200)
        # A graph Laplacian is exactly singular at shift 0 (every connected component has a
        # 0-eigenvalue with the all-ones eigenvector), so shift-invert factorization at
        # sigma=0 fails outright ("Factor is exactly singular") rather than just being
        # ill-conditioned — a known scipy/ARPACK pitfall for Laplacians, not a design bug to
        # work around silently. A tiny positive shift keeps the matrix invertible while still
        # targeting the smallest eigenvalues.
        lam_all = np.sort(
            eigsh(laplacian, k=k, sigma=1e-8, which="LM", return_eigenvectors=False)
        )
        near_zero = int(np.sum(lam_all <= _ZERO_EIGENVALUE_TOL))
        lam = lam_all[lam_all > _ZERO_EIGENVALUE_TOL]
        if lam.size == 0:
            out[m] = (np.inf, 0.0, 0.0, near_zero)
            continue
        h2 = np.sqrt((1.0 / (2 * n)) * np.sum(1.0 / lam))
        r_nodal = 1.0 / (h2 / np.sqrt(n))
        out[m] = (float(h2), float(r_nodal), float(r_nodal / m), near_zero)
    return out


def m_star(curve: dict) -> int:
    """Y-a: argmax_m robustness-per-neighbour."""
    return max(curve, key=lambda m: curve[m][2])


def is_connected_at(curve: dict, m: int) -> bool:
    """Y-d: the m-NN consensus graph is connected iff its Laplacian has exactly one
    near-zero eigenvalue (one connected component)."""
    _h2, _r_nodal, _r_per_m, near_zero = curve[m]
    return near_zero <= 1
