//! Rust-native H2 spike (roadmap.md §3 item 5 / §6.2 Phase 14: "H2 Rust-native
//! sparse-eigensolver spike ... reproduces a known Laplacian spectrum, completes within
//! budget"). Mirrors `claude/python/murmuration/analysis/h2.py` field-for-field: brute-force
//! (not kd-tree — spike scale only) m-nearest-neighbour graph, max-symmetrized adjacency,
//! Laplacian L = D - A, `k = min(N-1, 200)` smallest eigenvalues, same H2/R_nodal/R_per_m
//! formulas and the same `near_zero`/zero-eigenvalue-tolerance connectivity check.
//!
//! Candidate evaluated here: `nalgebra`'s dense `SymmetricEigen` (full spectrum, then take
//! the smallest `k`) rather than a true sparse partial solver (h2.py's `scipy.sparse.linalg.
//! eigsh` shift-invert Lanczos) — proves the *math* matches first; a sparse partial solver
//! (`faer`, `nalgebra-sparse`, or a hand-rolled Lanczos) is the follow-up needed before this
//! scales to N=2600 (dense eigendecomposition is O(N^3), fine for a spike, not for that N).

use nalgebra::{DMatrix, SymmetricEigen};

pub const ZERO_EIGENVALUE_TOL: f64 = 1e-9;

pub struct H2Result {
    pub h2: f64,
    pub r_nodal: f64,
    pub r_per_m: f64,
    pub near_zero: usize,
}

pub fn knn_indices(pos: &[[f64; 3]], i: usize, m: usize) -> Vec<usize> {
    let n = pos.len();
    let mut dists: Vec<(f64, usize)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            ((dx * dx + dy * dy + dz * dz).sqrt(), j)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(m).map(|(_, j)| j).collect()
}

/// Graph Laplacian L = D - A of the m-nearest-neighbour graph, uniform weight 1/m,
/// undirected via `a = max(a, a^T)` — matches h2.py's `a.maximum(a.T)` exactly.
pub fn laplacian(pos: &[[f64; 3]], m: usize) -> DMatrix<f64> {
    let n = pos.len();
    let w = 1.0 / m as f64;
    let mut dir = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in knn_indices(pos, i, m) {
            dir[(i, j)] = w;
        }
    }
    let mut a = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                a[(i, j)] = dir[(i, j)].max(dir[(j, i)]);
            }
        }
    }
    let mut lap = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let deg: f64 = a.row(i).sum();
        lap[(i, i)] = deg;
        for j in 0..n {
            if i != j {
                lap[(i, j)] = -a[(i, j)];
            }
        }
    }
    lap
}

pub fn eigenvalues_sorted(lap: &DMatrix<f64>) -> Vec<f64> {
    let eig = SymmetricEigen::new(lap.clone());
    let mut v: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Mirrors h2.py's `h2_curve` body for one `m`: same `k = min(N-1, 200)` truncation (h2.py's
/// `eigsh(k=...)` only ever returns that many of the smallest-magnitude eigenvalues after the
/// shift-invert, so a full dense spectrum is truncated the same way here for a fair
/// comparison), same near-zero count, same H2/R_nodal/R_per_m formulas.
pub fn h2_at_m(pos: &[[f64; 3]], m: usize) -> H2Result {
    let n = pos.len();
    let lap = laplacian(pos, m);
    let eigs = eigenvalues_sorted(&lap);
    let k = (n - 1).min(200);
    let lam_all = &eigs[..k];
    let near_zero = lam_all.iter().filter(|&&x| x <= ZERO_EIGENVALUE_TOL).count();
    let lam: Vec<f64> = lam_all
        .iter()
        .copied()
        .filter(|&x| x > ZERO_EIGENVALUE_TOL)
        .collect();
    if lam.is_empty() {
        return H2Result {
            h2: f64::INFINITY,
            r_nodal: 0.0,
            r_per_m: 0.0,
            near_zero,
        };
    }
    let sum_inv: f64 = lam.iter().map(|&x| 1.0 / x).sum();
    let h2 = ((1.0 / (2.0 * n as f64)) * sum_inv).sqrt();
    let r_nodal = 1.0 / (h2 / (n as f64).sqrt());
    let r_per_m = r_nodal / m as f64;
    H2Result {
        h2,
        r_nodal,
        r_per_m,
        near_zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph_laplacian(n: usize) -> DMatrix<f64> {
        let mut lap = DMatrix::<f64>::from_element(n, n, -1.0);
        for i in 0..n {
            lap[(i, i)] = (n - 1) as f64;
        }
        lap
    }

    fn cycle_graph_laplacian(n: usize) -> DMatrix<f64> {
        let mut lap = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            lap[(i, i)] = 2.0;
            let next = (i + 1) % n;
            let prev = (i + n - 1) % n;
            lap[(i, next)] = -1.0;
            lap[(i, prev)] = -1.0;
        }
        lap
    }

    // Known closed-form spectra (no h2.py involved) — this is the spike's core exit gate.

    #[test]
    fn complete_graph_k_n_has_known_spectrum() {
        // K_n Laplacian eigenvalues: 0 (multiplicity 1), n (multiplicity n-1).
        let n = 8;
        let eigs = eigenvalues_sorted(&complete_graph_laplacian(n));
        assert!(eigs[0].abs() < 1e-9, "expected 0, got {}", eigs[0]);
        for &lam in &eigs[1..] {
            assert!((lam - n as f64).abs() < 1e-9, "expected {n}, got {lam}");
        }
    }

    #[test]
    fn cycle_graph_c_n_has_known_spectrum() {
        // C_n Laplacian eigenvalues: 2 - 2*cos(2*pi*k/n), k = 0..n-1.
        let n = 10;
        let eigs = eigenvalues_sorted(&cycle_graph_laplacian(n));
        let mut expected: Vec<f64> = (0..n)
            .map(|k| 2.0 - 2.0 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in eigs.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got} want {want}");
        }
    }

    fn sample_cloud(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                [t.sin() * 5.0, t.cos() * 5.0, (t * 0.7).sin() * 3.0]
            })
            .collect()
    }

    #[test]
    fn knn_laplacian_rows_sum_to_zero() {
        let lap = laplacian(&sample_cloud(12), 3);
        for i in 0..lap.nrows() {
            let row_sum: f64 = lap.row(i).sum();
            assert!(row_sum.abs() < 1e-9, "row {i} sum {row_sum}");
        }
    }

    #[test]
    fn knn_laplacian_smallest_eigenvalue_is_near_zero_for_connected_graph() {
        let eigs = eigenvalues_sorted(&laplacian(&sample_cloud(12), 3));
        assert!(eigs[0].abs() < 1e-9);
    }

    #[test]
    fn h2_at_m_is_finite_and_positive_for_a_connected_graph() {
        let r = h2_at_m(&sample_cloud(12), 3);
        assert!(r.h2.is_finite() && r.h2 > 0.0);
        assert!(r.r_nodal > 0.0);
        assert_eq!(r.near_zero, 1);
    }

    // Cross-check against claude/python/murmuration/analysis/h2.py: same 12-point cloud
    // (`sample_cloud(12)`), same m, run through h2.py's own h2_curve directly (see
    // sci/param_table.md / this crate's module doc for how these numbers were produced —
    // recorded as a fixture rather than shelling out to Python at test time, matching this
    // repo's `fixtures/golden/` pattern).
    #[test]
    fn h2_at_m_matches_h2_py_on_the_same_point_cloud() {
        let r = h2_at_m(&sample_cloud(12), 3);
        // Recorded from: python3 -c "... h2_curve(sample_cloud(12), m_values=[3]) ..."
        // against claude/python/murmuration/analysis/h2.py, 2026-08-05.
        let expected_h2 = 0.7295956926228141;
        let expected_r_nodal = 4.7479743235389735;
        let expected_near_zero = 1usize;
        assert!(
            (r.h2 - expected_h2).abs() < 1e-6,
            "h2: got {} want {}",
            r.h2,
            expected_h2
        );
        assert!(
            (r.r_nodal - expected_r_nodal).abs() < 1e-6,
            "r_nodal: got {} want {}",
            r.r_nodal,
            expected_r_nodal
        );
        assert_eq!(r.near_zero, expected_near_zero);
    }
}
