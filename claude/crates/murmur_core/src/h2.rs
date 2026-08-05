//! H₂ robustness (Young 2013) — Rust-native path (design/03_observables_bindings.md §3.1,
//! roadmap.md Phase 14). Promoted from `claude/spikes/rust_h2/` (2026-08-05) once that spike
//! validated the approach — see its own module doc for the closed-form-spectrum proof and the
//! candidate-evaluation record (`nalgebra`'s dense `SymmetricEigen`, not yet a sparse partial
//! solver — still O(N³), fine at this project's slice scale, not for a 300k-boid run).
//!
//! Mirrors `claude/python/murmuration/analysis/h2.py` field-for-field: brute-force (not
//! spatial-index-accelerated — see the spike's own note) m-nearest-neighbour graph,
//! max-symmetrized adjacency, Laplacian `L = D - A`, `k = min(N-1, 200)` truncation, the same
//! H₂/R_nodal/R_per_m formulas and the same near-zero-eigenvalue connectivity check.
//!
//! First real consumer: `murmur_young` (Track C Tier C2, Phase 14) — the `FlockingMode` this
//! path exists for, computing a genuine per-step m* rather than a fixed fallback. Also the
//! prerequisite for `RequestMetric{H2Curve}` (`batch.rs`) ever being real over the C ABI.

use nalgebra::{DMatrix, SymmetricEigen};

use crate::math::Vec3;

pub const ZERO_EIGENVALUE_TOL: f64 = 1e-9;

#[derive(Debug, Clone, Copy)]
pub struct H2Result {
    pub h2: f64,
    pub r_nodal: f64,
    pub r_per_m: f64,
    pub near_zero: usize,
}

fn knn_indices(pos: &[Vec3], i: usize, m: usize) -> Vec<usize> {
    let n = pos.len();
    let mut dists: Vec<(f64, usize)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| ((pos[j] - pos[i]).len(), j))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(m).map(|(_, j)| j).collect()
}

/// Graph Laplacian `L = D - A` of the m-nearest-neighbour graph, uniform weight `1/m`,
/// undirected via `a = max(a, a^T)` — matches `h2.py`'s `a.maximum(a.T)` exactly.
pub fn laplacian(pos: &[Vec3], m: usize) -> DMatrix<f64> {
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

/// Mirrors `h2.py`'s `h2_curve` body for one `m`: same `k = min(N-1, 200)` truncation, same
/// near-zero count, same H₂/R_nodal/R_per_m formulas.
pub fn h2_at_m(pos: &[Vec3], m: usize) -> H2Result {
    let n = pos.len();
    let lap = laplacian(pos, m);
    let eigs = eigenvalues_sorted(&lap);
    let k = (n - 1).min(200);
    let lam_all = &eigs[..k];
    let near_zero = lam_all
        .iter()
        .filter(|&&x| x <= ZERO_EIGENVALUE_TOL)
        .count();
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

/// Young 2013's Y-a: `m* = argmax_m` robustness-per-neighbour, `R_per_m`. `m_values` should
/// span the plausible range (e.g. `2..=12`, matching `sci/todo.md`'s own sweep) — `m=1` is
/// deliberately excluded by convention (a 1-NN graph's `R_per_m` is often degenerate/
/// disconnected and not a meaningful candidate for the optimum, matching `h2.py`'s own
/// documented range starting at 1 but empirically never selecting it as `m*`).
pub fn m_star(pos: &[Vec3], m_values: impl IntoIterator<Item = usize>) -> Option<usize> {
    m_values
        .into_iter()
        .filter(|&m| m < pos.len())
        .map(|m| (m, h2_at_m(pos, m)))
        .max_by(|(_, a), (_, b)| a.r_per_m.partial_cmp(&b.r_per_m).unwrap())
        .map(|(m, _)| m)
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

    #[test]
    fn complete_graph_k_n_has_known_spectrum() {
        let n = 8;
        let eigs = eigenvalues_sorted(&complete_graph_laplacian(n));
        assert!(eigs[0].abs() < 1e-9, "expected 0, got {}", eigs[0]);
        for &lam in &eigs[1..] {
            assert!((lam - n as f64).abs() < 1e-9, "expected {n}, got {lam}");
        }
    }

    #[test]
    fn cycle_graph_c_n_has_known_spectrum() {
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

    fn sample_cloud(n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                Vec3::new(t.sin() * 5.0, t.cos() * 5.0, (t * 0.7).sin() * 3.0)
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
    fn h2_at_m_matches_the_python_h2_py_reference_on_the_same_point_cloud() {
        // Recorded from claude/spikes/rust_h2's own cross-check against
        // claude/python/murmuration/analysis/h2.py (2026-08-05, same fixture cloud/m).
        let r = h2_at_m(&sample_cloud(12), 3);
        let expected_h2 = 0.7295956926228141;
        let expected_r_nodal = 4.7479743235389735;
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
        assert_eq!(r.near_zero, 1);
    }

    #[test]
    fn m_star_picks_the_argmax_of_r_per_m() {
        let pos = sample_cloud(30);
        let m = m_star(&pos, 2..=12).expect("connected cloud should have a well-defined m*");
        let chosen = h2_at_m(&pos, m).r_per_m;
        for candidate in 2..=12 {
            if candidate == m {
                continue;
            }
            assert!(
                chosen >= h2_at_m(&pos, candidate).r_per_m - 1e-12,
                "m*={m} should maximize R_per_m, but m={candidate} scored higher"
            );
        }
    }

    #[test]
    fn m_star_is_none_for_a_population_too_small_for_any_candidate_m() {
        let pos = sample_cloud(3);
        assert_eq!(m_star(&pos, 5..=12), None);
    }
}
