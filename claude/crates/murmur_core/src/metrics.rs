//! Cheap per-frame metrics (design/03_observables_bindings.md §1) — O(N) or O(N log N)
//! reductions over the columns, no heavy linear algebra. Heavy quantities (H₂, PCA shape,
//! convex-hull density, τρ) live in Python analysis instead (§3 of that doc).
//!
//! **Scope note on `opacity_ext` (Θ′).** The design computes Θ′ by rasterizing a silhouette
//! disk of radius `body_radius` per boid — but `body_radius` is `OcclusionParams`, owned by
//! whichever `FlockingMode` plugin uses occlusion (design/01_core.md §4), not by core. Since Θ′
//! is explicitly a report-only quantity (never the P-a acceptance anchor — that's `opacity_int`
//! off the occlusion-toolkit-filled `theta` column), `Simulation::step()` does not compute it
//! automatically; [`external_opacity`] is provided here as a working, tested toolkit function a
//! host can call directly with whatever radius it wants (matching design's own note that a
//! higher-fidelity multi-viewpoint version belongs in Python's `opacity.py` anyway).

use crate::boids::BoidColumns;
use crate::math::{Vec3, MIN_LEN};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Metrics {
    /// α = |(1/N) Σ v̂ᵢ| ∈ [0, 1]
    pub polarisation: f64,
    /// Largest eigenvalue of the nematic order tensor — axis alignment without direction.
    pub nematic_order: f64,
    /// Θ̄ — mean per-boid internal opacity, from the occlusion toolkit's `theta` column.
    pub opacity_int: f64,
    /// Θ′ — external silhouette opacity; left `0.0` unless a caller populates it via
    /// [`external_opacity`] (see module doc).
    pub opacity_ext: f64,
    /// R_max = maxᵢ<ⱼ |rᵢ − rⱼ| (exact O(N²), fine at slice scale N≤2600).
    pub r_max: f64,
    /// σ_r = mean distance to the centroid.
    pub dispersion: f64,
    /// Mean nearest-neighbour distance (exact O(N²) at slice scale).
    pub mean_nn: f64,
    pub mean_speed: f64,
    /// L, self-normalised by (cruise_speed · gyration radius) — large for a milling vortex,
    /// ≈0 for a straight stream.
    pub angular_momentum: f64,
    pub count: u32,
    pub step: u64,
}

impl Metrics {
    /// `cruise_speed` (v0) is the `angular_momentum` normaliser (design/03 §1.1); an empty
    /// (zero active boid) column returns all-zero metrics rather than dividing by zero.
    pub fn collect(boids: &BoidColumns, step: u64, cruise_speed: f64) -> Metrics {
        let active: Vec<u32> = boids.iter_active().collect();
        let n = active.len();
        if n == 0 {
            return Metrics {
                step,
                ..Metrics::default()
            };
        }
        let inv = 1.0 / n as f64;

        let mut vhat_sum = Vec3::ZERO;
        let mut com = Vec3::ZERO;
        let mut speed_sum = 0.0;
        let mut theta_sum = 0.0;
        for &i in &active {
            let v = boids.vel[i as usize];
            let s = v.len();
            if s > MIN_LEN {
                vhat_sum += v / s;
            }
            com += boids.pos[i as usize];
            speed_sum += s;
            theta_sum += boids.theta[i as usize];
        }
        com *= inv;
        let polarisation = vhat_sum.len() * inv;
        let mean_speed = speed_sum * inv;
        let opacity_int = theta_sum * inv;

        let mut disp_sum = 0.0;
        let mut gyr_sq_sum = 0.0;
        let mut ang = Vec3::ZERO;
        let mut q = SymMat3::ZERO;
        for &i in &active {
            let r = boids.pos[i as usize] - com;
            disp_sum += r.len();
            gyr_sq_sum += r.len_sq();
            ang += r.cross(boids.vel[i as usize]);
            let v = boids.vel[i as usize];
            let s = v.len();
            if s > MIN_LEN {
                let vh = v / s;
                q.add_outer(vh);
            }
        }
        let dispersion = disp_sum * inv;
        let r_g = (gyr_sq_sum * inv).sqrt();
        let angular_momentum = if r_g > MIN_LEN && cruise_speed > MIN_LEN {
            ang.len() * inv / (cruise_speed * r_g)
        } else {
            0.0
        };

        let q_traceless = q.scale(inv).sub_diag(1.0 / 3.0).scale(1.5);
        let nematic_order = q_traceless.largest_eigenvalue();

        let r_max = r_max_exact(boids, &active);
        let mean_nn = mean_nearest_neighbour_exact(boids, &active);

        Metrics {
            polarisation,
            nematic_order,
            opacity_int,
            opacity_ext: 0.0,
            r_max,
            dispersion,
            mean_nn,
            mean_speed,
            angular_momentum,
            count: n as u32,
            step,
        }
    }
}

fn r_max_exact(boids: &BoidColumns, active: &[u32]) -> f64 {
    let mut max_d = 0.0_f64;
    for a in 0..active.len() {
        for b in (a + 1)..active.len() {
            let d = (boids.pos[active[a] as usize] - boids.pos[active[b] as usize]).len();
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

fn mean_nearest_neighbour_exact(boids: &BoidColumns, active: &[u32]) -> f64 {
    if active.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for &i in active {
        let pi = boids.pos[i as usize];
        let mut min_d = f64::INFINITY;
        for &j in active {
            if i == j {
                continue;
            }
            let d = (boids.pos[j as usize] - pi).len();
            if d < min_d {
                min_d = d;
            }
        }
        sum += min_d;
    }
    sum / active.len() as f64
}

/// Rasterizes the union of every active boid's silhouette disk (radius `body_radius`) onto a
/// plane ⟂ `view_axis`, over a `resolution`×`resolution` grid spanning the projected bounding
/// box (padded by `body_radius`) — Θ′ = covered_cells / total_cells (design/03 §1.1). O(N +
/// resolution²); intended for occasional/on-demand use, not every step (see module doc).
pub fn external_opacity(
    boids: &BoidColumns,
    body_radius: f64,
    view_axis: Vec3,
    resolution: u32,
) -> f64 {
    let active: Vec<u32> = boids.iter_active().collect();
    if active.is_empty() || resolution == 0 {
        return 0.0;
    }
    let axis = view_axis.normalized();
    let axis = if axis == Vec3::ZERO {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        axis
    };
    // Any unit vector not parallel to `axis` gives a stable basis via two cross products.
    let seed = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(seed).normalized();
    let v = axis.cross(u);

    let projected: Vec<(f64, f64)> = active
        .iter()
        .map(|&i| {
            let p = boids.pos[i as usize];
            (p.dot(u), p.dot(v))
        })
        .collect();

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in &projected {
        min_x = min_x.min(x - body_radius);
        max_x = max_x.max(x + body_radius);
        min_y = min_y.min(y - body_radius);
        max_y = max_y.max(y + body_radius);
    }
    let (span_x, span_y) = (max_x - min_x, max_y - min_y);
    if span_x <= MIN_LEN || span_y <= MIN_LEN {
        return if projected.len() == 1 { 1.0 } else { 0.0 };
    }

    let res = resolution as usize;
    let mut covered = vec![false; res * res];
    let cell_w = span_x / res as f64;
    let cell_h = span_y / res as f64;
    for &(cx, cy) in &projected {
        let r_cells_x = (body_radius / cell_w).ceil() as i64 + 1;
        let r_cells_y = (body_radius / cell_h).ceil() as i64 + 1;
        let ci = ((cx - min_x) / cell_w) as i64;
        let cj = ((cy - min_y) / cell_h) as i64;
        for di in -r_cells_x..=r_cells_x {
            for dj in -r_cells_y..=r_cells_y {
                let i = ci + di;
                let j = cj + dj;
                if i < 0 || j < 0 || i >= res as i64 || j >= res as i64 {
                    continue;
                }
                let cell_center_x = min_x + (i as f64 + 0.5) * cell_w;
                let cell_center_y = min_y + (j as f64 + 0.5) * cell_h;
                let d = ((cell_center_x - cx).powi(2) + (cell_center_y - cy).powi(2)).sqrt();
                if d <= body_radius {
                    covered[i as usize * res + j as usize] = true;
                }
            }
        }
    }
    covered.iter().filter(|&&c| c).count() as f64 / (res * res) as f64
}

/// A symmetric 3×3 matrix, stored by its 6 independent entries. Private to this module — used
/// only for the nematic order tensor's largest eigenvalue.
#[derive(Clone, Copy)]
struct SymMat3 {
    xx: f64,
    yy: f64,
    zz: f64,
    xy: f64,
    xz: f64,
    yz: f64,
}

impl SymMat3 {
    const ZERO: SymMat3 = SymMat3 {
        xx: 0.0,
        yy: 0.0,
        zz: 0.0,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };

    /// Adds the outer product `v vᵀ` (v assumed unit, but works for any v).
    fn add_outer(&mut self, v: Vec3) {
        self.xx += v.x * v.x;
        self.yy += v.y * v.y;
        self.zz += v.z * v.z;
        self.xy += v.x * v.y;
        self.xz += v.x * v.z;
        self.yz += v.y * v.z;
    }

    fn scale(self, s: f64) -> SymMat3 {
        SymMat3 {
            xx: self.xx * s,
            yy: self.yy * s,
            zz: self.zz * s,
            xy: self.xy * s,
            xz: self.xz * s,
            yz: self.yz * s,
        }
    }

    /// Subtracts `s` from each diagonal entry (e.g. `s = trace/3` to make traceless).
    fn sub_diag(self, s: f64) -> SymMat3 {
        SymMat3 {
            xx: self.xx - s,
            yy: self.yy - s,
            zz: self.zz - s,
            ..self
        }
    }

    fn trace(&self) -> f64 {
        self.xx + self.yy + self.zz
    }

    fn det(&self) -> f64 {
        self.xx * (self.yy * self.zz - self.yz * self.yz)
            - self.xy * (self.xy * self.zz - self.yz * self.xz)
            + self.xz * (self.xy * self.yz - self.yy * self.xz)
    }

    /// Closed-form largest eigenvalue of a real symmetric 3×3 matrix (standard trigonometric
    /// solution — see e.g. Smith 1961 / the "Eigenvalue algorithm" article on symmetric 3×3
    /// matrices). Degenerates cleanly to the max diagonal entry when the matrix is already
    /// diagonal (off-diagonal norm ≈ 0).
    fn largest_eigenvalue(&self) -> f64 {
        let p1 = self.xy * self.xy + self.xz * self.xz + self.yz * self.yz;
        if p1 <= 1e-18 {
            return self.xx.max(self.yy).max(self.zz);
        }
        let q = self.trace() / 3.0;
        let b = self.sub_diag(q);
        let p2 = b.xx * b.xx + b.yy * b.yy + b.zz * b.zz + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let r = (b.scale(1.0 / p).det() / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;
        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + std::f64::consts::TAU / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3;
        eig1.max(eig2).max(eig3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boids::Species;

    fn boids_from(pos_vel: &[(Vec3, Vec3)]) -> BoidColumns {
        let mut b = BoidColumns::with_capacity(pos_vel.len() as u32);
        for &(p, v) in pos_vel {
            b.add(p, v, Species::Prey, 0);
        }
        b
    }

    #[test]
    fn empty_flock_is_all_zero() {
        let b = BoidColumns::with_capacity(0);
        let m = Metrics::collect(&b, 5, 1.0);
        assert_eq!(m.count, 0);
        assert_eq!(m.step, 5);
        assert_eq!(m.polarisation, 0.0);
    }

    #[test]
    fn perfectly_aligned_flock_has_polarisation_one() {
        let b = boids_from(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        assert!((m.polarisation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn anti_aligned_pair_has_zero_polarisation_and_unit_nematic_order() {
        let b = boids_from(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        assert!(
            m.polarisation < 1e-9,
            "α should be ~0, got {}",
            m.polarisation
        );
        assert!(
            (m.nematic_order - 1.0).abs() < 1e-9,
            "S should be ~1, got {}",
            m.nematic_order
        );
    }

    #[test]
    fn symmetric_translating_pair_has_zero_angular_momentum() {
        let b = boids_from(&[
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        assert!(
            m.angular_momentum.abs() < 1e-9,
            "got {}",
            m.angular_momentum
        );
    }

    #[test]
    fn orbiting_pair_has_nonzero_angular_momentum() {
        let b = boids_from(&[
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        assert!(m.angular_momentum > 0.5, "got {}", m.angular_momentum);
    }

    #[test]
    fn r_max_and_mean_nn_on_a_known_triangle() {
        // 3 boids on a line at 0, 1, 3 -> pairwise distances 1, 2, 3; nearest-neighbour
        // distances: 1 (for boid@0, nearest is @1), 1 (for @1, nearest is @0), 2 (for @3,
        // nearest is @1) -> mean_nn = (1+1+2)/3 = 4/3.
        let b = boids_from(&[
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        assert!((m.r_max - 3.0).abs() < 1e-9);
        assert!((m.mean_nn - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn dispersion_and_mean_speed_on_a_known_configuration() {
        let b = boids_from(&[
            (Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0)), // speed 5
            (Vec3::new(-2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)), // speed 0
        ]);
        let m = Metrics::collect(&b, 0, 1.0);
        // centroid = (0,0,0); each boid is distance 2 away -> dispersion = 2
        assert!((m.dispersion - 2.0).abs() < 1e-9);
        assert!((m.mean_speed - 2.5).abs() < 1e-9);
    }

    #[test]
    fn external_opacity_of_a_single_boid_matches_circle_in_square_ratio() {
        // A single boid's projected disk (radius r) sits centred in its own 2r×2r bounding
        // box: covered/total = area(circle)/area(square) = πr²/(2r)² = π/4.
        let b = boids_from(&[(Vec3::ZERO, Vec3::ZERO)]);
        let theta_ext = external_opacity(&b, 1.0, Vec3::new(0.0, 0.0, 1.0), 64);
        let expected = std::f64::consts::PI / 4.0;
        assert!(
            (theta_ext - expected).abs() < 0.02,
            "expected ~{expected:.4} (circle-in-square), got {theta_ext:.4}"
        );
    }

    #[test]
    fn external_opacity_increases_as_boids_spread_out_less() {
        let spread_out = boids_from(&[
            (Vec3::new(-50.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO),
        ]);
        let clustered = boids_from(&[
            (Vec3::new(-0.5, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO),
        ]);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let theta_spread = external_opacity(&spread_out, 1.0, axis, 32);
        let theta_clustered = external_opacity(&clustered, 1.0, axis, 32);
        assert!(
            theta_clustered > theta_spread,
            "clustered {theta_clustered} should exceed spread {theta_spread}"
        );
    }

    #[test]
    fn external_opacity_handles_empty_flock_without_panicking() {
        let b = BoidColumns::with_capacity(0);
        assert_eq!(external_opacity(&b, 1.0, Vec3::new(0.0, 0.0, 1.0), 16), 0.0);
    }
}
