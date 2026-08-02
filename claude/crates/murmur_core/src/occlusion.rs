//! Occlusion pass — core toolkit, not an algorithm (design/01_core.md §7). Any `FlockingMode`
//! plugin may call `occlude()`, not only `PearceProjection`. Algorithm verbatim from
//! `sci/sim_new.md` §21 (f64).
//!
//! **Implementation note on `sigma`.** `design/01_core.md` §12.2's illustrative `occlude()`
//! reads `p.sigma` off the old undifferentiated `SimParams`; per this design's real params
//! split (§4), `sigma` ("how many nearest visible neighbours to average for alignment") is
//! `PearceParams`' own field, not toolkit-level `OcclusionParams` — a non-Pearce `FlockingMode`
//! calling `occlude()` may want a different count, or none at all. `occlude()` therefore takes
//! `sigma` as an explicit parameter, separate from `&OcclusionParams`, so the toolkit function
//! depends on exactly what's actually shared (§4's own note that illustrative snippets get
//! finalized "when Phase 1–2 write the real code").
//!
//! **Implementation note on `align`.** `VisibleView::align`'s doc contract (§7) promises the
//! *mean* velocity of the σ nearest visible neighbours; §12.2's reference code sums but never
//! divides (harmless there, since `PearceProjection` immediately normalizes it to a direction
//! — dividing by a positive scalar doesn't change that). This implementation divides by the
//! actual count, honouring the documented contract so `align` stays correct for any future
//! consumer that wants the vector's magnitude too, not only Pearce's direction-only use.

use crate::math::{Vec3, MIN_LEN, MIN_LEN2};
use crate::neighbor::Neighbor;
use crate::params::OcclusionParams;

pub struct VisibleView {
    /// Σ sinα·d̂ / Σ sinα — magnitude preserved (the density-regulation signal); `Vec3::ZERO`
    /// when the visible set is empty.
    pub delta_hat: Vec3,
    /// Internal opacity Θ = 1 − Π(1 − Ωⱼ/4π), ∈ [0, 1].
    pub theta: f64,
    /// Mean velocity of the σ nearest visible neighbours; `Vec3::ZERO` if none.
    pub align: Vec3,
}

#[derive(Clone, Copy)]
struct Cap {
    dir: Vec3,
    vel: Vec3,
    dist: f64,
    sin_a: f64,
    cos_a: f64,
}

/// Reused per-thread scratch for the occlusion pass — allocated once, never freed mid-run;
/// buffers only grow (never shrink) as they warm up to a composition's typical candidate count.
#[derive(Default)]
pub struct OcclusionScratch {
    caps: Vec<Cap>,
    visible: Vec<Cap>,
}

/// Computes the visible view for one observer. `heading` should be a unit vector (the
/// observer's normalized velocity); `candidates` are distance-filtered but not yet sorted.
pub fn occlude(
    heading: Vec3,
    candidates: &[Neighbor],
    params: &OcclusionParams,
    sigma: u32,
    scratch: &mut OcclusionScratch,
) -> VisibleView {
    scratch.caps.clear();
    for n in candidates {
        let d = n.distance.max(MIN_LEN);
        if params.blind_cone_enabled
            && n.direction.dot(-heading) >= params.blind_cone_half_angle.cos()
        {
            continue; // inside rear blind cone
        }
        let b_eff = if params.anisotropic_enabled && n.velocity.len_sq() > MIN_LEN2 {
            let a = params.body_radius * params.anisotropy;
            let cos_psi = n.direction.dot(n.velocity.normalized()).abs();
            let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();
            ((a * sin_psi).powi(2) + (params.body_radius * cos_psi).powi(2)).sqrt()
        } else {
            params.body_radius
        };
        let alpha = (b_eff / d).min(1.0).asin();
        scratch.caps.push(Cap {
            dir: n.direction,
            vel: n.velocity,
            dist: d,
            sin_a: alpha.sin(),
            cos_a: alpha.cos(),
        });
    }

    scratch
        .caps
        .sort_unstable_by(|x, y| x.dist.total_cmp(&y.dist)); // nearest-first
    scratch.caps.truncate(params.max_candidates as usize);

    scratch.visible.clear();
    'cand: for c in &scratch.caps {
        // closest-first visibility cull: occluded if inside a nearer *visible* cap
        for v in &scratch.visible {
            if c.dir.dot(v.dir) >= v.cos_a {
                continue 'cand;
            }
        }
        scratch.visible.push(*c);
    }

    let (mut delta, mut sin_sum, mut transp) = (Vec3::ZERO, 0.0, 1.0);
    let mut align_sum = Vec3::ZERO;
    let mut align_count: u32 = 0;
    for (i, v) in scratch.visible.iter().enumerate() {
        delta += v.dir * v.sin_a; // δ̂ numerator
        sin_sum += v.sin_a;
        transp *= 1.0 - (1.0 - v.cos_a) * 0.5; // Θ = 1 − Π(1 − Ωⱼ/4π)
        if (i as u32) < sigma {
            align_sum += v.vel;
            align_count += 1;
        }
    }
    let delta_hat = if sin_sum > MIN_LEN {
        delta / sin_sum
    } else {
        Vec3::ZERO
    }; // NOT normalised
    let align = if align_count > 0 {
        align_sum / align_count as f64
    } else {
        Vec3::ZERO
    };

    VisibleView {
        delta_hat,
        theta: 1.0 - transp,
        align,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(body_radius: f64) -> OcclusionParams {
        OcclusionParams::builder()
            .body_radius(body_radius)
            .max_candidates(50)
            .blind_cone_enabled(true)
            .blind_cone_half_angle(0.524)
            .build()
            .unwrap()
    }

    fn neighbor(direction: Vec3, distance: f64, velocity: Vec3) -> Neighbor {
        Neighbor {
            index: 0,
            distance,
            direction: direction.normalized(),
            velocity,
        }
    }

    // (i) cull: a farther cap directly behind a nearer, wider cap is occluded.
    #[test]
    fn farther_candidate_on_the_same_bearing_is_culled_by_a_nearer_cap() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let near = neighbor(Vec3::new(1.0, 0.0, 0.0), 2.0, Vec3::new(1.0, 0.0, 0.0));
        let far_same_bearing = neighbor(Vec3::new(1.0, 0.0, 0.0), 10.0, Vec3::new(0.0, 1.0, 0.0));
        let p = params(1.5); // b/d for `near` = 0.75 -> a wide ~48.6deg cap
        let mut scratch = OcclusionScratch::default();

        let view = occlude(heading, &[near, far_same_bearing], &p, 2, &mut scratch);

        // If both were visible, align (mean of up to 2) would blend both velocities; since the
        // far one is culled, align must equal exactly the near one's velocity.
        assert!((view.align - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9);
    }

    // (ii) rear blind cone: a candidate directly behind the heading is dropped entirely.
    #[test]
    fn candidate_inside_the_rear_blind_cone_is_dropped() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let behind = neighbor(-heading, 3.0, Vec3::new(1.0, 0.0, 0.0));
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();

        let view = occlude(heading, &[behind], &p, 4, &mut scratch);

        assert_eq!(view.theta, 0.0, "an empty visible set has zero opacity");
        assert_eq!(view.delta_hat, Vec3::ZERO);
        assert_eq!(view.align, Vec3::ZERO);
    }

    // (iii) |δ̂| ≈ 0 isotropic (six symmetric directions cancel exactly) / ≈ 1 at the edge (every
    // candidate on the same bearing reinforces).
    #[test]
    fn delta_hat_magnitude_is_zero_isotropic_and_one_at_the_edge() {
        let heading = Vec3::new(0.0, 0.0, 1.0); // blind cone excludes -heading, so skip it
        let isotropic = [
            neighbor(Vec3::new(1.0, 0.0, 0.0), 5.0, Vec3::ZERO),
            neighbor(Vec3::new(-1.0, 0.0, 0.0), 5.0, Vec3::ZERO),
            neighbor(Vec3::new(0.0, 1.0, 0.0), 5.0, Vec3::ZERO),
            neighbor(Vec3::new(0.0, -1.0, 0.0), 5.0, Vec3::ZERO),
        ];
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();
        let view = occlude(heading, &isotropic, &p, 0, &mut scratch);
        assert!(
            view.delta_hat.len() < 1e-9,
            "expected ~0, got {:?}",
            view.delta_hat
        );

        let edge = [
            neighbor(Vec3::new(1.0, 0.0, 0.0), 5.0, Vec3::ZERO),
            neighbor(Vec3::new(1.0, 0.0, 0.0), 8.0, Vec3::ZERO),
            neighbor(Vec3::new(1.0, 0.0, 0.0), 12.0, Vec3::ZERO),
        ];
        let view = occlude(heading, &edge, &p, 0, &mut scratch);
        assert!(
            (view.delta_hat.len() - 1.0).abs() < 1e-9,
            "expected ~1, got {:?}",
            view.delta_hat
        );
    }

    // (iv) Θ increases monotonically as a single neighbour approaches.
    #[test]
    fn theta_increases_monotonically_as_a_neighbor_approaches() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();
        let distances = [20.0, 10.0, 5.0, 2.0, 1.2];
        let mut prev_theta = -1.0;
        for &d in &distances {
            let n = neighbor(Vec3::new(1.0, 0.0, 0.0), d, Vec3::ZERO);
            let view = occlude(heading, &[n], &p, 0, &mut scratch);
            assert!(
                view.theta > prev_theta,
                "theta should strictly increase as distance shrinks: at d={d}, theta={}, prev={prev_theta}",
                view.theta
            );
            prev_theta = view.theta;
        }
    }

    // (v) anisotropy: broadside presentation (velocity ⟂ bearing) occludes more than end-on
    // (velocity ∥ bearing) at the same distance.
    #[test]
    fn anisotropic_broadside_has_higher_opacity_than_end_on() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = OcclusionParams::builder()
            .body_radius(1.0)
            .anisotropy(3.0)
            .anisotropic_enabled(true)
            .max_candidates(50)
            .blind_cone_enabled(true)
            .blind_cone_half_angle(0.524)
            .build()
            .unwrap();
        let mut scratch = OcclusionScratch::default();

        let bearing = Vec3::new(1.0, 0.0, 0.0);
        let broadside = neighbor(bearing, 5.0, Vec3::new(0.0, 1.0, 0.0)); // heading ⟂ bearing
        let end_on = neighbor(bearing, 5.0, Vec3::new(1.0, 0.0, 0.0)); // heading ∥ bearing

        let view_broadside = occlude(heading, &[broadside], &p, 0, &mut scratch);
        let view_end_on = occlude(heading, &[end_on], &p, 0, &mut scratch);

        assert!(
            view_broadside.theta > view_end_on.theta,
            "broadside theta={} should exceed end-on theta={}",
            view_broadside.theta,
            view_end_on.theta
        );
    }

    // (vi) NaN-safety across empty/single/coincident/zero-velocity candidate sets.
    #[test]
    fn empty_candidate_set_is_finite_and_zero() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();
        let view = occlude(heading, &[], &p, 4, &mut scratch);
        assert_eq!(view.delta_hat, Vec3::ZERO);
        assert_eq!(view.theta, 0.0);
        assert_eq!(view.align, Vec3::ZERO);
    }

    #[test]
    fn single_candidate_is_finite() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 3.0, Vec3::new(0.0, 1.0, 0.0));
        let view = occlude(heading, &[n], &p, 4, &mut scratch);
        assert!(view.delta_hat.is_finite());
        assert!(view.theta.is_finite());
        assert!(view.align.is_finite());
    }

    #[test]
    fn coincident_candidate_does_not_produce_nan() {
        // A Neighbor at distance 0 shouldn't occur in practice (RadiusGather guards against
        // it), but occlude() defends independently via `n.distance.max(MIN_LEN)`.
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = params(1.0);
        let mut scratch = OcclusionScratch::default();
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 0.0, Vec3::new(1.0, 0.0, 0.0));
        let view = occlude(heading, &[n], &p, 4, &mut scratch);
        assert!(view.delta_hat.is_finite());
        assert!(view.theta.is_finite());
        assert!(view.align.is_finite());
    }

    #[test]
    fn zero_velocity_candidate_falls_back_to_isotropic_without_nan() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = OcclusionParams::builder()
            .body_radius(1.0)
            .anisotropy(2.0)
            .anisotropic_enabled(true)
            .max_candidates(50)
            .blind_cone_enabled(true)
            .blind_cone_half_angle(0.524)
            .build()
            .unwrap();
        let mut scratch = OcclusionScratch::default();
        // velocity = ZERO would make `.normalized()` degenerate if not guarded.
        let n = neighbor(Vec3::new(1.0, 0.0, 0.0), 3.0, Vec3::ZERO);
        let view = occlude(heading, &[n], &p, 4, &mut scratch);
        assert!(view.delta_hat.is_finite());
        assert!(view.theta.is_finite());
        assert!(view.align.is_finite());
    }

    #[test]
    fn align_is_the_mean_not_the_sum_of_the_sigma_nearest_visible_velocities() {
        let heading = Vec3::new(0.0, 0.0, 1.0);
        let p = params(0.1); // small body_radius: caps don't overlap, both stay visible
        let mut scratch = OcclusionScratch::default();
        let a = neighbor(Vec3::new(1.0, 0.0, 0.0), 5.0, Vec3::new(2.0, 0.0, 0.0));
        let b = neighbor(Vec3::new(0.0, 1.0, 0.0), 5.0, Vec3::new(0.0, 4.0, 0.0));
        let view = occlude(heading, &[a, b], &p, 2, &mut scratch);
        assert!(
            (view.align - Vec3::new(1.0, 2.0, 0.0)).len() < 1e-9,
            "got {:?}",
            view.align
        );
    }
}
