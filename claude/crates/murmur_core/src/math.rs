//! `Vec3` — f64, `#[repr(C)]` so numpy can view arrays of these zero-copy (design/01_core.md §1).

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Shared floor for normalisation/division guards, used throughout the core and its plugins.
pub const MIN_LEN: f64 = 1e-9;
pub const MIN_LEN2: f64 = MIN_LEN * MIN_LEN;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn len_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn len(self) -> f64 {
        self.len_sq().sqrt()
    }

    /// Returns `Vec3::ZERO` for a zero (or near-zero) vector — never NaN.
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let l = self.len();
        if l > MIN_LEN {
            self * (1.0 / l)
        } else {
            Vec3::ZERO
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    #[inline]
    pub fn lerp(self, rhs: Vec3, t: f64) -> Vec3 {
        self + (rhs - self) * t
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, rhs: f64) -> Vec3 {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs;
    }
}

impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

/// Clamps `v`'s length to `max`, preserving direction; leaves `v` untouched if already within
/// bound or too small to have a meaningful direction (design/01_core.md §12.1). Shared by
/// `InstantResponse` and the Band `SpeedModel`.
#[inline]
pub fn clamp_len(v: Vec3, max: f64) -> Vec3 {
    let l = v.len();
    if l > max && l > MIN_LEN {
        v * (max / l)
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPS * (1.0 + a.abs().max(b.abs()))
    }

    fn vec_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    /// Tiny deterministic PRNG (xorshift64*) so property tests need no external crate and are
    /// still reproducible across runs.
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        /// Uniform f64 in [-scale, scale].
        fn next_f64(&mut self, scale: f64) -> f64 {
            let bits = self.next_u64();
            let unit = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64); // [0, 1)
            (unit * 2.0 - 1.0) * scale
        }
        fn next_vec3(&mut self, scale: f64) -> Vec3 {
            Vec3::new(
                self.next_f64(scale),
                self.next_f64(scale),
                self.next_f64(scale),
            )
        }
    }

    #[test]
    fn normalized_zero_is_zero_never_nan() {
        let v = Vec3::ZERO.normalized();
        assert_eq!(v, Vec3::ZERO);
        assert!(v.is_finite());

        let tiny = Vec3::new(1e-300, 0.0, 0.0).normalized();
        assert_eq!(tiny, Vec3::ZERO);
        assert!(tiny.is_finite());
    }

    #[test]
    fn normalized_nonzero_has_unit_length() {
        let mut rng = Xorshift64(0xC0FFEE);
        for _ in 0..1000 {
            let v = rng.next_vec3(1e6);
            if v.len_sq() <= MIN_LEN2 {
                continue;
            }
            let n = v.normalized();
            assert!(approx_eq(n.len(), 1.0), "len={} for v={:?}", n.len(), v);
        }
    }

    #[test]
    fn dot_is_commutative_and_bilinear() {
        let mut rng = Xorshift64(1);
        for _ in 0..1000 {
            let a = rng.next_vec3(100.0);
            let b = rng.next_vec3(100.0);
            let c = rng.next_vec3(100.0);
            assert!(approx_eq(a.dot(b), b.dot(a)));
            // distributivity: a . (b + c) == a.b + a.c
            assert!(approx_eq(a.dot(b + c), a.dot(b) + a.dot(c)));
            // len_sq is self-dot
            assert!(approx_eq(a.len_sq(), a.dot(a)));
        }
    }

    #[test]
    fn cross_is_anticommutative_and_orthogonal() {
        let mut rng = Xorshift64(2);
        for _ in 0..1000 {
            let a = rng.next_vec3(100.0);
            let b = rng.next_vec3(100.0);
            let cross = a.cross(b);
            assert!(vec_approx_eq(cross, -(b.cross(a))));
            // a x a == 0
            assert!(vec_approx_eq(a.cross(a), Vec3::ZERO));
            // a x b is perpendicular to both a and b
            assert!(approx_eq(a.dot(cross) / (1.0 + a.len() * cross.len()), 0.0));
            assert!(approx_eq(b.dot(cross) / (1.0 + b.len() * cross.len()), 0.0));
        }
    }

    #[test]
    fn lerp_identities() {
        let mut rng = Xorshift64(3);
        for _ in 0..1000 {
            let a = rng.next_vec3(100.0);
            let b = rng.next_vec3(100.0);
            assert!(vec_approx_eq(a.lerp(b, 0.0), a));
            assert!(vec_approx_eq(a.lerp(b, 1.0), b));
            assert!(vec_approx_eq(a.lerp(a, rng.next_f64(1.0)), a));
            let t = rng.next_f64(1.0);
            let mid = a.lerp(b, t);
            // lerp is affine: matches the direct formula a + (b-a)*t
            assert!(vec_approx_eq(mid, a + (b - a) * t));
        }
    }

    #[test]
    fn is_finite_rejects_nan_and_inf() {
        assert!(Vec3::ZERO.is_finite());
        assert!(!Vec3::new(f64::NAN, 0.0, 0.0).is_finite());
        assert!(!Vec3::new(0.0, f64::INFINITY, 0.0).is_finite());
        assert!(!Vec3::new(0.0, 0.0, f64::NEG_INFINITY).is_finite());
    }

    #[test]
    fn arithmetic_ops_agree_with_componentwise_definition() {
        let mut rng = Xorshift64(4);
        for _ in 0..500 {
            let a = rng.next_vec3(50.0);
            let b = rng.next_vec3(50.0);
            let s = rng.next_f64(10.0);

            assert!(vec_approx_eq(
                a + b,
                Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
            ));
            assert!(vec_approx_eq(
                a - b,
                Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
            ));
            assert!(vec_approx_eq(-a, Vec3::new(-a.x, -a.y, -a.z)));
            assert!(vec_approx_eq(a * s, Vec3::new(a.x * s, a.y * s, a.z * s)));
            if s.abs() > MIN_LEN {
                assert!(vec_approx_eq(a / s, Vec3::new(a.x / s, a.y / s, a.z / s)));
            }

            let mut c = a;
            c += b;
            assert!(vec_approx_eq(c, a + b));
            let mut d = a;
            d -= b;
            assert!(vec_approx_eq(d, a - b));
            let mut e = a;
            e *= s;
            assert!(vec_approx_eq(e, a * s));
        }
    }

    #[test]
    fn clamp_len_bounds_length_and_preserves_direction() {
        let v = Vec3::new(3.0, 4.0, 0.0); // len = 5
        let clamped = clamp_len(v, 2.0);
        assert!(approx_eq(clamped.len(), 2.0));
        assert!(vec_approx_eq(clamped.normalized(), v.normalized()));

        let unclamped = clamp_len(v, 10.0);
        assert_eq!(unclamped, v);

        assert_eq!(clamp_len(Vec3::ZERO, 5.0), Vec3::ZERO);
    }

    #[test]
    fn repr_c_layout_matches_three_f64() {
        // Zero-copy numpy views (design/01_core.md §1) depend on this layout.
        assert_eq!(std::mem::size_of::<Vec3>(), 3 * std::mem::size_of::<f64>());
        assert_eq!(std::mem::align_of::<Vec3>(), std::mem::align_of::<f64>());
    }
}
