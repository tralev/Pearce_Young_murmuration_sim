//! `OpenSpace` — the first `Domain` plugin (design/01_core.md §3). Unbounded ℝ³: `delta = b −
//! a`, `apply` is a no-op. Selected as the slice's default because the science (marginal
//! opacity, density scaling, R_max non-fragmentation) requires a free-flying, self-sizing
//! flock — not because it is structurally privileged over any other `Domain` plugin
//! (design/00_overview.md §2's governing rule).

use murmur_core::{Domain, PluginParams, Registry, Vec3};

pub struct OpenSpace;

impl Domain for OpenSpace {
    fn delta(&self, a: Vec3, b: Vec3) -> Vec3 {
        b - a
    }
    fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
    fn name(&self) -> &'static str {
        "open"
    }
}

/// Registers `OpenSpace` under the name `"open"` (design/02_plugins.md §1 — explicit
/// registration, not `inventory`).
pub fn register(r: &mut Registry) {
    r.register_domain("open", |_p: &PluginParams| Box::new(OpenSpace));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforms_to_domain_contract() {
        murmur_conformance::domain(&OpenSpace);
    }

    #[test]
    fn delta_is_b_minus_a() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(OpenSpace.delta(a, b), b - a);
    }

    #[test]
    fn apply_is_a_noop() {
        let mut pos = Vec3::new(100.0, -50.0, 3.0);
        let mut vel = Vec3::new(1.0, 2.0, 3.0);
        let (pos0, vel0) = (pos, vel);
        OpenSpace.apply(&mut pos, &mut vel, 1.0);
        assert_eq!(pos, pos0);
        assert_eq!(vel, vel0);
    }

    #[test]
    fn registered_name_resolves_via_the_registry() {
        let mut reg = Registry::new();
        register(&mut reg);
        let d = reg.resolve_domain("open", &PluginParams::new()).unwrap();
        assert_eq!(d.name(), "open");
    }

    /// Proves the `Domain` seam is real, not just `OpenSpace`'s correctness: a trivial stub
    /// implementing the same trait registers and resolves under a different name, alongside
    /// the real plugin (roadmap.md Phase 3 exit gate).
    #[test]
    fn a_different_domain_plugin_swaps_in_via_the_same_seam() {
        struct StubDomain;
        impl Domain for StubDomain {
            fn delta(&self, _a: Vec3, _b: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn apply(&self, _pos: &mut Vec3, _vel: &mut Vec3, _dt: f64) {}
            fn name(&self) -> &'static str {
                "stub_domain"
            }
        }
        let mut reg = Registry::new();
        register(&mut reg);
        reg.register_domain("stub_domain", |_p| Box::new(StubDomain));

        let open = reg.resolve_domain("open", &PluginParams::new()).unwrap();
        let stub = reg
            .resolve_domain("stub_domain", &PluginParams::new())
            .unwrap();
        assert_eq!(open.name(), "open");
        assert_eq!(stub.name(), "stub_domain");
    }
}
