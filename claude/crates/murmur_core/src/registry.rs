//! The static plugin registry (design/02_plugins.md §1). A plugin is registered **by name** via
//! an **explicit function call** (`pub fn register(r: &mut Registry)` in the plugin's own
//! crate) — not `inventory`/link-time auto-collection (decision D17). Dispatch is dynamic
//! (`Box<dyn Trait>`); there is no runtime `.so` loading. An empty `Registry` is a valid,
//! legal value — nothing is registered until explicitly asked for.

use std::collections::HashMap;

use crate::domain::Domain;
use crate::error::ConfigError;
use crate::init::{Initializer, NoiseSource};
use crate::modes::{FlockingMode, SteeringModifier};
use crate::neighbor::NeighborSelection;
use crate::spatial_index::SpatialIndex;
use crate::speed_model::SpeedModel;
use crate::step_hook::StepHook;

/// Opaque, plugin-specific configuration passed to a factory function. The FFI/Python boundary
/// is a serialization boundary anyway (design/01_core.md §4.2's `SetParam` dotted-path note),
/// so a simple string-keyed f64 map is enough for a plugin factory to pull its own typed fields
/// out of; richer typed params live on the constructed plugin itself, not here.
#[derive(Debug, Clone, Default)]
pub struct PluginParams(HashMap<String, f64>);

impl PluginParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, key: impl Into<String>, value: f64) -> Self {
        self.0.insert(key.into(), value);
        self
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.0.get(key).copied()
    }

    pub fn get_or(&self, key: &str, default: f64) -> f64 {
        self.0.get(key).copied().unwrap_or(default)
    }
}

/// A non-fatal, construction-time self-correction a plugin made (or flagged) while validating
/// itself against `CoreParams`/its co-composed sockets (design/01_core.md §4.1). `plugin` is
/// the same name a `Registry` factory is keyed under (`self.name()`); nothing here is fatal —
/// a hard-invalid combination is still a `ConfigError` from the plugin's own `PluginParams`
/// resolution (e.g. `PearceParamsBuilder::build()` already rejects `phi_p + phi_a > 1.0`),
/// unchanged by this addition.
#[derive(Debug, Clone, PartialEq)]
pub struct Warning {
    pub plugin: &'static str,
    pub message: String,
}

type ModeFactory = fn(&PluginParams) -> Box<dyn FlockingMode>;
type ModifierFactory = fn(&PluginParams) -> Box<dyn SteeringModifier>;
type DomainFactory = fn(&PluginParams) -> Box<dyn Domain>;
type SpatialIndexFactory = fn(&PluginParams) -> Box<dyn SpatialIndex>;
type NeighborSelectionFactory = fn(&PluginParams) -> Box<dyn NeighborSelection>;
type SpeedModelFactory = fn(&PluginParams) -> Box<dyn SpeedModel>;
type InitFactory = fn(&PluginParams) -> Box<dyn Initializer>;
type NoiseFactory = fn(&PluginParams) -> Box<dyn NoiseSource>;
type StepHookFactory = fn(&PluginParams) -> Box<dyn StepHook>;

#[derive(Default)]
pub struct Registry {
    modes: HashMap<&'static str, ModeFactory>,
    modifiers: HashMap<&'static str, ModifierFactory>,
    domains: HashMap<&'static str, DomainFactory>,
    spatial_indices: HashMap<&'static str, SpatialIndexFactory>,
    neighbor_selectors: HashMap<&'static str, NeighborSelectionFactory>,
    speed_models: HashMap<&'static str, SpeedModelFactory>,
    inits: HashMap<&'static str, InitFactory>,
    noises: HashMap<&'static str, NoiseFactory>,
    step_hooks: HashMap<&'static str, StepHookFactory>,
}

fn unknown(socket: &'static str, name: &str) -> ConfigError {
    ConfigError::UnknownPlugin {
        socket,
        name: name.to_string(),
    }
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_mode(&mut self, name: &'static str, f: ModeFactory) {
        self.modes.insert(name, f);
    }
    pub fn resolve_mode(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn FlockingMode>, ConfigError> {
        self.modes
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("FlockingMode", name))
    }

    pub fn register_modifier(&mut self, name: &'static str, f: ModifierFactory) {
        self.modifiers.insert(name, f);
    }
    pub fn resolve_modifier(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn SteeringModifier>, ConfigError> {
        self.modifiers
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("SteeringModifier", name))
    }

    pub fn register_domain(&mut self, name: &'static str, f: DomainFactory) {
        self.domains.insert(name, f);
    }
    pub fn resolve_domain(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn Domain>, ConfigError> {
        self.domains
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("Domain", name))
    }

    pub fn register_spatial_index(&mut self, name: &'static str, f: SpatialIndexFactory) {
        self.spatial_indices.insert(name, f);
    }
    pub fn resolve_spatial_index(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn SpatialIndex>, ConfigError> {
        self.spatial_indices
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("SpatialIndex", name))
    }

    pub fn register_neighbor_selection(&mut self, name: &'static str, f: NeighborSelectionFactory) {
        self.neighbor_selectors.insert(name, f);
    }
    pub fn resolve_neighbor_selection(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn NeighborSelection>, ConfigError> {
        self.neighbor_selectors
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("NeighborSelection", name))
    }

    pub fn register_speed_model(&mut self, name: &'static str, f: SpeedModelFactory) {
        self.speed_models.insert(name, f);
    }
    pub fn resolve_speed_model(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn SpeedModel>, ConfigError> {
        self.speed_models
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("SpeedModel", name))
    }

    pub fn register_init(&mut self, name: &'static str, f: InitFactory) {
        self.inits.insert(name, f);
    }
    pub fn resolve_init(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn Initializer>, ConfigError> {
        self.inits
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("Initializer", name))
    }

    pub fn register_noise(&mut self, name: &'static str, f: NoiseFactory) {
        self.noises.insert(name, f);
    }
    pub fn resolve_noise(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn NoiseSource>, ConfigError> {
        self.noises
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("NoiseSource", name))
    }

    pub fn register_step_hook(&mut self, name: &'static str, f: StepHookFactory) {
        self.step_hooks.insert(name, f);
    }
    pub fn resolve_step_hook(
        &self,
        name: &str,
        p: &PluginParams,
    ) -> Result<Box<dyn StepHook>, ConfigError> {
        self.step_hooks
            .get(name)
            .map(|f| f(p))
            .ok_or_else(|| unknown("StepHook", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::modes::{BoidCtx, SteerIntent};
    use crate::occlusion::OcclusionScratch;
    use crate::rng::Rng;

    struct DummyMode;
    impl FlockingMode for DummyMode {
        fn desired(
            &self,
            ctx: BoidCtx<'_>,
            _s: &mut OcclusionScratch,
            _r: &mut Rng,
        ) -> SteerIntent {
            SteerIntent {
                desired_v: ctx.vel,
                extra_force: Vec3::ZERO,
                theta: 0.0,
            }
        }
        fn name(&self) -> &'static str {
            "dummy_mode"
        }
    }

    fn make_dummy_mode(_p: &PluginParams) -> Box<dyn FlockingMode> {
        Box::new(DummyMode)
    }

    #[test]
    fn unregistered_name_is_unknown_plugin_error() {
        let reg = Registry::new();
        match reg.resolve_mode("pearce", &PluginParams::new()) {
            Err(e) => assert_eq!(
                e,
                ConfigError::UnknownPlugin {
                    socket: "FlockingMode",
                    name: "pearce".into()
                }
            ),
            Ok(_) => panic!("expected UnknownPlugin"),
        }
    }

    #[test]
    fn empty_registry_resolves_nothing() {
        let reg = Registry::new();
        assert!(reg.resolve_domain("open", &PluginParams::new()).is_err());
        assert!(reg
            .resolve_spatial_index("hash_grid", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_neighbor_selection("radius_gather", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_modifier("instant_response", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_speed_model("band", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_init("sphere_volume", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_noise("uniform_sphere", &PluginParams::new())
            .is_err());
        assert!(reg
            .resolve_step_hook("predator", &PluginParams::new())
            .is_err());
    }

    #[test]
    fn registered_name_resolves_to_a_working_plugin() {
        let mut reg = Registry::new();
        reg.register_mode("dummy_mode", make_dummy_mode);
        let mode = reg
            .resolve_mode("dummy_mode", &PluginParams::new())
            .unwrap();
        assert_eq!(mode.name(), "dummy_mode");
    }

    #[test]
    fn plugin_params_roundtrip() {
        let p = PluginParams::new().with("phi_p", 0.03).with("phi_a", 0.80);
        assert_eq!(p.get("phi_p"), Some(0.03));
        assert_eq!(p.get("missing"), None);
        assert_eq!(p.get_or("missing", 1.0), 1.0);
    }
}
