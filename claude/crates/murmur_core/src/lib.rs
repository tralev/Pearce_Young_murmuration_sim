//! `murmur_core` — infrastructure only.
//!
//! Storage, math, RNG, trait definitions, the registry, the occlusion/kernel toolkit, and the
//! generic `step()` pipeline. No concrete flocking algorithm or strategy lives here — every
//! implementation (`PearceProjection`, `OpenSpace`, `HashGrid`, ...) is a plugin in its own
//! crate. See `design/00_overview.md` §2, "Core vs. plugin — the governing rule".

pub mod batch;
pub mod boids;
pub mod domain;
pub mod error;
pub mod h2;
pub mod init;
pub mod kernels;
pub mod math;
pub mod metrics;
pub mod modes;
pub mod neighbor;
pub mod occlusion;
pub mod params;
pub mod pipeline;
pub mod registry;
pub mod rng;
pub mod spatial_index;
pub mod speed_model;
pub mod step_hook;

pub use batch::{
    BoidSnapshot, Checkpoint, CheckpointBuffer, Command, CommandError, InterpolationHint,
    PredatorSnapshot, SessionHeader,
};
pub use boids::{BoidColumns, Species};
pub use domain::Domain;
pub use error::ConfigError;
pub use h2::{h2_at_m, m_star, H2Result};
pub use init::{Initializer, NoiseSource};
pub use kernels::{AlignmentKernel, CohesionKernel, SeparationKernel};
pub use math::{clamp_len, Vec3, MIN_LEN, MIN_LEN2};
pub use metrics::{external_opacity, Metrics};
pub use modes::{BoidCtx, FlockingMode, SteerIntent, SteeringModifier};
pub use neighbor::{Neighbor, NeighborSelection};
pub use occlusion::{occlude, OcclusionScratch, VisibleView};
pub use params::{CoreParams, OcclusionParams, DT_MAX};
pub use pipeline::{Composition, SimConfig, Simulation};
pub use registry::{PluginParams, Registry};
pub use rng::{sample_unit_sphere, uniform01, Rng};
pub use spatial_index::SpatialIndex;
pub use speed_model::SpeedModel;
pub use step_hook::{
    BoidCheckpointFields, CsgOp, EnvironmentSnapshot, ObstacleNodeSnapshot,
    ObstaclePrimitiveSnapshot, RippleSnapshot, RippleTrainSnapshot, SceneCheckpointFields, SimView,
    StepHook, WanderSnapshot,
};
