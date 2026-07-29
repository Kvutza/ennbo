//! Experimental ENNX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing entry points in the crate root.

pub use crate::optimizer::{
    MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta, Optimizer, RegionBatch,
    RegionCandidate, SharingPolicy, Telemetry,
};
pub use crate::optimizer_factory::create_optimizer_enn_multi_tr;
pub use crate::trials::{
    Ask as WeightAsk, BpannHistory, Center as WeightCenter, Leaf as WeightLeaf,
    Search as WeightSearch, Trial as WeightTrial,
};
pub use crate::weights::{ComputeBackend, WeightBlock, WeightSelectConfig, WeightSelectResult};
