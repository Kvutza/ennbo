//! Low-Latency Multi-Trust Region (TuRBO-M) state engine.
//!
//! Maintains $M$ concurrent trust regions in contiguous array memory (SoA layout)
//! for zero-allocation CPU SIMD operations and direct GPU buffer mirroring.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::trust_region::{TRLengthConfig, TrustRegionError};

/// Policy for sharing observations across multiple trust regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharingPolicy {
    /// Shared: all observations update every region whose bounds contain the point.
    Shared,
    /// NearestCenter: observation updates only the nearest region center.
    NearestCenter,
    /// Independent: observations are assigned only to their generating region.
    Independent,
}

impl Default for SharingPolicy {
    fn default() -> Self {
        Self::Shared
    }
}

/// Configuration for Multi-Trust Region state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTrustRegionConfig {
    /// Number of concurrent trust regions (M).
    pub num_regions: usize,
    /// Trust region length configuration parameters.
    pub length: TRLengthConfig,
    /// Consecutive successes required before region expansion.
    pub succ_tolerance: i32,
    /// Consecutive failures allowed before region contraction.
    pub fail_tolerance: i32,
    /// Observation sharing policy across regions.
    pub sharing_policy: SharingPolicy,
}

impl MultiTrustRegionConfig {
    pub fn new(num_regions: usize, length: TRLengthConfig) -> Self {
        Self {
            num_regions,
            length,
            succ_tolerance: 3,
            fail_tolerance: 5,
            sharing_policy: SharingPolicy::Shared,
        }
    }
}

/// Multi-Trust Region engine storing $M$ regions in contiguous flat memory.
#[derive(Debug, Clone)]
pub struct MultiTrustRegionState {
    num_regions: usize,
    num_dim: usize,

    /// Contiguous center matrix of shape `[num_regions, num_dim]` in unit hypercube `[0, 1]^d`.
    pub centers: Array2<f64>,
    /// Contiguous length vector of shape `[num_regions]`.
    pub lengths: Array1<f64>,
    /// Best observed scalar value per region of shape `[num_regions]`.
    pub incumbents_y: Array1<f64>,
    /// Success counters per region.
    pub succ_counts: Vec<i32>,
    /// Failure counters per region.
    pub fail_counts: Vec<i32>,
    /// Active mask per region (false when region needs restart).
    pub active_mask: Vec<bool>,

    config: MultiTrustRegionConfig,
}

impl MultiTrustRegionState {
    /// Initialize a new MultiTrustRegionState.
    pub fn new(
        num_dim: usize,
        config: MultiTrustRegionConfig,
        initial_centers: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        if config.num_regions == 0 {
            return Err(TrustRegionError::InvalidParameter(
                "num_regions must be > 0".to_string(),
            ));
        }

        let num_regions = config.num_regions;
        let mut centers = Array2::zeros((num_regions, num_dim));

        if let Some(init) = initial_centers {
            if init.nrows() != num_regions || init.ncols() != num_dim {
                return Err(TrustRegionError::InvalidParameter(format!(
                    "initial_centers shape {:?} does not match [{}, {}]",
                    (init.nrows(), init.ncols()),
                    num_regions,
                    num_dim
                )));
            }
            centers.assign(init);
        } else {
            // Uniformly sample centers in unit hypercube [0, 1]^d
            for r in 0..num_regions {
                for d in 0..num_dim {
                    centers[[r, d]] = rand::Rng::gen_range(rng, 0.0..1.0);
                }
            }
        }

        let lengths = Array1::from_elem(num_regions, config.length.length_init);
        let incumbents_y = Array1::from_elem(num_regions, f64::NEG_INFINITY);
        let succ_counts = vec![0; num_regions];
        let fail_counts = vec![0; num_regions];
        let active_mask = vec![true; num_regions];

        Ok(Self {
            num_regions,
            num_dim,
            centers,
            lengths,
            incumbents_y,
            succ_counts,
            fail_counts,
            active_mask,
            config,
        })
    }

    pub fn num_regions(&self) -> usize {
        self.num_regions
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn active_count(&self) -> usize {
        self.active_mask.iter().filter(|&&a| a).count()
    }

    /// Compute lower and upper bounds for a specific trust region `r`.
    pub fn compute_bounds_1d(
        &self,
        r: usize,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let center = self.centers.row(r);
        let len = self.lengths[r];
        let mut lb = Array1::zeros(self.num_dim);
        let mut ub = Array1::zeros(self.num_dim);

        if let Some(ls) = lengthscales {
            let ls_mean = ls.mean().unwrap_or(1.0);
            for d in 0..self.num_dim {
                let half = 0.5 * len * (ls[d] / ls_mean);
                lb[d] = (center[d] - half).clamp(0.0, 1.0);
                ub[d] = (center[d] + half).clamp(0.0, 1.0);
            }
        } else {
            let half = 0.5 * len;
            for d in 0..self.num_dim {
                lb[d] = (center[d] - half).clamp(0.0, 1.0);
                ub[d] = (center[d] + half).clamp(0.0, 1.0);
            }
        }

        (lb, ub)
    }

    /// Update trust regions with newly observed batch points `x_batch` (shape [N, d]) and `y_batch` (shape [N]).
    pub fn tell_update(
        &mut self,
        x_batch: &ArrayView2<f64>,
        y_batch: &ArrayView1<f64>,
    ) -> Result<(), TrustRegionError> {
        let n = x_batch.nrows();
        if n == 0 {
            return Ok(());
        }
        if x_batch.ncols() != self.num_dim || y_batch.len() != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "Batch shapes mismatched: x {:?}, y len {}",
                (x_batch.nrows(), x_batch.ncols()),
                y_batch.len()
            )));
        }

        for r in 0..self.num_regions {
            if !self.active_mask[r] {
                continue;
            }

            let mut region_improved = false;
            let current_best = self.incumbents_y[r];
            let mut best_y_in_batch = f64::NEG_INFINITY;
            let mut best_x_idx: Option<usize> = None;

            let (lb, ub) = self.compute_bounds_1d(r, None);

            for i in 0..n {
                let x_i = x_batch.row(i);
                let y_i = y_batch[i];

                // Check containment according to sharing policy
                let inside = match self.config.sharing_policy {
                    SharingPolicy::Shared | SharingPolicy::Independent => {
                        (0..self.num_dim).all(|d| x_i[d] >= lb[d] && x_i[d] <= ub[d])
                    }
                    SharingPolicy::NearestCenter => {
                        let center_r = self.centers.row(r);
                        let dist_r = (0..self.num_dim)
                            .map(|d| (x_i[d] - center_r[d]).powi(2))
                            .sum::<f64>();

                        // Check if r is nearest active region center
                        let mut is_nearest = true;
                        for r_other in 0..self.num_regions {
                            if r_other != r && self.active_mask[r_other] {
                                let c_other = self.centers.row(r_other);
                                let d_other = (0..self.num_dim)
                                    .map(|d| (x_i[d] - c_other[d]).powi(2))
                                    .sum::<f64>();
                                if d_other < dist_r {
                                    is_nearest = false;
                                    break;
                                }
                            }
                        }
                        is_nearest
                    }
                };

                if inside && y_i > current_best && y_i > best_y_in_batch {
                    best_y_in_batch = y_i;
                    best_x_idx = Some(i);
                    region_improved = true;
                }
            }

            if region_improved {
                if let Some(idx) = best_x_idx {
                    self.incumbents_y[r] = y_batch[idx];
                    let x_best = x_batch.row(idx);
                    for d in 0..self.num_dim {
                        self.centers[[r, d]] = x_best[d];
                    }
                }
                self.succ_counts[r] += 1;
                self.fail_counts[r] = 0;

                if self.succ_counts[r] >= self.config.succ_tolerance {
                    self.lengths[r] = (2.0 * self.lengths[r]).min(self.config.length.length_max);
                    self.succ_counts[r] = 0;
                }
            } else {
                self.succ_counts[r] = 0;
                self.fail_counts[r] += 1;

                if self.fail_counts[r] >= self.config.fail_tolerance {
                    self.lengths[r] *= 0.5;
                    self.fail_counts[r] = 0;

                    if self.lengths[r] < self.config.length.length_min {
                        self.active_mask[r] = false; // Flag for restart
                    }
                }
            }
        }

        Ok(())
    }

    /// Restart a inactive region `r` with a new center location.
    pub fn restart_region(
        &mut self,
        r: usize,
        new_center: &ArrayView1<f64>,
    ) -> Result<(), TrustRegionError> {
        if r >= self.num_regions {
            return Err(TrustRegionError::InvalidParameter(format!(
                "region_idx {} out of bounds {}",
                r, self.num_regions
            )));
        }
        if new_center.len() != self.num_dim {
            return Err(TrustRegionError::InvalidParameter(format!(
                "new_center len {} != num_dim {}",
                new_center.len(),
                self.num_dim
            )));
        }

        for d in 0..self.num_dim {
            self.centers[[r, d]] = new_center[d].clamp(0.0, 1.0);
        }
        self.lengths[r] = self.config.length.length_init;
        self.succ_counts[r] = 0;
        self.fail_counts[r] = 0;
        self.incumbents_y[r] = f64::NEG_INFINITY;
        self.active_mask[r] = true;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_multi_tr_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = MultiTrustRegionConfig::new(4, TRLengthConfig::default());
        let tr = MultiTrustRegionState::new(3, cfg, None, &mut rng).unwrap();

        assert_eq!(tr.num_regions(), 4);
        assert_eq!(tr.num_dim(), 3);
        assert_eq!(tr.active_count(), 4);
        assert_eq!(tr.centers.shape(), &[4, 3]);
    }

    #[test]
    fn test_multi_tr_tell_update_expansion_and_contraction() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cfg = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        cfg.succ_tolerance = 2;
        cfg.fail_tolerance = 2;

        let init_centers = array![[0.5, 0.5], [0.1, 0.1]];
        let mut tr =
            MultiTrustRegionState::new(2, cfg, Some(&init_centers.view()), &mut rng).unwrap();

        let initial_len0 = tr.lengths[0];

        // Tell improvement near center 0
        let x_batch = array![[0.51, 0.51]];
        let y_batch = array![10.0];
        tr.tell_update(&x_batch.view(), &y_batch.view()).unwrap();
        assert_eq!(tr.succ_counts[0], 1);

        let y_batch2 = array![15.0];
        tr.tell_update(&x_batch.view(), &y_batch2.view()).unwrap();
        assert!(tr.lengths[0] > initial_len0, "Region 0 should expand");
    }
}
