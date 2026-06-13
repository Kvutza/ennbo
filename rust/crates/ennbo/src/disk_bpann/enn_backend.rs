//! Disk-backed ENN backend wrapping the standalone `bpann` crate.

use std::path::PathBuf;

use bpann::BpannBackend;
use ndarray::{Array1, Array2, ArrayView2};

use crate::backend::TrainRowsAtResult;
use crate::error::ENNError;
use crate::index::IndexDriver;

fn bpann_err(e: bpann::BpannError) -> ENNError {
    match e {
        bpann::BpannError::InvalidShape { expected, got } => ENNError::InvalidShape { expected, got },
        bpann::BpannError::InvalidParameter(s) => ENNError::InvalidParameter(s),
    }
}

pub struct DiskBpannEnnBackend {
    inner: BpannBackend,
    driver: IndexDriver,
    num_metrics: usize,
}

impl DiskBpannEnnBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::BpAnnDisk {
            return Err(ENNError::InvalidParameter(
                "DiskBpannEnnBackend requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        let num_metrics = train_y.ncols();
        Ok(Self {
            inner: BpannBackend::new(work_dir, train_x, train_y, train_yvar, scale_x, x_scale)
                .map_err(bpann_err)?,
            driver,
            num_metrics,
        })
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, ENNError> {
        Ok(Self {
            inner: BpannBackend::new_empty(work_dir, num_dim, num_metrics).map_err(bpann_err)?,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
        })
    }

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn pending_unindexed_count(&self) -> usize {
        self.inner.pending_rows()
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.inner.pending_flush_threshold()
    }

    pub fn append_syncs_at_threshold(&self) -> bool {
        !self.inner.defer_append_indexing()
    }

    pub fn defer_index_sync_for_search(&self) -> bool {
        true
    }

    pub fn is_index_stale(&self) -> bool {
        false
    }

    pub fn mark_index_stale(&mut self) {
        self.inner.mark_index_stale();
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        Ok(())
    }

    pub fn schedule_background_flush(&mut self) -> Result<(), ENNError> {
        if self.inner.defer_append_indexing()
            && self.pending_unindexed_count() >= self.pending_flush_threshold()
        {
            self.inner.ensure_index_sync().map_err(bpann_err)?;
        }
        Ok(())
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        self.inner.append_rows(x, y, yvar).map_err(bpann_err)
    }

    pub fn ensure_index_sync(
        &mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        self.inner
            .ensure_index_sync_with_scale(scale_x, x_scale)
            .map_err(bpann_err)
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        self.inner.train_rows_at(indices).map_err(bpann_err)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(
            self.inner.mmap_row_slice(i).map_err(bpann_err)?.to_vec(),
        ))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        let (_, y, _) = self.inner.train_rows_at(&[i]).map_err(bpann_err)?;
        Ok(y.row(0).to_owned())
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        let (_, _, yvar) = self.inner.train_rows_at(&[i]).map_err(bpann_err)?;
        Ok(yvar.map(|v| v.row(0).to_owned()))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.inner
            .search(x, search_k as usize, exclude_nearest)
            .map_err(bpann_err)
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.inner.index_memory_bytes())
    }
}
