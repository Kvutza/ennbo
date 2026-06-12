use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};

use crate::distance::row_to_f32;
use crate::error::BpannError;
use crate::index::{
    brute_force_topk_mmap, search_exhaustive_leaves, BpannIndex, DEFAULT_LEAF_CAPACITY,
};
use crate::merge::merge_topk_candidates;
use crate::mmap_store::MmapColumnStore;
use crate::observation::{
    self as obs, TrainRowsAt, INDEX_BACKEND, MAX_NUM_DIM, MAX_RECORD_STRIDE,
};

pub const PAPER_TEX_PATH: &str = "papers/bpann_2511.15557v1.tex";
pub const DEFAULT_PENDING_FLUSH_THRESHOLD: usize = 1000;

pub struct BpannBackend {
    work_dir: PathBuf,
    train_x: MmapColumnStore,
    train_y: MmapColumnStore,
    train_yvar: Option<MmapColumnStore>,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    x_scale: Array1<f64>,
    index_dir: PathBuf,
    index: Option<BpannIndex>,
    indexed_rows: usize,
    pending_flush_threshold: usize,
    defer_append_indexing: bool,
    pending_unindexed: AtomicUsize,
    index_dirty: Mutex<bool>,
}

impl BpannBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
    ) -> Result<Self, BpannError> {
        obs::validate_dim_limits(train_x.ncols())?;
        fs::create_dir_all(&work_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        obs::validate_index_backend(&work_dir, INDEX_BACKEND)?;

        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let mut train_x_store =
            MmapColumnStore::mmap_open_or_create(work_dir.join("train_x.bin"), num_dim, None)?;
        let mut train_y_store =
            MmapColumnStore::mmap_open_or_create(work_dir.join("train_y.bin"), num_metrics, None)?;
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store.mmap_append(&train_x.view())?;
            train_y_store.mmap_append(&train_y.view())?;
        }
        let train_yvar_store =
            obs::open_or_append_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;

        let n = train_x_store.nrows;
        let index_dir = work_dir.join("index");
        let indexed_rows = obs::load_indexed_rows(&work_dir).unwrap_or(0).min(n);
        let index = if index_dir.join("header.json").exists() && indexed_rows > 0 {
            Some(BpannIndex::open(index_dir.clone())?)
        } else {
            None
        };

        obs::write_metadata(
            &work_dir,
            n,
            num_dim,
            num_metrics,
            scale_x,
            indexed_rows,
        )?;

        Ok(Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            index_dir,
            index,
            indexed_rows,
            pending_flush_threshold: DEFAULT_PENDING_FLUSH_THRESHOLD,
            defer_append_indexing: true,
            pending_unindexed: AtomicUsize::new(n.saturating_sub(indexed_rows)),
            index_dirty: Mutex::new(indexed_rows < n),
        })
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, BpannError> {
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            false,
            Array1::ones(num_dim),
        )
    }

    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        self
    }

    pub fn with_defer_append_indexing(mut self, defer: bool) -> Self {
        self.defer_append_indexing = defer;
        self
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.pending_flush_threshold
    }

    pub fn defer_append_indexing(&self) -> bool {
        self.defer_append_indexing
    }

    pub fn pending_rows(&self) -> usize {
        self.pending_unindexed.load(Ordering::Relaxed)
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn indexed_rows(&self) -> usize {
        self.indexed_rows
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    pub fn append_row(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        yvar: Option<&Array1<f64>>,
    ) -> Result<(), BpannError> {
        let x2 = x.clone().insert_axis(ndarray::Axis(0));
        let y2 = y.clone().insert_axis(ndarray::Axis(0));
        let yv2 = yvar.map(|v| v.clone().insert_axis(ndarray::Axis(0)));
        self.append_rows(
            &x2.view(),
            &y2.view(),
            yv2.as_ref().map(|a| a.view()).as_ref(),
        )
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), BpannError> {
        if x.nrows() == 0 {
            return Ok(());
        }
        if x.ncols() != self.num_dim || y.ncols() != self.num_metrics || x.nrows() != y.nrows() {
            return Err(BpannError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        obs::check_append_row_limit(self.len() + x.nrows())?;
        self.train_x.mmap_append(x)?;
        self.train_y.mmap_append(y)?;
        obs::append_yvar_on_add(
            &self.work_dir,
            self.num_metrics,
            &mut self.train_yvar,
            yvar,
        )?;
        self.pending_unindexed
            .fetch_add(x.nrows(), Ordering::Relaxed);
        *self.index_dirty.lock().expect("index_dirty") = true;
        obs::write_metadata(
            &self.work_dir,
            self.len(),
            self.num_dim,
            self.num_metrics,
            self.scale_x,
            self.indexed_rows,
        )?;
        if !self.defer_append_indexing
            && self.pending_rows() >= self.pending_flush_threshold
        {
            self.ensure_index_sync()?;
        }
        Ok(())
    }

    pub fn ensure_index_sync(&mut self) -> Result<(), BpannError> {
        let end = self.len();
        if self.indexed_rows >= end {
            return Ok(());
        }
        self.build_index_range(0, end)?;
        *self.index_dirty.lock().expect("index_dirty") = false;
        Ok(())
    }

    fn build_index_range(&mut self, start: usize, end: usize) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let seed = std::env::var("BPANN_BUILD_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(start as u64);
        let mut vectors = Vec::with_capacity(end - start);
        let mut row_ids = Vec::with_capacity(end - start);
        let mut vec_buf = Vec::with_capacity(self.num_dim);
        for i in start..end {
            let row = self.train_x.mmap_row_slice(i)?;
            row_to_f32(row, self.scale_x, self.x_scale.as_slice().unwrap(), &mut vec_buf);
            row_ids.push(i as u32);
            vectors.push(vec_buf.clone());
        }
        let index = BpannIndex::build_from_rows(&row_ids, &vectors, self.num_dim, DEFAULT_LEAF_CAPACITY, seed, self.index_dir.clone())?;
        self.index = Some(index);
        self.indexed_rows = end;
        self.pending_unindexed
            .store(0, Ordering::Relaxed);
        obs::write_metadata(
            &self.work_dir,
            self.len(),
            self.num_dim,
            self.num_metrics,
            self.scale_x,
            self.indexed_rows,
        )?;
        Ok(())
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAt, BpannError> {
        obs::train_rows_at(
            self.len(),
            &self.train_x,
            &self.train_y,
            self.train_yvar.as_ref(),
            indices,
        )
    }

    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), BpannError> {
        let total = self.len();
        let n_query = queries.nrows();
        if total == 0 {
            return Ok((Array2::zeros((n_query, 0)), Array2::zeros((n_query, 0))));
        }
        let indexed = self.indexed_rows;
        let k_eff = search_k.min(total);
        let pool_k = if exclude_nearest {
            (search_k + 1).min(total)
        } else {
            k_eff
        };
        let mut dist2s = Array2::zeros((n_query, k_eff));
        let mut indices = Array2::zeros((n_query, k_eff));
        let scale_x = self.scale_x;
        let x_scale = self.x_scale.view();
        let pending_start = indexed;
        let has_pending = pending_start < total;
        let mut query_buf = vec![0.0f64; self.num_dim];
        let mut query_f32 = Vec::with_capacity(self.num_dim);

        let index_k = if exclude_nearest {
            pool_k.max(k_eff * 2)
        } else {
            k_eff
        };

        for q in 0..n_query {
            let query_row = queries.slice(ndarray::s![q, ..]);
            query_row.assign_to(&mut query_buf);
            row_to_f32(
                &query_buf,
                scale_x,
                x_scale.as_slice().unwrap(),
                &mut query_f32,
            );

            let leg_a: Vec<(u32, f32)> = if indexed > 0 {
                if let Some(ref index) = self.index {
                    search_exhaustive_leaves(index, &query_f32, index_k.max(1))
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if !has_pending {
                let merged = merge_topk_candidates(
                    &self.train_x,
                    &query_buf,
                    &leg_a,
                    &[],
                    k_eff,
                    pool_k,
                    exclude_nearest,
                    scale_x,
                    x_scale.as_slice().unwrap(),
                )?;
                for (j, (id, dist)) in merged.into_iter().enumerate() {
                    dist2s[[q, j]] = dist;
                    indices[[q, j]] = id as i64;
                }
                continue;
            }

            let leg_b = brute_force_topk_mmap(
                &self.train_x,
                pending_start,
                total,
                &query_buf,
                k_eff,
                scale_x,
                x_scale.as_slice().unwrap(),
            )?;

            let merged = merge_topk_candidates(
                &self.train_x,
                &query_buf,
                &leg_a,
                &leg_b,
                k_eff,
                pool_k,
                exclude_nearest,
                scale_x,
                x_scale.as_slice().unwrap(),
            )?;

            for (j, (id, dist)) in merged.into_iter().enumerate() {
                dist2s[[q, j]] = dist;
                indices[[q, j]] = id as i64;
            }
        }
        Ok((dist2s, indices))
    }

    pub fn index_snapshot(&self) -> Option<&BpannIndex> {
        self.index.as_ref()
    }

    pub fn page_bytes(&self) -> Vec<u8> {
        self.index.as_ref().map(|i| i.page_bytes()).unwrap_or_default()
    }

    pub fn mmap_row_slice(&self, i: usize) -> Result<&[f64], BpannError> {
        self.train_x.mmap_row_slice(i)
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.index
            .as_ref()
            .map(|i| i.index_memory_bytes())
            .unwrap_or(0)
    }

    pub fn reopen(work_dir: PathBuf) -> Result<Self, BpannError> {
        let meta_path = work_dir.join("metadata.json");
        let text = fs::read_to_string(&meta_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let num_dim = crate::observation::parse_json_usize_field(&text, "num_dim")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_dim".to_string()))?;
        let num_metrics = crate::observation::parse_json_usize_field(&text, "num_metrics")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_metrics".to_string()))?;
        let scale_x = text.contains("\"scale_x\":true");
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            scale_x,
            Array1::ones(num_dim),
        )
    }
}

pub fn open_rejects_num_dim(num_dim: usize) -> Result<(), BpannError> {
    obs::validate_dim_limits(num_dim)
}

pub fn open_rejects_record_stride(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if record_stride > MAX_RECORD_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_RECORD_STRIDE}"
        )));
    }
    if num_dim > MAX_NUM_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_NUM_DIM}"
        )));
    }
    Ok(())
}
