use std::ffi::c_void;
use std::sync::Arc;
use std::time::{Duration, Instant};

use metal::{ComputePipelineState, MTLSize};
use ndarray::{Array2, ArrayView2};

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k};
use crate::apple_gpu::Runtime;
use crate::index::IndexError;

const THREADS: u64 = 256;
const TILE_ROWS: usize = 1024;
const MAX_K: usize = 1024;
const SOURCE: &str = include_str!("metal_index.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    dim: u32,
    queries: u32,
    tile_start: u32,
    tile_rows: u32,
    k: u32,
}

pub(crate) struct MetalIndex {
    runtime: Arc<Runtime>,
    distance: Vec<ComputePipelineState>,
    distance_choice: Option<usize>,
    local_topk: ComputePipelineState,
    merge: ComputePipelineState,
    init_results: ComputePipelineState,
    rows: metal::Buffer,
    host_rows: Vec<f32>,
    row_capacity: usize,
    scratch: Scratch,
    num_dim: usize,
}

struct Scratch {
    query: metal::Buffer,
    tile_distances: metal::Buffer,
    local_distances: metal::Buffer,
    local_indices: metal::Buffer,
    result_distances: metal::Buffer,
    result_indices: metal::Buffer,
    query_capacity: usize,
    k_capacity: usize,
}

impl Scratch {
    fn new(runtime: &Runtime) -> Self {
        let empty = || runtime.buffer::<u32>(1);
        Self {
            query: empty(),
            tile_distances: empty(),
            local_distances: empty(),
            local_indices: empty(),
            result_distances: empty(),
            result_indices: empty(),
            query_capacity: 0,
            k_capacity: 0,
        }
    }

    fn ensure(&mut self, runtime: &Runtime, dim: usize, queries: usize, k: usize) {
        if queries <= self.query_capacity && k <= self.k_capacity {
            return;
        }
        self.query_capacity = next_capacity(queries);
        self.k_capacity = next_capacity(k);
        self.query = runtime.buffer::<f32>(self.query_capacity * dim);
        self.tile_distances = runtime.buffer::<f32>(self.query_capacity * TILE_ROWS);
        self.local_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.local_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
        self.result_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.result_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
    }
}

impl MetalIndex {
    pub(crate) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::with_agx(num_dim, train, false)
    }

    pub(crate) fn new_agx(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::with_agx(num_dim, train, true)
    }

    fn with_agx(num_dim: usize, train: &ArrayView2<f64>, agx: bool) -> Result<Self, IndexError> {
        let runtime = Runtime::shared().map_err(IndexError::InvalidParameter)?;
        let pipeline = |name: &str| -> Result<ComputePipelineState, IndexError> {
            let result = if agx {
                runtime.agx_pipeline(SOURCE, "index", name)
            } else {
                runtime.pipeline(SOURCE, "index", name)
            };
            result.map_err(IndexError::InvalidParameter)
        };
        let names: &[&str] = if agx {
            &["distance_rows", "distance_rows_2", "distance_rows_4"]
        } else {
            &["distance_rows"]
        };
        let distance = names
            .iter()
            .map(|name| pipeline(name))
            .collect::<Result<Vec<_>, _>>()?;
        let local_topk = pipeline("local_topk")?;
        let merge = pipeline("merge_topk")?;
        let init_results = pipeline("init_results")?;
        let scratch = Scratch::new(&runtime);
        let mut index = Self {
            rows: runtime.buffer::<u32>(1),
            runtime,
            distance,
            distance_choice: (!agx).then_some(0),
            local_topk,
            merge,
            init_results,
            host_rows: Vec::new(),
            row_capacity: 0,
            scratch,
            num_dim,
        };
        index.rebuild(train)?;
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.host_rows.len() / self.num_dim
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.row_capacity
            .saturating_mul(self.num_dim)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn rebuild(&mut self, train: &ArrayView2<f64>) -> Result<(), IndexError> {
        self.check_rows(train)?;
        self.host_rows = arr2_rows_to_f32(train);
        self.upload_rows();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        self.check_rows(rows)?;
        let start = self.len();
        let values = arr2_rows_to_f32(rows);
        self.host_rows.extend_from_slice(&values);
        if self.len() > self.row_capacity {
            self.upload_rows();
        } else {
            self.write_f32(&self.rows, start * self.num_dim, &values);
        }
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        self.check_rows(queries)?;
        if k_eff == 0 || k_eff > MAX_K {
            return Err(IndexError::InvalidParameter(format!(
                "Metal index supports 1..={MAX_K} neighbors, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok(pad_neighbor_cols_to_search_k(
                Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                Array2::zeros((queries.nrows(), 0)),
                search_k,
            ));
        }

        let query_values = arr2_rows_to_f32(queries);
        self.scratch
            .ensure(&self.runtime, self.num_dim, queries.nrows(), k_eff);
        self.write_f32(&self.scratch.query, 0, &query_values);
        self.calibrate_distance(queries.nrows())?;

        let command_buffer = self.runtime.queue.new_command_buffer();
        let init_params = Params {
            rows: to_u32(self.len(), "row count")?,
            dim: to_u32(self.num_dim, "dimension")?,
            queries: to_u32(queries.nrows(), "query count")?,
            tile_start: 0,
            tile_rows: 0,
            k: to_u32(k_eff, "neighbor count")?,
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.init_results);
        encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
        encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
        set_params(encoder, 2, &init_params);
        dispatch(encoder, queries.nrows() * k_eff);
        encoder.end_encoding();

        for tile_start in (0..self.len()).step_by(TILE_ROWS) {
            let tile_rows = (self.len() - tile_start).min(TILE_ROWS);
            let params = Params {
                rows: to_u32(self.len(), "row count")?,
                dim: to_u32(self.num_dim, "dimension")?,
                queries: to_u32(queries.nrows(), "query count")?,
                tile_start: to_u32(tile_start, "tile start")?,
                tile_rows: to_u32(tile_rows, "tile rows")?,
                k: to_u32(k_eff, "neighbor count")?,
            };

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.distance[self.distance_choice.expect("distance schedule calibrated")],
            );
            encoder.set_buffer(0, Some(&self.rows), 0);
            encoder.set_buffer(1, Some(&self.scratch.query), 0);
            encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
            set_params(encoder, 3, &params);
            dispatch(encoder, queries.nrows() * TILE_ROWS);
            encoder.end_encoding();

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.local_topk);
            encoder.set_buffer(0, Some(&self.scratch.tile_distances), 0);
            encoder.set_buffer(1, Some(&self.scratch.local_distances), 0);
            encoder.set_buffer(2, Some(&self.scratch.local_indices), 0);
            set_params(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: queries.nrows() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.merge);
            encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
            encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
            encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
            encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
            set_params(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: queries.nrows() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let distances = unsafe {
            std::slice::from_raw_parts(
                self.scratch.result_distances.contents().cast::<f32>(),
                queries.nrows() * k_eff,
            )
        };
        let indices = unsafe {
            std::slice::from_raw_parts(
                self.scratch.result_indices.contents().cast::<u32>(),
                queries.nrows() * k_eff,
            )
        };
        let mut out_dist = Array2::zeros((queries.nrows(), k_eff));
        let mut out_idx = Array2::zeros((queries.nrows(), k_eff));
        for q in 0..queries.nrows() {
            for k in 0..k_eff {
                out_dist[[q, k]] = f64::from(distances[q * k_eff + k]);
                out_idx[[q, k]] = i64::from(indices[q * k_eff + k]);
            }
        }
        Ok(pad_neighbor_cols_to_search_k(out_dist, out_idx, search_k))
    }

    fn calibrate_distance(&mut self, queries: usize) -> Result<(), IndexError> {
        if self.distance_choice.is_some() {
            return Ok(());
        }
        let query_count = queries.min(64);
        let params = Params {
            rows: to_u32(self.len(), "row count")?,
            dim: to_u32(self.num_dim, "dimension")?,
            queries: to_u32(query_count, "query count")?,
            tile_start: 0,
            tile_rows: to_u32(self.len().min(TILE_ROWS), "tile rows")?,
            k: 1,
        };
        self.run_distance(0, &params);
        let reference = self.distance_values(query_count);
        let runtime = Arc::clone(&self.runtime);
        let choice = runtime
            .schedule(
                "knn.distance",
                &[self.num_dim],
                self.distance.len(),
                |schedule| {
                    self.run_distance(schedule, &params);
                    if !same_distances(&reference, &self.distance_values(query_count)) {
                        return Ok(None);
                    }
                    let mut times = [Duration::MAX; 3];
                    for elapsed in &mut times {
                        let start = Instant::now();
                        self.run_distance(schedule, &params);
                        *elapsed = start.elapsed();
                    }
                    times.sort_unstable();
                    Ok(Some(times[1]))
                },
            )
            .map_err(IndexError::InvalidParameter)?;
        self.distance_choice = Some(choice);
        Ok(())
    }

    fn run_distance(&self, schedule: usize, params: &Params) {
        let command_buffer = self.runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance[schedule]);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.query), 0);
        encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
        set_params(encoder, 3, params);
        dispatch(encoder, params.queries as usize * TILE_ROWS);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn distance_values(&self, queries: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.scratch.tile_distances.contents().cast::<f32>(),
                queries * TILE_ROWS,
            )
            .to_vec()
        }
    }

    fn check_rows(&self, rows: &ArrayView2<f64>) -> Result<(), IndexError> {
        if rows.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows.ncols(),
            });
        }
        Ok(())
    }

    fn upload_rows(&mut self) {
        self.row_capacity = next_capacity(self.len());
        self.rows = self.runtime.buffer::<f32>(self.row_capacity * self.num_dim);
        self.write_f32(&self.rows, 0, &self.host_rows);
    }

    fn write_f32(&self, buffer: &metal::Buffer, offset: usize, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents().cast::<f32>().add(offset),
                values.len(),
            );
        }
    }
}

fn same_distances(left: &[f32], right: &[f32]) -> bool {
    left.iter().zip(right).all(|(&a, &b)| {
        a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4 * (1.0 + a.abs()))
    })
}

fn next_capacity(value: usize) -> usize {
    value
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(value.max(1))
}

fn set_params(encoder: &metal::ComputeCommandEncoderRef, slot: u64, params: &Params) {
    encoder.set_bytes(
        slot,
        std::mem::size_of::<Params>() as u64,
        (params as *const Params).cast::<c_void>(),
    );
}

fn dispatch(encoder: &metal::ComputeCommandEncoderRef, count: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: count.div_ceil(THREADS as usize) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
}

fn to_u32(value: usize, name: &str) -> Result<u32, IndexError> {
    u32::try_from(value)
        .map_err(|_| IndexError::InvalidParameter(format!("{name} exceeds u32 range")))
}
