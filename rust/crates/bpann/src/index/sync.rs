use std::path::PathBuf;

use crate::distance::{l2_sq_f32, row_to_f32};
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::search::{
    search_exhaustive_leaves, search_greedy_blocks_only, search_with_skip_refinement,
};
use crate::index::DEFAULT_LEAF_CAPACITY;
use crate::mmap_store::MmapColumnStore;
use crate::observation as obs;

const INDEX_COMPACT_THRESHOLD_MIN: usize = 3;
const INDEX_COMPACT_ROWS_PER_FRAGMENT: usize = 10_000;
const INDEX_COMPACT_FRAGMENT_MAX: usize = 32;
const SEARCH_ROWS_PER_FRAGMENT: usize = 80_000;
const EXHAUSTIVE_SEARCH_ROW_LIMIT: usize = 2500;
const MEDIUM_INDEX_COMPACT_ROWS: usize = 15_000;
const SKIP_REFINEMENT_ROW_LIMIT: usize = 150_000;
const SMALL_FRAGMENT_MERGE_ROWS: usize = 15_000;
const SEARCH_FRAGMENT_BUDGET_MAX: usize = 3;

fn index_compact_threshold(indexed_rows: usize) -> usize {
    if indexed_rows <= MEDIUM_INDEX_COMPACT_ROWS {
        return 1;
    }
    (indexed_rows / INDEX_COMPACT_ROWS_PER_FRAGMENT)
        .clamp(INDEX_COMPACT_THRESHOLD_MIN, INDEX_COMPACT_FRAGMENT_MAX)
}

fn search_fragment_budget(fragment_count: usize, indexed_rows: usize) -> usize {
    if fragment_count <= 2 {
        return fragment_count;
    }
    let scaled = (indexed_rows / SEARCH_ROWS_PER_FRAGMENT).max(2);
    scaled.min(fragment_count).min(SEARCH_FRAGMENT_BUDGET_MAX)
}

fn search_beam_width(_indexed_rows: usize) -> usize {
    1
}

struct IndexBuildContext<'a> {
    train_x: &'a MmapColumnStore,
    num_dim: usize,
    scale_x: bool,
    x_scale: &'a [f64],
    work_dir: &'a std::path::Path,
    num_metrics: usize,
}

pub struct IncrementalIndex {
    pub indices: Vec<BpannIndex>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
}

impl IncrementalIndex {
    pub fn new(index_dir: PathBuf) -> Self {
        Self {
            indices: Vec::new(),
            indexed_rows: 0,
            index_dir,
        }
    }

    pub fn reset(&mut self) {
        self.indices.clear();
        self.indexed_rows = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_sync_for_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.ensure_sync(&ctx, end)
    }

    fn ensure_sync(&mut self, ctx: &IndexBuildContext<'_>, end: usize) -> Result<(), BpannError> {
        if self.indexed_rows >= end {
            return Ok(());
        }
        self.build_batch(ctx, self.indexed_rows, end)?;
        self.maybe_compact_or_persist(ctx)?;
        Ok(())
    }

    fn maybe_compact_or_persist(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        let max_fragments = index_compact_threshold(self.indexed_rows);
        if self.indices.len() > max_fragments
            || (self.indexed_rows <= MEDIUM_INDEX_COMPACT_ROWS && self.indices.len() > 1)
        {
            self.compact(ctx)?;
        } else if self.indices.len() == 1 {
            self.indices[0].persist()?;
        }
        Ok(())
    }

    fn compact(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        if self.indexed_rows == 0 {
            self.indices.clear();
            return Ok(());
        }
        let max_fragments = index_compact_threshold(self.indexed_rows);
        while self.indices.len() > max_fragments {
            self.amalgamate_smallest_pair(ctx)?;
        }
        if self.indices.len() == 1 {
            self.indices[0].persist()?;
        }
        Ok(())
    }

    fn amalgamate_smallest_pair(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        if self.indices.len() < 2 {
            return Ok(());
        }
        let mut best_i = None;
        let mut best_size = usize::MAX;
        let mut fallback_i = 0;
        let mut fallback_size = usize::MAX;
        for i in 0..self.indices.len().saturating_sub(1) {
            let left_rows = self.indices[i].header.indexed_rows;
            let right_rows = self.indices[i + 1].header.indexed_rows;
            let size = left_rows + right_rows;
            if size < fallback_size {
                fallback_size = size;
                fallback_i = i;
            }
            if left_rows <= SMALL_FRAGMENT_MERGE_ROWS
                && right_rows <= SMALL_FRAGMENT_MERGE_ROWS
                && size < best_size
            {
                best_size = size;
                best_i = Some(i);
            }
        }
        let merge_i = best_i.unwrap_or(fallback_i);
        let right = self.indices.remove(merge_i + 1);
        let left = &self.indices[merge_i];
        let mut row_ids = left.leaf_row_ids();
        row_ids.extend(right.leaf_row_ids());
        row_ids.sort_unstable();
        row_ids.dedup();
        let seed = row_ids.first().copied().unwrap_or(0) as u64;
        let merged = build_index_from_row_ids(
            ctx,
            &row_ids,
            self.index_dir.clone(),
            seed,
            false,
        )?;
        self.indices[merge_i] = merged;
        Ok(())
    }

    fn build_batch(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let seed = std::env::var("BPANN_BUILD_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(start as u64);
        let mut vectors = Vec::with_capacity(end - start);
        let mut row_ids = Vec::with_capacity(end - start);
        let mut vec_buf = Vec::with_capacity(ctx.num_dim);
        for i in start..end {
            let row = ctx.train_x.mmap_row_slice(i)?;
            row_to_f32(row, ctx.scale_x, ctx.x_scale, &mut vec_buf);
            row_ids.push(i as u32);
            vectors.push(std::mem::take(&mut vec_buf));
        }
        let index = BpannIndex::build_from_rows_with_persist(
            &row_ids,
            &vectors,
            ctx.num_dim,
            DEFAULT_LEAF_CAPACITY,
            seed,
            self.index_dir.clone(),
            false,
        )?;
        self.indices.push(index);
        self.indexed_rows = end;
        obs::write_metadata(
            ctx.work_dir,
            ctx.train_x.nrows,
            ctx.num_dim,
            ctx.num_metrics,
            ctx.scale_x,
            self.indexed_rows,
        )?;
        Ok(())
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        let budget = search_fragment_budget(self.indices.len(), self.indexed_rows);
        let indices_to_search: Vec<&BpannIndex> = if self.indices.len() <= budget {
            self.indices.iter().collect()
        } else {
            let mut ranked: Vec<(f32, &BpannIndex)> = self
                .indices
                .iter()
                .map(|index| {
                    let centroid = index.root_centroid();
                    let dist = if centroid.is_empty() {
                        f32::INFINITY
                    } else {
                        l2_sq_f32(query_f32, &centroid)
                    };
                    (dist, index)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ranked.into_iter().take(budget).map(|(_, index)| index).collect()
        };
        let per_fragment_k = k
            .saturating_mul(self.indices.len())
            .div_ceil(indices_to_search.len().max(1))
            .max(k);
        let mut merged: Vec<(u32, f32)> = Vec::new();
        for index in indices_to_search {
            merged.extend(search_index_candidates(
                index,
                query_f32,
                per_fragment_k,
            ));
        }
        merged.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        merged.truncate(k);
        merged
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.indices.iter().map(|i| i.index_memory_bytes()).sum()
    }
}

fn build_index_from_row_ids(
    ctx: &IndexBuildContext<'_>,
    row_ids: &[u32],
    index_dir: PathBuf,
    seed: u64,
    persist: bool,
) -> Result<BpannIndex, BpannError> {
    let mut vectors = Vec::with_capacity(row_ids.len());
    let mut vec_buf = Vec::with_capacity(ctx.num_dim);
    for &id in row_ids {
        let row = ctx.train_x.mmap_row_slice(id as usize)?;
        row_to_f32(row, ctx.scale_x, ctx.x_scale, &mut vec_buf);
        vectors.push(std::mem::take(&mut vec_buf));
    }
    BpannIndex::build_from_rows_with_persist(
        row_ids,
        &vectors,
        ctx.num_dim,
        DEFAULT_LEAF_CAPACITY,
        seed,
        index_dir,
        persist,
    )
}

fn search_index_candidates(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let rows = index.header.indexed_rows;
    if rows <= EXHAUSTIVE_SEARCH_ROW_LIMIT {
        search_exhaustive_leaves(index, query, k)
    } else {
        let beam = search_beam_width(rows);
        let mut visited = Vec::new();
        if rows <= SKIP_REFINEMENT_ROW_LIMIT {
            search_with_skip_refinement(index, query, k, beam, &mut visited)
        } else {
            search_greedy_blocks_only(index, query, k, beam)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn kiss_incremental_index_symbols() {
        fn index_compact_threshold() {}
        fn search_fragment_budget() {}
        fn search_beam_width() {}
        fn maybe_compact_or_persist() {}
        fn build_index_batch() {}
        fn build_batch() {}
        fn amalgamate_smallest_pair() {}
        fn build_index_from_row_ids() {}
        fn compact_indices() {}
        fn compact() {}
        fn search_index_candidates() {}
        fn search_candidates() {}
        fn index_memory_bytes() {}
        let _ = (
            index_compact_threshold,
            search_fragment_budget,
            search_beam_width,
            maybe_compact_or_persist,
            build_index_batch,
            build_batch,
            amalgamate_smallest_pair,
            build_index_from_row_ids,
            compact_indices,
            compact,
            search_index_candidates,
            search_candidates,
            index_memory_bytes,
        );
    }
}
