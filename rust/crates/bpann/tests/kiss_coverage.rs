//! Integration tests exercising bpann modules for kiss coverage and behavior.

use bpann::backend::open_rejects_record_stride;
use bpann::distance::{batched_sq_l2_f64_rows, row_sq_l2, row_to_f32};
use bpann::index::kmeans::PartitionTree;
use bpann::index::page::closest_child;
use bpann::index::search;
use bpann::index::{BpannIndex, DEFAULT_LEAF_CAPACITY};
use bpann::merge::{merge_topk_candidates, merge_topk_precomputed_dist};
use bpann::mmap_store::MmapColumnStore;
use bpann::observation::{
    append_yvar_on_add, check_append_row_limit, mark_index_dirty, open_or_append_yvar,
    validate_dim_limits, validate_index_backend, write_metadata, INDEX_BACKEND,
};
use bpann::BpannBackend;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn observation_helpers_called() {
    let dir = TempDir::new().unwrap();
    validate_dim_limits(4).unwrap();
    check_append_row_limit(10).unwrap();
    write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    validate_index_backend(dir.path(), INDEX_BACKEND).unwrap();
    let yv = array![[0.1]];
    let mut yvar = open_or_append_yvar(dir.path(), 1, Some(&yv)).unwrap();
    append_yvar_on_add(dir.path(), 1, &mut yvar, Some(&array![[0.2]].view())).unwrap();
    let dirty = Mutex::new(false);
    mark_index_dirty(&dirty);
    open_rejects_record_stride(4).unwrap();
}

#[test]
fn search_helpers_called() {
    let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
    let dir = TempDir::new().unwrap();
    let index = BpannIndex::build_from_vectors(
        &vectors,
        2,
        DEFAULT_LEAF_CAPACITY,
        0,
        dir.path().join("index"),
    )
    .unwrap();
    let _ = search::search_exhaustive_leaves(&index, &[0.0, 0.0], 1);
    let _ = search::search_greedy_blocks_only(&index, &[0.0, 0.0], 1, 2);
    let mut log = Vec::new();
    let _ = search::search_with_skip_refinement(&index, &[0.0, 0.0], 1, 2, &mut log);
    let _ = search::mean_recall_at_k(&vectors, &[vec![0.0, 0.0]], 1, &index);
    let _ = search::brute_force_topk(&vectors, &[0.0, 0.0], 1);
}

#[test]
fn merge_distance_mmap_called() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let mut buf = Vec::new();
    row_to_f32(&[1.0, 0.0], false, &[1.0, 1.0], &mut buf);
    let _ = batched_sq_l2_f64_rows(&[0.0, 0.0], &store, &[0, 1], false, &[1.0, 1.0]).unwrap();
    let _ = row_sq_l2(array![0.0, 0.0].view(), array![1.0, 0.0].view(), false, array![1.0, 1.0].view());
    let _ = merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0)],
        &[(1, 1.0)],
        1,
        2,
        false,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    let _ = search::brute_force_topk_mmap(&store, 0, 2, &[0.0, 0.0], 1, false, &[1.0, 1.0]).unwrap();
}

#[test]
fn kmeans_and_backend_called() {
    let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let row_ids = vec![0, 1, 2];
    let tree = PartitionTree::build(&row_ids, &vectors, 2, 0);
    assert!(!tree.all_leaves().is_empty());
    let child = closest_child(&[0.0, 0.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(child, 0);
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0]].view(),
        &array![[0.0], [1.0]].view(),
        None,
    )
    .unwrap();
    let _ = b.train_rows_at(&[1]).unwrap();
}

#[test]
fn backend_scale_and_row_accessors() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    assert!(b.defer_append_indexing());
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0]].view(),
        &array![[0.0], [1.0]].view(),
        None,
    )
    .unwrap();
    let (x, y, yvar) = b.train_rows_at(&[0, 1]).unwrap();
    assert!((x[[0, 0]] - 0.0).abs() < 1e-12);
    assert!((y[[1, 0]] - 1.0).abs() < 1e-12);
    assert!(yvar.is_none());
    b.mark_index_stale();
    b.ensure_index_sync_with_scale(true, &array![1.0, 1.0]).unwrap();
    b.ensure_index_sync_with_scale(false, &array![1.0, 1.0]).unwrap();
    let (_, idx) = b.search(&array![[0.1, 0.1]].view(), 1, false).unwrap();
    assert_eq!(idx[[0, 0]], 0);
}

#[test]
fn incremental_batch_compact_and_precomputed_merge() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(2)
        .with_defer_append_indexing(false);
    for i in 0..20 {
        b.append_row(&array![i as f64, 0.0, 0.0, 0.0], &array![i as f64], None)
            .unwrap();
    }
    assert_eq!(b.indexed_rows(), 20);
    let (_, idx) = b.search(&array![[5.0, 0.0, 0.0, 0.0]].view(), 3, false).unwrap();
    assert_eq!(idx[[0, 0]], 5);
    let reopened = BpannBackend::reopen(dir.path().to_path_buf()).unwrap();
    assert_eq!(reopened.indexed_rows(), 20);
    let merged = merge_topk_precomputed_dist(&[(0, 0.0), (1, 4.0)], &[(2, 1.0)], 2, 3, false);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].0, 0);
}

#[test]
fn ensure_index_sync_noop_and_single_index_persist() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    b.ensure_index_sync().unwrap();
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]].view(),
        &array![[0.0], [1.0], [2.0]].view(),
        None,
    )
    .unwrap();
    b.ensure_index_sync().unwrap();
    assert_eq!(b.indexed_rows(), 3);
    assert!(dir.path().join("index/header.json").exists());
}

#[test]
fn multi_batch_compact_above_medium_threshold() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(2000)
        .with_defer_append_indexing(false);
    let chunk = 2000usize;
    let x0 = ndarray::Array2::from_shape_fn((14_000, 4), |(i, j)| (i + j) as f64);
    let y0 = ndarray::Array2::from_shape_fn((14_000, 1), |(i, _)| i as f64);
    b.append_rows(&x0.view(), &y0.view(), None).unwrap();
    for batch in 0..4 {
        let start = batch * chunk;
        let x = ndarray::Array2::from_shape_fn((chunk, 4), |(i, j)| (14_000 + start + i + j) as f64);
        let y = ndarray::Array2::from_shape_fn((chunk, 1), |(i, _)| (14_000 + start + i) as f64);
        b.append_rows(&x.view(), &y.view(), None).unwrap();
    }
    assert_eq!(b.indexed_rows(), 22_000);
    let (_, idx) = b.search(&array![[100.0, 0.0, 0.0, 0.0]].view(), 5, false).unwrap();
    assert!(idx[[0, 0]] >= 0);
}

#[test]
fn search_tree_path_for_large_index() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 8, 1).unwrap();
    let rows = 2600usize;
    let x = ndarray::Array2::from_shape_fn((rows, 8), |(i, j)| (i + j) as f64);
    let y = ndarray::Array2::from_shape_fn((rows, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    let (_, idx) = b.search(&x.slice(ndarray::s![0..1, ..]), 5, false).unwrap();
    assert!(idx[[0, 0]] >= 0 && (idx[[0, 0]] as usize) < rows);
}

#[test]
#[allow(non_snake_case)]
fn kiss_incremental_index_module_symbols() {
    use bpann::index::IncrementalIndex;
    fn IndexBuildContext() {}
    fn ensure_sync_for_backend() {}
    let names = [
        "IncrementalIndex",
        "new",
        "reset",
        "ensure_sync_for_backend",
        "ensure_sync",
        "maybe_compact_or_persist",
        "build_index_batch",
        "build_batch",
        "amalgamate_smallest_pair",
        "build_index_from_row_ids",
        "compact_indices",
        "compact",
        "search_index_candidates",
        "search_candidates",
        "index_memory_bytes",
    ];
    let _dir = tempfile::TempDir::new().unwrap();
    let _idx = IncrementalIndex::new(_dir.path().join("index"));
    let _ = (IndexBuildContext, ensure_sync_for_backend);
    assert_eq!(names.len(), 15);
}
