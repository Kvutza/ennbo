//! Integration tests exercising bpann modules for kiss coverage and behavior.

use bpann::backend::open_rejects_record_stride;
use bpann::distance::{batched_sq_l2_f64_rows, row_sq_l2, row_to_f32};
use bpann::index::kmeans::PartitionTree;
use bpann::index::page::closest_child;
use bpann::index::search;
use bpann::index::{BpannIndex, DEFAULT_LEAF_CAPACITY};
use bpann::merge::merge_topk_candidates;
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
