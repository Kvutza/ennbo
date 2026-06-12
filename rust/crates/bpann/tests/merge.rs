use bpann::merge;
use bpann::mmap_store::MmapColumnStore;
use ndarray::array;
use tempfile::TempDir;

#[test]
fn merge_topk_candidates() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let merged = merge::merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0), (1, 1.0)],
        &[],
        1,
        2,
        true,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    assert_eq!(merged[0].0, 1);
}
