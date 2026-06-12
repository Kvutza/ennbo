use ennbo::disk_hnsw::hnsw;
use ennbo::disk_hnsw::HnswHeader;
use ennbo::disk_hnsw::insert;
use ennbo::disk_hnsw::store::RamGraph;
use ennbo::knn::MmapColumnStore;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

#[test]
fn search() {
    let mut graph = RamGraph::new(2);
    let mut header = HnswHeader {
        entry_point: 0,
        max_level: 0,
        num_dim: 2,
    };
    let mut rng = StdRng::seed_from_u64(1);
    insert(&mut graph, &mut header, 0, &[0.0, 0.0], &mut rng);
    insert(&mut graph, &mut header, 1, &[1.0, 0.0], &mut rng);
    let hits = hnsw::search(&graph, &header, &[0.0, 0.0], 1, 16, 2);
    assert!(!hits.is_empty());
}

#[test]
fn brute_force_topk() {
    let vecs = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
    let top = hnsw::brute_force_topk(&vecs, &[0.0, 0.0], 1);
    assert_eq!(top[0].0, 0);
}

#[test]
fn brute_force_topk_mmap() {
    let dir = TempDir::new().unwrap();
    let rows = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store.mmap_append(&rows.view()).unwrap();
    let top = hnsw::brute_force_topk_mmap(&store, 0, 2, &[0.0, 0.0], 1, false, &[]).unwrap();
    assert_eq!(top[0].0, 0);
}

#[test]
fn merge_topk_candidates() {
    let dir = TempDir::new().unwrap();
    let rows = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store.mmap_append(&rows.view()).unwrap();
    let merged = hnsw::merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0)],
        &[(1, 1.0)],
        1,
        2,
        false,
        false,
        &[],
    )
    .unwrap();
    assert_eq!(merged.len(), 1);
}
