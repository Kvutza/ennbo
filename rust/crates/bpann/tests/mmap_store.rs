use bpann::mmap_store;
use ndarray::array;
use tempfile::TempDir;

#[test]
#[allow(non_snake_case)]
fn MmapColumnStore() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[1.0, 2.0]].view()).unwrap();
    assert_eq!(store.mmap_row_slice(0).unwrap()[0], 1.0);
}

#[test]
fn mmap_open_or_create() {
    let dir = TempDir::new().unwrap();
    mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
}

#[test]
fn mmap_append() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
}

#[test]
fn mmap_row_slice() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 1.0]].view()).unwrap();
    assert_eq!(store.mmap_row_slice(0).unwrap()[1], 1.0);
}

#[test]
fn mmap_gather() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view()).unwrap();
    assert_eq!(store.mmap_gather(&[1]).unwrap().nrows(), 1);
}
