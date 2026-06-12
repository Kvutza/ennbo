use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, ArrayView2};

use crate::error::BpannError;
use crate::mmap_store::MmapColumnStore;

pub const FORMAT_VERSION: u32 = 1;
pub const MAX_NUM_DIM: usize = 1024;
pub const MAX_RECORD_STRIDE: usize = 8 * 1024 * 1024;
pub const INDEX_BACKEND: &str = "bpann_disk";

pub type TrainRowsAt = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub fn validate_dim_limits(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if num_dim > MAX_NUM_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_NUM_DIM}"
        )));
    }
    if record_stride > MAX_RECORD_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_RECORD_STRIDE}"
        )));
    }
    Ok(())
}

pub fn check_append_row_limit(new_n: usize) -> Result<(), BpannError> {
    if new_n >= u32::MAX as usize {
        return Err(BpannError::InvalidParameter(
            "row count exceeds u32::MAX".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_index_backend(work_dir: &Path, expected: &str) -> Result<(), BpannError> {
    if let Some(backend) = load_index_backend(work_dir) {
        if backend != expected {
            return Err(BpannError::InvalidParameter(format!(
                "work_dir index_backend is {backend}, expected {expected}"
            )));
        }
    }
    Ok(())
}

pub fn load_indexed_rows(work_dir: &Path) -> Option<usize> {
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_json_usize_field(&text, "indexed_rows")
}

pub fn load_index_backend(work_dir: &Path) -> Option<String> {
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_json_string_field(&text, "index_backend")
}

pub fn write_metadata(
    work_dir: &Path,
    num_obs: usize,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    indexed_rows: usize,
) -> Result<(), BpannError> {
    let json = format!(
        "{{\"format_version\":{FORMAT_VERSION},\"num_obs\":{num_obs},\"num_dim\":{num_dim},\"num_metrics\":{num_metrics},\"scale_x\":{scale_x},\"index_backend\":\"{INDEX_BACKEND}\",\"indexed_rows\":{indexed_rows}}}"
    );
    fs::write(work_dir.join("metadata.json"), json)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub fn open_or_append_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: Option<&Array2<f64>>,
) -> Result<Option<MmapColumnStore>, BpannError> {
    if let Some(yv) = train_yvar {
        let yv_path = work_dir.join("train_yvar.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(yv_path, num_metrics, None)?;
        if store.nrows == 0 {
            store.mmap_append(&yv.view())?;
        }
        Ok(Some(store))
    } else {
        Ok(None)
    }
}

pub fn append_yvar_on_add(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: &mut Option<MmapColumnStore>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), BpannError> {
    match (train_yvar.as_mut(), yvar) {
        (Some(store), Some(yv)) => store.mmap_append(yv)?,
        (None, Some(yv)) => {
            let yv_path = work_dir.join("train_yvar.bin");
            let mut store = MmapColumnStore::mmap_open_or_create(yv_path, num_metrics, None)?;
            store.mmap_append(yv)?;
            *train_yvar = Some(store);
        }
        _ => {}
    }
    Ok(())
}

pub fn train_rows_at(
    n: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<TrainRowsAt, BpannError> {
    for &i in indices {
        if i >= n {
            return Err(BpannError::InvalidParameter(format!(
                "train_rows_at index {i} out of range [0, {n})"
            )));
        }
    }
    let x = train_x.mmap_gather(indices)?;
    let y = train_y.mmap_gather(indices)?;
    let yvar = train_yvar
        .map(|s| s.mmap_gather(indices))
        .transpose()?;
    Ok((x, y, yvar))
}

pub fn mark_index_dirty(index_dirty: &Mutex<bool>) {
    *index_dirty.lock().expect("index_dirty mutex poisoned") = true;
}

pub(crate) fn parse_json_usize_field(text: &str, field: &str) -> Option<usize> {
    let needle = format!("\"{field}\":");
    text.split_once(&needle).and_then(|(_, tail)| {
        tail.trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|digits| digits.parse().ok())
    })
}

pub fn parse_json_string_field(text: &str, field: &str) -> Option<String> {
    let marker = format!("\"{field}\":\"");
    text.split_once(&marker).and_then(|(_, tail)| {
        tail.split_once('"').map(|(value, _)| value.to_string())
    })
}
