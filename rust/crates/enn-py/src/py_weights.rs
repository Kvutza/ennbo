use ennbo::{
    apply_sparse, blocks_for_words, draw_sparse, merge_values, missing_words, select_weights,
    sparse_union, sparse_xor, take_words, AcquisitionKind, ComputeBackend, WeightAsk, WeightBlock,
    WeightLeaf, WeightSearch, WeightSelectConfig, WeightTrial,
};
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn array1_vec<T: Copy + Element>(array: PyReadonlyArray1<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

fn array2_vec<T: Copy + Element>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

fn int4_blocks(raw: Vec<(usize, usize, f32, f32, f32)>) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(offset, length, 4, quantization_scale, metric_scale, weight)
                    .map_err(err)
            },
        )
        .collect()
}

fn mixed_blocks(raw: Vec<(usize, usize, u8, f32, f32, f32)>) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, bits, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(
                    offset,
                    length,
                    bits,
                    quantization_scale,
                    metric_scale,
                    weight,
                )
                .map_err(err)
            },
        )
        .collect()
}

fn trial_leaves(raw: Vec<(usize, usize, u8, f32, f32, f32)>) -> PyResult<Vec<WeightLeaf>> {
    raw.into_iter()
        .map(|(offset, length, bits, scale, weight, radius)| {
            WeightLeaf::new(offset, length, bits, scale, weight, radius).map_err(err)
        })
        .collect()
}

#[pyclass(name = "WeightSearch", unsendable)]
pub struct PyWeightSearch {
    inner: WeightSearch,
    pending: Option<WeightTrial>,
}

#[pymethods]
impl PyWeightSearch {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,backend="auto"))]
    fn new(
        base: PyReadonlyArray1<'_, u8>,
        base_value: f32,
        leaves: Vec<(usize, usize, u8, f32, f32, f32)>,
        capacity: usize,
        backend: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: WeightSearch::new(
                &array1_vec(base),
                base_value,
                trial_leaves(leaves)?,
                capacity,
                ComputeBackend::parse(backend).map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }

    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32)> {
        let trial = self
            .inner
            .ask(
                &array1_vec(seeds),
                WeightAsk {
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)?;
        self.pending = Some(trial);
        Ok((trial.index, trial.seed, trial.score))
    }

    fn row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let trial = self
            .pending
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        Ok(self.inner.row(trial).map_err(err)?.into_pyarray_bound(py))
    }

    fn tell(&mut self, value: f32, accept: bool) -> PyResult<()> {
        let trial = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        self.inner.tell(trial, value, accept).map_err(err)
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    #[getter]
    fn row_bytes(&self) -> usize {
        self.inner.row_bytes()
    }
}

fn weight_ucb<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<WeightBlock>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    backend: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    let observation_count = observations.as_array().nrows();
    let candidate_count = candidates.as_array().nrows();
    let observation_bytes = array2_vec(&observations);
    let candidate_bytes = array2_vec(&candidates);
    let outcome_vec = array1_vec(outcomes);
    let result = select_weights(
        &observation_bytes,
        observation_count,
        &outcome_vec,
        &candidate_bytes,
        candidate_count,
        &blocks,
        WeightSelectConfig {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            y_scale,
            beta,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
            backend: ComputeBackend::parse(backend).map_err(err)?,
        },
    )
    .map_err(err)?;
    let row_bytes = candidates.as_array().ncols();
    let start = result.index * row_bytes;
    let selected = candidate_bytes[start..start + row_bytes].to_vec();
    Ok((selected.into_pyarray_bound(py), result.index, result.score))
}

#[pyfunction(name = "weight_int4_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,backend="auto"))]
pub fn weight_int4_select_ucb_py<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    backend: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        int4_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        backend,
    )
}

#[pyfunction(name = "weight_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,backend="auto"))]
pub fn weight_select_ucb_py<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    backend: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        mixed_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        backend,
    )
}

#[pyfunction(name = "sparse_union")]
pub fn sparse_union_py<'py>(
    py: Python<'py>,
    rows: Vec<PyReadonlyArray1<'_, u32>>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let owned: Vec<Vec<u32>> = rows.into_iter().map(array1_vec).collect();
    let refs: Vec<&[u32]> = owned.iter().map(Vec::as_slice).collect();
    Ok(sparse_union(&refs).into_pyarray_bound(py))
}

#[pyfunction(name = "sparse_xor")]
pub fn sparse_xor_py<'py>(
    py: Python<'py>,
    left_words: PyReadonlyArray1<'_, u32>,
    left_masks: PyReadonlyArray1<'_, u32>,
    right_words: PyReadonlyArray1<'_, u32>,
    right_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, masks) = sparse_xor(
        &array1_vec(left_words),
        &array1_vec(left_masks),
        &array1_vec(right_words),
        &array1_vec(right_masks),
    )
    .map_err(err)?;
    Ok((words.into_pyarray_bound(py), masks.into_pyarray_bound(py)))
}

#[pyfunction(name = "sparse_missing")]
pub fn sparse_missing_py<'py>(
    py: Python<'py>,
    cached: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(missing_words(&array1_vec(cached), &array1_vec(query)).into_pyarray_bound(py))
}

#[pyfunction(name = "sparse_merge")]
pub fn sparse_merge_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    extra_words: PyReadonlyArray1<'_, u32>,
    extra_values: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, values) = merge_values(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(extra_words),
        &array1_vec(extra_values),
    )
    .map_err(err)?;
    Ok((words.into_pyarray_bound(py), values.into_pyarray_bound(py)))
}

#[pyfunction(name = "sparse_take")]
pub fn sparse_take_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(
        take_words(&array1_vec(words), &array1_vec(values), &array1_vec(query))
            .map_err(err)?
            .into_pyarray_bound(py),
    )
}

#[pyfunction(name = "sparse_apply")]
pub fn sparse_apply_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    move_words: PyReadonlyArray1<'_, u32>,
    move_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(apply_sparse(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(move_words),
        &array1_vec(move_masks),
    )
    .map_err(err)?
    .into_pyarray_bound(py))
}

#[pyfunction(name = "sparse_blocks")]
pub fn sparse_blocks_py(
    words: PyReadonlyArray1<'_, u32>,
    word_ends: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
) -> PyResult<(Vec<(usize, usize, u8)>, usize)> {
    blocks_for_words(
        &array1_vec(words),
        &array1_vec(word_ends),
        &array1_vec(widths),
    )
    .map_err(err)
}

#[pyfunction(name = "sparse_draw")]
#[pyo3(signature=(count,size,dimension,parameter_ends,parameter_starts,word_offsets,widths,seed))]
pub fn sparse_draw_py<'py>(
    py: Python<'py>,
    count: usize,
    size: usize,
    dimension: u64,
    parameter_ends: PyReadonlyArray1<'_, u64>,
    parameter_starts: PyReadonlyArray1<'_, u64>,
    word_offsets: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
    seed: u64,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let rows = draw_sparse(
        count,
        size,
        dimension,
        &array1_vec(parameter_starts),
        &array1_vec(parameter_ends),
        &array1_vec(word_offsets),
        &array1_vec(widths),
        seed,
    )
    .map_err(err)?;
    let word_rows = PyList::empty_bound(py);
    let mask_rows = PyList::empty_bound(py);
    for (words, masks) in rows {
        word_rows.append(words.into_pyarray_bound(py))?;
        mask_rows.append(masks.into_pyarray_bound(py))?;
    }
    Ok((word_rows, mask_rows))
}

#[allow(clippy::too_many_arguments)]
fn sparse_select_impl(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    acquisition: &str,
    seed: u64,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    backend: &str,
) -> PyResult<(usize, f32)> {
    if observation_words.len() != observation_masks.len()
        || candidate_words.len() != candidate_masks.len()
    {
        return Err(PyValueError::new_err(
            "sparse word and mask row counts must match",
        ));
    }
    let base = array1_vec(base);
    let indices = array1_vec(indices);
    let observation_words: Vec<Vec<u32>> = observation_words.into_iter().map(array1_vec).collect();
    let observation_masks: Vec<Vec<u32>> = observation_masks.into_iter().map(array1_vec).collect();
    let candidate_words: Vec<Vec<u32>> = candidate_words.into_iter().map(array1_vec).collect();
    let candidate_masks: Vec<Vec<u32>> = candidate_masks.into_iter().map(array1_vec).collect();
    let observation_bytes =
        pack_sparse_rows(&base, &indices, &observation_words, &observation_masks)?;
    let candidate_bytes = pack_sparse_rows(&base, &indices, &candidate_words, &candidate_masks)?;
    let outcome_vec = array1_vec(outcomes);
    let result = select_weights(
        &observation_bytes,
        observation_words.len(),
        &outcome_vec,
        &candidate_bytes,
        candidate_words.len(),
        &mixed_blocks(blocks)?,
        WeightSelectConfig {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            y_scale,
            beta,
            acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
            seed,
            backend: ComputeBackend::parse(backend).map_err(err)?,
        },
    )
    .map_err(err)?;
    Ok((result.index, result.score))
}

#[pyfunction(name = "sparse_select")]
#[pyo3(signature=(base,indices,observation_words,observation_masks,outcomes,candidate_words,candidate_masks,blocks,acquisition,seed,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,backend="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_select_py(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    acquisition: &str,
    seed: u64,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    backend: &str,
) -> PyResult<(usize, f32)> {
    sparse_select_impl(
        base,
        indices,
        observation_words,
        observation_masks,
        outcomes,
        candidate_words,
        candidate_masks,
        blocks,
        acquisition,
        seed,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        backend,
    )
}

#[pyfunction(name = "sparse_select_ucb")]
#[pyo3(signature=(base,indices,observation_words,observation_masks,outcomes,candidate_words,candidate_masks,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_select_ucb_py(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
) -> PyResult<(usize, f32)> {
    sparse_select_impl(
        base,
        indices,
        observation_words,
        observation_masks,
        outcomes,
        candidate_words,
        candidate_masks,
        blocks,
        "ucb",
        0,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        "auto",
    )
}

fn pack_sparse_rows(
    base: &[u32],
    indices: &[u32],
    words: &[Vec<u32>],
    masks: &[Vec<u32>],
) -> PyResult<Vec<u8>> {
    if base.len() != indices.len() {
        return Err(PyValueError::new_err(
            "base and sparse index arrays must have the same length",
        ));
    }
    let mut bytes = Vec::with_capacity(words.len() * base.len() * std::mem::size_of::<u32>());
    for (row_words, row_masks) in words.iter().zip(masks) {
        if row_words.len() != row_masks.len() {
            return Err(PyValueError::new_err(
                "sparse row words and masks must have the same length",
            ));
        }
        let mut row = base.to_vec();
        for (&word, &mask) in row_words.iter().zip(row_masks) {
            let position = indices
                .binary_search(&word)
                .map_err(|_| PyValueError::new_err(format!("sparse word {word} is not indexed")))?;
            row[position] ^= mask;
        }
        for value in row {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn py_weights_symbols_are_linked() {
        let _ = (
            std::mem::size_of::<PyWeightSearch>(),
            weight_int4_select_ucb_py,
            weight_select_ucb_py,
            sparse_union_py,
            sparse_xor_py,
            sparse_missing_py,
            sparse_merge_py,
            sparse_take_py,
            sparse_apply_py,
            sparse_blocks_py,
            sparse_draw_py,
            sparse_select_py,
            sparse_select_ucb_py,
            std::mem::size_of::<PyArray1<u8>>,
        );
    }
}
