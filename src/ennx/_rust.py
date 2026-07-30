from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from . import ennx_rust as _ext
except ImportError as exc:  # pragma: no cover - exercised when extension unavailable
    raise ImportError(
        "Rust extension submodule `ennx.ennx_rust` is not available"
    ) from exc


hypervolume_2d_max = _ext.hypervolume.hypervolume_2d_max
normal_hash_batch_multi_seed_fast = _ext.hash.normal_hash_batch_multi_seed_fast
standardize_y = _ext.util.standardize_y
pareto_front_2d_maximize = _ext.util.pareto_front_2d_maximize
calculate_sobol_indices = _ext.util.calculate_sobol_indices
sobol_sequence = _ext.util.sobol_sequence
arms_from_pareto_fronts = _ext.util.arms_from_pareto_fronts
set_config_path = _ext.util.set_config_path
ensure_config_file = _ext.util.ensure_config_file
EpistemicNearestNeighbors = _ext.model.EpistemicNearestNeighbors
ENNParams = _ext.model.ENNParams
ENNStatefulFitter = _ext.fit.ENNStatefulFitter
subsample_loglik = _ext.fit.subsample_loglik
Optimizer = _ext.optimizer.Optimizer
create_optimizer_enn = _ext.optimizer.create_optimizer_enn
create_optimizer_enn_multi_tr = _ext.optimizer.create_optimizer_enn_multi_tr
create_optimizer_zero = _ext.optimizer.create_optimizer_zero
create_optimizer_lhd = _ext.optimizer.create_optimizer_lhd
dense_apply = _ext.optimizer.dense_apply
dense_dist2 = _ext.optimizer.dense_dist2
dense_linear = _ext.optimizer.dense_linear
DenseLinear = _ext.optimizer.DenseLinear


__all__ = [
    "ENNParams",
    "ENNStatefulFitter",
    "EpistemicNearestNeighbors",
    "DenseLinear",
    "Optimizer",
    "arms_from_pareto_fronts",
    "calculate_sobol_indices",
    "create_optimizer_enn",
    "create_optimizer_enn_multi_tr",
    "create_optimizer_lhd",
    "create_optimizer_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "ensure_config_file",
    "hypervolume_2d_max",
    "normal_hash_batch_multi_seed_fast",
    "pareto_front_2d_maximize",
    "set_config_path",
    "sobol_sequence",
    "standardize_y",
    "subsample_loglik",
]
