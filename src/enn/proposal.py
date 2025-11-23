from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import numpy as np


def select_enn_pareto(
    x_cand: np.ndarray,
    num_arms: int,
    x_obs_list: list,
    y_obs_list: list,
    k: Optional[int],
    var_scale: float,
    hnsw_threshold: Optional[int],
    rng,
    fallback_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    import numpy as np

    from .core import EpistemicNearestNeighbors
    from .enn_params import ENNParams
    from .turbo_utils import pareto_front

    if len(x_obs_list) == 0:
        return fallback_fn(x_cand, num_arms)
    y_obs_array = np.asarray(y_obs_list, dtype=float)
    if y_obs_array.size == 0:
        return fallback_fn(x_cand, num_arms)
    y = y_obs_array.reshape(-1, 1)
    yvar = np.zeros_like(y, dtype=float)
    x_obs_array = np.asarray(x_obs_list, dtype=float)
    enn_model = EpistemicNearestNeighbors(
        x_obs_array,
        y,
        yvar,
        hnsw_threshold=hnsw_threshold,
    )
    if len(enn_model) == 0:
        return fallback_fn(x_cand, num_arms)
    if k is None:
        from .fit import enn_fit

        n_obs = len(enn_model)
        if n_obs < 3:
            return fallback_fn(x_cand, num_arms)
        best_params = enn_fit(
            enn_model,
            num_fit_candidates=30,
            num_fit_samples=10,
            rng=rng,
        )
        k = best_params.k
        var_scale = best_params.var_scale
    params = ENNParams(k=k, var_scale=var_scale)
    posterior = enn_model.posterior(x_cand, params=params)
    mu = posterior.mu[:, 0]
    se = posterior.se[:, 0]
    remaining_idx = np.arange(mu.size, dtype=int)
    chosen_list = []
    while len(chosen_list) < num_arms and remaining_idx.size > 0:
        mu_remaining = mu[remaining_idx]
        se_remaining = se[remaining_idx]
        mask = pareto_front(mu_remaining, se_remaining)
        idx_front = np.sort(remaining_idx[mask])
        if idx_front.size == 0:
            break
        needed = num_arms - len(chosen_list)
        if idx_front.size <= needed:
            chosen_list.extend(idx_front.tolist())
        else:
            selected = rng.choice(idx_front, size=needed, replace=False)
            chosen_list.extend(selected.tolist())
        remaining_idx = remaining_idx[~mask]
    if len(chosen_list) == 0:
        return fallback_fn(x_cand, num_arms)
    chosen = np.asarray(chosen_list[:num_arms], dtype=int)
    return from_unit_fn(x_cand[chosen])


def select_uniform(
    x_cand: np.ndarray,
    num_arms: int,
    num_dim: int,
    rng,
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if x_cand.ndim != 2 or x_cand.shape[1] != num_dim:
        raise ValueError(x_cand.shape)
    if x_cand.shape[0] < num_arms:
        raise ValueError((x_cand.shape[0], num_arms))
    idx = rng.choice(x_cand.shape[0], size=num_arms, replace=False)
    return from_unit_fn(x_cand[idx])


def select_gp_thompson(
    x_cand: np.ndarray,
    num_arms: int,
    x_obs_list: list,
    y_obs_list: list,
    num_dim: int,
    gp_num_steps: int,
    rng,
    gp_y_mean: float,
    gp_y_std: float,
    select_sobol_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, float, float]:
    import contextlib

    import gpytorch
    import numpy as np
    import torch

    from .turbo_utils import fit_gp

    @contextlib.contextmanager
    def _torch_rng_context(generator):
        old_state = torch.get_rng_state()
        try:
            torch.set_rng_state(generator.get_state())
            yield
        finally:
            torch.set_rng_state(old_state)

    if len(x_obs_list) == 0:
        return select_sobol_fn(x_cand, num_arms), gp_y_mean, gp_y_std
    model, _likelihood, new_gp_y_mean, new_gp_y_std = fit_gp(
        x_obs_list,
        y_obs_list,
        num_dim,
        num_steps=gp_num_steps,
    )
    if model is None:
        return select_sobol_fn(x_cand, num_arms), gp_y_mean, gp_y_std
    x_torch = torch.as_tensor(x_cand, dtype=torch.float32)
    seed = int(rng.integers(2**31 - 1))
    gen = torch.Generator(device=x_torch.device)
    gen.manual_seed(seed)
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        _torch_rng_context(gen),
    ):
        posterior = model.posterior(x_torch)
        samples = posterior.sample(
            sample_shape=torch.Size([1]),
        )
    ts = samples[0].reshape(-1)
    scores = ts.detach().cpu().numpy().reshape(-1)
    scores = new_gp_y_mean + new_gp_y_std * scores
    if x_cand.shape[0] < num_arms:
        raise ValueError((x_cand.shape[0], num_arms))
    idx = np.argpartition(-scores, num_arms - 1)[:num_arms]
    return from_unit_fn(x_cand[idx]), new_gp_y_mean, new_gp_y_std
