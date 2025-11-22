from __future__ import annotations

from typing import Iterable

import numpy as np

from .core import EpistemicNearestNeighbors


def subsample_loglik(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    yvar: np.ndarray | None = None,
    *,
    k: int,
    var_scale: float,
    P: int = 10,
    rng: np.random.Generator,
) -> float:
    if x.ndim != 2:
        raise ValueError(x.shape)
    if y.ndim != 1:
        raise ValueError(y.shape)
    if x.shape[0] != y.shape[0]:
        raise ValueError((x.shape, y.shape))
    if yvar is not None and yvar.shape != y.shape:
        raise ValueError((y.shape, yvar.shape))
    if P <= 0:
        raise ValueError(P)
    n = x.shape[0]
    if n == 0:
        return 0.0
    if len(model) <= 1:
        return 0.0
    P_actual = min(P, n)
    if P_actual == n:
        indices = np.arange(n, dtype=int)
    else:
        indices = rng.permutation(n)[:P_actual]
    x_selected = x[indices]
    y_selected = y[indices]
    if not np.isfinite(y_selected).all():
        return 0.0
    posterior = model.posterior(
        x_selected, k=k, var_scale=var_scale, exclude_nearest=True
    )
    mu = posterior.mu
    se = posterior.se
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]
        se = se[:, 0]
    if (
        mu.ndim != 1
        or se.ndim != 1
        or mu.shape != y_selected.shape
        or se.shape != y_selected.shape
    ):
        raise ValueError((mu.shape, se.shape, y_selected.shape))
    if not np.isfinite(mu).all() or not np.isfinite(se).all():
        return 0.0
    if np.any(se <= 0.0):
        return 0.0
    y_std = float(np.std(y))
    if not np.isfinite(y_std) or y_std <= 0.0:
        y_std = 1.0
    y_scaled = y_selected / y_std
    mu_scaled = mu / y_std
    se_scaled = se / y_std
    if not np.isfinite(se_scaled).all() or np.any(se_scaled <= 0.0):
        return 0.0
    diff = y_scaled - mu_scaled
    var_scaled = se_scaled**2
    log_term = np.log(2.0 * np.pi * var_scaled)
    quad = diff**2 / var_scaled
    loglik = -0.5 * np.sum(log_term + quad)
    if not np.isfinite(loglik):
        return 0.0
    return float(loglik)


def enn_fit(
    model: EpistemicNearestNeighbors,
    k_values: Iterable[float] | None = None,
    var_scale_values: Iterable[float] | None = None,
    *,
    num_iterations: int = 1,
    P: int = 10,
    rng: np.random.Generator,
) -> dict[str, float]:
    train_x = model.train_x
    train_y = model.train_y
    train_yvar = model.train_yvar
    if train_y.shape[1] != 1 or train_yvar.shape[1] != 1:
        raise ValueError((train_y.shape, train_yvar.shape))
    y = train_y[:, 0]
    yvar = train_yvar[:, 0]
    num_obs = len(model)
    max_k = min(100, max(1, num_obs))
    if k_values is None:
        if max_k == 1:
            k_values = [1.0]
        else:
            k_values = np.logspace(0.0, np.log10(float(max_k)), num=30)
    k_list = []
    for v in k_values:
        i = int(round(float(v)))
        if 3 <= i <= max_k:
            if i not in k_list:
                k_list.append(i)
    if not k_list:
        k_list = [1]
    if var_scale_values is None:
        var_scale_values = np.logspace(-4.0, 3.0, num=30)
    var_scale_list = [float(v) for v in var_scale_values]
    best_k: int | None = None
    best_var_scale: float | None = None
    best_k_mll: float | None = None
    for k in k_list:
        value = subsample_loglik(
            model,
            train_x,
            y,
            yvar,
            k=k,
            var_scale=float(np.median(var_scale_list)),
            P=P,
            rng=rng,
        )
        if best_k_mll is None or value > best_k_mll:
            best_k_mll = value
            best_k = k
    if best_k is None:
        best_k = k_list[0]
    best_var_scale_mll: float | None = None
    for var_scale in var_scale_list:
        value = subsample_loglik(
            model,
            train_x,
            y,
            yvar,
            k=best_k,
            var_scale=var_scale,
            P=P,
            rng=rng,
        )
        if best_var_scale_mll is None or value > best_var_scale_mll:
            best_var_scale_mll = value
            best_var_scale = var_scale
    if best_var_scale is None:
        best_var_scale = var_scale_list[0]
    return {"k": float(best_k), "var_scale": float(best_var_scale)}
