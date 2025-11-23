from __future__ import annotations


def subsample_loglik(
    model,
    x,
    y,
    *,
    paramss: list,
    P: int = 10,
    rng,
) -> list:
    import numpy as np

    if x.ndim != 2:
        raise ValueError(x.shape)
    if y.ndim != 1:
        raise ValueError(y.shape)
    if x.shape[0] != y.shape[0]:
        raise ValueError((x.shape, y.shape))
    if P <= 0:
        raise ValueError(P)
    if len(paramss) == 0:
        raise ValueError("paramss must be non-empty")
    n = x.shape[0]
    if n == 0:
        return [0.0] * len(paramss)
    if len(model) <= 1:
        return [0.0] * len(paramss)
    P_actual = min(P, n)
    if P_actual == n:
        indices = np.arange(n, dtype=int)
    else:
        indices = rng.permutation(n)[:P_actual]
    x_selected = x[indices]
    y_selected = y[indices]
    if not np.isfinite(y_selected).all():
        return [0.0] * len(paramss)
    post_batch = model.batch_posterior(x_selected, paramss, exclude_nearest=True)
    mu_batch = post_batch.mu
    se_batch = post_batch.se
    if mu_batch.shape[2] == 1:
        mu_batch = mu_batch[:, :, 0]
        se_batch = se_batch[:, :, 0]
    num_params = len(paramss)
    if mu_batch.shape != (num_params, P_actual) or se_batch.shape != (
        num_params,
        P_actual,
    ):
        raise ValueError((mu_batch.shape, se_batch.shape, (num_params, P_actual)))
    y_std = float(np.std(y))
    if not np.isfinite(y_std) or y_std <= 0.0:
        y_std = 1.0
    y_scaled = y_selected / y_std
    mu_scaled = mu_batch / y_std
    se_scaled = se_batch / y_std
    result = []
    for i in range(num_params):
        mu_i = mu_scaled[i]
        se_i = se_scaled[i]
        if not np.isfinite(mu_i).all() or not np.isfinite(se_i).all():
            result.append(0.0)
            continue
        if np.any(se_i <= 0.0):
            result.append(0.0)
            continue
        diff = y_scaled - mu_i
        var_scaled = se_i**2
        log_term = np.log(2.0 * np.pi * var_scaled)
        quad = diff**2 / var_scaled
        loglik = -0.5 * np.sum(log_term + quad)
        if not np.isfinite(loglik):
            result.append(0.0)
            continue
        result.append(float(loglik))
    return result


def enn_fit(
    model,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng,
) -> dict[str, float]:
    import numpy as np

    from .enn_params import ENNParams

    train_x = model.train_x
    train_y = model.train_y
    train_yvar = model.train_yvar
    if train_y.shape[1] != 1 or train_yvar.shape[1] != 1:
        raise ValueError((train_y.shape, train_yvar.shape))
    y = train_y[:, 0]
    var_scale_log_min = -3.0
    var_scale_log_max = 3.0
    var_scale_log_values = np.linspace(
        var_scale_log_min, var_scale_log_max, num=num_fit_candidates
    )
    var_scale_values = [10**v for v in var_scale_log_values]
    if len(var_scale_values) == 0:
        return {"var_scale": 1.0}
    paramss = [ENNParams(k=k, var_scale=var_scale) for var_scale in var_scale_values]
    logliks = subsample_loglik(
        model, train_x, y, paramss=paramss, P=num_fit_samples, rng=rng
    )
    best_idx: int | None = None
    best_mll: float | None = None
    for i, loglik in enumerate(logliks):
        if best_mll is None or loglik > best_mll:
            best_mll = loglik
            best_idx = i
    if best_idx is None:
        return {"var_scale": float(var_scale_values[0])}
    return {"var_scale": float(var_scale_values[best_idx])}
