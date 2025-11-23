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
    num_obs = len(model)
    max_k = min(100, max(3, num_obs))
    num_k = int(np.ceil(np.sqrt(float(num_fit_candidates))))
    num_var_scale = int(np.ceil(float(num_fit_candidates) / float(num_k)))
    k_log_min = np.log10(3.0)
    k_log_max = np.log10(float(max_k))
    k_log_values = np.linspace(k_log_min, k_log_max, num=num_k)
    k_values = [int(round(10**v)) for v in k_log_values]
    k_values = [k for k in k_values if 3 <= k <= max_k]
    k_values = sorted(set(k_values))
    if not k_values:
        k_values = [3]
    var_scale_log_min = -3.0
    var_scale_log_max = 3.0
    var_scale_log_values = np.linspace(
        var_scale_log_min, var_scale_log_max, num=num_var_scale
    )
    var_scale_values = [10**v for v in var_scale_log_values]
    pairs = []
    for k in k_values:
        for var_scale in var_scale_values:
            pairs.append((k, var_scale))
    if len(pairs) > num_fit_candidates:
        indices = rng.choice(len(pairs), size=num_fit_candidates, replace=False)
        pairs = [pairs[i] for i in indices]
    if len(pairs) == 0:
        return {"k": float(k_values[0]), "var_scale": float(var_scale_values[0])}
    paramss = [ENNParams(k=k, var_scale=var_scale) for k, var_scale in pairs]
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
        return {"k": float(pairs[0][0]), "var_scale": float(pairs[0][1])}
    return {"k": float(pairs[best_idx][0]), "var_scale": float(pairs[best_idx][1])}
