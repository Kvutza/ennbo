from __future__ import annotations


def subsample_loglik(
    model,
    x,
    y,
    *,
    k: int,
    var_scale: float,
    P: int = 10,
    rng,
) -> float:
    import numpy as np

    if x.ndim != 2:
        raise ValueError(x.shape)
    if y.ndim != 1:
        raise ValueError(y.shape)
    if x.shape[0] != y.shape[0]:
        raise ValueError((x.shape, y.shape))
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
    model,
    *,
    num_tries: int,
    P: int = 10,
    rng,
) -> dict[str, float]:
    import numpy as np

    train_x = model.train_x
    train_y = model.train_y
    train_yvar = model.train_yvar
    if train_y.shape[1] != 1 or train_yvar.shape[1] != 1:
        raise ValueError((train_y.shape, train_yvar.shape))
    y = train_y[:, 0]
    num_obs = len(model)
    max_k = min(100, max(3, num_obs))
    num_k = int(np.ceil(np.sqrt(float(num_tries))))
    num_var_scale = int(np.ceil(float(num_tries) / float(num_k)))
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
    if len(pairs) > num_tries:
        indices = rng.choice(len(pairs), size=num_tries, replace=False)
        pairs = [pairs[i] for i in indices]
    best_k: int | None = None
    best_var_scale: float | None = None
    best_mll: float | None = None
    for k, var_scale in pairs:
        value = subsample_loglik(
            model,
            train_x,
            y,
            k=k,
            var_scale=var_scale,
            P=P,
            rng=rng,
        )
        if best_mll is None or value > best_mll:
            best_mll = value
            best_k = k
            best_var_scale = var_scale
    if best_k is None:
        best_k = k_values[0]
    if best_var_scale is None:
        best_var_scale = var_scale_values[0]
    return {"k": float(best_k), "var_scale": float(best_var_scale)}
