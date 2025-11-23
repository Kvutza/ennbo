from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpytorch.likelihoods import GaussianLikelihood

    from .turbo_gp import TurboGP


def fit_gp(
    x_obs_list: list,
    y_obs_list: list,
    num_dim: int,
    *,
    num_steps: int = 50,
) -> tuple[
    "TurboGP | None",
    "GaussianLikelihood | None",
    float,
    float,
]:
    import numpy as np
    import torch
    from gpytorch.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from .turbo_gp import TurboGP

    x = np.asarray(x_obs_list, dtype=float)
    y = np.asarray(y_obs_list, dtype=float)
    n = x.shape[0]
    if n == 0:
        return None, None, 0.0, 1.0
    if n == 1:
        gp_y_mean = float(y[0])
        gp_y_std = 1.0
        return None, None, gp_y_mean, gp_y_std
    gp_y_mean = float(np.mean(y))
    y_centered = y - gp_y_mean
    gp_y_std = float(np.std(y_centered))
    if not np.isfinite(gp_y_std) or gp_y_std <= 0.0:
        gp_y_std = 1.0
    z = y_centered / gp_y_std
    train_x = torch.as_tensor(x, dtype=torch.float32)
    train_y = torch.as_tensor(z, dtype=torch.float32)
    noise_constraint = Interval(5e-4, 0.2)
    lengthscale_constraint = Interval(0.005, float(np.sqrt(num_dim)))
    outputscale_constraint = Interval(0.05, 20.0)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        dtype=train_y.dtype
    )
    model = TurboGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=num_dim,
    ).to(dtype=train_x.dtype)
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood, gp_y_mean, gp_y_std


def latin_hypercube(num_points: int, num_dim: int, *, rng) -> object:
    import numpy as np

    cut = np.linspace(0.0, 1.0, num_points + 1)
    a = cut[:num_points]
    b = cut[1 : num_points + 1]
    rdpoints = np.zeros((num_points, num_dim))
    for j in range(num_dim):
        u = rng.uniform(size=num_points)
        rdpoints[:, j] = u * (b - a) + a
        rng.shuffle(rdpoints[:, j])
    return rdpoints


def argmax_random_tie(values, *, rng) -> int:
    import numpy as np

    if values.ndim != 1:
        raise ValueError(values.shape)
    max_val = float(np.max(values))
    idx = np.nonzero(values >= max_val)[0]
    if idx.size == 0:
        return int(rng.integers(values.size))
    if idx.size == 1:
        return int(idx[0])
    j = int(rng.integers(idx.size))
    return int(idx[j])


def pareto_front(mu, se) -> object:
    import numpy as np

    if mu.shape != se.shape or mu.ndim != 1:
        raise ValueError((mu.shape, se.shape))
    n = mu.size
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.argsort(-mu)
    mu_sorted = mu[order]
    se_sorted = se[order]
    is_pareto_sorted = np.zeros_like(mu_sorted, dtype=bool)
    best_se = float("inf")
    for i in range(n):
        if se_sorted[i] < best_se:
            best_se = float(se_sorted[i])
            is_pareto_sorted[i] = True
    is_pareto = np.zeros_like(is_pareto_sorted, dtype=bool)
    is_pareto[order] = is_pareto_sorted
    return is_pareto
