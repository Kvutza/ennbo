from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gpytorch.likelihoods import GaussianLikelihood

    from .core import EpistemicNearestNeighbors
    from .turbo_gp import TurboGP
    from .turbo_mode import TurboMode


def _latin_hypercube(num_points: int, num_dim: int, *, rng) -> object:
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


def _sobol_like(num_points: int, num_dim: int, *, rng) -> object:
    from scipy.stats import qmc

    seed = int(rng.integers(1_000_000))
    engine = qmc.Sobol(d=num_dim, scramble=True, seed=seed)
    return engine.random(num_points)


def _argmax_random_tie(values, *, rng) -> int:
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


def _pareto_front(mu, se) -> object:
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


class TurboOptimizer:
    def __init__(
        self,
        bounds,
        mode,
        num_arms: int,
        *,
        num_candidates: Optional[int] = None,
        rng,
    ) -> None:
        import numpy as np
        from scipy.stats import qmc

        from .trust_region_state import _TrustRegionState

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(bounds.shape)
        self._bounds = np.asarray(bounds, dtype=float)
        self._num_dim = self._bounds.shape[0]
        self._mode = mode
        tr_num_arms = int(num_arms)
        if tr_num_arms <= 0:
            raise ValueError(tr_num_arms)
        if num_candidates is None:
            num_candidates = 100 * self._num_dim
        self._num_candidates = int(num_candidates)
        if self._num_candidates <= 0:
            raise ValueError(self._num_candidates)
        self._rng = rng
        sobol_seed = int(self._rng.integers(1_000_000))
        self._sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        self._x_obs_list: list = []
        self._y_obs_list: list = []
        self._tr_state = _TrustRegionState(num_dim=self._num_dim, num_arms=tr_num_arms)
        self._gp_y_mean: float = 0.0
        self._gp_y_std: float = 1.0
        self._gp_num_steps: int = 50
        self._enn_model: EpistemicNearestNeighbors | None = None

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def mode(self) -> TurboMode:
        return self._mode

    def ask(self, num_arms: int) -> object:
        from .turbo_mode import TurboMode

        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if len(self._x_obs_list) == 0:
            return self._draw_initial(num_arms)
        if self._tr_state.needs_restart():
            self._tr_state.restart()
        x_center = self._best_x()[None, :]
        lb_local, ub_local = self._tr_state.create_bounds(x_center)
        lb_local = lb_local[0]
        ub_local = ub_local[0]
        x_cand = self._sample_candidates(lb_local, ub_local, self._num_candidates)
        if self._mode == TurboMode.TURBO_ZERO:
            return self._select_sobol(x_cand, num_arms)
        if self._mode == TurboMode.TURBO_ONE:
            return self._select_gp_thompson(x_cand, num_arms)
        if self._mode == TurboMode.TURBO_ENN:
            return self._select_enn_pareto(x_cand, num_arms)
        raise RuntimeError(self._mode)

    def tell(self, x, y) -> None:
        import numpy as np

        from .turbo_mode import TurboMode

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        if x.shape[0] == 0:
            return
        x_unit = self._to_unit(x)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        self._tr_state.update(y_obs_array)
        if self._mode == TurboMode.TURBO_ENN:
            self._update_enn_model()

    def _fit_gp(self) -> tuple[TurboGP | None, GaussianLikelihood | None]:
        import numpy as np
        import torch
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood

        from .turbo_gp import TurboGP

        x = np.asarray(self._x_obs_list, dtype=float)
        y = np.asarray(self._y_obs_list, dtype=float)
        n = x.shape[0]
        if n == 0:
            return None, None
        if n == 1:
            self._gp_y_mean = float(y[0])
            self._gp_y_std = 1.0
            return None, None
        self._gp_y_mean = float(np.mean(y))
        y_centered = y - self._gp_y_mean
        self._gp_y_std = float(np.std(y_centered))
        if not np.isfinite(self._gp_y_std) or self._gp_y_std <= 0.0:
            self._gp_y_std = 1.0
        z = y_centered / self._gp_y_std
        train_x = torch.as_tensor(x, dtype=torch.float32)
        train_y = torch.as_tensor(z, dtype=torch.float32)
        noise_constraint = Interval(5e-4, 0.2)
        lengthscale_constraint = Interval(0.005, float(np.sqrt(self._num_dim)))
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
            ard_dims=self._num_dim,
        ).to(dtype=train_x.dtype)
        model.train()
        likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(self._gp_num_steps):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        return model, likelihood

    def _draw_initial(self, num_arms: int) -> object:
        unit = _latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return self._from_unit(unit)

    def _best_x(self) -> object:
        import numpy as np

        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        if y_obs_array.size == 0:
            raise RuntimeError("no observations")
        idx = _argmax_random_tie(y_obs_array, rng=self._rng)
        x_obs_array = np.asarray(self._x_obs_list, dtype=float)
        return x_obs_array[idx]

    def _to_unit(self, x) -> object:
        import numpy as np

        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        if np.any(ub <= lb):
            raise ValueError(self._bounds)
        return (x - lb) / (ub - lb)

    def _from_unit(self, x_unit) -> object:
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x_unit * (ub - lb)

    def _sample_candidates(self, lb, ub, num_candidates: int) -> object:
        unit = self._sobol_engine.random(num_candidates)
        return lb + unit * (ub - lb)

    def _select_sobol(self, x_cand, num_arms: int) -> object:
        if x_cand.ndim != 2 or x_cand.shape[1] != self._num_dim:
            raise ValueError(x_cand.shape)
        if x_cand.shape[0] < num_arms:
            raise ValueError((x_cand.shape[0], num_arms))
        idx = self._rng.choice(x_cand.shape[0], size=num_arms, replace=False)
        return self._from_unit(x_cand[idx])

    def _select_gp_thompson(self, x_cand, num_arms: int) -> object:
        import gpytorch
        import numpy as np
        import torch

        if len(self._x_obs_list) == 0:
            return self._select_sobol(x_cand, num_arms)
        model, _likelihood = self._fit_gp()
        if model is None:
            return self._select_sobol(x_cand, num_arms)
        x_torch = torch.as_tensor(x_cand, dtype=torch.float32)
        gen = torch.Generator(device=x_torch.device)
        seed = int(self._rng.integers(2**31 - 1))
        gen.manual_seed(seed)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(x_torch)
            numel = int(posterior.event_shape.numel())
            base = torch.randn(
                (1, numel),
                generator=gen,
                dtype=x_torch.dtype,
                device=x_torch.device,
            )
            samples = posterior.rsample(
                sample_shape=torch.Size([1]),
                base_samples=base,
            )
        ts = samples[0].reshape(-1)
        scores = ts.detach().cpu().numpy().reshape(-1)
        scores = self._gp_y_mean + self._gp_y_std * scores
        if x_cand.shape[0] < num_arms:
            raise ValueError((x_cand.shape[0], num_arms))
        idx = np.argpartition(-scores, num_arms - 1)[:num_arms]
        return self._from_unit(x_cand[idx])

    def _update_enn_model(self) -> None:
        import numpy as np

        from .core import EpistemicNearestNeighbors

        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        if y_obs_array.size == 0:
            self._enn_model = None
            return
        y = y_obs_array.reshape(-1, 1)
        yvar = np.zeros_like(y, dtype=float)
        x_obs_array = np.asarray(self._x_obs_list, dtype=float)
        self._enn_model = EpistemicNearestNeighbors(
            x_obs_array,
            y,
            yvar,
            hnsw_threshold=None,
        )

    def _select_enn_pareto(self, x_cand, num_arms: int) -> object:
        import numpy as np

        if self._enn_model is None or len(self._enn_model) == 0:
            return self._select_sobol(x_cand, num_arms)
        posterior = self._enn_model.posterior(
            x_cand,
            k=min(10, max(1, len(self._enn_model))),
            var_scale=1.0,
            exclude_nearest=False,
        )
        mu = posterior.mu[:, 0]
        se = posterior.se[:, 0]
        mask = _pareto_front(mu, se)
        idx_pareto = np.nonzero(mask)[0]
        if idx_pareto.size == 0:
            return self._select_sobol(x_cand, num_arms)
        if idx_pareto.size >= num_arms:
            chosen = self._rng.choice(idx_pareto, size=num_arms, replace=False)
        else:
            base = list(idx_pareto)
            extra = self._rng.choice(
                idx_pareto, size=num_arms - idx_pareto.size, replace=True
            )
            chosen = np.asarray(base + list(extra), dtype=int)
        return self._from_unit(x_cand[chosen])
