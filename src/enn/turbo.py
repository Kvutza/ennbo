from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.stats import qmc

from .core import EpistemicNearestNeighbors
from .enn_normal import ENNNormal


class TurboMode(Enum):
    TURBO_ONE = auto()
    TURBO_ZERO = auto()
    TURBO_ENN = auto()


@dataclass
class _TrustRegionState:
    num_dim: int
    num_arms: int
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    prev_num_obs: int = 0

    def __post_init__(self) -> None:
        self.failure_tolerance = int(
            np.ceil(
                max(
                    4.0 / float(self.num_arms),
                    float(self.num_dim) / float(self.num_arms),
                )
            )
        )
        self.success_tolerance = 3

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 1:
            raise ValueError(values.shape)
        if values.size == 0:
            return
        new_values = values[self.prev_num_obs :]
        if new_values.size == 0:
            return
        if not np.isfinite(self.best_value):
            self.best_value = float(np.max(new_values))
            self.prev_num_obs = values.size
            return
        improved = np.max(new_values) > self.best_value
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.length = 0.5 * self.length
            self.failure_counter = 0
        self.best_value = max(self.best_value, float(np.max(new_values)))
        self.prev_num_obs = values.size

    def needs_restart(self) -> bool:
        return self.length < self.length_min

    def restart(self) -> None:
        self.length = self.length_init
        self.failure_counter = 0
        self.success_counter = 0
        self.best_value = -float("inf")
        self.prev_num_obs = 0

    def create_bounds(self, x_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            x_center.ndim != 2
            or x_center.shape[0] != 1
            or x_center.shape[1] != self.num_dim
        ):
            raise ValueError(x_center.shape)
        lb = np.clip(x_center - 0.5 * self.length, 0.0, 1.0)
        ub = np.clip(x_center + 0.5 * self.length, 0.0, 1.0)
        return lb, ub


def _latin_hypercube(
    num_points: int, num_dim: int, *, rng: np.random.Generator
) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, num_points + 1)
    u = rng.uniform(size=(num_points, num_dim))
    a = cut[:num_points]
    b = cut[1 : num_points + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    for j in range(num_dim):
        rng.shuffle(rdpoints[:, j])
    return rdpoints


def _sobol_like(
    num_points: int, num_dim: int, *, rng: np.random.Generator
) -> np.ndarray:
    seed = int(rng.integers(1_000_000))
    engine = qmc.Sobol(d=num_dim, scramble=True, seed=seed)
    return engine.random(num_points)


def _argmax_random_tie(values: np.ndarray, *, rng: np.random.Generator) -> int:
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


def _pareto_front(mu: np.ndarray, se: np.ndarray) -> np.ndarray:
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
        bounds: np.ndarray,
        mode: TurboMode,
        num_arms: int,
        *,
        num_candidates: Optional[int] = None,
        rng: np.random.Generator,
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(bounds.shape)
        self._bounds = np.asarray(bounds, dtype=float)
        self._num_dim = self._bounds.shape[0]
        self._mode = mode
        self._num_arms = int(num_arms)
        if self._num_arms <= 0:
            raise ValueError(self._num_arms)
        if num_candidates is None:
            num_candidates = 100 * self._num_dim
        self._num_candidates = int(num_candidates)
        if self._num_candidates <= 0:
            raise ValueError(self._num_candidates)
        self._rng = rng
        self._x_obs = np.zeros((0, self._num_dim), dtype=float)
        self._y_obs = np.zeros((0,), dtype=float)
        self._tr_state = _TrustRegionState(
            num_dim=self._num_dim, num_arms=self._num_arms
        )
        self._gp_alpha: np.ndarray | None = None
        self._gp_train_x: np.ndarray | None = None
        self._gp_noise: float = 1e-6
        self._gp_y_mean: float = 0.0
        self._gp_y_std: float = 1.0
        self._enn_model: EpistemicNearestNeighbors | None = None

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def mode(self) -> TurboMode:
        return self._mode

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._x_obs.shape[0] == 0:
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

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        if x.shape[0] == 0:
            return
        x_unit = self._to_unit(x)
        self._x_obs = np.vstack([self._x_obs, x_unit])
        self._y_obs = np.concatenate([self._y_obs, y])
        self._tr_state.update(self._y_obs)
        if self._mode == TurboMode.TURBO_ENN:
            self._update_enn_model()

    def _fit_gp(self) -> None:
        x = self._x_obs
        y = self._y_obs
        n = x.shape[0]
        if n == 0:
            raise RuntimeError("no observations")
        if n == 1:
            self._gp_y_mean = float(y[0])
            self._gp_y_std = 1.0
            self._gp_train_x = x.copy()
            self._gp_alpha = np.array([0.0], dtype=float)
            return
        self._gp_y_mean = float(np.mean(y))
        y_centered = y - self._gp_y_mean
        self._gp_y_std = float(np.std(y_centered))
        if not np.isfinite(self._gp_y_std) or self._gp_y_std <= 0.0:
            self._gp_y_std = 1.0
        z = y_centered / self._gp_y_std
        lengthscale = 0.2 * np.ones(self._num_dim, dtype=float)
        diff = x[:, None, :] - x[None, :, :]
        dist2 = np.sum((diff / lengthscale) ** 2, axis=-1)
        kxx = np.exp(-0.5 * dist2)
        kxx = kxx + self._gp_noise * np.eye(n, dtype=float)
        alpha = np.linalg.solve(kxx, z)
        self._gp_train_x = x.copy()
        self._gp_alpha = alpha

    def _gp_posterior(self, x_cand: np.ndarray) -> ENNNormal:
        if self._gp_train_x is None or self._gp_alpha is None:
            raise RuntimeError("gp model is not fitted")
        train_x = self._gp_train_x
        alpha = self._gp_alpha
        lengthscale = 0.2 * np.ones(self._num_dim, dtype=float)
        diff = x_cand[:, None, :] - train_x[None, :, :]
        dist2 = np.sum((diff / lengthscale) ** 2, axis=-1)
        k_x = np.exp(-0.5 * dist2)
        mu = k_x @ alpha
        k_xx = np.ones(x_cand.shape[0], dtype=float)
        diff_train = train_x[:, None, :] - train_x[None, :, :]
        dist2_train = np.sum((diff_train / lengthscale) ** 2, axis=-1)
        k_train = np.exp(-0.5 * dist2_train)
        k_train = k_train + self._gp_noise * np.eye(train_x.shape[0], dtype=float)
        v = np.linalg.solve(k_train, k_x.T)
        var = k_xx - np.sum(k_x * v.T, axis=1)
        var = np.maximum(var, 1e-9)
        se = np.sqrt(var)
        return ENNNormal(mu=mu[:, None], se=se[:, None])

    def _draw_initial(self, num_arms: int) -> np.ndarray:
        unit = _latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return self._from_unit(unit)

    def _best_x(self) -> np.ndarray:
        if self._y_obs.size == 0:
            raise RuntimeError("no observations")
        idx = _argmax_random_tie(self._y_obs, rng=self._rng)
        return self._x_obs[idx]

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        if np.any(ub <= lb):
            raise ValueError(self._bounds)
        return (x - lb) / (ub - lb)

    def _from_unit(self, x_unit: np.ndarray) -> np.ndarray:
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x_unit * (ub - lb)

    def _sample_candidates(
        self, lb: np.ndarray, ub: np.ndarray, num_candidates: int
    ) -> np.ndarray:
        unit = _sobol_like(num_candidates, self._num_dim, rng=self._rng)
        return lb + unit * (ub - lb)

    def _select_sobol(self, x_cand: np.ndarray, num_arms: int) -> np.ndarray:
        if x_cand.ndim != 2 or x_cand.shape[1] != self._num_dim:
            raise ValueError(x_cand.shape)
        if x_cand.shape[0] < num_arms:
            raise ValueError((x_cand.shape[0], num_arms))
        idx = self._rng.choice(x_cand.shape[0], size=num_arms, replace=False)
        return self._from_unit(x_cand[idx])

    def _select_gp_thompson(self, x_cand: np.ndarray, num_arms: int) -> np.ndarray:
        if self._x_obs.shape[0] == 0:
            return self._select_sobol(x_cand, num_arms)
        if self._gp_alpha is None or self._gp_train_x is None:
            self._fit_gp()
        posterior = self._gp_posterior(x_cand)
        samples = posterior.sample(1, rng=self._rng)
        scores = samples[:, 0, 0]
        if x_cand.shape[0] < num_arms:
            raise ValueError((x_cand.shape[0], num_arms))
        idx = np.argpartition(-scores, num_arms - 1)[:num_arms]
        return self._from_unit(x_cand[idx])

    def _update_enn_model(self) -> None:
        if self._y_obs.size == 0:
            self._enn_model = None
            return
        y = self._y_obs.reshape(-1, 1)
        yvar = np.zeros_like(y, dtype=float)
        self._enn_model = EpistemicNearestNeighbors(
            self._x_obs,
            y,
            yvar,
            hnsw_threshold=None,
        )

    def _select_enn_pareto(self, x_cand: np.ndarray, num_arms: int) -> np.ndarray:
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
