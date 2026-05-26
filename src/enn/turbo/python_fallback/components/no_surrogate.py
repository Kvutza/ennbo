from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .posterior_result import PosteriorResult
from .surrogate_result import SurrogateResult

if TYPE_CHECKING:
    from numpy.random import Generator


class NoSurrogate:
    def __init__(self) -> None:
        self._x_obs: np.ndarray | None = None
        self._y_obs: np.ndarray | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return None

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: Generator | None = None,
    ) -> SurrogateResult:
        del y_var, num_steps, rng
        self._x_obs = np.asarray(x_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        self._y_obs = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
        return SurrogateResult(model=None, lengthscales=None)

    def predict(self, x: np.ndarray) -> PosteriorResult:
        if self._x_obs is None or self._y_obs is None:
            raise RuntimeError("NoSurrogate.predict requires fit() first")
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self._x_obs.shape[0] or not np.allclose(x, self._x_obs):
            raise RuntimeError("NoSurrogate.predict only works for training points")
        return PosteriorResult(mu=self._y_obs.copy(), sigma=None)

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: Generator,
    ) -> np.ndarray:
        del rng
        if self._y_obs is None:
            raise RuntimeError("NoSurrogate.sample requires fit() first")
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        m = self._y_obs.shape[1] if self._y_obs.ndim == 2 else 1
        return np.broadcast_to(self._y_obs.reshape(1, n, m), (num_samples, n, m)).copy()
