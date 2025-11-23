from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .turbo_utils import argmax_random_tie, fit_gp, latin_hypercube, pareto_front

if TYPE_CHECKING:
    from .core import EpistemicNearestNeighbors
    from .turbo_mode import TurboMode


class TurboOptimizer:
    def __init__(
        self,
        bounds,
        mode,
        num_arms: int,
        *,
        num_candidates: Optional[int] = None,
        rng,
        hnsw_threshold: Optional[int] = None,
        num_fit_samples: int = 10,
        num_fit_candidates: int = 100,
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
        self._hnsw_threshold = hnsw_threshold
        self._num_fit_samples = num_fit_samples
        self._num_fit_candidates = num_fit_candidates

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

    def _draw_initial(self, num_arms: int) -> object:
        unit = latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return self._from_unit(unit)

    def _best_x(self) -> object:
        import numpy as np

        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        if y_obs_array.size == 0:
            raise RuntimeError("no observations")
        idx = argmax_random_tie(y_obs_array, rng=self._rng)
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
        model, _likelihood, self._gp_y_mean, self._gp_y_std = fit_gp(
            self._x_obs_list,
            self._y_obs_list,
            self._num_dim,
            num_steps=self._gp_num_steps,
        )
        if model is None:
            return self._select_sobol(x_cand, num_arms)
        x_torch = torch.as_tensor(x_cand, dtype=torch.float32)
        seed = int(self._rng.integers(2**31 - 1))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gen = torch.Generator(device=x_torch.device)
            gen.manual_seed(seed)
            old_state = torch.get_rng_state()
            torch.set_rng_state(gen.get_state())
            posterior = model.posterior(x_torch)
            samples = posterior.sample(
                sample_shape=torch.Size([1]),
            )
            torch.set_rng_state(old_state)
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
            hnsw_threshold=self._hnsw_threshold,
        )

    def _select_enn_pareto(self, x_cand, num_arms: int) -> object:
        import numpy as np

        from .enn_params import ENNParams
        from .fit import enn_fit

        if self._enn_model is None or len(self._enn_model) == 0:
            return self._select_sobol(x_cand, num_arms)
        result = enn_fit(
            self._enn_model,
            num_fit_candidates=self._num_fit_candidates,
            num_fit_samples=self._num_fit_samples,
            rng=self._rng,
        )
        k = int(result["k"])
        var_scale = float(result["var_scale"])
        params = ENNParams(k=k, var_scale=var_scale)
        posterior = self._enn_model.posterior(x_cand, params=params)
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
                selected = self._rng.choice(idx_front, size=needed, replace=False)
                chosen_list.extend(selected.tolist())
            remaining_idx = remaining_idx[~mask]
        if len(chosen_list) == 0:
            return self._select_sobol(x_cand, num_arms)
        chosen = np.asarray(chosen_list[:num_arms], dtype=int)
        return self._from_unit(x_cand[chosen])
