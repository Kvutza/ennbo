from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .proposal import select_enn_pareto, select_gp_thompson, select_uniform
from .turbo_utils import argmax_random_tie, from_unit, latin_hypercube, raasp, to_unit

if TYPE_CHECKING:
    import numpy as np

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
        k: Optional[int] = None,
        var_scale: float = 1.0,
        trailing_obs: Optional[int] = None,
        num_init: Optional[int] = None,
    ) -> None:
        import numpy as np
        from scipy.stats import qmc

        from .trust_region_state import TrustRegionState

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(bounds.shape)
        self._bounds = np.asarray(bounds, dtype=float)
        self._num_dim = self._bounds.shape[0]
        self._mode = mode
        tr_num_arms = int(num_arms)
        if tr_num_arms <= 0:
            raise ValueError(tr_num_arms)
        if num_candidates is None:
            num_candidates = min(5000, 100 * self._num_dim)
        from .turbo_mode import TurboMode

        if mode == TurboMode.TURBO_ENN:
            num_candidates = max(num_candidates, 10 * tr_num_arms)
        self._num_candidates = int(num_candidates)
        if self._num_candidates <= 0:
            raise ValueError(self._num_candidates)
        self._rng = rng
        sobol_seed = int(self._rng.integers(1_000_000))
        self._sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        self._x_obs_list: list = []
        self._y_obs_list: list = []

        if mode == TurboMode.TURBO_ONE or mode == TurboMode.TURBO_ENN:
            self._x_tr_list: list = []
            self._y_tr_list: list = []
        else:
            self._x_tr_list = None
            self._y_tr_list = None
        self._tr_state = TrustRegionState(num_dim=self._num_dim, num_arms=tr_num_arms)
        self._gp_y_mean: float = 0.0
        self._gp_y_std: float = 1.0
        self._gp_num_steps: int = 50
        self._hnsw_threshold = hnsw_threshold
        if k is not None:
            k_val = int(k)
            if k_val < 3:
                raise ValueError(f"k must be >= 3, got {k_val}")
            self._k = k_val
        else:
            self._k = None
        var_scale_val = float(var_scale)
        if var_scale_val <= 0.0:
            raise ValueError(f"var_scale must be > 0, got {var_scale_val}")
        self._var_scale = var_scale_val
        if trailing_obs is not None:
            trailing_obs_val = int(trailing_obs)
            if trailing_obs_val <= 0:
                raise ValueError(f"trailing_obs must be > 0, got {trailing_obs_val}")
            self._trailing_obs = trailing_obs_val
        else:
            self._trailing_obs = None
        if num_init is None:
            num_init = 2 * self._num_dim
        num_init_val = int(num_init)
        if num_init_val <= 0:
            raise ValueError(f"num_init must be > 0, got {num_init_val}")
        self._num_init = num_init_val
        self._init_lhd = from_unit(
            latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
            self._bounds,
        )
        self._init_idx = 0

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def mode(self) -> TurboMode:
        return self._mode

    def ask(self, num_arms: int) -> np.ndarray:
        from .turbo_mode import TurboMode

        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._mode == TurboMode.LHD_ONLY:
            return self._draw_initial(num_arms)
        if (
            self._mode == TurboMode.TURBO_ONE
            and self._x_tr_list is not None
            and len(self._x_tr_list) == 0
        ):
            return self._get_init_lhd_points(num_arms)
        if self._init_idx < self._num_init:
            if len(self._x_obs_list) == 0:
                fallback_fn = None
            else:
                fallback_fn = self._ask_normal
            return self._get_init_lhd_points(num_arms, fallback_fn=fallback_fn)
        if len(self._x_obs_list) == 0:
            return self._draw_initial(num_arms)
        return self._ask_normal(num_arms)

    def _ask_normal(self, num_arms: int) -> np.ndarray:
        from .turbo_mode import TurboMode

        if self._tr_state.needs_restart():
            self._tr_state.restart()
            if self._mode == TurboMode.TURBO_ONE or self._mode == TurboMode.TURBO_ENN:
                self._x_tr_list.clear()
                self._y_tr_list.clear()
                self._init_idx = 0
                self._init_lhd = from_unit(
                    latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
                    self._bounds,
                )
                return self._get_init_lhd_points(num_arms)
        if (
            self._mode == TurboMode.TURBO_ONE or self._mode == TurboMode.TURBO_ENN
        ) and self._x_tr_list is not None:
            x_center = self._best_x_tr()
        else:
            x_center = self._best_x()

        def from_unit_fn(x):
            return from_unit(x, self._bounds)

        gp_model = None
        gp_y_mean_fitted = None
        gp_y_std_fitted = None
        weights = None

        if self._mode == TurboMode.TURBO_ONE or self._mode == TurboMode.TURBO_ENN:
            if len(self._x_tr_list) == 0:
                return self._get_init_lhd_points(num_arms)
            if self._trailing_obs is not None:
                x_tr_slice = self._x_tr_list[-self._trailing_obs :]
                y_tr_slice = self._y_tr_list[-self._trailing_obs :]
            else:
                x_tr_slice = self._x_tr_list
                y_tr_slice = self._y_tr_list

            if self._mode == TurboMode.TURBO_ONE:
                import numpy as np

                from .turbo_utils import fit_gp

                gp_model, _likelihood, gp_y_mean_fitted, gp_y_std_fitted = fit_gp(
                    x_tr_slice,
                    y_tr_slice,
                    self._num_dim,
                    num_steps=self._gp_num_steps,
                )
                if gp_model is not None:
                    weights = (
                        gp_model.covar_module.base_kernel.lengthscale.cpu()
                        .detach()
                        .numpy()
                        .ravel()
                    )
                    weights = weights / weights.mean()
                    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))

        lb_local, ub_local = self._tr_state._compute_bounds_1d(x_center, weights)

        x_cand = raasp(
            x_center,
            lb_local,
            ub_local,
            self._num_candidates,
            num_pert=20,
            rng=self._rng,
            sobol_engine=self._sobol_engine,
        )

        def fallback_fn(x, n):
            return select_uniform(x, n, self._num_dim, self._rng, from_unit_fn)

        if self._mode == TurboMode.TURBO_ZERO:
            return select_uniform(
                x_cand,
                num_arms,
                self._num_dim,
                self._rng,
                from_unit_fn,
            )
        if self._mode == TurboMode.TURBO_ONE:
            selected, self._gp_y_mean, self._gp_y_std, _ = select_gp_thompson(
                x_cand,
                num_arms,
                x_tr_slice,
                y_tr_slice,
                self._num_dim,
                self._gp_num_steps,
                self._rng,
                self._gp_y_mean,
                self._gp_y_std,
                fallback_fn,
                from_unit_fn,
                model=gp_model,
                new_gp_y_mean=gp_y_mean_fitted,
                new_gp_y_std=gp_y_std_fitted,
            )
            return selected
        if self._mode == TurboMode.TURBO_ENN:
            return select_enn_pareto(
                x_cand,
                num_arms,
                x_tr_slice,
                y_tr_slice,
                self._k,
                self._var_scale,
                self._hnsw_threshold,
                self._rng,
                fallback_fn,
                from_unit_fn,
            )
        raise RuntimeError(self._mode)

    def _append_observations(self, x, y) -> None:
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        if x.shape[0] == 0:
            return
        x_unit = to_unit(x, self._bounds)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        if self._x_tr_list is not None:
            self._x_tr_list.extend(x_unit.tolist())
            self._y_tr_list.extend(y.tolist())
        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        self._tr_state.update(y_obs_array)

    def tell(self, x, y) -> None:
        self._append_observations(x, y)

    def _best_x_from_lists(self, x_list, y_list, error_msg: str) -> np.ndarray:
        import numpy as np

        y_array = np.asarray(y_list, dtype=float)
        if y_array.size == 0:
            raise RuntimeError(error_msg)
        idx = argmax_random_tie(y_array, rng=self._rng)
        x_array = np.asarray(x_list, dtype=float)
        return x_array[idx]

    def _draw_initial(self, num_arms: int) -> np.ndarray:
        unit = latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return from_unit(unit, self._bounds)

    def _get_init_lhd_points(self, num_arms: int, fallback_fn=None) -> np.ndarray:
        import numpy as np

        remaining_init = self._num_init - self._init_idx
        num_to_return = min(num_arms, remaining_init)
        result = self._init_lhd[self._init_idx : self._init_idx + num_to_return]
        self._init_idx += num_to_return
        if num_to_return < num_arms:
            remaining = num_arms - num_to_return
            if fallback_fn is not None:
                result = np.vstack([result, fallback_fn(remaining)])
            else:
                result = np.vstack([result, self._draw_initial(remaining)])
        return result

    def _best_x(self) -> np.ndarray:
        return self._best_x_from_lists(
            self._x_obs_list, self._y_obs_list, "no observations"
        )

    def _best_x_tr(self) -> np.ndarray:
        return self._best_x_from_lists(
            self._x_tr_list, self._y_tr_list, "no trust-region observations"
        )
