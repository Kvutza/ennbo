from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrustRegionState:
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
        import numpy as np

        self.failure_tolerance = int(
            np.ceil(
                max(
                    4.0 / float(self.num_arms),
                    float(self.num_dim) / float(self.num_arms),
                )
            )
        )
        self.success_tolerance = 3

    def update(self, values) -> None:
        import numpy as np

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
        improved = np.max(new_values) > self.best_value + 1e-3 * np.abs(self.best_value)
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

    def _compute_bounds_1d(self, x_center, weights=None):
        import numpy as np

        if x_center.ndim != 1 or x_center.shape[0] != self.num_dim:
            raise ValueError(x_center.shape)
        if weights is None:
            half_length = 0.5 * self.length
            lb = np.clip(x_center - half_length, 0.0, 1.0)
            ub = np.clip(x_center + half_length, 0.0, 1.0)
        else:
            if weights.ndim != 1 or weights.shape[0] != self.num_dim:
                raise ValueError(weights.shape)
            half_length = weights * self.length / 2.0
            lb = np.clip(x_center - half_length, 0.0, 1.0)
            ub = np.clip(x_center + half_length, 0.0, 1.0)
        return lb, ub

    def create_bounds(self, x_center) -> tuple:
        if (
            x_center.ndim != 2
            or x_center.shape[0] != 1
            or x_center.shape[1] != self.num_dim
        ):
            raise ValueError(x_center.shape)
        lb_1d, ub_1d = self._compute_bounds_1d(x_center[0])
        lb = lb_1d[None, :]
        ub = ub_1d[None, :]
        return lb, ub
