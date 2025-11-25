from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class GumbelTrustRegion:
    num_dim: int
    length: float = 1.0

    def update(self, values: np.ndarray | Any) -> None:
        import numpy as np

        from .enn_util import gumbel_expected_max

        n = len(values)
        if n <= 1:
            self.length = 1.0
            return
        y_max = float(np.max(values))
        y_median = float(np.median(values))
        y_std = float(np.std(values))
        denom = 2.0 * gumbel_expected_max(n)
        if denom <= 0:
            denom = 1.0
        signal = ((y_max - y_median) / (1e-6 + y_std) / denom) ** 2
        scale = 1.0 / (1e-6 + signal)
        self.length = float(np.clip(scale, 0.1, 1.0))

    def needs_restart(self) -> bool:
        return False

    def restart(self) -> None:
        pass

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        length = self.length
        if weights is None:
            half_length = 0.5 * length
        else:
            half_length = weights * length / 2.0
        lb = np.clip(x_center - half_length, 0.0, 1.0)
        ub = np.clip(x_center + half_length, 0.0, 1.0)
        return lb, ub
