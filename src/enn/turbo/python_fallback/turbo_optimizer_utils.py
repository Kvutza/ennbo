from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import TellInputs

if TYPE_CHECKING:
    import numpy as np


def sobol_seed_for_state(
    seed_base: int, *, restart_generation: int, n_obs: int, num_arms: int
) -> int:
    mask64 = (1 << 64) - 1
    x = int(seed_base) & mask64
    x ^= (int(restart_generation) + 1) * 0xD1342543DE82EF95 & mask64
    x ^= (int(n_obs) + 1) * 0x9E3779B97F4A7C15 & mask64
    x ^= (int(num_arms) + 1) * 0xBF58476D1CE4E5B9 & mask64
    x = (x + 0x9E3779B97F4A7C15) & mask64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & mask64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & mask64
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFF)


def reset_timing(opt: object) -> None:
    setattr(opt, "_dt_fit", 0.0)
    setattr(opt, "_dt_gen", 0.0)
    setattr(opt, "_dt_sel", 0.0)


def validate_tell_inputs(
    x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None, num_dim: int
) -> TellInputs:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != num_dim:
        raise ValueError(x.shape)
    if y.ndim == 2:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = y.shape[1]
    elif y.ndim == 1:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = 1
    else:
        raise ValueError(y.shape)
    if y_var is not None:
        y_var = np.asarray(y_var, dtype=float)
        if y_var.shape != y.shape:
            raise ValueError((y.shape, y_var.shape))
    return TellInputs(x=x, y=y, y_var=y_var, num_metrics=num_metrics)
