from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["get_gp_posterior_suppress_warning", "record_duration", "torch_seed_context"]


@contextlib.contextmanager
def record_duration(set_dt: Callable[[float], None]) -> Iterator[None]:
    import time

    t0 = time.perf_counter()  # testmon
    try:
        yield
    finally:
        set_dt(time.perf_counter() - t0)


@contextlib.contextmanager
def torch_seed_context(
    seed: int, device: torch.device | Any | None = None
) -> Iterator[None]:
    import torch

    devices: list[int] | None = None
    if device is not None and getattr(device, "type", None) == "cuda":
        idx = 0 if getattr(device, "index", None) is None else int(device.index)
        devices = [idx]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        if (
            device is not None
            and getattr(device, "type", None) == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "manual_seed")
        ):
            torch.mps.manual_seed(int(seed))
        yield


def get_gp_posterior_suppress_warning(model: Any, x_torch: Any) -> Any:
    import warnings

    try:
        from gpytorch.utils.warnings import GPInputWarning
    except ImportError:
        GPInputWarning = None
    if GPInputWarning is None:
        return model.posterior(x_torch)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The input matches the stored training data\..*",
            category=GPInputWarning,
        )
        return model.posterior(x_torch)
