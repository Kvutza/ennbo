from __future__ import annotations

import numpy as np

from ennx.turbo.rust_optimizer import (
    RustOptimizer,
    _ObsView,
    create_optimizer,
)


def test_rust_optimizer_kiss_surface_has_view_and_factory():
    from ennx.turbo import rust_optimizer as ro

    assert RustOptimizer.__init__ is not None
    assert create_optimizer is ro.create_optimizer
    v = _ObsView(np.array([[0.0]]))
    assert v.view().shape == (1, 1)
    bounds = np.array([[0.0, 1.0]], dtype=float)
    cfg = __import__(
        "ennx.turbo.config", fromlist=["turbo_zero_config"]
    ).turbo_zero_config(num_init=1)
    create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(0))


def test_failure_tolerance_dimension_override_reaches_rust_optimizer():
    from ennx import _rust

    num_dim = 100
    bounds = np.column_stack((np.zeros(num_dim), np.ones(num_dim)))
    optimizer = _rust.create_optimizer_enn(
        bounds,
        2,
        1,
        7,
        config_overrides={
            "failure_tolerance_dim": 1.0,
            "min_candidates": 8,
            "max_candidates": 8,
        },
    )

    initial = optimizer.ask(1, 11)
    optimizer.tell(initial, np.array([[1.0]]), 12)
    assert optimizer.tr_length() == 0.8

    # The first post-init tell initializes upstream's O(batch) trust-region
    # history; the following four failures hit max(4 / 1, 1 / 1) = 4.
    for i in range(5):
        candidate = optimizer.ask(1, 20 + i)
        optimizer.tell(candidate, np.array([[0.0]]), 30 + i)

    assert optimizer.tr_length() == 0.4
