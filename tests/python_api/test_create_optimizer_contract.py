from __future__ import annotations

import inspect

import numpy as np
import pytest

from ennx import Telemetry, create_optimizer, turbo_enn_config, turbo_zero_config
from ennx.turbo.config import lhd_only_config
from ennx.turbo.optimizer import Optimizer


def test_factory_signature():
    assert set(inspect.signature(create_optimizer).parameters) == {
        "bounds",
        "config",
        "rng",
    }


@pytest.mark.parametrize(
    "cfg",
    [
        turbo_enn_config(num_init=2),
        turbo_zero_config(num_init=2),
        lhd_only_config(num_init=2),
    ],
)
def test_optimizer_contract(cfg):
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(42))
    x = opt.ask(2)

    assert isinstance(opt, Optimizer)
    assert x.shape == (2, 2)
    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert isinstance(opt.telemetry(), Telemetry)
