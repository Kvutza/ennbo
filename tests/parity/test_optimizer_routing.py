from __future__ import annotations

import numpy as np
import pytest

from ennx import create_optimizer, turbo_enn_config, turbo_one_config, turbo_zero_config
from ennx.turbo.config import ENNSurrogateConfig, lhd_only_config
from ennx.turbo.config.encode import ENN_K, enn_k, supports
from ennx.turbo.optimizer import Optimizer

pytest.importorskip("ennx._rust")

BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "cfg",
    [turbo_enn_config(), turbo_zero_config(), lhd_only_config(), turbo_one_config()],
)
def test_configs_use_native_core(cfg):
    assert supports(cfg)
    opt = create_optimizer(bounds=BOUNDS, config=cfg, rng=np.random.default_rng(0))
    assert isinstance(opt, Optimizer)


def test_default_enn_k():
    cfg = turbo_enn_config(enn=ENNSurrogateConfig(k=None))
    assert enn_k(cfg) == ENN_K
