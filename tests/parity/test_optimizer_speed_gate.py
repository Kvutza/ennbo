from __future__ import annotations

import pytest

pytest.importorskip("ennx._rust")
pytestmark = pytest.mark.slow


def test_optimizer_speed_ci_subset():
    from .parity_speed_gate import assert_ci_speed_gate

    assert_ci_speed_gate()
