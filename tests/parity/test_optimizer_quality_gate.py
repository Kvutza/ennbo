from __future__ import annotations

import pytest

pytest.importorskip("ennx._rust")
pytestmark = pytest.mark.slow


def test_optimizer_quality_ci_subset():
    from .parity_quality_gate import assert_ci_quality_gate

    assert_ci_quality_gate()
