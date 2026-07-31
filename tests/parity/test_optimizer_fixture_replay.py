from __future__ import annotations

import pytest

from ennx.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    assert_fixture_contracts,
    load_fixture,
)
from ennx.turbo.optimizer_fixtures.replay import _config_for_fixture

pytest.importorskip("ennx._rust")


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_optimizer_replays_fixtures(name: str):
    data = load_fixture(name)
    config = _config_for_fixture(name)
    assert_fixture_contracts(data, config)
