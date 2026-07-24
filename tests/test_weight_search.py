import numpy as np

from enn.enn_rust import optimizer


def _leaves():
    return [
        (0, 257, 4, 0.25, 1.0, 0.75),
        (257, 263, 8, 0.5, 0.5, 1.0),
    ]


def _base():
    row_bytes = (257 + 1) // 2 + 263
    return np.asarray(
        [(index * 37 + 11) & 0xFF for index in range(row_bytes)],
        dtype=np.uint8,
    )


def _ask(backend):
    search = optimizer.WeightSearch(_base(), 0.25, _leaves(), 4, backend)
    _, _, _ = search.ask(np.asarray([17], dtype=np.uint64), 1.0, 1)
    search.tell(0.75, True)
    index, seed, score = search.ask(
        np.asarray([19, 23, 29, 31], dtype=np.uint64),
        0.65,
        2,
        beta=1.3,
    )
    return index, seed, score, np.asarray(search.row())


def test_weight_search_keeps_state_across_ask_and_tell():
    cpu = _ask("cpu")
    assert cpu[0] in range(4)
    assert cpu[1] in {19, 23, 29, 31}
    assert np.isfinite(cpu[2])
    assert cpu[3].shape == _base().shape
    assert not np.array_equal(cpu[3], _base())


def test_weight_search_metal_matches_cpu():
    cpu = _ask("cpu")
    metal = _ask("metal")
    assert metal[:2] == cpu[:2]
    assert np.isclose(metal[2], cpu[2], atol=1.0e-5)
    assert np.array_equal(metal[3], cpu[3])
