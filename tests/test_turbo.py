import numpy as np
import pytest

from enn.turbo import (
    TurboMode,
    TurboOptimizer,
    _argmax_random_tie,
    _latin_hypercube,
    _pareto_front,
    _sobol_like,
    _TrustRegionState,
)


def _sphere(x: np.ndarray) -> np.ndarray:
    return -np.sum(x**2, axis=1)


def _run_bo(mode: TurboMode, num_steps: int = 15) -> float:
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = TurboOptimizer(bounds=bounds, mode=mode, num_arms=4, rng=rng)
    best = -np.inf
    for _ in range(num_steps):
        x = opt.ask(num_arms=4)
        y = _sphere(x)
        opt.tell(x, y)
        best = max(best, float(np.max(y)))
    return best


def test_turbo_zero_ask_tell_and_shape():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ZERO,
        num_arms=3,
        rng=rng,
    )
    x0 = opt.ask(num_arms=3)
    assert x0.shape == (3, 2)
    assert np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    y0 = _sphere(x0)
    opt.tell(x0, y0)
    x1 = opt.ask(num_arms=3)
    assert x1.shape == (3, 2)


def test_turbo_one_improves_on_sphere():
    best = _run_bo(TurboMode.TURBO_ONE, num_steps=12)
    assert best > -0.5


def test_turbo_zero_reasonable_on_sphere():
    best = _run_bo(TurboMode.TURBO_ZERO, num_steps=12)
    assert best > -1.5


def test_turbo_enn_uses_enn_and_is_reasonable():
    best = _run_bo(TurboMode.TURBO_ENN, num_steps=12)
    assert best > -1.5


def test_latin_hypercube_stratification_and_bounds():
    rng = np.random.default_rng(0)
    n = 8
    d = 3
    x = _latin_hypercube(n, d, rng=rng)
    assert x.shape == (n, d)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    for j in range(d):
        xs = np.sort(x[:, j])
        for k in range(n):
            lo = k / n
            hi = (k + 1) / n
            in_bin = (xs >= lo) & (xs <= hi + 1e-8)
            assert np.any(in_bin)


def test_sobol_like_unit_cube_and_reproducible():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    x1 = _sobol_like(16, 2, rng=rng1)
    x2 = _sobol_like(16, 2, rng=rng2)
    assert x1.shape == (16, 2)
    assert np.all(x1 >= 0.0) and np.all(x1 <= 1.0)
    assert np.allclose(x1, x2)


def test_argmax_random_tie_uses_rng_and_is_deterministic():
    values = np.array([1.0, 2.0, 2.0, 0.0], dtype=float)
    rng = np.random.default_rng(0)
    idx1 = _argmax_random_tie(values, rng=rng)
    assert idx1 in (1, 2)
    rng = np.random.default_rng(0)
    idx2 = _argmax_random_tie(values, rng=rng)
    assert idx1 == idx2


def test_pareto_front_simple_case():
    mu = np.array([1.0, 0.9, 0.8], dtype=float)
    se = np.array([1.0, 0.5, 0.7], dtype=float)
    mask = _pareto_front(mu, se)
    assert mask.dtype == bool
    assert mask.shape == mu.shape
    assert mask[0]
    assert mask[1]
    assert not mask[2]


def test_trust_region_state_update_and_restart_and_bounds():
    state = _TrustRegionState(num_dim=2, num_arms=2)
    values = []
    for v in [0.0, 1.0, 2.0]:
        values.append(v)
        state.update(np.array(values, dtype=float))
    x_center = np.zeros((1, 2), dtype=float)
    lb, ub = state.create_bounds(x_center)
    assert lb.shape == (1, 2)
    assert ub.shape == (1, 2)
    state.length = state.length_min / 2.0
    assert state.needs_restart()
    state.restart()
    assert state.length == state.length_init


@pytest.mark.parametrize(
    "mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ONE, TurboMode.TURBO_ENN]
)
def test_turbo_behavior_independent_of_affine_x(mode: TurboMode) -> None:
    bounds1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    bounds2 = np.array([[2.0, 4.0], [-3.0, 1.0]], dtype=float)
    num_arms = 3
    num_steps = 8
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    opt1 = TurboOptimizer(
        bounds=bounds1,
        mode=mode,
        num_arms=num_arms,
        rng=rng1,
    )
    opt2 = TurboOptimizer(
        bounds=bounds2,
        mode=mode,
        num_arms=num_arms,
        rng=rng2,
    )

    def to_unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return (x - lb) / (ub - lb)

    for _ in range(num_steps):
        x1 = opt1.ask(num_arms=num_arms)
        x2 = opt2.ask(num_arms=num_arms)
        u1 = to_unit(x1, bounds1)
        u2 = to_unit(x2, bounds2)
        # Optimizer behavior in unit space should not depend on how x is scaled or centered.
        assert np.allclose(u1, u2)
        # Use a common objective defined on unit coordinates to ensure identical y across runs.
        z1 = 2.0 * u1 - 1.0
        z2 = 2.0 * u2 - 1.0
        y1 = _sphere(z1)
        y2 = _sphere(z2)
        assert np.allclose(y1, y2)
        opt1.tell(x1, y1)
        opt2.tell(x2, y2)


@pytest.mark.parametrize(
    "mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ONE, TurboMode.TURBO_ENN]
)
def test_turbo_behavior_independent_of_affine_y(mode: TurboMode) -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    num_steps = 8

    def run_with_transform(scale: float, shift: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        opt = TurboOptimizer(
            bounds=bounds,
            mode=mode,
            num_arms=num_arms,
            rng=rng,
        )
        unit_trajectory = []
        for _ in range(num_steps):
            x = opt.ask(num_arms=num_arms)
            # With these bounds, x already lives in unit space.
            u = x.copy()
            z = 2.0 * u - 1.0
            base_y = _sphere(z)
            y = scale * base_y + shift
            opt.tell(x, y)
            unit_trajectory.append(u)
        return np.stack(unit_trajectory, axis=0)

    traj_base = run_with_transform(scale=1.0, shift=0.0)
    traj_affine = run_with_transform(scale=2.0, shift=0.5)
    # The sequence of unit-space query points should be invariant to affine
    # rescalings (scale and center) of the observed y values.
    assert np.allclose(traj_base, traj_affine)
