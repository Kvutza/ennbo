from __future__ import annotations

import pytest


def test_ennnormal_sample_shape_and_clip():
    import numpy as np

    from enn.enn_normal import ENNNormal

    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 1.0]], dtype=float)
    se = np.array([[1.0, 2.0]], dtype=float)
    normal = ENNNormal(mu=mu, se=se)
    samples = normal.sample(5, clip=1.0, rng=rng)
    assert samples.shape == (1, 2, 5)
    assert np.all(samples >= mu.min() - 2.0)
    assert np.all(samples <= mu.max() + 2.0)


def test_epistemic_nearest_neighbors_posterior_and_var_scale():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, var_scale=1.0)
    post = model.posterior(x_test, params=params, exclude_nearest=False)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    post_changed = model.posterior(
        x_test, params=ENNParams(k=5, var_scale=0.5), exclude_nearest=True
    )
    assert post_changed.mu.shape == (4, 1)
    assert post_changed.se.shape == (4, 1)


def test_epistemic_nearest_neighbors_with_no_observations_returns_prior_like_posterior():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    d = 3
    x = np.zeros((0, d), dtype=float)
    y = np.zeros((0, 1), dtype=float)
    yvar = np.ones_like(y, dtype=float)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(
        x_test, params=ENNParams(k=3, var_scale=1.0), exclude_nearest=False
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.allclose(post.mu, 0.0)
    assert np.allclose(post.se, 1.0)


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_epistemic_nearest_neighbors_with_few_observations_has_valid_posterior(
    num_obs: int,
):
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    d = 3
    x = rng.standard_normal((num_obs, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(
        x_test, params=ENNParams(k=3, var_scale=1.0), exclude_nearest=False
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_batch_posterior_matches_individual_posterior_calls():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    paramss = [
        ENNParams(k=3, var_scale=1.0),
        ENNParams(k=5, var_scale=0.5),
        ENNParams(k=7, var_scale=2.0),
    ]
    post_batch = model.batch_posterior(x_test, paramss, exclude_nearest=False)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, exclude_nearest=False)
        assert np.allclose(post_batch.mu[i], post.mu)
        assert np.allclose(post_batch.se[i], post.se)


def test_batch_posterior_matches_individual_posterior_calls_with_exclude_nearest():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    paramss = [
        ENNParams(k=3, var_scale=1.0),
        ENNParams(k=5, var_scale=0.5),
    ]
    post_batch = model.batch_posterior(x_test, paramss, exclude_nearest=True)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, exclude_nearest=True)
        assert np.allclose(post_batch.mu[i], post.mu)
        assert np.allclose(post_batch.se[i], post.se)


def test_epistemic_nearest_neighbors_with_sobol_indices():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + 0.1 * x[:, 1] + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    yvar = 0.1 * np.ones_like(y)
    model_default = EpistemicNearestNeighbors(x, y, yvar, sobol_indices=False)
    model_sobol = EpistemicNearestNeighbors(x, y, yvar, sobol_indices=True)
    assert not np.allclose(model_default._x_scale, model_sobol._x_scale)
    assert np.all(model_sobol._x_scale > 0)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, var_scale=1.0)
    post_default = model_default.posterior(x_test, params=params, exclude_nearest=False)
    post_sobol = model_sobol.posterior(x_test, params=params, exclude_nearest=False)
    assert post_sobol.mu.shape == (4, 1)
    assert post_sobol.se.shape == (4, 1)
    assert np.all(np.isfinite(post_sobol.mu))
    assert np.all(np.isfinite(post_sobol.se))
    assert not np.allclose(post_default.mu, post_sobol.mu) or not np.allclose(
        post_default.se, post_sobol.se
    )


def test_epistemic_nearest_neighbors_sobol_indices_requires_single_metric():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors

    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 2))
    yvar = 0.1 * np.ones_like(y)
    with pytest.raises(
        ValueError, match="sobol_indices=True requires train_y to have exactly 1 metric"
    ):
        EpistemicNearestNeighbors(x, y, yvar, sobol_indices=True)
