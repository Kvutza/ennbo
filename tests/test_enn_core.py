import numpy as np
import pytest

from enn import ENNNormal, EpistemicNearestNeighbors


def test_ennnormal_sample_shape_and_clip():
    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 1.0]], dtype=float)
    se = np.array([[1.0, 2.0]], dtype=float)
    normal = ENNNormal(mu=mu, se=se)
    samples = normal.sample(5, clip=1.0, rng=rng)
    assert samples.shape == (1, 2, 5)
    assert np.all(samples >= mu.min() - 2.0)
    assert np.all(samples <= mu.max() + 2.0)


def test_epistemic_nearest_neighbors_posterior_and_var_scale_and_hnsw_threshold():
    np.random.seed(0)
    n = 20
    d = 3
    x = np.random.randn(n, d)
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model_flat = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=None)
    model_hnsw = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=5)
    x_test = np.random.randn(4, d)
    post_flat = model_flat.posterior(x_test, k=3, var_scale=1.0, exclude_nearest=False)
    post_hnsw = model_hnsw.posterior(x_test, k=3, var_scale=1.0, exclude_nearest=True)
    assert post_flat.mu.shape == (4, 1)
    assert post_flat.se.shape == (4, 1)
    assert post_hnsw.mu.shape == (4, 1)
    assert post_hnsw.se.shape == (4, 1)
    post_changed = model_flat.posterior(
        x_test, k=5, var_scale=0.5, exclude_nearest=True
    )
    assert post_changed.mu.shape == (4, 1)
    assert post_changed.se.shape == (4, 1)


def test_epistemic_nearest_neighbors_with_no_observations_returns_prior_like_posterior():
    np.random.seed(0)
    d = 3
    x = np.zeros((0, d), dtype=float)
    y = np.zeros((0, 1), dtype=float)
    yvar = np.ones_like(y, dtype=float)
    model = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=None)
    x_test = np.random.randn(5, d)
    post = model.posterior(x_test, k=3, var_scale=1.0, exclude_nearest=False)
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.allclose(post.mu, 0.0)
    assert np.allclose(post.se, 1.0)


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_epistemic_nearest_neighbors_with_few_observations_has_valid_posterior(
    num_obs: int,
):
    np.random.seed(0)
    d = 3
    x = np.random.randn(num_obs, d)
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=None)
    x_test = np.random.randn(5, d)
    post = model.posterior(x_test, k=3, var_scale=1.0, exclude_nearest=False)
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))
