import numpy as np

from enn import EpistemicNearestNeighbors, enn_fit, subsample_loglik


def test_subsample_loglik_and_enn_fit_improve_hyperparameters():
    np.random.seed(0)
    n = 40
    d = 2
    x = np.random.randn(n, d)
    true_w = np.array([1.5, -0.5])
    y_mean = x @ true_w
    noise_std = 0.1
    noise = noise_std * np.random.randn(n)
    y = (y_mean + noise).reshape(-1, 1)
    yvar = (noise_std**2) * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=None)
    base_ll = subsample_loglik(
        model,
        x,
        y[:, 0],
        yvar[:, 0],
        k=1,
        var_scale=1.0,
        P=20,
    )
    result = enn_fit(
        model, k_values=None, var_scale_values=None, num_iterations=1, P=20
    )
    assert "k" in result and "var_scale" in result
    assert result["k"] >= 1
    assert result["var_scale"] > 0.0
    tuned_ll = subsample_loglik(
        model,
        x,
        y[:, 0],
        yvar[:, 0],
        k=int(result["k"]),
        var_scale=float(result["var_scale"]),
        P=20,
    )
    assert tuned_ll >= base_ll
