def test_subsample_loglik_and_enn_fit_improve_hyperparameters():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.fit import enn_fit, subsample_loglik

    rng = np.random.default_rng(0)
    n = 40
    d = 2
    x = rng.standard_normal((n, d))
    true_w = np.array([1.5, -0.5])
    y_mean = x @ true_w
    noise_std = 0.1
    noise = noise_std * rng.standard_normal(n)
    y = (y_mean + noise).reshape(-1, 1)
    yvar = (noise_std**2) * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar, hnsw_threshold=None)
    rng_base = np.random.default_rng(0)
    base_ll = subsample_loglik(
        model,
        x,
        y[:, 0],
        k=1,
        var_scale=1.0,
        P=20,
        rng=rng_base,
    )
    rng_fit = np.random.default_rng(1)
    result = enn_fit(
        model,
        num_tries=30,
        P=20,
        rng=rng_fit,
    )
    assert "k" in result and "var_scale" in result
    assert result["k"] >= 1
    assert result["var_scale"] > 0.0
    rng_tuned = np.random.default_rng(2)
    tuned_ll = subsample_loglik(
        model,
        x,
        y[:, 0],
        k=int(result["k"]),
        var_scale=float(result["var_scale"]),
        P=20,
        rng=rng_tuned,
    )
    assert tuned_ll >= base_ll
