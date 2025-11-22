from __future__ import annotations


class EpistemicNearestNeighbors:
    def __init__(
        self,
        train_x,
        train_y,
        train_yvar,
        hnsw_threshold: int | None = None,
        hnsw_M: int = 32,
    ) -> None:
        import numpy as np

        if train_x.ndim != 2:
            raise ValueError(train_x.shape)
        if train_y.ndim != 2 or train_yvar.ndim != 2:
            raise ValueError((train_y.shape, train_yvar.shape))
        if train_x.shape[0] != train_y.shape[0] or train_y.shape != train_yvar.shape:
            raise ValueError((train_x.shape, train_y.shape, train_yvar.shape))
        self._train_x = np.asarray(train_x, dtype=float)
        self._train_y = np.asarray(train_y, dtype=float)
        self._train_yvar = np.asarray(train_yvar, dtype=float)
        self._num_obs, self._num_dim = self._train_x.shape
        _, self._num_metrics = self._train_y.shape
        self._eps_var = 1e-9
        self._x_scale = np.std(self._train_x, axis=0).astype(float)
        self._x_scale = np.maximum(self._x_scale, 1e-6)
        self._index = None
        self._hnsw_threshold = hnsw_threshold
        self._hnsw_M = int(hnsw_M)
        self._build_index()

    @property
    def train_x(self) -> object:
        return self._train_x

    @property
    def train_y(self) -> object:
        return self._train_y

    @property
    def train_yvar(self) -> object:
        return self._train_yvar

    @property
    def num_outputs(self) -> int:
        return self._num_metrics

    def __len__(self) -> int:
        return self._num_obs

    def _build_index(self) -> None:
        import faiss
        import numpy as np

        if self._num_obs == 0:
            return
        x_scaled = self._train_x / self._x_scale
        x_scaled = x_scaled.astype(np.float32, copy=False)
        if self._hnsw_threshold is not None and self._num_obs > self._hnsw_threshold:
            index = faiss.IndexHNSWFlat(self._num_dim, self._hnsw_M)
        else:
            index = faiss.IndexFlatL2(self._num_dim)
        index.add(x_scaled)
        self._index = index

    def _search(self, x, k: int, exclude_nearest: bool) -> tuple:
        import numpy as np

        if self._index is None or len(self) == 0:
            raise RuntimeError("index is not initialized")
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(len(self))
            k = min(k + 1, len(self))
        else:
            k = min(k, len(self))
        x_scaled = x / self._x_scale
        x_scaled = x_scaled.astype(np.float32, copy=False)
        dist2s, idx = self._index.search(x_scaled, k)
        if exclude_nearest:
            dist2s = dist2s[:, 1:]
            idx = idx[:, 1:]
        return dist2s.astype(float), idx.astype(int)

    def _calc_enn_normal(
        self,
        dist2s,
        y,
        yvar,
        var_scale: float,
    ):
        import numpy as np

        from .enn_normal import ENNNormal

        if dist2s.ndim != 2:
            raise ValueError(dist2s.shape)
        if y.shape != yvar.shape:
            raise ValueError((y.shape, yvar.shape))
        if var_scale <= 0.0:
            raise ValueError(var_scale)
        batch_size, num_neighbors = dist2s.shape
        if y.shape[0] != batch_size or y.shape[1] != num_neighbors:
            raise ValueError((dist2s.shape, y.shape))
        num_metrics = y.shape[2]
        if num_neighbors == 0:
            mu = np.zeros((batch_size, num_metrics), dtype=float)
            se = np.ones((batch_size, num_metrics), dtype=float)
            return ENNNormal(mu, se)
        if num_neighbors == 1:
            mu = y[:, 0, :]
            epistemic_var = np.ones((batch_size, num_metrics), dtype=float)
            noise_var = yvar[:, 0, :]
            vvar = epistemic_var + noise_var
            vvar = np.maximum(vvar, self._eps_var)
            se = np.sqrt(vvar)
            return ENNNormal(mu.astype(float, copy=False), se.astype(float, copy=False))
        dist2s_expanded = dist2s[..., np.newaxis]
        var_component = var_scale * dist2s_expanded + yvar
        w = 1.0 / (self._eps_var + var_component)
        norm = np.sum(w, axis=1)
        mu = np.sum(w * y, axis=1) / norm
        epistemic_var = 1.0 / norm
        noise_var = np.sum(w * yvar, axis=1) / norm
        vvar = epistemic_var + noise_var
        vvar = np.maximum(vvar, self._eps_var)
        mu = mu.astype(float, copy=False)
        se = np.sqrt(vvar).astype(float, copy=False)
        return ENNNormal(mu, se)

    def posterior(
        self,
        x,
        *,
        k: int,
        var_scale: float,
        exclude_nearest: bool = False,
    ):
        import numpy as np

        from .enn_normal import ENNNormal

        if len(self) == 0:
            if x.ndim != 2:
                raise ValueError(x.shape)
            batch_size = x.shape[0]
            mu = np.zeros((batch_size, self._num_metrics), dtype=float)
            se = np.ones((batch_size, self._num_metrics), dtype=float)
            return ENNNormal(mu, se)
        dist2s, idx = self._search(x, k=k, exclude_nearest=exclude_nearest)
        y_neighbors = self._train_y[idx]
        yvar_neighbors = self._train_yvar[idx]
        return self._calc_enn_normal(dist2s, y_neighbors, yvar_neighbors, var_scale)
