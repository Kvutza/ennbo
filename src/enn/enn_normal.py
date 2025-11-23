from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ENNNormal:
    mu: object
    se: object

    def sample(
        self,
        num_samples: int,
        rng,
        clip=None,
    ) -> object:
        import numpy as np

        size = (*self.se.shape, num_samples)
        eps = rng.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)
        return np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps
