from dataclasses import dataclass

import numpy as np


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray

    def sample(self, num_samples, clip=None) -> np.ndarray:
        if isinstance(num_samples, tuple):
            if len(num_samples) != 1:
                raise ValueError(num_samples)
            num_samples = num_samples[0]
        size = list(self.se.shape)
        size.append(int(num_samples))
        eps = np.random.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)
        return np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps
