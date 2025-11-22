from __future__ import annotations

import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class TurboGP(ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        lengthscale_constraint: Interval,
        outputscale_constraint: Interval,
        ard_dims: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_dims,
            lengthscale_constraint=lengthscale_constraint,
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=outputscale_constraint,
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self, x: torch.Tensor) -> MultivariateNormal:
        return self(x)
