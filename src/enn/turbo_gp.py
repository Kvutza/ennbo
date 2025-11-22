from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from gpytorch.models import ExactGP


class TurboGP(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims: int,
    ) -> None:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

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

    def forward(self, x) -> object:
        from gpytorch.distributions import MultivariateNormal

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self, x) -> object:
        return self(x)
