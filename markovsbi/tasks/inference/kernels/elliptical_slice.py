from chex import PRNGKey
from jaxtyping import Array, PyTree


import jax
from typing import Any, Callable, NamedTuple, Tuple


import jax.numpy as jnp

from markovsbi.tasks.inference.kernels.base import MCMCKernel

import blackjax
from blackjax.mcmc.elliptical_slice import EllipSliceState, EllipSliceInfo


class EllipticalSliceParams(NamedTuple):
    pass


class EllipticalSliceKernel(MCMCKernel):
    params: EllipticalSliceParams

    def __init__(
        self,
        logdensity_fn: Callable,
        cov_matrix: Array,
        mean: Array,
        is_loglikelihood: bool = False,
    ) -> None:
        if not is_loglikelihood:
            cov_sqrt = jnp.linalg.cholesky(cov_matrix)

            def log_density_fn_wrapper(x):
                return logdensity_fn(x) - 0.5 * jnp.dot(
                    x - mean, jax.scipy.linalg.cho_solve((cov_sqrt, True), x - mean)
                )

            self.logdensity_fn = log_density_fn_wrapper
        else:
            self.logdensity_fn = logdensity_fn

        self.init_fn = blackjax.elliptical_slice.init
        self.update_fn = blackjax.elliptical_slice.build_kernel(cov_matrix, mean)

    def init_params(self, position: PyTree):
        return EllipticalSliceParams()

    def init_state(self, position: PyTree) -> EllipSliceState:
        self.init_params(position)
        return self.init_fn(position, self.logdensity_fn)

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        raise NotImplementedError("adapt_params method must be implemented")

    def __call__(
        self, key: PRNGKey, state: EllipSliceState
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        return self.update_fn(key, state, self.logdensity_fn)
