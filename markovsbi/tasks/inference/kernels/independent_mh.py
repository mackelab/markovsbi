from chex import PRNGKey
from jaxtyping import Array, PyTree


from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from markovsbi.tasks.inference.kernels.base import MCMCKernel

import blackjax
from blackjax.mcmc.random_walk import RWState, RWInfo


class ProposalParams(NamedTuple):
    pass


class IndependentMetropolisHastingKernel(MCMCKernel):
    params: ProposalParams

    def __init__(
        self,
        logdensity_fn: Callable,
        proposal_fn: Callable,
        proposal_logpdf: Callable,
    ) -> None:
        self.logdensity_fn = logdensity_fn
        self.proposal_fn = proposal_fn
        self.proposal_logpdf = proposal_logpdf
        self.init_fn = blackjax.irmh.init
        self.update_fn = blackjax.irmh.build_kernel()

    def init_params(self, position: PyTree):
        return ProposalParams()

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[RWState, RWInfo]:
        raise NotImplementedError("adapt_params method must be implemented")

    def init_state(self, position: PyTree) -> RWState:
        self.init_params(position)
        return self.init_fn(position, self.logdensity_fn)

    def __call__(self, key: PRNGKey, state: RWState) -> Tuple[RWState, RWInfo]:
        proposal = lambda k: self.proposal_fn(k, self.params)

        proposal_log_prob = lambda x, _: self.proposal_logpdf(x, self.params)

        return self.update_fn(
            key, state, self.logdensity_fn, proposal, proposal_log_prob
        )


class GaussianParams(NamedTuple):
    mean: Array
    cov: Optional[Array]
    scale: Array


def gaussian_proposal(key, params: GaussianParams):
    mean = params.mean
    eps = jax.random.normal(key, mean.shape)

    if len(params.scale.shape) == 1:
        new_position = mean + jnp.sqrt(params.scale) * eps
    elif len(params.scale.shape) == 2:
        new_position = mean + jnp.dot(params.scale, eps)
    else:
        raise ValueError("Invalid scale shape")

    return new_position


def gaussian_logpdf(state, params: GaussianParams):
    x = state.position
    mean = params.mean
    if len(params.scale.shape) == 1:
        return jax.scipy.stats.norm.logpdf(x, mean, params.scale).sum()
    else:
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean, params.cov)


class GaussianIMHKernel(IndependentMetropolisHastingKernel):
    params: GaussianParams

    def __init__(
        self,
        logdensity_fn: Callable,
        mean: Array,
        scale: Array,
        cov: Optional[Array] = None,
    ) -> None:
        super().__init__(logdensity_fn, gaussian_proposal, gaussian_logpdf)
        self.mean = jnp.atleast_1d(mean)
        self.scale = jnp.atleast_1d(scale)

        if cov is not None:
            if len(self.scale.shape) == 2:
                self.cov = jnp.dot(self.scale, self.scale.T)
        else:
            self.cov = cov

    def init_params(self, position: PyTree):
        mean = jnp.asarray(self.mean)
        mean = jnp.broadcast_to(mean, position.shape)
        self.params = GaussianParams(mean=mean, cov=self.cov, scale=self.scale)
        return self.params

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[RWState, RWInfo]:
        raise NotImplementedError("adapt_params method must be implemented")
