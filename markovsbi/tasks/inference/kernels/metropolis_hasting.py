from chex import PRNGKey
from jaxtyping import Array, PyTree


from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from markovsbi.tasks.inference.kernels.base import MCMCKernel
from markovsbi.tasks.inference.kernels.adaptation import (
    step_size_adaption,
    step_size_and_scale_adaption,
)

import blackjax
from blackjax.mcmc.random_walk import RWState, RWInfo


class MetropolisHastingParams(NamedTuple):
    pass


class MetropolisHastingKernel(MCMCKernel):
    params: MetropolisHastingParams

    def __init__(
        self,
        logdensity_fn: Callable,
        transition_proposal_fn: Callable,
        transition_proposal_logpdf: Optional[Callable] = None,
    ) -> None:
        self.logdensity_fn = logdensity_fn
        self.transition_proposal_fn = transition_proposal_fn
        self.transition_proposal_logpdf = transition_proposal_logpdf
        self.init_fn = blackjax.rmh.init
        self.update_fn = blackjax.rmh.build_kernel()

    def init_params(self, position: PyTree):
        return MetropolisHastingParams()

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[RWState, RWInfo]:
        raise NotImplementedError("adapt_params method must be implemented")

    def init_state(self, position: PyTree) -> RWState:
        self.init_params(position)
        return self.init_fn(position, self.logdensity_fn)

    def __call__(self, key: PRNGKey, state: RWState) -> Tuple[RWState, RWInfo]:
        transition_generator = lambda k, x: self.transition_proposal_fn(
            k, x, self.params
        )
        if self.transition_proposal_logpdf is not None:
            proposal_logdensity_fn = lambda x, y: self.transition_proposal_logpdf(
                x, y, self.params
            )
        else:
            proposal_logdensity_fn = None

        return self.update_fn(
            key, state, self.logdensity_fn, transition_generator, proposal_logdensity_fn
        )


class GaussianMHParameters(NamedTuple):
    step_size: float
    scale: Array


def gaussian_transition_proposal(key, position, params: GaussianMHParameters):
    flat_position, unflatten = ravel_pytree(position)
    eps = jax.random.normal(key, flat_position.shape)
    if params.scale is None:
        new_position = flat_position + params.step_size * eps
    elif len(params.scale.shape) == 1:
        new_position = flat_position + params.step_size * params.scale * eps
    elif len(params.scale.shape) == 2:
        new_position = flat_position + params.step_size * jnp.dot(params.scale, eps)
    else:
        raise ValueError("Invalid scale shape")

    return unflatten(new_position)


class GaussianMHKernel(MetropolisHastingKernel):
    params: GaussianMHParameters

    def __init__(
        self,
        logdensity_fn: Callable,
        step_size: float = 2.38,
        scale: Optional[Array] = None,
        is_scale_diagonal: bool = True,
    ) -> None:
        self.scale = scale
        if scale is not None:
            if len(scale.shape) == 1:
                self.is_scale_diagonal = True
            else:
                self.is_scale_diagonal = False
        else:
            self.is_scale_diagonal = is_scale_diagonal
        self.step_size = step_size

        self.params = GaussianMHParameters(step_size=step_size, scale=scale)
        super().__init__(logdensity_fn, gaussian_transition_proposal, None)

    def init_params(self, position: PyTree):
        flat_position, _ = ravel_pytree(position)
        dim = flat_position.shape[0]
        if self.scale is None:
            if self.is_scale_diagonal:
                scale = jnp.ones((dim,))
            else:
                scale = jnp.eye(dim)
        else:
            scale = self.scale

        step_size = self.step_size / jnp.sqrt(dim)

        self.params = GaussianMHParameters(step_size=step_size, scale=scale)

        return self.params

    def adapt_params(
        self,
        key: PRNGKey,
        position: PyTree,
        num_steps: int = 100,
        method="step_size_and_scale",
        target_acceptance_rate=0.234,
        **kwargs: Any,
    ) -> Tuple[RWState, RWInfo]:
        if method == "step_size":
            adaption_alg = step_size_adaption(
                blackjax.rmh,
                self.logdensity_fn,
                self.params,
                gaussian_transition_proposal,
                target_acceptance_rate=target_acceptance_rate,
                **kwargs,
            )

            results, info = adaption_alg.run(key, position, num_steps)
            step_size = results.parameters["step_size"]
            self.params = GaussianMHParameters(
                step_size=step_size, scale=self.params.scale
            )
        elif method == "step_size_and_scale":
            adaption_alg = step_size_and_scale_adaption(
                blackjax.rmh,
                self.logdensity_fn,
                self.params,
                gaussian_transition_proposal,
                target_acceptance_rate=target_acceptance_rate,
                **kwargs,
            )
            results, info = adaption_alg.run(key, position, num_steps)
            step_size = results.parameters["step_size"]
            scale = results.parameters["scale"]
            self.params = GaussianMHParameters(step_size=step_size, scale=scale)
        else:
            raise ValueError("Invalid method")

        return results.state, info
