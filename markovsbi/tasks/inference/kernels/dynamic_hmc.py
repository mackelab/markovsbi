from chex import PRNGKey
from jaxtyping import Array, PyTree


import jax
from typing import Any, Callable, NamedTuple, Optional, Tuple


import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from markovsbi.tasks.inference.kernels.base import MCMCKernel

import blackjax
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.hmc import HMCInfo
from blackjax.mcmc.dynamic_hmc import halton_trajectory_length


def halton_trajectory_length_fns(average_trajectory_length: float):
    def halton_init_random_arg_fn():
        return jnp.array(0, dtype=jnp.int32)

    def halton_next_random_arg_fn(index: Array):
        return jnp.array(index + 1, dtype=jnp.int32)

    def halton_next_integration_steps_fn(random_arg: Array, **kwargs):
        return halton_trajectory_length(random_arg, average_trajectory_length)

    return (
        halton_init_random_arg_fn,
        halton_next_random_arg_fn,
        halton_next_integration_steps_fn,
    )


def random_trajectory_length_fns(average_trajectory_length: int):
    def random_init_random_arg_fn():
        return jax.random.PRNGKey(0)

    def random_next_random_arg_fn(random_arg: Array):
        return jax.random.split(random_arg)[1]

    def random_next_integration_steps_fn(random_arg: Array, **kwargs):
        return jax.random.randint(
            random_arg, shape=(), minval=1, maxval=2 * average_trajectory_length
        )

    return (
        random_init_random_arg_fn,
        random_next_random_arg_fn,
        random_next_integration_steps_fn,
    )


class DynamicHMCParams(NamedTuple):
    step_size: float
    inverse_mass_matrix: Array


class DynamicHMCKernel(MCMCKernel):
    params: DynamicHMCParams

    def __init__(
        self,
        logdensity_fn: Callable,
        step_size: float = 1e-2,
        average_integration_steps: int = 20,
        integration_steps_sequence: str = "halton",
        inverse_mass_matrix: Optional[Array] = None,
        is_mass_matrix_diagonal: bool = True,
        init_random_arg_fn: Optional[Callable] = None,
        next_random_arg_fn: Optional[Callable] = None,
        integration_steps_fn: Optional[Callable] = None,
    ) -> None:
        self.logdensity_fn = logdensity_fn
        self.num_integration_steps = average_integration_steps

        self._inital_step_size = step_size
        self._inital_inverse_mass_matrix = inverse_mass_matrix
        if inverse_mass_matrix is not None:
            if len(inverse_mass_matrix.shape) == 1:
                self.is_mass_matrix_diagonal = True
            else:
                self.is_mass_matrix_diagonal = False
        else:
            self.is_mass_matrix_diagonal = is_mass_matrix_diagonal

        if integration_steps_sequence == "halton":
            random_arg_init_fn, random_arg_next_fn, integration_steps_fn = (
                halton_trajectory_length_fns(average_integration_steps)
            )
        elif integration_steps_sequence == "random":
            random_arg_init_fn, random_arg_next_fn, integration_steps_fn = (
                random_trajectory_length_fns(average_integration_steps)
            )
        else:
            raise ValueError(
                "Invalid integration_steps_sequence, specify specific functions for init, next and integration_steps_fn"
            )

        if init_random_arg_fn is not None:
            random_arg_init_fn = init_random_arg_fn
        if next_random_arg_fn is not None:
            random_arg_next_fn = next_random_arg_fn
        if integration_steps_fn is not None:
            integration_steps_fn = integration_steps_fn

        self.random_arg_init_fn = random_arg_init_fn
        self.random_arg_next_fn = random_arg_next_fn
        self.integration_steps_fn = integration_steps_fn

        self.init_fn = blackjax.dynamic_hmc.init
        self.update_fn = blackjax.dynamic_hmc.build_kernel(
            next_random_arg_fn=self.random_arg_next_fn,
            integration_steps_fn=self.integration_steps_fn,
        )

    def init_params(self, position: PyTree):
        flat_position, _ = ravel_pytree(position)
        dim = flat_position.shape[0]
        if self._inital_inverse_mass_matrix is None:
            if self.is_mass_matrix_diagonal:
                inverse_mass_matrix = jnp.ones((dim,))
            else:
                inverse_mass_matrix = jnp.eye(dim)
        self.params = DynamicHMCParams(
            step_size=self._inital_step_size, inverse_mass_matrix=inverse_mass_matrix
        )
        return self.params

    def init_state(self, position: PyTree) -> DynamicHMCState:
        params = self.init_params(position)
        random_arg = self.random_arg_init_fn()
        return self.init_fn(position, self.logdensity_fn, random_arg)

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[DynamicHMCState, HMCInfo]:
        raise NotImplementedError("adapt_params method must be implemented")

    def __call__(
        self, key: PRNGKey, state: DynamicHMCState
    ) -> Tuple[DynamicHMCState, HMCInfo]:
        return self.update_fn(key, state, self.logdensity_fn, *self.params)
