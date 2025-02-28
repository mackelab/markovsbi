from chex import PRNGKey
from jaxtyping import Array, PyTree


import jax
from typing import Callable, NamedTuple, Optional, Tuple


import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from markovsbi.tasks.inference.kernels.base import MCMCKernel

import blackjax
from blackjax.mcmc.hmc import HMCState, HMCInfo


class HMCParams(NamedTuple):
    step_size: float
    inverse_mass_matrix: Array


class HMCKernel(MCMCKernel):
    params: HMCParams

    def __init__(
        self,
        logdensity_fn: Callable,
        step_size: float = 1e-2,
        num_integration_steps: int = 10,
        inverse_mass_matrix: Optional[Array] = None,
        is_mass_matrix_diagonal: bool = True,
    ) -> None:
        self.logdensity_fn = logdensity_fn
        self.num_integration_steps = num_integration_steps

        self._inital_step_size = step_size
        self._inital_inverse_mass_matrix = inverse_mass_matrix
        if inverse_mass_matrix is not None:
            if len(inverse_mass_matrix.shape) == 1:
                self.is_mass_matrix_diagonal = True
            else:
                self.is_mass_matrix_diagonal = False
        else:
            self.is_mass_matrix_diagonal = is_mass_matrix_diagonal

        self.init_fn = blackjax.hmc.init
        self.update_fn = blackjax.hmc.build_kernel()

    def init_params(self, position: PyTree):
        flat_position, _ = ravel_pytree(position)
        dim = flat_position.shape[0]
        if self._inital_inverse_mass_matrix is None:
            if self.is_mass_matrix_diagonal:
                inverse_mass_matrix = jnp.ones((dim,))
            else:
                inverse_mass_matrix = jnp.eye(dim)
        else:
            inverse_mass_matrix = self.inverse_mass_matrix

        self.params = HMCParams(
            step_size=self._inital_step_size, inverse_mass_matrix=inverse_mass_matrix
        )

    def adapt_params(
        self,
        key: PRNGKey,
        position: Array,
        num_steps: int = 100,
        target_acceptance_rate: float = 0.8,
        method: str = "window",
    ):
        if method == "window":
            adaption_alg = blackjax.window_adaptation(
                blackjax.hmc,
                self.logdensity_fn,
                initial_step_size=self._inital_step_size,
                num_integration_steps=self.num_integration_steps,
                is_mass_matrix_diagonal=self.is_mass_matrix_diagonal,
                target_acceptance_rate=target_acceptance_rate,
            )
        elif method == "pathfinder":
            adaption_alg = blackjax.pathfinder_adaptation(
                blackjax.hmc,
                self.logdensity_fn,
                initial_step_size=self._inital_step_size,
                num_integration_steps=self.num_integration_steps,
                target_acceptance_rate=target_acceptance_rate,
            )
        else:
            raise ValueError(f"Adaption method {method} not supported")

        results, info = jax.jit(adaption_alg.run, static_argnums=(2,))(
            key, position, num_steps
        )
        self.params = HMCParams(
            step_size=results.parameters["step_size"],
            inverse_mass_matrix=results.parameters["inverse_mass_matrix"],
        )
        state = results.state
        return state, info

    def init_state(self, position: Array) -> HMCState:
        self.init_params(position)
        state = self.init_fn(position, self.logdensity_fn)
        return state

    def __call__(self, key: PRNGKey, state: HMCState) -> Tuple[HMCState, HMCInfo]:
        new_state, info = self.update_fn(
            key,
            state,
            self.logdensity_fn,
            num_integration_steps=self.num_integration_steps,
            *self.params,
        )
        return new_state, info


class NUTSKernel(HMCKernel):
    def __init__(
        self,
        logdensity_fn: Callable,
        step_size: float = 1e-2,
        max_treedepth: int = 10,
        inverse_mass_matrix: Optional[Array] = None,
        is_mass_matrix_diagonal: bool = True,
    ) -> None:
        super().__init__(
            logdensity_fn,
            step_size,
            num_integration_steps=1,
            inverse_mass_matrix=inverse_mass_matrix,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        )
        self.max_treedepth = max_treedepth
        self.init_fn = blackjax.nuts.init
        self.update_fn = blackjax.nuts.build_kernel()

    def init_state(self, position: Array) -> HMCState:
        self.init_params(position)
        state = self.init_fn(position, self.logdensity_fn)
        return state

    def __call__(self, key: PRNGKey, state: HMCState) -> Tuple[HMCState, HMCInfo]:
        new_state, info = self.update_fn(
            key,
            state,
            self.logdensity_fn,
            # max_treedepth=self.max_treedepth,
            *self.params,
        )
        return new_state, info
