from chex import PRNGKey
from jaxtyping import Array, PyTree


import jax
from typing import Any, Callable, NamedTuple, Tuple



from markovsbi.tasks.inference.kernels.base import MCMCKernel

import blackjax
from blackjax.mcmc.mala import MALAState, MALAInfo


class MALAParams(NamedTuple):
    step_size: float


class MALAKernel(MCMCKernel):
    params: MALAParams

    def __init__(
        self,
        logdensity_fn: Callable,
        step_size: float = 1e-3,
    ) -> None:
        self.logdensity_fn = logdensity_fn
        self.step_size = step_size
        self.init_fn = blackjax.mala.init
        self.update_fn = blackjax.mala.build_kernel()

    def init_params(self, position: PyTree):
        params = MALAParams(step_size=self.step_size)
        self.params = params
        return params

    def init_state(self, position: PyTree) -> MALAState:
        self.init_params(position)
        return self.init_fn(position, self.logdensity_fn)

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[MALAState, MALAInfo]:
        raise NotImplementedError("adapt_params method must be implemented")

    def __call__(self, key: PRNGKey, state: MALAState) -> Tuple[MALAState, MALAInfo]:
        return self.update_fn(key, state, self.logdensity_fn, *self.params)


# TODO ULA kernel


def init(position: Array, logdensity_fn: Callable) -> MALAState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return MALAState(position, logdensity, logdensity_grad)


def build_kernel() -> Callable:
    def kernel(
        rng_key: PRNGKey, state: MALAState, logdensity_fn: Callable, step_size: float
    ) -> tuple[MALAState, MALAInfo]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_langevin(grad_fn)

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)
        new_state = MALAState(*new_state)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = MALAInfo(p_accept, do_accept)

        return accepted_state, info

    return kernel
