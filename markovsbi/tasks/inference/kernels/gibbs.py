from chex import PRNGKey
from jaxtyping import Array, ArrayLike


from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp




from blackjax.base import State, Info

from blackjax.base import SamplingAlgorithm


class GibbsState(NamedTuple):
    position: Dict[str, ArrayLike]
    inner_state: Dict[str, State]


class GibbsInfo(NamedTuple):
    inner_info: Dict[str, Info]


def init(
    position: Dict[str, ArrayLike],
    logdensity_fn: Callable,
    inner_kernel: Dict,
    rng_key: Optional[PRNGKey] = None,
) -> GibbsState:
    inner_state = {}
    for k in position.keys():

        def logdensity_k(value):
            kwargs = position.copy()
            kwargs[k] = value
            return logdensity_fn(**kwargs)

        # inspect for keyword argument "rng_key"
        if "rng_key" in inner_kernel[k].init.__code__.co_varnames:
            inner_state[k] = inner_kernel[k].init(
                position[k], logdensity_k, rng_key=rng_key
            )
        else:
            inner_state[k] = inner_kernel[k].init(position[k], logdensity_k)

    return GibbsState(position=position, inner_state=inner_state)


def build_kernel(
    inner_kernel: Dict,
    inner_kernel_kwargs: Optional[Dict[str, Any]] = None,
    inner_kernel_steps: Optional[Dict[str, int]] = None,
) -> Callable:
    _kernels = {k: inner_kernel[k].build_kernel() for k in inner_kernel.keys()}

    def kernel(
        rng_key: PRNGKey,
        state: GibbsState,
        logdensity_fn: Callable,
        **kwargs,
    ) -> Tuple[GibbsState, GibbsInfo]:
        inner_info = {}
        inner_state = {}
        new_position = state.position.copy()

        for k in state.position.keys():
            if inner_kernel_steps is None:
                num_steps = 1
            else:
                num_steps = inner_kernel_steps[k] or 1

            rng_key, *rng_keys = jax.random.split(rng_key, num_steps + 1)

            def logdensity_k(value):
                kwargs = new_position.copy()
                kwargs[k] = value
                return logdensity_fn(**kwargs)

            if inner_kernel_kwargs is None:
                kwargs = {}
            else:
                kwargs = inner_kernel_kwargs[k] or {}

            new_inner_state, new_inner_info = _kernels[k](
                rng_keys[0], state.inner_state[k], logdensity_k, **kwargs
            )

            if num_steps > 1:
                carry = (new_inner_state, new_inner_info)

                def one_step(carry, key):
                    state, info = carry
                    state, info = _kernels[k](key, state, logdensity_k, **kwargs)
                    return (state, info), None

                (new_inner_state, new_inner_info), _ = jax.lax.scan(
                    one_step, carry, jnp.array(rng_keys[1:])
                )

            new_position[k] = new_inner_state.position
            inner_state[k] = new_inner_state
            inner_info[k] = new_inner_info

        return GibbsState(position=new_position, inner_state=inner_state), GibbsInfo(
            inner_info=inner_info
        )

    return kernel


class gibbs:
    """Implements a Gibbs sampler."""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        inner_kernel: Dict,
        inner_kernel_kwargs: Optional[Dict] = None,
        inner_kernel_steps: Optional[Dict[str, int]] = None,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(inner_kernel, inner_kernel_kwargs, inner_kernel_steps)

        def init_fn(position: Array, rng_key=None):
            return cls.init(
                position,
                logdensity_fn,
                inner_kernel,
                rng_key=rng_key,
            )

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                inner_kernel_kwargs=inner_kernel_kwargs,
                inner_kernel_steps=inner_kernel_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)
