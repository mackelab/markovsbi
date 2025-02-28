from functools import partial
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jax import Array
from blackjax.base import SamplingAlgorithm

from typing import NamedTuple, Callable

from markovsbi.tasks.inference.kernels.base import MCMCKernel


class SliceParams(NamedTuple):
    pass


class SliceKernel(MCMCKernel):
    params: SliceParams

    def __init__(
        self,
        log_density_fn: Callable,
        step_size: float = 0.5,
        max_steps: int = 100,
        slice_fn="linear",
        slice_param_fn=None,
        slice_fn_kwargs={},
    ) -> None:
        self.log_density_fn = log_density_fn
        self.step_size = step_size
        self.max_steps = max_steps

        if slice_fn == "linear":
            slice_fn = linear_slice_fn
            slice_param_fn = sample_random_direction
        elif slice_fn == "poly":
            slice_fn = partial(polynomial_slice_fn, **slice_fn_kwargs)
            slice_param_fn = partial(sample_random_polynomial, **slice_fn_kwargs)
        elif slice_fn == "axis":
            slice_fn = axis_slice_fn
            slice_param_fn = sample_random_index
        elif slice_fn == "none":
            slice_fn = lambda pos, theta: lambda t: pos + t
            slice_param_fn = lambda key, pos: None
        else:
            assert isinstance(slice_fn, Callable), (
                "Default not supported, please provide a callabel"
            )
            assert isinstance(slice_param_fn, Callable), (
                "For non defaults a slice param function must be provided"
            )

        self.init_fn = slice.init
        self.update_fn = slice.build_kernel(slice_fn, slice_param_fn)

    def init_params(self, position: Array):
        self.params = SliceParams()
        return self.params

    def init_state(self, position: Array, rng_key: PRNGKey):
        return self.init_fn(position, self.log_density_fn, rng_key)

    def __call__(self, rng_key: PRNGKey, state):
        return self.update_fn(
            rng_key, state, self.log_density_fn, self.max_steps, self.step_size
        )


class SliceState(NamedTuple):
    position: Array
    logdensity: Array
    random_arg_slice: Array


class SliceInfo(NamedTuple):
    num_evals: int
    proposal: SliceState


def sample_random_direction(key: PRNGKey, position: Array):
    direction = jax.random.normal(key, shape=position.shape)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    return direction


def linear_slice_fn(position: Array, theta: Array):
    def linear_slice(t: float):
        return position + t * theta

    return linear_slice


def sample_random_polynomial(key: PRNGKey, position: Array, degree: int = 3):
    a = jax.random.normal(key, (degree,) + position.shape)
    a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a


def polynomial_slice_fn(position: Array, theta: Array, degree: int = 3):
    degrees = jnp.arange(1, degree + 1)
    # Factorial of degrees
    factorial = jnp.cumprod(degrees)
    # Factors
    factors = 1.0 / factorial

    def polynomial_slice_fn(t: float):
        t_powers = jnp.power(t, degrees) * factors
        return position + jnp.sum(theta * t_powers[:, None], axis=0)

    return polynomial_slice_fn


def sample_random_index(key: PRNGKey, position: Array):
    idx = jax.random.randint(key, (), minval=0, maxval=position.shape[0])
    return idx


def axis_slice_fn(position: Array, theta: Array):
    def axis_slice_fn(t: float):
        new_position = position.at[theta].add(t)
        return new_position

    return axis_slice_fn


def init(position: Array, logdensity_fn, rng_key: Array):
    log_density = logdensity_fn(position)
    return SliceState(position, log_density, rng_key)


def build_kernel(
    slice_fn_builder: Callable = linear_slice_fn,
    slice_fn_arg: Callable = sample_random_direction,
):
    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        log_density_fn: Callable,
        max_steps: int = 100,
        step_size: float = 0.5,
        **kwargs,
    ):
        rng_key, key_slice, key_rejections = jax.random.split(rng_key, 3)
        direction = slice_fn_arg(key_slice, state.position, **kwargs)
        log_density = state.logdensity
        u = jax.random.uniform(key_rejections, shape=())
        y = jnp.squeeze(jnp.log(u) + log_density)

        slice_fn = slice_fn_builder(state.position, direction)

        t_lower, t_upper, evals = lower_upper_bracket(
            log_density_fn, slice_fn, y, step_size, max_steps
        )

        x_new, log_density, evals_reject = accept_reject_slice(
            log_density_fn, slice_fn, key_rejections, t_lower, t_upper, y, max_steps
        )

        new_state = SliceState(x_new, log_density, rng_key)
        info = SliceInfo(evals + evals_reject, state)
        return new_state, info

    return kernel


class slice:
    """Implement a generalized slice sampler."""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        max_steps: int = 100,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: Array, rng_key=None):
            return cls.init(position, logdensity_fn, rng_key=rng_key)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, max_steps, step_size)

        return SamplingAlgorithm(init_fn, step_fn)


def lower_upper_bracket(
    log_density_fn: Callable,
    slice_fn: Callable,
    log_density_bound: Array,
    step_size: float,
    max_steps: int,
):
    # Bracket expansion phase
    def cond_fn(carry):
        i, _, _, mask = carry
        return jnp.any(mask) & (i < max_steps)

    def body_fn(carry):
        i, t, step_size, mask = carry

        t += step_size
        x = slice_fn(t)
        potential = log_density_fn(x)
        mask = potential >= log_density_bound

        return (i + 1, t, step_size, mask)

    evals1, t_upper, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, jnp.array(step_size), step_size, True)
    )
    evals2, t_lower, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, jnp.array(-step_size), -step_size, True)
    )
    total_evals = evals1 + evals2

    return t_lower, t_upper, total_evals


def accept_reject_slice(
    log_density_fn: Callable,
    slice_fn: Callable,
    key: PRNGKey,
    t_lower: Array,
    t_upper: Array,
    log_density_bound: Array,
    max_steps: int,
):
    def cond_fn_reject(carry):
        i, _, _, _, _, _, mask_reject = carry
        return jnp.any(mask_reject) & (i < max_steps)

    def body_fn_reject(carry):
        i, key, t_lower, t_upper, _, h, mask_reject = carry

        key, key_reject = jax.random.split(key)
        t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
        x_new = slice_fn(t_new)
        h = log_density_fn(x_new)

        mask_reject = h <= log_density_bound

        new_t_lower = jax.lax.cond(t_new < 0, lambda: t_new, lambda: t_lower)
        new_t_upper = jax.lax.cond(t_new > 0, lambda: t_new, lambda: t_upper)

        return (i + 1, key, new_t_lower, new_t_upper, x_new, h, mask_reject)

    key, key_reject = jax.random.split(key)
    t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
    x_new = slice_fn(t_new)
    h = log_density_fn(x_new)
    mask_reject = h <= log_density_bound
    new_t_lower = jax.lax.cond(t_new <= 0, lambda: t_new, lambda: t_lower)
    new_t_upper = jax.lax.cond(t_new >= 0, lambda: t_new, lambda: t_upper)

    evals, _, _, _, x_new, h, _ = jax.lax.while_loop(
        cond_fn_reject,
        body_fn_reject,
        (1, key, new_t_lower, new_t_upper, x_new, h, mask_reject),
    )

    return x_new, h, evals
