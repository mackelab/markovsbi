import jax
import jax.numpy as jnp
from jaxtyping import Array

import blackjax
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.adaptation.base import AdaptationInfo, AdaptationResults
from blackjax.base import AdaptationAlgorithm

from typing import Callable, NamedTuple



def cholesky_update(L, u):
    """
    Update the Cholesky decomposition of a matrix after a rank-1 update i.e.

    C = L @ L.T + multiplier * u @ u.T

    Args:
    L: A [D, D] lower triangular matrix, the Cholesky factor of the original matrix.
    u: A [D,] vector, the update vector.

    Returns:
    The updated [D, D] lower triangular matrix.
    """
    D = L.shape[0]
    indices = jnp.arange(D)

    def body_fun(i, vals):
        L, u = vals
        r = jnp.sqrt(L[i, i] ** 2 + u[i] ** 2)
        c = r / L[i, i]
        s = u[i] / L[i, i]
        L = L.at[i, i].set(r)

        mask = indices > i
        col_update = (L[:, i] + s * u) / c
        col_update = jnp.where(mask, col_update, L[:, i])
        L = L.at[:, i].set(col_update)
        u_update = c * u - s * L[:, i]
        u = jnp.where(mask, u_update, u)

        return (L, u)

    L, u = jax.lax.fori_loop(0, D, body_fun, (L, u))

    return L


# window_adaptation -> HMC, NUTs single chain
# chees_adaptation -> dynamic HMC, NUTs single chain
# meads_adaptation -> GHMC
# pathfinder_adaptation -> HMC, NUTs single chain


def step_size_adaption(
    algorithm: blackjax.rmh,
    logdensity_fn,
    params,
    transition_generator_fn,
    transition_logdensity_fn=None,
    target_acceptance_rate: float = 0.23,
    inital_step_size: float = 2.34,
    t0=10,
    gamma=0.05,
    kappa=0.75,
    **kwargs,
):
    if not hasattr(params, "step_size"):
        raise ValueError("The params object must have a step_size attribute")

    mcmc_kernel = algorithm.build_kernel()
    adapt_init, adapt_step, adapt_final = dual_averaging_adaptation(
        target_acceptance_rate,
        t0=t0,
        gamma=gamma,
        kappa=kappa,
    )

    def one_step(carry, key):
        state, adaption_state = carry
        step_size = jnp.exp(adaption_state.log_step_size)
        params_dict = params._asdict()
        params_dict["step_size"] = step_size
        updated_params = params.__class__(**params_dict)
        transition_generator = lambda k, x: transition_generator_fn(
            k, x, updated_params
        )
        if transition_logdensity_fn is not None:
            proposal_logdensity = lambda x, y: transition_logdensity_fn(
                x, y, updated_params
            )
        else:
            proposal_logdensity = None
        state, info = mcmc_kernel(
            key,
            state,
            logdensity_fn,
            transition_generator=transition_generator,
            proposal_logdensity_fn=proposal_logdensity,
            **kwargs,
        )
        adaption_state = adapt_step(adaption_state, info.acceptance_rate)
        return (state, adaption_state), AdaptationInfo(state, info, adaption_state)

    def run(rng_key, position, num_steps):
        initial_state = algorithm.init(position, logdensity_fn)
        init_adaptiation_state = adapt_init(jnp.log(inital_step_size))

        keys = jax.random.split(rng_key, num_steps)
        init_carry = (initial_state, init_adaptiation_state)
        (final_state, final_adaptation_state), info = jax.lax.scan(
            one_step, init_carry, keys
        )
        step_size = adapt_final(final_adaptation_state)
        parameters = {"step_size": step_size}
        result = AdaptationResults(final_state, parameters)
        return result, info

    return AdaptationAlgorithm(run)


def step_size_and_scale_adaption(
    algorithm: blackjax.rmh,
    logdensity_fn,
    params,
    transition_generator_fn,
    transition_logdensity_fn=None,
    target_acceptance_rate: float = 0.23,
    inital_step_size: float = 2.34,
    is_diagonal_matrix: bool = True,
    t0=10,
    gamma=0.05,
    kappa=0.75,
    **kwargs,
):
    if not hasattr(params, "step_size") or not hasattr(params, "scale"):
        raise ValueError("The params object must have a step_size and scale attribute")

    mcmc_kernel = algorithm.build_kernel()
    adapt_init, adapt_step, adapt_final = dual_averaging_adaptation(
        target_acceptance_rate,
        t0=t0,
        gamma=gamma,
        kappa=kappa,
    )
    adapt_init_scale, adapt_step_scale, adapt_final_scale = square_root_algorithm(
        is_diagonal_matrix=is_diagonal_matrix
    )

    def one_step(carry, key):
        state, ss_state, sr_state = carry
        step_size = jnp.exp(ss_state.log_step_size)
        scale, _, _ = adapt_final_scale(sr_state)
        params_dict = params._asdict()
        params_dict["step_size"] = step_size
        params_dict["scale"] = scale
        updated_params = params.__class__(**params_dict)
        transition_generator = lambda k, x: transition_generator_fn(
            k, x, updated_params
        )
        if transition_logdensity_fn is not None:
            proposal_logdensity = lambda x, y: transition_logdensity_fn(
                x, y, updated_params
            )
        else:
            proposal_logdensity = None
        state, info = mcmc_kernel(
            key,
            state,
            logdensity_fn,
            transition_generator=transition_generator,
            proposal_logdensity_fn=proposal_logdensity,
            **kwargs,
        )
        ss_state = adapt_step(ss_state, info.acceptance_rate)
        sr_state = adapt_step_scale(sr_state, jnp.array(state.position))
        return (state, ss_state, sr_state), AdaptationInfo(
            state, info, (ss_state, sr_state)
        )

    def run(rng_key, position, num_steps):
        initial_state = algorithm.init(position, logdensity_fn)
        ss_state = adapt_init(jnp.log(inital_step_size))
        sr_state = adapt_init_scale(position.shape[0])

        keys = jax.random.split(rng_key, num_steps)
        init_carry = (initial_state, ss_state, sr_state)
        (final_state, final_ss_state, final_sr_state), info = jax.lax.scan(
            one_step, init_carry, keys
        )
        step_size = adapt_final(final_ss_state)
        scale, _, _ = adapt_final_scale(final_sr_state)

        parameters = {"step_size": step_size, "scale": scale}
        result = AdaptationResults(final_state, parameters)
        return result, info

    return AdaptationAlgorithm(run)


class SquareRootState(NamedTuple):
    mean: Array
    L: Array
    sample_size: int


def square_root_algorithm(
    is_diagonal_matrix: bool,
) -> tuple[Callable, Callable, Callable]:
    def init(n_dims: int) -> SquareRootState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Parameters
        ----------
        n_dims: int
            The number of dimensions of the problem, which corresponds to the size
            of the corresponding square mass matrix.

        """
        sample_size = 0
        mean = jnp.zeros((n_dims,))
        if is_diagonal_matrix:
            L = jnp.ones((n_dims,))
        else:
            L = jnp.eye(n_dims)
        return SquareRootState(mean, L, sample_size)

    def update(sq_state: SquareRootState, value: Array) -> SquareRootState:
        mean, L, sample_size = sq_state
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size

        if is_diagonal_matrix:
            updated_delta = value - mean
            L = L + delta * updated_delta
        else:
            # This might be slightly biased, ...
            L = cholesky_update(L, delta)

        return SquareRootState(mean, L, sample_size)

    def final(
        sq_state: SquareRootState,
    ) -> tuple[Array, int, Array]:
        mean, L, sample_size = sq_state
        if is_diagonal_matrix:
            L = jnp.sqrt(L / (sample_size - 1))
        else:
            L = L / jnp.sqrt(sample_size - 1)

        return L, sample_size, mean

    return init, update, final
