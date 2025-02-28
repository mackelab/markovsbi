
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from .base import FilterAPI
from markovsbi.tasks.inference.smc.resampling import resample_systematic
from blackjax.smc.ess import ess


class ParticleFilterState(NamedTuple):
    particles: ArrayLike
    log_weights: ArrayLike
    t: Optional[ArrayLike]


class ParticleFilterInfo(NamedTuple):
    ancestors: ArrayLike
    log_likelihood: ArrayLike
    ess: ArrayLike
    is_observed: bool


def init(
    particles: ArrayLike,
    t: Optional[ArrayLike] = None,
    log_weights: Optional[ArrayLike] = None,
) -> ParticleFilterState:
    # Initialize state for a particle filter
    num_particles = particles.shape[0]
    if log_weights is None:
        log_weights = jnp.full((num_particles,), fill_value=-jnp.log(num_particles))
    else:
        assert log_weights.shape[0] == num_particles, (
            "The number of particles and log weights must be the same."
        )
        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    return ParticleFilterState(particles, log_weights, t)


def resample_when_ess_below(ess: float, ess_threshold: float = 0.5):
    return ess < ess_threshold


def build_kernel(
    log_likelihood_fn: Callable,
    transition_fn: Callable,
    transition_logdensity_fn: Optional[Callable] = None,
    proposal_transition_fn: Optional[Callable] = None,
    proposal_logdensity_fn: Optional[Callable] = None,
    resample_criterion: Callable = resample_when_ess_below,
    resample_fn: Callable = resample_systematic,
    unbiased_gradients: bool = False,
):
    """Build a particle filter kernel.

    Args:
        log_likelihood_fn (Callable): Log likelihood function of the model.
        transition_fn (Callable): Transition function p(x_t|x_{t-1}) of the model.
        transition_logdensity_fn (Optional[Callable], optional): Computes logdensity function of the transition function. Defaults to None.
        proposal_transition_fn (Optional[Callable], optional): Transition based on a proposal. Defaults to None.
        proposal_logdensity_fn (Optional[Callable], optional): Proposal density_fn. Defaults to None.
        resample_criterion (Callable, optional): _description_. Defaults to resample_when_ess_below.
        resample_fn (Callable, optional): _description_. Defaults to systematic.
        unbiased_gradients (bool, optional): _description_. Defaults to True.
    """

    def kernel(
        state: ParticleFilterState,
        t: Optional[float | int] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[ArrayLike] = None,
    ):
        assert rng_key is not None, (
            "You must provide a random key for the particle filter kernel."
        )
        # Unpack state
        particles = state.particles
        log_weights = state.log_weights
        rng_key, rng_key_predict, rng_key_resample = jax.random.split(rng_key, 3)
        log_num_particles = jnp.log(particles.shape[0])

        # Predict new particles
        if proposal_logdensity_fn is None:
            new_particles = transition_fn(rng_key_predict, particles, t)
        else:
            new_particles = proposal_transition_fn(rng_key_predict, particles, t)

        # Update weights
        is_observed = observed is not None
        if is_observed:
            # Update step
            log_weights = log_likelihood_fn(new_particles, observed, t) + log_weights
            if (
                transition_logdensity_fn is not None
                and proposal_logdensity_fn is not None
            ):
                log_weights = log_weights + (
                    transition_logdensity_fn(new_particles, particles, t)
                    - proposal_logdensity_fn(new_particles, particles, t)
                )
            log_normalizer = jax.scipy.special.logsumexp(log_weights)
            log_weights = log_weights - log_normalizer
            log_likelihood = log_normalizer - log_num_particles

        else:
            if (
                transition_logdensity_fn is not None
                and proposal_logdensity_fn is not None
            ):
                log_weights += transition_logdensity_fn(
                    new_particles, particles, t
                ) - proposal_logdensity_fn(new_particles, particles, t)
                log_normalizer = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_normalizer
            else:
                new_log_weights = log_weights
            log_likelihood = 0.0  # Without observation we don't have a logZ

        # Resample if necessary
        effective_samples_size = ess(log_weights)
        do_resample = resample_criterion(effective_samples_size / log_weights.shape[0])

        def resample(key, log_weights, particles):
            new_particles, new_log_weights, idx = resample_fn(
                key, log_weights, particles
            )
            if unbiased_gradients:
                new_log_weights += log_weights[idx] - jax.lax.stop_gradient(
                    log_weights[idx]
                )

            return new_particles, new_log_weights, idx

        def no_resample(key, log_weights, particles):
            return particles, log_weights, jnp.arange(particles.shape[0])

        new_particles, new_log_weights, ancestors = jax.lax.cond(
            do_resample,
            resample,
            no_resample,
            rng_key_resample,
            log_weights,
            new_particles,
        )

        new_state = ParticleFilterState(new_particles, new_log_weights, t)
        info = ParticleFilterInfo(
            ancestors, log_likelihood, effective_samples_size, is_observed
        )

        return new_state, info

    return kernel


class ParticleFilter(FilterAPI):
    """Particle filter inference algorithm.

    This class implements the particle filter algorithm. The particle filter is a sequential Monte Carlo method that approximates the filtering distribution of a state-space model.
    The particle filter is a generalization of the Kalman filter to non-linear and non-Gaussian models.

    To build a particle filter, you need to provide the following functions:
    Args:
        log_likelihood_fn (Callable): Log likelihood function of the model $p(y_t|x_t).
        transition_fn (Callable): Transition function $p(x_t|x_{t-1})$ of the model.
        transition_logdensity_fn (Optional[Callable], optional): Computes logdensity function of the transition function. Defaults to None.
        proposal_transition_fn (Optional[Callable], optional): Transition based on a proposal. Defaults to None.
        proposal_logdensity_fn (Optional[Callable], optional): Proposal density_fn. Defaults to None.
        resample_criterion (Callable, optional): _description_. Defaults to resample_when_ess_below.
        resample_fn (Callable, optional): _description_. Defaults to systematic.
        unbiased_gradients (bool, optional): _description_. Defaults to True.
    """

    init = init
    build_kernel = build_kernel
