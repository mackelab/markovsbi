from functools import partial
import math
import warnings
from markovsbi.tasks.base import Task

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

import blackjax

from markovsbi.utils.prior_utils import GeneralDistribution, Normal


def build_log_likelihood(task: Task):
    def log_likelihood(theta, x_obs):
        mean1 = x_obs[:-1] + theta
        mean2 = x_obs[:-1] - theta
        xs = x_obs[1:]

        component1 = jax.scipy.stats.norm.logpdf(xs, loc=mean1, scale=task.sigma).sum(
            -1
        ) + jnp.log(0.5)
        component2 = jax.scipy.stats.norm.logpdf(xs, loc=mean2, scale=task.sigma).sum(
            -1
        ) + jnp.log(0.5)

        return jax.scipy.special.logsumexp(
            jnp.stack([component1, component2], axis=0), axis=0
        ).sum()

    return log_likelihood


def build_sampler(
    task: Task,
    x_obs: ArrayLike,
    burn_in: int = 500,
    num_chains: int = 100,
    num_sir=50,
    thin=6,
):
    log_likelihood = build_log_likelihood(task)
    prior = task.get_prior()

    def log_density_fn(theta):
        return log_likelihood(theta, x_obs) + prior.log_prob(theta)

    hmc_kernel = blackjax.hmc.build_kernel()
    inverse_mass_matrix = jnp.ones(task.input_shape)

    def mcmc(key, num_steps, burn_in):
        key, subkey = jax.random.split(key)
        theta0 = prior.sample(subkey, (num_sir,))
        weights = jax.vmap(log_density_fn)(theta0)
        theta0 = theta0[jnp.argmax(weights)]
        state = blackjax.hmc.init(theta0, log_density_fn)

        def one_step(carry, key):
            state, step_size, iter = carry
            state, info = hmc_kernel(
                key,
                state,
                logdensity_fn=log_density_fn,
                num_integration_steps=5,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            )
            acceptance_rate = info.acceptance_rate

            def update_step_size(step_size, acceptance_rate):
                return step_size * jnp.where(acceptance_rate >= 0.8, 1.1, 0.9)

            def not_update_step_size(step_size, acceptance_rate):
                return step_size

            step_size = jax.lax.cond(
                (iter < burn_in),
                update_step_size,
                not_update_step_size,
                step_size,
                acceptance_rate,
            )

            iter = iter + 1

            # jax.debug.print("step_size", step_size)

            return (state, step_size, iter), state.position

        keys = jax.random.split(key, num_steps)
        init_step_size = 0.1
        init_carry = (state, init_step_size, 0)
        _, thetas = jax.lax.scan(one_step, init_carry, keys)
        return thetas

    @partial(jax.jit, static_argnums=(1,))
    def sample(key, shape):
        num_samples = math.prod(shape)
        chains = min(num_samples, num_chains)
        num_samples = thin * (num_samples // chains) + burn_in
        keys = jax.random.split(key, chains)
        thetas = jax.vmap(partial(mcmc, num_steps=num_samples, burn_in=burn_in))(keys)
        thetas = thetas[:, burn_in:]
        thetas = thetas[:, ::thin]
        thetas = thetas.reshape(shape + (thetas.shape[-1],))
        return thetas[:, :num_samples]

    def log_prob(theta):
        warnings.warn("log_prob is not normalized")
        _log_density_fn = log_density_fn
        for _ in range(theta.ndim - 1):
            _log_density_fn = jax.vmap(_log_density_fn)
        return _log_density_fn(theta)

    return sample, log_prob


class MixtureRW(Task):
    def __init__(
        self,
        D: int,
        alpha=1.0,
        sigma0=10.0,
        sigma=0.5,
        x0_min=-3,
        x0_max=3,
        proposal="naive",
    ):
        self.D = D
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.sigma = sigma
        self.simga0 = sigma0
        self.alpha = alpha
        self.proposal = proposal
        super().__init__("SimpleN{}".format(D))

    @property
    def input_shape(self):
        return (self.D,)

    @property
    def condition_shape(self):
        return (self.D,)

    def get_data(self, key, num_simulations, T, x0=None):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        if x0 is None:
            if self.proposal == "naive":
                x0 = jax.random.normal(key2, (num_simulations, self.D)) * self.simga0
            elif self.proposal == "pred":
                key_construct, key_propose = jax.random.split(key2)
                proposal = self.pred_proposal(key_construct)
                x0 = jax.vmap(proposal)(jax.random.split(key_propose, num_simulations))

        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_prior(self):
        return Normal(jnp.zeros((self.D,)), jnp.ones((self.D,)))

    def get_simulator(self):
        def simulator(rng_key, theta: Array, T: int, x0=None):
            def one_step(x_t, key):
                key1, key2 = jax.random.split(key)
                u = jax.random.uniform(key1) - 0.5
                sign = jnp.sign(u)
                out = (
                    x_t
                    + sign * theta
                    + jax.random.normal(key2, shape=(self.D,)) * self.sigma
                )
                return out, out

            if x0 is None:
                x0 = jnp.zeros((self.D,))
            keys = jax.random.split(rng_key, T - 1)
            _, xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([x0[None, ...], xs], axis=0)
            return xs

        return simulator

    def get_true_posterior(
        self, x_o: ArrayLike, burn_in=600, num_chains=100, num_sir=50
    ):
        sample_fn, log_prob_fn = build_sampler(
            self, x_o, burn_in=burn_in, num_chains=num_chains, num_sir=num_sir
        )
        event_shape = self.input_shape
        return GeneralDistribution(event_shape, sample_fn, log_prob_fn)


class MixtureRW2D(MixtureRW):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class MixtureRW5D(MixtureRW):
    def __init__(self, *args, **kwargs):
        super().__init__(5, *args, **kwargs)
