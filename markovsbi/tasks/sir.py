from functools import partial
import math
import warnings
from markovsbi.tasks.base import Task

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

from markovsbi.utils.prior_utils import GeneralDistribution, Normal

from markovsbi.tasks.inference.filter import filter_log_likelihood
from markovsbi.tasks.inference.filtering.particle_filter import ParticleFilter
from markovsbi.tasks.inference.kernels import gibbs
import blackjax


def drift_sir(t, x, gamma, beta):
    S = x[0]
    I = x[1]
    R = x[2]
    # predator and prey are swapped
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I

    dx = jnp.array([dS, dI, dR])
    return dx


def diffusion_sir(t, x, sigma1=0.1, sigma2=0.05):
    return jnp.array([[0.01, 0.0, 0.0], [0.0, x[1] * sigma1, 0.0], [0.0, 0.0, 0.01]])


def build_log_likelihood(task: Task, n_particles=200, obs_sigma=5e-2):
    simulator = task.get_simulator()

    def transition_fn(key, x, t, theta=None):
        keys = jax.random.split(key, x.shape[0])
        _transiton = lambda k, x: simulator(k, theta, 2, x0=x)[-1]
        return jax.vmap(_transiton)(keys, x)

    def observed_log_likelihood_fn(x, y, t):
        # Approximate dirac delta

        return jax.scipy.stats.norm.logpdf(x, y, obs_sigma).sum(axis=-1)

    # Initialize particles at right values

    def unbiased_log_likelihood_fn(theta, key, x_o):
        _transiton_fn = partial(transition_fn, theta=theta)
        kernel = ParticleFilter(observed_log_likelihood_fn, _transiton_fn)
        ts = jnp.arange(x_o.shape[0])
        keys_init, key_transition = jax.random.split(key)
        keys = jax.random.split(keys_init, n_particles)
        init_particles = x_o[0] * jnp.ones((n_particles, 3))
        log_Z = filter_log_likelihood(
            key_transition, ts, ts[1:], x_o[1:], kernel, init_particles
        )
        return log_Z.sum()

    return unbiased_log_likelihood_fn


def build_sampler(
    task: Task,
    x_obs: ArrayLike,
    burn_in: int = 500,
    num_chains: int = 50,
    num_sir=50,
    thin=5,
    n_particles=200,
    obs_ll=None,
):
    if x_obs.shape[0] <= 3:
        obs_ll = 1e-3
        step_size = 0.2
    elif x_obs.shape[0] <= 20:
        obs_ll = 1e-2
        step_size = 0.1
    else:
        obs_ll = 1e-1
        step_size = 0.1

    if obs_ll is not None:
        obs_ll = jnp.array(obs_ll)

    log_likelihood = build_log_likelihood(
        task, obs_sigma=obs_ll, n_particles=n_particles
    )
    prior = task.get_prior()

    def log_density_fn(theta, key):
        return log_likelihood(theta, key, x_obs) + prior.log_prob(theta)

    T = x_obs.shape[0]
    # kernel1 = slice
    kernel1 = blackjax.rmh
    kernel2 = blackjax.irmh
    kernels = {"theta": kernel1, "key": kernel2}
    inner_kernel_kwargs = {
        # "theta": {"step_size": 0.05 + 1 / T, "max_steps": 20},
        "theta": {
            "transition_generator": lambda k, x: x
            + step_size * jax.random.normal(k, shape=x.shape)
        },
        "key": {
            "proposal_distribution": lambda key: jax.random.split(key)[1],
            "proposal_logdensity_fn": lambda x, x_old: 0.0,
        },
    }
    # inner_kernel_steps ={ "theta": 1, "key": 1}
    inner_kernel_steps = {"theta": 5, "key": 1}

    kernel = gibbs(
        log_density_fn,
        kernels,
        inner_kernel_kwargs=inner_kernel_kwargs,
        inner_kernel_steps=inner_kernel_steps,
    )

    def mcmc(key, num_steps):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        subkey1_propose, subkey_sir = jax.random.split(subkey1)
        subkeys_sir = jax.random.split(subkey_sir, num_sir)
        theta0 = prior.sample(subkey1_propose, (num_sir,))
        weights = jax.vmap(log_density_fn)(theta0, subkeys_sir)
        theta0 = theta0[jnp.argmax(weights)]
        init_key = subkey3
        position = {"theta": theta0, "key": init_key}
        state = kernel.init(position, subkey2)

        def one_step(state, key):
            state, _ = kernel.step(key, state)
            return state, state.position["theta"]

        keys = jax.random.split(key, num_steps)
        _, thetas = jax.lax.scan(one_step, state, keys)
        return thetas

    @partial(jax.jit, static_argnums=(1,))
    def sample(key, shape):
        num_samples = math.prod(shape)
        chains = min(num_samples, num_chains)
        num_samples = thin * (num_samples // chains) + burn_in
        keys = jax.random.split(key, chains)
        thetas = jax.vmap(partial(mcmc, num_steps=num_samples))(keys)
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


class SIR(Task):
    def __init__(
        self,
        x0_min=0.0,
        x0_max=5.0,
        dt=0.01,
        inner_steps=5,
        sigma1=0.1,
        sigma2=0.1,
        proposal="naive",
    ):
        super().__init__("SIR")
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.dt = dt
        self.inner_steps = inner_steps
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        def initial_dist(key):
            key1, key2 = jax.random.split(key)
            S0 = jax.random.uniform(
                key1, (), minval=self.x0_max / 2, maxval=self.x0_max
            )
            I = jax.random.uniform(key2, (), minval=0.0, maxval=self.x0_max / 2)
            return jnp.array([S0, I, 0.0])

        self.initial_distribution = initial_dist
        self.proposal = proposal

    @property
    def input_shape(self):
        return (2,)

    @property
    def condition_shape(self):
        return (3,)

    def get_prior(self):
        return Normal(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    def pred_proposal(self, key, num_sims=100, T_max=50):
        key1, key2 = jax.random.split(key)
        prior = self.get_prior()
        simulator = self.get_simulator()
        thetas_pred = prior.sample(key1, (num_sims,))
        x_pred = jax.vmap(lambda k, theta: simulator(k, theta, T_max))(
            jax.random.split(key2, num_sims), thetas_pred
        )
        x_pred_average = x_pred.reshape(-1, x_pred.shape[-1])

        def proposal(key):
            key, key2 = jax.random.split(key)
            idx = jax.random.randint(key, (), 0, x_pred_average.shape[0])
            x = x_pred_average[idx]
            eps = jax.random.normal(key2, x.shape) * x.shape[-1] / 10
            out = x + eps
            return jnp.where(out > 0, out, x)

        return proposal

    def get_simulator(self):
        def simulator(rng_key, theta: Array, T: int, x0=None):
            gamma, beta = theta
            gamma = jax.nn.sigmoid(gamma) * 2.0
            beta = jax.nn.sigmoid(beta) * 2.0

            if x0 is None:
                rng_key, key = jax.random.split(rng_key)
                x0 = self.initial_distribution(key)

            def one_step(x, key):
                def integrate_step(x, key):
                    xdt = drift_sir(0, x, gamma=gamma, beta=beta)
                    dWt = jax.random.normal(key, shape=(3,)) * jnp.sqrt(self.dt)
                    B = diffusion_sir(0, x)
                    x = x + xdt * self.dt + jnp.dot(B, dWt)
                    return x

                keys = jax.random.split(key, self.inner_steps)
                x = jax.lax.fori_loop(
                    0, self.inner_steps, lambda i, x: integrate_step(x, keys[i]), x
                )
                return x, x

            keys = jax.random.split(rng_key, T - 1)
            _, xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([x0[None], xs], axis=0)
            xs = jnp.clip(xs, 0)
            return xs

        return simulator

    def get_data(self, key, num_simulations, T, x0=None):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        if x0 is None:
            if self.proposal == "naive":
                x0 = jax.random.uniform(
                    key2, (num_simulations, 3), minval=self.x0_min, maxval=self.x0_max
                )
            elif self.proposal == "pred":
                key_construct, key_propose = jax.random.split(key2)
                proposal = self.pred_proposal(key_construct)
                x0 = jax.vmap(proposal)(jax.random.split(key_propose, num_simulations))
            else:
                pass
        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_true_posterior(
        self,
        x_o: ArrayLike,
        burn_in=500,
        num_chains=50,
        num_sir=50,
        thin=5,
        n_particles=500,
        obs_ll=None,
    ):
        sample_fn, log_prob_fn = build_sampler(
            self,
            x_o,
            burn_in=burn_in,
            num_chains=num_chains,
            num_sir=num_sir,
            thin=thin,
            n_particles=n_particles,
            obs_ll=obs_ll,
        )
        return GeneralDistribution(self.input_shape, sample_fn, log_prob_fn)
