from functools import partial
import math
from typing import Callable, Tuple
import warnings
from markovsbi.tasks.base import Task

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp
from jax.scipy.linalg import expm

from markovsbi.tasks.inference.filtering.kalman_filter import kalman_filter
from markovsbi.tasks.inference.filter import filter, filter_log_likelihood
# from .utils.linalg import matrix_fraction_decomposition

import blackjax
from markovsbi.utils.prior_utils import GeneralDistribution, Normal


def matrix_fraction_decomposition(
    t0: ArrayLike, t1: ArrayLike, A: ArrayLike, B: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Matrix fraction decomposition

    Returns the transition matrix and covariance. Is exact if A and B are truely time independent

    Args:
        t0 (float): New time point
        t1 (float): Old time point
        A (Array): Drift matrix
        B (Array): Diffusion matrix

    Returns:
        Tuple[Array]: Transition matrix and covariance
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[A, jnp.dot(B, B.T)], [jnp.zeros((d, d)), -A.T]])
    M = expm(blockmatrix * (t1 - t0))
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    return Phi, Q


def build_log_likelihood(task: Task):
    def log_likelihood(theta, x_obs):
        T = x_obs.shape[0]
        A = task.A(theta)
        B = task.B(theta)
        dt = task.dt * task.inner_steps
        transition_matrix, transtion_cov = matrix_fraction_decomposition(0, dt, A, B)

        mu0 = task.x0
        cov0 = jnp.zeros((mu0.shape[0], mu0.shape[0])) + jnp.eye(mu0.shape[0]) * 1e-5
        C = jnp.eye(mu0.shape[0])
        D = jnp.zeros((mu0.shape[0], mu0.shape[0])) + jnp.eye(mu0.shape[0]) * 1e-4

        kernel = kalman_filter(transition_matrix, transtion_cov, C, D)
        ts = jnp.arange(T)
        ll = filter_log_likelihood(
            jax.random.PRNGKey(0), ts, ts[1:], x_obs[1:], kernel, mu0, cov0
        )
        ll = jnp.nan_to_num(ll, nan=-1e6, posinf=-1e6, neginf=-1e6)
        return ll

    return log_likelihood


def build_filter(task: Task):
    def kf(theta, x_obs):
        T = x_obs.shape[0]
        A = task.A(theta)
        B = task.B(theta)
        dt = task.dt * task.inner_steps
        transition_matrix, transtion_cov = matrix_fraction_decomposition(0, dt, A, B)

        mu0 = task.x0
        cov0 = jnp.zeros((mu0.shape[0], mu0.shape[0])) + jnp.eye(mu0.shape[0]) * 1e-6
        C = jnp.eye(mu0.shape[0])
        D = jnp.eye(mu0.shape[0]) * 1e-6

        kernel = kalman_filter(transition_matrix, transtion_cov, C, D)
        ts = jnp.arange(T)
        states, _ = filter(
            jax.random.PRNGKey(0), ts, ts[1:], x_obs[1:], kernel, mu0, cov0
        )
        return states.mean, states.cov

    return kf


def build_sampler(
    task: Task,
    x_obs: ArrayLike,
    burn_in: int = 500,
    num_chains: int = 100,
    num_sir=50,
    thin=5,
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
                return step_size * jnp.where(acceptance_rate >= 0.6, 1.1, 0.9)

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


class LinearSDE(Task):
    def __init__(
        self,
        name: str,
        A: Callable,
        B: Callable | ArrayLike,
        x0: ArrayLike,
        dt: float = 0.005,
        sigma_data0: float = 1.0,
        inner_steps: int = 20,
        proposal="naive",
    ):
        super().__init__(name)
        self.A = A
        self.B = B
        self.x0 = x0
        self.dt = dt
        self.inner_steps = inner_steps
        self.sigma_data0 = sigma_data0
        self.proposal = proposal

    @property
    def input_shape(self):
        return self.x0.shape

    @property
    def condition_shape(self):
        return self.x0.shape

    def get_simulator(self):
        def simulator(rng_key, theta: ArrayLike, T: int, x0=None):
            if x0 is None:
                x0 = self.x0

            A = self.A(theta)
            B = self.B(theta)

            def one_step(x, key):
                def one_inner_step(x, key):
                    dWt = jax.random.normal(key, shape=(x.shape[-1],)) * jnp.sqrt(
                        self.dt
                    )
                    x = x + jnp.dot(A, x) * self.dt + jnp.dot(B, dWt)
                    return x, None

                keys = jax.random.split(key, self.inner_steps)
                x, _ = jax.lax.scan(one_inner_step, x, keys)
                return x, x

            keys = jax.random.split(rng_key, T - 1)
            _, xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([jnp.expand_dims(x0, 0), xs], axis=0)
            return xs

        return simulator

    def get_data(self, key, num_simulations, T):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        if self.proposal == "naive":
            x0 = (
                jax.random.normal(key2, shape=(num_simulations, self.x0.shape[0]))
                * self.sigma_data0
            )
        elif "pred":
            key_construct, key_propose = jax.random.split(key2)
            proposal = self.pred_proposal(key_construct)
            x0 = jax.vmap(proposal)(jax.random.split(key_propose, num_simulations))
        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_true_posterior(
        self, x_o: ArrayLike, burn_in=100, num_chains=100, num_sir=50
    ):
        sample_fn, log_prob_fn = build_sampler(
            self, x_o, burn_in=burn_in, num_chains=num_chains, num_sir=num_sir
        )
        event_shape = self.input_shape
        return GeneralDistribution(event_shape, sample_fn, log_prob_fn)


class PeriodicSDE(LinearSDE):
    def __init__(self, sigma1: float = 0.1, sigma2: float = 0.1, proposal="naive"):
        def A(theta):
            return jnp.array([[0.0, -(theta[0] ** 2)], [theta[1] ** 2, 0.0]])

        def B(theta):
            return jnp.array([[sigma1, 0.0], [0.0, sigma2]])

        x0 = jnp.array([-0.5, 0.5])
        super().__init__("prediodic_sde", A, B, x0, proposal=proposal)

    def get_prior(self):
        return Normal(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))


class DataHighDimSDE(LinearSDE):
    def __init__(self, x_dim=10, sigma=1.0, proposal="naive"):
        def A(theta):
            theta1 = jax.nn.sigmoid(theta[0])
            return (
                jnp.diag(-theta1 * jnp.ones(x_dim - 1), k=-1)
                + jnp.diag(theta1 * jnp.ones(x_dim - 1), k=1)
                - jnp.diag(jnp.ones(x_dim))
            )

        def B(theta):
            theta2 = jax.nn.sigmoid(theta[1])
            return jnp.eye(x_dim) * theta2 * sigma

        x0 = jnp.zeros(x_dim)
        super().__init__(
            "data_high_dim_sde", A, B, x0, dt=0.01, inner_steps=50, proposal=proposal
        )
        self.x_dim = x_dim
        self.sigma_data0 = sigma

    @property
    def input_shape(self):
        return (2,)

    @property
    def condition_shape(self):
        return (2 * self.x_dim,)

    def get_prior(self):
        return Normal(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))


class GeneralSDE(LinearSDE):
    @property
    def input_shape(self):
        return (2 * self.d**2,)

    @property
    def condition_shape(self):
        return (2 * self.d**2,)

    def __init__(self, d=2, proposal="naive"):
        self.d = d

        def A(theta):
            theta_sub = 0.5 * theta[: d**2].reshape((d, d))

            theta_sub = theta_sub - 2 * jnp.eye(d)
            return theta_sub

        def B(theta):
            theta_sub = 0.5 * theta[d**2 :].reshape((d, d)) + 0.5 * jnp.eye(d)
            return theta_sub

        x0 = jnp.zeros(d)
        super().__init__(
            "general_sde", A, B, x0, dt=0.001, inner_steps=20, proposal=proposal
        )

        self.sigma_data0 = 0.5

    def get_prior(self):
        return Normal(jnp.zeros((2 * self.d**2,)), jnp.ones((2 * self.d**2,)))

    def get_true_posterior(
        self, x_o: ArrayLike, burn_in=500, num_chains=100, num_sir=50
    ):
        sample_fn, log_prob_fn = build_sampler(
            self, x_o, burn_in=burn_in, num_chains=num_chains, num_sir=num_sir
        )
        return GeneralDistribution(self.input_shape, sample_fn, log_prob_fn)
