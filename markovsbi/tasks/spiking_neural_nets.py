from markovsbi.tasks.base import Task

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from markovsbi.utils.prior_utils import Normal


def input_current(t, t_on=25.0, t_off=75.0, I_max=1.0):
    return jnp.where((t > t_on) & (t < t_off), I_max, 0.0)


class LIFSpikingNN(Task):
    def __init__(
        self,
        n_neurons: int = 10,
        dt: float = 0.005,
        inner_steps: int = 100,
        i_on: float = -jnp.inf,  # We need to condition the model on this if provided
        i_off: float = jnp.inf,  # We need to condition the model on this if provided
        threshold: float = 1.0,
        reset: float = 0.0,
        max_mu: float = 2.0,
        min_mu: float = 0.1,
        sigma: float = 0.01,
        weight_scale: float = 0.9,
    ):
        super().__init__("lif")
        self.n_neurons = n_neurons
        self.dt = dt
        self.inner_steps = inner_steps
        self.i_on = i_on
        self.i_off = i_off
        self.num_params = n_neurons + n_neurons**2
        self.threshold = threshold
        self.reset = reset
        self.max_mu = max_mu
        self.min_mu = min_mu
        self.sigma = sigma
        self.weight_scale = weight_scale

    def reparametrize(self, theta: ArrayLike) -> ArrayLike:
        mu = theta[: self.n_neurons]
        mu = jax.nn.sigmoid(mu)
        mu = mu * (self.max_mu - self.min_mu) + self.min_mu

        W_synaptic = theta[self.n_neurons :].reshape(self.n_neurons, self.n_neurons)
        # W_synaptic = W_synaptic / jnp.linalg.matrix_norm(W_synaptic, ord=2)
        # W_synaptic = W_synaptic * self.weight_scale

        return mu, W_synaptic

    def get_prior(self):
        return Normal(jnp.zeros(self.num_params), jnp.ones(self.num_params))

    def get_simulator(self):
        def simulator(
            key: ArrayLike, theta: ArrayLike, T: int, x0=None, return_spikes=False
        ) -> ArrayLike:
            mu, W_synaptic = self.reparametrize(theta)
            dt = self.dt
            if x0 is None:
                V0 = jnp.zeros(self.n_neurons)
            else:
                V0 = x0

            def step(V, key):
                def inner_step(V, key):
                    dWt = jax.random.normal(key, shape=V.shape) * jnp.sqrt(dt)
                    synaptic_input = W_synaptic @ V
                    dV = (
                        mu
                        * (
                            -V
                            + synaptic_input
                            + input_current(1.0, t_on=self.i_on, t_off=self.i_off)
                        )
                        * dt
                    )  # TODO Time dependent input current
                    V_new = V + dV + self.sigma * dWt

                    spike_new = jnp.where(V_new > self.threshold, 1.0, 0.0)
                    V_new = jnp.where(V_new > self.threshold, self.reset, V_new)

                    return V_new, spike_new

                keys = jax.random.split(key, self.inner_steps)
                V_new, spikes = jax.lax.scan(inner_step, V, keys)
                if return_spikes:
                    spiked = jnp.any(spikes, axis=0)
                    return V_new, (V_new, spiked)
                else:
                    return V_new, V_new

            keys = jax.random.split(key, T)
            _, V = jax.lax.scan(step, V0, keys)
            return V

        return simulator

    def get_data(self, key, num_simulations, T) -> ArrayLike:
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        x0 = (
            jax.random.beta(
                key2,
                a=6,
                b=1,
                shape=(num_simulations, self.n_neurons),
            )
            * 3
            * self.threshold
            - 1.5 * self.threshold
        )
        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)
        return {"thetas": thetas, "xs": xs}


class SparseLIFNetwork(LIFSpikingNN):
    def __init__(
        self,
        n_neurons: int = 10,
        sparsity: float = 0.1,
        sparsity_key=jax.random.PRNGKey(0),
        **kwargs,
    ):
        super().__init__(n_neurons=n_neurons, **kwargs)
        self.sparsity = sparsity
        self.connectivty_mask = jax.random.bernoulli(
            sparsity_key, sparsity, (n_neurons, n_neurons)
        )
        num_weights = jnp.sum(self.connectivty_mask)
        self.num_params = n_neurons + num_weights

    def reparametrize(self, theta: ArrayLike) -> ArrayLike:
        mu = theta[: self.n_neurons]
        mu = jax.nn.sigmoid(mu)
        mu = mu * (self.max_mu - self.min_mu) + self.min_mu

        W_synaptic = jnp.zeros((self.n_neurons, self.n_neurons))
        ws = theta[self.n_neurons :]
        W_synaptic = W_synaptic.at[self.connectivty_mask].set(ws)

        return mu, W_synaptic
