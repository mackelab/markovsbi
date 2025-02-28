from markovsbi.tasks.base import Task

import jax
from jax import Array
import jax.numpy as jnp


from markovsbi.utils.prior_utils import Normal


class SimpleND(Task):
    def __init__(self, D: int, x0_min=-100, x0_max=100):
        self.D = D
        self.x0_min = x0_min
        self.x0_max = x0_max
        super().__init__("SimpleN{}".format(D))

    @property
    def input_shape(self):
        return (self.D,)

    @property
    def condition_shape(self):
        return (self.D,)

    def get_data(self, key, num_simulations, T):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        x0 = jax.random.uniform(
            key2, (num_simulations, self.D), minval=self.x0_min, maxval=self.x0_max
        )
        xs = simulators(key3, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_prior(self):
        return Normal(jnp.zeros((self.D,)), jnp.ones((self.D,)))

    def get_simulator(self):
        def simulator(rng_key, theta: Array, T: int, x0=None):
            def one_step(x_t, key):
                return x_t + theta + jax.random.normal(key, shape=(self.D,))

            if x0 is None:
                x0 = jnp.zeros((self.D,))
            keys = jax.random.split(rng_key, T - 1)
            xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([x0, xs], axis=0)
            return xs

        return simulator

    def get_true_posterior(self, x_o: Array):
        T = x_o.shape[0] - 1
        x_first = x_o[0, None]
        x_last = x_o[-1, None]

        mean_posterior = (x_last - x_first) / (T + 1)
        std_posterior = 1 / jnp.sqrt(T + 1)
        return Normal(mean_posterior, std_posterior)


class Simple1D(SimpleND):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class Simple2D(SimpleND):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)


class SimpleNDStationary(Task):
    def __init__(self, D: int, alpha: float = 0.9, sigma0_scale=1.0):
        self.D = D
        self.alpha = alpha
        # Asymptotic variance of the stationary distribution
        self.sigma = jnp.sqrt(1 / (1 - self.alpha) ** 2) * sigma0_scale
        super().__init__("SimpleN{}Stationary".format(self.D))

    @property
    def input_shape(self):
        return (self.D,)

    @property
    def condition_shape(self):
        return (self.D,)

    def get_data(self, key, num_simulations, T):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        x0 = jax.random.normal(key2, (num_simulations, self.D)) * self.sigma
        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_prior(self):
        return Normal(jnp.zeros((self.D,)), jnp.ones((self.D,)))

    def get_simulator(self):
        def simulator(rng_key, theta: Array, T: int, x0=None):
            def one_step(x_t, key):
                x_t = self.alpha * x_t + theta + jax.random.normal(key, shape=(self.D,))
                return x_t, x_t

            if x0 is None:
                x0 = jnp.zeros((self.D,))
            keys = jax.random.split(rng_key, T - 1)
            _, xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([x0[None, ...], xs], axis=0)
            return xs

        return simulator

    def get_true_posterior(self, x_o: Array):
        T = x_o.shape[0] - 1
        x_next = x_o[1:]
        x_current = x_o[:-1]
        mean_posterior = (
            1 / (T + 1) * jnp.sum((x_next - self.alpha * x_current), axis=0)
        )
        std_posterior = 1 / jnp.sqrt(T + 1)
        return Normal(mean_posterior, std_posterior)


class Simple1DStationary(SimpleNDStationary):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class Simple2DStationary(SimpleNDStationary):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)


class Simple10DStationary(SimpleNDStationary):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)


class SimpleNDNonstationary(Task):
    def __init__(self, D: int, alpha: float = 0.9):
        self.D = D
        self.alpha = alpha
        # Asymptotic variance of the stationary distribution
        self.sigma = jnp.sqrt(1 / (1 - self.alpha) ** 2)
        super().__init__("SimpleN{}Nonstationary".format(self.D))

    @property
    def input_shape(self):
        return (self.D,)

    @property
    def condition_shape(self):
        return (self.D,)

    def get_data(self, key, num_simulations, T, max_T):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3, key4 = jax.random.split(key, 4)
        thetas = prior.sample(key1, (num_simulations,))
        x0 = jax.random.normal(key2, (num_simulations, self.D)) * self.sigma
        t0 = jax.random.randint(key4, (num_simulations, 1), 0, max_T - 1)

        ts = t0 + jnp.arange(0, T)[None, ...]

        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0, 0))(
            keys, thetas, T, ts, x0
        )  # x0 = None is violated here.

        return {"thetas": thetas, "xs": xs, "ts": ts}

    def get_prior(self):
        return Normal(jnp.zeros((self.D,)), jnp.ones((self.D,)))

    def get_simulator(self):
        def simulator(rng_key, theta: Array, T: int, ts=None, x0=None):
            def one_step(state, t):
                x_t, key = state
                key, key_noise = jax.random.split(key)
                x_t = (
                    (self.alpha + (1 / (t + 1))) * x_t
                    + theta
                    + jax.random.normal(key_noise, shape=(self.D,))
                )
                return (x_t, key), x_t

            if x0 is None:
                x0 = jnp.zeros((self.D,))

            if ts is None:
                ts = jnp.arange(1, T + 1)

            _, xs = jax.lax.scan(one_step, (x0, rng_key), ts[1:])
            xs = jnp.concatenate([x0[None, ...], xs], axis=0)
            return xs

        return simulator

    def get_true_posterior(self, x_o: Array):
        d = x_o.shape[1]
        print(d)
        T = x_o.shape[0] - 1
        x_next = x_o[1:]
        x_current = x_o[:-1]

        time_correction = jnp.array([1 / (t + 1) for t in range(T)])

        time_correction = jnp.stack([time_correction for _ in range(d)], axis=1)
        print(time_correction.shape)
        print(((0.9 + time_correction) * x_current).shape)
        mean_posterior = (1 / (T + 1)) * jnp.sum(
            (x_next - (0.9 + time_correction) * x_current), axis=0
        )
        std_posterior = 1 / jnp.sqrt(T + 1)

        # x_o = x_o.reshape(-1)
        # T = x_o.shape[0] - 1
        # x_next = x_o[1:]
        # x_current = x_o[:-1]
        # time_correction = jnp.array([1 / (t + 1) for t in range(T)])
        # mean_posterior = (1 / (T + 1)) * jnp.sum(
        #    (x_next - (self.alpha + time_correction) * x_current), axis=0
        # )

        # std_posterior = 1 / jnp.sqrt(T + 1)

        return Normal(mean_posterior, std_posterior)


class Simple1DNonstationary(SimpleNDNonstationary):
    def __init__(self):
        super().__init__(1)


class Simple2DNonstationary(SimpleNDNonstationary):
    def __init__(self):
        super().__init__(2)


class Simple10DNonstationary(SimpleNDNonstationary):
    def __init__(self):
        super().__init__(10)
