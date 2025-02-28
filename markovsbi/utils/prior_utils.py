from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from abc import ABC, abstractmethod


class Distribution(ABC):
    _event_shape: tuple

    @property
    def event_shape(self) -> tuple:
        return self._event_shape

    @property
    def mean(self) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def var(self) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key: jnp.ndarray, shape: tuple = ()) -> jnp.ndarray:
        pass

    @abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        pass


@jax.tree_util.register_pytree_node_class
class GeneralDistribution(Distribution):
    def __init__(self, event_shape, sample_fn, log_prob_fn):
        self.sample_fn = sample_fn
        self.log_prob_fn = log_prob_fn
        self._event_shape = event_shape

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.log_prob_fn(x)

    def sample(self, key: jnp.ndarray, shape: tuple = ()) -> jnp.ndarray:
        return self.sample_fn(key, shape)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda x: self.log_prob_fn(x).sum())(x)

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.sample_fn, self.log_prob_fn)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "GeneralDistribution":
        return cls(aux_data["event_shape"], *children)


@jax.tree_util.register_pytree_node_class
class Normal(Distribution):
    def __init__(self, mu: ArrayLike, std: ArrayLike):
        self.mu = mu
        self.std = std
        self._event_shape = mu.shape

    @property
    def mean(self) -> jnp.ndarray:
        return self.mu

    @property
    def var(self) -> jnp.ndarray:
        return self.std**2

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        return jax.random.normal(key, shape + self.event_shape) * self.std + self.mu

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.stats.norm.logpdf(x, loc=self.mu, scale=self.std).sum(axis=-1)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return -(x - self.mu) / self.std**2

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.mu, self.std)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "Normal":
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Uniform(Distribution):
    def __init__(self, lower: ArrayLike, upper: ArrayLike):
        self.lower = lower
        self.upper = upper
        self._event_shape = lower.shape

    @property
    def mean(self) -> jnp.ndarray:
        return (self.lower + self.upper) / 2

    @property
    def var(self) -> jnp.ndarray:
        return (self.upper - self.lower) ** 2 / 12

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        return (
            jax.random.uniform(key, shape=shape + self.event_shape)
            * (self.upper - self.lower)
            + self.lower
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.stats.uniform.logpdf(
            x, loc=self.lower, scale=self.upper - self.lower
        ).sum(axis=-1)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        zeros = jnp.zeros_like(x)
        infs = jnp.inf * zeros
        return jnp.where(jnp.logical_and(x >= self.lower, x <= self.upper), zeros, infs)

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.lower, self.upper)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "Uniform":
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class MixtureNormal(Distribution):
    def __init__(self, mus: ArrayLike, stds: ArrayLike, log_weights: ArrayLike):
        self.mus = mus
        self.stds = stds
        self.log_weights = log_weights
        self._event_shape = mus.shape[1:]

    @property
    def mean(self) -> jnp.ndarray:
        weights = jnp.exp(self.log_weights)
        return (self.mus * weights[..., None]).sum(axis=0)

    @property
    def var(self) -> jnp.ndarray:
        weights = jnp.exp(self.log_weights)
        return ((self.stds**2 + self.mus**2) * weights[..., None]).sum(
            axis=0
        ) - self.mean**2

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(
            subkey,
            jnp.arange(self.mus.shape[0]),
            p=jnp.exp(self.log_weights),
            shape=shape,
        )
        return (
            jax.random.normal(key, shape + self.event_shape) * self.stds[idx]
            + self.mus[idx]
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.vmap(
            lambda mu, sigma: jax.scipy.stats.norm.logpdf(x, loc=mu, scale=sigma).sum(
                axis=-1, keepdims=True
            )
        )(self.mus, self.stds)

        return jnp.squeeze(
            jax.scipy.special.logsumexp(
                log_probs + self.log_weights[..., None], axis=0
            ),
            -1,
        )

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        scores = jax.vmap(lambda mu, sigma, w: -(x - mu) / sigma**2 * w)(
            self.mus, self.stds, jnp.exp(self.log_weights)
        )
        return scores.sum(axis=0)

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.mus, self.stds, self.log_weights)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "MixtureNormal":
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class MultivariateNormal(Distribution):
    def __init__(self, mu: ArrayLike, cov: ArrayLike):
        self.mu = mu
        self.cov = cov
        self._event_shape = mu.shape

    @property
    def mean(self) -> jnp.ndarray:
        return self.mu

    @property
    def var(self) -> jnp.ndarray:
        return jnp.diag(self.cov)

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        return jax.random.multivariate_normal(key, self.mu, self.cov, shape)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.linalg.solve(self.cov, (x - self.mu).T).T

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.mu, self.cov)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "MultivariateNormal":
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Empirical(Distribution):
    def __init__(self, data: ArrayLike, weights: Optional[ArrayLike] = None):
        self.data = data
        if weights is None:
            weights = jnp.ones(data.shape[0]) / data.shape[0]
        self.weights = weights
        self._event_shape = data.shape[1:]

    @property
    def mean(self) -> jnp.ndarray:
        return self.data.mean(axis=0)

    @property
    def var(self) -> jnp.ndarray:
        return self.data.var(axis=0)

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        idx = jax.random.choice(
            key, jnp.arange(self.data.shape[0]), shape=shape, p=self.weights
        )
        return self.data[idx]

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(x.shape[0])

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(x.shape)

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        children = (self.data, self.weights)
        aux_data = {"event_shape": self._event_shape}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: Tuple) -> "Empirical":
        return cls(*children)
