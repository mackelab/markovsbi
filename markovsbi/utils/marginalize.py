from typing import Tuple
import jax
import jax.numpy as jnp


from jax.typing import ArrayLike
from markovsbi.utils.prior_utils import (
    GeneralDistribution,
    Normal,
    MultivariateNormal,
    MixtureNormal,
    Uniform,
    Distribution,
    Empirical,
)


def marginalize(p: Distribution, m: ArrayLike, s: ArrayLike):
    if isinstance(p, Normal):
        return marginalize_gaussian(p, m, s)
    elif isinstance(p, MultivariateNormal):
        return marginalize_multivariate_gaussian(p, m, s)
    elif isinstance(p, Uniform):
        return marginalize_uniform(p, m, s)
    elif isinstance(p, MixtureNormal):
        return marginalize_mixture_normal(p, m, s)
    elif isinstance(p, Empirical):
        return marginalize_empirical(p, m, s)
    else:
        raise NotImplementedError(
            "Marginalization not implemented for this distribution"
        )


def marginalize_gaussian(
    p: Normal, a: ArrayLike, s: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Diagonal Gaussian marginalization of y = a*x + s*eps, where eps ~ N(0,1)

    Args:
        mu1 (ArrayLike): Mean of x
        sigma1 (ArrayLike): Standard deviation of x
        a (ArrayLike): Scaling factor
        s (ArrayLike): Standard deviation of noise

    Returns:
        Tuple: New mean and standard deviation of y
    """

    new_mu = a * p.mu
    new_std = jnp.sqrt(a**2 * p.std**2 + s**2)

    return Normal(new_mu, new_std)


def marginalize_multivariate_gaussian(
    p: MultivariateNormal, m: ArrayLike, s: ArrayLike
):
    mu1 = p.mu
    cov1 = p.cov

    d = cov1.shape[-1]
    m = jnp.atleast_1d(m)
    s = jnp.atleast_1d(s)
    if m.shape[0] == 1:
        m = jnp.tile(m, d)
    if s.shape[0] == 1:
        s = jnp.tile(s, d)
    m = jnp.diag(m)
    s = jnp.diag(s)
    new_mu = m @ mu1
    new_cov = m @ cov1 @ m.T + s @ s.T

    return MultivariateNormal(new_mu, new_cov)


def marginalize_uniform(p: Uniform, m: ArrayLike, s: ArrayLike):
    a = p.lower
    b = p.upper
    diff = b - a
    logZ = -(m + jnp.log(diff).sum(axis=-1))

    event_shape = a.shape

    def sample_fn(key, shape):
        key1, key2 = jax.random.split(key)
        u = jax.random.uniform(key1, shape + event_shape) * (b - a) + a
        eps = jax.random.normal(key2, shape + event_shape)
        return m * u + s * eps

    def log_prob(x):
        t1 = jax.scipy.stats.norm.cdf(m * b, x, s)
        t2 = jax.scipy.stats.norm.cdf(m * a, x, s)
        diff = t1 - t2
        return jnp.log(diff).sum(axis=-1) + logZ

    return GeneralDistribution(event_shape, sample_fn, log_prob)


def marginalize_mixture_normal(p: MixtureNormal, m, s):
    pi = p.log_weights
    mus = p.mus
    stds = p.stds

    m = jnp.atleast_1d(m)
    s = jnp.atleast_1d(s)

    new_mus = m[None, ...] * mus
    new_stds = jnp.sqrt(m[None, ...] ** 2 * stds**2 + s[None, ...] ** 2)

    return MixtureNormal(new_mus, new_stds, pi)


def marginalize_empirical(p: Empirical, m, s) -> MixtureNormal:
    mus = p.data
    stds = jnp.zeros_like(mus)

    new_mus = m * mus
    new_stds = jnp.sqrt(m**2 * stds**2 + s**2)
    return MixtureNormal(new_mus, new_stds, p.weights)
