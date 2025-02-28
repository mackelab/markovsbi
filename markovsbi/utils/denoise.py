import jax
import jax.numpy as jnp


from jax.typing import ArrayLike
from markovsbi.utils.prior_utils import (
    Empirical,
    Normal,
    MultivariateNormal,
    MixtureNormal,
    Distribution,
)

from markovsbi.utils.marginalize import marginalize


def denoise_tweedie(p: Distribution, m: ArrayLike, s: ArrayLike, x_t: ArrayLike):
    p_t = marginalize(p, m, s)
    score = p_t.score(x_t)
    x_0 = (x_t + s**2 * score) / m
    J = jax.jacfwd(p_t.score)(x_t)
    cov_0 = s**2 / m**2 * (jnp.eye(x_t.shape[0]) + s**2 * J)
    return x_0, cov_0


def denoise(p: Distribution, m: ArrayLike, s: ArrayLike, x_t: ArrayLike):
    if isinstance(p, Normal):
        return denoise_gaussian(p, m, s, x_t)
    elif isinstance(p, MultivariateNormal):
        return denoise_multivaraite_gaussian(p, m, s, x_t)
    # elif isinstance(p, Uniform):
    #     return denoise_uniform(p, m, s)
    elif isinstance(p, MixtureNormal):
        return denoise_mixture_normal(p, m, s, x_t)
    elif isinstance(p, Empirical):
        return denoise_empirical(p, m, s, x_t)
    else:
        raise NotImplementedError("Denoising not implemented for this distribution")


def denoise_gaussian(p: Normal, m: ArrayLike, s: ArrayLike, x_t: ArrayLike):
    mu0 = p.mu
    std0 = p.std

    # Calculate the posterior variance
    var = 1 / (1 / std0**2 + m**2 / s**2)

    # Calculate the posterior mean
    mean = var * (mu0 / std0**2 + m * (x_t / s**2))
    std = jnp.sqrt(var)

    return Normal(mean, std)


def denoise_multivaraite_gaussian(
    p: MultivariateNormal, m: ArrayLike, s: ArrayLike, x_t: ArrayLike
):
    mu0 = p.mu
    cov0 = p.cov

    var = jnp.linalg.inv(jnp.linalg.inv(cov0) + m**2 / s**2 * jnp.eye(cov0.shape[0]))
    mean = var @ (jnp.linalg.inv(cov0) @ mu0 + m / s**2 * x_t)
    return MultivariateNormal(mean, var)


def denoise_mixture_normal(
    p: MixtureNormal, m: ArrayLike, s: ArrayLike, x_t: ArrayLike
):
    mus = p.mus
    stds = p.stds
    log_weights = p.log_weights

    m = jnp.atleast_1d(m)
    s = jnp.atleast_1d(s)
    m = m[None, ...]
    s = s[None, ...]

    precission1 = m**2 / s**2
    mean1 = x_t / m

    precission2 = 1 / stds**2
    mean2 = mus

    var = 1 / (precission1 + precission2)
    mean = var * (precission1 * mean1 + precission2 * mean2)

    mean_diff = (mean1 - mean2) ** 2
    variances_added = 1 / precission1 + 1 / precission2
    log_term1 = -0.5 * jnp.log((2 * jnp.pi * variances_added))
    log_term2 = -0.5 * mean_diff / variances_added
    # print(log_term1, log_term2)
    weight_term = log_term1 + log_term2
    weight_term = jnp.sum(weight_term, axis=-1)

    log_weights += weight_term

    log_weights -= jax.scipy.special.logsumexp(log_weights, axis=0)

    std = jnp.sqrt(var)

    return MixtureNormal(mean, std, log_weights)


def denoise_empirical(p: Empirical, m: ArrayLike, s: ArrayLike, x_t: ArrayLike):
    mus = p.data
    stds = jnp.zeros_like(mus)
    weights = p.weights

    m = jnp.atleast_1d(m)
    s = jnp.atleast_1d(s)
    m = m[None, ...]
    s = s[None, ...]

    # Calculate the posterior variance
    var = 1 / (1 / stds**2 + m**2 / s**2)

    # Calculate the posterior mean
    mean = var * (mus / stds**2 + m * x_t / s**2)
    std = jnp.sqrt(var)

    # Calculate the posterior weights
    _f = lambda x, mean, std: jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std).sum(
        axis=-1
    )
    log_weights = jnp.log(weights)
    log_weights += jax.vmap(_f, in_axes=(None, 0, 0))(x_t, mean, std)
    log_weights -= jax.scipy.special.logsumexp(log_weights, axis=0)
    weights = jnp.exp(log_weights)

    return MixtureNormal(mean, std, weights)
