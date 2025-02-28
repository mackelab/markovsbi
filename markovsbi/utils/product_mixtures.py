from functools import partial
import jax
import jax.numpy as jnp


from markovsbi.utils.prior_utils import MixtureNormal
from markovsbi.utils.denoise import denoise
from jax.typing import ArrayLike

from markovsbi.utils.prior_utils import Normal


@partial(jax.jit, static_argnums=(6,))
def product_of_gaussian_mixture(
    log_w1: ArrayLike,
    m1: ArrayLike,
    p1: ArrayLike,
    log_w2: ArrayLike,
    m2: ArrayLike,
    p2: ArrayLike,
    normalize_weights: bool = False,
):
    """Compute the product of two Gaussian mixtures.

    Args:
        w1 (ArrayLike): Weights of the first Gaussian mixture (K,)
        m1 (ArrayLike): Means of the first Gaussian mixture (K, d)
        p1 (ArrayLike): Precisions of the first Gaussian mixture (K, d) or (K, d, d)
        w2 (ArrayLike): Weights of the second Gaussian mixture (K,)
        m2 (ArrayLike): Means of the second Gaussian mixture (K, d)
        p2 (ArrayLike): Precisions of the second Gaussian mixture (K, d) or (K, d, d)

    Returns:
        Tuple: New weights, means and precisions of the product Gaussian mixture
    """
    is_diag = p1.ndim <= 2
    K_new = log_w1.shape[0] * log_w2.shape[0]
    d = 0 if m1.ndim == 1 else m1.shape[1]

    # Compute new variances and stds using broadcasting

    new_precissions = p1[:, None, ...] + p2[None, :, ...]

    # Compute new means using broadcasting
    if is_diag:
        new_means = (
            1
            / new_precissions
            * (p1[:, None] * m1[:, None] + p2[None, :] * m2[None, :])
        )
    else:
        precission_adjusted_m1 = jnp.einsum("ij,ikj->ik", p1, m1)
        precission_adjusted_m2 = jnp.einsum("ij,ikj->ik", p2, m2)
        new_means = jnp.linalg.solve(
            new_precissions, precission_adjusted_m1 + precission_adjusted_m2
        ).ravel()

    # Compute the product of two Gaussian densities (unnormalized weights)
    if is_diag:
        mean_diff = (m1[:, None] - m2[None, :]) ** 2
        variances_added = 1 / p1[:, None] + 1 / p2[None, :]
        log_term1 = -0.5 * jnp.log((2 * jnp.pi * variances_added))
        log_term2 = -0.5 * mean_diff / variances_added
        weight_term = log_term1 + log_term2
        weight_term = jnp.sum(weight_term, axis=-1) if d > 1 else weight_term
        weight_term = jnp.squeeze(weight_term)
    else:
        raise NotImplementedError("Only exponent_term=True is supported")

    # print((log_w1[:, None] + log_w2[None, :]).shape, weight_term.shape)

    new_log_weights = ((log_w1[:, None] + log_w2[None, :]) + weight_term).ravel()

    if normalize_weights:
        new_log_weights -= jax.scipy.special.logsumexp(new_log_weights)

    # Return in right shapes
    new_means = new_means.reshape(K_new, d) if d > 1 else new_means.ravel()
    precission_shape = (K_new, d, d) if not is_diag else (K_new, d)
    new_precissions = (
        new_precissions.reshape(*precission_shape) if d > 1 else new_precissions.ravel()
    )

    return new_log_weights, new_means, new_precissions


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def product_of_mixtures(
    weights: list,
    means: list,
    precisions: list,
    normalize_weights=False,
    reduce_components=False,
    reduce_method="prune",
    min_components=50,
    max_components_scale=2,
):
    """Compute the product of a list of Gaussian mixtures.

    Args:
        weights (ArrayLike): Weights of the Gaussian mixtures (K, ...)
        means (ArrayLike): Means of the Gaussian mixtures (K, ..., d)
        precisions (ArrayLike): Precisions of the Gaussian mixtures (K, ..., d) or (K, ..., d, d)

    Returns:
        Tuple: New weights, means and precisions of the product Gaussian mixture
    """

    # Compute the product of the first two mixtures
    new_weights, new_means, new_precisions = product_of_gaussian_mixture(
        weights[0],
        means[0],
        precisions[0],
        weights[1],
        means[1],
        precisions[1],
        normalize_weights=normalize_weights,
    )

    # Compute the product of the rest of the mixtures
    for i in range(2, len(weights)):
        max_components = min_components + i * max_components_scale

        if reduce_components:
            # jax.debug.print(
            #     "Clusters before: {new_means}", new_means=new_means.shape[0]
            # )
            # jax.debug.print("{new_precisions}", new_precisions=new_precisions)

            if reduce_method == "prune":
                idx = jnp.argsort(new_weights)[::-1]
                new_weights = new_weights[idx][:max_components]
                new_means = new_means[idx][:max_components]
                new_precisions = new_precisions[idx][:max_components]
            elif reduce_method == "merge":
                new_weights, new_means, new_precisions = (
                    reduce_mixture_components_simple(
                        new_weights,
                        new_means,
                        new_precisions,
                        max_components=max_components,
                    )
                )
            else:
                raise ValueError(f"Unknown reduce method: {reduce_method}")

            # jax.debug.print("{w}", w=new_weights)
            # jax.debug.print("{new_means}", new_means=new_means)
            # jax.debug.print("{new_precisions}", new_precisions=new_precisions)
            # jax.debug.print("Clusters after: {new_means}", new_means=new_means.shape[0])

        print("Num components ", new_weights.shape[0])
        new_weights, new_means, new_precisions = product_of_gaussian_mixture(
            new_weights,
            new_means,
            new_precisions,
            weights[i],
            means[i],
            precisions[i],
            normalize_weights=normalize_weights,
        )

    return new_weights, new_means, new_precisions


def kl_divergence_gaussian(mean1, prec1, mean2, prec2):
    """
    Compute the KL divergence between two Gaussian distributions.
    Inputs:
        mean1, mean2: Means of the distributions (vectors).
        prec1, prec2: Precision matrices (or vectors if diagonal) of the distributions.
    """

    # Check if prec1 and prec2 are vectors or matrices
    if jnp.ndim(prec1) < 2:
        # Diagonal case (precision vectors)
        mean_diff = mean1 - mean2
        kl = 0.5 * (jnp.log(prec1 / prec2) + (1 / prec1 + mean_diff**2) * prec2 - 1.0)
    else:
        # Full precision matrices
        d = mean1.shape[0]
        precision1 = prec1
        precision2 = prec2

        cov2 = jnp.linalg.inv(precision2)
        mean_diff = mean1 - mean2

        term1 = jnp.trace(jnp.matmul(cov2, precision1))
        term2 = jnp.matmul(jnp.matmul(mean_diff.T, precision2), mean_diff)
        term3 = jnp.log(jnp.linalg.det(precision2) / jnp.linalg.det(precision1))

        kl = 0.5 * (term1 + term2 - d + term3)

    return kl.sum()


@partial(jax.jit, static_argnums=(3,))
def reduce_mixture_components_simple(weights, means, precissions, max_components):
    """Reduce the number of components in a Gaussian mixture."""

    if len(weights) <= max_components:
        return weights, means, precissions

    # Comute pairise distance between means
    idx = jnp.argsort(weights)[::-1]
    weights = weights[idx]
    means = means[idx]
    precissions = precissions[idx]

    # Largest weights first
    mew_weights = weights[: max_components // 2]
    mew_means = means[: max_components // 2]
    new_precissions = precissions[: max_components // 2]

    means = means[max_components // 2 :]
    weights = weights[max_components // 2 :]
    precissions = precissions[max_components // 2 :]

    mean_diff = means[:, None] - means[None, :]
    mean_dist = jnp.linalg.norm(mean_diff, axis=-1)

    # mean_dist = jnp.where(mean_dist < merge_tol, mean_dist, jnp.inf)
    mean_dist = mean_dist + jnp.diag(jnp.inf * jnp.ones(mean_dist.shape[0]))

    # Sort by distance
    idx = jnp.argsort(mean_dist.flatten())
    # Unravel indices
    i, j = jnp.unravel_index(idx, mean_dist.shape)
    # Remove duplicates due to symmetry

    i = i[::2][: max_components - max_components // 2]
    j = j[::2][: max_components - max_components // 2]

    # Merge components
    def merge_compontents(i, j):
        return merge_gaussians(
            weights[i], means[i], precissions[i], weights[j], means[j], precissions[j]
        )

    merged_weights, merged_means, merged_precissions = jax.vmap(merge_compontents)(i, j)

    new_weights = jnp.concatenate([mew_weights, merged_weights])
    new_means = jnp.concatenate([mew_means, merged_means])
    new_precissions = jnp.concatenate([new_precissions, merged_precissions])

    return new_weights, new_means, new_precissions


@partial(jax.jit, static_argnums=(3, 4, 5))
def reduce_mixture_components(
    weights, means, precissions, max_components, merge_tol=10.0, normalize_weights=False
):
    """Reduce the number of components in a Gaussian mixture."""
    # New parameters
    if weights.shape[0] <= max_components:
        return weights, means, precissions

    # Initialize new parameters
    weights_new = jnp.full((max_components,), -jnp.inf)
    means_new = jnp.zeros((max_components,) + means.shape[1:])
    precissions_new = jnp.ones((max_components,) + precissions.shape[1:])

    def scan_fn(carry, i):
        j, merged_components, weights_new, means_new, precissions_new = carry
        current_mean = means[i]
        current_prec = precissions[i]
        kl_divs = jax.vmap(kl_divergence_gaussian, in_axes=(0, 0, None, None))(
            means, precissions, current_mean, current_prec
        )
        kl_divs = jnp.where(merged_components, jnp.inf, kl_divs)
        kl_divs = kl_divs.at[i].set(jnp.inf)
        # jax.debug.print("{x}", x=kl_divs)
        # Find the component with the smallest KL divergence
        l = jnp.argmin(kl_divs)
        val = kl_divs[l]

        def merge_components(
            j, merged_components, weights_new, means_new, precissions_new
        ):
            # Merge the two components
            new_weight, new_mean, new_prec = merge_gaussians(
                weights[i],
                means[i],
                precissions[i],
                weights[l],
                means[l],
                precissions[l],
            )
            weights_new = weights_new.at[j].set(new_weight)
            means_new = means_new.at[j].set(new_mean)
            precissions_new = precissions_new.at[j].set(new_prec)
            merged_components = merged_components.at[l].set(True)
            merged_components = merged_components.at[i].set(True)
            j += 1
            return j, merged_components, weights_new, means_new, precissions_new

        def no_merge(j, merged_components, weights_new, means_new, precissions_new):
            def add_current_component(
                j, merged_components, weights_new, means_new, precissions_new
            ):
                weights_new = weights_new.at[j].set(weights[i])
                means_new = means_new.at[j].set(means[i])
                precissions_new = precissions_new.at[j].set(precissions[i])
                merged_components = merged_components.at[i].set(True)
                j += 1
                return j, merged_components, weights_new, means_new, precissions_new

            def drop_current_component(
                j, merged_components, weights_new, means_new, precissions_new
            ):
                return j, merged_components, weights_new, means_new, precissions_new

            results = jax.lax.cond(
                (~merged_components[i] & (j < max_components)),
                add_current_component,
                drop_current_component,
                j,
                merged_components,
                weights_new,
                means_new,
                precissions_new,
            )
            return results

        cond = val < merge_tol

        j, merged_components, weights_new, means_new, precissions_new = jax.lax.cond(
            cond,
            merge_components,
            no_merge,
            j,
            merged_components,
            weights_new,
            means_new,
            precissions_new,
        )

        return (j, merged_components, weights_new, means_new, precissions_new), None

    # Iterated by components, starting with largest weight
    indices = jnp.argsort(weights)[::-1]
    merged_components = jnp.array([False] * len(weights))
    j = 0
    init_carry = (j, merged_components, weights_new, means_new, precissions_new)
    final_carry, _ = jax.lax.scan(scan_fn, init_carry, indices)
    _, _, weights_new, means_new, precissions_new = final_carry

    if normalize_weights:
        weights_new -= jax.scipy.special.logsumexp(weights_new)

    return weights_new, means_new, precissions_new


def merge_gaussians(log_weight1, mean1, prec1, log_weight2, mean2, prec2):
    """Merge two Gaussian components using precision (inverse of variance)."""
    weight1 = jnp.exp(log_weight1)
    weight2 = jnp.exp(log_weight2)
    # Compute the new weight
    new_weight = jax.scipy.special.logsumexp(jnp.array([log_weight1, log_weight2]))

    # Precision is the inverse of variance, so total precision is the sum of weighted precisions
    new_prec = (weight1 * prec1 + weight2 * prec2) / jnp.exp(new_weight)

    # Compute the new mean using the precision-weighted average
    new_mean = (weight1 * prec1 * mean1 + weight2 * prec2 * mean2) / (
        jnp.exp(new_weight) * new_prec
    )

    return new_weight, new_mean, new_prec


def xi(m, p):
    d = m.shape[-1] if m.ndim > 0 else 1
    if p.ndim < 2:
        return -0.5 * (
            d * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(p)) + jnp.sum(m**2 * p)
        )
    else:
        mahalanobis = jnp.einsum("...i, ...ij, ...j", m, p, m)
        logdet = jnp.linalg.slogdet(p)[1]
        return -0.5 * (d * jnp.log(2 * jnp.pi) - logdet + mahalanobis)


def build_gauss_mixture_correction_term(
    prior: Normal,
    posteriors: list[MixtureNormal],
    reduce_components=False,
    max_components_scale=2,
    min_components=50,
):
    def correction_term(theta_a, m, s):
        denoise_prior = denoise(prior, m, s, theta_a)
        denoise_posteriors = [denoise(p, m, s, theta_a) for p in posteriors]

        log_weights = [p.log_weights for p in denoise_posteriors]
        means = [p.mus for p in denoise_posteriors]
        precissions = [1 / p.stds**2 + 0.5 for p in denoise_posteriors]

        posterior_log_weights, posterior_means, posterior_precissions = (
            product_of_mixtures(
                log_weights,
                means,
                precissions,
                reduce_components=reduce_components,
                max_components_scale=max_components_scale,
                min_components=min_components,
            )
        )

        T = len(posteriors)
        prior_mean = denoise_prior.mean
        prior_precission = 1 / denoise_prior.var

        # Ensure validity
        posterior_precissions = jnp.clip(
            posterior_precissions, min=(1 - T) * prior_precission + 0.1
        )

        Lams = posterior_precissions + (1 - T) * prior_precission
        etas = (
            posterior_precissions * posterior_means
            + (1 - T) * prior_precission * prior_mean
        )

        # jax.debug.print("{Lams}", Lams=Lams)
        xi_posteriors = jax.vmap(xi)(posterior_means, posterior_precissions)
        xi_prior = xi(prior_mean, prior_precission)
        xi_alls = jax.vmap(xi)(1 / Lams * etas, Lams)

        total = xi_posteriors + (1 - T) * xi_prior - xi_alls

        return jax.scipy.special.logsumexp(posterior_log_weights + total)

    return correction_term


def build_gauss_mixture_correction_term_approximate(
    prior: Normal, posteriors: list[MixtureNormal]
):
    def correction_term(theta_a, m, s):
        denoise_prior = denoise(prior, m, s, theta_a)
        denoise_posteriors = [denoise(p, m, s, theta_a) for p in posteriors]

        xs = jnp.linspace(-15, 15, 5000)
        log_prob_prior = jax.vmap(denoise_prior.log_prob)(xs)

        log_prob_posteriors = [
            jax.vmap(posterior1.log_prob)(xs) for posterior1 in denoise_posteriors
        ]
        T = len(posteriors)
        log_prob_prod = (1 - T) * log_prob_prior + sum(log_prob_posteriors)

        prob_prod = jnp.exp(log_prob_prod)
        integral = jnp.trapezoid(prob_prod, xs)
        correction = jnp.log(integral)

        return correction

    return correction_term


def kmeans(X, k, max_iter=200):
    # Initialize centroids randomly
    centroids = X[:k]

    # Define a function to assign points to clusters
    def assign_points(X, centroids):
        distances = jnp.sum((X[:, None] - centroids) ** 2, axis=-1)
        return jnp.argmin(distances, axis=-1)

    # Define a function to update centroids
    def update_centroids(X, clusters, centroids, k):
        for i in range(k):
            cluster_mask = clusters == i
            cluster_mask = cluster_mask[:, None]
            centeroid = jnp.sum(cluster_mask * X, axis=0) / jnp.sum(cluster_mask)
            centroids = centroids.at[i].set(centeroid)
        return centroids

    def scan_fn(carry, _):
        centroids, clusters = carry
        clusters = assign_points(X, centroids)
        new_centroids = update_centroids(X, clusters, centroids, k)
        return (new_centroids, clusters), None

    # Iterate until convergence
    initial_clusters = jnp.arange(len(X)) % k
    init_carry = (centroids, initial_clusters)
    final_carry, _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(max_iter))
    centroids, clusters = final_carry

    return clusters, centroids


def compute_mixture_params(X, centroids, clusters):
    mean = centroids
    K = len(centroids)
    weights = jnp.bincount(clusters, length=K) / X.shape[0]

    stds = jnp.zeros((K, X.shape[-1]))
    for i in range(K):
        cluster_mask = clusters == i
        cluster_mask = cluster_mask[:, None]
        mean_diff = X - mean[i]
        variance = jnp.sum(cluster_mask * mean_diff**2, axis=0) / (
            jnp.sum(cluster_mask) - 1
        )
        stds = stds.at[i].set(jnp.sqrt(variance))

    return weights, mean, stds
