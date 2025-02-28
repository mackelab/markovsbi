import jax
import jax.numpy as jnp

from typing import Callable, Optional
from jax.typing import ArrayLike
from markovsbi.models.utils import get_windows
from markovsbi.utils.marginalize import marginalize
from markovsbi.utils.denoise import denoise
from markovsbi.utils.product_mixtures import build_gauss_mixture_correction_term
from markovsbi.utils.util_linalg import (
    mv_diag_or_dense,
    solve_diag_or_dense,
    add_diag_or_dense,
)


class ScoreFn:
    """Base class for score functions.

    This will return the score of the score network. Child classes will transform the
    score to the desired form.
    """

    requires_hyperparameters: bool = False

    def __init__(self, score_net: Callable, params: ArrayLike, sde, prior) -> None:
        self.score_net = score_net
        self.params = params
        self.sde = sde
        self.prior = prior

    def estimate_hyperparameters(self, x_o, input_shape, key, t_o=None):
        pass

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.squeeze(base_score, axis=0)
        score = jnp.atleast_1d(base_score)
        return score


class FNPEScoreFn(ScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params: ArrayLike,
        sde,
        prior,
        prior_score_weight: Optional[Callable] = None,
    ) -> None:
        """The FNPE score function as proposed by Geffner et al. (2022).

        Args:
            score_net (Callable): Score estimation network of the form f(params,a,theta,x_o)
            params (ArrayLike): Parameters of the score network
            sde (_type_): SDE of forward model
            prior (_type_): Prior distribution
        """
        super().__init__(score_net, params, sde, prior)

        if prior_score_weight is None:

            def prior_score_weight(a):
                return (self.sde.T_max - a) / self.sde.T_max

        self.prior_score_weight = prior_score_weight

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)

        # Prior score
        prior_score = self.prior.score(theta)
        prior_score_marg_approx = self.prior_score_weight(a) * prior_score

        # Accumulate the scores
        N = base_score.shape[0]
        score = (1 - N) * prior_score_marg_approx + base_score.sum(0)
        return score


class UncorrectedScoreFn(ScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params: ArrayLike,
        sde,
        prior,
        marginal_prior_score_fn: Optional[Callable] = None,
    ) -> None:
        """The uncorrected score function

        Args:
            score_net (Callable): Score estimation network of the form f(params,a,theta,x_o)
            params (ArrayLike): Parameters of the score network
            sde (_type_): SDE of forward model
            prior (_type_): Prior distribution
            marginal_prior_score_fn (Callable): Function that returns the prior score $\nabla_\theta_a \log p_a(\theta_a)$. Should be of form f(a,theta).

        """
        super().__init__(score_net, params, sde, prior)
        if marginal_prior_score_fn is None:
            # Automatic marginalization of the prior
            def marginal_prior_score_fn(a, theta):
                m = sde.mu(a)
                std = sde.std(a)
                p_t = marginalize(self.prior, m, std)
                return p_t.score(theta)

        self.marginal_prior_score_fn = marginal_prior_score_fn

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)

        # Marginal prior score
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        return (1 - N) * prior_score + base_score.sum(0)


class UncorrectedScoreFn_improved(ScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params: ArrayLike,
        sde,
        prior,
        marginal_prior_score_fn: Optional[Callable] = None,
    ) -> None:
        """The uncorrected score function

        Args:
            score_net (Callable): Score estimation network of the form f(params,a,theta,x_o)
            params (ArrayLike): Parameters of the score network
            sde (_type_): SDE of forward model
            prior (_type_): Prior distribution
            marginal_prior_score_fn (Callable): Function that returns the prior score $\nabla_\theta_a \log p_a(\theta_a)$. Should be of form f(a,theta).

        """
        super().__init__(score_net, params, sde, prior)
        if marginal_prior_score_fn is None:
            # Automatic marginalization of the prior
            def marginal_prior_score_fn(a, theta):
                m = sde.mu(a)
                std = sde.std(a)
                p_t = marginalize(self.prior, m, std)
                return p_t.score(theta)

        self.marginal_prior_score_fn = marginal_prior_score_fn

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)

        # Marginal prior score
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        return (1 / N**a) * ((1 - N) * prior_score + base_score.sum(0))


class GaussCorrectedScoreFn(UncorrectedScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params,
        sde,
        prior,
        posterior_precission_est_fn: Optional[Callable | ArrayLike] = None,
        marginal_prior_score_fn: Optional[Callable] = None,
        marginal_denoising_prior_precission_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Corrected score function assuming a Gaussian posterior with variance determined by marginal_posterior_precission_est_fn.

        Args:
            score_net (Callable): Score estimation network of the form f(params,a,theta,x_o)
            params (_type_): Parameters of the score network
            sde (_type_): SDE of forward model
            T_min (_type_): Minimum time point
            T_max (_type_): Maximum time point
            marginal_prior_score_fn (_type_): Marginal prior score function
            marginal_prior_precission_fn (_type_): Marginal prior precission function
            marginal_posterior_precission_est_fn (_type_): Posterior precission estimation function
        """
        super().__init__(score_net, params, sde, prior, marginal_prior_score_fn)
        self.posterior_precission_est_fn = posterior_precission_est_fn
        if self.posterior_precission_est_fn is None:
            self.requires_hyperparameters = True
        else:
            self.requires_hyperparameters = False

        # Compute the marginal prior precission
        if marginal_denoising_prior_precission_fn is None:

            def marginal_denoising_prior_precission_fn(a, theta):
                m = self.sde.mu(a)
                std = self.sde.std(a)
                p_denoise = denoise(self.prior, m, std, theta)

                return 1 / p_denoise.var

        self.marginal_denoising_prior_precission_fn = (
            marginal_denoising_prior_precission_fn
        )

        def marginal_posterior_precission_fn(a, theta, x_o, N):
            precissions_posteriors = self.posterior_precission_est_fn(x_o)
            precissions_posteriors = jnp.atleast_2d(precissions_posteriors)

            # If one constant precission is given, tile it
            if precissions_posteriors.shape[0] < N:
                precissions_posteriors = jnp.tile(precissions_posteriors, (N, 1))

            # Denoising posterior via Bayes rule
            m = self.sde.mu(a)
            std = self.sde.std(a)

            if precissions_posteriors.ndim == 3:
                I = jnp.eye(precissions_posteriors.shape[-1])
            else:
                I = jnp.ones_like(precissions_posteriors)

            marginal_precissions = m**2 / std**2 * I + precissions_posteriors
            return marginal_precissions

        self.marginal_posterior_precission_est_fn = marginal_posterior_precission_fn
        self.kwargs = kwargs

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        # Marginal prior precission
        prior_precission = self.marginal_denoising_prior_precission_fn(a, theta)
        # Marginal posterior variance estimates
        posterior_precissions = self.marginal_posterior_precission_est_fn(
            a, theta, x_o, N
        )

        # Total precission
        term1 = (1 - N) * prior_precission
        term2 = jnp.sum(posterior_precissions, axis=0)
        Lam = add_diag_or_dense(term1, term2)

        # Weighted scores
        weighted_prior_score = mv_diag_or_dense(prior_precission, prior_score)
        weighted_posterior_scores = jax.vmap(mv_diag_or_dense)(
            posterior_precissions, base_score
        )

        # Accumulate the scores
        score = (1 - N) * weighted_prior_score + jnp.sum(
            weighted_posterior_scores, axis=0
        )

        # Solve the linear system
        score = solve_diag_or_dense(Lam, score)

        return score

    def estimate_hyperparameters(
        self,
        x_o,
        input_shape,
        key,
        ensure_valid=True,
        num_steps=200,
        num_samples=1000,
        diag=True,
        window_size=2,
        window_stride=1,
        precission_nugget=0.1,
        t_o=None,
    ):
        window_size = self.kwargs.get("window_size", window_size)
        window_stride = window_size - self.kwargs.get("window_stride", window_stride)
        precission_nugget = self.kwargs.get("precission_nugget", precission_nugget)
        diag = self.kwargs.get("diag", diag)
        ensure_valid = self.kwargs.get("ensure_valid", ensure_valid)
        num_steps = self.kwargs.get("num_steps", num_steps)
        num_samples = self.kwargs.get("num_samples", num_samples)

        if x_o.shape[0] <= window_size:
            self.posterior_precission_est_fn = lambda x: 1 / self.prior.var
        else:
            from markovsbi.sampling.kernels.euler_maruyama import EulerMaruyama
            from markovsbi.sampling.sample import Diffuser

            base_score = ScoreFn(self.score_net, self.params, self.sde, self.prior)
            num_samples = num_samples * input_shape[0]

            kernel = EulerMaruyama(base_score)
            length = num_steps
            time_grid = jnp.linspace(self.sde.T_min, self.sde.T_max, length)
            sampler = Diffuser(kernel, time_grid, input_shape)

            individual_x_o = get_windows(x_o, window_size, window_stride, axis=0)
            # print(individual_x_o)
            if t_o is None:
                individual_samples = jax.vmap(
                    jax.vmap(sampler.sample, in_axes=(0, None)), in_axes=(None, 0)
                )(jax.random.split(key, num_samples), individual_x_o)
            else:
                t_o_windows = get_windows(t_o, window_size, window_stride, axis=0)

                individual_samples = jax.vmap(
                    jax.vmap(sampler.sample, in_axes=(0, None, None)),
                    in_axes=(None, 0, 0),
                )(jax.random.split(key, num_samples), individual_x_o, t_o_windows)

                # print(individual_samples.shape)

            # Empirically estimate posterior covariances
            if diag == True:
                individual_precissions = 1 / jnp.var(individual_samples, axis=1)
            else:
                cov_matrix = jax.vmap(jnp.cov)(
                    jnp.transpose(individual_samples, axes=(0, 2, 1))
                ) + 1e-3 * jnp.eye(input_shape[0])
                individual_precissions = jnp.linalg.inv(cov_matrix)

            # Ensure the overall precission will be larger than the prior precission
            # Otherwise, this correction will become invalid!
            N = individual_precissions.shape[0]
            if diag:
                prior_precission = 1 / self.prior.var
            else:
                # TODO This may not be correct for non-diagonal precission matrices
                prior_precission = jnp.diag(1 / self.prior.var)

            if ensure_valid:
                term1 = (1 - N) * prior_precission
                term2 = jnp.sum(individual_precissions, axis=0)
                Lam = add_diag_or_dense(term1, term2)

                if diag:
                    average_diff = (
                        jnp.where(Lam > 0, 0.0, -Lam) / (N - 1) + precission_nugget
                    )
                    individual_precissions_corrected = (
                        individual_precissions + average_diff
                    )
                else:
                    eigenvalues, eigenvectors = jnp.linalg.eigh(Lam)
                    eigenvalues = jnp.where(eigenvalues <= 0, -eigenvalues, 0.0) / (
                        N - 1
                    )
                    Lam_corr = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
                    Lam_corr += precission_nugget * jnp.eye(Lam.shape[0])
                    individual_precissions_corrected = (
                        individual_precissions + Lam_corr[None, ...]
                    )
                self.posterior_precission_est_fn = (
                    lambda x: individual_precissions_corrected
                )
            else:
                self.posterior_precission_est_fn = lambda x: individual_precissions


class GaussMixtureScoreFn(UncorrectedScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params,
        sde,
        prior,
        marginal_prior_score_fn=None,
        **kwargs,
    ):
        super().__init__(score_net, params, sde, prior, marginal_prior_score_fn)
        self.requires_hyperparameters = True
        self.correction_term_fn = None
        self.kwargs = kwargs

    def build_correction_term(
        self,
        posteriors=None,
        reduce_compontents=False,
        max_components_scale=2,
        min_components=10,
    ):
        correction_term = build_gauss_mixture_correction_term(
            self.prior,
            posteriors,
            reduce_compontents,
            max_components_scale,
            min_components=min_components,
        )
        correction_term_grad = jax.grad(correction_term)

        def correction_term_fn(x, a):
            a = jnp.clip(a, self.sde.T_min, self.sde.T_max)
            m = self.sde.mu(a)
            std = self.sde.std(a)

            return correction_term_grad(x, m, std)

        self.correction_term_fn = correction_term_fn
        self.correction_term = correction_term

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)

        # Marginal prior score
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        uncorrected_score = (1 - N) * prior_score + base_score.sum(0)

        correction_term = self.correction_term_fn(theta, a)

        # jax.debug.print(
        #     "Correction norm: {n}, Uncorrected norm: {m}",
        #     n=jnp.linalg.norm(correction_term, axis=-1).max(),
        #     m=jnp.linalg.norm(uncorrected_score, axis=-1).max(),
        # )

        score = uncorrected_score + correction_term
        return score

    def estimate_hyperparameters(
        self,
        x_o,
        input_shape,
        key,
        ensure_valid=True,
        num_steps=200,
        num_samples=500,
        K=2,
        window_size=2,
        window_stride=1,
        precission_nugget=0.1,
        reduce_compontents=False,
        max_components_scale=2,
        min_components=10,
    ):
        window_size = self.kwargs.get("window_size", window_size)
        window_stride = window_size - self.kwargs.get("window_stride", window_stride)
        precission_nugget = self.kwargs.get("precission_nugget", precission_nugget)
        K = self.kwargs.get("K", K)
        ensure_valid = self.kwargs.get("ensure_valid", ensure_valid)
        num_steps = self.kwargs.get("num_steps", num_steps)
        num_samples = self.kwargs.get("num_samples", num_samples)

        if x_o.shape[0] <= window_size:
            self.correction_term_fn = lambda x, a: jnp.zeros_like(x)
        else:
            from markovsbi.sampling.kernels.euler_maruyama import EulerMaruyama
            from markovsbi.sampling.sample import Diffuser
            from markovsbi.utils.product_mixtures import kmeans, compute_mixture_params
            from markovsbi.utils.prior_utils import MixtureNormal

            base_score = ScoreFn(self.score_net, self.params, self.sde, self.prior)
            num_samples = num_samples * input_shape[0]

            kernel = EulerMaruyama(base_score)
            length = num_steps
            time_grid = jnp.linspace(self.sde.T_min, self.sde.T_max, length)
            sampler = Diffuser(kernel, time_grid, input_shape)

            individual_x_o = get_windows(x_o, window_size, window_stride, axis=0)
            # print(individual_x_o)
            individual_samples = jax.vmap(
                jax.vmap(sampler.sample, in_axes=(0, None)), in_axes=(None, 0)
            )(jax.random.split(key, num_samples), individual_x_o)

            clusters, centeroids = jax.vmap(kmeans, in_axes=(0, None))(
                individual_samples, K
            )
            weights, means, stds = jax.vmap(compute_mixture_params)(
                individual_samples, centeroids, clusters
            )

            posteriors = [
                MixtureNormal(m, s, jnp.log(w)) for m, s, w in zip(means, stds, weights)
            ]
            self.build_correction_term(
                posteriors,
                reduce_compontents=reduce_compontents,
                max_components_scale=max_components_scale,
                min_components=min_components,
            )


class CorrectedScoreFn(GaussCorrectedScoreFn):
    def __init__(
        self,
        score_net: Callable,
        params,
        sde,
        prior,
        marginal_prior_score_fn: Optional[Callable] = None,
        marginal_denoising_prior_precission_fn: Optional[Callable] = None,
        marginal_denosing_posterior_precission_fn: Optional[Callable] = None,
        ensure_valid=True,
        precission_nugget=0.1,
    ) -> None:
        """Corrected score function assuming a Gaussian posterior with variance determined by marginal_posterior_precission_est_fn.

        Args:
            score_net (Callable): Score estimation network of the form f(params,a,theta,x_o)
            params (_type_): Parameters of the score network
            sde (_type_): SDE of forward model
            T_min (_type_): Minimum time point
            T_max (_type_): Maximum time point
            marginal_prior_score_fn (_type_): Marginal prior score function
            marginal_prior_precission_fn (_type_): Marginal prior precission function
        """
        super().__init__(
            score_net,
            params,
            sde,
            prior,
            None,
            marginal_prior_score_fn=marginal_prior_score_fn,
            marginal_denoising_prior_precission_fn=marginal_denoising_prior_precission_fn,
        )
        self.ensure_valid = ensure_valid
        self.precission_nugget = precission_nugget

        if marginal_denosing_posterior_precission_fn is not None:
            self.marginal_denoising_posterior_precission_fn = (
                marginal_denosing_posterior_precission_fn
            )

    def marginal_denoising_posterior_precission_fn(self, a, theta, x_o):
        d = theta.shape[-1]
        jac = jax.jacfwd(lambda x: self.score_net(self.params, a, x, x_o))(theta)

        # For a valid covariance jac must be symmetric!
        jac = 0.5 * (jac + jnp.transpose(jac, (0, 2, 1)))

        m = self.sde.mu(a)
        std = self.sde.std(a)

        # print(denoising_posterior_precission)
        if d > 1:
            denoising_posterior_precission = (
                m**2 / std**2 * jnp.linalg.inv(jnp.eye(d) + std**2 * jac)
            )

            denoising_posterior_precission = jax.vmap(project_psd)(
                denoising_posterior_precission
            )
        else:
            denoising_posterior_precission = m**2 / std**2 * 1 / (1 + std**2 * jac)
            denoising_posterior_precission = jnp.clip(
                denoising_posterior_precission, 1e-2, 100000.0
            )
        return denoising_posterior_precission

    def ensure_valid_precission(
        self, denoising_prior_precission, denoising_posterior_precission
    ):
        d = denoising_posterior_precission.shape[-1]
        N = denoising_posterior_precission.shape[0]

        denoising_posterior_precission = jnp.where(
            jnp.isnan(denoising_posterior_precission),
            2 * jnp.repeat(jnp.diag(denoising_prior_precission)[None, ...], N, axis=0),
            denoising_posterior_precission,
        )

        term1 = (1 - N) * denoising_prior_precission
        term2 = jnp.sum(denoising_posterior_precission, axis=0)
        Lam = add_diag_or_dense(term1, term2)

        if N > 1:
            if d > 1:
                eigenvalues, eigenvectors = jnp.linalg.eigh(Lam)
                eigenvalues = jnp.where(eigenvalues <= 0, -eigenvalues, 0.0) / (N - 1)
                Lam_corr = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
                Lam_corr += self.precission_nugget * jnp.eye(Lam.shape[0])
                denoising_posterior_precission = (
                    denoising_posterior_precission + Lam_corr[None, ...]
                )
            else:
                Lam = jnp.diag(Lam)
                average_diff = (
                    jnp.where(Lam > 0, 0.0, -Lam) / (N - 1) + self.precission_nugget
                )
                Lam_corr = jnp.diag(average_diff)

                denoising_posterior_precission = (
                    denoising_posterior_precission + Lam_corr
                )

            return denoising_prior_precission, denoising_posterior_precission
        else:
            return denoising_prior_precission, denoising_posterior_precission

    def __call__(self, a, theta, x_o, **kwargs):
        base_score = self.score_net(self.params, a, theta, x_o, **kwargs)
        base_score = jnp.atleast_1d(base_score)
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        # Denoising prior precission
        prior_precission = self.marginal_denoising_prior_precission_fn(a, theta)

        # Denoising posterior precission
        if N > 1:
            posterior_precissions = self.marginal_denoising_posterior_precission_fn(
                a, theta, x_o
            )
        else:
            posterior_precissions = jnp.diag(prior_precission)[None, ...]

        if self.ensure_valid:
            prior_precission, posterior_precissions = self.ensure_valid_precission(
                prior_precission, posterior_precissions
            )

        # Total precission
        term1 = (1 - N) * prior_precission
        term2 = jnp.sum(posterior_precissions, axis=0)
        Lam = add_diag_or_dense(term1, term2)

        # Weighted scores
        weighted_prior_score = mv_diag_or_dense(prior_precission, prior_score)
        weighted_posterior_scores = jax.vmap(mv_diag_or_dense)(
            posterior_precissions, base_score
        )

        # Accumulate the scores
        score = (1 - N) * weighted_prior_score + jnp.sum(
            weighted_posterior_scores, axis=0
        )

        # Solve the linear system
        score = solve_diag_or_dense(Lam, score)

        return score


def project_psd(A, min_eig=1e-3):
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, min_eig)
    return eigvecs @ jnp.diag(eigvals) @ jnp.transpose(eigvecs)
