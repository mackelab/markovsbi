import jax
from markovsbi.sampling.score_fn import ScoreFn


def build_tweedies_denoiser(score_fn: ScoreFn):
    mean_fn = lambda a: score_fn.sde.mu(a)
    std_fn = lambda a: score_fn.sde.std(a)

    def mean_estimator(a, theta, x_o):
        m = mean_fn(a)
        std = std_fn(a)
        score = score_fn(a, theta, x_o)
        x_0_est = (theta + std**2 * score) / m
        return x_0_est

    def cov_estimator(a, theta, x_o):
        m = mean_fn(a)
        std = std_fn(a)
        score_jac = jax.jacfwd(mean_estimator, argnums=1)(a, theta, x_o)
        var = std**2 / m * score_jac
        return var

    return mean_estimator, cov_estimator
