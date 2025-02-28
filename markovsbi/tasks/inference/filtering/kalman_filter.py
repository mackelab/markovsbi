from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from .base import FilterAPI


class KalmanFilterState(NamedTuple):
    mean: ArrayLike
    cov: ArrayLike
    t: Optional[ArrayLike]


class KalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    cov_pred: ArrayLike
    log_likelihood: Optional[ArrayLike] = None


def init(
    mean: ArrayLike, cov: ArrayLike, t: Optional[ArrayLike] = None
) -> KalmanFilterState:
    return KalmanFilterState(mean, cov, t)


def unpack_matrix(A: ArrayLike, t_old: ArrayLike, t: Optional[ArrayLike] = None):
    if isinstance(A, Callable):
        if t is not None:
            return A(t_old, t)
        else:
            return A(t_old)
    else:
        return A


# This is the discrete time Kalman filter for a linear Gaussian model of the form:
# x_t = A_t x_{t-1} + C**1/2 @ w_t
def build_discrete_kernel(
    transition_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    transition_covariance_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_covariance: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
) -> Callable:
    def kernel(
        state: KalmanFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:
        mu0 = state.mean
        cov0 = state.cov
        t_old = state.t
        is_observed = observed is not None

        Phi = unpack_matrix(transition_matrix, t_old, t)
        Q = unpack_matrix(transition_covariance_matrix, t_old, t)

        # Predict
        mu1_ = jnp.dot(Phi, mu0)
        cov1_ = jnp.dot(Phi, jnp.dot(cov0, Phi.T)) + Q

        if is_observed:
            C = unpack_matrix(observation_matrix, t)
            R = unpack_matrix(observation_covariance, t)

            # Kalman gain
            y = observed
            y_ = C @ mu1_
            r = y - y_
            S = C @ cov1_ @ C.T
            S = S + R
            K = jnp.linalg.solve(S.T, (cov1_ @ C.T).T).T

            # Update mean and covariance
            mu1 = mu1_ + K @ r
            cov1 = cov1_ - K @ C @ cov1_
            cov1 = 0.5 * (cov1 + cov1.T)  # Ensure symmetry

            # log_likelihood = -0.5 * (
            #     jnp.linalg.slogdet(S)[1] + r.T @ jnp.linalg.solve(S, r)
            # )
            log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(y, y_, S)

            return KalmanFilterState(mu1, cov1, t), KalmanFilterInfo(
                mu1_, cov1_, log_likelihood
            )
        else:
            return KalmanFilterState(mu1_, cov1_, t), KalmanFilterInfo(
                mu1_, cov1_, jnp.array(0.0)
            )

    return kernel


# API


class kalman_filter(FilterAPI):
    r"""
    Kalman filter for a linear Gaussian state space model.

    $$dx_t = A_t x_t + B_t dw_t \qquad y_t = \mathcal{N}(y_t; C x_t, R_t)$$

    To build a Kalman filter kernel, we require the following components:

    Args:
        transition_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Transition matrix A_t
        transition_covariance_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Transition covariance matrix Q_t
        observation_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Observation matrix C_t
        observation_covariance (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Observation covariance matrix R_t
    """

    init = init
    build_kernel = build_discrete_kernel
