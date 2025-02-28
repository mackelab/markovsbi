from markovsbi.sampling.score_fn import ScoreFn

import jax
import jax.numpy as jnp

from markovsbi.sampling.kernels.base import Kernel, SDEState


def init_state(position, x_o, a, score_fn: ScoreFn, **kwargs):
    return SDEState(position, score_fn(a, position, x_o, **kwargs), a)


def build_kernel(score_fn: ScoreFn, eta=1.0):
    _drift = score_fn.sde.drift
    _diffusion = score_fn.sde.diffusion

    def kernel(key, state, a_new, x_o, **kwargs):
        a_old = state.a
        position = state.position
        score = state.score

        dt = a_new - a_old
        drift_forward = _drift(a_old, position)
        diffusion_forward = eta * _diffusion(a_old, position)
        drift_backward = drift_forward - (1 + eta**2) / 2 * diffusion_forward**2 * score

        position = (
            position
            + drift_backward * dt
            + diffusion_forward
            * jax.random.normal(key, position.shape)
            * jnp.sqrt(jnp.abs(dt))
        )

        score = score_fn(a_new, position, x_o, **kwargs)

        return SDEState(position, score, a_new)

    return kernel


class EulerMaruyama(Kernel):
    init_state = init_state
    build_kernel = build_kernel
