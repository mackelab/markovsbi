from markovsbi.sampling.score_fn import ScoreFn

import jax
import jax.numpy as jnp

from markovsbi.sampling.kernels.base import Kernel, SDEState
from markovsbi.sampling.kernels.euler_maruyama import EulerMaruyama


def build_langevin_corrector(
    score_fn: ScoreFn, step_size: float = 1e-3, steps: int = 10
):
    def langevin_correction(key, state, x_o, **kwargs):
        def one_step(state, key):
            position = state.position
            score = state.score
            key, subkey = jax.random.split(key)

            position = (
                position
                + step_size * score
                + jnp.sqrt(2 * step_size) * jax.random.normal(subkey, position.shape)
            )
            score = score_fn(state.a, position, x_o, **kwargs)

            state = state._replace(position=position, score=score)
            return state, None

        keys = jax.random.split(key, steps)
        state, _ = jax.lax.scan(one_step, state, keys)
        return state

    return langevin_correction


def init_state(position, x_o, a, score_fn: ScoreFn, **kwargs):
    return SDEState(position, score_fn(a, position, x_o, **kwargs), a)


def build_kernel(
    score_fn: ScoreFn,
    predictor: Kernel = EulerMaruyama,
    corrector: str = "langevin",
    **kwargs,
):
    if corrector == "langevin":
        corrector = build_langevin_corrector(score_fn, **kwargs)
    else:
        raise ValueError(f"Corrector '{corrector}' not implemented")

    predictor_kernel = predictor.build_kernel(score_fn)

    def kernel(key, state, a_new, x_o, **kwargs):
        state = predictor_kernel(key, state, a_new, x_o, **kwargs)
        state = corrector(key, state, x_o, **kwargs)
        return state

    return kernel


class PredictorCorrector(Kernel):
    init_state = init_state
    build_kernel = build_kernel
