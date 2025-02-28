from markovsbi.sampling.score_fn import ScoreFn

import jax
import jax.numpy as jnp

from markovsbi.sampling.kernels.base import Kernel, SDEState



def init_state(position, x_o, a, score_fn: ScoreFn, **kwargs):
    return SDEState(position, score_fn(a, position, x_o, **kwargs), a)


def vp_default_bridge(alpha, alpha_new, std, std_new, t1, t0):
    return std_new / std * jnp.sqrt((1 - alpha / alpha_new))


def build_kernel(score_fn: ScoreFn, bridge_sde_std_fn=vp_default_bridge, eta=1.0):
    alpha_fn = lambda a: score_fn.sde.mu(a)
    std_fn = lambda a: score_fn.sde.std(a)

    def kernel(key, state, a_new, x_o, **kwargs):
        a_old = state.a
        position = state.position
        score = state.score

        alpha = alpha_fn(a_old)
        std = std_fn(a_old)
        alpha_new = alpha_fn(a_new)
        std_new = std_fn(a_new)
        std_bridge = eta * bridge_sde_std_fn(
            alpha, alpha_new, std, std_new, a_new, a_old
        )
        # We need that std_bridge >= std otherwise invalid
        std_bridge = jnp.clip(std_bridge, max=std_new)

        epsilon_pred = -std * score
        x0_pred = (position - std * epsilon_pred) / alpha

        # Correction term for difference in std
        bridge_correction = jnp.sqrt(std_new**2 - std_bridge**2) * epsilon_pred
        bridge_noise = std_bridge * jax.random.normal(key, position.shape)

        # New position
        new_position = alpha_new * x0_pred + bridge_correction + bridge_noise

        score = score_fn(a_new, new_position, x_o, **kwargs)

        return SDEState(new_position, score, a_new)

    return kernel


class DDIM(Kernel):
    init_state = init_state
    build_kernel = build_kernel
