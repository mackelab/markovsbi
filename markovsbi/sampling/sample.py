import jax
import jax.numpy as jnp


from jax.typing import ArrayLike
from functools import partial
from typing import Tuple, Optional, Callable


from markovsbi.sampling.kernels import Kernel


def clip_transform(min_val, max_val):
    def transform(state):
        return state._replace(position=jnp.clip(state.position, min_val, max_val))

    return transform


class Diffuser:
    def __init__(
        self,
        kernel: Kernel,
        grid: ArrayLike,
        theta_shape: Tuple,
        mean_end: ArrayLike = 0.0,
        std_end: ArrayLike = 1.0,
        transform_initital: Optional[Callable] = None,
        transform_state: Optional[Callable] = None,
    ):
        """This is a class that samples from a reverse diffusion process.

        Args:
            kernel (Kernel): A transition kernel
            grid (Array): A grid of time points
            theta_shape (Tuple): The shape of the theta
            mean_end (ArrayLike, optional): End mean. Defaults to 0..
            std_end (ArrayLike, optional): End std. Defaults to 1..
            transform_state (Callable, optional): A function that transfomr the state. Defaults to None.
        """
        self.kernel = kernel
        self.theta_shape = theta_shape
        self.grid = grid
        self.transform_state = transform_state
        self.transform_initital = transform_initital
        self.mean_end = mean_end
        self.std_end = std_end

    @partial(jax.jit, static_argnums=(0,))
    def sample(
        self, key: jax.random.PRNGKey, x_o: ArrayLike, t_o: Optional[ArrayLike] = None
    ):
        key_init, key_sample = jax.random.split(key)

        if self.transform_initital is not None:
            m, s = self.transform_initital(self.mean_end, self.std_end, x_o, t_o)
        else:
            m, s = self.mean_end, self.std_end

        initial_position = m + s * jax.random.normal(key_init, self.theta_shape)
        state = self.kernel.init(initial_position, x_o, self.grid[-1], time=t_o)

        def body_fn(carry, a_new):
            key, state = carry
            key, key_noise = jax.random.split(key)
            state = self.kernel(key_noise, state, a_new, x_o, time=t_o)
            if self.transform_state is not None:
                state = self.transform_state(state)
            return (key, state), None

        (_, state), _ = jax.lax.scan(body_fn, (key_sample, state), self.grid[::-1][1:])

        return state.position

    @partial(jax.jit, static_argnums=(0,))
    def simulate(
        self, key: jax.random.PRNGKey, x_o: ArrayLike, t_o: Optional[ArrayLike] = None
    ):
        key_init, key_sample = jax.random.split(key)

        initial_position = self.mean_end + self.std_end * jax.random.normal(
            key_init, self.theta_shape
        )
        state = self.kernel.init(initial_position, x_o, self.grid[-1], time=t_o)

        def body_fn(carry, a_new):
            key, state = carry
            key, key_noise = jax.random.split(key)
            state = self.kernel(key_noise, state, a_new, x_o, time=t_o)
            if self.transform_state is not None:
                state = self.transform_state(state)
            return (key, state), state.position

        _, positions = jax.lax.scan(body_fn, (key_sample, state), self.grid[::-1][1:])

        positions = jnp.concatenate([initial_position[None], positions], axis=0)

        return positions
