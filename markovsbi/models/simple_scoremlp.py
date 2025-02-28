import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Callable, Optional

import haiku as hk



from markovsbi.models.utils import get_windows


class GaussianFourierEmbedding(hk.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable=False,
        name: str = "gaussian_fourier_embedding",
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "gaussian_fourier_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim
        self.learnable = learnable

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        B = hk.get_parameter(
            "B", [half_dim, inputs.shape[-1]], init=hk.initializers.RandomNormal()
        )
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


def precondition_functions(sde, shring_var=0.5):
    var_true = shring_var * sde.std0**2
    mean_true = sde.mu0

    std_fn = sde.std
    mean_fn = sde.mu

    def c_in(a, x):
        # Time dependent z-score
        return jnp.sqrt(var_true / (std_fn(a) ** 2 + mean_fn(a) ** 2 * var_true)) * (
            x - mean_fn(a) * mean_true
        )

    def c_noise(a):
        # Time to noise
        return std_fn(a)

    def c_out(a, x, y):
        scale = 1 / (var_true * mean_fn(a) ** 2 + std_fn(a) ** 2)
        return -scale * x + scale * mean_fn(a) * y

    return c_in, c_noise, c_out


def precondition_functions_v2(sde, shring_var=0.5):
    var_true = shring_var * sde.std0**2
    mean_true = sde.mu0

    std_fn = sde.std
    mean_fn = sde.mu

    def c_in(a, x):
        # Time dependent z-score
        return jnp.sqrt(var_true / (std_fn(a) ** 2 + mean_fn(a) ** 2 * var_true)) * (
            x - mean_fn(a) * mean_true
        )

    def c_noise(a):
        # Time to noise
        return std_fn(a)

    def c_out(a, x, y):
        approx_mean = mean_fn(a) * mean_true
        approx_var = mean_fn(a) ** 2 * var_true + std_fn(a) ** 2
        gaussian_approx_score = (x - approx_mean) / approx_var
        scale = mean_fn(a) / approx_var
        return -scale * y - gaussian_approx_score

    return c_in, c_noise, c_out


def precondition_functions_v3(sde, shring_var=0.5):
    var_true = shring_var * sde.std0**2
    mean_true = sde.mu0

    std_fn = sde.std
    mean_fn = sde.mu

    def c_in(a, x):
        # Time dependent z-score
        return jnp.sqrt(var_true / (std_fn(a) ** 2 + mean_fn(a) ** 2 * var_true)) * (
            x - mean_fn(a) * mean_true
        )

    def c_noise(a):
        # Time to noise
        return std_fn(a)

    def c_out(a, x, y):
        approx_mean = mean_fn(a) * mean_true
        approx_var = mean_fn(a) ** 2 * var_true + std_fn(a) ** 2
        gaussian_approx_score = (x - approx_mean) / approx_var
        scale = mean_fn(a) / std_fn(a)
        return -scale * y - gaussian_approx_score

    return c_in, c_noise, c_out


def build_score_mlp(
    window_size=int,
    markov_order: int = 1,
    hidden_dim: int = 50,
    activation: Callable = jax.nn.gelu,
    num_hidden: int = 3,
    layer_norm: bool = True,
    skip_connection: bool = True,
    time_embedding_dim: int = 16,
    use_variance_scaling_init: bool = True,
    x_o_processing="linear",
    c_out: Callable = lambda t, x, y: y,
    c_in: Callable = lambda t, x: x,
    c_noise: Callable = lambda t: t,
    c_context: Callable = lambda x: x,
):
    """Simple MLP for score function estimation with

    Args:
        window_size (int): Window size.
        hidden_dim (int, optional): Hidden dimension. Defaults to 50.
        activation (Callable, optional): Activation function. Defaults to jax.nn.gelu.
        num_hidden (int, optional): Hidden layers. Defaults to 6.
        layer_norm (bool, optional): Layer norm. Defaults to True.
        output_scale_fn (Optional[Callable], optional): Output score scaling. Defaults to None.
    """

    def score_net(
        a: ArrayLike,  # Diffusion time [B,]
        theta: ArrayLike,  # Parameters [B, D1]
        x_o: ArrayLike,  # Observations [B, T, D2]
        time: Optional[ArrayLike] = None,  # Time series Time [B,]
        return_last_hidden: bool = False,
    ):
        a = jnp.asarray(a)
        if time is not None:
            time = jnp.asarray(time)
        theta = jnp.asarray(theta)
        x_o = jnp.asarray(x_o)

        # Formating inputs
        if x_o.ndim == 2:
            x_o = x_o[None, ...]
        if theta.ndim == 1:
            theta = theta[None, ...]
        if theta.ndim == 3:
            theta = jnp.squeeze(theta, axis=1)

        b, d1 = theta.shape

        x_o = c_context(x_o)
        x_o = x_o.reshape(b, -1, x_o.shape[-1])

        b, t, d2 = x_o.shape

        # Transform time by c_noise
        a = a.reshape(b, 1, 1)
        noise = c_noise(a)
        # Transform input by c_in
        theta = theta[:, None, :]
        theta_in = c_in(a, theta)

        stride = window_size - markov_order

        # Context window
        context = get_windows(
            x_o, window_size, stride=stride, axis=-2
        )  # [B, T-window_size+1, window_size, D2]
        # print(context)
        # print(context.shape)
        dividable = int((t - window_size + 1) % stride != 0)
        context = context.transpose(0, 2, 1, 3)
        if x_o_processing == "linear" or x_o_processing == "mlp":
            context = context.reshape(
                b, (t - window_size + 1) // stride + dividable, window_size * d2
            )  # [B, T-window_size+1, window_size*D2]

        # print(context)
        # print(context.shape)

        a_embedding = GaussianFourierEmbedding(time_embedding_dim)(noise)  # [B, 1, 128]
        a_embedding = hk.Linear(hidden_dim)(a_embedding)  # [B, 1, hidden_dim]

        if time is not None:
            time = time.reshape(b, t, 1)
            time_window = get_windows(time, window_size, stride=stride, axis=-2)
            dividable = int((t - window_size + 1) % stride != 0)
            time_window = time_window.reshape(
                b, (t - window_size + 1) // stride + dividable, window_size
            )
            time_window = time_window.reshape(
                b, (t - window_size + 1) // stride + dividable, window_size
            )
            # Only for window_size = 1 working
            time_dim = 10
            input_hidden_dim = hidden_dim - time_dim

            h1 = hk.Linear(input_hidden_dim // 2)(context)
            h2 = GaussianFourierEmbedding(time_dim)(time_window)

            h3 = hk.Linear(input_hidden_dim - (input_hidden_dim // 2))(theta_in)

            h3 = jnp.repeat(h3, h1.shape[1], axis=1)
            h = jnp.concatenate([h1, h3, h2], axis=-1)
        else:
            if x_o_processing == "linear":
                h1 = hk.Linear(hidden_dim // 2)(context)
            elif x_o_processing == "mlp":
                mlp = hk.nets.MLP(
                    [hidden_dim + context.shape[-1], hidden_dim // 2],
                    activation=activation,
                )
                h1 = mlp(context)
            elif x_o_processing == "gru":
                net = hk.GRU(hidden_dim // 2)
                state = net.initial_state(None)
                h1, _ = jax.vmap(jax.vmap(lambda x: net(x, state)))(context)
                h1 = h1[:, :, -1, :]
            else:
                raise ValueError("x_o_processing should be linear or gru")

            h2 = hk.Linear(hidden_dim - hidden_dim // 2)(theta_in)
            h2 = jnp.repeat(h2, h1.shape[1], axis=1)
            h = jnp.concatenate([h1, h2], axis=-1)

        h += a_embedding
        h = activation(h)

        if use_variance_scaling_init:
            init = hk.initializers.VarianceScaling(2 / (num_hidden - 1))
        else:
            init = None

        for _ in range(num_hidden - 1):
            h_new = hk.Linear(hidden_dim, w_init=init)(h)
            h_new += a_embedding
            h_new = activation(h_new)

            if skip_connection:
                h_new += h

            if layer_norm:
                h = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                )(h_new)

        out = hk.Linear(theta.shape[-1])(h)  # [B, T-window_size+1, D1]

        # Output transformation
        out = c_out(a, theta, out)

        # Some output conventions
        if b == 1:
            out = out.squeeze(0)
        else:
            out = out.reshape(b, t - window_size + 1, d1)

        if return_last_hidden:
            return out, h
        else:
            return out

    init_fn, model_fn = hk.without_apply_rng(hk.transform(score_net))
    return init_fn, model_fn
