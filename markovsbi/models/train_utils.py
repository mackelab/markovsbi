from functools import partial
import jax
import jax.numpy as jnp

from .loss_fns import (
    denoising_score_matching_loss,
    score_matching_loss,
    sliced_score_matching,
)

LOSSES = ["dsm", "sm", "ssm"]


def build_batch_sampler(data, nonstationary=False, jit=True):
    thetas = data["thetas"]
    xs = data["xs"]

    if nonstationary == True:
        ts = data["ts"]

    def batch_sampler(rng, batch_size):
        idx = jax.random.choice(rng, jnp.arange(thetas.shape[0]), shape=(batch_size,))
        theta_batch = thetas[idx]
        x_batch = xs[idx]
        if nonstationary == True:
            t_batch = ts[idx]
            return (theta_batch, x_batch, t_batch)
        else:
            return (theta_batch, x_batch)

    if jit:
        return jax.jit(batch_sampler, static_argnums=(1,))
    else:
        return batch_sampler


def build_online_batch_sampler(task, T=2):
    @partial(jax.jit, static_argnums=(1,))
    def online_batch_sampler(rng, batch_size):
        data = task.get_data(rng, batch_size, T)
        theta_batch = data["thetas"]
        x_batch = data["xs"]
        theta_batch = jnp.nan_to_num(theta_batch)
        x_batch = jnp.nan_to_num(x_batch)
        return (theta_batch, x_batch)

    return online_batch_sampler


# Loss functions


def sample_times_uniform(rng, batch_size, T_min, T_max):
    return jax.random.uniform(rng, (batch_size,), minval=T_min, maxval=T_max)


def sample_beta(rng, batch_size, T_min, T_max, alpha=2.0, beta=4.0):
    return jax.random.beta(rng, alpha, beta, (batch_size,)) * (T_max - T_min) + T_min


def sample_times_repeated_uniform(rng, batch_size, T_min, T_max, num_repeats=10):
    batch_size_small = batch_size // num_repeats
    if (batch_size_small * num_repeats) != batch_size:
        batch_size_small += 1
    times = jax.random.uniform(rng, (batch_size_small,), minval=T_min, maxval=T_max)
    times = jnp.repeat(times, num_repeats)
    times = times[:batch_size]
    return times


def build_denoising_score_matching_loss_fn(
    score_net, sde, weight_fn, time_fn=sample_times_uniform, **globa_kwargs
):
    def loss_fn(params, key, theta_batch, x_batch, **kwargs):
        rng_time, rng_loss = jax.random.split(key, 2)
        batch_size = theta_batch.shape[0]
        times = time_fn(rng_time, batch_size, sde.T_min, sde.T_max)

        # Forward diffusion, do not perturb conditioned data
        loss = denoising_score_matching_loss(
            params,
            times,
            theta_batch,
            score_net,
            x_batch,
            mean_fn=sde.mean,
            std_fn=sde.std,
            weight_fn=weight_fn,
            rng_key=rng_loss,
            axis=-1,
            **kwargs,
            **globa_kwargs,
        )

        return loss

    return loss_fn


def build_score_matching_loss_fn(
    score_net, sde, weight_fn, time_fn=sample_times_uniform, **global_kwargs
):
    def loss_fn(params, key, theta_batch, x_batch, **kwargs):
        rng_time, rng_loss = jax.random.split(key, 2)
        batch_size = theta_batch.shape[0]
        # Generate data and random times
        times = time_fn(rng_time, batch_size, sde.T_min, sde.T_max)

        # Forward diffusion, do not perturb conditioned data
        loss = score_matching_loss(
            params,
            times,
            theta_batch,
            score_net,
            x_batch,
            mean_fn=sde.mean,
            std_fn=sde.std,
            weight_fn=weight_fn,
            rng_key=rng_loss,
            axis=-1,
            **kwargs,
            **global_kwargs,
        )

        return loss

    return loss_fn


def build_sliced_score_matching_loss_fn(
    score_net, sde, weight_fn, time_fn=sample_times_uniform, **global_kwargs
):
    def loss_fn(params, key, theta_batch, x_batch, **kwargs):
        rng_time, rng_loss = jax.random.split(key, 2)
        batch_size = theta_batch.shape[0]

        # Generate data and random times
        times = time_fn(rng_time, batch_size, sde.T_min, sde.T_max)

        # Forward diffusion, do not perturb conditioned data
        loss = sliced_score_matching(
            params,
            times,
            theta_batch,
            score_net,
            x_batch,
            mean_fn=sde.mean,
            std_fn=sde.std,
            weight_fn=weight_fn,
            rng_key=rng_loss,
            axis=-1,
            **kwargs,
            **global_kwargs,
        )

        return loss

    return loss_fn


def build_loss_fn(name: str, score_net, sde, weight_fn, **kwargs):
    if name.lower() == "dsm":
        return build_denoising_score_matching_loss_fn(
            score_net, sde, weight_fn, **kwargs
        )
    elif name.lower() == "sm":
        return build_score_matching_loss_fn(score_net, sde, weight_fn, **kwargs)
    elif name.lower() == "ssm":
        return build_sliced_score_matching_loss_fn(score_net, sde, weight_fn, **kwargs)
    else:
        raise ValueError(f"Loss function {name} not implemented.")
