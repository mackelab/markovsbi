from typing import List
from markovsbi.tasks.base import Task
from markovsbi.bm.api_utils import ModelAPI

import jax
import jax.numpy as jnp
import torch
import numpy as np
from sbi.utils.metrics import c2st

import time


def sliced_wasserstein_distance(rng, samples_true, samples_model, num_projections=1000):
    """Compute the sliced Wasserstein distance between two sets of samples."""

    # Compute the sliced Wasserstein distance
    # Compute the projections

    projections = jax.random.normal(rng, (num_projections, samples_true.shape[-1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1, keepdims=True)
    projections = jnp.transpose(projections)
    samples_true_proj = jnp.dot(samples_true, projections)
    samples_model_proj = jnp.dot(samples_model, projections)

    # Sort the projections
    samples_true_proj = jnp.sort(samples_true_proj, axis=0)
    samples_model_proj = jnp.sort(samples_model_proj, axis=0)

    # Compute the sliced Wasserstein distance
    swd = jnp.mean(jnp.abs(samples_true_proj - samples_model_proj))
    return swd


def eval(cfg, eval_methods: List[str], model: ModelAPI, task: Task, rng):
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_eval_xs = cfg.eval.num_eval_xs
    num_eval_steps = cfg.eval.num_steps_xs
    num_eval_samples = cfg.eval.num_eval_samples

    metric_values = []
    sampling_times = []
    for eval_method in eval_methods:
        metrics = []
        for num_steps in num_eval_steps:
            rng1, rng2, rng = jax.random.split(rng, 3)
            theta_eval = prior.sample(rng1, (num_eval_xs,))
            rngs = jax.random.split(rng2, num_eval_xs)
            xs_eval = jax.vmap(simulator, in_axes=(0, 0, None))(
                rngs, theta_eval, num_steps
            )

            metric = 0
            sampling_time = 0
            for i in range(num_eval_xs):
                rng, rng1, rng2 = jax.random.split(rng, 3)
                x_o = xs_eval[i]
                true_posterior = task.get_true_posterior(x_o)
                samples_true = true_posterior.sample(rng1, (num_eval_samples,))
                start_time = time.time()
                samples_model = model.sample(num_eval_samples, x_o, rng=rng2)
                end_time = time.time()
                sampling_time += (end_time - start_time) / num_eval_xs

                # print("Nan samples:", jnp.isnan(samples_model).sum())
                if eval_method == "c2st":
                    samples_true = torch.tensor(np.array(samples_true))
                    samples_model = torch.tensor(np.array(samples_model))
                    metric += c2st(samples_true, samples_model) / num_eval_xs
                elif eval_method == "swd":
                    samples_true = jnp.array(samples_true)
                    samples_model = jnp.array(samples_model)
                    metric += (
                        sliced_wasserstein_distance(rng, samples_true, samples_model)
                        / num_eval_xs
                    )

            metrics.append(float(metric))
            sampling_times.append(sampling_time)
        metric_values.append(metrics)
    return metric_values, sampling_times
