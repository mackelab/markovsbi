import argparse
import pandas as pd

import sys

sys.path


parser = argparse.ArgumentParser(description="Proposal")
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--key_number", type=int, default=0)
parser.add_argument("--obs_length", type=str, default=[2, 11])
parser.add_argument("--simulation_budget", type=int, default=10_000)
flags = parser.parse_args()

# %%
data_store = []

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import optax

from markovsbi.tasks import get_task
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn

# %%
jax.devices()

# %%

data_store_local = []


key = jax.random.PRNGKey(flags.key_number)

task = get_task("simple{}dnonstationary".format(flags.dim))
print("simple{}dnonstationary".format(flags.dim))
prior = task.get_prior()
simulator = task.get_simulator()


data = task.get_data(key, flags.simulation_budget, 2, max_T=100)


# %%
sde, weight_fn = init_sde(data, name="VPSDE")

# %%
key, key_init = jax.random.split(key)

# %%
c_in, c_noise, c_out = precondition_functions(sde)
init_fn, score_net = build_score_mlp(2,  num_hidden=5,c_in=c_in, c_noise=c_noise, c_out=c_out)
batch_sampler = build_batch_sampler(data, nonstationary=True)
loss_fn = build_loss_fn(
    "dsm",
    score_net,
    sde,
    weight_fn,
    control_variate=True,
    control_variate_optimal_scaling=True,
)
loss_fn = jax.jit(loss_fn)

# %%
theta_batch, x_batch, t_batch = batch_sampler(key_init, 10)
d = theta_batch.shape[1]


# %%
params = init_fn(key_init, jnp.ones((10,)), theta_batch, x_batch, t_batch)


num_inner_epochs = 50 * (flags.simulation_budget // 1000)

schedule = optax.cosine_onecycle_schedule(
    100 * num_inner_epochs,
    5e-4,
)
optimizer = optax.chain(optax.adaptive_grad_clip(20), optax.adamw(schedule))
opt_state = optimizer.init(params)


# %%
@jax.jit
def update(params, rng, opt_state, theta_batch, x_batch, t_batch):
    loss, grads = jax.value_and_grad(loss_fn)(
        params, rng, theta_batch, x_batch, time=t_batch
    )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

data_val = jax.tree_util.tree_map(lambda x: x[-1000:], data)
data = jax.tree_util.tree_map(lambda x: x[:-1000], data)
batch_sampler = build_batch_sampler(data, nonstationary=True)
batch_sampler_val = build_batch_sampler(data_val, nonstationary=True)


best_loss = jnp.inf
best_params = None
for i in range(100):
    l = 0.0
    for _ in range(num_inner_epochs):
        key, key_batch, key_loss = jax.random.split(key,3)
        theta_batch, x_batch, t_batch = batch_sampler(key_batch, 1000)
        loss, params, opt_state = update(
            params, key_loss, opt_state, theta_batch, x_batch, t_batch
        )
        l += loss / num_inner_epochs

    l_val = 0.0
    for _ in range(500):
        key, key_batch, key_loss = jax.random.split(key,3)
        theta_batch, x_batch, t_batch = batch_sampler_val(key_batch, 1000)
        loss = loss_fn(params, key_loss, theta_batch, x_batch, time=t_batch)
        l_val += loss / 500
    
    print("Training loss :", l)
    print("Validation loss:",l_val)

    if l_val < best_loss:
        best_loss = l_val
        best_params = params.copy()
    
params = best_params


# %%
from markovsbi.sampling.score_fn import (
    FNPEScoreFn,
    UncorrectedScoreFn,
    GaussCorrectedScoreFn,
    CorrectedScoreFn,
    ScoreFn,
)

# %%
score_fn = FNPEScoreFn(score_net, params, sde, prior)
score_fn = UncorrectedScoreFn(score_net, params, sde, prior)
score_fn = GaussCorrectedScoreFn(score_net, params, sde, prior)


from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama, PredictorCorrector, DDIM
from sbi.analysis import pairplot
from sympy import limit
import numpy as np
import torch
from sbi.utils.metrics import c2st

kernel = EulerMaruyama(score_fn)
time_grid = jnp.linspace(sde.T_min, sde.T_max, 500)
sampler = Diffuser(kernel, time_grid, (d,))


# %%
from functools import partial

num_observations = eval(flags.obs_length)
for num_obs in num_observations:
    num_obs = int(num_obs)
    metrics = 0.0
    for i in range(10):
        key, subkey = jax.random.split(key)
        theta_o = prior.sample(subkey)

        t_o = jnp.arange(0, num_obs).astype(jnp.float32)
        key, subkey = jax.random.split(key)
        x_o = simulator(subkey, theta_o, num_obs, ts=t_o)

        key, subkey = jax.random.split(key)
        sampler.kernel.score_fn.estimate_hyperparameters(
            x_o, task.input_shape, subkey, diag=d == 1, t_o=t_o, precission_nugget=0
        )

        key, subkey = jax.random.split(key)
        samples = jax.vmap(partial(sampler.sample, x_o=x_o, t_o=t_o))(
            jax.random.split(subkey, 1000)
        )
        key, subkey = jax.random.split(key)

        true_posterior = task.get_true_posterior(x_o)
        true_samples = true_posterior.sample(subkey, (1000,))

        distance = c2st(
            torch.tensor(np.array(samples))[:1000],
            torch.tensor(np.array(true_samples))[:1000],
        )
        metrics += float(distance / 10)
    print("C2ST: ", metrics)

    # %%

    data_store_local = []

    data_store_local.append(flags.dim)
    data_store_local.append(flags.simulation_budget)
    data_store_local.append(flags.key_number)
    data_store_local.append(num_obs)
    data_store_local.append(float(metrics))

    data_store.append(data_store_local)
    # print(data_store_local)

    # print(data_store)

    import pandas as pd
    
    # Append data to CSV file
    with open("Nonstationary_Proposal.csv", "a") as f:
        pd.DataFrame([data_store_local], columns=["dim", "num_simulations", "key_number", "obs_length", "C2ST"]).to_csv(f, header=False, index=False)