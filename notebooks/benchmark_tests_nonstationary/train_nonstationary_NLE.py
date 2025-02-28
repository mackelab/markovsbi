# %% [markdown]
#

import argparse
import pandas as pd

import sys

sys.path


parser = argparse.ArgumentParser(description="Nonstationary_NLE")
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--key_number", type=int, default=0)
parser.add_argument("--obs_length", type=str, default="[2, 11]")
parser.add_argument("--simulation_budget", type=int, default=10_000)
flags = parser.parse_args()


# %%
import sbi
from sbi.inference import SNLE, SNRE, SNPE
import torch


data_store = []


import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import optax

from markovsbi.tasks import get_task
from markovsbi.bm.api_utils import get_window_torch

sbi.__version__

# %%


key = jax.random.PRNGKey(flags.key_number)

task = get_task("simple{}dnonstationary".format(flags.dim))
prior = task.get_prior()
simulator = task.get_simulator()


data = task.get_data(key, flags.simulation_budget, 2, max_T=100)

# %%
from numpy import indices

d = task.input_shape[0]
print("d:", d)
prior_torch = torch.distributions.Independent(
    torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1
)


class EmbeddingNet(torch.nn.Module):
    def __init__(self, embedding_size=4):
        super(EmbeddingNet, self).__init__()
        self.embedding_size = embedding_size
        self.register_buffer("w", torch.randn(embedding_size // 2))
        self.buffer = torch.nn.Linear(1, 1)

    def forward(self, x):
        t = x[..., -2:]
        t0 = t[..., 0].unsqueeze(-1)
        t1 = t[..., 1].unsqueeze(-1)
        t0_emebdding = torch.cat([torch.sin(self.w * t0), torch.cos(x * t0)], dim=-1)
        t1_emebdding = torch.cat([torch.sin(self.w * t1), torch.cos(x * t1)], dim=-1)
        ts_embedding = torch.cat([t0_emebdding, t1_emebdding], dim=-1)
        full_embedding = torch.cat([x[..., :-2], ts_embedding], dim=-1)
        return full_embedding


from sbi.utils import likelihood_nn

flow = likelihood_nn(model="maf", embedding_net=EmbeddingNet())

inf = SNLE(prior_torch, density_estimator=flow)


thetas = data["thetas"]
xs = data["xs"]
ts = data["ts"]

thetas = torch.tensor(np.array(thetas), dtype=torch.float32)
xs = torch.tensor(np.array(xs), dtype=torch.float32)
xs = xs.reshape(xs.shape[0], -1)

ts = torch.tensor(np.array(ts), dtype=torch.float32)

thetas = torch.concatenate([thetas, ts], dim=-1)
T = int(xs.shape[1])


inf.append_simulations(thetas, xs)


density_estimator = inf.train(
    training_batch_size=1000,
    learning_rate=1e-3,
    validation_fraction=0.1,
    stop_after_epochs=20,
    max_num_epochs=2147483647,
    clip_max_norm=5.0,
)


from sbi.inference import MCMCPosterior, ratio_estimator_based_potential
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, mcmc_transform
from sbi.utils.torchutils import atleast_2d


def custom_potential(theta, x_o):
    # print(theta.shape, x_o.shape)
    theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
        theta=atleast_2d(theta), x=atleast_2d(x_o)
    )
    # print(theta_repeated.shape, x_repeated.shape)
    theta_repeated_expanded = torch.cat([theta_repeated, x_repeated[:, -2:]], dim=1)
    x_repeated = x_repeated[:, :-2]

    x_repeated = x_repeated.unsqueeze(0)

    out = density_estimator.log_prob(x_repeated, theta_repeated_expanded) + prior_torch.log_prob(
        theta_repeated
    )
    out = out.reshape(x_o.shape[0], -1).sum(0)
    return out


posterior = MCMCPosterior(
    custom_potential,
    inf._prior,
    theta_transform=mcmc_transform(inf._prior),
    num_chains=100,
)

from sbi.utils.metrics import c2st

num_observations = eval(flags.obs_length)
for num_obs in num_observations:
    num_obs = int(num_obs)
    metrics = 0.0
    for i in range(10):
        key, subkey = jax.random.split(key)
        theta_o = prior.sample(subkey)

        key, subkey = jax.random.split(key)

        x_o = torch.tensor(np.array(simulator(subkey, theta_o, num_obs)))
        t_o = torch.tensor(np.arange(num_obs, dtype=jnp.float32)).reshape(-1, 1)

        x_o_window = get_window_torch(x_o, 2)
        t_o_window = get_window_torch(t_o, 2)
        x_o_window = torch.concatenate([x_o_window, t_o_window], dim=1)

        n_samples = 1000
        x_o = torch.tensor(np.array(x_o))
        posterior._x_shape = (1, x_o.shape[0], x_o.shape[1])
        samples = posterior.sample((n_samples,), x=x_o_window, show_progress_bars=False)

        true_posterior = task.get_true_posterior(jnp.array(np.array(x_o)))
        true_samples = true_posterior.sample(key, (1000,))
        true_samples = torch.tensor(np.array(true_samples))

        distance = c2st(torch.tensor(np.array(samples))[:1000], true_samples[:1000])
        metrics += float(distance / 10)

    data_store_local = []

    data_store_local.append(flags.dim)
    data_store_local.append(flags.simulation_budget)
    data_store_local.append(flags.key_number)
    data_store_local.append(num_obs)
    data_store_local.append(float(metrics))

    data_store.append(data_store_local)
    print(data_store_local)

    print(data_store)

    import pandas as pd

    # Append data to CSV file
    with open("Nonstationary_NLE.csv", "a") as f:
        pd.DataFrame(
            [data_store_local],
            columns=["dim", "num_simulations", "key_number", "obs_length", "C2ST"],
        ).to_csv(f, header=False, index=False)
