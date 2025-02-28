# %%


import argparse
import pandas as pd

import sys

sys.path


parser = argparse.ArgumentParser(description="Nonstationary_NPE")
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--key_number", type=int, default=0)
parser.add_argument("--obs_length", type=str, default="[2, 11]")
parser.add_argument("--simulation_budget", type=int, default=10_000)
flags = parser.parse_args()


import sbi
from sbi.inference import SNLE, SNRE, SNPE
import torch


import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
import optax

from markovsbi.tasks import get_task
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn


import torch
torch.manual_seed(flags.key_number)


data_store = []


data_store_local = []
dim = flags.dim


key = jax.random.PRNGKey(flags.key_number)

task = get_task("simple{}dnonstationary".format(flags.dim))
prior = task.get_prior()
simulator = task.get_simulator()

print("flags.simulation_budget:", flags.simulation_budget)
print("flags.simulation_budget-1:", flags.simulation_budget - 1)

data = task.get_data(key, (flags.simulation_budget) // 10, 11, max_T=100)


prior_torch = torch.distributions.Independent(
    torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), 1
)


class EmbeddingNet(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=2):
        super(EmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = torch.nn.GRU(
            self.input_dim + 8, self.output_dim, num_layers=num_layers, batch_first=True
        )
        self.register_buffer("w", torch.randn((4,)))

    def forward(self, x):
        x, t = x[..., :-1], x[..., -1:]
        t1 = torch.sin(t * self.w)
        t2 = torch.cos(t * self.w)
        x = torch.cat([x, t1, t2], dim=-1)

        *batch_shape, n_samples, n_dim = x.shape
        x = x.view(-1, n_samples, n_dim)
        # Replace nans with zeros
        mask = torch.isnan(x).any(-1, keepdim=True).to(torch.int32)
        indices = torch.argmax(mask, dim=1).squeeze(1) - 1
        x = torch.nan_to_num(x, nan=0.0)
        hs = self.gru(x)[0]
        # print(hs.shape)
        x = hs[torch.arange(hs.shape[0]), indices, :]
        # print(x.shape)
        x = x.view(*batch_shape, self.output_dim)
        return x


inf = SNPE(
    prior_torch,
    density_estimator=sbi.utils.posterior_nn(
        model="nsf", embedding_net=EmbeddingNet(flags.dim, 50), z_score_x="none"
    ),
)
thetas = torch.tensor(np.array(data["thetas"]), dtype=torch.float32)
xs = torch.tensor(np.array(data["xs"]), dtype=torch.float32)
ts = torch.tensor(np.array(data["ts"]), dtype=torch.float32)

xs = torch.cat([xs, ts.unsqueeze(-1)], dim=-1)

xs2 = xs.clone()

num_sims = int(flags.simulation_budget)
T_max = 11

for t in range(2, T_max):
    idx = torch.randint(
        0,
        xs.shape[0],
        (
            int(
                0.1
                * num_sims
                / (T_max - 1)
            ),
        ),
    )
    xs_subset = xs[idx]
    theta_subset = thetas[idx]
    xs_subset[:, t:] = torch.nan  # Cut off data after t
    xs = torch.cat([xs, xs_subset], dim=0)
    thetas = torch.cat([thetas, theta_subset], dim=0)



inf.append_simulations(thetas, xs, exclude_invalid_x=False)
density_estimator = inf.train(training_batch_size=1000)

posterior = inf.build_posterior()


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
        t_o = torch.tensor(np.arange(num_obs, dtype=jnp.float32)).unsqueeze(-1)
        x_o_extended = torch.cat([x_o, t_o], dim=1)

        n_samples = 1000
        posterior._x_shape = (1, x_o_extended.shape[0], x_o_extended.shape[1])
        posterior.posterior_estimator._condition_shape = x_o_extended.shape
        samples = posterior.sample(
            (n_samples,), x=x_o_extended, show_progress_bars=False
        )

        true_posterior = task.get_true_posterior(jnp.array(np.array(x_o)))
        key, subkey = jax.random.split(key)
        true_samples = true_posterior.sample(subkey, (1000,))
        true_samples = torch.tensor(np.array(true_samples))

        print(samples.shape, true_samples.shape)

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
    with open("Nonstationary_NPE.csv", "a") as f:
        pd.DataFrame(
            [data_store_local],
            columns=["dim", "num_simulations", "key_number", "obs_length", "C2ST"],
        ).to_csv(f, header=False, index=False)
