import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("train_fnse_noisy.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


logger.info("Run script")
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import optax

from markovsbi.tasks import KolmogorovFlow
from markovsbi.utils.utils_plot import draw, vorticity
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import (
    build_batch_sampler,
    build_loss_fn,
    build_online_batch_sampler,
)

logger.info(jax.devices())

key = jax.random.PRNGKey(1)

task = KolmogorovFlow(64, sigma=5e-3)
prior = task.get_prior()
simulator = task.get_simulator()
T = 2

data = task.get_data(key, 500, T, init_T=0)
sde, weight_fn = init_sde(data)

key, key_init = jax.random.split(key)

import haiku as hk


def embedding_net(x):
    x = x.astype(jnp.float32)

    if x.ndim < 5:
        x = x[None]
        squeeze = True
    else:
        squeeze = False
    shape = x.shape
    x = x.transpose((0, 1, 3, 4, 2))
    x = x.reshape((-1, 64, 64, 2))

    h1 = hk.Conv2D(64, kernel_shape=6, stride=2, padding="VALID")(x)
    h1 = hk.GroupNorm(4)(h1)
    h1 = jax.nn.gelu(h1)
    # print(h1.shape)
    h2 = hk.Conv2D(32, kernel_shape=6, stride=2, padding="VALID")(h1)
    h2 = hk.GroupNorm(4)(h2)
    h2 = jax.nn.gelu(h2)
    # print(h2.shape)
    h3 = hk.Conv2D(16, kernel_shape=3, stride=1, padding="VALID")(h2)
    h3 = hk.GroupNorm(4)(h3)
    h3 = jax.nn.gelu(h3)
    # print(h3.shape)
    h4 = hk.Conv2D(8, kernel_shape=3, stride=1, padding="VALID")(h3)
    # print(h4.shape)
    h6 = hk.Flatten()(h4)
    out = hk.Linear(100)(h6)
    out = jax.nn.gelu(out)
    out = hk.Linear(100)(out)
    # print(out.shape)
    out = out.reshape(shape[:2] + (100,))
    out = jax.nn.gelu(out)
    # print(out.shape)
    if squeeze:
        out = out[0]
    return out


c_in, c_noise, c_out = precondition_functions(sde)
init_fn, score_net = build_score_mlp(
    T,
    num_hidden=5,
    hidden_dim=400,
    c_in=c_in,
    c_noise=c_noise,
    c_out=c_out,
    c_context=embedding_net,
)
batch_sampler = build_batch_sampler(data)
loss_fn = build_loss_fn(
    "dsm",
    score_net,
    sde,
    weight_fn,
    control_variate=True,
    control_variate_optimal_scaling=True,
)

theta_batch, x_batch = batch_sampler(key_init, 50)
d = theta_batch.shape[1]
logger.info(theta_batch.shape, x_batch.shape)

params = init_fn(key_init, jnp.ones((50,)), theta_batch, x_batch)

schedule = optax.cosine_onecycle_schedule(
    50 * 5000,
    5e-4,
)
optimizer = optax.chain(optax.adaptive_grad_clip(20.0), optax.adamw(schedule))
opt_state = optimizer.init(params)


loss_fn = jax.jit(loss_fn)


@jax.jit
def update(params, rng, opt_state, theta_batch, x_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def move_to_cpu(x):
    return jax.device_put(x, jax.devices("cpu")[0])


# Simulate data 200k
data = task.get_data(jax.random.key(0), 1_000, T, init_T=0)
data = jax.tree_util.tree_map(move_to_cpu, data)
data_old = data
for i in range(1, 100):
    data_new = task.get_data(jax.random.key(i), 1_000, T, x0=data_old["xs"][:, -1])
    data_new = jax.tree_util.tree_map(move_to_cpu, data_new)
    data["thetas"] = jnp.concatenate([data["thetas"], data_new["thetas"]])
    data["xs"] = jnp.concatenate([data["xs"], data_new["xs"]])
    data_old = data_new
data_old = task.get_data(jax.random.key(420), 10_000, T, init_T=0)
for i in range(100, 110):
    data_new = task.get_data(jax.random.key(i), 10_000, T, x0=data_old["xs"][:, -1])
    data_new = jax.tree_util.tree_map(move_to_cpu, data_new)
    data["thetas"] = jnp.concatenate([data["thetas"], data_new["thetas"]])
    data["xs"] = jnp.concatenate([data["xs"], data_new["xs"]])
    data_old = data_new


# Shuffle each array in the dictionary
import numpy as np

data = {k: jax.random.permutation(jax.random.key(0), v) for k, v in data.items()}
data_val = {k: v[-5_000:] for k, v in data.items()}
data_train = {k: np.array(v[:-5_000]) for k, v in data.items()}


batch_sampler_val = build_batch_sampler(data_val)


thetas_train = data_train["thetas"]
xs_train = data_train["xs"]

print("Training data: ", thetas_train.shape)


def batch_sampler(thetas_train, xs_train):
    idx = np.random.randint(0, xs_train.shape[0], size=500)
    theta_batch = thetas_train[idx]
    xs_train = xs_train[idx]
    return theta_batch, xs_train


np.random.seed(0)


best_loss = jnp.inf
best_params = None

for i in range(50):
    l = 0.0
    for _ in range(5000):
        key, key_batch, key_loss = jax.random.split(key, 3)
        theta_batch, x_batch = batch_sampler(thetas_train, xs_train)
        theta_batch = jax.device_put(theta_batch)
        x_batch = jax.device_put(x_batch)
        loss, params, opt_state = update(
            params, key_loss, opt_state, theta_batch, x_batch
        )
        l += loss / 5000
    l_val = 0.0
    for _ in range(200):
        key, key_batch, key_loss = jax.random.split(key, 3)
        theta_batch, x_batch = batch_sampler_val(key_batch, 500)
        theta_batch = jax.device_put(theta_batch)
        x_batch = jax.device_put(x_batch)
        loss = loss_fn(params, key_loss, theta_batch, x_batch)
        l_val += loss / 200

    logger.info(f"Training loss: {l}, Validation loss: {l_val}")

    if l_val < best_loss:
        best_loss = l_val
        best_params = params.copy()


jnp.save("params_fnse_best.pkl", best_params)
jnp.save("params_fnse.pkl", params)
