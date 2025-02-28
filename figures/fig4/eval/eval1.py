
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('eval.log')
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
from markovsbi.models.train_utils import build_batch_sampler,build_loss_fn, build_online_batch_sampler

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
    
    h1 = hk.Conv2D(64, kernel_shape=6, stride=2, padding='VALID')(x)
    h1 = hk.GroupNorm(4)(h1)
    h1 = jax.nn.gelu(h1)
    #print(h1.shape)
    h2 = hk.Conv2D(32, kernel_shape=6, stride=2, padding='VALID')(h1)
    h2 = hk.GroupNorm(4)(h2)
    h2 = jax.nn.gelu(h2)
    #print(h2.shape)
    h3 = hk.Conv2D(16, kernel_shape=3, stride=1, padding='VALID')(h2)
    h3 = hk.GroupNorm(4)(h3)
    h3 = jax.nn.gelu(h3)
    #print(h3.shape)
    h4 = hk.Conv2D(8, kernel_shape=3, stride=1, padding='VALID')(h3)
    #print(h4.shape)
    h6 = hk.Flatten()(h4)
    out = hk.Linear(100)(h6)
    out = jax.nn.gelu(out)
    out = hk.Linear(100)(out)
    #print(out.shape)
    out = out.reshape(shape[:2] + (100,))
    out = jax.nn.gelu(out)
    #print(out.shape)
    if squeeze:
        out = out[0]
    return out


c_in, c_noise, c_out = precondition_functions(sde)
init_fn, score_net = build_score_mlp(T, num_hidden=5,hidden_dim=400, c_in=c_in, c_noise=c_noise, c_out=c_out, c_context=embedding_net)
batch_sampler = build_batch_sampler(data)
loss_fn = build_loss_fn("dsm", score_net, sde,weight_fn,control_variate=True, control_variate_optimal_scaling=True)

theta_batch, x_batch = batch_sampler(key_init, 50)
d = theta_batch.shape[1]
logger.info(theta_batch.shape, x_batch.shape)

params = jnp.load("params_fnse.npy", allow_pickle=True).item()

from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama, PredictorCorrector, DDIM
from markovsbi.models.utils import get_windows
from markovsbi.sampling.score_fn import FNPEScoreFn, UncorrectedScoreFn, GaussCorrectedScoreFn,CorrectedScoreFn, ScoreFn

d=2
score_fn = GaussCorrectedScoreFn(score_net, params, sde, prior)
kernel = EulerMaruyama(score_fn)
length = 2000
time_grid = jnp.linspace(sde.T_min + 0.0, sde.T_max, length)
sampler = Diffuser(kernel, time_grid, (d,))

from functools import partial
@partial(jax.jit, static_argnums=(1,))
def get_theta_x_o(key, T=11):
    theta_o = prior.sample(key)
    x_o = simulator(key, theta_o, T)
    return theta_o, x_o

@jax.jit
def sample_fn(key, x_o):

    sampler.kernel.score_fn.estimate_hyperparameters(key, theta_o.shape, x_o, diag=False, precission_nugget=0.)
    
    samples = jax.vmap(sampler.sample, in_axes=(0, None))(jax.random.split(key, 1_000), x_o)
    return samples


calibration = []

for i in range(0, 200):
    key = jax.random.PRNGKey(i)
    key1, key2 = jax.random.split(key)
    theta_o, x_o = get_theta_x_o(key1, T=2)
    samples = sample_fn(key2, x_o)
    values = jnp.sum(theta_o < samples, axis=0)
    calibration.append(values)
    
    
calibration = jnp.stack(calibration)

jnp.save("calibration_1.npy", calibration)
