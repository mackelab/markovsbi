from abc import ABC, abstractmethod
import torch
import jax
import numpy as np
import jax.numpy as jnp


class ModelAPI(ABC):
    @abstractmethod
    def sample(self, n_samples, x_o, rng=None):
        pass

    @abstractmethod
    def log_prob(self, theta, x_o):
        pass


class DiffuserModel(ModelAPI):
    def __init__(
        self, params, score_net, sde, prior, num_steps: int, input_shape, cfg=None
    ):
        self.params = params
        self.score_net = score_net
        self.sde = sde
        self.num_steps = num_steps
        self.prior = prior
        self.cfg = cfg
        self.input_shape = input_shape

        self._score_fn = None
        self._diffuser = None

    @property
    def score_fn(self):
        return self._score_fn

    @score_fn.setter
    def score_fn(self, value):
        self._score_fn = value

    @property
    def diffuser(self):
        return self._diffuser

    @diffuser.setter
    def diffuser(self, value):
        self._diffuser = value

    def sample(self, n_samples, x_o, rng=None):
        assert self._diffuser is not None, "Please provide a diffuser for sampling."
        assert rng is not None, "Please provide a random key for sampling."
        if self._diffuser.kernel.score_fn.requires_hyperparameters:
            rng, subkey = jax.random.split(rng)
            self._diffuser.kernel.score_fn.estimate_hyperparameters(
                x_o, self._diffuser.theta_shape, subkey
            )
        rngs = jax.random.split(rng, n_samples)
        samples = jax.vmap(self.diffuser.sample, in_axes=(0, None))(rngs, x_o)
        return samples

    def log_prob(self, theta, x_o):
        raise NotImplementedError("DiffuserModel does not support log_prob.")

    def __getstate__(self):
        out = self.__dict__.copy()
        out["score_net"] = None
        out["_diffuser"] = None
        out["_score_fn"] = None

        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        _, score_net = build_model(self.cfg, self.sde)
        self.score_net = score_net
        set_factorization_method(self.cfg, self)
        set_diffusion_method(self.cfg, self)

        return self


class SBIModel(ModelAPI):
    def __init__(self, posterior, T, cfg=None):
        self.posterior = posterior
        self.T = T
        self.cfg = cfg

    def sample(self, n_samples, x_o, rng=None):
        x_o = torch.tensor(np.array(x_o))
        x_o_window = get_window_torch(x_o, self.T)
        return self.posterior.sample(
            (n_samples,), x=x_o_window, show_progress_bars=False
        )

    def log_prob(self, theta, x_o):
        x_o = torch.tensor(np.array(x_o))
        x_o_window = get_window_torch(x_o, self.T)
        return self.posterior.log_prob(theta, x=x_o_window)

    def __getstate__(self):
        self.posterior._posterior_sampler = None
        out = self.__dict__.copy()

        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        return self


class NPEModel(ModelAPI):
    def __init__(self, posterior, cfg=None):
        self.posterior = posterior
        self.cfg = cfg

    def sample(self, n_samples, x_o, rng=None):
        x_o = torch.tensor(np.array(x_o))
        self.posterior._x_shape = (1, x_o.shape[0], x_o.shape[1])
        self.posterior.posterior_estimator._condition_shape = x_o.shape
        return self.posterior.sample((n_samples,), x=x_o, show_progress_bars=False)

    def log_prob(self, theta, x_o):
        x_o = torch.tensor(np.array(x_o))
        self.posterior._x_shape = (1, x_o.shape[0], x_o.shape[1])
        return self.posterior.log_prob(theta, x=x_o)


class NSEModel(ModelAPI):
    def __init__(self, posterior, cfg=None):
        self.posterior = posterior
        self.cfg = cfg

    def sample(self, n_samples, x_o, rng=None):
        x_o = torch.tensor(np.array(x_o))
        self.posterior._x_shape = (1, x_o.shape[0], x_o.shape[1])
        self.posterior.score_estimator._condition_shape = x_o.shape
        return self.posterior.sample((n_samples,), x=x_o, show_progress_bars=False)

    def log_prob(self, theta, x_o):
        x_o = torch.tensor(np.array(x_o))
        self.posterior._x_shape = (1, x_o.shape[0], x_o.shape[1])
        return self.posterior.log_prob(theta, x=x_o)


class NPESummaryModel(ModelAPI):
    def __init__(self, posterior, embedding_net, cfg=None):
        self.posterior = posterior
        self.embedding_net = embedding_net
        self.cfg = cfg

    def sample(self, n_samples, x_o, rng=None):
        x_o = torch.tensor(np.array(x_o))
        s_o = self.embedding_net(x_o)
        return self.posterior.sample((n_samples,), x=s_o, show_progress_bars=False)

    def log_prob(self, theta, x_o):
        x_o = torch.tensor(np.array(x_o))
        s_o = self.embedding_net(x_o)
        return self.posterior.log_prob(theta, x=s_o)


def get_window_torch(tensor, win_size):
    """
    Transforms a tensor of shape [T, d] into a tensor of shape [b, win_size*d].

    Args:
        tensor: The input tensor of shape [T, d].
        win_size: The size of the sliding window.

    Returns:
        The transformed tensor of shape [b, win_size*d].
    """

    T, d = tensor.shape
    b = T - win_size + 1

    # Create an empty tensor to store the transformed data
    transformed_tensor = torch.zeros(b, win_size * d)

    # Iterate over the sliding window and copy the data into the transformed tensor
    for i in range(b):
        transformed_tensor[i, :] = tensor[i : i + win_size, :].view(-1)

    return transformed_tensor


from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama, PredictorCorrector, DDIM


from markovsbi.sampling.score_fn import (
    FNPEScoreFn,
    UncorrectedScoreFn,
    GaussCorrectedScoreFn,
    CorrectedScoreFn,
)

from markovsbi.models.simple_scoremlp import (
    build_score_mlp,
    precondition_functions,
    precondition_functions_v2,
)


def build_model(cfg, sde):
    num_steps = cfg.task.num_steps
    nn_params = cfg.method.neural_net

    if nn_params.name == "score_mlp":
        if nn_params.preconditioner == "v1":
            c_in, c_noise, c_out = precondition_functions(sde)
        elif nn_params.preconditioner == "v2":
            c_in, c_noise, c_out = precondition_functions_v2(sde)
        else:
            raise NotImplementedError()

        init_fn, score_net = build_score_mlp(
            num_steps,
            c_in=c_in,
            c_noise=c_noise,
            c_out=c_out,
            hidden_dim=nn_params.hidden_dim,
            num_hidden=nn_params.num_hidden,
            layer_norm=nn_params.layer_norm,
            skip_connection=nn_params.skip_connection,
            activation=eval(nn_params.activation),
            time_embedding_dim=nn_params.time_embedding_dim,
            x_o_processing=nn_params.x_o_processing,
        )
    elif nn_params.name == "score_mlp_with_embedding":
        pass
    else:
        raise NotImplementedError(f"Neural net {nn_params.name} not implemented")

    return init_fn, score_net


def set_factorization_method(cfg, model):
    factorization_method = cfg.method.sampler.factor_method
    name = factorization_method.name

    if name == "FNPE":
        if factorization_method.prior_score_weight == "default":
            prior_score_weight = None
        else:
            raise NotImplementedError("Custom prior score weight not implemented.")

        score_fn = FNPEScoreFn(
            model.score_net,
            model.params,
            model.sde,
            model.prior,
            prior_score_weight=prior_score_weight,
        )
    elif name == "UNCORRECTED":
        score_fn = UncorrectedScoreFn(
            model.score_net, model.params, model.sde, model.prior
        )
    elif name == "GAUS":
        if "prior" in factorization_method.posterior_precission_est_fn:
            scale, _ = factorization_method.posterior_precission_est_fn.split("_")
            prior_precission = float(scale) / model.prior.var
            posterior_precission_est_fn = lambda *args: prior_precission
        elif "posterior" in factorization_method.posterior_precission_est_fn:
            posterior_precission_est_fn = None  # Use automatic routine
        else:
            raise NotImplementedError(
                "Custom posterior precission estimation not implemented."
            )
        window_size = cfg.task.num_steps
        score_fn = GaussCorrectedScoreFn(
            model.score_net,
            model.params,
            model.sde,
            model.prior,
            posterior_precission_est_fn=posterior_precission_est_fn,
            window_size=window_size,
        )
    elif name == "JAC":
        score_fn = CorrectedScoreFn(
            model.score_net, model.params, model.sde, model.prior
        )
    else:
        raise NotImplementedError(f"Factorization method {name} not implemented.")

    model.score_fn = score_fn


def set_diffusion_method(cfg, model):
    diffusion_method = cfg.method.sampler.diffusion_method
    name = diffusion_method.name
    steps = diffusion_method.steps

    if name == "em":
        eta = diffusion_method.eta
        kernel = EulerMaruyama(model.score_fn, eta=eta)
    elif name == "pc":
        predictor_name = diffusion_method.predictor
        corrector_name = diffusion_method.corrector
        corrector_params = dict(diffusion_method.params_corrector)
        if predictor_name == "em":
            predictor = EulerMaruyama
        else:
            raise NotImplementedError(f"Predictor {predictor_name} not implemented.")

        kernel = PredictorCorrector(
            model.score_fn,
            predictor=predictor,
            corrector=corrector_name,
            **corrector_params,
        )
    elif name == "ddim":
        kernel = DDIM(model.score_fn)
    else:
        raise NotImplementedError(f"Diffusion method {name} not implemented.")

    time_grid = jnp.linspace(model.sde.T_min, model.sde.T_max, steps)
    if cfg.method.sampler.factor_method.name == "FNPE":

        def initial_transform(mean, std, x_o, t_o):
            N = x_o.shape[0]
            return mean, 1 / np.sqrt(N) * std

    else:
        initial_transform = None
    diffuser = Diffuser(
        kernel, time_grid, model.input_shape, transform_initital=initial_transform
    )

    model.diffuser = diffuser
