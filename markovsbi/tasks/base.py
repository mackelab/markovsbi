from abc import ABC, abstractmethod
from typing import Tuple, Callable

from markovsbi.utils.prior_utils import Distribution
import jax


class Task(ABC):
    def __init__(self, name: str):
        self.name = name

    def pred_proposal(self, key, num_sims=50, T_max=100):
        key1, key2 = jax.random.split(key)
        prior = self.get_prior()
        simulator = self.get_simulator()
        thetas_pred = prior.sample(key1, (num_sims,))
        x_pred = jax.vmap(lambda k, theta: simulator(k, theta, T_max))(
            jax.random.split(key2, num_sims), thetas_pred
        )
        x_pred_average = x_pred.reshape(-1, x_pred.shape[-1])

        def proposal(key):
            key, key2 = jax.random.split(key)
            idx = jax.random.randint(key, (), 0, x_pred_average.shape[0])
            x = x_pred_average[idx]
            eps = jax.random.normal(key2, x.shape) * 0.1
            out = x + eps
            return out

        return proposal

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def condition_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def get_prior(self) -> Distribution:
        pass

    @abstractmethod
    def get_simulator(self) -> Callable:
        pass

    def get_true_posterior(self, x_o) -> Distribution:
        raise NotImplementedError("This task does not have a true posterior.")
