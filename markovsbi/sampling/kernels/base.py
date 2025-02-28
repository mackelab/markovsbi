from typing import NamedTuple, Callable
from functools import partial

from jax.typing import ArrayLike
from markovsbi.sampling.score_fn import ScoreFn


class SDEState(NamedTuple):
    position: ArrayLike
    score: ArrayLike
    a: ArrayLike


class Kernel:
    @staticmethod
    def init_state(position, x_o, a, score_fn: ScoreFn, *args, **kwargs) -> SDEState:
        # Initialize the state
        pass

    @staticmethod
    def build_kernel(score_fn: ScoreFn, *args, **kwargs) -> Callable:
        # Single step of the kernel
        pass

    def __init__(self, score_fn: ScoreFn, *args, **kwargs) -> None:
        super().__init__()
        self.score_fn = score_fn
        self.init = partial(type(self).init_state, score_fn=score_fn)
        self._kernel = type(self).build_kernel(score_fn, *args, **kwargs)

    def __call__(self, key, state, a_new, x_o, **kwargs):
        return self._kernel(key, state, a_new, x_o, **kwargs)
