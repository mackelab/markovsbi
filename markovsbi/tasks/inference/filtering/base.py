from typing import Any, Optional, Tuple, NamedTuple, Callable
from jax.typing import ArrayLike



class API(type):
    """API class for algorithms"""

    def __str__(self):
        return self.__doc__

    def __repr__(self):
        text = self.__doc__
        return text


class FilterState(NamedTuple):
    """This is a NamedTuple that represents the state of a filter.

    It contains all the information **required** to run the filter.
    """

    pass


class FilterInfo(NamedTuple):
    """This is a NamedTuple that represents the information returned by a filter.

    It contains all useful information that can be extracted from the filter.

    """

    pass


class FilterKernel(NamedTuple):
    """This is a NamedTuple that represents a filter kernel."""

    init: Callable
    step: Callable

    def __call__(
        self,
        state: FilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[ArrayLike] = None,
    ) -> Tuple[FilterState, FilterInfo]:
        return self.step(state, t, observed, rng_key)


class FilterAPI(metaclass=API):
    @staticmethod
    def init(*args, **kwargs) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_kernel(*args, **kwargs) -> Any:
        raise NotImplementedError("build_kernel method must be implemented")

    def __new__(cls, *args, **kwargs) -> FilterKernel:
        return FilterKernel(cls.init, cls.build_kernel(*args, **kwargs))
