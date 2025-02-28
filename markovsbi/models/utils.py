import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3))
def get_windows(x: ArrayLike, window_size: int, stride: int, axis: int = 0):
    """
    Get windows of a given size from a 1D array.

    Parameters
    ----------
    x : Array
        1D array from which to extract windows.
    window_size : int
        Size of the windows to extract.

    Returns
    -------
    Array
        2D array of shape (len(x)-window_size+1, window_size) containing the windows.
    """
    # Make this vectorized
    ndim = len(x.shape)
    shape = x.shape

    def window_x(i):
        slice_start = [0] * ndim
        slice_sizes = list(shape)
        slice_start[axis] = i
        slice_sizes[axis] = window_size  # Change this line
        return jax.lax.dynamic_slice(x, slice_start, slice_sizes)

    idx = jnp.arange(0, shape[axis] - window_size + 1, stride)

    return jax.vmap(window_x, out_axes=axis)(idx)
