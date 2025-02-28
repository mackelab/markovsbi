from functools import partial
import math
from typing import Optional, Sequence, Callable
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax.random import PRNGKey
from .filtering.base import FilterState, FilterInfo, FilterKernel
from .filtering.particle_filter import ParticleFilter


def filter(
    key: PRNGKey,
    ts: ArrayLike,
    t_o: Optional[ArrayLike],
    x_o: Optional[ArrayLike],
    kernel: FilterKernel,
    *args,
    unpack_fn: Optional[Callable] = None,
    checkpoint_lengths: Optional[Sequence[int]] = None,
    unroll: int = 1,
    **kwargs,
):
    inital_state = kernel.init(*args, t=ts[0], **kwargs)
    if unpack_fn is None:
        unpack_fn = get_default_unpack_fn(kernel)

    def scan_fn(carry, t):
        state, key, i = carry
        key, subkey = jax.random.split(key)
        is_observed = t == t_o[i]

        def update_fn(subkey, state, i):
            state, info = kernel(state, t=t_o[i], observed=x_o[i], rng_key=subkey)
            return state, info, i + 1

        def predict_fn(subkey, state, i):
            state, info = kernel(state, t=t_o[i], rng_key=subkey)
            return state, info, i

        state, info, i = jax.lax.cond(
            is_observed, update_fn, predict_fn, subkey, state, i
        )
        out = unpack_fn(state, info)
        return (state, key, i), out

    carry = (inital_state, key, 0)

    if checkpoint_lengths is None:
        _, output = jax.lax.scan(scan_fn, carry, ts[1:], unroll=unroll)

    else:
        _, output = nested_checkpoint_scan(
            scan_fn, carry, ts[1:], nested_lengths=checkpoint_lengths, unroll=unroll
        )
        output = jax.tree_map(lambda x: jnp.concatenate([inital_output, x]), output)

    inital_output = unpack_fn(inital_state, None)
    output = jax.tree_map(
        lambda init_x, x: jnp.concatenate([init_x[None, ...], x]), inital_output, output
    )
    return output


def filter_log_likelihood(
    key, ts, t_o, x_o, kernel, *args, checkpoint_lengths=None, unroll=1, **kwargs
):
    output = filter(
        key,
        ts,
        t_o,
        x_o,
        kernel,
        *args,
        unpack_fn=unpack_loglikeliood,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
    ll = output.sum()
    return ll


def get_default_unpack_fn(kernel: FilterKernel):
    if isinstance(kernel, ParticleFilter):
        return lambda state, info: state.particles
    else:
        return lambda state, info: (state, info)


def unpack_loglikeliood(state: FilterState, info: FilterInfo):
    if info is not None and hasattr(info, "log_likelihood"):
        return info.log_likelihood
    else:
        return jnp.array(0.0)


def nested_checkpoint_scan(
    f,
    init,
    xs,
    length: Optional[int] = None,
    *,
    nested_lengths: Sequence[int],
    scan_fn: Callable = jax.lax.scan,
    checkpoint_fn: Callable = jax.checkpoint,  # Corrected type hint
    unroll: int = 1,
):
    """A version of lax.scan that supports recursive gradient checkpointing.

    Code taken from: https://github.com/google/jax/issues/2139

    The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
    the required `nested_lengths` argument.

    The key feature of `nested_checkpoint_scan` is that gradient calculations
    require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
    scans, which it achieves by re-evaluating the forward pass
    `len(nested_lengths) - 1` times.

    `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
    single element.

    Args:
        f: function to scan over.
        init: initial value.
        xs: scanned over values.
        length: leading length of all dimensions
        nested_lengths: required list of lengths to scan over for each level of
            checkpointing. The product of nested_lengths must match length (if
            provided) and the size of the leading axis for all arrays in ``xs``.
        scan_fn: function matching the API of lax.scan
        checkpoint_fn: function matching the API of jax.checkpoint.
    """

    if xs is not None:
        length = xs.shape[0]
    if length is not None and length != math.prod(nested_lengths):
        raise ValueError(f"inconsistent {length=} and {nested_lengths=}")

    def nested_reshape(x):
        x = jnp.asarray(x)
        new_shape = tuple(nested_lengths) + x.shape[1:]
        return x.reshape(new_shape)

    _scan_fn = partial(scan_fn, unroll=unroll)

    sub_xs = jax.tree_map(nested_reshape, xs)
    return _inner_nested_scan(f, init, sub_xs, nested_lengths, _scan_fn, checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_map(jnp.concatenate, out)
    return carry, stacked_out
