from functools import partial
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike


@partial(jax.jit, static_argnames=("precission",), inline=True)
def mv_diag_or_dense(
    A_diag_or_dense: ArrayLike, b: ArrayLike, precission=jax.lax.Precision.HIGHEST
) -> ArrayLike:
    """Dot product of a diagonal matrix and a dense matrix

    Args:
        A (Array): Diagonal matrix
        B (Array): Dense matrix

    Returns:
        Array: Dot product
    """
    A_diag_or_dense = jnp.asarray(A_diag_or_dense)
    dtype = jnp.result_type(A_diag_or_dense.dtype, b.dtype)
    A_diag_or_dense = A_diag_or_dense.astype(dtype)
    b = b.astype(dtype)
    ndim = A_diag_or_dense.ndim

    if ndim == 1:
        return jax.lax.mul(A_diag_or_dense, b)
    else:
        return jax.lax.dot(
            A_diag_or_dense, b, precision=precission, preferred_element_type=dtype
        )


@partial(jax.jit, inline=True)
def solve_diag_or_dense(A_diag_or_dense: ArrayLike, b: ArrayLike):
    """Solve a linear system with a diagonal matrix or a dense matrix

    Args:
        A_diag_or_dense (ArrayLike): _description_
        b (ArrayLike): _description_
        precission (_type_, optional): _description_. Defaults to jax.lax.Precision.HIGHEST.
    """

    A_diag_or_dense = jnp.asarray(A_diag_or_dense)
    dtype = jnp.result_type(A_diag_or_dense.dtype, b.dtype)
    A_diag_or_dense = A_diag_or_dense.astype(dtype)
    b = b.astype(dtype)
    ndim = A_diag_or_dense.ndim

    if ndim == 1:
        return jax.lax.div(b, A_diag_or_dense)
    else:
        return jax.scipy.linalg.solve(A_diag_or_dense, b)


@partial(jax.jit, inline=True)
def add_diag_or_dense(A_diag_or_dense: ArrayLike, B_diag_or_dense: ArrayLike):
    """Add two diagonal matrices or two dense matrices

    Args:
        A_diag_or_dense (ArrayLike): _description_
        B_diag_or_dense (ArrayLike): _description_

    Returns:
        ArrayLike: _description_
    """

    A_diag_or_dense = jnp.asarray(A_diag_or_dense)
    B_diag_or_dense = jnp.asarray(B_diag_or_dense)
    dtype = jnp.result_type(A_diag_or_dense.dtype, B_diag_or_dense.dtype)
    A_diag_or_dense = A_diag_or_dense.astype(dtype)
    B_diag_or_dense = B_diag_or_dense.astype(dtype)
    ndim1 = A_diag_or_dense.ndim
    ndim2 = B_diag_or_dense.ndim

    if ndim1 == 1 and ndim2 == 1:
        return A_diag_or_dense + B_diag_or_dense
    elif ndim1 == 2 and ndim2 == 2:
        return A_diag_or_dense + B_diag_or_dense
    elif ndim1 == 2 and ndim2 == 1:
        return A_diag_or_dense + jnp.diag(B_diag_or_dense)
    elif ndim1 == 1 and ndim2 == 2:
        return jnp.diag(A_diag_or_dense) + B_diag_or_dense
    else:
        raise ValueError("Invalid shapes")


def project_positive_definite_cone(A: ArrayLike):
    """Project a matrix onto the positive definite cone

    Args:
        A (ArrayLike): Matrix to project

    Returns:
        ArrayLike: Projected matrix
    """
    A = jnp.asarray(A)
    if A.ndim == 2:
        A = (A + A.T) / 2
        w, v = jnp.linalg.eigh(A)
        w = jnp.maximum(w, 0)
        return v @ jnp.diag(w) @ v.T
    else:
        return jnp.maximum(A, 0)
