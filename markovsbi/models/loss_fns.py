import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from functools import partial

from typing import Optional, Sequence, Callable
from jaxtyping import PyTree, Array


# Flow matching objectives


@partial(jax.jit, static_argnames=("model_fn", "mean_fn", "std_fn"))
def conditional_flow_and_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_source: Array,
    xs_target: Array,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    *args,
    estimate_score: bool = False,
):
    """This function computes the conditional flow matching loss and score matching loss. By setting estimate_score to False, only the conditional flow matching loss is computed. By setting estimate_score to True, both the conditional flow matching loss and score matching loss are computed.


    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_source (Array): Marginal distribution at time t=0, refered to as source distribution.
        xs_target (Array): Marginal distribution at time t, refered to as target distribution.
        model_fn (Callable): Model_fn that takes parameters, times, and samples as input and returns the vector field and optionally the marginal score. Should be a function of the form model_fn(params, times, xs_t) -> v_t(, s_t).
        mean_fn (Callable): The mean function of the Gaussian probability path, should satisfy the following:
                                - mean_fn(xs_source, xs_target, 0) -> xs_source
                                - mean_fn(xs_source, xs_target, 1) -> xs_target
                                - Lipschitz continuous in time
        std_fn (Callable): The standard deviation function of the Gaussian probability path, should satisfy the following:
                                - std_fn(xs_source, xs_target, 0) -> 0
                                - std_fn(xs_source, xs_target, 1) -> 0
                                - std_fn(xs_source, xs_target, t) > 0 for all t in [0, 1]
                                - Two times continuously differentiable in time.
        estimate_score (bool, optional): If set to true, both flow and score matching objectives are computed. Defaults to False.

    Returns:
        (loss_flow, Optional[loss_score]): Respective loss functions
    """
    # Sample x_t
    eps = jax.random.normal(key, shape=xs_source.shape)
    xs_t = (
        mean_fn(xs_source, xs_target, times) + std_fn(xs_source, xs_target, times) * eps
    )

    # Compute u_t -> For flow matching
    # This is valid for Gaussian probability paths, which is currented here.
    t = jnp.broadcast_to(
        times, xs_target.shape
    )  # Pad to x shape for jax.grad -> x.shape
    std_fn_grad = jax.grad(lambda x_s, x_t, t: std_fn(x_s, x_t, t).sum(), argnums=2)
    mean_fn_grad = jax.grad(lambda x_s, x_t, t: mean_fn(x_s, x_t, t).sum(), argnums=2)
    u_t = std_fn_grad(xs_source, xs_target, t) * eps + mean_fn_grad(
        xs_source, xs_target, t
    )

    # Compute loss
    if not estimate_score:
        # Compute vector field -> Flow matching loss
        v_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))

        return loss
    else:
        # Compute vector field and marginal score -> Flow matching loss + Score matching loss
        v_t, s_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))
        loss_score = jnp.mean(
            jnp.sum((s_t + 1 / std_fn(xs_source, xs_target, times) * eps) ** 2, axis=-1)
        )

        return loss, loss_score


def denoising_score_matching_loss(
    params: PyTree,
    times: Array,
    xs_target: Array,
    model_fn: Callable,
    *args,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Optional[Callable] = None,
    loss_mask: Optional[Array] = None,
    rng_key: Optional[PRNGKey] = None,
    rebalance_loss: bool = False,
    control_variate: bool = True,
    control_variate_cutoff: Optional[float] = None,
    control_variate_optimal_scaling: bool = False,
    axis: int = -1,
    **kwargs,
) -> Array:
    """This function computes the denoising score matching loss. Which can be used to train diffusion models.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random generator key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        loss_mask (Optional[Array]): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1).
        model_fn (Callable): Score model that takes parameters, times, and samples as input and returns the score. Should be a function of the form model_fn(params, times, xs_t, *args) -> s_t.
        mean_fn (Callable): Mean function of the SDE.
        std_fn (Callable): Std function of the SDE.
        weight_fn (Callable): Weight function for the loss.
        axis (int, optional): Axis to sum over. Defaults to -2.


    Returns:
        Array: Loss
    """
    assert rng_key is not None, (
        "rng_key must be provided for denoising score matching loss."
    )

    eps = jax.random.normal(rng_key, shape=xs_target.shape)
    mean_t = mean_fn(times, xs_target)
    std_t = std_fn(times, xs_target)

    xs_t = mean_t + std_t * eps

    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    score_pred = model_fn(params, times, xs_t, *args, **kwargs)
    score_pred = score_pred.reshape(xs_t.shape)
    score_target = -eps / std_t

    loss = jnp.sum((score_pred - score_target) ** 2, axis=axis)

    if control_variate:
        # Adds a control variate to the loss, which is efficient for small std_t
        s = model_fn(params, times, mean_t, *args, **kwargs)
        s = s.reshape(xs_target.shape)

        term1 = 2 / std_t * jnp.sum(eps * s, axis=axis, keepdims=True)
        term2 = jnp.sum(eps**2, axis=axis, keepdims=True) / std_t**2
        term3 = xs_target.shape[axis] / std_t**2

        cv = jnp.mean(term3 - term1 - term2, axis=axis)

        if control_variate_cutoff is not None:
            cv = jnp.where(std_t < control_variate_cutoff, cv, 0.0)

        # if control_variate_optimal_scaling:
        #     cv_var = jnp.var(cv)
        #     cv_loss_covar = jnp.mean((cv - jnp.mean(cv)) * (loss - jnp.mean(loss)))

        #     beta = cv_loss_covar / cv_var
        #     # jax.debug.print("{beta}", beta=beta)
        #     cv = -beta * cv

        loss = loss + cv

    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0, loss)

    if weight_fn is not None:
        weight = weight_fn(times)
        for _ in range(xs_target.ndim - 1):
            weight = weight[..., None]

        loss = weight * loss

    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)

    loss = jnp.mean(loss)

    return loss


def high_order_denosing_score_matching_loss(
    params: PyTree,
    times: Array,
    xs_target: Array,
    model_fn: Callable,
    *args,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Optional[Callable] = None,
    loss_mask: Optional[Array] = None,
    rng_key: Optional[PRNGKey] = None,
    rebalance_loss: bool = False,
    diagonal_2nd_order: bool = False,
    c: float = 0.1,
    stop_gradient_2nd_order: bool = True,
    control_variate1: bool = False,
    control_variate2: bool = False,
    axis: int = -1,
    **kwargs,
) -> Array:
    assert rng_key is not None, (
        "rng_key must be provided for denoising score matching loss."
    )

    eps = jax.random.normal(rng_key, shape=xs_target.shape)
    mean_t = mean_fn(times, xs_target)
    std_t = std_fn(times, xs_target)
    xs_t = mean_t + std_t * eps

    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    score_pred, second_order_score_pred = model_fn(params, times, xs_t, *args, **kwargs)
    score_pred = score_pred.reshape(xs_t.shape)
    score_target = -eps / std_t

    loss1 = jnp.sum((score_pred - score_target) ** 2, axis=axis, keepdims=True)

    if control_variate1:
        # Adds a control variate to the loss, which is efficient for small std_t
        s, _ = model_fn(params, times, mean_t, *args, **kwargs)
        s = s.reshape(xs_target.shape)

        term1 = 2 / std_t * jnp.sum(eps * s, axis=axis, keepdims=True)
        term2 = jnp.sum(eps**2, axis=axis, keepdims=True) / std_t**2
        term3 = xs_target.shape[axis] / std_t**2

        cv = jnp.mean(-term1 - term2 + term3, axis=axis, keepdims=True)
        loss1 = loss1 + cv

    if not control_variate2:
        if not diagonal_2nd_order:
            s2_target = jnp.einsum("...i,...j->...ij", eps, eps)
            s1_2 = jnp.einsum("...i,...j->...ij", score_pred, score_pred)
            if stop_gradient_2nd_order:
                s1_2 = jax.lax.stop_gradient(s1_2)
            I = jnp.eye(xs_target.shape[-1])
            loss2 = (second_order_score_pred + s1_2 + (I - s2_target) / std_t**2) ** 2
            loss2 = jnp.sum(loss2, axis=axis)
            loss2 = jnp.sum(loss2, axis=axis - 1, keepdims=True)
        else:
            s2_target = eps**2
            s1_2 = score_pred**2
            if stop_gradient_2nd_order:
                s1_2 = jax.lax.stop_gradient(s1_2)
            loss2 = (second_order_score_pred + s1_2 + (1 - s2_target) / std_t**2) ** 2
            loss2 = jnp.sum(loss2, axis=axis, keepdims=True)
    else:
        x_t_plus = mean_t + std_t * eps
        x_t_minus = mean_t - std_t * eps
        s_plus, s2_plus = model_fn(params, times, x_t_plus, *args, **kwargs)
        s_minus, s2_minus = model_fn(params, times, x_t_minus, *args, **kwargs)
        s_clean, s2_clean = model_fn(params, times, mean_t, *args, **kwargs)
        s_plus = s_plus.reshape(xs_target.shape)
        s_minus = s_minus.reshape(xs_target.shape)
        s_clean = s_clean.reshape(xs_target.shape)
        s2_plus = s2_plus.reshape(xs_target.shape + (xs_target.shape[-1],))
        s2_minus = s2_minus.reshape(xs_target.shape + (xs_target.shape[-1],))
        s2_clean = s2_clean.reshape(xs_target.shape + (xs_target.shape[-1],))

        s2_target = jnp.einsum("...i,...j->...ij", eps, eps)
        s1_2_plus = jnp.einsum("...i,...j->...ij", s_plus, s_plus)
        s1_2_minus = jnp.einsum("...i,...j->...ij", s_minus, s_minus)
        s1_2_clean = jnp.einsum("...i,...j->...ij", s_clean, s_clean)
        if stop_gradient_2nd_order:
            s1_2_plus = jax.lax.stop_gradient(s1_2_plus)
            s1_2_minus = jax.lax.stop_gradient(s1_2_minus)
            s1_2_clean = jax.lax.stop_gradient(s1_2_clean)

        phi_plus = s2_plus + s1_2_plus
        phi_minus = s2_minus + s1_2_minus
        phi_clean = s2_clean + s1_2_clean
        print(phi_plus.shape, phi_minus.shape, phi_clean.shape)

        loss2 = (
            phi_plus**2
            + phi_minus**2
            + 2
            * (jnp.eye(xs_target.shape[-1]) - s2_target)
            / std_t
            * (phi_plus + phi_minus - 2 * phi_clean)
        )
        loss2 = jnp.sum(loss2, axis=axis)

    print(loss1.shape, loss2.shape)
    loss = c * loss1 + (1 - c) * loss2

    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0, loss)

    if weight_fn is not None:
        weight = weight_fn(times)
        for _ in range(xs_target.ndim - 1):
            weight = weight[..., None]

        loss = weight * loss

    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)

    loss = jnp.mean(loss)
    return loss


def score_matching_loss(
    params: PyTree,
    times: Array,
    xs_target: Array,
    model_fn: Callable,
    *args,
    mean_fn: Optional[Callable] = None,
    std_fn: Optional[Callable] = None,
    weight_fn: Optional[Callable] = None,
    loss_mask: Optional[Array] = None,
    rebalance_loss: bool = False,
    rng_key: Optional[PRNGKey] = None,
    tikhonov: Optional[float] = None,
    jac_fn: Callable = jax.jacfwd,
    vmap_args: Optional[Sequence[int]] = None,
    axis: int = -1,
    **kwargs,
):
    """Score matching loss. Minimizing the Fisher divergence between the model and the target distribution, using partial integration trick.

    NOTE: This becomes inefficient when the dimension of the target distribution is high, as the Jacobian of the model is computed.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        model_fn (Callable): _description_
        args: Additional arguments to the model_fn.
        loss_mask (Optional[Array], optional): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1). Defaults to None.
        tikhonov (float, optional): Tikhonov regularization. Defaults to 0.0.
        jac_fn (Callable, optional): Jacobian function. Defaults to jax.jacfwd.

    Returns:
        Array: Loss
    """

    if mean_fn is not None and std_fn is not None:
        assert rng_key is not None, (
            "rng_key must be when mean_fn and std_fn are provided."
        )
        eps = jax.random.normal(rng_key, shape=xs_target.shape)
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        xs_t = mean_t + std_t * eps
    else:
        xs_t = xs_target

    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    _model_fn = partial(model_fn, **kwargs)
    args_vmap = (0,) * len(args) if vmap_args is None else vmap_args
    jac_model_fn = jax.vmap(
        jac_fn(_model_fn, argnums=2), in_axes=(None, 0, 0) + args_vmap
    )
    score = _model_fn(params, times, xs_t, *args)
    jac_score = jac_model_fn(params, times, xs_t, *args)

    loss = 0.5 * jnp.sum(score**2, axis=axis) + jnp.trace(
        jac_score, axis1=axis - 1, axis2=axis
    )

    if tikhonov:
        diag_jac = jnp.diagonal(jac_score, axis1=axis - 1, axis2=axis)
        loss += tikhonov * jnp.sum(diag_jac**2, axis=axis, keepdims=True)

    if loss_mask is not None:
        loss = jnp.where(loss_mask, loss, 0.0)

    if weight_fn is not None:
        weight = weight_fn(times)
        for _ in range(loss.ndim - 1):
            weight = weight[..., None]
        loss = weight * loss

    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)

    return jnp.mean(loss)


def sliced_score_matching(
    params: PyTree,
    times: Array,
    xs_target: Array,
    model_fn: Callable,
    *args,
    mean_fn: Optional[Callable] = None,
    std_fn: Optional[Callable] = None,
    weight_fn: Optional[Callable] = None,
    loss_mask: Optional[Array] = None,
    rebalance_loss: bool = False,
    rng_key: Optional[PRNGKey] = None,
    num_slices: int = 1,
    sliced_dist: str = "normal",
    tikhonov: Optional[float] = None,
    vmap_args: Optional[Sequence[int]] = None,
    axis: int = -1,
    **kwargs,
):
    assert rng_key is not None, (
        "rng_key must be provided for sliced score matching loss."
    )

    rng_key_sample, rng_key_slice = jax.random.split(rng_key)

    if mean_fn is not None and std_fn is not None:
        assert rng_key is not None, (
            "rng_key must be when mean_fn and std_fn are provided."
        )
        eps = jax.random.normal(rng_key_sample, shape=xs_target.shape)
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        xs_t = mean_t + std_t * eps
    else:
        xs_t = xs_target

    def value_and_jvp(t, x, v, *args):
        value, jvp = jax.jvp(
            lambda x: model_fn(params, t, x, *args, **kwargs), (x,), (v,)
        )

        sliced_value = jnp.sum(value * v, axis)
        sliced_jvp = jnp.sum(jvp * v, axis)

        if tikhonov is not None:
            reg = tikhonov * jnp.sum((jvp * v) ** 2, axis)
        else:
            reg = jnp.zeros_like(sliced_value)
        return sliced_value, sliced_jvp, reg

    # Slice directions
    if sliced_dist == "normal":
        v = jax.random.normal(rng_key_slice, shape=(num_slices, *xs_t.shape))
    elif sliced_dist == "rademacher":
        v = jax.random.rademacher(rng_key_slice, shape=(num_slices, *xs_t.shape))
        v = v.astype(jnp.float32)
    elif sliced_dist == "ball":
        v = jax.random.ball(
            rng_key_slice,
            xs_t.shape[-1],
            shape=(num_slices, *xs_t.shape[:-1]),
        )
    else:
        raise ValueError("Invalid sliced_dist")

    args_vmap = (0,) * len(args) if vmap_args is None else vmap_args
    _value_and_jvp = jax.vmap(value_and_jvp, in_axes=(0, 0, 0) + args_vmap)
    sliced_score, jac_trace, reg = jax.vmap(
        _value_and_jvp, in_axes=(None, None, 0) + (None,) * len(args)
    )(times, xs_t, v, *args)

    loss = 0.5 * sliced_score**2 + jac_trace + reg

    # Average over slices
    loss = jnp.mean(loss, axis=0)

    if weight_fn is not None:
        weight = weight_fn(times)
        for _ in range(loss.ndim - 1):
            weight = weight[..., None]
        loss = weight * loss

    if loss_mask is not None:
        loss = jnp.mean(jnp.where(loss_mask, loss, 0.0))

    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)

    return jnp.mean(loss)
