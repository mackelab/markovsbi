import jax
import jax.numpy as jnp

import optax

from markovsbi.bm.api_utils import DiffuserModel, build_model

from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn
from markovsbi.utils.sde_utils import init_sde


def run_train_factorized_diffusion(cfg, task, data, rng):
    # Initialize SDE
    rng, rng_init = jax.random.split(rng)
    sde, weight_fn = init_sde(
        data, name=cfg.method.sde.name, **dict(cfg.method.sde.params)
    )

    # Build model
    init_fn, score_net = build_model(cfg, sde)

    # Build batch sampler and loss function
    loss_fn = build_loss_fn(
        cfg.method.loss_fn.name,
        score_net,
        sde,
        weight_fn,
        **dict(cfg.method.loss_fn.params),
    )
    t_val = jnp.linspace(sde.T_min, sde.T_max, 500)
    val_loss = build_loss_fn(
        cfg.method.loss_fn.name,
        score_net,
        sde,
        # lambda t: jnp.array(1.0),
        weight_fn,
        time_fn=lambda rng, batch_size, T_min, T_max: t_val,
        **dict(cfg.method.loss_fn.params),
    )
    val_loss = jax.jit(val_loss)

    # Train model
    params_train = cfg.method.params_train

    if "validation" in dict(params_train):
        validation = True
        validation_size = min(
            params_train.validation, int(0.1 * cfg.task.num_simulations)
        )
        data_train = jax.tree_util.tree_map(lambda x: x[:-validation_size], data)
        data_val = jax.tree_util.tree_map(lambda x: x[-validation_size:], data)
        data = data_train
    else:
        validation = False

    batch_sampler = build_batch_sampler(data)
    if validation:
        batch_sampler_val = build_batch_sampler(data_val)

    # Initialize model
    theta_batch, x_batch = batch_sampler(rng_init, 10)
    params = init_fn(rng_init, jnp.ones((10,)), theta_batch, x_batch)

    if params_train.scheduler == "cosine":
        schedule = optax.cosine_onecycle_schedule(
            int(
                params_train.num_epochs
                * params_train.num_inner_epochs
                * (cfg.task.num_simulations // params_train.training_batch_size)
            ),
            params_train.learning_rate,
        )
    elif params_train.scheduler == "constant":
        schedule = params_train.learning_rate
    else:
        raise NotImplementedError(
            f"Scheduler {params_train.scheduler} not implemented."
        )
    optimizer_class = eval(params_train.optimizer)
    if params_train.ema:
        optimizer = optax.chain(
            optax.adaptive_grad_clip(params_train.clip_max_norm),
            optimizer_class(schedule),
            optax.ema(decay=params_train.ema_decay),
        )
    else:
        optimizer = optax.chain(
            optax.adaptive_grad_clip(params_train.clip_max_norm),
            optimizer_class(schedule),
        )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, rng, opt_state, theta_batch, x_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    best_loss = jnp.inf
    best_params = None
    inner_iter = int(
        params_train.num_inner_epochs
        * (cfg.task.num_simulations // params_train.training_batch_size + 1)
    )
    for i in range(params_train.num_epochs):
        l = 0.0
        l_val = 0.0
        for _ in range(inner_iter):
            rng, rng_batch, rng_loss = jax.random.split(rng, 3)
            theta_batch, x_batch = batch_sampler(
                rng_batch, params_train.training_batch_size
            )
            loss, params, opt_state = update(
                params, rng_loss, opt_state, theta_batch, x_batch
            )
            l += loss / inner_iter

        if validation:
            for _ in range(500):
                rng, rng_loss, rng_batch_val = jax.random.split(rng, 3)
                l_val += (
                    val_loss(
                        params,
                        rng_loss,
                        *batch_sampler_val(
                            rng_batch_val,
                            500,  # params_train.training_batch_size
                        ),
                    )
                    / 500
                )

        print("Training loss:", l)
        print("Validation loss:", l_val)
        if validation:
            l = l_val
        if l < best_loss:
            best_loss = l
            best_params = params.copy()

    params = best_params if best_params is not None else params

    # Initialize sampler
    prior = task.get_prior()
    num_steps = cfg.method.sampler.diffusion_method.steps
    model = DiffuserModel(
        params, score_net, sde, prior, num_steps, task.input_shape, cfg=cfg
    )

    return model
