import jax.numpy as jnp


from markovsbi.utils.sde import VPSDE, subVPSDE, VESDE


def init_sde(data, name="VPSDE", **kwargs):
    thetas = data["thetas"]
    # VPSDE
    if name.upper() == "VPSDE":
        T_max = kwargs.get("T_max", 1.0)
        T_min = kwargs.get("T_min", 1e-2)
        beta_min = kwargs.get("beta_min", 1e-1)
        beta_max = kwargs.get("beta_max", 1e1)
        mu0 = jnp.mean(thetas, axis=0)
        std0 = jnp.std(thetas, axis=0)

        sde = VPSDE(
            beta_min=beta_min,
            beta_max=beta_max,
            T_min=T_min,
            T_max=T_max,
            mu0=mu0,
            std0=std0,
        )

        min_clip = kwargs.get("min_clip", 1e-2)

        def weight_fn(t):
            return jnp.clip(
                1 - jnp.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t),
                min_clip,
            )

        return sde, weight_fn

    elif name.upper() == "VESDE":
        T_max = kwargs.get("T_max", 1.0)
        T_min = kwargs.get("T_min", 1e-2)
        sigma_min = kwargs.get("sigma_min", 1e-2)
        sigma_max = kwargs.get("sigma_max", 5.0)
        mu0 = jnp.mean(thetas, axis=0)
        std0 = jnp.std(thetas, axis=0)

        sde = VESDE(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            T_min=T_min,
            T_max=T_max,
            mu0=mu0,
            std0=std0,
        )

        min_clip = kwargs.get("min_clip", -jnp.inf)

        def weight_fn(t):
            return jnp.clip(sde.diffusion(t, jnp.ones((1, 1, 1))) ** 2, min_clip)

        sde.T_min = T_min
        sde.T_max = T_max

        return sde, weight_fn

    elif name.upper() == "SUBVPSDE":
        T_max = kwargs.get("T_max", 1.0)
        T_min = kwargs.get("T_min", 1e-1)
        beta_min = kwargs.get("beta_min", 1e-1)
        beta_max = kwargs.get("beta_max", 1e1)
        mu0 = jnp.mean(thetas, axis=0)
        std0 = jnp.std(thetas, axis=0)

        sde = subVPSDE(
            beta_min=beta_min,
            beta_max=beta_max,
            T_min=T_min,
            T_max=T_max,
            mu0=mu0,
            std0=std0,
        )

        min_clip = kwargs.get("min_clip", 1e-10)

        def weight_fn(t):
            return jnp.clip(sde.diffusion(t, jnp.ones((1, 1, 1))) ** 2, min_clip)

        sde.T_min = T_min
        sde.T_max = T_max

        return sde, weight_fn
