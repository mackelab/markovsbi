from functools import partial
from markovsbi.tasks.base import Task
import jax_cfd.base as cfd
import math

from markovsbi.utils.prior_utils import Normal

import jax
from jax import Array
import jax.numpy as jnp



class KolmogorovFlow(Task):
    def __init__(self, size: int, dt=0.001, inner_steps=100, sigma=None):
        super().__init__("kolmogorov_flow")

        self.dt = dt
        self.inner_steps = inner_steps
        self.size = size
        self.grid = cfd.grids.Grid(
            shape=(size, size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        self.bc = cfd.boundaries.periodic_boundary_conditions(2)
        self.sigma = sigma

        self.forcing = cfd.forcings.simple_turbulence_forcing(
            grid=self.grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )

        def initial_distribution(key, init_T=0) -> jax.Array:
            key, key_sub, key_sub2 = jax.random.split(key, 3)
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=self.grid,
                maximum_velocity=3.0,
                peak_wavenumber=3.0,
            )
            if init_T > 0:
                x0 = jnp.stack((u.data, v.data))
                prior = self.get_prior()
                simulator = self.get_simulator()
                thetas = prior.sample(key_sub)
                xs = simulator(key_sub2, thetas, init_T, x0)

                return xs[-1]
            else:
                x0 = jnp.stack((u.data, v.data))
                return x0

        self.initial_distribution = initial_distribution

    @property
    def input_shape(self):
        return (2,)

    @property
    def condition_shape(self):
        return (self.size, self.size)

    def get_prior(self):
        return Normal(jnp.zeros((2,)), jnp.ones((2,)))

    def get_data(self, key, num_simulations, T, init_T=0, x0=None):
        prior = self.get_prior()
        simulators = self.get_simulator()
        key1, key2, key3 = jax.random.split(key, 3)
        thetas = prior.sample(key1, (num_simulations,))
        keys2 = jax.random.split(key2, num_simulations)
        if x0 is None:
            x0 = jax.vmap(partial(self.initial_distribution, init_T=init_T))(keys2)
        keys = jax.random.split(key3, num_simulations)
        xs = jax.vmap(simulators, in_axes=(0, 0, None, 0))(keys, thetas, T, x0)

        return {"thetas": thetas, "xs": xs}

    def get_simulator(self):
        dt = self.dt
        grid = self.grid
        inner_steps = self.inner_steps
        f = self.forcing
        bc = self.bc

        def simulator(rng_key, theta: Array, T: int, x0=None):
            density = theta[0]
            reynolds = theta[1]

            # Transorm density between 0 and 1
            density = jax.nn.sigmoid(density)
            reynolds = jax.nn.sigmoid(reynolds)
            reynolds = reynolds * 1999 + 1

            step = cfd.funcutils.repeated(
                f=cfd.equations.semi_implicit_navier_stokes(
                    grid=grid,
                    forcing=f,
                    dt=dt,
                    density=density,
                    viscosity=1 / reynolds,
                ),
                steps=inner_steps,
            )

            def one_step(uv: jax.Array, key) -> jax.Array:
                u, v = cfd.initial_conditions.wrap_variables(
                    var=tuple(uv),
                    grid=grid,
                    bcs=(bc, bc),
                )

                u, v = step((u, v))

                x = jnp.stack((u.data, v.data))

                if self.sigma is not None:
                    x += self.sigma * jax.random.normal(key, x.shape)

                return x, x

            if x0 is None:
                rng_key, subkey = jax.random.split(rng_key)
                x0 = self.initial_distribution(subkey)
            keys = jax.random.split(rng_key, T - 1)
            _, xs = jax.lax.scan(one_step, x0, keys)
            xs = jnp.concatenate([x0[None], xs], axis=0)
            # xs = jax.vmap(vorticity)(xs)
            xs = jnp.nan_to_num(xs, nan=0.0, posinf=3.0, neginf=-3.0)
            return xs

        return simulator
