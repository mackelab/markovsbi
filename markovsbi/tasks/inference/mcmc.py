from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class MCMC:
    _print_every: int = 10
    _print_length: int = 30

    def __init__(
        self,
        kernel,
        verbose: bool = False,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self._acceptance_rate = 0.0

    @partial(jax.jit, static_argnums=(0,))
    def run(self, key, state: NamedTuple, num_steps: int):
        def body_fn(i, val):
            key, state = val
            key, new_key = jax.random.split(key)
            new_state, info = self.kernel(key, state)

            if self.verbose:
                jax.lax.cond(
                    i % self._print_every == 0,
                    lambda: jax.debug.callback(
                        self._print_progress, i, num_steps, info
                    ),
                    lambda: None,
                )

            return (new_key, new_state)

        val = (key, state)
        _, out_state = jax.lax.fori_loop(0, num_steps, body_fn, val)
        return out_state

    def sample(self, key, state: NamedTuple, num_samples: int, thinning: int = 1):
        num_steps = num_samples * thinning
        samples = jnp.empty((num_samples,) + state.position.shape)

        def scan_fn(carry, i):
            samples, key, state = carry
            key, new_key = jax.random.split(key)
            new_state, info = self.kernel(key, state)
            samples = jax.lax.cond(
                i % thinning == thinning - 1,
                lambda: samples.at[i // thinning].set(new_state.position),
                lambda: samples,
            )

            if self.verbose:
                jax.lax.cond(
                    i % self._print_every == 0,
                    lambda: jax.debug.callback(
                        self._print_progress, i, num_steps, info
                    ),
                    lambda: None,
                )

            return (samples, new_key, new_state), None

        carry = (samples, key, state)
        (samples, _, state), _ = jax.lax.scan(scan_fn, carry, jnp.arange(num_steps))

        return samples, state

    def _print_progress(self, iteration, total, info):
        percent = 100 * (iteration / float(total) + self._print_every / total)
        percent = min(percent, 100)

        percent = ("{0:." + str(2) + "f}").format(percent)

        filled_length = int(self._print_length * iteration // total + 1)
        bar = "█" * filled_length + "-" * (self._print_length - filled_length)

        progress_bar = f"\rProgress: |{bar}| {percent}%"

        if info is not None:
            if hasattr(info, "acceptance_rate"):
                self._acceptance_rate = (
                    0.8 * self._acceptance_rate + 0.2 * info.acceptance_rate
                )
                progress_bar += f" | accept_prob: {self._acceptance_rate:.2f} |"

        print(progress_bar, end="\r")

        if iteration == total:
            print()
