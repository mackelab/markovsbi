from typing import Optional
import jax.numpy as jnp

from jax.typing import ArrayLike
from abc import ABC, abstractmethod


class SDE(ABC):
    T_min: float
    T_max: float

    def __init__(self, T_min=1e-2, T_max=1.0, mu0=None, std0=None):
        self.T_min = T_min
        self.T_max = T_max
        self.mu0 = mu0
        self.std0 = std0

    @abstractmethod
    def mu(self, t: ArrayLike, *args):
        pass

    @abstractmethod
    def std(self, t: ArrayLike, *args):
        pass

    def mean(self, t: ArrayLike, x0=None):
        alpha = self.mu(t)
        while len(alpha.shape) < len(x0.shape):
            alpha = alpha[..., None]
        return alpha * x0

    @abstractmethod
    def drift(self, t: ArrayLike, x: Optional[ArrayLike]):
        pass

    @abstractmethod
    def diffusion(self, t: ArrayLike, x: Optional[ArrayLike]):
        pass


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: ArrayLike = 1e-1,
        beta_max: ArrayLike = 1e1,
        T_min=1e-2,
        T_max=1.0,
        mu0=None,
        std0=None,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(T_min=T_min, T_max=T_max, mu0=mu0, std0=std0)

    def beta(self, t: ArrayLike):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mu(self, t: ArrayLike, *args):
        alpha = jnp.exp(
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        return alpha

    def std(self, t: ArrayLike, x: Optional[ArrayLike] = None):
        alpha = jnp.exp(
            -0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min
        )

        var = 1.0 - alpha
        std = jnp.sqrt(var)
        if x is not None:
            while len(std.shape) < len(x.shape):
                std = std[..., None]
        return std

    def drift(self, t: ArrayLike, x: ArrayLike):
        a = -0.5 * self.beta(t)
        return a * x

    def diffusion(self, t: ArrayLike, x: ArrayLike):
        return jnp.sqrt(self.beta(t))


class subVPSDE(VPSDE):
    def std(self, t: ArrayLike, x: Optional[ArrayLike] = None):
        std = 1.0 - jnp.exp(
            -0.5 * t**2.0 * (self.beta_max - self.beta_min) - t * self.beta_min
        )

        if x is not None:
            while len(std.shape) < len(x.shape):
                std = std[..., None]
        return std

    def diffusion(self, t: ArrayLike, x: ArrayLike):
        out = jnp.sqrt(
            jnp.abs(
                self.beta(t)
                * (
                    1
                    - jnp.exp(
                        -2 * self.beta_min * t - (self.beta_max - self.beta_min) * t**2
                    )
                )
            )
        )
        return out


class VESDE(SDE):
    def __init__(
        self, sigma_min=1e-5, sigma_max=5.0, T_min=0.01, T_max=1, mu0=None, std0=None
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(T_min, T_max, mu0, std0)

    def sigma(self, t: ArrayLike):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def mu(self, t: ArrayLike, *args):
        return jnp.ones_like(t)

    def std(self, t: ArrayLike, x: Optional[ArrayLike] = None, **kwargs):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        if x is not None:
            while len(std.shape) < len(x.shape):
                std = std[..., None]
        return std

    def drift(self, t: ArrayLike, x: ArrayLike):
        return jnp.zeros_like(x)

    def diffusion(self, t: ArrayLike, x: ArrayLike):
        g = self.sigma(t) * jnp.sqrt((2 * jnp.log(self.sigma_max / self.sigma_min)))
        while len(g.shape) < len(x.shape):
            g = g[..., None]
        return g
