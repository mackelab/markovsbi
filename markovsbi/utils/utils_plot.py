import seaborn

import numpy as np
from jax.typing import ArrayLike

import jax.numpy as jnp

from PIL import Image


def vorticity2rgb(
    w: ArrayLike,
    vmin: float = -1.25,
    vmax: float = 1.25,
) -> ArrayLike:
    w = np.asarray(w)
    w = (w - vmin) / (vmax - vmin)
    w = 2 * w - 1
    w = np.sign(w) * np.abs(w) ** 0.8
    w = (w + 1) / 2
    w = seaborn.cm.icefire(w)
    w = 256 * w[..., :3]
    w = w.astype(np.uint8)

    return w


def circular_pad_jax(y):
    # Get the dimensions of the input tensor
    shape = jnp.shape(y)
    # Define the pad widths (1 for each dimension)
    pad_widths = [(0, 0) if i == 0 else (1, 1) for i in range(len(shape))]
    # Circular padding in JAX
    padded_y = jnp.pad(y, pad_widths, mode="wrap")
    return padded_y


def vorticity(x):
    h, w = x.shape[-2:]

    # Pad the input array circularly
    y = circular_pad_jax(x)

    # Compute du/dx and dv/dy using finite differences
    du_dx = (y[0, 2:, 1:-1] - y[0, :-2, 1:-1]) / 2.0
    dv_dy = (y[1, 1:-1, 2:] - y[1, 1:-1, :-2]) / 2.0

    # Compute vorticity (du/dx - dv/dy)
    vort = du_dx - dv_dy

    # Reshape back to original shape

    vort = vort.reshape((h, w))

    return vort


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    w = vorticity2rgb(w, **kwargs)
    w = w[(None,) * (5 - w.ndim)]

    M, N, H, W, _ = w.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask = mask[(None,) * (4 - mask.ndim)]

    img = Image.new(
        "RGB",
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=(255, 255, 255),
    )

    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )

            img.paste(Image.fromarray(w[i][j]), offset)

            if mask is not None:
                img.paste(
                    Image.new("L", size=(W, H), color=240),
                    offset,
                    Image.fromarray(~mask[i][j]),
                )

    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    else:
        return img
