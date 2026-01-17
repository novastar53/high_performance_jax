from functools import partial

import jax
import jax.numpy as jnp

from jax.experimental import pallas as pl

def matmul_kernel(x_ref, y_ref, o_ref):
    x_reg = x_ref[...]
    y_reg = y_ref[...]
    o_reg = x_reg @ y_reg
    o_ref[...] = o_reg

@partial(jax.jit, static_argnums=(0,))
def matmul(G, x, y):
    o = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(G, G),
        in_specs=[
            pl.BlockSpec((x.shape[0] // G, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // G), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec((x.shape[0] // G, y.shape[1] // G), lambda i, j: (i, j))
    )(x, y)
    return o
