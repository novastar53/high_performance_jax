from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
    # Load x and y from SRAM into registers
    x_regs = x_sram_ref[:, :]
    y_regs = y_sram_ref[:, :]
    # Execute a vectorized matmul
    z_regs = x_regs @ y_regs
    # Store the output values in registers back into SRAM
    z_sram_ref[:, :] = z_regs


@partial(jax.jit, static_argnums=(0,))
def matmul(G: int, x: jax.Array, y: jax.Array):
    # pallas_call will first allocate scratch buffers for `x` and `y` in SRAM.
    # It will then copy `x` and `y` from HBM into SRAM.
    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(G, G),
        in_specs=[
            pl.BlockSpec((x.shape[0] // G, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // G), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec(
            (x.shape[0] // G, y.shape[1] // G), lambda i, j: (i, j)
        )
    )(x, y)
    # pallas_call will also copy the output from SRAM back into HBM.
    return z


B = 4
D = 512
G = 32
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (B, D, D), dtype=jnp.float32)
y = jax.random.normal(k2, (B, D, D), dtype=jnp.float32)
f = partial(matmul, G)
z = jax.vmap(f)(x, y)
print(jnp.allclose(z, x @ y, atol=0.06))
    

