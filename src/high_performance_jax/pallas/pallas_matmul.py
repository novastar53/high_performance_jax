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


@jax.jit
def matmul(x: jax.Array, y: jax.Array):
    # pallas_call will first allocate scratch buffers for `x` and `y` in SRAM.
    # It will then copy `x` and `y` from HBM into SRAM.
    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(4, 4),
        in_specs=[
            pl.BlockSpec((x.shape[0] // 4, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // 4), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec(
            (x.shape[0] // 4, y.shape[1] // 4), lambda i, j: (i, j)
        )
    )(x, y)
    # pallas_call will also copy the output from SRAM back into HBM.
    return z


D = 128
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (4, D, D))
y = jax.random.normal(k2, (4, D, D))
z = jax.vmap(matmul)(x, y)

print(jnp.allclose(z, x @ y, atol=0.05))
    

