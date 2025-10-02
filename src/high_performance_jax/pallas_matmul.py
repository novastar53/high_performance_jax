from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

D = 128

def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
    z_ref[...] = activation(x_ref[...] @ y_ref[...])


def matmul(x: jax.Array, y: jax.Array, *, activation):
    return pl.pallas_call(
        partial(matmul_kernel, activation=activation),
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


k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (4, D, D))
y = jax.random.normal(k2, (4, D, D))

z = jax.vmap(partial(matmul, activation=jax.nn.relu))(x, y)
print(jnp.allclose(z, jax.nn.relu(x @ y), atol=0.05))
    

