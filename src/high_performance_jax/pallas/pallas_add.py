from functools import partial

import numpy as np
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y
  

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
     add_vectors_kernel,
     out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

out = add_vectors(jnp.arange(8), jnp.arange(8))
print(out.shape, out.dtype, out, out.device)


total_shape = (4096, 4096)
block_shape = (4096, 4096)


def add_matrices_pipelined_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def add_matrices_pipelined(x: jax.Array, y: jax.Array):
    return pl.pallas_call(
        add_matrices_pipelined_kernel,
        grid=tuple(total // block for (total, block) in zip(total_shape, block_shape)),
        in_specs=[
            pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
            pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j))
        ],
        out_specs = pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
        out_shape = jax.ShapeDtypeStruct(total_shape, dtype=jnp.float32),
    )(x, y)


x = jax.random.uniform(jax.random.key(0), total_shape, dtype=jnp.float32)
y = jax.random.uniform(jax.random.key(1), total_shape, dtype=jnp.float32)
result = add_matrices_pipelined(x, y)
np.testing.assert_array_equal(
    result, x + y
)


# Reduction (TPU)
'''
def correct_sum_kernel(x_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += x_ref[...]


def correct_sum(x: jax.Array, block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
    reduction_size, *out_shape = x.shape
    grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
    return pl.pallas_call(
        correct_sum_kernel,
        grid=grid,
        in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
        out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
    )(x)


x = jnp.ones((8, 1024, 1024))
print(jnp.sum(x, axis=0))

result = correct_sum(x)
print(result)
'''