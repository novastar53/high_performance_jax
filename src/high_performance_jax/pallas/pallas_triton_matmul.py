import functools
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

def matmul_kernel(a_ref, b_ref, c_ref, *, K: int):
    # a_ref: [BM, K]   block of rows of A
    # b_ref: [K, BN]   block of cols of B
    # c_ref: [BM, BN]  output tile

    a_block = a_ref[...]
    b_block = b_ref[...]

    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)

    def body(t, acc):
        k0 = t * BLOCK_K
        a_tile = jax.lax.dynamic_slice(a_block, (0, k0), (BLOCK_M, BLOCK_K))
        b_tile = jax.lax.dynamic_slice(b_block, (k0, 0), (BLOCK_K, BLOCK_M))
        return acc + a_tile @ b_tile

    num_k_tiles = K // BLOCK_K
    acc = jax.lax.fori_loop(0, num_k_tiles, body, acc)
    c_ref[...] = acc.astype(c_ref.dtype)


def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    grid = (M // BLOCK_M, N // BLOCK_N)
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    return pl.pallas_call(
        functools.partial(matmul_kernel, K=K),
        out_shape=out_shape,
        grid=grid,
        # Each kernel instance sees the full A,B,C; program_id picks the tile
        in_specs=[
            pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),  # A
            pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),  # B
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),  # C
        interpret=True
        #compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
    )(a, b)

# Example usage (on GPU):
a = jax.random.normal(jax.random.key(0), (64, 64), dtype=jnp.float32)
b = jax.random.normal(jax.random.key(1), (64, 64), dtype=jnp.float32)
c = matmul(a, b)
jnp.allclose(c, a @ b, atol=1e-3, rtol=1e-3)