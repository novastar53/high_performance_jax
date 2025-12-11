import functools
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

def matmul_kernel(a_ref, b_ref, c_ref, *, k: int):
    # 2D program id: which output tile (block) am I computing?
    pid_m = pl.program_id(0)  # which block of rows
    pid_n = pl.program_id(1)  # which block of cols

    # Compute the row/col range for this block
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    m_slice = pl.dslice(m0, BLOCK_M)   # rows [m0, m0 + BLOCK_M)
    n_slice = pl.dslice(n0, BLOCK_N)   # cols [n0, n0 + BLOCK_N)

    # Accumulator tile in registers/SMEM
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)

    # Loop over K dimension in BLOCK_K chunks
    def body(kk, acc):
        k0 = kk * BLOCK_K
        k_slice = pl.dslice(k0, BLOCK_K)  # [k0, k0 + BLOCK_K)

        # Load tiles explicitly from global memory
        a_tile = plgpu.load(a_ref.at[m_slice, k_slice], other=0.0)
        b_tile = plgpu.load(b_ref.at[k_slice, n_slice], other=0.0)

        # Tile matmul: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N)
        acc = acc + pl.dot(a_tile, b_tile)
        return acc

    num_k_tiles = k // BLOCK_K
    acc = lax.fori_loop(0, num_k_tiles, body, acc)

    # Write the result tile back to global memory
    plgpu.store(c_ref.at[m_slice, n_slice], acc.astype(c_ref.dtype))

def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    grid = (M // BLOCK_M, N // BLOCK_N)
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    return pl.pallas_call(
        functools.partial(matmul_kernel, k=K),
        out_shape=out_shape,
        grid=grid,
        # Each kernel instance sees the full A,B,C; program_id picks the tile
        in_specs=[
            pl.BlockSpec((None, None), lambda i, j: (0, 0)),  # A
            pl.BlockSpec((None, None), lambda i, j: (0, 0)),  # B
        ],
        out_specs=pl.BlockSpec((None, None), lambda i, j: (0, 0)),  # C
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
    )(a, b)

# Example usage (on GPU):
# a = jax.random.normal(jax.random.key(0), (64, 64), dtype=jnp.float32)
# b = jax.random.normal(jax.random.key(1), (64, 64), dtype=jnp.float32)
# c = matmul(a, b)
# jnp.allclose(c, a @ b, atol=1e-3, rtol=1e-3)