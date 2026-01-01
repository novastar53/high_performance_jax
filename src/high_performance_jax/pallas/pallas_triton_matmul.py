import time
import functools

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 8
NUM_STAGES = 4

INTERPRET_MODE = False # Set to False on GPU

def matmul_kernel(a_ref, b_ref, c_ref, *, K: int, num_k_tiles: int):
    # a_ref: [BM, K]   block of rows of A
    # b_ref: [K, BN]   block of cols of B
    # c_ref: [BM, BN]  output tile

    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)

    def body(t, acc):
        k_idx = pl.dslice(t * BLOCK_K, BLOCK_K)
        a_tile = plgpu.load(a_ref.at[:, k_idx])
        b_tile = plgpu.load(b_ref.at[k_idx, :])
        return acc + pl.dot(a_tile, b_tile).astype(jnp.float32)

    acc = jax.lax.fori_loop(0, num_k_tiles, body, acc)
    plgpu.store(c_ref, acc.astype(c_ref.dtype))


@jax.jit
def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    grid = (M // BLOCK_M, N // BLOCK_N)
    num_k_tiles = K // BLOCK_K
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    return pl.pallas_call(
        functools.partial(matmul_kernel, K=K, num_k_tiles=num_k_tiles),
        out_shape=out_shape,
        grid=grid,
        # Each kernel instance sees the full A,B,C; program_id picks the tile
        in_specs=[
            pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),  # A
            pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),  # B
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),  # C
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
    )(a, b)




key = jax.random.key(0)
M = N = K = 4096  # pick something big enough and divisible by BM/BN/BK

a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)

# JIT both
jax_mm_jit = jax.jit(lambda x, y: x @ y)

# Warmup (compile + first run)
_ = matmul(a, b).block_until_ready()
_ = jax_mm_jit(a, b).block_until_ready()

def bench(fn, *args, iters=10):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()   # very important
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times)//2]   # median

t_pallas = bench(matmul, a, b)
t_jax    = bench(jax_mm_jit, a, b)

print(f"Pallas matmul: {t_pallas*1e3:.2f} ms")
print(f"JAX   matmul: {t_jax*1e3:.2f} ms")
print(f"Speedup (baseline / pallas): {t_jax / t_pallas:.2f}x")
