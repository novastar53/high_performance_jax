import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

INTERPRET_MODE = False  # Set to True to run on CPU.

BLOCK_M = 128
BLOCK_N = 32
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 2


def _matmul_kernel(a_ref, b_ref, c_ref, *, num_k_tiles: int):
    acc = jnp.zeros((1, BLOCK_M, BLOCK_N), dtype=jnp.float32)

    def body(t, acc):
        k_idx = pl.dslice(t * BLOCK_K, BLOCK_K)
        a_tile = plgpu.load(a_ref.at[0, :, k_idx])
        b_tile = plgpu.load(b_ref.at[0, :, k_idx])
        return acc + pl.dot(a_tile, b_tile, trans_b=True).astype(jnp.float32)

    acc = jax.lax.fori_loop(0, num_k_tiles, body, acc)
    plgpu.store(c_ref, acc.astype(c_ref.dtype))

@jax.jit
def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    outer_dims = a.shape[:-2]
    a_flat = a.reshape(-1, a.shape[-2], a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-2], b.shape[-1])
    B = a_flat.shape[0]
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-2]

    grid = (B, pl.cdiv(M, BLOCK_M), pl.cdiv(N, BLOCK_N))
    num_k_tiles = pl.cdiv(K, BLOCK_K)
    out_shape = jax.ShapeDtypeStruct((B, M, N), a.dtype)

    c_flat = pl.pallas_call(
        functools.partial(_matmul_kernel, num_k_tiles=num_k_tiles),
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_M, K), lambda i, j, k: (i, j, 0)),  # A
            pl.BlockSpec((1, BLOCK_N, K), lambda i, j, k: (i, j, 0)),  # B
        ],
        out_specs=pl.BlockSpec((1, BLOCK_M, BLOCK_N), lambda i, j, k: (i, j, k)),  # C
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        ),
    )(a_flat, b_flat)
    c = c_flat.reshape(outer_dims + (M, N))
    return c


def _bench(fn, *args, iters=10):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times)/len(times)


if __name__ == "__main__":
    key = jax.random.key(0)
    B = 2
    T = 2
    M = N = K = 1024  # pick something big enough and divisible by BM/BN/BK

    a = jax.random.normal(key, (B, T, M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(key, (B, T, N, K), dtype=jnp.bfloat16)

    jax_mm_jit = jax.jit(lambda x, y: x @ y)

    c_matmul = matmul(a, b).block_until_ready()
    c_jax = jax_mm_jit(a, b).block_until_ready()
    assert c_matmul.shape == c_jax.shape == (B, T, M, N)

    t_pallas = _bench(matmul, a, b)
    t_jax = _bench(jax_mm_jit, a, b)

    print(f"Pallas matmul: {t_pallas*1e3:.2f} ms")
    print(f"JAX   matmul: {t_jax*1e3:.2f} ms")
    print(f"Speedup (baseline / pallas): {t_jax / t_pallas:.2f}x")
