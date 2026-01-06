import math
import time
import functools

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

INTERPRET_MODE = False # Set to False on GPU
DTYPE = jnp.float32

_AUTOTUNE_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
]

_AUTOTUNE_CACHE = {}
_COMPILED_CACHE = {}


def _matmul_kernel_factory(block_m: int, block_n: int, block_k: int):
    def matmul_kernel(a_ref, b_ref, c_ref, *, K: int, num_k_tiles: int):
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)

        def body(t, acc):
            k_idx = pl.dslice(t * block_k, block_k)
            a_tile = plgpu.load(a_ref.at[:, k_idx])
            b_tile = plgpu.load(b_ref.at[k_idx, :])
            return acc + pl.dot(a_tile, b_tile).astype(jnp.float32)

        acc = jax.lax.fori_loop(0, num_k_tiles, body, acc)
        plgpu.store(c_ref, acc.astype(c_ref.dtype))

    return matmul_kernel


def _matmul_with_config(a: jax.Array, b: jax.Array, *, config: dict) -> jax.Array:
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % block_m == 0 and N % block_n == 0 and K % block_k == 0

    grid = (pl.cdiv(M, block_m), pl.cdiv(N, block_n))
    num_k_tiles = pl.cdiv(K, block_k)
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)
    kernel = _matmul_kernel_factory(block_m, block_n, block_k)

    return pl.pallas_call(
        functools.partial(kernel, K=K, num_k_tiles=num_k_tiles),
        out_shape=out_shape,
        grid=grid,
        # Each kernel instance sees the full A,B,C; program_id picks the tile
        in_specs=[
            pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),  # A
            pl.BlockSpec((K, block_n), lambda i, j: (0, j)),  # B
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),  # C
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        ),
    )(a, b)


def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    config = _autotune_config(a, b)
    key = (tuple(sorted(config.items())))
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        fn = jax.jit(lambda x, y, cfg=config: _matmul_with_config(x, y, config=cfg))
        _COMPILED_CACHE[key] = fn
    return fn(a, b)


def _autotune_config(a: jax.Array, b: jax.Array) -> dict:
    key = (a.shape, b.shape, a.dtype, b.dtype)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    best_config = None
    best_time = float("inf")
    for config in _AUTOTUNE_CONFIGS:
        block_m = config["BLOCK_M"]
        block_n = config["BLOCK_N"]
        block_k = config["BLOCK_K"]
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            continue
        if M % block_m or N % block_n or K % block_k:
            continue
        key = (tuple(sorted(config.items())))
        fn = _COMPILED_CACHE.get(key)
        if fn is None:
            fn = jax.jit(lambda x, y, cfg=config: _matmul_with_config(x, y, config=cfg))
            _COMPILED_CACHE[key] = fn
        _ = fn(a, b).block_until_ready()
        t = _bench(fn, a, b, iters=5)
        if t < best_time:
            best_time = t
            best_config = config

    if best_config is None:
        raise ValueError("No valid autotune config for given shapes.")

    _AUTOTUNE_CACHE[key] = best_config
    return best_config

