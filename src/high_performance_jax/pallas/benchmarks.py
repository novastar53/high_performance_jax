import time
from functools import partial

import jax
import jax.numpy as jnp

from high_performance_jax.pallas.pallas_matmul_naive import matmul as pallas_matmul
from high_performance_jax.pallas.pallas_softmax import manual_softmax, online_softmax, softmax
from high_performance_jax.pallas.pallas_triton_matmul import DTYPE as TRITON_DTYPE
from high_performance_jax.pallas.pallas_triton_matmul import _AUTOTUNE_CACHE as TRITON_AUTOTUNE_CACHE
from high_performance_jax.pallas.pallas_triton_matmul import _COMPILED_CACHE as TRITON_COMPILED_CACHE
from high_performance_jax.pallas.pallas_triton_matmul import _autotune_config as triton_autotune_config
from high_performance_jax.pallas.pallas_triton_matmul import matmul as triton_matmul


def bench(fn, *args, iters=10):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def bench_softmax():
    d = 1024
    key = jax.random.key(0)
    logits = jax.random.normal(shape=(d, d), key=key)

    out_jax = jax.nn.softmax(logits)
    out_manual = manual_softmax(logits)
    out_online = online_softmax(logits)
    out_pallas = softmax(logits)

    assert jnp.allclose(jnp.squeeze(out_jax), out_online)
    assert jnp.allclose(jnp.squeeze(out_jax), out_manual)
    assert jnp.allclose(jnp.squeeze(out_jax), out_pallas)

    softmax_jit = jax.jit(jax.nn.softmax)
    softmax_manual_jit = jax.jit(manual_softmax)

    _ = softmax_jit(logits).block_until_ready()
    _ = softmax_manual_jit(logits).block_until_ready()
    _ = softmax(logits).block_until_ready()

    t_jax = bench(softmax_jit, logits)
    t_manual = bench(softmax_manual_jit, logits)
    t_pallas = bench(softmax, logits)

    print(f"Jax Softmax: {t_jax*1e3:.2f} ms")
    print(f"Manual Softmax: {t_manual*1e3:.2f} ms")
    print(f"Pallas Softmax: {t_pallas*1e3:.2f} ms")
    print(f"Speedup (jax / pallas): {t_jax / t_pallas:.2f}x")


def bench_matmul():
    b = 1
    d = 256
    g = 16
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (b, d, d), dtype=jnp.float32)
    y = jax.random.normal(k2, (b, d, d), dtype=jnp.float32)
    f = partial(pallas_matmul, g)

    matmul_jit = jax.jit(lambda x, y: x @ y)

    _ = matmul_jit(x, y).block_until_ready()
    _ = jax.vmap(f)(x, y).block_until_ready()

    t_jax = bench(matmul_jit, x, y)
    t_pallas = bench(jax.vmap(f), x, y)

    print(f"Jax Matmul: {t_jax*1e3:.2f} ms")
    print(f"Pallas Matmul: {t_pallas*1e3:.2f} ms")
    print(f"Speedup (jax / pallas): {t_jax / t_pallas:.2f}x")


def bench_triton_matmul():
    m = n = k = 1024
    key = jax.random.key(0)
    a = jax.random.normal(key, (m, k), dtype=TRITON_DTYPE)
    b = jax.random.normal(key, (k, n), dtype=TRITON_DTYPE)

    best_config = triton_autotune_config(a, b)
    print(f"Chosen Triton config: {best_config}")
    compiled_key = tuple(sorted(best_config.items()))
    tuned_mm = TRITON_COMPILED_CACHE[compiled_key]

    jax_mm_jit = jax.jit(lambda x, y: x @ y)

    _ = tuned_mm(a, b).block_until_ready()
    _ = jax_mm_jit(a, b).block_until_ready()

    t_pallas = bench(tuned_mm, a, b)
    t_jax = bench(jax_mm_jit, a, b)

    print(f"Pallas Triton Matmul: {t_pallas*1e3:.2f} ms")
    print(f"Jax Matmul: {t_jax*1e3:.2f} ms")
    print(f"Speedup (jax / pallas triton): {t_jax / t_pallas:.2f}x")


if __name__ == "__main__":
    bench_softmax()
    bench_matmul()
    bench_triton_matmul()
