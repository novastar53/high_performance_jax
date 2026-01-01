from functools import partial
import time

import jax
import jax.numpy as jnp

from jax.experimental import pallas as pl

def matmul_kernel(x_ref, y_ref, o_ref):
    x_reg = x_ref[...]
    y_reg = y_ref[...]
    o_reg = x_reg @ y_reg
    o_ref[...] = o_reg

@partial(jax.jit, static_argnums=(0,))
def matmul(G, x, y):
    o = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(G, G),
        in_specs=[
            pl.BlockSpec((x.shape[0] // G, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // G), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec((x.shape[0] // G, y.shape[1] // G), lambda i, j: (i, j))
    )(x, y)
    return o

B = 1
D = 256
G = 16
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (B, D, D), dtype=jnp.float32)
y = jax.random.normal(k2, (B, D, D), dtype=jnp.float32)
f = partial(matmul, G)
#z = jax.vmap(f)(x, y)

matmul_jit = jax.jit(lambda x, y: x @ y)

# Warmup (compile + first run)
_ = matmul_jit(x, y).block_until_ready()
_ = jax.vmap(f)(x, y).block_until_ready()

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

t_jax       = bench(matmul_jit, x, y)
t_pallas    = bench(jax.vmap(f), x, y)

print(f"Jax Matmul: {t_jax*1e3:.2f} ms")
print(f"Pallas Matmul: {t_pallas*1e3:.2f} ms")
print(f"Speedup (jax / pallas): {t_jax / t_pallas:.2f}x")