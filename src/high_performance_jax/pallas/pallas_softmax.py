import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plgpu


INTERPRET_MODE = True # Set to False on GPU

# Pallas softmax
BLK_SIZE = 1024

# Manual softmax (jax)
def manual_softmax(logits):
    max_rows = jnp.max(logits, axis=-1)
    s = jnp.exp(logits - max_rows[..., None])
    l = jnp.sum(s, axis=-1)
    l = l[..., None]
    return s / l 


def softmax_kernel(x_ref, m_ref, l_ref):

    G = x_ref.shape[1]

    max_reg = jnp.full((BLK_SIZE,), -jnp.finfo(jnp.float32).max, dtype=jnp.float32) 
    l_reg = jnp.zeros((BLK_SIZE,), dtype=jnp.float32) 

    def body(t, args):
        max_reg, l_reg = args
        idx = pl.dslice(t * BLK_SIZE, BLK_SIZE)
        row_ids = pl.program_id(0) * BLK_SIZE + jnp.arange(BLK_SIZE)
        row_mask = row_ids < x_ref.shape[0]
        x_tile = plgpu.load(x_ref.at[:, idx], mask=row_mask[:, None], other=-jnp.inf)
        x_tile = x_tile.astype(jnp.float32)
        max_tile = jnp.max(x_tile, axis=-1)
        max_new = jnp.maximum(max_reg, max_tile)
        l_reg = l_reg * jnp.exp(max_reg - max_new) + jnp.sum(jnp.exp(x_tile - max_new), axis=-1)
        max_reg = max_new
        return max_reg, l_reg
        
    max_reg, l_reg = jax.lax.fori_loop(0, G, body, (max_reg, l_reg))
    m_ref[...] = max_reg
    l_ref[...] = l_reg


@jax.jit
def softmax(logits):
    G = pl.cdiv(logits.shape[0], BLK_SIZE)
    m, l = pl.pallas_call(
        partial(softmax_kernel),
        out_shape=[jax.ShapeDtypeStruct((logits.shape[0],), jnp.float32), 
                   jax.ShapeDtypeStruct((logits.shape[0],), jnp.float32)],
        grid=(G,),
        in_specs=[pl.BlockSpec((BLK_SIZE, logits.shape[1]), lambda i: (i, 0))],
        out_specs=[pl.BlockSpec((BLK_SIZE,), lambda i: (i,)),
                   pl.BlockSpec((BLK_SIZE,), lambda i: (i,))],
        interpret=INTERPRET_MODE
    )(logits)
    return m, l


D = 4096*4
key = jax.random.key(0)
logits = jax.random.normal(shape=(D, D), key=key)


max_pl, l = softmax(logits)
max_gt = jnp.max(logits, axis=-1)
l_gt = jnp.sum(jnp.exp(logits - max_gt), axis=-1)
print(max_pl)
print(max_gt)
print(l)
print(l_gt)
assert(jnp.allclose(jnp.squeeze(max_pl),max_gt))
assert(jnp.allclose(jnp.squeeze(l_gt),l))

# JIT compile
softmax_jit = jax.jit(jax.nn.softmax)
softmax_manual_jit = jax.jit(manual_softmax)
softmax_pallas_jit = lambda x: softmax(x)[1]

# Warmup (compile + first run)
_ = softmax_jit(logits).block_until_ready()
_ = softmax_manual_jit(logits).block_until_ready()
_ = softmax_pallas_jit(logits).block_until_ready()

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

t_jax       = bench(softmax_jit, logits)
t_manual    = bench(softmax_manual_jit, logits)
t_pallas    = bench(softmax_pallas_jit, logits)

print(f"Jax Softmax: {t_jax*1e3:.2f} ms")
print(f"Manual Softmax: {t_manual*1e3:.2f} ms")
print(f"Pallas Softmax: {t_pallas*1e3:.2f} ms")
print(f"Speedup (jax / manual): {t_jax / t_pallas:.2f}x")