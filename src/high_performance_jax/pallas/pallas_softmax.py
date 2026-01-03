import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plgpu


INTERPRET_MODE = True # Set to False on GPU

# Pallas softmax
BLK_SIZE = 128

# Manual softmax (jax)
def manual_softmax(logits):
    max_rows = jnp.max(logits, axis=-1)
    s = jnp.exp(logits - max_rows[..., None])
    l = jnp.sum(s, axis=-1)
    l = l[..., None]
    return s / l 

# Online softmax
def online_softmax(logits):
    out = jnp.zeros_like(logits)
    m = jnp.full((logits.shape[0],), -jnp.inf)
    l = jnp.zeros((logits.shape[0],))
    for i in range(0, logits.shape[0], BLK_SIZE):
        for j in range(0, logits.shape[1], BLK_SIZE):
            block = logits[i:i+BLK_SIZE, j:j+BLK_SIZE]
            block_max = jnp.max(block, axis=-1)
            curr_max = m[i:i+BLK_SIZE]
            new_max = jnp.maximum(curr_max, block_max)
            m = m.at[i:i+BLK_SIZE].set(new_max)
            l_block = l[i:i+BLK_SIZE]
            l_block = l_block * jnp.exp(curr_max - new_max) + jnp.sum(
                jnp.exp(block - new_max[:, None]), axis=-1
            )
            l = l.at[i:i+BLK_SIZE].set(l_block)
        out_block = jnp.exp(logits[i:i+BLK_SIZE, :] - m[i:i+BLK_SIZE][:, None]) / l[i:i+BLK_SIZE][:, None]
        out = out.at[i:i+BLK_SIZE, :].set(out_block)
    
    return out


def softmax_kernel(x_ref, m_ref, l_ref, out_ref, *, n_col_blocks, n_rows, n_cols):
    max_reg = jnp.full((BLK_SIZE,), -jnp.inf, dtype=jnp.float32) 
    l_reg = jnp.zeros((BLK_SIZE,), dtype=jnp.float32) 
    row_ids = pl.program_id(0) * BLK_SIZE + jnp.arange(BLK_SIZE)
    row_mask = row_ids < n_rows

    def stats_body(t, args):
        max_reg, l_reg = args
        idx = pl.dslice(t * BLK_SIZE, BLK_SIZE)
        col_ids = t * BLK_SIZE + jnp.arange(BLK_SIZE)
        cols_mask = col_ids < n_cols
        mask = row_mask[:, None] & cols_mask[None, :]

        x_tile = plgpu.load(
            x_ref.at[:, idx],
            mask=mask,
            other=-jnp.inf,
        ).astype(jnp.float32)
        max_tile = jnp.max(x_tile, axis=-1)
        max_new = jnp.maximum(max_reg, max_tile)
        l_update = l_reg * jnp.exp(max_reg - max_new) + jnp.sum(
            jnp.exp(x_tile - max_new[:, None]), axis=-1
        )
        max_reg = jnp.where(row_mask, max_new, max_reg)
        l_reg = jnp.where(row_mask, l_update, l_reg)
        return max_reg, l_reg
        
    max_reg, l_reg = jax.lax.fori_loop(0, n_col_blocks, stats_body, (max_reg, l_reg))

    def out_body(t, _):
        idx = pl.dslice(t * BLK_SIZE, BLK_SIZE)
        col_ids = t * BLK_SIZE + jnp.arange(BLK_SIZE)
        cols_mask = col_ids < n_cols
        mask = row_mask[:, None] & cols_mask[None, :]

        x_tile = plgpu.load(
            x_ref.at[:, idx],
            mask=mask,
            other=-jnp.inf,
        ).astype(jnp.float32)
        out_tile = jnp.exp(x_tile - max_reg[:, None]) / l_reg[:, None]
        plgpu.store(out_ref.at[:, idx], out_tile.astype(jnp.float32), mask=mask)

    _ = jax.lax.fori_loop(0, n_col_blocks, out_body, None)

    plgpu.store(m_ref, max_reg, mask=row_mask)
    plgpu.store(l_ref, l_reg, mask=row_mask)


@jax.jit
def softmax(logits):
    n_row_blocks = pl.cdiv(logits.shape[0], BLK_SIZE)
    n_col_blocks = pl.cdiv(logits.shape[1], BLK_SIZE)
    return pl.pallas_call(
        partial(softmax_kernel, n_col_blocks=n_col_blocks, n_rows=logits.shape[0], n_cols=logits.shape[1]),
        out_shape=[jax.ShapeDtypeStruct((logits.shape[0],), jnp.float32),
                   jax.ShapeDtypeStruct((logits.shape[0],), jnp.float32),
                   jax.ShapeDtypeStruct(logits.shape, jnp.float32)],
        grid=(n_row_blocks,),
        in_specs=[pl.BlockSpec((BLK_SIZE, logits.shape[1]), lambda i: (i, 0))],
        out_specs=[pl.BlockSpec((BLK_SIZE,), lambda i: (i,)),
                   pl.BlockSpec((BLK_SIZE,), lambda i: (i,)),
                   pl.BlockSpec((BLK_SIZE, logits.shape[1]), lambda i: (i, 0))],
        interpret=INTERPRET_MODE
    )(logits)


D = 1024 
key = jax.random.key(0)
logits = jax.random.normal(shape=(D, D), key=key)

out_manual = manual_softmax(logits)
out_online = online_softmax(logits)
max_pl, l, out_pl = softmax(logits)
max_gt = jnp.max(logits, axis=-1)
l_gt = jnp.sum(jnp.exp(logits - max_gt[..., None]), axis=-1)

# Check correctness
assert(jnp.allclose(jnp.squeeze(out_manual),out_online))
assert(jnp.allclose(jnp.squeeze(out_manual),out_pl))
assert(jnp.allclose(jnp.squeeze(l_gt),l))
assert(jnp.allclose(jnp.squeeze(max_pl),max_gt))

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
print(f"Speedup (jax / pallas): {t_jax / t_pallas:.2f}x")
