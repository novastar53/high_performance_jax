import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plgpu


INTERPRET_MODE = False # Set to False on GPU

# Pallas softmax
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 3


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
    for i in range(0, logits.shape[0], BLOCK_M):
        for j in range(0, logits.shape[1], BLOCK_N):
            block = logits[i:i+BLOCK_M, j:j+BLOCK_N]
            block_max = jnp.max(block, axis=-1)
            curr_max = m[i:i+BLOCK_M]
            new_max = jnp.maximum(curr_max, block_max)
            m = m.at[i:i+BLOCK_M].set(new_max)
            l_block = l[i:i+BLOCK_M]
            l_block = l_block * jnp.exp(curr_max - new_max) + jnp.sum(
                jnp.exp(block - new_max[:, None]), axis=-1
            )
            l = l.at[i:i+BLOCK_M].set(l_block)
        out_block = jnp.exp(logits[i:i+BLOCK_M, :] - m[i:i+BLOCK_M][:, None]) / l[i:i+BLOCK_M][:, None]
        out = out.at[i:i+BLOCK_M, :].set(out_block)
    
    return out


def softmax_kernel(x_ref, out_ref, *, n_col_blocks, n_rows, n_cols):
    max_reg = jnp.full((BLOCK_M,), -jnp.inf, dtype=jnp.float32) 
    l_reg = jnp.zeros((BLOCK_M,), dtype=jnp.float32) 
    row_ids = pl.program_id(0) * BLOCK_M + jnp.arange(BLOCK_M)
    row_mask = row_ids < n_rows

    def stats_body(t, args):
        max_reg, l_reg = args
        idx = pl.dslice(t * BLOCK_N, BLOCK_N)
        col_ids = t * BLOCK_N + jnp.arange(BLOCK_N)
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
        return max_new, l_update
        
    max_reg, l_reg = jax.lax.fori_loop(0, n_col_blocks, stats_body, (max_reg, l_reg))

    def out_body(t, _):
        idx = pl.dslice(t * BLOCK_N, BLOCK_N)
        col_ids = t * BLOCK_N + jnp.arange(BLOCK_N)
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


@jax.jit
def softmax(logits):
    n_row_blocks = pl.cdiv(logits.shape[0], BLOCK_M)
    n_col_blocks = pl.cdiv(logits.shape[1], BLOCK_N)
    return pl.pallas_call(
        partial(softmax_kernel, n_col_blocks=n_col_blocks, n_rows=logits.shape[0], n_cols=logits.shape[1]),
        out_shape=jax.ShapeDtypeStruct(logits.shape, jnp.float32),
        grid=(n_row_blocks,),
        in_specs=[pl.BlockSpec((BLOCK_M, logits.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((BLOCK_M, logits.shape[1]), lambda i: (i, 0)),
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        ),
    )(logits)


D = 4096 * 2
key = jax.random.key(0)
logits = jax.random.normal(shape=(D, D), key=key)

out_jax = jax.nn.softmax(logits)
out_manual = manual_softmax(logits)
out_online = online_softmax(logits)
out_pl = softmax(logits)

# Check correctness
assert(jnp.allclose(jnp.squeeze(out_jax),out_online))
assert(jnp.allclose(jnp.squeeze(out_jax),out_manual))
assert(jnp.allclose(jnp.squeeze(out_jax),out_pl))

# JIT compile
softmax_jit = jax.jit(jax.nn.softmax)
softmax_manual_jit = jax.jit(manual_softmax)

# Warmup (compile + first run)
_ = softmax_jit(logits).block_until_ready()
_ = softmax_manual_jit(logits).block_until_ready()
_ = softmax(logits).block_until_ready()

def bench(fn, *args, iters=100):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()   # very important
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times)  / len(times)  # median

t_jax       = bench(softmax_jit, logits)
t_manual    = bench(softmax_manual_jit, logits)
t_pallas    = bench(softmax, logits)

print(f"Jax Softmax: {t_jax*1e3:.2f} ms")
print(f"Manual Softmax: {t_manual*1e3:.2f} ms")
print(f"Pallas Softmax: {t_pallas*1e3:.2f} ms")
print(f"Speedup (jax / pallas): {t_jax / t_pallas:.2f}x")
