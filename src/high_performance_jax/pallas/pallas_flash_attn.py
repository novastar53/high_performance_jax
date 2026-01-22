"""Pallas Flash Attention implementation.

Input shapes: Q, K, V are (B, H, T, D) where:
    B = batch size
    H = number of heads
    T = sequence length
    D = head dimension

Standard attention computes:
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

This normally requires materializing the full N×N attention matrix. Flash attention avoids this by:

1. Computing Q·Kᵀ tiles on-the-fly: Instead of loading pre-computed attention scores, compute S_tile = Q_block @ K_block.T / sqrt(d) inside your column loop (where you currently load x_tile).
2. Accumulating the output with V using online correction: Just as you correct l_reg when the max changes (l_reg * exp(max_old - max_new)), you need to apply the same correction to a running output accumulator:
O_acc = O_acc * exp(m_old - m_new) + P_tile @ V_block
2. where P_tile = exp(S_tile - m_new).
3. Final normalization: After all K/V blocks are processed, divide by l_reg to get the final output.

The kernel structure would be similar to your softmax kernel, but:
- Input refs: Q, K, V (instead of just x)
- The inner loop computes the matmul Q_block @ K_block.T rather than loading from a pre-computed matrix
- You maintain an output accumulator O_acc that gets rescaled alongside l_reg
- Output is O_acc / l_reg

This keeps memory usage at O(N) instead of O(N²) since you never store the full attention matrix.
"""
from functools import partial

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


INTERPRET_MODE = True  # Set to False on GPU

BLOCK_T = 64
NUM_WARPS = 4
NUM_STAGES = 2


# Reference implementation

@jax.jit
def mha_reference(q, k, v):
    """Reference multi-head attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    probs = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
    logits_max = jnp.max(logits, axis=-1)
    logits_shift = logits - logits_max[..., None]
    logsumexp =  logits_max + jnp.log(jnp.sum(jnp.exp(logits_shift), axis=-1))
    return o, logsumexp


# Flash Attention Forward

def flash_attention_fwd_kernel(q_ref, k_ref, v_ref, o_ref, logsumexp_ref, *, scale, num_k_blocks):
    """Flash attention forward kernel."""
    q_reg = plgpu.load(q_ref.at[0, :, :]).astype(jnp.float32)
    o_reg = jnp.zeros(q_reg.shape, jnp.float32)
    max_reg = jnp.full((BLOCK_T,), -jnp.inf, dtype=jnp.float32)
    l_reg = jnp.zeros((BLOCK_T,), dtype=jnp.float32)
    logsumexp_reg = jnp.zeros((BLOCK_T,), dtype=jnp.float32)

    def num_body(t, args):
        max_reg, l_reg, o_reg = args
        idx = pl.dslice(t * BLOCK_T, BLOCK_T)
        k_blk = plgpu.load(k_ref.at[0, idx, :]).astype(jnp.float32)
        v_blk = plgpu.load(v_ref.at[0, idx, :]).astype(jnp.float32)
        s_blk = pl.dot(q_reg, k_blk, trans_b=True) / scale
        max_blk = jnp.maximum(max_reg, jnp.max(s_blk, axis=-1))
        s_blk = jnp.exp(s_blk - max_blk[:, None])
        l_blk = jnp.sum(s_blk, axis=-1)
        o_blk = pl.dot(s_blk, v_blk)
        return (max_blk, 
                l_reg * jnp.exp(max_reg - max_blk) + l_blk, 
                o_reg * jnp.exp(max_reg - max_blk)[:, None] + o_blk)

    max_reg, l_reg, o_reg = jax.lax.fori_loop(0, num_k_blocks, num_body, (max_reg, l_reg, o_reg))
    logsumexp_reg = max_reg + jnp.log(l_reg)
    o_reg = o_reg / l_reg[:, None]
    plgpu.store(o_ref.at[0, :, :], o_reg.astype(o_ref.dtype))
    plgpu.store(logsumexp_ref.at[0, :], logsumexp_reg.astype(logsumexp_ref.dtype))



@jax.jit
def flash_attention_fwd(q, k, v):
    """Flash attention forward pass."""
    B, H, T, C = q.shape
    B_flat = B*H
    q_flat = q.reshape(-1, T, C)
    k_flat = k.reshape(-1, T, C)
    v_flat = v.reshape(-1, T, C)
    scale = math.sqrt(C)
    num_k_blocks = pl.cdiv(T, BLOCK_T)

    grid = (B_flat, pl.cdiv(T, BLOCK_T))

    out_flat, logsumexp = pl.pallas_call(
        partial(flash_attention_fwd_kernel, scale=scale, num_k_blocks=num_k_blocks),
        out_shape=[
            jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
            jax.ShapeDtypeStruct((B*H, T), q_flat.dtype)
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_T, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0))
        ],
        out_specs=[
            pl.BlockSpec((1, BLOCK_T, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, BLOCK_T), lambda b, t: (b, t))
        ],
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
    )(q_flat, k_flat, v_flat)
    out = out_flat.reshape(q.shape)
    logsumexp = logsumexp.reshape(B, H, T)
    return out, logsumexp



# Flash Attention Backward

def flash_attention_bwd_kernel(q_ref, k_ref, v_ref, o_ref, do_ref, logsumexp_pref,
                                dq_ref, dk_ref, dv_ref, *, num_kv_blocks, scale):
    """Flash attention backward kernel."""
    pass


@jax.jit
def flash_attention_bwd(q, k, v, o, l, m, do):
    """Flash attention backward pass."""
    pass


# Custom VJP wrapper

@jax.custom_vjp
def flash_attention(q, k, v):
    """Flash attention with custom backward pass."""
    o, logsumexp = flash_attention_fwd(q, k, v)
    return o, logsumexp


def flash_attention_fwd_rule(q, k, v):
    """Forward rule for custom_vjp."""
    pass


def flash_attention_bwd_rule(res, do):
    """Backward rule for custom_vjp."""
    pass


flash_attention.defvjp(flash_attention_fwd_rule, flash_attention_bwd_rule)


if __name__ == "__main__":
    import time

    def _bench(fn, *args, iters=10):
        for _ in range(3):  # warmup
            fn(*args).block_until_ready()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args).block_until_ready()
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    B, H, T, D = 2, 4, 256, 64
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, H, T, D), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=jnp.float32)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=jnp.float32)

    # Forward check
    o_ref, logsumexp_ref = mha_reference(q, k, v)
    print(f"Reference output shape: {o_ref.shape}")

    # Backward check (reference)
    def loss_ref(q, k, v):
        O, _ = mha_reference(q, k, v)
        return jnp.sum(O * do)

    dq_ref, dk_ref, dv_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    print(f"Reference gradient shapes: dq={dq_ref.shape}, dk={dk_ref.shape}, dv={dv_ref.shape}")

    o_flash, logsumexp_flash = flash_attention(q, k, v)
    assert jnp.allclose(o_flash, o_ref, atol=1e-2, rtol=1e-2)
    print(logsumexp_ref)
    print("*****")
    print(logsumexp_flash)
    assert jnp.allclose(logsumexp_flash, logsumexp_ref, atol=1e-2, rtol=1e-2)
    print("Forward pass check passed!")
    #
    # def loss_flash(q, k, v):
    #     return jnp.sum(flash_attention(q, k, v) * do)
    #
    # dq_flash, dk_flash, dv_flash = jax.grad(loss_flash, argnums=(0, 1, 2))(q, k, v)
    # assert jnp.allclose(dq_flash, dq_ref, atol=1e-2, rtol=1e-2)
    # assert jnp.allclose(dk_flash, dk_ref, atol=1e-2, rtol=1e-2)
    # assert jnp.allclose(dv_flash, dv_ref, atol=1e-2, rtol=1e-2)
    # print("Backward pass check passed!")
