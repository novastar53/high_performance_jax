"""Pallas Flash Attention implementation.

Input shapes: Q, K, V are (B, H, T, D) where:
    B = batch size
    H = number of heads
    T = sequence length
    D = head dimension

Standard attention computes:
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

This keeps memory usage at O(N) instead of O(N²) since you never store the full attention matrix.
"""
from functools import partial
import argparse
import sys

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _parse_global_args():
    parser = argparse.ArgumentParser(description="Flash attention kernel config", add_help=False)
    parser.add_argument("--block-r", type=int, default=64, help="Block size R (Q rows)")
    parser.add_argument("--block-c", type=int, default=64, help="Block size C (KV cols)")
    parser.add_argument("--num-warps", type=int, default=4, help="Number of warps")
    parser.add_argument("--num-stages", type=int, default=3, help="Number of pipeline stages")
    parser.add_argument("--window-size", type=str, default="-1,-1", help="Sliding window as 'left,right' tuple (e.g., '-1,0' for causal, '64,64' for bidirectional)")
    parser.add_argument("--interpret-mode", action="store_true", help="Use Pallas interpreter mode")
    args, _ = parser.parse_known_args()
    return args

_args = _parse_global_args()

INTERPRET_MODE = _args.interpret_mode
BLOCK_R = _args.block_r
BLOCK_C = _args.block_c
NUM_WARPS = _args.num_warps
NUM_STAGES = _args.num_stages
WINDOW_SIZE = _args.window_size
parts = _args.window_size.split(',')
WINDOW_SIZE = (int(parts[0].strip()), int(parts[1].strip()))
IS_CAUSAL = (WINDOW_SIZE == (-1,0))
IS_BIDIRECTIONAL = (WINDOW_SIZE == (-1,-1))

DTYPE = jnp.bfloat16
JAX_SDPA_IMPL = "cudnn"

if INTERPRET_MODE:
    JAX_SDPA_IMPL = "xla"



# Reference implementations

@jax.jit
def mha_reference(q, k, v):
    """Reference multi-head attention (materializes N×N matrix).

    Computes: softmax(Q @ K^T / sqrt(d)) @ V
    Input shape: (B, H, T, D) - batch, heads, sequence, head_dim
    """
    d = q.shape[-1]
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(d)
    T = q.shape[2]
    left, right = WINDOW_SIZE
    left_limit = None if left == -1 else left
    right_limit = None if right == -1 else right
    if left_limit is not None or right_limit is not None:
        q_idx = jnp.arange(T)
        k_idx = jnp.arange(T)
        mask = jnp.ones((T, T), dtype=jnp.bool_)
        if left_limit is not None:
            mask = mask & (q_idx[:, None] - left_limit <= k_idx[None, :])
        if right_limit is not None:
            mask = mask & (q_idx[:, None] + right_limit >= k_idx[None, :])
        logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', probs, v)


def _compute_logsumexp(q, k):
    """Compute logsumexp of attention logits (for testing flash attention)."""
    d = q.shape[-1]
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(d)
    T = q.shape[2]
    left, right = WINDOW_SIZE
    left_limit = None if left == -1 else left
    right_limit = None if right == -1 else right
    if left_limit is not None or right_limit is not None:
        q_idx = jnp.arange(T)
        k_idx = jnp.arange(T)
        mask = jnp.ones((T, T), dtype=jnp.bool_)
        if left_limit is not None:
            mask = mask & (q_idx[:, None] - left_limit <= k_idx[None, :])
        if right_limit is not None:
            mask = mask & (q_idx[:, None] + right_limit >= k_idx[None, :])
        logits = jnp.where(mask, logits, -jnp.inf)
    logits_max = jnp.max(logits, axis=-1)
    logits_shift = logits - logits_max[..., None]
    return logits_max + jnp.log(jnp.sum(jnp.exp(logits_shift), axis=-1))


@jax.jit
def cudnn_attention(q, k, v):
    """JAX cuDNN flash attention wrapper.

    Wraps jax.nn.dot_product_attention with layout transposition.
    Input/output shape: (B, H, T, D) - batch, heads, sequence, head_dim
    """
    # Transpose from (B, H, T, D) to (B, T, H, D) for jax.nn.dot_product_attention
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    if IS_CAUSAL:
        out = jax.nn.dot_product_attention(q_t, k_t, v_t, implementation=JAX_SDPA_IMPL, is_causal=True)
    elif IS_BIDIRECTIONAL:
        out = jax.nn.dot_product_attention(q_t, k_t, v_t, implementation=JAX_SDPA_IMPL, is_causal=False)
    else:
        print(WINDOW_SIZE)
        out = jax.nn.dot_product_attention(q_t, k_t, v_t, implementation=JAX_SDPA_IMPL, local_window_size=WINDOW_SIZE) #, is_causal=Fal)
    return jnp.transpose(out, (0, 2, 1, 3))  # Back to (B, H, T, D)


def flash_attn_jax_wrapper(q, k, v):
    """Reference nshepperd/flash_attn_jax implementation (C++ CUDA kernels).

    Wraps flash_attn_jax.flash_mha with layout conversion.
    Requires: pip install flash-attn-jax (already in pyproject.toml [gpu])

    Input shape: (B, H, T, D) - batch, heads, sequence, head_dim
    Output shape: (B, H, T, D)
    """
    from flash_attn_jax import flash_mha

    # nshepperd/flash_attn_jax expects (n, l, h, d) format
    # Our format is (B, H, T, D), so transpose axes 1 and 2
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    if IS_CAUSAL:
        out = flash_mha(q_t, k_t, v_t, is_causal=True)
    elif IS_BIDIRECTIONAL:
        out = flash_mha(q_t, k_t, v_t, is_causal=False)
    else:
        out = flash_mha(q_t, k_t, v_t, window_size=WINDOW_SIZE)
    return jnp.transpose(out, (0, 2, 1, 3))


# Flash Attention Forward

def flash_attention_fwd_kernel(q_ref, k_ref, v_ref, o_ref, logsumexp_ref, *, qk_scale, num_k_blocks):
    """Flash attention forward kernel."""
    q_reg = plgpu.load(q_ref.at[0, :, :])
    o_reg = jnp.zeros(q_reg.shape, jnp.float32)
    max_reg = jnp.full((BLOCK_R,), -jnp.inf, dtype=jnp.float32)
    l_reg = jnp.zeros((BLOCK_R,), dtype=jnp.float32)

    blk_idx = pl.program_id(1)
    q_idx = BLOCK_R * blk_idx + jnp.arange(BLOCK_R)

    def body(t, args):
        max_reg, l_reg, o_reg = args
        idx = pl.dslice(t * BLOCK_C, BLOCK_C)
        k_blk = plgpu.load(k_ref.at[0, idx, :])
        v_blk = plgpu.load(v_ref.at[0, idx, :])

        def compute_block(_):
            s_blk = pl.dot(q_reg, k_blk, trans_b=True, precision='float32') * qk_scale
            if IS_CAUSAL:
                kv_idx = BLOCK_C * t + jnp.arange(BLOCK_C)
                causal_mask = kv_idx[None, :] > q_idx[:, None]
                s_blk_masked = jnp.where(causal_mask, -jnp.inf, s_blk)
            elif IS_BIDIRECTIONAL:
                s_blk_masked = s_blk
            else:
                kv_idx = BLOCK_C * t + jnp.arange(BLOCK_C)
                mask = jnp.full((BLOCK_R, BLOCK_C), fill_value=True, dtype=jnp.bool)
                mask = mask & (q_idx[:, None] - WINDOW_SIZE[0] <= kv_idx[None, :])
                mask = mask & (q_idx[:, None] + WINDOW_SIZE[1] >= kv_idx[None, :])
                s_blk_masked = jnp.where(mask, s_blk, -jnp.inf)

            max_blk = jnp.maximum(max_reg, jnp.max(s_blk_masked, axis=-1))
            p_blk = jnp.exp(s_blk_masked - max_blk[:, None])
            l_blk = jnp.sum(p_blk, axis=-1)
            o_blk = pl.dot(p_blk.astype(v_blk.dtype), v_blk)
            alpha = jnp.exp(max_reg - max_blk)
            return (max_blk,
                    l_reg * alpha + l_blk,
                    o_reg * alpha[:, None] + o_blk)

        def skip_block(_):
            return (max_reg, l_reg, o_reg)

        # Skip blocks entirely outside the attention window
        if IS_CAUSAL:
            return jax.lax.cond(t > blk_idx, skip_block, compute_block, None)
        elif IS_BIDIRECTIONAL:
            return compute_block(None)
        else:
            skip_right = t * BLOCK_C > blk_idx * BLOCK_R + (BLOCK_R - 1) + WINDOW_SIZE[1]
            skip_left = (t + 1) * BLOCK_C - 1 < blk_idx * BLOCK_R - WINDOW_SIZE[0]
            return jax.lax.cond(skip_right | skip_left, skip_block, compute_block, None)

    max_reg, l_reg, o_reg = jax.lax.fori_loop(0, num_k_blocks, body, (max_reg, l_reg, o_reg))
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
    qk_scale = 1.0 / math.sqrt(C)
    num_k_blocks = pl.cdiv(T, BLOCK_C)
    grid = (B_flat, pl.cdiv(T, BLOCK_R))

    out_flat, logsumexp = pl.pallas_call(
        partial(flash_attention_fwd_kernel, qk_scale=qk_scale, num_k_blocks=num_k_blocks),
        out_shape=[
            jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
            jax.ShapeDtypeStruct((B*H, T), q_flat.dtype)
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0))
        ],
        out_specs=[
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, BLOCK_R), lambda b, t: (b, t))
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
#
# Three separate kernels to avoid atomic operations:
# 1. Preprocess: D = rowsum(O ⊙ dO)
# 2. dK/dV: outer loop over KV blocks, inner loop over Q blocks
# 3. dQ: outer loop over Q blocks, inner loop over KV blocks


def flash_attention_bwd_preprocess_kernel(o_ref, do_ref, d_ref):
    """Compute D = rowsum(O ⊙ dO) for backward pass.

    D is used in the softmax backward: dS = P ⊙ (dP - D)
    where D_i = sum_j(dO_ij * O_ij)
    """
    o_reg = plgpu.load(o_ref)
    do_reg = plgpu.load(do_ref)
    d_reg = jnp.sum((o_reg * do_reg).astype(jnp.float32), axis=-1)
    plgpu.store(d_ref, d_reg.astype(d_ref.dtype))


def flash_attention_bwd_preprocess(o_flat, do_flat):
    """Preprocess for backward: compute D = rowsum(O ⊙ dO)."""
    B_flat, T, C = o_flat.shape
    grid = (B_flat, pl.cdiv(T, BLOCK_R))

    d_flat = pl.pallas_call(
        partial(flash_attention_bwd_preprocess_kernel),
        out_shape=jax.ShapeDtypeStruct((B_flat, T), o_flat.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)),  # o
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)),  # do
        ],
        out_specs=pl.BlockSpec((1, BLOCK_R), lambda b, t: (b, t)),  # d
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
    )(o_flat, do_flat)
    return d_flat


def flash_attention_bwd_dkv_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dk_ref, dv_ref,
    *, qk_scale, softmax_scale, num_q_blocks,
):
    """Compute dK and dV gradients.

    Grid: (batch*heads, num_kv_blocks)
    - Outer parallel dimension: KV blocks
    - Inner loop: iterate over all Q blocks

    For each KV block, we accumulate:
        dV += P^T @ dO
        dK += dS^T @ Q  (where dS = P ⊙ (dP - D))
    """
    k_reg = plgpu.load(k_ref.at[0, :, :])
    v_reg = plgpu.load(v_ref.at[0, :, :])
    kv_blk_idx = pl.program_id(1)
    kv_idx = BLOCK_C * kv_blk_idx + jnp.arange(BLOCK_C)

    dk_acc = jnp.zeros(dk_ref.shape, dtype=jnp.float32)
    dv_acc = jnp.zeros(dv_ref.shape, dtype=jnp.float32)

    def body(t, carry):
        dk_acc, dv_acc = carry
        idx = pl.dslice(t * BLOCK_R, BLOCK_R)
        q_blk = plgpu.load(q_ref.at[0, idx, :])
        do_blk = plgpu.load(do_ref.at[0, idx, :])
        logsumexp_blk = plgpu.load(logsumexp_ref.at[0, idx])
        d_blk = plgpu.load(d_ref.at[0, idx])

        def compute_block(_):
            s_blk = pl.dot(q_blk, k_reg, trans_b=True, precision='float32') * qk_scale
            if IS_CAUSAL:
                q_idx = BLOCK_R * t + jnp.arange(BLOCK_R)
                causal_mask = kv_idx[None, :] > q_idx[:, None]
                s_blk_masked = jnp.where(causal_mask, -jnp.inf, s_blk)
            else:
                s_blk_masked = s_blk
            p_blk = jnp.exp(s_blk_masked - logsumexp_blk[..., None])
            dp_blk = pl.dot(do_blk, v_reg, trans_b=True)
            ds_blk = p_blk * (dp_blk - d_blk[..., None]) * softmax_scale
            dv_new = dv_acc + pl.dot(p_blk.astype(do_blk.dtype), do_blk, trans_a=True)
            dk_new = dk_acc + pl.dot(ds_blk.astype(q_blk.dtype), q_blk, trans_a=True)
            return (dk_new, dv_new)

        def skip_block(_):
            return (dk_acc, dv_acc)

        # For causal: skip Q blocks that can't see this KV block (t < kv_blk_idx)
        if IS_CAUSAL:
            return jax.lax.cond(t < kv_blk_idx, skip_block, compute_block, None)
        else:
            return compute_block(None)

    dk_acc, dv_acc = jax.lax.fori_loop(0, num_q_blocks, body, (dk_acc, dv_acc))
    plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
    plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))


def flash_attention_bwd_dkv(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, qk_scale, softmax_scale):
    """Compute dK and dV using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_q_blocks = pl.cdiv(T, BLOCK_R)
    grid = (B_flat, pl.cdiv(T, BLOCK_C))  # Outer loop over KV blocks

    dk_flat, dv_flat = pl.pallas_call(
        partial(flash_attention_bwd_dkv_kernel, qk_scale=qk_scale, softmax_scale=softmax_scale, num_q_blocks=num_q_blocks),
        out_shape=[
            jax.ShapeDtypeStruct(k_flat.shape, k_flat.dtype),  # dK
            jax.ShapeDtypeStruct(v_flat.shape, v_flat.dtype),  # dV
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # q (full sequence for inner loop)
            pl.BlockSpec((1, BLOCK_C, C), lambda b, t: (b, t, 0)), # k (blocked)
            pl.BlockSpec((1, BLOCK_C, C), lambda b, t: (b, t, 0)), # v (blocked)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # do (full sequence for inner loop)
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),             # logsumexp (full sequence)
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),             # d (full sequence)
        ],
        out_specs=[
            pl.BlockSpec((1, BLOCK_C, C), lambda b, t: (b, t, 0)), # dk (blocked)
            pl.BlockSpec((1, BLOCK_C, C), lambda b, t: (b, t, 0)), # dv (blocked)
        ],
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
    )(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat)
    return dk_flat, dv_flat


def flash_attention_bwd_dq_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dq_ref,
    *, qk_scale, softmax_scale, num_kv_blocks
):
    """Compute dQ gradient.

    Grid: (batch*heads, num_q_blocks)
    - Outer parallel dimension: Q blocks
    - Inner loop: iterate over all KV blocks

    For each Q block, we accumulate:
        dQ += dS @ K  (where dS = P ⊙ (dP - D))
    """
    q_reg = plgpu.load(q_ref.at[0, :, :])
    do_reg = plgpu.load(do_ref.at[0, :, :])
    logsumexp_reg = plgpu.load(logsumexp_ref.at[0, :])
    d_reg = plgpu.load(d_ref.at[0, :])
    q_blk_idx = pl.program_id(1)
    q_idx = BLOCK_R * q_blk_idx + jnp.arange(BLOCK_R)
    dq_acc = jnp.zeros(dq_ref.shape, dtype=jnp.float32)  # Use float32 for numerical stability

    def body(t, carry):
        dq_acc = carry
        idx = pl.dslice(t * BLOCK_C, BLOCK_C)
        k_blk = plgpu.load(k_ref.at[0, idx, :])
        v_blk = plgpu.load(v_ref.at[0, idx, :])

        def compute_block(_):
            s_blk = pl.dot(q_reg, k_blk, trans_b=True, precision='float32') * qk_scale
            if IS_CAUSAL:
                kv_idx = BLOCK_C * t + jnp.arange(BLOCK_C)
                causal_mask = kv_idx[None, :] > q_idx[:, None]
                s_blk_masked = jnp.where(causal_mask, -jnp.inf, s_blk)
            else:
                s_blk_masked = s_blk
            p_blk = jnp.exp(s_blk_masked - logsumexp_reg[..., None])
            dp_blk = pl.dot(do_reg, v_blk, trans_b=True)
            ds_blk = p_blk * (dp_blk - d_reg[..., None]) * softmax_scale
            dq_new = dq_acc + pl.dot(ds_blk.astype(k_blk.dtype), k_blk)
            return dq_new

        def skip_block(_):
            return dq_acc

        # For causal: skip KV blocks this Q can't see (t > q_blk_idx)
        if IS_CAUSAL:
            return jax.lax.cond(t > q_blk_idx, skip_block, compute_block, None)
        else:
            return compute_block(None)

    dq_acc = jax.lax.fori_loop(0, num_kv_blocks, body, dq_acc)
    plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))


def flash_attention_bwd_dq(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, qk_scale, softmax_scale):
    """Compute dQ using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_kv_blocks = pl.cdiv(T, BLOCK_C)
    grid = (B_flat, pl.cdiv(T, BLOCK_R))  # Outer loop over Q blocks

    dq_flat = pl.pallas_call(
        partial(flash_attention_bwd_dq_kernel, qk_scale=qk_scale, softmax_scale=softmax_scale, num_kv_blocks=num_kv_blocks),
        out_shape=jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)), # q (blocked)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # k (full sequence for inner loop)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # v (full sequence for inner loop)
            pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)), # do (blocked)
            pl.BlockSpec((1, BLOCK_R), lambda b, t: (b, t)),       # logsumexp (blocked)
            pl.BlockSpec((1, BLOCK_R), lambda b, t: (b, t)),       # d (blocked)
        ],
        out_specs=pl.BlockSpec((1, BLOCK_R, C), lambda b, t: (b, t, 0)),  # dq (blocked)
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
    )(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat)
    return dq_flat


@jax.jit
def flash_attention_bwd(q, k, v, o, logsumexp, do):
    """Flash attention backward pass using 3 separate kernels."""
    B, H, T, C = q.shape
    qk_scale = 1.0 / math.sqrt(C)
    softmax_scale = 1.0 / math.sqrt(C)

    # Flatten batch and head dimensions
    q_flat = q.reshape(-1, T, C)
    k_flat = k.reshape(-1, T, C)
    v_flat = v.reshape(-1, T, C)
    o_flat = o.reshape(-1, T, C)
    do_flat = do.reshape(-1, T, C)
    logsumexp_flat = logsumexp.reshape(-1, T)

    # Kernel 1: Preprocess - compute D = rowsum(O ⊙ dO)
    d_flat = flash_attention_bwd_preprocess(o_flat, do_flat)

    # Kernel 2: Compute dK, dV (outer loop over KV blocks)
    dk_flat, dv_flat = flash_attention_bwd_dkv(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, qk_scale, softmax_scale
    )

    # Kernel 3: Compute dQ (outer loop over Q blocks)
    dq_flat = flash_attention_bwd_dq(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, qk_scale, softmax_scale
    )

    return (
        dq_flat.reshape(q.shape),
        dk_flat.reshape(k.shape),
        dv_flat.reshape(v.shape),
    )


# Custom VJP wrapper

@jax.custom_vjp
def flash_attention(q, k, v):
    """Flash attention with custom backward pass."""
    o, _ = flash_attention_fwd(q, k, v)
    return o


def flash_attention_fwd_rule(q, k, v):
    """Forward rule for custom_vjp.

    Returns the output and residuals needed for backward pass.
    """
    o, logsumexp = flash_attention_fwd(q, k, v)
    return o, (q, k, v, o, logsumexp)


def flash_attention_bwd_rule(res, do):
    """Backward rule for custom_vjp.

    Takes residuals from forward and upstream gradient dO,
    returns gradients (dQ, dK, dV).
    """
    q, k, v, o, logsumexp = res
    dq, dk, dv = flash_attention_bwd(q, k, v, o, logsumexp, do)
    return dq, dk, dv


flash_attention.defvjp(flash_attention_fwd_rule, flash_attention_bwd_rule)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Flash attention kernel tests")
    parser.add_argument("--block-r", type=int, default=BLOCK_R, help="Block size R (Q rows)")
    parser.add_argument("--block-c", type=int, default=BLOCK_C, help="Block size C (KV cols)")
    parser.add_argument("--num-warps", type=int, default=NUM_WARPS, help="Number of warps")
    parser.add_argument("--num-stages", type=int, default=NUM_STAGES, help="Number of pipeline stages")
    parser.add_argument("--window-size", type=str, default=None, help="Sliding window as 'left,right' tuple (e.g., '-1,0' for causal, '64,64' for bidirectional)")
    parser.add_argument("--interpret-mode", action="store_true", help="Use Pallas interpreter mode")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (B)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads (H)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (T)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (D)")
    args = parser.parse_args()

    B, H, T, D = args.batch_size, args.num_heads, args.seq_len, args.head_dim
    print(f"Config: BLOCK_R={args.block_r}, BLOCK_C={args.block_c}, NUM_WARPS={args.num_warps}, NUM_STAGES={args.num_stages}, WINDOW_SIZE={WINDOW_SIZE}, INTERPRET_MODE={args.interpret_mode}")
    print(f"Testing with shapes: B={B}, H={H}, T={T}, D={D}")

    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, H, T, D), dtype=DTYPE)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=DTYPE)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=DTYPE)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=DTYPE)

    # Forward check
    o_ref = mha_reference(q, k, v)
    logsumexp_ref = _compute_logsumexp(q, k)
    o_flash, logsumexp_flash = flash_attention_fwd(q, k, v)
    o_cudnn_ref = cudnn_attention(q, k, v)

    # Test nshepperd/flash_attn_jax reference (skip in INTERPRET_MODE)
    if not INTERPRET_MODE:
        o_flash_attn_jax = flash_attn_jax_wrapper(q, k, v)
        assert jnp.allclose(o_flash_attn_jax, o_ref, atol=1e-2, rtol=1e-2), \
            f"flash_attn_jax max diff: {jnp.max(jnp.abs(o_flash_attn_jax - o_ref))}"
        print("flash_attn_jax reference check passed!")

    assert jnp.allclose(o_cudnn_ref, o_ref, atol=1e-1, rtol=1e-2),f"o max diff: {jnp.max(jnp.abs(o_cudnn_ref - o_ref))}"
    assert jnp.allclose(o_flash, o_cudnn_ref, atol=1e-1, rtol=1e-2), f"o max diff: {jnp.max(jnp.abs(o_flash - o_cudnn_ref))}"
    assert jnp.allclose(o_flash, o_ref, atol=1e-1, rtol=1e-2), f"o max diff: {jnp.max(jnp.abs(o_flash - o_ref))}"
    assert jnp.allclose(logsumexp_flash, logsumexp_ref, atol=1e-2, rtol=1e-2), f"logsumexp max diff: {jnp.max(jnp.abs(logsumexp_flash - logsumexp_ref))}"
    print("Forward pass check passed!")

    # Backward check (reference)
    def loss_ref(q, k, v):
        return jnp.sum(mha_reference(q, k, v) * do)

    dq_ref, dk_ref, dv_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    # Test preprocess kernel: D = rowsum(O * dO)
    # Use reference output for ground truth to avoid compounding bfloat16 errors
    d_ref = jnp.sum((o_ref * do).astype(jnp.float32), axis=-1).astype(DTYPE)  # (B, H, T)
    o_flat = o_ref.reshape(-1, T, D)
    do_flat = do.reshape(-1, T, D)
    d_flash = flash_attention_bwd_preprocess(o_flat, do_flat).reshape(B, H, T)

    # For bfloat16, we need looser tolerance due to precision limits
    # 0.0625 = 2^-4 is the expected precision error for accumulated sums
    assert jnp.allclose(d_flash, d_ref, atol=1e-1, rtol=1e-2), f"D max diff: {jnp.max(jnp.abs(d_flash - d_ref))}"
    print("Preprocess kernel (D) check passed!")

    dq_flash, dk_flash, dv_flash = flash_attention_bwd(q, k, v, o_flash, logsumexp_flash, do)
    assert jnp.allclose(dq_flash, dq_ref, atol=1e-1, rtol=1e-2), f"dQ max diff: {jnp.max(jnp.abs(dq_flash - dq_ref))}"
    assert jnp.allclose(dk_flash, dk_ref, atol=1e-1, rtol=1e-2), f"dK max diff: {jnp.max(jnp.abs(dk_flash - dk_ref))}"
    assert jnp.allclose(dv_flash, dv_ref, atol=1e-1, rtol=1e-2), f"dV max diff: {jnp.max(jnp.abs(dv_flash - dv_ref))}"
    print("Backward pass check passed!")

    if not INTERPRET_MODE:
        # Timing comparison with JAX built-in dot_product_attention
        print("\n" + "="*60)
        print("Timing Comparison")
        print("="*60)

        q_bench = jax.random.normal(keys[0], (B, H, T, D), dtype=DTYPE)
        k_bench = jax.random.normal(keys[1], (B, H, T, D), dtype=DTYPE)
        v_bench = jax.random.normal(keys[2], (B, H, T, D), dtype=DTYPE)
        do_bench = jax.random.normal(keys[3], (B, H, T, D), dtype=DTYPE)

        print(f"Benchmark shape: B={B}, H={H}, T={T}, D={D}")

        def _bench(fn, warmup=3, iters=20):
            """Benchmark a function."""
            # Warmup
            for _ in range(warmup):
                out = fn()
                jax.block_until_ready(out)
            # Bench
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                out = fn()
                jax.block_until_ready(out)
                times.append(time.perf_counter() - t0)
            return np.median(times) * 1000

        # Create jitted forward functions
        jax_fwd = jax.jit(cudnn_attention)
        flash_fwd = jax.jit(flash_attention)
        ref_fwd = jax.jit(mha_reference)
        flash_attn_jax_fwd = jax.jit(flash_attn_jax_wrapper) if not INTERPRET_MODE else None

        # Get vjp functions for backward pass
        _, jax_vjp = jax.vjp(cudnn_attention, q_bench, k_bench, v_bench)
        _, flash_vjp = jax.vjp(flash_attention, q_bench, k_bench, v_bench)
        #_, ref_vjp = jax.vjp(mha_reference, q_bench, k_bench, v_bench)
        if not INTERPRET_MODE:
            _, flash_attn_jax_vjp = jax.vjp(flash_attn_jax_wrapper, q_bench, k_bench, v_bench)

        print("\nForward pass:")
        t_jax = _bench(lambda: jax_fwd(q_bench, k_bench, v_bench))
        print(f"  JAX dot_product_attention: {t_jax:.3f} ms")
        t_ours = _bench(lambda: flash_fwd(q_bench, k_bench, v_bench))
        print(f"  Our flash_attention:       {t_ours:.3f} ms")
        #t_ref = _bench(lambda: ref_fwd(q_bench, k_bench, v_bench))
        #print(f"  Reference (materialized):  {t_ref:.3f} ms")
        if not INTERPRET_MODE:
            t_flash_attn_jax = _bench(lambda: flash_attn_jax_fwd(q_bench, k_bench, v_bench))
            print(f"  flash_attn_jax (C++ CUDA):  {t_flash_attn_jax:.3f} ms")

        print("\nBackward pass only:")
        t_jax_bwd = _bench(lambda: jax_vjp(do_bench))
        print(f"  JAX dot_product_attention: {t_jax_bwd:.3f} ms")
        t_ours_bwd = _bench(lambda: flash_vjp(do_bench))
        print(f"  Our flash_attention:       {t_ours_bwd:.3f} ms")
        #t_ref_bwd = _bench(lambda: ref_vjp(do_bench))
        #print(f"  Reference (materialized):  {t_ref_bwd:.3f} ms")
        if not INTERPRET_MODE:
            t_flash_attn_jax_bwd = _bench(lambda: flash_attn_jax_vjp(do_bench))
            print(f"  flash_attn_jax (C++ CUDA):  {t_flash_attn_jax_bwd:.3f} ms")

        print("\nTotal (Forward + Backward):")
        print(f"  JAX dot_product_attention: {t_jax + t_jax_bwd:.3f} ms")
        print(f"  Our flash_attention:       {t_ours + t_ours_bwd:.3f} ms")
        #print(f"  Reference (materialized):  {t_ref + t_ref_bwd:.3f} ms")
        if not INTERPRET_MODE:
            print(f"  flash_attn_jax (C++ CUDA):  {t_flash_attn_jax + t_flash_attn_jax_bwd:.3f} ms")
