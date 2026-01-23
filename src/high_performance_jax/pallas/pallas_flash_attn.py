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

BLOCK_R = 64
BLOCK_C = 64
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
    max_reg = jnp.full((BLOCK_R,), -jnp.inf, dtype=jnp.float32)
    l_reg = jnp.zeros((BLOCK_R,), dtype=jnp.float32)
    logsumexp_reg = jnp.zeros((BLOCK_R,), dtype=jnp.float32)

    def num_body(t, args):
        max_reg, l_reg, o_reg = args
        idx = pl.dslice(t * BLOCK_C, BLOCK_C)
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
    num_k_blocks = pl.cdiv(T, BLOCK_C)
    grid = (B_flat, pl.cdiv(T, BLOCK_R))

    out_flat, logsumexp = pl.pallas_call(
        partial(flash_attention_fwd_kernel, scale=scale, num_k_blocks=num_k_blocks),
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
    o_reg = plgpu.load(o_ref).astype(jnp.float32)
    do_reg = plgpu.load(do_ref).astype(jnp.float32)
    d_reg = jnp.sum(o_reg * do_reg, axis=-1)
    plgpu.store(d_ref, d_reg.astype(d_ref.dtype))


def flash_attention_bwd_preprocess(o_flat, do_flat):
    """Preprocess for backward: compute D = rowsum(O ⊙ dO)."""
    B_flat, T, C = o_flat.shape
    grid = (B_flat, pl.cdiv(T, BLOCK_R))

    d_flat = pl.pallas_call(
        flash_attention_bwd_preprocess_kernel,
        out_shape=jax.ShapeDtypeStruct((B_flat, T), jnp.float32),
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
    *, scale, num_q_blocks
):
    """Compute dK and dV gradients.

    Grid: (batch*heads, num_kv_blocks)
    - Outer parallel dimension: KV blocks
    - Inner loop: iterate over all Q blocks

    For each KV block, we accumulate:
        dV += P^T @ dO
        dK += dS^T @ Q  (where dS = P ⊙ (dP - D))
    """
    # Load K, V block for this program
    k_reg = plgpu.load(k_ref.at[0, :, :]).astype(jnp.float32)
    v_reg = plgpu.load(v_ref.at[0, :, :]).astype(jnp.float32)

    # Initialize dK_acc, dV_acc to zeros (use float32 for numerical stability)
    dk_acc = jnp.zeros(dk_ref.shape, dtype=jnp.float32)
    dv_acc = jnp.zeros(dv_ref.shape, dtype=jnp.float32)
    # Loop over all Q blocks:
    def body(t, carry):
        dk_acc, dv_acc = carry
        idx = pl.dslice(t * BLOCK_R, BLOCK_R)
    #   Load Q, dO, logsumexp, D for current Q block
        q_blk = plgpu.load(q_ref.at[0, idx, :]).astype(jnp.float32)
        do_blk = plgpu.load(do_ref.at[0, idx, :]).astype(jnp.float32)
        logsumexp_blk = plgpu.load(logsumexp_ref.at[0, idx]).astype(jnp.float32)
        d_blk = plgpu.load(d_ref.at[0, idx]).astype(jnp.float32)
    #   Recompute P = softmax(Q @ K^T / scale) using logsumexp
        s_blk = pl.dot(q_blk, k_reg, trans_b=True) / scale
        p_blk = jnp.exp(s_blk - logsumexp_blk[..., None])
    #   Compute dP = dO @ V^T
        dp_blk = pl.dot(do_blk, v_reg, trans_b=True)
    #   Compute dS = P * (dP - D)
        ds_blk = p_blk * (dp_blk - d_blk[..., None]) / scale
    #   Accumulate: dV += P^T @ dO, dK += dS^T @ Q
        dv_acc += pl.dot(p_blk, do_blk, trans_a=True)
        dk_acc += pl.dot(ds_blk, q_blk, trans_a=True)
        return dk_acc, dv_acc
        
    dk_acc, dv_acc = jax.lax.fori_loop(0, num_q_blocks, body, (dk_acc, dv_acc))
    # Store dK, dV
    plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
    plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))


def flash_attention_bwd_dkv(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale):
    """Compute dK and dV using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_q_blocks = pl.cdiv(T, BLOCK_R)
    grid = (B_flat, pl.cdiv(T, BLOCK_C))  # Outer loop over KV blocks

    dk_flat, dv_flat = pl.pallas_call(
        partial(flash_attention_bwd_dkv_kernel, scale=scale, num_q_blocks=num_q_blocks),
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
    *, scale, num_kv_blocks
):
    """Compute dQ gradient.

    Grid: (batch*heads, num_q_blocks)
    - Outer parallel dimension: Q blocks
    - Inner loop: iterate over all KV blocks

    For each Q block, we accumulate:
        dQ += dS @ K  (where dS = P ⊙ (dP - D))
    """
    # Load Q, dO, logsumexp, D for this Q block
    q_reg = plgpu.load(q_ref.at[0, :, :]).astype(jnp.float32)
    do_reg = plgpu.load(do_ref.at[0, :, :]).astype(jnp.float32)
    logsumexp_reg = plgpu.load(logsumexp_ref.at[0, :]).astype(jnp.float32)
    d_reg = plgpu.load(d_ref.at[0, :]).astype(jnp.float32)
    dq_acc = jnp.zeros(dq_ref.shape, dtype=jnp.float32)  # Use float32 for numerical stability
    # Loop over all KV blocks:
    def body(t, carry):
        dq_acc = carry
    #   Load K, V for current KV block
        idx = pl.dslice(t * BLOCK_C, BLOCK_C)
        k_blk = plgpu.load(k_ref.at[0, idx, :]).astype(jnp.float32)
        v_blk = plgpu.load(v_ref.at[0, idx, :]).astype(jnp.float32)
    #   Recompute P = softmax(Q @ K^T / scale) using logsumexp
        s_blk = pl.dot(q_reg, k_blk, trans_b=True) / scale
        p_blk = jnp.exp(s_blk - logsumexp_reg[..., None])
    #   Compute dP = dO @ V^T
        dp_blk = pl.dot(do_reg, v_blk, trans_b=True)
    #   Compute dS = P * (dP - D)
        ds_blk = p_blk * ( dp_blk - d_reg[..., None] ) / scale
    #   Accumulate: dQ += dS @ K
        dq_acc += pl.dot(ds_blk, k_blk)
        return dq_acc
    # Store dQ
    dq_acc = jax.lax.fori_loop(0, num_kv_blocks, body, dq_acc)
    plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))


def flash_attention_bwd_dq(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale):
    """Compute dQ using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_kv_blocks = pl.cdiv(T, BLOCK_C)
    grid = (B_flat, pl.cdiv(T, BLOCK_R))  # Outer loop over Q blocks

    dq_flat = pl.pallas_call(
        partial(flash_attention_bwd_dq_kernel, scale=scale, num_kv_blocks=num_kv_blocks),
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
    scale = math.sqrt(C)

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
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale
    )

    # Kernel 3: Compute dQ (outer loop over Q blocks)
    dq_flat = flash_attention_bwd_dq(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale
    )

    return (
        dq_flat.reshape(q.shape),
        dk_flat.reshape(k.shape),
        dv_flat.reshape(v.shape),
    )


# =============================================================================
# V2: Fused backward kernel (single kernel for dQ, dK, dV)
# =============================================================================
# Constraint: BLOCK_R == BLOCK_C (both must be equal)
# This allows us to use the same grid index as both Q block and KV block index

def flash_attention_bwd_kernel_v2(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dq_ref, dk_ref, dv_ref,
    *, scale, num_blocks
):
    """Fused backward kernel computing dQ, dK, dV in a single kernel.

    Grid: (batch*heads, num_blocks)
    - program_id(1) is used as BOTH KV block index (scan 1) AND Q block index (scan 2)
    - Constraint: num_q_blocks == num_kv_blocks, i.e., BLOCK_R == BLOCK_C
    - All inputs are full sequences; we use block_idx to select which block to load

    Scan 1: Compute dK, dV for block_idx as KV block
        - Load K, V at block_idx, loop over all Q blocks
    Scan 2: Compute dQ for block_idx as Q block
        - Load Q, dO at block_idx, loop over all KV blocks
    """
    block_idx = pl.program_id(1)
    BLOCK = BLOCK_R  # == BLOCK_C

    # =========================================================================
    # Scan 1: Compute dK, dV
    # block_idx = KV block index, loop over Q blocks
    # =========================================================================
    # Load K, V for this KV block (using block_idx to index into full sequence)
    kv_idx = pl.dslice(block_idx * BLOCK, BLOCK)
    k_reg = plgpu.load(k_ref.at[0, kv_idx, :]).astype(jnp.float32)
    v_reg = plgpu.load(v_ref.at[0, kv_idx, :]).astype(jnp.float32)

    # Accumulators match output ref shapes (1, BLOCK, C)
    dk_acc = jnp.zeros((1, BLOCK, k_ref.shape[2]), dtype=jnp.float32)
    dv_acc = jnp.zeros((1, BLOCK, v_ref.shape[2]), dtype=jnp.float32)

    def dkv_body(t, carry):
        dk_acc, dv_acc = carry
        idx = pl.dslice(t * BLOCK, BLOCK)
        # Load Q, dO, logsumexp, D for current Q block
        q_blk = plgpu.load(q_ref.at[0, idx, :]).astype(jnp.float32)
        do_blk = plgpu.load(do_ref.at[0, idx, :]).astype(jnp.float32)
        logsumexp_blk = plgpu.load(logsumexp_ref.at[0, idx]).astype(jnp.float32)
        d_blk = plgpu.load(d_ref.at[0, idx]).astype(jnp.float32)
        # Recompute P
        s_blk = pl.dot(q_blk, k_reg, trans_b=True) / scale
        p_blk = jnp.exp(s_blk - logsumexp_blk[..., None])
        # Compute dP, dS
        dp_blk = pl.dot(do_blk, v_reg, trans_b=True)
        ds_blk = p_blk * (dp_blk - d_blk[..., None]) / scale
        # Accumulate dV, dK (add leading dimension for 3D accumulators)
        dv_acc += pl.dot(p_blk, do_blk, trans_a=True)[None, :, :]
        dk_acc += pl.dot(ds_blk, q_blk, trans_a=True)[None, :, :]
        return dk_acc, dv_acc

    dk_acc, dv_acc = jax.lax.fori_loop(0, num_blocks, dkv_body, (dk_acc, dv_acc))
    plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
    plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))

    # =========================================================================
    # Scan 2: Compute dQ
    # block_idx = Q block index, loop over KV blocks
    # =========================================================================
    # Load Q, dO, logsumexp, D for this Q block (using block_idx to index into full sequence)
    q_idx = pl.dslice(block_idx * BLOCK, BLOCK)
    q_reg = plgpu.load(q_ref.at[0, q_idx, :]).astype(jnp.float32)
    do_reg = plgpu.load(do_ref.at[0, q_idx, :]).astype(jnp.float32)
    logsumexp_reg = plgpu.load(logsumexp_ref.at[0, q_idx]).astype(jnp.float32)
    d_reg = plgpu.load(d_ref.at[0, q_idx]).astype(jnp.float32)

    # Accumulator matches output ref shape (1, BLOCK, C)
    dq_acc = jnp.zeros((1, BLOCK, q_ref.shape[2]), dtype=jnp.float32)

    def dq_body(t, carry):
        dq_acc = carry
        idx = pl.dslice(t * BLOCK, BLOCK)
        # Load K, V for current KV block
        k_blk = plgpu.load(k_ref.at[0, idx, :]).astype(jnp.float32)
        v_blk = plgpu.load(v_ref.at[0, idx, :]).astype(jnp.float32)
        # Recompute P
        s_blk = pl.dot(q_reg, k_blk, trans_b=True) / scale
        p_blk = jnp.exp(s_blk - logsumexp_reg[..., None])
        # Compute dP, dS
        dp_blk = pl.dot(do_reg, v_blk, trans_b=True)
        ds_blk = p_blk * (dp_blk - d_reg[..., None]) / scale
        # Accumulate dQ (add leading dimension for 3D accumulator)
        dq_acc += pl.dot(ds_blk, k_blk)[None, :, :]
        return dq_acc

    dq_acc = jax.lax.fori_loop(0, num_blocks, dq_body, dq_acc)
    plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))


def flash_attention_bwd_v2_call(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale):
    """Pallas call wrapper for fused backward kernel."""
    B_flat, T, C = q_flat.shape
    assert BLOCK_R == BLOCK_C, "V2 kernel requires BLOCK_R == BLOCK_C"
    BLOCK = BLOCK_R
    num_blocks = pl.cdiv(T, BLOCK)
    grid = (B_flat, num_blocks)

    dq_flat, dk_flat, dv_flat = pl.pallas_call(
        partial(flash_attention_bwd_kernel_v2, scale=scale, num_blocks=num_blocks),
        out_shape=[
            jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),   # dQ
            jax.ShapeDtypeStruct(k_flat.shape, k_flat.dtype),   # dK
            jax.ShapeDtypeStruct(v_flat.shape, v_flat.dtype),   # dV
        ],
        grid=grid,
        in_specs=[
            # All inputs are full sequences - kernel uses block_idx to select blocks
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # q (full)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # k (full)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # v (full)
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),       # do (full)
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),             # logsumexp (full)
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),             # d (full)
        ],
        out_specs=[
            # Outputs are blocked - each program writes one block
            pl.BlockSpec((1, BLOCK, C), lambda b, t: (b, t, 0)),   # dq (blocked)
            pl.BlockSpec((1, BLOCK, C), lambda b, t: (b, t, 0)),   # dk (blocked)
            pl.BlockSpec((1, BLOCK, C), lambda b, t: (b, t, 0)),   # dv (blocked)
        ],
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
    )(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat)
    return dq_flat, dk_flat, dv_flat


@jax.jit
def flash_attention_bwd_v2(q, k, v, o, logsumexp, do):
    """Flash attention backward pass V2: fused single kernel."""
    B, H, T, C = q.shape
    scale = math.sqrt(C)

    # Flatten batch and head dimensions
    q_flat = q.reshape(-1, T, C)
    k_flat = k.reshape(-1, T, C)
    v_flat = v.reshape(-1, T, C)
    o_flat = o.reshape(-1, T, C)
    do_flat = do.reshape(-1, T, C)
    logsumexp_flat = logsumexp.reshape(-1, T)

    # Kernel 1: Preprocess - compute D = rowsum(O ⊙ dO)
    d_flat = flash_attention_bwd_preprocess(o_flat, do_flat)

    # Kernel 2: Fused dQ, dK, dV
    dq_flat, dk_flat, dv_flat = flash_attention_bwd_v2_call(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale
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


# V2 Custom VJP wrapper (uses fused backward kernel)

@jax.custom_vjp
def flash_attention_v2(q, k, v):
    """Flash attention V2 with fused backward pass."""
    o, _ = flash_attention_fwd(q, k, v)
    return o


def flash_attention_v2_fwd_rule(q, k, v):
    """Forward rule for custom_vjp."""
    o, logsumexp = flash_attention_fwd(q, k, v)
    return o, (q, k, v, o, logsumexp)


def flash_attention_v2_bwd_rule(res, do):
    """Backward rule using fused V2 kernel."""
    q, k, v, o, logsumexp = res
    dq, dk, dv = flash_attention_bwd_v2(q, k, v, o, logsumexp, do)
    return dq, dk, dv


flash_attention_v2.defvjp(flash_attention_v2_fwd_rule, flash_attention_v2_bwd_rule)


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

    B, H, T, D = 2, 2, 256, 64
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

    # Test forward pass (using flash_attention_fwd directly to get logsumexp)
    o_flash, logsumexp_flash = flash_attention_fwd(q, k, v)
    assert jnp.allclose(o_flash, o_ref, atol=1e-2, rtol=1e-2)
    assert jnp.allclose(logsumexp_flash, logsumexp_ref, atol=1e-2, rtol=1e-2)
    print("Forward pass check passed!")

    # Test preprocess kernel: D = rowsum(O * dO)
    d_ref = jnp.sum(o_ref * do, axis=-1)  # (B, H, T)
    o_flat = o_flash.reshape(-1, T, D)
    do_flat = do.reshape(-1, T, D)
    d_flash = flash_attention_bwd_preprocess(o_flat, do_flat).reshape(B, H, T)
    assert jnp.allclose(d_flash, d_ref, atol=1e-2, rtol=1e-2), f"D max diff: {jnp.max(jnp.abs(d_flash - d_ref))}"
    print("Preprocess kernel (D) check passed!")

    # Test backward pass (dQ only for now, dK/dV return dummy zeros)
    dq_flash, dk_flash, dv_flash = flash_attention_bwd(q, k, v, o_flash, logsumexp_flash, do)
    assert jnp.allclose(dq_flash, dq_ref, atol=1e-2, rtol=1e-2), f"dQ max diff: {jnp.max(jnp.abs(dq_flash - dq_ref))}"
    print("dQ check passed!")

    assert jnp.allclose(dk_flash, dk_ref, atol=1e-2, rtol=1e-2), f"dK max diff: {jnp.max(jnp.abs(dk_flash - dk_ref))}"
    assert jnp.allclose(dv_flash, dv_ref, atol=1e-2, rtol=1e-2), f"dV max diff: {jnp.max(jnp.abs(dv_flash - dv_ref))}"
    print("Backward pass V1 (3 kernels) check passed!")

    # Test backward pass V2 (fused kernel)
    dq_flash_v2, dk_flash_v2, dv_flash_v2 = flash_attention_bwd_v2(q, k, v, o_flash, logsumexp_flash, do)
    assert jnp.allclose(dq_flash_v2, dq_ref, atol=1e-2, rtol=1e-2), f"V2 dQ max diff: {jnp.max(jnp.abs(dq_flash_v2 - dq_ref))}"
    assert jnp.allclose(dk_flash_v2, dk_ref, atol=1e-2, rtol=1e-2), f"V2 dK max diff: {jnp.max(jnp.abs(dk_flash_v2 - dk_ref))}"
    assert jnp.allclose(dv_flash_v2, dv_ref, atol=1e-2, rtol=1e-2), f"V2 dV max diff: {jnp.max(jnp.abs(dv_flash_v2 - dv_ref))}"
    print("Backward pass V2 (fused kernel) check passed!")

    # Timing comparison with JAX built-in dot_product_attention
    print("\n" + "="*60)
    print("Timing Comparison")
    print("="*60)

    # Use larger sizes for meaningful timing
    # Use float16 for cuDNN compatibility
    B_bench, H_bench, T_bench, D_bench = 4, 8, 1024, 64
    q_bench = jax.random.normal(keys[0], (B_bench, H_bench, T_bench, D_bench), dtype=jnp.float16)
    k_bench = jax.random.normal(keys[1], (B_bench, H_bench, T_bench, D_bench), dtype=jnp.float16)
    v_bench = jax.random.normal(keys[2], (B_bench, H_bench, T_bench, D_bench), dtype=jnp.float16)
    do_bench = jax.random.normal(keys[3], (B_bench, H_bench, T_bench, D_bench), dtype=jnp.float16)

    print(f"Benchmark shape: B={B_bench}, H={H_bench}, T={T_bench}, D={D_bench}")

    def _bench_fwd(fn, q, k, v, iters=20):
        # Warmup
        for _ in range(3):
            out = fn(q, k, v)
            jax.block_until_ready(out)
        # Bench
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(q, k, v)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times) * 1000  # ms

    def _bench_bwd(fn, q, k, v, do, iters=20):
        # Warmup
        for _ in range(3):
            grads = jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v) * do), argnums=(0, 1, 2))(q, k, v)
            jax.block_until_ready(grads)
        # Bench
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            grads = jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v) * do), argnums=(0, 1, 2))(q, k, v)
            jax.block_until_ready(grads)
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times) * 1000  # ms

    # JAX built-in (note: different input layout - expects (B, T, H, D))
    @jax.jit
    def jax_dot_product_attention(q, k, v):
        # Transpose from (B, H, T, D) to (B, T, H, D) for jax.nn.dot_product_attention
        q_t = jnp.transpose(q, (0, 2, 1, 3))
        k_t = jnp.transpose(k, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))
        out = jax.nn.dot_product_attention(q_t, k_t, v_t, implementation='cudnn')
        return jnp.transpose(out, (0, 2, 1, 3))  # Back to (B, H, T, D)

    # Our flash attention V1 (3 separate backward kernels)
    @jax.jit
    def our_flash_attention_v1(q, k, v):
        return flash_attention(q, k, v)

    # Our flash attention V2 (fused backward kernel)
    @jax.jit
    def our_flash_attention_v2(q, k, v):
        return flash_attention_v2(q, k, v)

    # Reference (materialized attention matrix)
    @jax.jit
    def reference_attention(q, k, v):
        o, _ = mha_reference(q, k, v)
        return o

    print("\nForward pass:")
    t_jax = _bench_fwd(jax_dot_product_attention, q_bench, k_bench, v_bench)
    print(f"  JAX dot_product_attention: {t_jax:.3f} ms")
    t_ours = _bench_fwd(our_flash_attention_v1, q_bench, k_bench, v_bench)
    print(f"  Our flash_attention:       {t_ours:.3f} ms")
    t_ref = _bench_fwd(reference_attention, q_bench, k_bench, v_bench)
    print(f"  Reference (materialized):  {t_ref:.3f} ms")

    print("\nBackward pass:")
    t_jax_bwd = _bench_bwd(jax_dot_product_attention, q_bench, k_bench, v_bench, do_bench)
    print(f"  JAX dot_product_attention: {t_jax_bwd:.3f} ms")
    t_ours_v1_bwd = _bench_bwd(our_flash_attention_v1, q_bench, k_bench, v_bench, do_bench)
    print(f"  Our flash_attention V1:    {t_ours_v1_bwd:.3f} ms (3 kernels)")
    t_ours_v2_bwd = _bench_bwd(our_flash_attention_v2, q_bench, k_bench, v_bench, do_bench)
    print(f"  Our flash_attention V2:    {t_ours_v2_bwd:.3f} ms (fused)")
    t_ref_bwd = _bench_bwd(reference_attention, q_bench, k_bench, v_bench, do_bench)
    print(f"  Reference (materialized):  {t_ref_bwd:.3f} ms")
