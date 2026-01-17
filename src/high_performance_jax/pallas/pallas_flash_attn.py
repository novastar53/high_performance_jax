"""Pallas Flash Attention implementation.

Input shapes: Q, K, V are (B, H, N, D) where:
    B = batch size
    H = number of heads
    N = sequence length
    D = head dimension
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


INTERPRET_MODE = True  # Set to False on GPU

BLOCK_Q = 64
BLOCK_KV = 64
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
    return o


# Flash Attention Forward

def flash_attention_fwd_kernel(q_ref, k_ref, v_ref, o_ref, *, num_kv_blocks, scale):
    """Flash attention forward kernel."""
    pass


@jax.jit
def flash_attention_fwd(q, k, v):
    """Flash attention forward pass."""
    pass


# Flash Attention Backward

def flash_attention_bwd_kernel(q_ref, k_ref, v_ref, o_ref, do_ref, l_ref, m_ref,
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
    pass


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

    B, H, N, D = 2, 4, 256, 64
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, H, N, D), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, H, N, D), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, H, N, D), dtype=jnp.float32)
    do = jax.random.normal(keys[3], (B, H, N, D), dtype=jnp.float32)

    # Forward check
    o_ref = mha_reference(q, k, v)
    print(f"Reference output shape: {o_ref.shape}")

    # Backward check (reference)
    def loss_ref(q, k, v):
        return jnp.sum(mha_reference(q, k, v) * do)

    dq_ref, dk_ref, dv_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    print(f"Reference gradient shapes: dq={dq_ref.shape}, dk={dk_ref.shape}, dv={dv_ref.shape}")

    # TODO: Uncomment once implemented
    # o_flash = flash_attention(q, k, v)
    # assert jnp.allclose(o_flash, o_ref, atol=1e-2, rtol=1e-2)
    # print("Forward pass check passed!")
    #
    # def loss_flash(q, k, v):
    #     return jnp.sum(flash_attention(q, k, v) * do)
    #
    # dq_flash, dk_flash, dv_flash = jax.grad(loss_flash, argnums=(0, 1, 2))(q, k, v)
    # assert jnp.allclose(dq_flash, dq_ref, atol=1e-2, rtol=1e-2)
    # assert jnp.allclose(dk_flash, dk_ref, atol=1e-2, rtol=1e-2)
    # assert jnp.allclose(dv_flash, dv_ref, atol=1e-2, rtol=1e-2)
    # print("Backward pass check passed!")
