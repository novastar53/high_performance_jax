from functools import partial
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

INTERPRET_MODE = False
BLOCK_R = 64
BLOCK_C = 64
NUM_WARPS = 4
NUM_STAGES = 3
DTYPE = jnp.bfloat16
WINDOW_SIZE = 128


@jax.jit
def sliding_window_attn_ref(q, k, v, window_size):
    B, H, T, C = q.shape
    scale = 1.0 / jnp.sqrt(C)
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    
    q_pos = jnp.arange(T)
    k_pos = jnp.arange(T)
    mask = jnp.abs(q_pos[:, None] - k_pos[None, :]) <= (window_size // 2)
    
    logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', probs, v)


def sliding_window_fwd_kernel(q_ref, k_ref, v_ref, o_ref, *, scale, num_k_blocks, window_size):
    o_dummy = jnp.zeros(o_ref.shape, dtype=o_ref.dtype)
    plgpu.store(o_ref, o_dummy)


def sliding_window_bwd_preprocess_kernel(o_ref, do_ref, d_ref):
    d_dummy = jnp.zeros(d_ref.shape, dtype=d_ref.dtype)
    plgpu.store(d_ref, d_dummy)


def sliding_window_bwd_preprocess(o_flat, do_flat):
    B_flat, T, _ = o_flat.shape
    return jnp.zeros((B_flat, T), dtype=o_flat.dtype)


def sliding_window_bwd_dkv_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dk_ref, dv_ref,
    *, scale, num_q_blocks, window_size
):
    dk_dummy = jnp.zeros(dk_ref.shape, dtype=dk_ref.dtype)
    dv_dummy = jnp.zeros(dv_ref.shape, dtype=dv_ref.dtype)
    plgpu.store(dk_ref, dk_dummy)
    plgpu.store(dv_ref, dv_dummy)


def sliding_window_bwd_dkv(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale):
    return jnp.zeros(k_flat.shape, dtype=k_flat.dtype), jnp.zeros(v_flat.shape, dtype=v_flat.dtype)


def sliding_window_bwd_dq_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dq_ref,
    *, scale, num_kv_blocks, window_size
):
    dq_dummy = jnp.zeros(dq_ref.shape, dtype=dq_ref.dtype)
    plgpu.store(dq_ref, dq_dummy)


def sliding_window_bwd_dq(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat, scale):
    return jnp.zeros(q_flat.shape, dtype=q_flat.dtype)


@jax.jit
def sliding_window_fwd_with_lse(q, k, v):
    return jnp.zeros(q.shape, dtype=q.dtype), jnp.zeros(q.shape[:-1], dtype=q.dtype)


@jax.jit
def sliding_window_bwd(q, k, v, o, logsumexp, do):
    return jnp.zeros(q.shape, dtype=q.dtype), jnp.zeros(k.shape, dtype=k.dtype), jnp.zeros(v.shape, dtype=v.dtype)


@jax.custom_vjp
def sliding_window_attention(q, k, v):
    return jnp.zeros(q.shape, dtype=q.dtype)


def sliding_window_fwd_rule(q, k, v):
    o = jnp.zeros(q.shape, dtype=q.dtype)
    return o, (q, k, v, o, jnp.zeros(q.shape[:-1], dtype=q.dtype))


def sliding_window_bwd_rule(res, do):
    q, k, v, o, logsumexp = res
    return jnp.zeros(q.shape, dtype=q.dtype), jnp.zeros(k.shape, dtype=k.dtype), jnp.zeros(v.shape, dtype=v.dtype)


sliding_window_attention.defvjp(sliding_window_fwd_rule, sliding_window_bwd_rule)


@jax.jit
def sliding_window_attn(q, k, v):
    return sliding_window_attention(q, k, v)


if __name__ == "__main__":
    import time
    
    B, H, T, D = 2, 2, 256, 64
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)
    
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=DTYPE)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=DTYPE)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=DTYPE)
    
    o_ref = sliding_window_attn_ref(q, k, v, WINDOW_SIZE)
    print(f"Reference output shape: {o_ref.dtype, o_ref.shape}")
    
    o_pallas = sliding_window_attn(q, k, v)
    assert jnp.allclose(o_pallas, o_ref, atol=1e-2, rtol=1e-2)
    print("Forward pass check passed!")
    
    print("\n" + "="*60)
    print("Timing Comparison")
    print("="*60)
    
    B_bench, H_bench, T_bench, D_bench = 4, 8, 4096, 64
    q_bench = jax.random.normal(keys[0], (B_bench, H_bench, T_bench, D_bench), dtype=DTYPE)
    k_bench = jax.random.normal(keys[1], (B_bench, H_bench, T_bench, D_bench), dtype=DTYPE)
    v_bench = jax.random.normal(keys[2], (B_bench, H_bench, T_bench, D_bench), dtype=DTYPE)
    
    print(f"Benchmark shape: B={B_bench}, H={H_bench}, T={T_bench}, D={D_bench}")
    
    def _bench(fn, warmup=3, iters=20):
        for _ in range(warmup):
            out = fn()
            jax.block_until_ready(out)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return jnp.array(times).median() * 1000
    
    ref_fwd = jax.jit(lambda q, k, v: sliding_window_attn_ref(q, k, v, WINDOW_SIZE))
    pallas_fwd = jax.jit(sliding_window_attn)
    
    try:
        from flash_attn_jax import flash_mha
        
        def flash_attn_jax_wrapper(q, k, v):
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            v_t = jnp.transpose(v, (0, 2, 1, 3))
            out = flash_mha(q_t, k_t, v_t, window_size=(WINDOW_SIZE, WINDOW_SIZE))
            return jnp.transpose(out, (0, 2, 1, 3))
        
        flash_attn_jax_fwd = jax.jit(flash_attn_jax_wrapper)
        use_flash_attn_jax = True
    except ImportError:
        print("flash-attn-jax not available")
        use_flash_attn_jax = False
    
    print("\nForward pass:")
    t_ref = _bench(lambda: ref_fwd(q_bench, k_bench, v_bench))
    print(f"  Reference implementation: {t_ref:.3f} ms")
    t_pallas = _bench(lambda: pallas_fwd(q_bench, k_bench, v_bench))
    print(f"  Pallas implementation:    {t_pallas:.3f} ms")
    if use_flash_attn_jax:
        t_flash_attn_jax = _bench(lambda: flash_attn_jax_fwd(q_bench, k_bench, v_bench))
        print(f"  flash_attn_jax:          {t_flash_attn_jax:.3f} ms")
