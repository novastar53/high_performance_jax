import time
import jax
import jax.numpy as jnp


def generate_inputs(batch, n_heads, seq_len, d_head, key):
    key, subkey = jax.random.split(key)
    q = jax.random.normal(subkey, (batch, n_heads, seq_len, d_head), dtype=jnp.bfloat16)
    k = jax.random.normal(subkey, (batch, n_heads, seq_len, d_head), dtype=jnp.bfloat16)
    v = jax.random.normal(subkey, (batch, n_heads, seq_len, d_head), dtype=jnp.bfloat16)
    return q, k, v, key

def benchmark_attention(batch=32, n_heads=8, seq_len=2048*16, d_head=64,
                        window=(16, 0), n_warmup=5, n_iters=1000):
    key = jax.random.PRNGKey(0)

    # JIT versions
    full_attn = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation='cudnn'))
    local_attn = jax.jit(
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, local_window_size=window, implementation='cudnn')
    )

    # Warmup
    for _ in range(n_warmup):
        q, k, v, key = generate_inputs(batch, n_heads, seq_len, d_head, key) 
        _ = full_attn(q, k, v).block_until_ready()
        _ = local_attn(q, k, v).block_until_ready()

    # Full attention timing
    start = time.time()
    for _ in range(n_iters):
        q, k, v, key = generate_inputs(batch, n_heads, seq_len, d_head, key) 
        _ = full_attn(q, k, v).block_until_ready()
    full_time = (time.time() - start) / n_iters

    # Local attention timing
    start = time.time()
    for _ in range(n_iters):
        q, k, v, key = generate_inputs(batch, n_heads, seq_len, d_head, key) 
        _ = local_attn(q, k, v).block_until_ready()
    local_time = (time.time() - start) / n_iters

    print(f"batch={batch}, heads={n_heads}, seq_len={seq_len}, d_head={d_head}, window={window}")
    print(f"Full attention avg: {full_time * 1e3:.3f} ms")
    print(f"Sliding-window attention avg: {local_time * 1e3:.3f} ms")


if __name__ == "__main__":
    benchmark_attention()