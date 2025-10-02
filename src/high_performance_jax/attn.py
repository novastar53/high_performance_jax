"""Compare performance of nnx.MultiHeadAttention vs jax.nn.dot_product_attention.

This script tries to import a MultiHeadAttention from `flax.nnx` (alias nnx). If
not available, it will skip the nnx test and only benchmark jax.nn.dot_product_attention.

Usage: run from repository root with PYTHONPATH=src:
    PYTHONPATH=src python3 src/high_performance_jax/attn.py
"""

import time
from typing import Tuple

import jax
import jax.numpy as jnp

try:
    import flax.nnx as nnx
except Exception:
    nnx = None


def make_qkv(batch: int, seq_len: int, d_model: int):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (batch, seq_len, d_model), dtype=jnp.float32)
    K = jax.random.normal(k2, (batch, seq_len, d_model), dtype=jnp.float32)
    V = jax.random.normal(k3, (batch, seq_len, d_model), dtype=jnp.float32)
    return Q, K, V


def bench_dot_product(batch: int, seq_len: int, heads: int, head_dim: int, warmup: int = 2, runs: int = 10) -> Tuple[float, float]:
    d_model = heads * head_dim
    Q, K, V = make_qkv(batch, seq_len, d_model)

    # Reshape to [batch, seq, heads, head_dim]
    Qh = Q.reshape(batch, seq_len, heads, head_dim)
    Kh = K.reshape(batch, seq_len, heads, head_dim)
    Vh = V.reshape(batch, seq_len, heads, head_dim)

    def attn_fn(Q, K, V):
        return jax.nn.dot_product_attention(query=Q, key=K, value=V, mask=None, bias=None)

    jit_fn = jax.jit(attn_fn)

    # warmup
    out = None
    for _ in range(warmup):
        out = jit_fn(Qh, Kh, Vh)
    if out is not None:
        jax.block_until_ready(out)

    timings = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = jit_fn(Qh, Kh, Vh)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    import statistics
    return statistics.mean(timings), statistics.stdev(timings) if len(timings) > 1 else 0.0


def bench_nnx_mha(batch: int, seq_len: int, heads: int, head_dim: int, warmup: int = 2, runs: int = 10) -> Tuple[float, float]:
    if nnx is None:
        raise RuntimeError('flax.nnx (nnx) is not available in this environment')

    d_model = heads * head_dim
    Q, K, V = make_qkv(batch, seq_len, d_model)

    # Some nnx.MultiHeadAttention APIs expect [batch, seq, d_model]
    # Build a simple module instance if possible. We'll attempt to call a functional API
    # if nnx exposes it; otherwise, try creating an nnx.Module.

    # Try a functional call pattern first: nnx.MultiHeadAttention(...)(query, key, value)
    try:
        # instantiate with required args
        rngs = nnx.Rngs(default=0)
        in_features = heads * head_dim
        mha = nnx.MultiHeadAttention(in_features=in_features, num_heads=heads, rngs=rngs)
    except Exception as e:
        # Fallback: signal inability to construct
        raise RuntimeError(f'Failed to construct nnx.MultiHeadAttention: {e}')

    if mha is None:
        raise RuntimeError('Could not construct nnx.MultiHeadAttention (API mismatch)')

    # Wrap call in jit
    def mha_fn(q, k, v):
        # Call MHA in non-decode mode (decode=False) so cache paths aren't used.
        try:
            return mha(q, decode=False)
        except TypeError:
            try:
                return mha(q, k, v, decode=False)
            except TypeError:
                # last resorts
                try:
                    return mha(q)
                except TypeError:
                    return mha(q, k, v)

    jit_mha = jax.jit(mha_fn)

    # Initialize autoregressive cache if MHA provides the helper
    try:
        if hasattr(mha, 'init_cache'):
            # Some nnx MHA implementations expect the full input shape
            mha.init_cache(Q.shape)
    except Exception:
        # ignore cache init failures and proceed; the call may still work in non-decode mode
        pass

    # warmup
    out = None
    for _ in range(warmup):
        out = jit_mha(Q, K, V)
    if out is not None:
        jax.block_until_ready(out)

    timings = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = jit_mha(Q, K, V)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    import statistics
    return statistics.mean(timings), statistics.stdev(timings) if len(timings) > 1 else 0.0


def compare(batch=32, seq_len=4096, heads=8, head_dim=256, warmup: int = 2, runs: int = 10):
    print(f'Parameters: batch={batch}, seq_len={seq_len}, heads={heads}, head_dim={head_dim}')
    dp_mean, dp_std = bench_dot_product(batch, seq_len, heads, head_dim, warmup=warmup, runs=runs)
    print(f'jax.nn.dot_product_attention: mean={dp_mean:.2f} ms std={dp_std:.2f} ms')

    if nnx is not None:
        try:
            nnx_mean, nnx_std = bench_nnx_mha(batch, seq_len, heads, head_dim, warmup=warmup, runs=runs)
            print(f'nnx.MultiHeadAttention: mean={nnx_mean:.2f} ms std={nnx_std:.2f} ms')
        except Exception as e:
            print(f'Could not benchmark nnx.MultiHeadAttention: {e}')
    else:
        print('flax.nnx not available; skipped nnx.MultiHeadAttention benchmark')


if __name__ == '__main__':
    compare()
