import jax
import jax.numpy as jnp
import time
from typing import Callable

try:
  import flax.nnx as nnx  # type: ignore
except Exception:
  nnx = None


def make_sliding_mask(seq_len: int, window: int) -> jnp.ndarray:
  """Create a boolean attention mask with True inside a sliding window."""
  mask = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)
  for i in range(0, seq_len - window + 1):
    mask = mask.at[i:i+window, i:i+window].set(True)
  return mask


def make_causal_mask(seq_len: int) -> jnp.ndarray:
  """Create a lower-triangular causal attention mask."""
  return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def _make_attention_impl(backend: str,
                         batch: int,
                         seq_len: int,
                         heads: int,
                         head_dim: int) -> Callable:
  backend = backend.lower()
  if backend not in {'dot_product', 'nnx'}:
    raise ValueError(f"Unsupported attention backend '{backend}'. Choose from 'dot_product' or 'nnx'.")

  if backend == 'dot_product':
    def dot_product_attn(Q, K, V, mask=None, causal: bool = False):
      return jax.nn.dot_product_attention(query=Q,
                                          key=K,
                                          value=V,
                                          bias=None,
                                          mask=mask,
                                          is_causal=causal)

    return dot_product_attn

  if nnx is None:
    raise RuntimeError("flax.nnx is not available; cannot use 'nnx' attention backend")

  d_model = heads * head_dim
  rngs = nnx.Rngs(default=0)

  try:
    mha = nnx.MultiHeadAttention(in_features=d_model, num_heads=heads, rngs=rngs)
  except Exception as exc:  # pragma: no cover - defensive for mismatched nnx versions
    raise RuntimeError(f"Failed to construct nnx.MultiHeadAttention: {exc}") from exc

  def nnx_attn(Q, K, V, mask=None, causal: bool = False):
    q = Q.reshape(batch, seq_len, d_model)
    k = K.reshape(batch, seq_len, d_model)
    v = V.reshape(batch, seq_len, d_model)

    attn_mask = mask
    if attn_mask is None and causal:
      attn_mask = make_causal_mask(seq_len)[None, None, :, :]
    if attn_mask is not None:
      attn_mask = jnp.asarray(attn_mask, dtype=jnp.bool_)

    kwargs = {'decode': False}
    if attn_mask is not None:
      kwargs['mask'] = attn_mask

    try:
      out = mha(q, k, v, **kwargs)
    except TypeError:
      try:
        out = mha(q, **kwargs)
      except TypeError:
        out = mha(q)

    if out.ndim == 3 and out.shape[-1] == d_model:
      return out.reshape(batch, seq_len, heads, head_dim)
    return out

  return nnx_attn


def benchmark_compare(batch: int = 8,
                      seq_len: int = 1024,
                      heads: int = 8,
                      head_dim: int = 64,
                      window: int = 64,
                      warmup: int = 2,
                      runs: int = 10,
                      backend: str = 'dot_product'):
  """Compare timings for sliding-window vs causal attention masks.

  Prints results for each mask and returns a tuple of (sliding_mean, causal_mean).
  """
  key = jax.random.PRNGKey(0)
  k1, k2, k3 = jax.random.split(key, 3)
  Q = jax.random.normal(k1, (batch, seq_len, heads, head_dim), dtype=jnp.bfloat16)
  K = jax.random.normal(k2, (batch, seq_len, heads, head_dim), dtype=jnp.bfloat16)
  V = jax.random.normal(k3, (batch, seq_len, heads, head_dim), dtype=jnp.bfloat16)

  sliding = make_sliding_mask(seq_len, window)[None, None, :, :]
  causal = make_causal_mask(seq_len)[None, None, :, :]

  attn_impl = _make_attention_impl(backend=backend,
                                   batch=batch,
                                   seq_len=seq_len,
                                   heads=heads,
                                   head_dim=head_dim)

  # JIT compile separately for each case (sliding mask vs causal mask) to get representative performance
  jit_sliding = jax.jit(lambda Q, K, V, mask: attn_impl(Q, K, V, mask, causal=False))
  jit_causal = jax.jit(lambda Q, K, V, mask: attn_impl(Q, K, V, mask, causal=True))

  def run_benchmark(jit_fn: Callable, mask, name: str):
    # warmup
    out = None
    for _ in range(warmup):
      out = jit_fn(Q, K, V, mask)
    if out is not None:
      jax.block_until_ready(out)

    timings = []
    print(f"Benchmarking {name} ({backend}): batch={batch}, seq_len={seq_len}, heads={heads}, head_dim={head_dim}, window={window if name=='sliding' else 'N/A'}")
    for i in range(runs):
      t0 = time.perf_counter()
      out = jit_fn(Q, K, V, mask)
      jax.block_until_ready(out)
      t1 = time.perf_counter()
      ms = (t1 - t0) * 1000.0
      timings.append(ms)
      print(f"  run {i+1}/{runs}: {ms:.2f} ms")

    import statistics
    mean_ms = statistics.mean(timings)
    std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"{name} mean {mean_ms:.2f} ms ± {std_ms:.2f} ms\n")
    return mean_ms, std_ms

  sliding_stats = run_benchmark(jit_sliding, sliding, 'sliding')
  causal_stats = run_benchmark(jit_causal, causal if backend != 'dot_product' else None, 'causal')

  print(f"Comparison: sliding={sliding_stats[0]:.2f} ms, causal={causal_stats[0]:.2f} ms")
  return sliding_stats, causal_stats


if __name__ == '__main__':
  # small default compare run; adjust params as needed
  benchmark_compare(batch=16,
                    seq_len=1024*12,
                    heads=8,
                    head_dim=64,
                    window=1024,
                    warmup=2,
                    runs=5,
                    backend='nnx')
