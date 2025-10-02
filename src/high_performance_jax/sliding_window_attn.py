import jax
import jax.numpy as jnp
import time
from typing import Tuple


def make_sliding_mask(seq_len: int, window: int) -> jnp.ndarray:
  """Create a boolean attention mask of shape (seq_len, seq_len) where True
  indicates allowed attention within a sliding window of size `window`.
  """
  mask = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)
  for i in range(0, seq_len - window + 1):
    mask = mask.at[i:i+window, i:i+window].set(True)
  return mask


def make_causal_mask(seq_len: int) -> jnp.ndarray:
  """Create a causal (lower-triangular) attention mask of shape (seq_len, seq_len).
  True indicates allowed attention (positions can attend to earlier positions and themselves).
  """
  return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def benchmark_compare(batch: int = 8, seq_len: int = 1024, heads: int = 8, head_dim: int = 64, window: int = 64, warmup: int = 2, runs: int = 10):
  """Compare timings for sliding-window vs causal attention masks.

  Prints results for each mask and returns a tuple of (sliding_mean, causal_mean).
  """
  key = jax.random.PRNGKey(0)
  k1, k2, k3 = jax.random.split(key, 3)
  Q = jax.random.normal(k1, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
  K = jax.random.normal(k2, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
  V = jax.random.normal(k3, (batch, seq_len, heads, head_dim), dtype=jnp.float32)

  sliding = make_sliding_mask(seq_len, window)[None, None, :, :]
  causal = make_causal_mask(seq_len)[None, None, :, :]

  def attn_fn(Q, K, V, mask):
    return jax.nn.dot_product_attention(query=Q, key=K, value=V, bias=None, mask=mask)

  # JIT compile separately for each mask (to get representative performance)
  jit_sliding = jax.jit(attn_fn)
  jit_causal = jax.jit(attn_fn)

  def run_benchmark(jit_fn, mask, name: str):
    # warmup
    out = None
    for _ in range(warmup):
      out = jit_fn(Q, K, V, mask)
    if out is not None:
      jax.block_until_ready(out)

    timings = []
    print(f"Benchmarking {name}: batch={batch}, seq_len={seq_len}, heads={heads}, head_dim={head_dim}, window={window if name=='sliding' else 'N/A'}")
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
  causal_stats = run_benchmark(jit_causal, causal, 'causal')

  print(f"Comparison: sliding={sliding_stats[0]:.2f} ms, causal={causal_stats[0]:.2f} ms")
  return sliding_stats, causal_stats


def benchmark_dot_product_attention(batch: int = 8, seq_len: int = 1024, heads: int = 8, head_dim: int = 64, window: int = 64, warmup: int = 2, runs: int = 10) -> Tuple[float, float]:
  """Benchmark jax.nn.dot_product_attention with a sliding window mask.

  Returns (mean_ms, std_ms) for the timed runs.
  """
  key = jax.random.PRNGKey(0)
  k1, k2, k3 = jax.random.split(key, 3)
  # shapes: [batch, seq_len, heads, head_dim]
  Q = jax.random.normal(k1, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
  K = jax.random.normal(k2, (batch, seq_len, heads, head_dim), dtype=jnp.float32)
  V = jax.random.normal(k3, (batch, seq_len, heads, head_dim), dtype=jnp.float32)

  # create mask of shape (seq_len, seq_len) and expand to attention mask expected by dot_product_attention
  base_mask = make_sliding_mask(seq_len, window)
  # dot_product_attention expects an array broadcastable to [batch, heads, seq_len, seq_len]
  attn_mask = base_mask[None, None, :, :]

  # JIT compile the attention function
  def attn_fn(Q, K, V, mask):
    return jax.nn.dot_product_attention(query=Q, key=K, value=V, bias=None, mask=mask)

  jit_fn = jax.jit(attn_fn)

  # Warmup
  print(f"Warming up JAX compilation and first runs (warmup={warmup})...")
  out = None
  for _ in range(warmup):
    out = jit_fn(Q, K, V, attn_mask)
  # Ensure computation finished
  if out is not None:
    jax.block_until_ready(out)

  timings = []
  print(f"Running {runs} timed runs: batch={batch}, seq_len={seq_len}, heads={heads}, head_dim={head_dim}, window={window}")
  for i in range(runs):
    t0 = time.perf_counter()
    out = jit_fn(Q, K, V, attn_mask)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    timings.append(ms)
    print(f"run {i+1}/{runs}: {ms:.2f} ms")

  import statistics
  mean_ms = statistics.mean(timings)
  std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
  print(f"Mean {mean_ms:.2f} ms ± {std_ms:.2f} ms over {runs} runs")
  return mean_ms, std_ms


if __name__ == '__main__':
  # small default compare run; adjust params as needed
  benchmark_compare(batch=32, seq_len=2048, heads=8, head_dim=64, window=64, warmup=2, runs=5)
