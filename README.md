# High Performance JAX

Tutorials and recipes for achieving high performance with JAX. This repository contains educational implementations of performance-critical components for machine learning: custom GPU kernels, model sharding, attention mechanisms, and more.

## Installation

```bash
# Install dependencies (auto-detects platform: GPU, TPU, or Metal)
make install

# Development install with Jupyter kernel
make dev
```

## Project Structure

```
src/high_performance_jax/
├── pallas/                    # Custom JAX kernels using Pallas DSL
│   ├── pallas_softmax.py      # Online softmax with custom forward/backward
│   ├── pallas_flash_attn.py   # Flash attention implementation
│   └── pallas_matmul.py       # Matrix multiplication kernels
├── triton/                    # Native Triton GPU kernels (Linux only)
│   └── flash_attention.py     # Full flash attention with autograd
├── profiling.py               # Reusable profiling utilities
├── model_sharding.py          # Flax NNX model sharding
├── moe.py                     # Mixture of Experts with expert parallelism
└── sharded_vmap.py            # Combining vmap with sharding

notebooks/                     # Interactive tutorials
├── pallas-softmax.ipynb       # Pallas softmax blogpost
├── pallas-flash-attn.ipynb    # Flash attention blogpost
└── basic_sharding.ipynb       # Sharding tutorial

scripts/
├── profile_attention.py       # Example profiling script
└── plot_roofline.py          # Generate roofline plots from JSON data
```

## Profiling

This repository includes reusable profiling utilities based on XProf for analyzing kernel performance.

### Quick Start

```python
from high_performance_jax.profiling import profile, profile_comparison, quick_profile

# Profile a single operation
with profile("my_kernel"):
    result = my_kernel(x, y)
    result.block_until_ready()

# Compare implementations
profile_comparison(
    "attention",
    ("flash", lambda: flash_attention(q, k, v)),
    ("reference", lambda: mha_reference(q, k, v)),
)

# One-liner for quick profiling
quick_profile("matmul", lambda: (x @ y).block_until_ready())
```

### Profiling Workflow

Traces are saved to `traces/{YYYY-MM-DD}/{name}/` within the repository, organized by date.

#### Option 1: Download Traces (Recommended)

This approach downloads traces to your local machine so you can analyze them at leisure without keeping a GPU running.

1. **On the remote GPU machine**, run your profiling script:
   ```bash
   python scripts/profile_attention.py --seq-len 1024 --backward
   ```

2. **Download traces** to your local machine:
   ```bash
   make download-traces h=<remote_host> k=<keyfile>
   ```

3. **View locally** with xprof:
   ```bash
   make xprof-serve
   # Open http://localhost:8791 in your browser
   ```

#### Option 2: SSH Tunnel (Real-time)

View traces in real-time while connected to the remote machine.

1. **On the remote GPU machine**, run profiling and start the server:
   ```bash
   python scripts/profile_attention.py --seq-len 1024 --backward
   python scripts/profile_attention.py --serve
   ```

2. **SSH tunnel** from your local machine:
   ```bash
   make xprof-tunnel h=<remote_host> k=<keyfile>
   ```

3. **Open** http://localhost:8791 in your browser.

### Makefile Commands

```bash
make download-traces h=<host> k=<keyfile>  # Download traces from remote
make xprof-serve                            # Start xprof server locally
make xprof-list                             # List available traces
make xprof-tunnel h=<host> k=<keyfile>      # SSH tunnel for remote viewing
```

## Roofline Analysis

Generate roofline plots to analyze performance characteristics of attention implementations.

### Generate New Benchmarks

```bash
make roofline batch=4 heads=8 head-dim=64 seq-lengths=128,256,512,1024,2048,4096
```

This runs benchmarks and saves:
- `traces/roofline_data_{GPU_MODEL}_{timestamp}.json` - Benchmark data
- `traces/roofline_fwd_{GPU_MODEL}_{timestamp}.png` - Forward pass plot
- `traces/roofline_bwd_{GPU_MODEL}_{timestamp}.png` - Backward pass plot

### Generate Plots from Existing JSON

```bash
make plot-roofline json=traces/roofline_data_NVIDIA_RTX_4000_Ada_20260125_123456.json output-dir=traces
```

Or run directly:

```bash
uv run python scripts/plot_roofline.py traces/roofline_data_*.json --output-dir=traces
```

## Key Patterns

### Pallas Kernels

Use `pl.pallas_call` with `BlockSpec` for input/output tiling:

```python
out = pl.pallas_call(
    kernel_fn,
    out_shape=jax.ShapeDtypeStruct(shape, dtype),
    grid=(batch, num_blocks),
    in_specs=[pl.BlockSpec((BLOCK_M, BLOCK_N), lambda b, i: (b, i, 0))],
    out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda b, i: (b, i, 0)),
)(inputs)
```

### Custom Gradients

Implement `jax.custom_vjp` for kernels requiring custom backward passes:

```python
@jax.custom_vjp
def my_kernel(x):
    return forward(x)

def my_kernel_fwd(x):
    y = forward(x)
    return y, (x, y)  # residuals for backward

def my_kernel_bwd(res, g):
    x, y = res
    return (backward(x, y, g),)

my_kernel.defvjp(my_kernel_fwd, my_kernel_bwd)
```

### Sharding with Flax NNX

```python
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.sharding.Mesh(devices, ('data', 'model'))
sharding = NamedSharding(mesh, P('data', 'model'))

# Apply sharding constraint to intermediates
x = jax.lax.with_sharding_constraint(x, sharding)
```

## Platform Notes

- Set `INTERPRET_MODE = True` for CPU/interpret mode, `False` for GPU execution in Pallas code
- GPU dependencies: `uv sync --extra gpu`
- TPU dependencies: `uv sync --extra tpu`
- Triton kernels only work on Linux with NVIDIA GPUs

## Development

```bash
make dev          # Install with dev dependencies
make lab          # Start Jupyter lab
make lint         # Run linting
make format       # Format code
```
