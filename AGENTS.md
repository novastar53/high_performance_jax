# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

This is a tutorials and recipes repository for achieving high performance with JAX. It contains educational implementations of performance-critical components for machine learning: custom GPU kernels, model sharding, attention mechanisms, and more.

## Development Setup

Uses `uv` for Python package management with platform-specific JAX installations (GPU, TPU, or Metal).

```bash
# Install dependencies (auto-detects platform)
make install

# Development install with Jupyter kernel
make dev

# Start Jupyter lab
make lab

# SSH tunnel for remote Jupyter (usage: make jupyter-ssh-tunnel h=hostname k=keyfile)
make jupyter-ssh-tunnel h=${h} k=${k}
```

## Code Architecture

### Source Structure (`src/high_performance_jax/`)

**Pallas Kernels** (`pallas/`): Custom JAX kernels using the Pallas DSL for GPU/TPU
- `pallas_softmax.py` - Online softmax with custom forward/backward passes via `jax.custom_vjp`
- `pallas_flash_attn.py`, `pallas_flash_attn_ref.py` - Flash attention implementations
- `pallas_matmul.py`, `pallas_triton_matmul.py` - Matrix multiplication kernels

**Triton Kernels** (`triton/`): Native Triton GPU kernels (Linux GPU only)
- `flash_attention.py` - Full flash attention with forward and backward passes using `torch.autograd.Function`

**Sharding & Distributed** (root level):
- `model_sharding.py` - Flax NNX model sharding with `NamedSharding` and `MeshRules`
- `moe.py` - Mixture of Experts with expert parallelism using `jax.lax.with_sharding_constraint`
- `sharded_vmap.py` - Combining vmap with sharding

**CUDA Examples** (`cuda/`): Raw CUDA kernels for learning purposes
- Build with `make -C src/high_performance_jax/cuda`

### Notebooks (`notebooks/`)

Interactive tutorials covering sharding, attention, numerics, and kernel development. The `pallas-softmax.ipynb` notebook accompanies the Pallas softmax blogpost.

## Key Patterns

**Pallas kernel structure**: Use `pl.pallas_call` with `BlockSpec` for input/output tiling, `plgpu.load/store` for memory access, and `jax.lax.fori_loop` for iteration within kernels.

**Sharding with Flax NNX**: Define `MeshRules` dataclass mapping logical names to mesh axes. Apply sharding via `nnx.with_partitioning` on initializers and `jax.lax.with_sharding_constraint` on intermediates.

**Custom gradients**: Implement `jax.custom_vjp` with `defvjp(fwd, bwd)` for kernels requiring custom backward passes (see `pallas_softmax.py`).

## Platform Notes

- Set `INTERPRET_MODE = True` for CPU/interpret mode, `False` for GPU execution in Pallas code
- GPU dependencies require the `gpu` extra: `uv sync --extra gpu`
- TPU dependencies require the `tpu` extra with a specific libtpu wheel
- Triton kernels only work on Linux with NVIDIA GPUs

## Running Python Scripts

**IMPORTANT**: Always use `uv run` instead of `python` when running scripts that import from the `src/` directory:

```bash
# Correct - uses project's virtual environment with JAX installed
uv run python scripts/my_script.py

# Incorrect - uses system Python which may not have JAX or wrong version
python scripts/my_script.py
```

This is critical for:
- Scripts that import `high_performance_jax` (which needs `uv sync` dependencies)
- Scripts that need correct JAX platform detection (GPU/TPU/Metal)
- Benchmarking and profiling scripts that require proper JAX backend

**Set backend explicitly if needed**:
```bash
# Force GPU backend
JAX_PLATFORMS=gpu uv run python scripts/my_script.py

# Force CPU backend
JAX_PLATFORMS=cpu uv run python scripts/my_script.py
```

## Blog Writing Guidelines

When writing content for blogposts or notebooks, follow these style guidelines:

1. **Use short paragraphs** - Keep paragraphs concise for better readability.

2. **Avoid bullets and lists** - Write paragraphs instead of using bullet points or lists. Exception: Explicitly sequential steps like algorithms or processes.

3. **Avoid contrastive phrasing** - Don't use constructions like "it's not this - it's that." State what something is directly.

4. **Avoid emojis** - Don't use emojis in technical content.

5. **Use LaTeX for math** - Write mathematical expressions using LaTeX syntax ($$...$$ or $...$).

6. **Don't use backticks for variable names** - Write variable names inline without code formatting (e.g., use flash_attention_fwd instead of `flash_attention_fwd`).

7. **Use LaTeX for matrix dimensions** - write $(B, T, C)$ rather than `(B, T, C)` and write $B$ rather than `B` or B'.

8. **Use LaTeX for matrix math** - instead of A@B^T or `A@B^T`, prefer using latex notation $AB^T$ instead.

## Plotting Guidelines

When creating matplotlib plots for notebooks or blog posts, follow these style guidelines:

1. **Use default matplotlib fontsizes** - Do not specify explicit `fontsize=` parameters in `set_xlabel()`, `set_ylabel()`, `set_title()`, `legend()`, `annotate()`, or `text()` calls. Let matplotlib use its default sizes.

2. **Avoid bold fonts** - Do not use `fontweight='bold'` on labels or titles. Keep text at normal weight.

3. **Use consistent figure sizes**:
   - For single plots: `figsize=(10, 4)` or use matplotlib default
   - For subplots with 1 row, 2 columns: `figsize=(10, 4)`
   - Do not use `suptitle()` for overall titles - titles should be per-plot

4. **Keep styling minimal** - Let matplotlib's defaults and the seaborn-v0_8-darkgrid style do the work. Avoid overly-customized styling that makes plots look cluttered.

5. **Reference style** - Look at `how-residual-connections-work.ipynb` in the blog repo for examples of the clean, minimalist plotting style to emulate.

Example correct plotting pattern:
```python
fig, ax = plt.subplots(figsize=(10, 4)
ax.plot(x, y, 'o-', label='Series A')
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')
ax.set_title('Plot Title')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
```
