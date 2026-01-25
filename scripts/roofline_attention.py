#!/usr/bin/env python3
"""Generate roofline plot for attention implementations.

This script benchmarks naive MHA and flash attention across multiple sequence
lengths, measures performance using xProf, and generates a roofline plot
showing compute-bound vs memory-bound regions for RTX 4000 Ada.

Usage:
    python scripts/roofline_attention.py
    make roofline
"""

import sys
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from high_performance_jax.profiling import (
    configure,
    profile_function,
    _get_trace_path,
    _get_default_trace_dir,
)

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Check backend and set interpret mode for CPU
if jax.default_backend() == 'cpu':
    # Modify INTERPRET_MODE for CPU execution
    import high_performance_jax.pallas.pallas_flash_attn as flash_attn_module
    flash_attn_module.INTERPRET_MODE = True
    flash_attention = flash_attn_module.flash_attention
    mha_reference = flash_attn_module.mha_reference
    cudnn_attention = flash_attn_module.cudnn_attention
else:
    from high_performance_jax.pallas.pallas_flash_attn import (
        flash_attention,
        mha_reference,
        cudnn_attention,
    )


# GPU Specifications dictionary mapping GPU_MODEL values to specs
GPU_MODEL_TO_SPECS = {
    # NVIDIA RTX 4000 Ada
    "NVIDIA RTX 4000 Ada Generation": {
        "peak_compute_tflops": 26.7,
        "peak_compute_tflops_tc": 106.91,
        "tensor_tflops_datasheet": 327.6,
        "peak_bandwidth_gb_s": 360.0,
        "ridge_ai": 26.7e3 / 360.0,
        "ridge_ai_tc": 106.91e3 / 360.0,
    },
}


def get_gpu_specs(gpu_model: str) -> dict:
    """Get GPU specifications for given GPU model.

    Args:
        gpu_model: GPU model name (e.g., "NVIDIA RTX 4000 Ada")

    Returns:
        Dictionary with GPU specs including name added from GPU_MODEL_TO_SPECS
    """
    if gpu_model in GPU_MODEL_TO_SPECS:
        specs = GPU_MODEL_TO_SPECS[gpu_model].copy()
        specs["name"] = gpu_model
        return specs

    print(f"Warning: GPU model '{gpu_model}' not found in GPU_MODEL_TO_SPECS")
    print("Available GPU models: " + ", ".join(GPU_MODEL_TO_SPECS.keys()))
    print("Using default RTX 4000 Ada specs - please update GPU_MODEL_TO_SPECS")
    return {
        "name": gpu_model,
        "peak_compute_tflops": 26.7,
        "peak_compute_tflops_tc": 106.91,
        "tensor_tflops_datasheet": 327.6,
        "peak_bandwidth_gb_s": 360.0,
        "ridge_ai": 26.7e3 / 360.0,
        "ridge_ai_tc": 106.91e3 / 360.0,
    }


def calculate_flops_fwd(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for forward attention (same for all implementations).

    Forward:
    - Q @ K.T: B*H*T*T*D * 2 (mult + add per element)
    - Softmax: B*H*T*T * ~5 ops (exp, sum, max, div, sub)
    - P @ V: B*H*T*T*D * 2 (mult + add per element)
    Total forward: 4*B*H*T^2*D + 5*B*H*T^2
    """
    fwd_matmul = 4 * B * H * T * T * D  # 2 matmuls * 2 ops each
    fwd_softmax = 5 * B * H * T * T     # Softmax ops
    return fwd_matmul + fwd_softmax


def calculate_flops_bwd_naive(B: int, H: int, T: int, D: int) -> float:
    """Calculate backward FLOPs for naive attention.

    Naive attention stores the full attention matrix, so backward doesn't
    need to recompute. Uses standard autodiff.
    - dV = P^T @ dO: 2*T²*D
    - dP = dO @ V^T: 2*T²*D
    - dQ = dS @ K: 2*T²*D
    - dK = dS^T @ Q: 2*T²*D
    Total: ~8*B*H*T^2*D
    """
    return 8 * B * H * T * T * D


def calculate_flops_bwd_pallas(B: int, H: int, T: int, D: int) -> float:
    """Calculate backward FLOPs for Pallas flash attention.

    Our Pallas implementation recomputes attention twice in backward:
    - dKV kernel: recomputes S = Q @ K^T, computes dP, dV, dK (4 matmuls)
    - dQ kernel: recomputes S = Q @ K^T, computes dP, dQ (3 matmuls)
    Total: ~14*B*H*T^2*D
    """
    bwd_dkv = 8 * B * H * T * T * D   # S recompute + dP + dV + dK
    bwd_dq = 6 * B * H * T * T * D    # S recompute + dP + dQ
    return bwd_dkv + bwd_dq


def calculate_flops_bwd_cudnn(B: int, H: int, T: int, D: int) -> float:
    """Calculate backward FLOPs for cuDNN flash attention.

    cuDNN uses optimized backward that recomputes attention only once:
    - Single fused backward: recomputes S once, computes dQ, dK, dV
    Total: ~10*B*H*T^2*D
    """
    return 10 * B * H * T * T * D


def calculate_bytes_fwd_naive(B: int, H: int, T: int, D: int, bytes_per_elem: int = 2) -> float:
    """Calculate bytes transferred for naive MHA forward pass.

    Naive MHA materializes full attention matrix:
    - Read Q, K, V: B*H*T*D * 3
    - Write attention matrix: B*H*T*T  <-- THE BIG ONE
    - Write output O: B*H*T*D
    """
    return (
        B * H * T * D * 3 * bytes_per_elem +  # Q, K, V read
        B * H * T * T * bytes_per_elem +       # Attention matrix write
        B * H * T * D * bytes_per_elem          # Output O write
    )


def calculate_bytes_bwd_naive(B: int, H: int, T: int, D: int, bytes_per_elem: int = 2) -> float:
    """Calculate bytes transferred for naive MHA backward pass.

    - Read Q, K, V, O, dO, attention matrix: lots of traffic
    - Write dQ, dK, dV: B*H*T*D * 3
    """
    return (
        B * H * T * D * 5 * bytes_per_elem +  # Q, K, V, O, dO read
        B * H * T * T * bytes_per_elem +       # Attention matrix read
        B * H * T * D * 3 * bytes_per_elem     # dQ, dK, dV write
    )


def calculate_bytes_fwd_flash(B: int, H: int, T: int, D: int, bytes_per_elem: int = 2) -> float:
    """Calculate bytes transferred for flash attention forward pass.

    Flash attention computes in tiles, only stores logsumexp:
    - Read Q, K, V: B*H*T*D * 3
    - Write logsumexp: B*H*T  <-- TINY
    - Write output O: B*H*T*D
    """
    return (
        B * H * T * D * 3 * bytes_per_elem +  # Q, K, V read
        B * H * T * bytes_per_elem +           # logsumexp write
        B * H * T * D * bytes_per_elem          # Output O write
    )


def calculate_bytes_bwd_flash(B: int, H: int, T: int, D: int, bytes_per_elem: int = 2) -> float:
    """Calculate bytes transferred for flash attention backward pass.

    - Read Q, K, V, O, dO, logsumexp
    - Write dQ, dK, dV
    """
    return (
        B * H * T * D * 5 * bytes_per_elem +  # Q, K, V, O, dO read
        B * H * T * bytes_per_elem +           # logsumexp read
        B * H * T * D * 3 * bytes_per_elem     # dQ, dK, dV write
    )


def benchmark_attention(
    B: int, H: int, T: int, D: int,
    dtype=jnp.float16,
    warmup_iters: int = 3,
    profile_iters: int = 5,
) -> dict:
    """Benchmark a single configuration.

    Returns:
        Dict with 'naive', 'flash', and 'cudnn' results containing separate
        forward and backward metrics (time, gflops_s, ai, bw_gb_s).
    """
    print(f"\nBenchmarking: B={B}, H={H}, T={T}, D={D}")

    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=dtype)

    # Calculate FLOPs and bytes for this configuration (separate fwd/bwd)
    bytes_per_elem = 2 if dtype == jnp.float16 else 4

    # Forward FLOPs (same for all implementations)
    flops_fwd = calculate_flops_fwd(B, H, T, D)

    # Backward FLOPs (different per implementation)
    flops_bwd_naive = calculate_flops_bwd_naive(B, H, T, D)
    flops_bwd_pallas = calculate_flops_bwd_pallas(B, H, T, D)
    flops_bwd_cudnn = calculate_flops_bwd_cudnn(B, H, T, D)

    # Forward bytes
    bytes_fwd_naive = calculate_bytes_fwd_naive(B, H, T, D, bytes_per_elem)
    bytes_fwd_flash = calculate_bytes_fwd_flash(B, H, T, D, bytes_per_elem)

    # Backward bytes
    bytes_bwd_naive = calculate_bytes_bwd_naive(B, H, T, D, bytes_per_elem)
    bytes_bwd_flash = calculate_bytes_bwd_flash(B, H, T, D, bytes_per_elem)

    def _bench(fn, warmup=warmup_iters, iters=profile_iters):
        """Benchmark a function, return median time in seconds."""
        for _ in range(warmup):
            out = fn()
            jax.block_until_ready(out)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return np.median(times)

    # Create jitted forward functions
    naive_fwd = jax.jit(mha_reference)
    flash_fwd = jax.jit(flash_attention)
    cudnn_fwd = jax.jit(cudnn_attention)

    # Get vjp functions for backward pass and benchmark
    print("  Warming up and timing naive MHA...")
    _, naive_vjp = jax.vjp(mha_reference, q, k, v)
    naive_fwd_time = _bench(lambda: naive_fwd(q, k, v))
    naive_bwd_time = _bench(lambda: naive_vjp(do))

    print("  Warming up and timing flash attention...")
    _, flash_vjp = jax.vjp(flash_attention, q, k, v)
    flash_fwd_time = _bench(lambda: flash_fwd(q, k, v))
    flash_bwd_time = _bench(lambda: flash_vjp(do))

    print("  Warming up and timing cuDNN attention...")
    _, cudnn_vjp = jax.vjp(cudnn_attention, q, k, v)
    cudnn_fwd_time = _bench(lambda: cudnn_fwd(q, k, v))
    cudnn_bwd_time = _bench(lambda: cudnn_vjp(do))

    # Calculate metrics for each pass
    def calc_metrics(time_s, flops, bytes_transferred):
        return {
            "time_ms": time_s * 1000,
            "gflops_s": flops / (time_s * 1e9),
            "ai": flops / bytes_transferred,
            "bw_gb_s": bytes_transferred / (time_s * 1e9),
        }

    naive_fwd_metrics = calc_metrics(naive_fwd_time, flops_fwd, bytes_fwd_naive)
    naive_bwd_metrics = calc_metrics(naive_bwd_time, flops_bwd_naive, bytes_bwd_naive)
    flash_fwd_metrics = calc_metrics(flash_fwd_time, flops_fwd, bytes_fwd_flash)
    flash_bwd_metrics = calc_metrics(flash_bwd_time, flops_bwd_pallas, bytes_bwd_flash)
    cudnn_fwd_metrics = calc_metrics(cudnn_fwd_time, flops_fwd, bytes_fwd_flash)
    cudnn_bwd_metrics = calc_metrics(cudnn_bwd_time, flops_bwd_cudnn, bytes_bwd_flash)

    # Print results
    naive_total = naive_fwd_time + naive_bwd_time
    flash_total = flash_fwd_time + flash_bwd_time
    cudnn_total = cudnn_fwd_time + cudnn_bwd_time

    print(f"  Naive:  fwd {naive_fwd_metrics['time_ms']:.3f}ms ({naive_fwd_metrics['gflops_s']:.0f} GFLOP/s, AI={naive_fwd_metrics['ai']:.0f}) | "
          f"bwd {naive_bwd_metrics['time_ms']:.3f}ms ({naive_bwd_metrics['gflops_s']:.0f} GFLOP/s, AI={naive_bwd_metrics['ai']:.0f})")
    print(f"  Flash:  fwd {flash_fwd_metrics['time_ms']:.3f}ms ({flash_fwd_metrics['gflops_s']:.0f} GFLOP/s, AI={flash_fwd_metrics['ai']:.0f}) | "
          f"bwd {flash_bwd_metrics['time_ms']:.3f}ms ({flash_bwd_metrics['gflops_s']:.0f} GFLOP/s, AI={flash_bwd_metrics['ai']:.0f})")
    print(f"  cuDNN:  fwd {cudnn_fwd_metrics['time_ms']:.3f}ms ({cudnn_fwd_metrics['gflops_s']:.0f} GFLOP/s, AI={cudnn_fwd_metrics['ai']:.0f}) | "
          f"bwd {cudnn_bwd_metrics['time_ms']:.3f}ms ({cudnn_bwd_metrics['gflops_s']:.0f} GFLOP/s, AI={cudnn_bwd_metrics['ai']:.0f})")
    print(f"  Speedup (flash vs naive): {naive_total/flash_total:.2f}x")
    print(f"  Speedup (cuDNN vs naive): {naive_total/cudnn_total:.2f}x")

    return {
        "naive": {
            "fwd": naive_fwd_metrics,
            "bwd": naive_bwd_metrics,
        },
        "flash": {
            "fwd": flash_fwd_metrics,
            "bwd": flash_bwd_metrics,
        },
        "cudnn": {
            "fwd": cudnn_fwd_metrics,
            "bwd": cudnn_bwd_metrics,
        },
        "speedup": naive_total / flash_total,
        "speedup_cudnn": naive_total / cudnn_total,
    }


def generate_roofline_plot(
    results: dict,
    pass_type: str = "fwd",
    gpu_model: str = None,
    dtype: str = None,
    output_path: Path | None = None,
):
    """Generate roofline plot for forward or backward pass.

    Args:
        results: Dict with 'sequence_lengths', 'naive', 'flash', 'cudnn' data
        pass_type: "fwd" for forward pass, "bwd" for backward pass
        gpu_model: GPU model name (e.g., "NVIDIA RTX 4000 Ada")
        dtype: Data type ("float16" or "float32") - determines which ridge to show
        output_path: Path to save PNG plot
    """
    if gpu_model is None:
        gpu_model = os.environ.get("GPU_MODEL", "NVIDIA RTX 4000 Ada")

    if dtype is None:
        dtype = results.get("config", {}).get("dtype", "float16")

    gpu = get_gpu_specs(gpu_model)
    pass_name = "Forward" if pass_type == "fwd" else "Backward"

    # Determine which ridge to use based on dtype
    if dtype == "float32":
        ridge_ai = gpu['ridge_ai']
        ridge_label = "Ridge (FP32)"
    else:
        ridge_ai = gpu['ridge_ai_tc']
        ridge_label = "Ridge (BF16/FP16)"

    seq_lengths = np.array(results["sequence_lengths"])

    # Extract metrics for the specified pass
    naive_ai = np.array([r["ai"] for r in results["naive"][pass_type]])
    flash_ai = np.array([r["ai"] for r in results["flash"][pass_type]])
    cudnn_ai = np.array([r["ai"] for r in results["cudnn"][pass_type]])
    naive_perf = np.array([r["gflops_s"] for r in results["naive"][pass_type]])
    flash_perf = np.array([r["gflops_s"] for r in results["flash"][pass_type]])
    cudnn_perf = np.array([r["gflops_s"] for r in results["cudnn"][pass_type]])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine axis range based on actual data
    all_ai = np.concatenate([naive_ai, flash_ai, cudnn_ai])
    all_perf = np.concatenate([naive_perf, flash_perf, cudnn_perf])
    ai_min = min(all_ai.min(), ridge_ai) / 2
    ai_max = max(all_ai.max(), ridge_ai) * 2

    # Roofline calculations
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 100)

    # Memory roof (diagonal)
    # GB/s * FLOPs/byte = 10^9 bytes/s * FLOPs/byte = 10^9 FLOPs/s = GFLOP/s
    memory_roof = gpu["peak_bandwidth_gb_s"] * ai_range

    # Compute roof (horizontal) - use appropriate peak based on dtype
    if dtype == "float32":
        compute_roof = gpu["peak_compute_tflops"] * 1000 * np.ones_like(ai_range)
        compute_label = f'FP32 roof ({gpu["peak_compute_tflops"]:.1f} TFLOP/s)'
    else:
        compute_roof = gpu["peak_compute_tflops_tc"] * 1000 * np.ones_like(ai_range)
        compute_label = f'FP16 TC roof ({gpu["peak_compute_tflops_tc"]:.1f} TFLOP/s)'

    # Cap memory roof at compute roof
    memory_roof = np.minimum(memory_roof, compute_roof)

    # Plot roofs
    ax.plot(ai_range, memory_roof, 'k--', linewidth=2, alpha=0.7, label='Memory roof')
    ax.plot(ai_range, compute_roof, 'r--', linewidth=2, alpha=0.7, label=compute_label)

    # Plot actual performance
    ax.scatter(naive_ai, naive_perf, marker='o', s=150, c='red',
               edgecolors='black', linewidth=1.5, label='Naive MHA', zorder=5)
    ax.scatter(flash_ai, flash_perf, marker='s', s=150, c='blue',
               edgecolors='black', linewidth=1.5, label='Flash (Pallas)', zorder=5)
    ax.scatter(cudnn_ai, cudnn_perf, marker='^', s=150, c='green',
               edgecolors='black', linewidth=1.5, label='cuDNN Flash', zorder=5)

    # Annotate sequence lengths (only for largest/smallest to avoid clutter)
    for i, (ai, perf, T) in enumerate(zip(naive_ai, naive_perf, seq_lengths)):
        if i == 0 or i == len(seq_lengths) - 1:
            ax.annotate(f'T={T}', (ai, perf), fontsize=8,
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom')
    for i, (ai, perf, T) in enumerate(zip(flash_ai, flash_perf, seq_lengths)):
        if i == 0 or i == len(seq_lengths) - 1:
            ax.annotate(f'T={T}', (ai, perf), fontsize=8,
                        xytext=(0, -15), textcoords='offset points',
                        ha='center', va='top')
    for i, (ai, perf, T) in enumerate(zip(cudnn_ai, cudnn_perf, seq_lengths)):
        if i == 0 or i == len(seq_lengths) - 1:
            ax.annotate(f'T={T}', (ai, perf), fontsize=8,
                        xytext=(10, 0), textcoords='offset points',
                        ha='left', va='center')

    # Ridge point annotation
    ridge_perf_fp32 = gpu["peak_compute_tflops"] * 1000
    ax.axvline(ridge_ai, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai, ridge_perf_fp32 * 0.1, f'  {ridge_label}\n  AI={ridge_ai:.1f}',
            fontsize=10, rotation=90, va='bottom', ha='right')

    # Region annotations
    ax.text(ridge_ai / 3, ridge_perf_fp32 * 1.2, 'Memory-Bound\n(AI < Ridge)',
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.text(ridge_ai * 3, ridge_perf_fp32 * 1.2, 'Compute-Bound\n(AI > Ridge)',
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Labels and styling
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis limits based on data range
    ax.set_xlim(ai_min, ai_max)
    perf_min = all_perf.min() / 2
    perf_max = max(all_perf.max(), compute_roof[0]) * 1.5
    ax.set_ylim(perf_min, perf_max)

    ax.set_title(f'Roofline Analysis ({pass_name} Pass, {dtype.upper()}): Naive vs Flash (Pallas) vs cuDNN\n{gpu["name"]}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Add GPU specs text box
    specs_text_lines = [
        "GPU Specifications:",
        f"  CUDA peak: {gpu['peak_compute_tflops']:.1f} TFLOP/s",
        f"  TC peak: {gpu['peak_compute_tflops_tc']:.1f} TFLOP/s",
        f"  Bandwidth: {gpu['peak_bandwidth_gb_s']:.1f} GB/s",
    ]
    specs_text = "\n".join(specs_text_lines)

    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_filename_safe = gpu_model.replace(" ", "_").replace("/", "_")
        output_path = _get_default_trace_dir() / f"roofline_{pass_type}_{gpu_filename_safe}_{timestamp}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{pass_name} pass plot saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate roofline plot for attention")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--seq-lengths", type=str, default="128,256,512,1024,2048,4096",
                       help="Comma-separated sequence lengths (x2 scaling)")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "float32"], help="Data type")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Profile iterations")
    parser.add_argument("--trace-dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Configure trace directory
    if args.trace_dir:
        configure(trace_dir=args.trace_dir)

    # Get GPU model from environment variable
    gpu_model = os.environ.get("GPU_MODEL", "NVIDIA RTX 4000 Ada")
    print(f"GPU Model: {gpu_model}")

    dtype = jnp.float16 if args.dtype == "float16" else jnp.float32
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    print("=" * 70)
    print("ROOFLINE ANALYSIS FOR ATTENTION")
    print("=" * 70)

    # Print GPU info using specs from GPU_MODEL_TO_SPECS
    gpu = get_gpu_specs(gpu_model)
    print(f"\nGPU: {gpu['name']}")
    print(f"  Peak Compute (CUDA cores):   {gpu['peak_compute_tflops']:.1f} TFLOP/s")
    print(f"  Peak Compute (Tensor cores): {gpu['peak_compute_tflops_tc']:.1f} TFLOP/s (observed)")
    print(f"  Peak Memory Bandwidth:       {gpu['peak_bandwidth_gb_s']:.1f} GB/s")
    print(f"  Ridge AI (CUDA):             {gpu['ridge_ai']:.1f} FLOPs/byte")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Batch size:      {args.batch}")
    print(f"  Number of heads: {args.heads}")
    print(f"  Head dimension:  {args.head_dim}")
    print(f"  Data type:       {args.dtype}")
    print(f"  Sequence lengths: {seq_lengths}")

    # Check backend
    print(f"\nBackend: {jax.default_backend()}")
    if jax.default_backend() != 'gpu':
        print("\n" + "!" * 70)
        print("WARNING: JAX backend is NOT 'gpu'!")
        print("Roofline analysis requires GPU backend for accurate results.")
        print("CPU backend will show poor performance and invalid roofline comparison.")
        print("Set JAX backend with: JAX_PLATFORMS=gpu <command>")
        print("!" * 70)
        print()

    print("\n" + "=" * 70)
    print("BENCHMARKING")
    print("=" * 70)

    # Benchmark across sequence lengths
    results = {
        "gpu_model": gpu_model,
        "config": {
            "B": args.batch,
            "H": args.heads,
            "D": args.head_dim,
            "dtype": args.dtype,
        },
        "sequence_lengths": seq_lengths,
        "naive": {"fwd": [], "bwd": []},
        "flash": {"fwd": [], "bwd": []},
        "cudnn": {"fwd": [], "bwd": []},
        "speedups": [],
        "speedups_cudnn": [],
    }

    for T in seq_lengths:
        benchmark_result = benchmark_attention(
            args.batch, args.heads, T, args.head_dim,
            dtype=dtype, warmup_iters=args.warmup, profile_iters=args.iters
        )

        results["naive"]["fwd"].append(benchmark_result["naive"]["fwd"])
        results["naive"]["bwd"].append(benchmark_result["naive"]["bwd"])
        results["flash"]["fwd"].append(benchmark_result["flash"]["fwd"])
        results["flash"]["bwd"].append(benchmark_result["flash"]["bwd"])
        results["cudnn"]["fwd"].append(benchmark_result["cudnn"]["fwd"])
        results["cudnn"]["bwd"].append(benchmark_result["cudnn"]["bwd"])
        results["speedups"].append(benchmark_result["speedup"])
        results["speedups_cudnn"].append(benchmark_result["speedup_cudnn"])

    # Print summary tables
    print("\n" + "=" * 120)
    print("FORWARD PASS SUMMARY")
    print("=" * 120)
    print(f"{'T':<6} {'Naive':<10} {'Flash':<10} {'cuDNN':<10} "
          f"{'Naive':<12} {'Flash':<12} {'cuDNN':<12} "
          f"{'Naive':<10} {'Flash':<10} {'cuDNN':<10}")
    print(f"{'':<6} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} "
          f"{'(GFLOP/s)':<12} {'(GFLOP/s)':<12} {'(GFLOP/s)':<12} "
          f"{'(AI)':<10} {'(AI)':<10} {'(AI)':<10}")
    print("-" * 120)

    for i, T in enumerate(seq_lengths):
        print(f"{T:<6} "
              f"{results['naive']['fwd'][i]['time_ms']:<10.3f} "
              f"{results['flash']['fwd'][i]['time_ms']:<10.3f} "
              f"{results['cudnn']['fwd'][i]['time_ms']:<10.3f} "
              f"{results['naive']['fwd'][i]['gflops_s']:<12.0f} "
              f"{results['flash']['fwd'][i]['gflops_s']:<12.0f} "
              f"{results['cudnn']['fwd'][i]['gflops_s']:<12.0f} "
              f"{results['naive']['fwd'][i]['ai']:<10.0f} "
              f"{results['flash']['fwd'][i]['ai']:<10.0f} "
              f"{results['cudnn']['fwd'][i]['ai']:<10.0f}")

    print("\n" + "=" * 120)
    print("BACKWARD PASS SUMMARY")
    print("=" * 120)
    print(f"{'T':<6} {'Naive':<10} {'Flash':<10} {'cuDNN':<10} "
          f"{'Naive':<12} {'Flash':<12} {'cuDNN':<12} "
          f"{'Naive':<10} {'Flash':<10} {'cuDNN':<10}")
    print(f"{'':<6} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} "
          f"{'(GFLOP/s)':<12} {'(GFLOP/s)':<12} {'(GFLOP/s)':<12} "
          f"{'(AI)':<10} {'(AI)':<10} {'(AI)':<10}")
    print("-" * 120)

    for i, T in enumerate(seq_lengths):
        print(f"{T:<6} "
              f"{results['naive']['bwd'][i]['time_ms']:<10.3f} "
              f"{results['flash']['bwd'][i]['time_ms']:<10.3f} "
              f"{results['cudnn']['bwd'][i]['time_ms']:<10.3f} "
              f"{results['naive']['bwd'][i]['gflops_s']:<12.0f} "
              f"{results['flash']['bwd'][i]['gflops_s']:<12.0f} "
              f"{results['cudnn']['bwd'][i]['gflops_s']:<12.0f} "
              f"{results['naive']['bwd'][i]['ai']:<10.0f} "
              f"{results['flash']['bwd'][i]['ai']:<10.0f} "
              f"{results['cudnn']['bwd'][i]['ai']:<10.0f}")

    avg_speedup = np.mean(results["speedups"])
    avg_speedup_cudnn = np.mean(results["speedups_cudnn"])
    print(f"\nAverage total speedup (flash vs naive):  {avg_speedup:.2f}x")
    print(f"Average total speedup (cuDNN vs naive):  {avg_speedup_cudnn:.2f}x")

    # Verify GFLOPs/s against peak compute
    max_flash_fwd = max(r["gflops_s"] for r in results["flash"]["fwd"])
    max_flash_bwd = max(r["gflops_s"] for r in results["flash"]["bwd"])
    max_cudnn_fwd = max(r["gflops_s"] for r in results["cudnn"]["fwd"])
    max_cudnn_bwd = max(r["gflops_s"] for r in results["cudnn"]["bwd"])
    peak_fp32 = gpu["peak_compute_tflops"] * 1000
    peak_tc = gpu["peak_compute_tflops_tc"] * 1000

    print(f"\nPeak verification:")
    print(f"  GPU FP32 Peak:     {peak_fp32:.0f} GFLOP/s ({gpu['peak_compute_tflops']:.1f} TFLOP/s)")
    print(f"  GPU FP16 TC Peak:  {peak_tc:.0f} GFLOP/s ({gpu['peak_compute_tflops_tc']:.1f} TFLOP/s)")
    print(f"  Pallas fwd: {max_flash_fwd:.0f} GFLOP/s ({max_flash_fwd/peak_fp32*100:.1f}% of CUDA peak)")
    print(f"  Pallas bwd: {max_flash_bwd:.0f} GFLOP/s ({max_flash_bwd/peak_fp32*100:.1f}% of CUDA peak)")
    print(f"  cuDNN fwd:  {max_cudnn_fwd:.0f} GFLOP/s ({max_cudnn_fwd/peak_tc*100:.1f}% of TC peak)")
    print(f"  cuDNN bwd:  {max_cudnn_bwd:.0f} GFLOP/s ({max_cudnn_bwd/peak_tc*100:.1f}% of TC peak)")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_filename_safe = gpu_model.replace(" ", "_").replace("/", "_")
    json_path = _get_default_trace_dir() / f"roofline_data_{gpu_filename_safe}_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate separate forward and backward roofline plots
    if not args.no_plot:
        fwd_plot = generate_roofline_plot(results, pass_type="fwd", gpu_model=gpu_model, dtype=args.dtype)
        bwd_plot = generate_roofline_plot(results, pass_type="bwd", gpu_model=gpu_model, dtype=args.dtype)
        print(f"\nDone! View plots at:")
        print(f"  Forward:  {fwd_plot}")
        print(f"  Backward: {bwd_plot}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
