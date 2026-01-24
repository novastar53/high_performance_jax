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
import time
from datetime import datetime
from pathlib import Path

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

# Check backend and set interpret mode for CPU
if jax.default_backend() == 'cpu':
    # Modify INTERPRET_MODE for CPU execution
    import high_performance_jax.pallas.pallas_flash_attn as flash_attn_module
    flash_attn_module.INTERPRET_MODE = True
    flash_attention = flash_attn_module.flash_attention
else:
    from high_performance_jax.pallas.pallas_flash_attn import flash_attention


# GPU Specifications for RTX 4000 Ada
GPU_SPECS = {
    "rtx4000-ada": {
        "name": "NVIDIA RTX 4000 Ada",
        "peak_compute_tflops": 26.7,  # FP16 with tensor cores
        "peak_bandwidth_gb_s": 360.0,  # GDDR6
        "ridge_ai": 26.7e3 / 360.0,  # TFLOPs/s / GB/s -> FLOPs/byte
    }
}


def mha_reference(q, k, v):
    """Reference multi-head attention (materializes N×N matrix)."""
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', probs, v)


def calculate_flops(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for forward+backward attention.

    Forward:
    - Q @ K.T: B*H*T*T*D * 2 (mult + add per element)
    - Softmax: B*H*T*T * ~5 ops (exp, sum, max, div, sub)
    - Attention @ V: B*H*T*T*D * 2 (mult + add per element)

    Backward: ~2.5x forward (gradient computation more expensive)

    Total: 2.5 * (2 * B * H * T^2 * D + B * H * T^2 * 5)
    """
    fwd_matmul = 2 * B * H * T * T * D  # Q@K + Attention@V
    fwd_softmax = 5 * B * H * T * T        # Softmax ops
    total_fwd = fwd_matmul + fwd_softmax
    total = 2.5 * total_fwd  # Backward ~2.5x forward
    return total


def calculate_bytes_naive(B: int, H: int, T: int, D: int) -> float:
    """Calculate bytes transferred for naive MHA (fwd+bwd).

    Naive MHA materializes full attention matrix:
    - Q, K, V (input): B*H*T*D * 3 * 4 bytes
    - Attention matrix (T×T): B*H*T*T * 4 bytes  <-- THE BIG ONE
    - Output O: B*H*T*D * 4 bytes

    Backward: similar traffic plus gradients, approximately 2x forward
    """
    fwd_bytes = (
        B * H * T * D * 3 * 4 +  # Q, K, V
        B * H * T * T * 4 +       # Attention matrix
        B * H * T * D * 4          # Output O
    )
    # Backward roughly doubles memory traffic (gradients similar size)
    return fwd_bytes * 2


def calculate_bytes_flash(B: int, H: int, T: int, D: int) -> float:
    """Calculate bytes transferred for flash attention (fwd+bwd).

    Flash attention computes in tiles, only stores logsumexp:
    - Q, K, V (input): B*H*T*D * 3 * 4 bytes
    - logsumexp: B*H*T * 4 bytes  <-- TINY RESIDUAL
    - Output O: B*H*T*D * 4 bytes

    Backward: similar traffic plus gradients, approximately 2x forward
    """
    fwd_bytes = (
        B * H * T * D * 3 * 4 +  # Q, K, V
        B * H * T * 4 +             # logsumexp
        B * H * T * D * 4            # Output O
    )
    # Backward roughly doubles memory traffic (gradients similar size)
    return fwd_bytes * 2


def benchmark_attention(
    B: int, H: int, T: int, D: int,
    dtype=jnp.float16,
    warmup_iters: int = 3,
    profile_iters: int = 5,
) -> dict:
    """Benchmark a single configuration.

    Returns:
        Dict with 'naive' and 'flash' results containing:
        - time_ms: average execution time
        - gflops_s: achieved GFLOPs/s
        - ai: arithmetic intensity (FLOPs/byte)
        - bw_gb_s: effective bandwidth
    """
    print(f"\nBenchmarking: B={B}, H={H}, T={T}, D={D}")

    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=dtype)

    # Calculate FLOPs and bytes for this configuration
    flops = calculate_flops(B, H, T, D)
    bytes_naive = calculate_bytes_naive(B, H, T, D)
    bytes_flash = calculate_bytes_flash(B, H, T, D)

    # Time naive MHA
    def ref_loss(q, k, v):
        return jnp.sum(mha_reference(q, k, v) * do)

    ref_fwd_bwd = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1, 2)))

    # Warm up naive
    print("  Warming up naive MHA...")
    for _ in range(warmup_iters):
        _, grads = ref_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)

    # Time naive MHA (more iterations for accuracy)
    print("  Timing naive MHA...")
    naive_times = []
    for _ in range(profile_iters):
        t0 = time.perf_counter()
        _, grads = ref_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)
        naive_times.append(time.perf_counter() - t0)

    # Use median for robustness (ignore outliers)
    naive_time_s = np.median(naive_times)
    naive_time_ms = naive_time_s * 1000
    naive_gflops_s = flops / (naive_time_s * 1e9)
    naive_ai = flops / bytes_naive
    naive_bw_gb_s = bytes_naive / (naive_time_s * 1e9)

    # Time flash attention
    def flash_loss(q, k, v):
        return jnp.sum(flash_attention(q, k, v) * do)

    flash_fwd_bwd = jax.jit(jax.value_and_grad(flash_loss, argnums=(0, 1, 2)))

    # Warm up flash attention
    print("  Warming up flash attention...")
    for _ in range(warmup_iters):
        _, grads = flash_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)

    # Time flash attention (more iterations for accuracy)
    print("  Timing flash attention...")
    flash_times = []
    for _ in range(profile_iters):
        t0 = time.perf_counter()
        _, grads = flash_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)
        flash_times.append(time.perf_counter() - t0)

    # Use median for robustness (ignore outliers)
    flash_time_s = np.median(flash_times)
    flash_time_ms = flash_time_s * 1000
    flash_gflops_s = flops / (flash_time_s * 1e9)
    flash_ai = flops / bytes_flash
    flash_bw_gb_s = bytes_flash / (flash_time_s * 1e9)

    speedup = naive_time_ms / flash_time_ms

    print(f"  Naive:  {naive_time_ms:.3f} ms, {naive_gflops_s:.2f} GFLOP/s, AI={naive_ai:.1f}")
    print(f"  Flash:   {flash_time_ms:.3f} ms, {flash_gflops_s:.2f} GFLOP/s, AI={flash_ai:.1f}")
    print(f"  Speedup: {speedup:.2f}x")

    return {
        "naive": {
            "time_ms": naive_time_ms,
            "gflops_s": naive_gflops_s,
            "ai": naive_ai,
            "bw_gb_s": naive_bw_gb_s,
        },
        "flash": {
            "time_ms": flash_time_ms,
            "gflops_s": flash_gflops_s,
            "ai": flash_ai,
            "bw_gb_s": flash_bw_gb_s,
        },
        "speedup": speedup,
    }


def generate_roofline_plot(
    results: dict,
    gpu_key: str = "rtx4000-ada",
    output_path: Path | None = None,
):
    """Generate roofline plot.

    Args:
        results: Dict with 'sequence_lengths', 'naive', 'flash' data
        gpu_key: Key for GPU specs in GPU_SPECS
        output_path: Path to save PNG plot
    """
    gpu = GPU_SPECS[gpu_key]

    seq_lengths = np.array(results["sequence_lengths"])
    naive_ai = np.array(results["naive"]["ai"])
    flash_ai = np.array(results["flash"]["ai"])
    naive_perf = np.array(results["naive"]["gflops_s"])
    flash_perf = np.array(results["flash"]["gflops_s"])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Roofline calculations
    ai_range = np.logspace(0.5, 4, 100)  # 3.16 to 10000 FLOPs/byte

    # Memory roof (diagonal)
    memory_roof = gpu["peak_bandwidth_gb_s"] * ai_range / 1000.0  # GB/s -> GFLOP/s

    # Compute roof (horizontal)
    compute_roof = gpu["peak_compute_tflops"] * np.ones_like(ai_range)

    # Ridge point
    ridge_ai = gpu["ridge_ai"]

    # Plot roofs
    ax.plot(ai_range, memory_roof, 'k--', linewidth=2, alpha=0.7, label='Memory roof')
    ax.plot(ai_range, compute_roof, 'r--', linewidth=2, alpha=0.7, label='Compute roof')

    # Plot actual performance
    ax.scatter(naive_ai, naive_perf, marker='o', s=150, c='red',
               edgecolors='black', linewidth=1.5, label='Naive MHA', zorder=5)
    ax.scatter(flash_ai, flash_perf, marker='s', s=150, c='blue',
               edgecolors='black', linewidth=1.5, label='Flash Attention', zorder=5)

    # Annotate sequence lengths
    for i, (ai, perf, T) in enumerate(zip(naive_ai, naive_perf, seq_lengths)):
        ax.annotate(f'T={T}', (ai, perf), fontsize=8,
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')
    for i, (ai, perf, T) in enumerate(zip(flash_ai, flash_perf, seq_lengths)):
        ax.annotate(f'T={T}', (ai, perf), fontsize=8,
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')

    # Ridge point annotation
    ridge_perf = gpu["peak_compute_tflops"]
    ax.axvline(ridge_ai, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai, ridge_perf * 0.1, f'  Ridge\n  AI={ridge_ai:.1f}',
            fontsize=10, rotation=90, va='bottom', ha='right')

    # Region annotations
    ax.text(ai_range[0] * 2, ridge_perf * 1.2, 'Memory-Bound\n(AI < Ridge)',
            fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.text(ai_range[-1] * 0.4, ridge_perf * 1.2, 'Compute-Bound\n(AI > Ridge)',
            fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Labels and styling
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Roofline Analysis: Naive MHA vs Flash Attention\n{gpu["name"]}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Add GPU specs text box
    specs_text = (
        f'GPU Specifications:\n'
        f'  Peak Compute: {gpu["peak_compute_tflops"]:.1f} TFLOP/s\n'
        f'  Peak BW: {gpu["peak_bandwidth_gb_s"]:.1f} GB/s\n'
        f'  Ridge AI: {ridge_ai:.1f} FLOPs/byte\n\n'
        f'  Note: 1 TFLOP/s = 1,000 GFLOP/s'
    )
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _get_default_trace_dir() / f"roofline_plot_{timestamp}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

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
    parser.add_argument("--gpu", type=str, default="rtx4000-ada", help="GPU model")
    parser.add_argument("--trace-dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Configure trace directory
    if args.trace_dir:
        configure(trace_dir=args.trace_dir)

    dtype = jnp.float16 if args.dtype == "float16" else jnp.float32
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    print("=" * 70)
    print("ROOFLINE ANALYSIS FOR ATTENTION")
    print("=" * 70)

    # Print GPU info
    gpu = GPU_SPECS[args.gpu]
    print(f"\nGPU: {gpu['name']}")
    print(f"  Peak Compute (FP16):    {gpu['peak_compute_tflops']:.1f} TFLOP/s")
    print(f"  Peak Memory Bandwidth:  {gpu['peak_bandwidth_gb_s']:.1f} GB/s")
    print(f"  Ridge AI:               {gpu['ridge_ai']:.1f} FLOPs/byte")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Batch size:      {args.batch}")
    print(f"  Number of heads: {args.heads}")
    print(f"  Head dimension:  {args.head_dim}")
    print(f"  Data type:       {args.dtype}")
    print(f"  Sequence lengths: {seq_lengths}")

    print("\n" + "=" * 70)
    print("BENCHMARKING")
    print("=" * 70)

    # Benchmark across sequence lengths
    results = {
        "gpu": args.gpu,
        "config": {
            "B": args.batch,
            "H": args.heads,
            "D": args.head_dim,
            "dtype": args.dtype,
        },
        "sequence_lengths": seq_lengths,
        "naive": {"time_ms": [], "gflops_s": [], "ai": [], "bw_gb_s": []},
        "flash": {"time_ms": [], "gflops_s": [], "ai": [], "bw_gb_s": []},
        "speedups": [],
    }

    for T in seq_lengths:
        benchmark_result = benchmark_attention(
            args.batch, args.heads, T, args.head_dim,
            dtype=dtype, warmup_iters=args.warmup, profile_iters=args.iters
        )

        results["naive"]["time_ms"].append(benchmark_result["naive"]["time_ms"])
        results["naive"]["gflops_s"].append(benchmark_result["naive"]["gflops_s"])
        results["naive"]["ai"].append(benchmark_result["naive"]["ai"])
        results["naive"]["bw_gb_s"].append(benchmark_result["naive"]["bw_gb_s"])

        results["flash"]["time_ms"].append(benchmark_result["flash"]["time_ms"])
        results["flash"]["gflops_s"].append(benchmark_result["flash"]["gflops_s"])
        results["flash"]["ai"].append(benchmark_result["flash"]["ai"])
        results["flash"]["bw_gb_s"].append(benchmark_result["flash"]["bw_gb_s"])

        results["speedups"].append(benchmark_result["speedup"])

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'T':<6} {'Naive':<12} {'Flash':<12} {'Naive':<12} {'Flash':<12} {'Naive':<10} {'Flash':<10} {'Speedup':<8}")
    print(f"{'':<6} {'(ms)':<12} {'(ms)':<12} {'(GFLOP/s)':<12} {'(GFLOP/s)':<12} {'(AI)':<10} {'(AI)':<10} {'':<8}")
    print("-" * 70)

    for i, T in enumerate(seq_lengths):
        print(f"{T:<6} {results['naive']['time_ms'][i]:<12.3f} "
              f"{results['flash']['time_ms'][i]:<12.3f} "
              f"{results['naive']['gflops_s'][i]:<12.2f} "
              f"{results['flash']['gflops_s'][i]:<12.2f} "
              f"{results['naive']['ai'][i]:<10.1f} "
              f"{results['flash']['ai'][i]:<10.1f} "
              f"{results['speedups'][i]:<8.2f}x")

    avg_speedup = np.mean(results["speedups"])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    # Verify GFLOPs/s against peak compute
    max_flash_gflops = np.max(results["flash"]["gflops_s"])
    peak_gflops = gpu["peak_compute_tflops"] * 1000  # Convert TFLOP/s to GFLOP/s
    utilization = (max_flash_gflops / peak_gflops) * 100
    print(f"\nPeak verification:")
    print(f"  GPU Peak Compute: {peak_gflops:.0f} GFLOP/s ({gpu['peak_compute_tflops']:.1f} TFLOP/s)")
    print(f"  Max Achieved: {max_flash_gflops:.0f} GFLOP/s")
    print(f"  Utilization: {utilization:.1f}%")
    print(f"  Note: 1 TFLOP/s = 1,000 GFLOP/s")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = _get_default_trace_dir() / f"roofline_data_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate plot
    if not args.no_plot:
        plot_path = generate_roofline_plot(results, gpu_key=args.gpu)
        print(f"\nDone! View plot at: {plot_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
