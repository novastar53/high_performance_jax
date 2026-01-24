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
    mha_reference = flash_attn_module.mha_reference
    cudnn_attention = flash_attn_module.cudnn_attention
else:
    from high_performance_jax.pallas.pallas_flash_attn import (
        flash_attention,
        mha_reference,
        cudnn_attention,
    )


# GPU Specifications for RTX 4000 Ada
# From NVIDIA datasheet:
# - FP32 CUDA cores: 26.7 TFLOP/s
# - FP16 CUDA cores: 26.7 TFLOP/s (1:1 with FP32)
# - Tensor Performance: 327.6 TFLOP/s (INT8/sparse modes)
# - Memory Bandwidth: 360 GB/s
#
# Observed performance:
# - Pallas flash attention: ~27 TFLOP/s
#   (Should use tensor cores via Triton, but achieves CUDA-core level perf.
#    Likely due to suboptimal block sizes, memory patterns, or loop overhead.)
# - cuDNN flash attention: ~55 TFLOP/s
#   (Highly optimized, achieves full tensor core throughput)
GPU_SPECS = {
    "rtx4000-ada": {
        "name": "NVIDIA RTX 4000 Ada",
        "peak_compute_tflops": 26.7,           # FP16/FP32 CUDA core peak
        "peak_compute_tflops_tc": 53.4,        # FP16 Tensor Core (observed via cuDNN)
        "tensor_tflops_datasheet": 327.6,      # Tensor perf from datasheet (INT8/sparse)
        "peak_bandwidth_gb_s": 360.0,          # GDDR6
        "ridge_ai": 26.7e3 / 360.0,            # Ridge point for CUDA cores
        "ridge_ai_tc": 53.4e3 / 360.0,         # Ridge point for Tensor Cores
    }
}


def calculate_flops_fwd(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for forward attention only.

    Forward:
    - Q @ K.T: B*H*T*T*D * 2 (mult + add per element)
    - Softmax: B*H*T*T * ~5 ops (exp, sum, max, div, sub)
    - P @ V: B*H*T*T*D * 2 (mult + add per element)
    Total forward: 4*B*H*T^2*D + 5*B*H*T^2
    """
    fwd_matmul = 4 * B * H * T * T * D  # 2 matmuls * 2 ops each
    fwd_softmax = 5 * B * H * T * T     # Softmax ops
    return fwd_matmul + fwd_softmax


def calculate_flops_pallas(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for Pallas flash attention (forward + backward).

    Our Pallas implementation recomputes attention twice in backward:
    - dKV kernel: recomputes S = Q @ K^T, computes dP, dV, dK (4 matmuls)
    - dQ kernel: recomputes S = Q @ K^T, computes dP, dQ (3 matmuls)
    Total backward: ~14*B*H*T^2*D

    Total: ~18*B*H*T^2*D (forward + backward)
    """
    total_fwd = calculate_flops_fwd(B, H, T, D)

    # Backward pass with flash attention recomputation (our implementation)
    bwd_dkv = 8 * B * H * T * T * D   # S recompute + dP + dV + dK
    bwd_dq = 6 * B * H * T * T * D    # S recompute + dP + dQ
    total_bwd = bwd_dkv + bwd_dq

    return total_fwd + total_bwd


def calculate_flops_cudnn(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for cuDNN flash attention (forward + backward).

    cuDNN uses optimized backward that recomputes attention only once:
    - Single fused backward: recomputes S once, computes dQ, dK, dV
    Total backward: ~10*B*H*T^2*D

    Total: ~14*B*H*T^2*D (forward + backward)
    """
    total_fwd = calculate_flops_fwd(B, H, T, D)

    # cuDNN backward - more efficient, recomputes once
    # S recompute: 2*T²*D, dP: 2*T²*D, dV: 2*T²*D, dQ: 2*T²*D, dK: 2*T²*D
    total_bwd = 10 * B * H * T * T * D

    return total_fwd + total_bwd


def calculate_flops_naive(B: int, H: int, T: int, D: int) -> float:
    """Calculate FLOPs for naive attention (forward + backward).

    Naive attention stores the full attention matrix, so backward doesn't
    need to recompute. Uses standard autodiff.

    Forward: 4*B*H*T^2*D
    Backward: ~8*B*H*T^2*D (dV, dP, dQ, dK matmuls)
    Total: ~12*B*H*T^2*D
    """
    total_fwd = calculate_flops_fwd(B, H, T, D)

    # Standard backward with stored attention matrix
    total_bwd = 8 * B * H * T * T * D

    return total_fwd + total_bwd


def calculate_bytes_naive(B: int, H: int, T: int, D: int, bytes_per_elem: int = 4) -> float:
    """Calculate bytes transferred for naive MHA (fwd+bwd).

    Naive MHA materializes full attention matrix:
    - Q, K, V (input): B*H*T*D * 3 * bytes_per_elem
    - Attention matrix (T×T): B*H*T*T * bytes_per_elem  <-- THE BIG ONE
    - Output O: B*H*T*D * bytes_per_elem

    Backward: similar traffic plus gradients, approximately 2x forward
    """
    fwd_bytes = (
        B * H * T * D * 3 * bytes_per_elem +  # Q, K, V
        B * H * T * T * bytes_per_elem +       # Attention matrix
        B * H * T * D * bytes_per_elem          # Output O
    )
    # Backward roughly doubles memory traffic (gradients similar size)
    return fwd_bytes * 2


def calculate_bytes_flash(B: int, H: int, T: int, D: int, bytes_per_elem: int = 4) -> float:
    """Calculate bytes transferred for flash attention (fwd+bwd).

    Flash attention computes in tiles, only stores logsumexp:
    - Q, K, V (input): B*H*T*D * 3 * bytes_per_elem
    - logsumexp: B*H*T * bytes_per_elem  <-- TINY RESIDUAL
    - Output O: B*H*T*D * bytes_per_elem

    Backward: similar traffic plus gradients, approximately 2x forward
    """
    fwd_bytes = (
        B * H * T * D * 3 * bytes_per_elem +  # Q, K, V
        B * H * T * bytes_per_elem +             # logsumexp
        B * H * T * D * bytes_per_elem            # Output O
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
        Dict with 'naive', 'flash', and 'cudnn' results containing:
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
    # Each implementation has different FLOP counts due to different backward algorithms
    bytes_per_elem = 2 if dtype == jnp.float16 else 4
    flops_naive = calculate_flops_naive(B, H, T, D)
    flops_pallas = calculate_flops_pallas(B, H, T, D)
    flops_cudnn = calculate_flops_cudnn(B, H, T, D)
    bytes_naive = calculate_bytes_naive(B, H, T, D, bytes_per_elem)
    bytes_flash = calculate_bytes_flash(B, H, T, D, bytes_per_elem)

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
    naive_gflops_s = flops_naive / (naive_time_s * 1e9)
    naive_ai = flops_naive / bytes_naive
    naive_bw_gb_s = bytes_naive / (naive_time_s * 1e9)

    # Time flash attention (our Pallas implementation)
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
    flash_gflops_s = flops_pallas / (flash_time_s * 1e9)
    flash_ai = flops_pallas / bytes_flash
    flash_bw_gb_s = bytes_flash / (flash_time_s * 1e9)

    # Time cuDNN attention (jax.nn.dot_product_attention)
    def cudnn_loss(q, k, v):
        return jnp.sum(cudnn_attention(q, k, v) * do)

    cudnn_fwd_bwd = jax.jit(jax.value_and_grad(cudnn_loss, argnums=(0, 1, 2)))

    # Warm up cuDNN attention
    print("  Warming up cuDNN attention...")
    for _ in range(warmup_iters):
        _, grads = cudnn_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)

    # Time cuDNN attention
    print("  Timing cuDNN attention...")
    cudnn_times = []
    for _ in range(profile_iters):
        t0 = time.perf_counter()
        _, grads = cudnn_fwd_bwd(q, k, v)
        jax.block_until_ready(grads)
        cudnn_times.append(time.perf_counter() - t0)

    cudnn_time_s = np.median(cudnn_times)
    cudnn_time_ms = cudnn_time_s * 1000
    cudnn_gflops_s = flops_cudnn / (cudnn_time_s * 1e9)
    cudnn_ai = flops_cudnn / bytes_flash  # cuDNN uses flash attention, same memory pattern
    cudnn_bw_gb_s = bytes_flash / (cudnn_time_s * 1e9)

    speedup_flash = naive_time_ms / flash_time_ms
    speedup_cudnn = naive_time_ms / cudnn_time_ms

    print(f"  Naive:  {naive_time_ms:.3f} ms, {naive_gflops_s:.2f} GFLOP/s, AI={naive_ai:.1f}")
    print(f"  Flash:  {flash_time_ms:.3f} ms, {flash_gflops_s:.2f} GFLOP/s, AI={flash_ai:.1f}")
    print(f"  cuDNN:  {cudnn_time_ms:.3f} ms, {cudnn_gflops_s:.2f} GFLOP/s, AI={cudnn_ai:.1f}")
    print(f"  Speedup (flash vs naive): {speedup_flash:.2f}x")
    print(f"  Speedup (cuDNN vs naive): {speedup_cudnn:.2f}x")

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
        "cudnn": {
            "time_ms": cudnn_time_ms,
            "gflops_s": cudnn_gflops_s,
            "ai": cudnn_ai,
            "bw_gb_s": cudnn_bw_gb_s,
        },
        "speedup": speedup_flash,
        "speedup_cudnn": speedup_cudnn,
    }


def generate_roofline_plot(
    results: dict,
    gpu_key: str = "rtx4000-ada",
    output_path: Path | None = None,
):
    """Generate roofline plot.

    Args:
        results: Dict with 'sequence_lengths', 'naive', 'flash', 'cudnn' data
        gpu_key: Key for GPU specs in GPU_SPECS
        output_path: Path to save PNG plot
    """
    gpu = GPU_SPECS[gpu_key]

    seq_lengths = np.array(results["sequence_lengths"])
    naive_ai = np.array(results["naive"]["ai"])
    flash_ai = np.array(results["flash"]["ai"])
    cudnn_ai = np.array(results["cudnn"]["ai"])
    naive_perf = np.array(results["naive"]["gflops_s"])
    flash_perf = np.array(results["flash"]["gflops_s"])
    cudnn_perf = np.array(results["cudnn"]["gflops_s"])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine axis range based on actual data
    all_ai = np.concatenate([naive_ai, flash_ai, cudnn_ai])
    all_perf = np.concatenate([naive_perf, flash_perf, cudnn_perf])
    ridge_ai = gpu['ridge_ai']  # Use FP32 ridge point (Pallas achieves FP32-level)
    ai_min = min(all_ai.min(), ridge_ai) / 2  # Include ridge point, with padding
    ai_max = max(all_ai.max(), ridge_ai) * 2

    # Roofline calculations
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 100)

    # Memory roof (diagonal)
    # GB/s * FLOPs/byte = 10^9 bytes/s * FLOPs/byte = 10^9 FLOPs/s = GFLOP/s
    memory_roof = gpu["peak_bandwidth_gb_s"] * ai_range

    # Compute roofs (horizontal) - show both FP32 and FP16 Tensor Core peaks
    compute_roof_fp32 = gpu["peak_compute_tflops"] * 1000 * np.ones_like(ai_range)
    compute_roof_tc = gpu["peak_compute_tflops_tc"] * 1000 * np.ones_like(ai_range)

    # Cap memory roof at FP16 TC compute roof (the higher one)
    memory_roof = np.minimum(memory_roof, compute_roof_tc)

    # Plot roofs
    ax.plot(ai_range, memory_roof, 'k--', linewidth=2, alpha=0.7, label='Memory roof')
    ax.plot(ai_range, compute_roof_fp32, 'r--', linewidth=2, alpha=0.7, label='FP32 roof (26.7 TFLOP/s)')
    ax.plot(ai_range, compute_roof_tc, 'g--', linewidth=2, alpha=0.7, label='FP16 TC roof (53.4 TFLOP/s)')

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
    ridge_perf = gpu["peak_compute_tflops"] * 1000  # Convert TFLOP/s to GFLOP/s
    ridge_ai = gpu['ridge_ai']
    ax.axvline(ridge_ai, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai, ridge_perf * 0.1, f'  Ridge\n  AI={ridge_ai:.1f}',
            fontsize=10, rotation=90, va='bottom', ha='right')

    # Region annotations - position relative to ridge point
    ax.text(ridge_ai / 3, ridge_perf * 1.2, 'Memory-Bound\n(AI < Ridge)',
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.text(ridge_ai * 3, ridge_perf * 1.2, 'Compute-Bound\n(AI > Ridge)',
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
    perf_max = max(all_perf.max(), ridge_perf) * 1.5
    ax.set_ylim(perf_min, perf_max)

    ax.set_title(f'Roofline Analysis: Naive vs Flash (Pallas) vs cuDNN\n{gpu["name"]}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Add GPU specs text box
    specs_text_lines = [
        "GPU Specifications:",
        f"  CUDA peak: {gpu['peak_compute_tflops']:.1f} TFLOP/s",
        f"  TC peak: {gpu['peak_compute_tflops_tc']:.1f} TFLOP/s",
        f"  Bandwidth: {gpu['peak_bandwidth_gb_s']:.1f} GB/s",
        "",
        "Pallas: ~CUDA-level perf",
        "cuDNN: full TC throughput",
    ]
    specs_text = "\n".join(specs_text_lines)

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
        "cudnn": {"time_ms": [], "gflops_s": [], "ai": [], "bw_gb_s": []},
        "speedups": [],
        "speedups_cudnn": [],
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

        results["cudnn"]["time_ms"].append(benchmark_result["cudnn"]["time_ms"])
        results["cudnn"]["gflops_s"].append(benchmark_result["cudnn"]["gflops_s"])
        results["cudnn"]["ai"].append(benchmark_result["cudnn"]["ai"])
        results["cudnn"]["bw_gb_s"].append(benchmark_result["cudnn"]["bw_gb_s"])

        results["speedups"].append(benchmark_result["speedup"])
        results["speedups_cudnn"].append(benchmark_result["speedup_cudnn"])

    # Print summary table
    print("\n" + "=" * 95)
    print("SUMMARY TABLE")
    print("=" * 95)
    print(f"{'T':<6} {'Naive':<10} {'Flash':<10} {'cuDNN':<10} "
          f"{'Naive':<12} {'Flash':<12} {'cuDNN':<12} {'Flash':<8} {'cuDNN':<8}")
    print(f"{'':<6} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} "
          f"{'(GFLOP/s)':<12} {'(GFLOP/s)':<12} {'(GFLOP/s)':<12} {'Speedup':<8} {'Speedup':<8}")
    print("-" * 95)

    for i, T in enumerate(seq_lengths):
        print(f"{T:<6} {results['naive']['time_ms'][i]:<10.3f} "
              f"{results['flash']['time_ms'][i]:<10.3f} "
              f"{results['cudnn']['time_ms'][i]:<10.3f} "
              f"{results['naive']['gflops_s'][i]:<12.2f} "
              f"{results['flash']['gflops_s'][i]:<12.2f} "
              f"{results['cudnn']['gflops_s'][i]:<12.2f} "
              f"{results['speedups'][i]:<8.2f}x "
              f"{results['speedups_cudnn'][i]:<8.2f}x")

    avg_speedup = np.mean(results["speedups"])
    avg_speedup_cudnn = np.mean(results["speedups_cudnn"])
    print(f"\nAverage speedup (flash vs naive):  {avg_speedup:.2f}x")
    print(f"Average speedup (cuDNN vs naive):  {avg_speedup_cudnn:.2f}x")

    # Verify GFLOPs/s against peak compute
    max_flash_gflops = np.max(results["flash"]["gflops_s"])
    max_cudnn_gflops = np.max(results["cudnn"]["gflops_s"])
    peak_fp32 = gpu["peak_compute_tflops"] * 1000
    peak_tc = gpu["peak_compute_tflops_tc"] * 1000

    print(f"\nPeak verification:")
    print(f"  GPU FP32 Peak:     {peak_fp32:.0f} GFLOP/s ({gpu['peak_compute_tflops']:.1f} TFLOP/s)")
    print(f"  GPU FP16 TC Peak:  {peak_tc:.0f} GFLOP/s ({gpu['peak_compute_tflops_tc']:.1f} TFLOP/s)")
    print(f"  Max Achieved (Pallas): {max_flash_gflops:.0f} GFLOP/s ({max_flash_gflops/peak_fp32*100:.1f}% of CUDA peak)")
    print(f"  Max Achieved (cuDNN):  {max_cudnn_gflops:.0f} GFLOP/s ({max_cudnn_gflops/peak_tc*100:.1f}% of TC peak)")
    print(f"\n  Note: Pallas/Triton should use TCs but achieves CUDA-level perf (optimization opportunity)")
    print(f"\n  FLOP counts (per B*H, forward + backward):")
    print(f"    Naive:  ~12*T^2*D (stores attention matrix, no recompute)")
    print(f"    Pallas: ~18*T^2*D (recomputes attention twice in backward)")
    print(f"    cuDNN:  ~14*T^2*D (recomputes attention once in backward)")

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
