#!/usr/bin/env python3
"""Generate roofline plots from existing benchmark data JSON.

This script reads benchmark data JSON files and generates roofline plots
for forward and backward passes.

Usage:
    python scripts/plot_roofline.py /path/to/roofline_data_*.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")


# GPU Specifications dictionary mapping GPU_MODEL values to specs
GPU_MODEL_TO_SPECS = {
    # NVIDIA RTX 4000 Ada
    "NVIDIA RTX 4000 Ada": {
        "peak_compute_tflops": 26.7,
        "peak_compute_tflops_tc": 106.91,
        "tensor_tflops_datasheet": 327.6,
        "peak_bandwidth_gb_s": 360.0,
        "ridge_ai": 26.7e3 / 360.0,
        "ridge_ai_tc": 106.91e3 / 360.0,
    },
}


def get_gpu_specs(gpu_model: str) -> dict:
    """Get GPU specifications for the given GPU model.

    Args:
        gpu_model: GPU model name (e.g., "NVIDIA RTX 4000 Ada")

    Returns:
        Dictionary with GPU specs including name added from GPU_MODEL_TO_SPECS
    """
    if gpu_model in GPU_MODEL_TO_SPECS:
        specs = GPU_MODEL_TO_SPECS[gpu_model].copy()
        specs["name"] = gpu_model
        return specs
    else:
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


def _get_default_trace_dir() -> Path:
    """Get the default trace directory."""
    trace_dir = os.environ.get("JAX_TRACE_DIR")
    if trace_dir:
        return Path(trace_dir)
    return Path(__file__).parent.parent / "traces"


def generate_roofline_plot(
    results: dict,
    pass_type: str = "fwd",
    gpu_model: str = None,
    output_path: Path | None = None,
):
    """Generate roofline plot for forward or backward pass.

    Args:
        results: Dict with 'sequence_lengths', 'naive', 'flash', 'cudnn' data
        pass_type: "fwd" for forward pass, "bwd" for backward pass
        gpu_model: GPU model name (e.g., "NVIDIA RTX 4000 Ada")
        output_path: Path to save PNG plot
    """
    if gpu_model is None:
        gpu_model = os.environ.get("GPU_MODEL", "NVIDIA RTX 4000 Ada")

    gpu = get_gpu_specs(gpu_model)
    pass_name = "Forward" if pass_type == "fwd" else "Backward"

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
    ridge_ai = gpu['ridge_ai']
    ai_min = min(all_ai.min(), ridge_ai) / 2
    ai_max = max(all_ai.max(), ridge_ai) * 2

    # Roofline calculations
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 100)

    # Memory roof (diagonal)
    memory_roof = gpu["peak_bandwidth_gb_s"] * ai_range

    # Compute roofs (horizontal) - show both FP32 and FP16 Tensor Core peaks
    compute_roof_fp32 = gpu["peak_compute_tflops"] * 1000 * np.ones_like(ai_range)
    compute_roof_tc = gpu["peak_compute_tflops_tc"] * 1000 * np.ones_like(ai_range)

    # Cap memory roof at FP16 TC compute roof
    memory_roof = np.minimum(memory_roof, compute_roof_tc)

    # Plot roofs
    ax.plot(ai_range, memory_roof, 'k--', linewidth=2, alpha=0.7, label='Memory roof')
    ax.plot(ai_range, compute_roof_fp32, 'r--', linewidth=2, alpha=0.7, label=f'FP32 roof ({gpu["peak_compute_tflops"]:.1f} TFLOP/s)')
    ax.plot(ai_range, compute_roof_tc, 'g--', linewidth=2, alpha=0.7, label=f'FP16 TC roof ({gpu["peak_compute_tflops_tc"]:.1f} TFLOP/s)')

    # Plot actual performance
    ax.scatter(naive_ai, naive_perf, marker='o', s=150, c='red',
               edgecolors='black', linewidth=1.5, label='Naive MHA', zorder=5)
    ax.scatter(flash_ai, flash_perf, marker='s', s=150, c='blue',
               edgecolors='black', linewidth=1.5, label='Flash (Pallas)', zorder=5)
    ax.scatter(cudnn_ai, cudnn_perf, marker='^', s=150, c='green',
               edgecolors='black', linewidth=1.5, label='cuDNN Flash', zorder=5)

    # Annotate sequence lengths
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
    ridge_ai = gpu['ridge_ai']
    ax.axvline(ridge_ai, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai, ridge_perf_fp32 * 0.1, f'  Ridge\n  AI={ridge_ai:.1f}',
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
    perf_max = max(all_perf.max(), ridge_perf_fp32, compute_roof_tc[0]) * 1.5
    ax.set_ylim(perf_min, perf_max)

    ax.set_title(f'Roofline Analysis ({pass_name} Pass): Naive vs Flash (Pallas) vs cuDNN\n{gpu["name"]}',
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
    parser = argparse.ArgumentParser(description="Generate roofline plots from benchmark JSON data")
    parser.add_argument("json_file", type=str, help="Path to roofline benchmark JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output plots")
    parser.add_argument("--forward-only", action="store_true", help="Only generate forward pass plot")
    parser.add_argument("--backward-only", action="store_true", help="Only generate backward pass plot")

    args = parser.parse_args()

    # Read JSON file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    with open(json_path) as f:
        results = json.load(f)

    print("=" * 70)
    print("ROOFLINE PLOT GENERATION FROM JSON")
    print("=" * 70)
    print(f"\nInput file: {json_path}")

    # Get GPU model from results or environment
    gpu_model = results.get("gpu_model") or os.environ.get("GPU_MODEL", "NVIDIA RTX 4000 Ada")
    print(f"GPU Model: {gpu_model}")

    # Get GPU specs
    gpu = get_gpu_specs(gpu_model)
    print(f"\nGPU: {gpu['name']}")
    print(f"  Peak Compute (CUDA cores):   {gpu['peak_compute_tflops']:.1f} TFLOP/s")
    print(f"  Peak Compute (Tensor cores): {gpu['peak_compute_tflops_tc']:.1f} TFLOP/s")
    print(f"  Peak Memory Bandwidth:       {gpu['peak_bandwidth_gb_s']:.1f} GB/s")

    # Print configuration
    config = results.get("config", {})
    print(f"\nConfiguration:")
    print(f"  Batch size:      {config.get('B', 'N/A')}")
    print(f"  Number of heads: {config.get('H', 'N/A')}")
    print(f"  Head dimension:  {config.get('D', 'N/A')}")
    print(f"  Data type:       {config.get('dtype', 'N/A')}")
    print(f"  Sequence lengths: {results.get('sequence_lengths', [])}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _get_default_trace_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    output_paths = []

    if not args.backward_only:
        fwd_plot_path = output_dir / json_path.stem.replace("roofline_data_", f"roofline_fwd_{gpu_model.replace(' ', '_')}.png")
        fwd_plot_path = generate_roofline_plot(results, pass_type="fwd", gpu_model=gpu_model, output_path=fwd_plot_path)
        output_paths.append(fwd_plot_path)

    if not args.forward_only:
        bwd_plot_path = output_dir / json_path.stem.replace("roofline_data_", f"roofline_bwd_{gpu_model.replace(' ', '_')}.png")
        bwd_plot_path = generate_roofline_plot(results, pass_type="bwd", gpu_model=gpu_model, output_path=bwd_plot_path)
        output_paths.append(bwd_plot_path)

    print("\n" + "=" * 70)
    print("Done! View plots at:")
    for path in output_paths:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    exit(main())
