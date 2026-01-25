#!/usr/bin/env python3
"""Profile flash attention implementations (forward and backward separately).

Usage:
    # On remote GPU machine:
    python scripts/profile_attention.py

    # Download traces and view locally:
    make download-traces
    make xprof-serve
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from high_performance_jax.profiling import (
    configure,
    profile_function,
    start_xprof_server,
    print_traces,
    get_trace_dir,
)
from high_performance_jax.pallas.pallas_flash_attn import flash_attention, mha_reference, cudnn_attention


def profile_attention(B: int, H: int, T: int, D: int, dtype=jnp.float16):
    """Profile forward and backward passes separately for each attention implementation."""
    print(f"\nProfiling attention: B={B}, H={H}, T={T}, D={D}, dtype={dtype}")

    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=dtype)

    gpu_model = os.environ.get("GPU_MODEL", "unknown").replace(" ", "_")
    shape_str = f"B{B}_H{H}_T{T}_D{D}_{gpu_model}"

    # Create jitted forward functions
    flash_fwd = jax.jit(flash_attention)
    ref_fwd = jax.jit(mha_reference)
    cudnn_fwd = jax.jit(cudnn_attention)

    # Get vjp functions for backward passes
    _, flash_vjp = jax.vjp(flash_attention, q, k, v)
    _, ref_vjp = jax.vjp(mha_reference, q, k, v)
    _, cudnn_vjp = jax.vjp(cudnn_attention, q, k, v)

    # Profile Flash Attention
    print("\n" + "="*60)
    print("Profiling: Flash Attention (forward)")
    print("="*60)
    profile_function(
        f"flash_fwd_{shape_str}",
        lambda: jax.block_until_ready(flash_fwd(q, k, v))
    )

    print("\n" + "="*60)
    print("Profiling: Flash Attention (backward)")
    print("="*60)
    profile_function(
        f"flash_bwd_{shape_str}",
        lambda: jax.block_until_ready(flash_vjp(do))
    )

    # Profile Reference Attention
    print("\n" + "="*60)
    print("Profiling: Reference Attention (forward)")
    print("="*60)
    profile_function(
        f"reference_fwd_{shape_str}",
        lambda: jax.block_until_ready(ref_fwd(q, k, v))
    )

    print("\n" + "="*60)
    print("Profiling: Reference Attention (backward)")
    print("="*60)
    profile_function(
        f"reference_bwd_{shape_str}",
        lambda: jax.block_until_ready(ref_vjp(do))
    )

    # Profile cuDNN Attention
    print("\n" + "="*60)
    print("Profiling: cuDNN Attention (forward)")
    print("="*60)
    profile_function(
        f"cudnn_fwd_{shape_str}",
        lambda: jax.block_until_ready(cudnn_fwd(q, k, v))
    )

    print("\n" + "="*60)
    print("Profiling: cuDNN Attention (backward)")
    print("="*60)
    profile_function(
        f"cudnn_bwd_{shape_str}",
        lambda: jax.block_until_ready(cudnn_vjp(do))
    )


def main():
    parser = argparse.ArgumentParser(description="Profile attention implementations")
    parser.add_argument("--serve", action="store_true", help="Start xprof server")
    parser.add_argument("--list", action="store_true", help="List available traces")
    parser.add_argument("--port", type=int, default=8791, help="xprof server port")
    parser.add_argument("--trace-dir", type=str, default=None, help="Directory for traces")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")

    args = parser.parse_args()

    if args.trace_dir:
        configure(trace_dir=args.trace_dir)

    if args.serve:
        start_xprof_server(port=args.port)
        return

    if args.list:
        print_traces()
        return

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"GPU Model: {os.environ.get('GPU_MODEL', 'not set')}")
    print(f"Traces will be saved to: {get_trace_dir()}")

    profile_attention(args.batch, args.heads, args.seq_len, args.head_dim)

    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)
    print("\nTo view traces:")
    print("  make download-traces")
    print("  make xprof-serve")


if __name__ == "__main__":
    main()
