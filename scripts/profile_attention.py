#!/usr/bin/env python3
"""Profile flash attention implementations.

This script demonstrates how to use the profiling module to compare
different attention implementations and analyze memory/compute patterns.

Usage:
    # On remote GPU machine:
    python scripts/profile_attention.py

    # Then start xprof server:
    python scripts/profile_attention.py --serve

    # From local machine, SSH tunnel:
    make xprof-tunnel h=<remote_host> k=<keyfile>

    # Open http://localhost:8791 in browser
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from high_performance_jax.profiling import (
    configure,
    profile,
    profile_comparison,
    profile_function,
    start_xprof_server,
    print_traces,
)
from high_performance_jax.pallas.pallas_flash_attn import (
    flash_attention,
    flash_attention_fwd,
    flash_attention_bwd,
)


def mha_reference(q, k, v):
    """Reference multi-head attention (materializes N×N matrix)."""
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    probs = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
    return o


def mha_reference_no_jit(q, k, v):
    """Reference attention without JIT (truly naive baseline)."""
    d = q.shape[-1]
    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    probs = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
    return o


def profile_forward_pass(B: int, H: int, T: int, D: int, dtype=jnp.float16):
    """Profile forward pass implementations."""
    print(f"\nProfiling forward pass: B={B}, H={H}, T={T}, D={D}, dtype={dtype}")

    key = jax.random.key(42)
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)

    # JIT compile the functions
    flash_jit = jax.jit(flash_attention)
    ref_jit = jax.jit(mha_reference)

    profile_comparison(
        f"attention_fwd_B{B}_H{H}_T{T}_D{D}",
        ("flash_attention", lambda: flash_jit(q, k, v).block_until_ready()),
        ("reference_jit", lambda: ref_jit(q, k, v).block_until_ready()),
        ("reference_no_jit", lambda: mha_reference_no_jit(q, k, v).block_until_ready()),
    )


def profile_backward_pass(B: int, H: int, T: int, D: int, dtype=jnp.float16):
    """Profile backward pass implementations."""
    print(f"\nProfiling backward pass: B={B}, H={H}, T={T}, D={D}, dtype={dtype}")

    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=dtype)

    # Create gradient functions
    def flash_loss(q, k, v):
        return jnp.sum(flash_attention(q, k, v) * do)

    def ref_loss(q, k, v):
        return jnp.sum(mha_reference(q, k, v) * do)

    flash_grad = jax.jit(jax.grad(flash_loss, argnums=(0, 1, 2)))
    ref_grad = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))

    profile_comparison(
        f"attention_bwd_B{B}_H{H}_T{T}_D{D}",
        ("flash_attention", lambda: jax.block_until_ready(flash_grad(q, k, v))),
        ("reference_jit", lambda: jax.block_until_ready(ref_grad(q, k, v))),
    )


def profile_memory_scaling(dtype=jnp.float16):
    """Profile memory usage across different sequence lengths."""
    print("\nProfiling memory scaling across sequence lengths...")

    B, H, D = 2, 4, 64
    seq_lengths = [256, 512, 1024, 2048]

    for T in seq_lengths:
        print(f"\n--- Sequence length: {T} ---")
        profile_forward_pass(B, H, T, D, dtype)


def main():
    parser = argparse.ArgumentParser(description="Profile attention implementations")
    parser.add_argument("--serve", action="store_true", help="Start xprof server")
    parser.add_argument("--list", action="store_true", help="List available traces")
    parser.add_argument("--port", type=int, default=8791, help="xprof server port")
    parser.add_argument("--trace-dir", type=str, default="/tmp/jax-traces",
                       help="Directory for traces")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--backward", action="store_true", help="Also profile backward pass")
    parser.add_argument("--scaling", action="store_true",
                       help="Profile memory scaling across sequence lengths")

    args = parser.parse_args()

    # Configure profiling
    configure(trace_dir=args.trace_dir)

    if args.serve:
        start_xprof_server(port=args.port)
        return

    if args.list:
        print_traces()
        return

    # Print device info
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    if args.scaling:
        profile_memory_scaling()
    else:
        # Profile forward pass
        profile_forward_pass(args.batch, args.heads, args.seq_len, args.head_dim)

        # Optionally profile backward pass
        if args.backward:
            profile_backward_pass(args.batch, args.heads, args.seq_len, args.head_dim)

    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)
    print("\nTo view traces:")
    print(f"  1. Start server: python {__file__} --serve")
    print(f"  2. SSH tunnel:   make xprof-tunnel h=<host> k=<keyfile>")
    print(f"  3. Open:         http://localhost:{args.port}")


if __name__ == "__main__":
    main()
