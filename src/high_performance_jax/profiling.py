"""Reusable JAX profiling utilities using XProf.

Usage:
    from high_performance_jax.profiling import profile, profile_comparison, start_xprof_server

    # Profile a single function
    with profile("my_kernel"):
        result = my_kernel(x, y)
        result.block_until_ready()

    # Compare two implementations
    profile_comparison(
        "attention_comparison",
        ("flash_attn", lambda: flash_attention(q, k, v)),
        ("reference", lambda: mha_reference(q, k, v)),
    )

    # Start xprof server for viewing
    start_xprof_server()  # Then SSH tunnel: ssh -L 8791:localhost:8791 user@host

Remote profiling setup:
    1. Run your script with profiling on the remote machine
    2. Start xprof server: python -c "from high_performance_jax.profiling import start_xprof_server; start_xprof_server()"
    3. SSH tunnel: ssh -L 8791:localhost:8791 user@remote_host
    4. Open http://localhost:8791 in your browser
"""

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import jax
import jax.numpy as jnp


# Default trace directory
DEFAULT_TRACE_DIR = Path("/tmp/jax-traces")


@dataclass
class ProfileConfig:
    """Configuration for profiling."""
    trace_dir: Path = DEFAULT_TRACE_DIR
    host_tracer_level: int = 2  # 0=disabled, 1=user events, 2=default, 3=verbose
    device_tracer_level: int = 1  # 0=disabled, 1=enabled
    python_tracer_level: int = 0  # 0=disabled, 1=enabled
    warmup_iters: int = 3
    profile_iters: int = 1

    def __post_init__(self):
        self.trace_dir = Path(self.trace_dir)


# Global config
_config = ProfileConfig()


def configure(
    trace_dir: str | Path | None = None,
    host_tracer_level: int | None = None,
    device_tracer_level: int | None = None,
    python_tracer_level: int | None = None,
    warmup_iters: int | None = None,
    profile_iters: int | None = None,
):
    """Configure global profiling settings."""
    global _config
    if trace_dir is not None:
        _config.trace_dir = Path(trace_dir)
    if host_tracer_level is not None:
        _config.host_tracer_level = host_tracer_level
    if device_tracer_level is not None:
        _config.device_tracer_level = device_tracer_level
    if python_tracer_level is not None:
        _config.python_tracer_level = python_tracer_level
    if warmup_iters is not None:
        _config.warmup_iters = warmup_iters
    if profile_iters is not None:
        _config.profile_iters = profile_iters


def _get_trace_path(name: str) -> Path:
    """Get the trace path for a given profile name."""
    _config.trace_dir.mkdir(parents=True, exist_ok=True)
    return _config.trace_dir / name


def _make_profiler_options() -> jax.profiler.ProfileOptions:
    """Create profiler options from current config."""
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = _config.host_tracer_level
    options.device_tracer_level = _config.device_tracer_level
    options.python_tracer_level = _config.python_tracer_level
    return options


@contextmanager
def profile(name: str, config: ProfileConfig | None = None):
    """Context manager for profiling a code block.

    Args:
        name: Name for this profile (used as directory name)
        config: Optional config override

    Usage:
        with profile("my_kernel"):
            result = my_kernel(x)
            result.block_until_ready()
    """
    cfg = config or _config
    trace_path = _get_trace_path(name)
    options = _make_profiler_options()

    print(f"Starting profile: {name}")
    print(f"Trace will be saved to: {trace_path}")

    jax.profiler.start_trace(str(trace_path), profiler_options=options)
    try:
        yield trace_path
    finally:
        jax.profiler.stop_trace()
        print(f"Profile saved to: {trace_path}")
        print(f"View with: xprof --port 8791 {trace_path}")


def profile_function(
    name: str,
    fn: Callable[[], Any],
    warmup_iters: int | None = None,
    profile_iters: int | None = None,
) -> Path:
    """Profile a function with warmup iterations.

    Args:
        name: Name for this profile
        fn: Function to profile (should call block_until_ready internally or return arrays)
        warmup_iters: Number of warmup iterations (default from config)
        profile_iters: Number of iterations to profile (default from config)

    Returns:
        Path to the trace directory
    """
    warmup = warmup_iters if warmup_iters is not None else _config.warmup_iters
    iters = profile_iters if profile_iters is not None else _config.profile_iters

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (list, tuple)):
            jax.block_until_ready(result)

    # Profile
    trace_path = _get_trace_path(name)
    with profile(name):
        for _ in range(iters):
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, (list, tuple)):
                jax.block_until_ready(result)

    return trace_path


def profile_comparison(
    name: str,
    *implementations: tuple[str, Callable[[], Any]],
    warmup_iters: int | None = None,
    profile_iters: int | None = None,
) -> dict[str, Path]:
    """Profile multiple implementations for comparison.

    Args:
        name: Base name for profiles
        *implementations: Tuples of (impl_name, fn) to profile
        warmup_iters: Number of warmup iterations
        profile_iters: Number of iterations to profile

    Returns:
        Dict mapping implementation names to trace paths

    Usage:
        paths = profile_comparison(
            "attention",
            ("flash", lambda: flash_attention(q, k, v)),
            ("reference", lambda: mha_reference(q, k, v)),
        )
    """
    results = {}
    for impl_name, fn in implementations:
        full_name = f"{name}_{impl_name}"
        print(f"\n{'='*60}")
        print(f"Profiling: {impl_name}")
        print('='*60)
        path = profile_function(full_name, fn, warmup_iters, profile_iters)
        results[impl_name] = path

    print(f"\n{'='*60}")
    print("All profiles complete. View with xprof:")
    print('='*60)
    for impl_name, path in results.items():
        print(f"  {impl_name}: xprof --port 8791 {path}")

    return results


def profile_forward_backward(
    name: str,
    forward_fn: Callable[[], Any],
    loss_fn: Callable[[], Any],
    warmup_iters: int | None = None,
    profile_iters: int | None = None,
) -> dict[str, Path]:
    """Profile both forward and backward passes.

    Args:
        name: Base name for profiles
        forward_fn: Function for forward pass only
        loss_fn: Function that computes a scalar loss (for backward pass)
        warmup_iters: Number of warmup iterations
        profile_iters: Number of iterations to profile

    Returns:
        Dict with 'forward' and 'backward' trace paths
    """
    # Create gradient function for backward pass
    grad_fn = jax.grad(lambda: loss_fn().sum())

    return profile_comparison(
        name,
        ("forward", forward_fn),
        ("backward", lambda: grad_fn()),
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
    )


def start_xprof_server(port: int = 8791, trace_dir: str | Path | None = None, blocking: bool = True):
    """Start the xprof server to view traces.

    Args:
        port: Port to serve on (default 8791)
        trace_dir: Directory containing traces (default: configured trace_dir)
        blocking: If True, block until server is stopped

    Usage:
        # On remote machine:
        start_xprof_server()

        # Then SSH tunnel from local:
        # ssh -L 8791:localhost:8791 user@remote_host

        # Open http://localhost:8791 in browser
    """
    trace_path = Path(trace_dir) if trace_dir else _config.trace_dir

    print(f"Starting xprof server on port {port}")
    print(f"Serving traces from: {trace_path}")
    print(f"\nTo view remotely:")
    print(f"  1. SSH tunnel: ssh -L {port}:localhost:{port} user@this_host")
    print(f"  2. Open: http://localhost:{port}")
    print(f"\nPress Ctrl+C to stop.")

    cmd = ["xprof", "--port", str(port), str(trace_path)]

    if blocking:
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nStopping xprof server.")
    else:
        return subprocess.Popen(cmd)


def list_traces() -> list[Path]:
    """List all available traces."""
    if not _config.trace_dir.exists():
        return []
    return sorted(_config.trace_dir.iterdir())


def print_traces():
    """Print all available traces with xprof commands."""
    traces = list_traces()
    if not traces:
        print(f"No traces found in {_config.trace_dir}")
        return

    print(f"Available traces in {_config.trace_dir}:")
    for trace in traces:
        print(f"  xprof --port 8791 {trace}")


# Convenience function for quick profiling
def quick_profile(name: str, fn: Callable[[], Any]) -> Path:
    """Quick one-liner profiling with sensible defaults.

    Usage:
        quick_profile("matmul", lambda: (x @ y).block_until_ready())
    """
    return profile_function(name, fn, warmup_iters=3, profile_iters=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JAX Profiling Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # xprof server command
    server_parser = subparsers.add_parser("serve", help="Start xprof server")
    server_parser.add_argument("--port", type=int, default=8791, help="Port to serve on")
    server_parser.add_argument("--trace-dir", type=str, default=str(DEFAULT_TRACE_DIR),
                               help="Directory containing traces")

    # list traces command
    list_parser = subparsers.add_parser("list", help="List available traces")
    list_parser.add_argument("--trace-dir", type=str, default=str(DEFAULT_TRACE_DIR),
                            help="Directory containing traces")

    args = parser.parse_args()

    if args.command == "serve":
        configure(trace_dir=args.trace_dir)
        start_xprof_server(port=args.port)
    elif args.command == "list":
        configure(trace_dir=args.trace_dir)
        print_traces()
    else:
        parser.print_help()
