"""cuTile matrix multiplication integrated with JAX.

This module provides a JAX-compatible matrix multiplication using cuTile kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

try:
    import cuda.tile as ct
except ImportError as e:
    raise ImportError(
        "cuTile not installed. Install with: pip install cuda-tile"
    ) from e


# Type alias for cuTile constants
from typing import TypeVar
ConstInt = TypeVar('ConstInt', bound=int)


@ct.kernel
def matmul_kernel_f32(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """Simple matmul kernel for float32.

    Each block computes a tm x tn tile of the output C.
    """
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    # Get block indices
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    # Number of tiles in K dimension
    num_tiles_k = ct.ceildiv(K, tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)

    # Loop over K dimension
    for k_idx in range(num_tiles_k):
        # Load tiles from A and B
        a_tile = ct.load(A, index=(bidx, k_idx), shape=(tm, tk))
        b_tile = ct.load(B, index=(k_idx, bidy), shape=(tk, tn))

        # Matrix multiply-accumulate
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # Store result
    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.kernel
def matmul_kernel_f16(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """Optimized matmul kernel for float16/bfloat16 using Tensor Cores."""
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    bidx = ct.bid(0)
    bidy = ct.bid(1)

    num_tiles_k = ct.ceildiv(K, tk)
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)

    for k_idx in range(num_tiles_k):
        # Load and cast to appropriate type for Tensor Cores
        a_tile = ct.load(A, index=(bidx, k_idx), shape=(tm, tk))
        b_tile = ct.load(B, index=(k_idx, bidy), shape=(tk, tn))
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # Cast back to output dtype and store
    result = accumulator.astype(C.dtype)
    ct.store(C, index=(bidx, bidy), tile=result)


def cutile_matmul_host(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Host-side wrapper for cuTile matmul.

    This function is called via jax.pure_callback to launch the cuTile kernel.

    Args:
        A: Input matrix of shape (M, K)
        B: Input matrix of shape (K, N)

    Returns:
        Output matrix of shape (M, N)
    """
    import torch

    # Convert JAX arrays to PyTorch tensors via DLPack
    A_torch = torch.from_dlpack(A)
    B_torch = torch.from_dlpack(B)

    M, K = A_torch.shape
    K2, N = B_torch.shape
    assert K == K2, f"Shape mismatch: {K} != {K2}"

    # Determine tile sizes based on dtype
    if A_torch.dtype in (torch.float16, torch.bfloat16):
        tm, tn, tk = 128, 256, 64
        kernel = matmul_kernel_f16
    else:
        tm, tn, tk = 64, 64, 32
        kernel = matmul_kernel_f32

    # Calculate grid dimensions
    grid_m = (M + tm - 1) // tm
    grid_n = (N + tn - 1) // tn
    grid = (grid_m, grid_n, 1)

    # Allocate output
    C_torch = torch.empty((M, N), device=A_torch.device, dtype=A_torch.dtype)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (A_torch, B_torch, C_torch, tm, tn, tk)
    )

    # Convert back to JAX array via DLPack
    return jnp.from_dlpack(C_torch)


def cutile_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """JAX-compatible cuTile matrix multiplication.

    Args:
        A: Left-hand side matrix, shape (..., M, K)
        B: Right-hand side matrix, shape (..., K, N)

    Returns:
        Result matrix, shape (..., M, N)

    Example:
        >>> A = jax.random.normal(jax.random.key(0), (1024, 512))
        >>> B = jax.random.normal(jax.random.key(1), (512, 2048))
        >>> C = cutile_matmul(A, B)  # Shape: (1024, 2048)
    """
    # Handle batch dimensions by vmap
    if A.ndim > 2 or B.ndim > 2:
        # Find batch shape
        batch_A = A.shape[:-2] if A.ndim > 2 else ()
        batch_B = B.shape[:-2] if B.ndim > 2 else ()

        if batch_A != batch_B:
            # Broadcasting
            batch_shape = jnp.broadcast_shapes(batch_A, batch_B)
            A = jnp.broadcast_to(A, batch_shape + A.shape[-2:])
            B = jnp.broadcast_to(B, batch_shape + B.shape[-2:])

        # Use vmap over batch dimensions
        return jax.vmap(cutile_matmul, in_axes=(0, 0))(A, B)

    # 2D case: use pure_callback
    result_shape = jax.ShapeDtypeStruct(
        shape=(A.shape[0], B.shape[1]),
        dtype=A.dtype
    )

    return jax.pure_callback(
        cutile_matmul_host,
        result_shape,
        A, B,
        vmap_method="sequential"
    )


def test_cutile_matmul():
    """Test the cuTile matmul against JAX reference."""
    print("Testing cuTile matmul integration with JAX...")

    # Test shapes
    test_cases = [
        (128, 128, 128),
        (256, 512, 256),
        (1024, 1024, 1024),
    ]

    for M, K, N in test_cases:
        print(f"\nTesting shape: A({M}, {K}) @ B({K}, {N})")

        # Create random inputs
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)

        for dtype in [jnp.float32, jnp.float16]:
            A = jax.random.normal(key1, (M, K), dtype=dtype)
            B = jax.random.normal(key2, (K, N), dtype=dtype)

            # cuTile result
            C_cutile = cutile_matmul(A, B)
            C_cutile.block_until_ready()

            # JAX reference
            C_ref = A @ B

            # Compare
            atol = 1e-2 if dtype == jnp.float16 else 1e-4
            rtol = 1e-2 if dtype == jnp.float16 else 1e-4

            max_diff = jnp.max(jnp.abs(C_cutile - C_ref))
            print(f"  dtype={dtype.__name__}: max_diff={max_diff:.6e}", end="")

            try:
                np.testing.assert_allclose(C_cutile, C_ref, atol=atol, rtol=rtol)
                print(" ✓")
            except AssertionError as e:
                print(f" ✗\n    {e}")
                raise

    # Test batched matmul
    print("\nTesting batched matmul...")
    batch = 4
    M, K, N = 128, 128, 128

    A = jax.random.normal(key1, (batch, M, K))
    B = jax.random.normal(key2, (batch, K, N))

    C_cutile = cutile_matmul(A, B)
    C_ref = jax.lax.batch_matmul(A, B)

    max_diff = jnp.max(jnp.abs(C_cutile - C_ref))
    print(f"  Batched ({batch}, {M}, {K}) @ ({batch}, {K}, {N}): max_diff={max_diff:.6e}", end="")

    try:
        np.testing.assert_allclose(C_cutile, C_ref, atol=1e-4, rtol=1e-4)
        print(" ✓")
    except AssertionError as e:
        print(f" ✗\n    {e}")
        raise

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_cutile_matmul()
