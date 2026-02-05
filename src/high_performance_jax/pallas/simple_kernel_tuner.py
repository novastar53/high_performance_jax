"""Simple flash attention kernel tuner.

Prints optimal hyperparameters for given Q, K, V shapes.
"""


def get_optimal_params(seq_len: int, head_dim: int = 64):
    """Get optimal kernel parameters for sequence length.

    Args:
        seq_len: Sequence length (T dimension)
        head_dim: Head dimension

    Returns:
        tuple: (block_r, block_c, num_warps, num_stages)
    """
    if seq_len <= 512:
        # Short sequences: small blocks, default config
        return 64, 64, 4, 3
    elif seq_len <= 2048:
        # Medium sequences: taller blocks for fewer kernel launches
        return 128, 64, 4, 3
    elif seq_len <= 4096:
        # Long sequences: max blocks, more warps for utilization
        return 128, 128, 8, 4
    else:
        # Very long sequences: max blocks, extra pipeline stage
        return 128, 128, 8, 5


def print_optimal_params(q_shape, k_shape, v_shape):
    """Print optimal hyperparameters for given shapes.

    Args:
        q_shape: Query shape (B, H, T, D)
        k_shape: Key shape (B, H, T, D)
        v_shape: Value shape (B, H, T, D)
    """
    B, H, T, D = q_shape
    head_dim = D

    block_r, block_c, num_warps, num_stages = get_optimal_params(T, head_dim)

    print("=" * 70)
    print("Optimal Flash Attention Kernel Parameters")
    print("=" * 70)
    print("Input shapes:")
    print(f"  Q: {q_shape}")
    print(f"  K: {k_shape}")
    print(f"  V: {v_shape}")
    print(f"\nOptimal parameters for seq_len={T}:")
    print(f"  BLOCK_R = {block_r}")
    print(f"  BLOCK_C = {block_c}")
    print(f"  NUM_WARPS = {num_warps}")
    print(f"  NUM_STAGES = {num_stages}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print optimal flash attention kernel parameters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (B)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of heads (H)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length (T)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (D)",
    )

    args = parser.parse_args()

    q_shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
    k_shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
    v_shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)

    print_optimal_params(q_shape, k_shape, v_shape)
