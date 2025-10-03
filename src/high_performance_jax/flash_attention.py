import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

# Block sizes (tune for your hardware!)
BLOCK_M = 128   # query block size
BLOCK_N = 128   # key block size
BLOCK_D = 64    # head_dim (must divide head_dim evenly)

def flash_attention_kernel(q_ref, k_ref, v_ref, o_ref, *, seq_len, dim):
    # Program IDs: [batch, head, query_block]
    pid_b = pl.program_id(0)  # batch index
    pid_h = pl.program_id(1)  # head index
    pid_m = pl.program_id(2)  # query block index

    offs_m = pid_m * BLOCK_M + jnp.arange(BLOCK_M)
    offs_d = jnp.arange(BLOCK_D)
    mask_m = offs_m < seq_len

    # Load Q block: shape [BLOCK_M, dim]
    q_tile = pl.load(q_ref, (pid_b, pid_h, offs_m[:, None], offs_d[None, :]),
                     mask=mask_m[:, None])

    # Running online softmax stats
    m_i = jnp.full((BLOCK_M, 1), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((BLOCK_M, 1), dtype=jnp.float32)
    acc = jnp.zeros((BLOCK_M, BLOCK_D), dtype=jnp.float32)

    # Loop over K/V blocks
    def body(pid_n, state):
        m_i, l_i, acc = state
        offs_n = pid_n * BLOCK_N + jnp.arange(BLOCK_N)
        mask_n = offs_n < seq_len

        # Load K and V tiles
        k_tile = pl.load(k_ref, (pid_b, pid_h, offs_n[:, None], offs_d[None, :]),
                         mask=mask_n[:, None])
        v_tile = pl.load(v_ref, (pid_b, pid_h, offs_n[:, None], offs_d[None, :]),
                         mask=mask_n[:, None])

        # Attention scores
        scores = jnp.dot(q_tile, k_tile.T) / jnp.sqrt(dim)

        # Causal mask
        causal_mask = (offs_m[:, None] >= offs_n[None, :])
        scores = jnp.where(causal_mask & mask_n[None, :], scores, -jnp.inf)

        # Block max
        m_ij = jnp.max(scores, axis=-1, keepdims=True)
        new_m_i = jnp.maximum(m_i, m_ij)

        # Rescale old sums
        exp_scores = jnp.exp(scores - new_m_i)
        exp_scores = jnp.nan_to_num(exp_scores)

        exp_m_diff = jnp.exp(m_i - new_m_i)
        l_i = l_i * exp_m_diff + jnp.sum(exp_scores, axis=-1, keepdims=True)
        acc = acc * exp_m_diff + jnp.dot(exp_scores, v_tile)

        return (new_m_i, l_i, acc)

    def loop_body(pid_n, state):
        return body(pid_n, state)

    num_key_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N
    state = (m_i, l_i, acc)
    state = jax.lax.fori_loop(0, num_key_blocks, loop_body, state)
    m_i, l_i, acc = state

    # Normalize
    out_tile = acc / l_i

    # Store result
    pl.store(o_ref, (pid_b, pid_h, offs_m[:, None], offs_d[None, :]),
             out_tile, mask=mask_m[:, None])

# Wrapper
def flash_attention(q, k, v):
    """
    q, k, v: [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, dim = q.shape
    o = jnp.empty_like(q)

    grid = (batch, n_heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
    flash_attention_kernel_p = pl.pallas_call(
        flash_attention_kernel,
        out_shape=jax.ShapeDtypeStruct(o.shape, o.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec(q.shape, (None, None, BLOCK_M, BLOCK_D)), # q
            pl.BlockSpec(k.shape, (None, None, BLOCK_N, BLOCK_D)), # k
            pl.BlockSpec(v.shape, (None, None, BLOCK_N, BLOCK_D)), # v
        ],
        out_specs=pl.BlockSpec(o.shape, (None, None, BLOCK_M, BLOCK_D)),
        kernel_kwargs=dict(seq_len=seq_len, dim=dim)
    )

    return flash_attention_kernel_p(q, k, v)


def reference_attention(q, k, v, causal=True):
    """Naive reference: [B, H, N, D]."""
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if causal:
        N = q.shape[-2]
        mask = jnp.tril(jnp.ones((N, N), dtype=bool))
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    return out

def test_flash_attention(batch=2, heads=4, seq_len=128, dim=64, atol=1e-4, rtol=1e-4):
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, heads, seq_len, dim), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, heads, seq_len, dim), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, heads, seq_len, dim), dtype=jnp.float32)

    # Run reference
    ref_out = reference_attention(q, k, v)

    # Run Pallas implementation
    pallas_out = flash_attention(q, k, v)

    # Compare
    max_diff = jnp.max(jnp.abs(ref_out - pallas_out))
    all_close = jnp.allclose(ref_out, pallas_out, atol=atol, rtol=rtol)

    print(f"max diff = {max_diff:.6f}")
    print("Outputs match:", all_close)

    return all_close

# Run on GPU
if __name__ == "__main__":
    print("Device:", jax.devices()[0])
    ok = test_flash_attention(batch=2, heads=4, seq_len=128, dim=64)
    assert ok, "FlashAttention kernel does not match reference!"