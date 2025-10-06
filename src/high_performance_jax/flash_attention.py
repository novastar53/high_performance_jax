import jax
import jax.numpy as jnp

# online softmax
x = jnp.array([1, 5, 2, 8])
z = 0
max_x = float("-inf")
for i in range(len(x)):
    new_max_x = max(x[0], max_x)
    z = z*jnp.exp(max_x - new_max_x) + jnp.exp(x[i] - new_max_x)
    max_x = new_max_x

result = jnp.exp(x - max_x) / z
standard_result = jax.nn.softmax(x)
assert(jnp.allclose(result, standard_result))


def print_grid(N, filled_cells):
    """
    Prints an NxN grid.
    
    Parameters:
    - N: size of the grid (int)
    - filled_cells: list of tuples (row, col) representing filled cells
    """
    # Convert filled_cells to a set for O(1) lookups
    filled = set(filled_cells)

    for i in range(N):
        row = ""
        for j in range(N):
            if (i, j) in filled:
                row += "X "   # filled cell
            else:
                row += ". "   # empty cell
        print(row.strip())

N, D = 24, 8

q = jax.random.randint(
    jax.random.key(0),
    (N, D), -5, 5
)
k = jax.random.randint(
    jax.random.key(1),
    (N, D), -5, 5
)
v = jax.random.randint(
    jax.random.key(2),
    (N, D), -5, 5
)


# regular attention
a = jnp.einsum('nd,md->nm', q, k)
a = jax.nn.softmax(a, axis=-1)
ref_result = jnp.einsum('nm,md->nd', a, v)


# manual attention
s = q @ k.T
manual_m = jnp.max(s, axis=-1)
s = s - manual_m[..., None]
manual_p = jnp.exp(s)
manual_z = jnp.sum(manual_p, axis=-1)
manual_result_unnorm = manual_p @ v
manual_result = manual_result_unnorm / manual_z[..., None]

assert(jnp.allclose(ref_result, manual_result))


# flash attention (python)
block_size = 4
n_blocks = N // block_size

computed_blocks = []
result = jnp.zeros_like(q, dtype=jnp.float32)
z = jnp.zeros((N,), dtype=jnp.float32)
m = jnp.full((N,), float('-inf'))
for row in range(n_blocks):
    q_blk = q[row*block_size:(row+1)*block_size, :]
    m_blk = m[row*block_size:(row+1)*block_size]
    z_blk = z[row*block_size:(row+1)*block_size]
    result_blk = result[row*block_size:(row+1)*block_size, :]
    for col in range(n_blocks):
        k_blk = k.T[:, col*block_size:(col+1)*block_size] 
        v_blk = v[col*block_size:(col+1)*block_size, :]
        s_blk = q_blk @ k_blk
        m_blk_old = m_blk
        m_blk = jnp.maximum(m_blk, jnp.max(s_blk, axis=-1))
        correction_factor = jnp.exp(m_blk_old - m_blk)
        p_blk = jnp.exp(s_blk - m_blk[..., None])
        z_blk = z_blk * correction_factor + jnp.sum(p_blk, axis=-1)
        r = p_blk @ v_blk
        result_blk = result_blk * correction_factor[..., None] + r
    m = m.at[row*block_size:(row+1)*block_size].set(m_blk)
    z = z.at[row*block_size:(row+1)*block_size].set(z_blk)
    result = result.at[row*block_size:(row+1)*block_size, :].set(result_blk)


assert(jnp.allclose(m, manual_m))
assert(jnp.allclose(z, manual_z))
assert(jnp.allclose(result, manual_result_unnorm))
result /= z[..., None]
assert(jnp.allclose(result, manual_result))



# flash attention (triton)
import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, dtype=torch.float16):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V  = Q.shape[-1], K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ) 
        .normal_(mean=0.0, std=0.5) 
        .requires_grad_()
    )

    K = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ) 
        .normal_(mean=0.0, std=0.5) 
        .requires_grad_()
    )

    V = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ) 
        .normal_(mean=0.0, std=0.5) 
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)



    # reference attention (torch)
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device='cuda'))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None


    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward()
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)


    










# flash attention (pallas)
# TODO