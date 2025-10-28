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
a = jnp.einsum('nd,md->nm', q, k) / jnp.sqrt(D)
a = jax.nn.softmax(a, axis=-1)
ref_result = jnp.einsum('nm,md->nd', a, v)


# manual attention
s = q @ k.T / jnp.sqrt(D)
manual_m = jnp.max(s, axis=-1)
s = s - manual_m[..., None]
manual_p = jnp.exp(s)
manual_z = jnp.sum(manual_p, axis=-1)
manual_result_unnorm = manual_p @ v
manual_result = manual_result_unnorm / manual_z[..., None]

assert(jnp.allclose(ref_result, manual_result, atol=0.01))


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
        s_blk = q_blk @ k_blk / jnp.sqrt(D)
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
assert(jnp.allclose(result, manual_result_unnorm, atol=1e-2))
result /= z[..., None]
assert(jnp.allclose(result, manual_result, atol=1e-2))



# flash attention (triton)
import torch
import triton
import triton.language as tl


def _attn_fwd_inner(
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # from 0 to the left of the diagonal
        lo, hi, = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN
    
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BlOCK_SIZE_KV):
        # Just let the compiler know that start_n 
        # is a multiple of BLOCCK_N, 
        # so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk -- #
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) * softmax_scale
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
        # Compute the exponential of each dot product, so now we are computing exp(q_ki) -m_ij)
        P_block = tl.math.exp(QK_block)
    
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)

        # Apply the correction factor to the previous l_i and and the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        # This computes the following: Q_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # O_block += P_block @ V_block

        m_i = m_ij 

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V[SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # # K[HEAD_DIM, SEQ_LEN]



@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with
    # a single head or a single batch
    index_batch_head = tl.program_id(1)
    
    # This indicates which batch this program is associated with
    # (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS

    # This indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by selecting
    # indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, #Q[index_batch, index_head, block_index_q * BLOCK_SIZE_q, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr( # V[index_batch, index_head, :, :]
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr( # K[index_batch, index_head, :, :]
        base = K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
            # We invert the strides w.r.t Q, so we transpose the matrix
        ),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1)
    )

    Q_block_ptr = tl.make_block_ptr( # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) 
    
    # offs_v: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV) # 0, 1, 2, 3

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')   

    # l_i: the running sum. We have one for each query (as we sum the attention score by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This tep runs for non-causal attention or for the blocks
        # to the left of the diagonal in causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q, 
            offs_kv,
            SEQ_LEN,
        )
    

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    m_i += tl.math.log(l_i) # This is needed to compute the logsumexp for the backwards pass

    O_block = O_block / l_i[i, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load( # O [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # (BLOCK_SIZEQ, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of DO
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
    )


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale, dtype=torch.float16):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V  = Q.shape[-1], K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # Which group of queries are we going to work with? 
            BATCH_SIZE * NUM_HEADS, # Which head of which batch element are we going to work with?
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return 0
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # Compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
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


test_op(BATCH_SIZE=1, NUM_HEADS=2, SEQ_LEN=4, HEAD_DIM=8, causal=False)
    




# flash attention (pallas)
# TODO
