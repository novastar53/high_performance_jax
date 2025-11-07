import torch
import triton
import triton.language as tl

# -------- Host-side setup (HBM allocation) --------
# Make a simple 2D tensor A in GPU memory (HBM): values 0..(N*M-1)
N, M = 8, 8
A = torch.arange(N * M, device='cuda', dtype=torch.float32).reshape(N, M)
print(A.stride(0), A.stride(1))

# Choose a block (tile) to load: size HxW starting at (row0, col0)
BLOCK_H, BLOCK_W = 4, 4
row0, col0 = 2, 3

# Output buffer (on GPU as well), will hold the loaded tile
out = torch.empty((BLOCK_H, BLOCK_W), device='cuda', dtype=A.dtype)


# -------- Device kernel --------
@triton.jit
def load_block_kernel(
    A_ptr,          # *base* pointer to A (flattened)
    out_ptr,        # pointer to output tile (contiguous)
    N, M,           # shape of A
    row0, col0,     # tile top-left
    stride_row,     # A.stride(0)
    stride_col,     # A.stride(1)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # This kernel is tiny; one program handles the whole block
    pid = tl.program_id(0)  # we won’t use pid for multi-tiling here

    # Row/col indices for the tile we want to load
    offs_i = row0 + tl.arange(0, BLOCK_H)                  # [BLOCK_H]
    offs_j = col0 + tl.arange(0, BLOCK_W)                  # [BLOCK_W]

    # Build a boolean mask to guard out-of-bounds loads
    mask = (offs_i[:, None] < N) & (offs_j[None, :] < M)   # [BLOCK_H, BLOCK_W]

    # Compute a 2D tensor of POINTERS into A:
    #   ptrs[i,j] = A_ptr + offs_i[i]*stride_row + offs_j[j]*stride_col
    ptrs = (
        A_ptr
        + offs_i[:, None] * stride_row
        + offs_j[None, :] * stride_col
    )

    # Load the tile into registers (masked OOB as 0.0)
    tile = tl.load(ptrs, mask=mask, other=0.0)             # [BLOCK_H, BLOCK_W]

    # Store to a contiguous output buffer so we can read it back easily
    out_ptrs = (
        out_ptr
        + tl.arange(0, BLOCK_H)[:, None] * stride_row
        + tl.arange(0, BLOCK_W)[None, :] * stride_col
    )
    tl.store(out_ptrs, tile)


# -------- Launch --------
grid = (1,)  # single program; this kernel handles one block
load_block_kernel[grid](
    A_ptr=A,
    out_ptr=out,
    N=N,
    M=M,
    row0=row0,
    col0=col0,
    stride_row=A.stride(0),
    stride_col=A.stride(1),
    BLOCK_H=BLOCK_H,
    BLOCK_W=BLOCK_W,
)

print("A (HBM):\n", A.cpu().numpy())
print(f"\nLoaded block A[{row0}:{row0+BLOCK_H}, {col0}:{col0+BLOCK_W}] -> out:\n", out.cpu().numpy())