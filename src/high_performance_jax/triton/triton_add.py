import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKSIZE
    offsets = block_start + tl.arange(0, BLOCKSIZE)
    mask = tl.full(offsets.shape, 1, tl.int1) # > n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)



def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x) 
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCKSIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCKSIZE=1024)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)

print(torch.allclose(output_torch, output_triton))