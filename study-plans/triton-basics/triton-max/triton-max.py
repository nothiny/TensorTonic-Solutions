import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0,BLOCK_SIZE)
    mask=offsets < n
    x=tl.load(x_ptr+offsets,mask=mask,other=float("-inf"))
    o=tl.max(x,axis=0)
    tl.atomic_max(out_ptr,o)
    pass


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch max_kernel on the provided tensor with a single-program reduction."""
    if x.dtype.is_floating_point:
        out.fill_(float("-inf"))
    else:
        out.fill_(torch.iinfo(x.dtype).min)
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)