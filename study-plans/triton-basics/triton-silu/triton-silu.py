import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0,BLOCK_SIZE)
    mask= offsets < n
    x=tl.load(x_ptr+offsets,mask=mask,other=0.0)
    y=x/(1+tl.exp(-x))
    tl.store(out_ptr+offsets,y,mask=mask)
    pass


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)