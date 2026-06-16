import torch
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(x_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid  = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n
    x=tl.load(x_ptr + offsets,mask=mask,other=0.0)
    block_sumsq = tl.sum(x * x, axis=0)
    tl.atomic_add(sumsq_ptr, block_sumsq)
    pass


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch l2_norm_kernel and finalize the square root."""
    n = x.numel()
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    l2_norm_kernel[grid](x, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)
    out.copy_(torch.sqrt(sumsq_buf))