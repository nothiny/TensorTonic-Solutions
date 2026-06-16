import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_x = x_ptr + row_idx * x_row_stride
    row_start_out = out_ptr + row_idx * out_row_stride
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # 加载整行，无效位置用 -inf 填充，避免影响 max
    x = tl.load(row_start_x + cols, mask=mask, other=float("-inf"))
    
    # 行最大值（标量）
    row_max = tl.max(x, axis=0)
    
    # 减去最大值，指数化
    x_shifted = x - row_max
    e = tl.exp(x_shifted)
    # 将无效位置显式置 0（虽然 exp(-inf)=0，但为了数值安全和后续求和）
    e = tl.where(mask, e, 0.0)
    
    # 行总和
    row_sum = tl.sum(e, axis=0)
    
    # 归一化
    out = e / row_sum
    tl.store(row_start_out + cols, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )