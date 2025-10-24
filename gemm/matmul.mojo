from math import iota
from sys import exit
from sys.info import has_accelerator
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx

alias block_size = 16
alias num_rows = 1024
alias num_cols = 1024
alias num_reduction = 4096


fn addmm(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    Bias: UnsafePointer[Float32],
    C: UnsafePointer[Float32]
    ) -> Matrix:
    """
    A@B + Bias = C
    """
    M, K, N = C.shape[0], A.shape[1], C.shape[1]
    row_idx = block_idx.x * block_dim.x + thread_idx.x
    col_idx = block_idx.y * block_dim.y + thread_idx.y
    idx = row_idx * K + col_idx
    if(row_idx<M and col_idx<N):
    


def main():
    @parameter
    if has_accelerator():
        print("Using accelerator and proceeding with matmul.")
        ctx = DeviceContext()
        A_h = ctx.enqueue_create_host_buffer[DType.float32](
            num_rows*num_reduction
        )
        B_h = ctx.enqueue_create_host_buffer[DType.float32](
            num_reduction*num_cols
        )
        C_h = ctx.enqueue_create_host_buffer[DType.float32](
            num_rows*num_cols
        )
        Bias_h = ctx.enqueue_create_host_buffer[DType.float32](
            num_rows
        )
        ctx.synchronize()

        iota(A_h.unsafe_ptr(), num_rows*num_reduction)
        iota(B_h.unsafe_ptr(), num_reduction*num_cols)
        iota(Bias_h.unsafe_ptr(), num_rows)
        print("A:", A_h)
        print("B:", B_h)
        print("Bias:", Bias_h)

        A_d = ctx.enqueue_create_buffer[DType.float32](
            num_rows*num_reduction
        )
        B_d = ctx.enqueue_create_buffer[DType.float32](
            num_reduction*num_cols
        )
        Bias_d = ctx.enqueue_create_buffer[DType.float32](
            num_rows
        )
        C_d = ctx.enqueue_create_buffer[DType.float32](
            num_rows*num_cols
        )

        # copy from host to device
        ctx.enqueue_copy(src_buf = A_h, dst_buf = A_d)
        ctx.enqueue_copy(src_buf = B_h, dst_buf = B_d)
        ctx.enqueue_copy(src_buf = Bias_h, dst_buf = Bias_d)

        addmm_kernel = ctx.compile_function_checked[
            addmm, addmm
        ]()

        ctx.enqueue_function_checked(
            addmm_kernel,
            A_d,
            B_d,
            Bias_d,
            C_d,
            grid_dim=((num_rows+1//block_size), (num_cols+1//block_size)),
            block_dim=(block_size, block_size),
        )

        ctx.enqueue_copy(src_buf=C_d, dst_buf=C_h)

        ctx.synchronize()
    else:
        print("No accelerator found. Exiting.")
        exit(0)