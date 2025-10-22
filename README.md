# 'Tis but a GEMM

<div align="center">
  <img width="384" height="256" alt="image" src="https://github.com/user-attachments/assets/b2eda7b4-96f5-458a-afd2-65c77e8292ff" />
</div>

A tangent back to the basics of GPU programming. How hard could it be?

| GPU        | WMMA | WGMMA | 2CTA MMA | TMEM | TCGEN |
|------------|:----:|:-----:|:--------:|:----:|:-----:|
| RTX 3090   |      |   X   |    x     |  x   |   x   |
| A100       |      |   X   |    x     |  x   |   x   |
| RTX 4090   |      |   X   |    x     |  x   |   x   |
| H100       |      |       |          |  x   |   x   |
| B200       |      |       |          |      |       |
| RTX 5090   |      |       |    ?     |  ?   |   x   |

## Implementations in:

0. PTX
1. CUDA C++
2. Triton
3. Mojo
4. CUTLASS (CuTe DSL)
5. PyTorch Eager
6. PyTorch Compile with Triton
7. PyTorch Compile with TensorRT
