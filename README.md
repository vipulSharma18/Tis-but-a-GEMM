# 'Tis but a GEMM

<div align="center">
  <img width="384" height="256" alt="image" src="https://github.com/user-attachments/assets/b2eda7b4-96f5-458a-afd2-65c77e8292ff" />
</div>

A tangent back to the basics of GPU programming.

Unique **Hardware Features**: https://docs.google.com/document/d/1_ct_ImtU7ulqEljhKXiBiT6t6G_icGmZXQyODSLPn5o/edit?tab=t.0

| GPU       | WMMA | WGMMA | 2CTA MMA | TMEM | TCGEN |
|------------|:----:|:-----:|:--------:|:----:|:-----:|
| A100       |    |     |        |    |     |
| H100       |    |     |        |    |     |
| RTX 4090   |    |     |        |    |     |
| B200       |    |     |        |    |     |
| RTX 5090   |    |     |        |    |     |

## Implementations in:

0. PTX
1. CUDA C++
2. Triton
3. Mojo
4. CUTLASS (CuTe DSL)
5. PyTorch Eager
6. PyTorch Compile with Triton
7. PyTorch Compile with TensorRT
