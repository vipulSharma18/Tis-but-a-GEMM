# Tis-but-a-GEMM
I can optimize GEMM all day.

A tangent back to the basics of GPU programming.

Unique Hardware Features: https://docs.google.com/document/d/1_ct_ImtU7ulqEljhKXiBiT6t6G_icGmZXQyODSLPn5o/edit?tab=t.0

GPUs that come to mind: A100, H100, RTX 4090, B200, RTX 5090

Features that come to mind: WMMA, WGMMA, 2CTA MMA, TMEM, and TCGEN.

Implementations in:

0. PTX
1. CUDA C++
2. Triton
3. Mojo
4. CUTLASS (CuTe DSL)
5. PyTorch Eager
6. PyTorch Compile with Triton
7. PyTorch Compile with TensorRT
