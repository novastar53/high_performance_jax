# High Performance JAX

Tutorials for high-performance JAX: custom GPU kernels (Pallas/Triton), model sharding, and attention mechanisms.

## Modules

```
src/high_performance_jax/
├── pallas/
│   ├── pallas_softmax.py          # Online softmax with custom forward/backward
│   ├── pallas_flash_attn.py       # Flash attention implementation
│   ├── pallas_matmul_naive.py     # Matrix multiplication kernel
│   ├── pallas_add.py              # Simple addition kernel
│   ├── benchmarks.py              # Benchmark utilities
│   └── simple_kernel_tuner.py     # Kernel parameter tuning
├── triton/                        # Native Triton kernels (Linux + NVIDIA only)
│   ├── flash_attention.py
│   └── triton_add.py
├── profiling.py                   # XProf-based profiling utilities
├── model_sharding.py              # Flax NNX sharding patterns
├── moe.py                         # Mixture of Experts with expert parallelism
├── sharded_vmap.py                # Combining vmap with sharding
├── adafactor.py                   # Adafactor optimizer
└── fp8_training.py                # FP8 precision training
```

## Pallas Flash Attention Benchmarks

```
Config: BLOCK_R=128, BLOCK_C=128, NUM_WARPS=8, NUM_STAGES=5, CAUSAL=True, INTERPRET_MODE=False
Testing with shapes: B=2, H=4, T=8192, D=64
flash_attn_jax reference check passed!
Forward pass check passed!
Preprocess kernel (D) check passed!
Backward pass check passed!

============================================================
Timing Comparison
============================================================
Benchmark shape: B=2, H=4, T=8192, D=64

Forward pass:
  JAX dot_product_attention: 1.461 ms
  Our flash_attention:       1.745 ms
  flash_attn_jax (C++ CUDA):  1.361 ms

Backward pass only:
  JAX dot_product_attention: 4.062 ms
  Our flash_attention:       6.092 ms
  flash_attn_jax (C++ CUDA):  3.257 ms

Total (Forward + Backward):
  JAX dot_product_attention: 5.524 ms
  Our flash_attention:       7.838 ms
  flash_attn_jax (C++ CUDA):  4.618 ms

Config: BLOCK_R=128, BLOCK_C=128, NUM_WARPS=4, NUM_STAGES=3, CAUSAL=True, INTERPRET_MODE=False
Testing with shapes: B=2, H=4, T=4096, D=64
flash_attn_jax reference check passed!
Forward pass check passed!
Preprocess kernel (D) check passed!
Backward pass check passed!

============================================================
Timing Comparison
============================================================
Benchmark shape: B=2, H=4, T=4096, D=64

Forward pass:
  JAX dot_product_attention: 0.540 ms
  Our flash_attention:       0.805 ms
  flash_attn_jax (C++ CUDA):  0.559 ms

Backward pass only:
  JAX dot_product_attention: 1.451 ms
  Our flash_attention:       3.466 ms
  flash_attn_jax (C++ CUDA):  1.256 ms

Total (Forward + Backward):
  JAX dot_product_attention: 1.990 ms
  Our flash_attention:       4.271 ms
  flash_attn_jax (C++ CUDA):  1.815 ms
```
