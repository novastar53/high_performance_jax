from datetime import datetime
import random
import string
import jax
import jax.numpy as jnp


from deepkit.utils import timeit


def measure_tpu_flops(A, B, op):
    match op:
        case "matadd":
            return (A.size + B.size)
        case "matmul":
            M, N = A.shape
            N, P = B.shape
            return M*P*(2*N - 1)
        case _:
            return 0


def matadd(A, B):
    return A + B


def matmul(A, B):
    return A @ B


def main():

    print(jax.devices())
    jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") # Set the default precision for matrix multiplication

    dim = 2**16
    A = jnp.ones((dim, dim), dtype=jnp.uint8)
    B = jnp.ones((dim, dim), dtype=jnp.uint8)
    task = "matmul"

    average_time_ms, trace_dir = timeit(matmul, A, B, task=task)
    flops = measure_tpu_flops(A, B, task)
    print(f"trace {trace_dir}: average time milliseconds: {average_time_ms:.2f} | "
          f"flops {flops:,} | "  
          f"tera flops/s {1000*flops/average_time_ms/10**12:0.4f}")

    matmul_jit = jax.jit(matmul)

    average_time_ms, trace_dir = timeit(matmul, A, B, task=task)
    print(f"trace {trace_dir}: average time milliseconds: {average_time_ms:.2f} | "
          f"flops {flops:,} | "  
          f"tera flops/s {1000*flops/average_time_ms/10**12:0.4f}")


if __name__ == "__main__":
    main()
