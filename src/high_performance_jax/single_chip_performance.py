import matplotlib.pyplot as plt
from datetime import datetime
import random
import string
import jax
import jax.numpy as jnp


from deepkit.utils import timeit

def measure_tpu_hbm_memory_transfer(A, B, op, dtype):
    bytes = jnp.dtype(dtype).itemsize

    match op:
        case "matadd":
            return 3 * (A.size + B.size) * bytes 
        case "matmul":
            return (A.size + B.size + A.shape[0]*B.shape[1]) * bytes
        case _:
            return 0

def measure_tpu_flops(A, B, op):
    match op:
        case "matadd":
            M, N = A.shape
            return M * N
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

    dtype = jnp.uint8
    print(jax.devices())
    jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") # Set the default precision for matrix multiplication

    dims = [2**i for i in range(2, 15)]
    times = []
    tflops = []
    hbm_rates = []

    for dim in dims:
        A = jnp.ones((dim, dim), dtype=dtype)
        B = jnp.ones((dim, dim), dtype=dtype)
        task = "matmul"

        average_time_ms = timeit(jax.jit(matmul), A, B)
        flops = measure_tpu_flops(A, B, task)
        times.append(average_time_ms)
        tflops.append(1000 * flops / average_time_ms / 10**12)
        hbm_xfer = measure_tpu_hbm_memory_transfer(A, B, task, dtype)
        hbm_rates.append(1000 * hbm_xfer / average_time_ms)

        print(f"dim {dim} | average time (ms): {average_time_ms:.2f} | "
              f"hbm xfer/s {hbm_rates[-1]:0.4f} | "
              f"flops {flops:,} | "
              f"tera flops/s {tflops[-1]:0.4f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(dims, times, marker='o')
    ax1.set_xscale('log')
    ax1.set_xlabel("Matrix Dimension (dim)")
    ax1.set_ylabel("Average Time (ms)")
    ax1.set_title("Matrix Multiplication Time vs Dimension")
    ax1.grid(True)

    ax2.plot(dims, tflops, marker='o')
    ax2.set_xscale('log')
    ax2.set_xlabel("Matrix Dimension (dim)")
    ax2.set_ylabel("TFLOPS/s")
    ax2.set_title("Performance in TFLOPS/s vs Dimension")
    ax2.grid(True)

    ax3.plot(dims, hbm_rates, marker='o')
    ax3.set_xscale('log')
    ax3.set_xlabel("Matrix Dimension (dim)")
    ax3.set_ylabel("HBM Transfer Rate (bytes/s)")
    ax3.set_title("HBM Transfer Rate vs Dimension")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
