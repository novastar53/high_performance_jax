import jax
import jax.numpy as jnp


def measure_tpu_hbm_memory_transfer(op, n, dim, dtype):
    bytes = jnp.dtype(dtype).itemsize

    match op:
        case "matadd":
            return n * 3 * (dim**2) * bytes 
        case "matadd3":
            return n * 6 * (dim**2) * bytes 
        case "matmul":
            return n * 3 * (dim**2) * bytes
        case "matmul3":
            return n * 6 * (dim**2) * bytes
        case _:
            return 0

def measure_tpu_flops(op, n, dim):
    match op:
        case "matadd":
            return n * dim**2
        case "matadd3":
            return n * 2*dim**2
        case "matmul":
            return n * (dim**2)*(2*dim - 1)
        case "matmul3":
            return n * 2*(dim**2)*(2*dim - 1)
        case _:
            return 0


def matadd(A, B):
    return A + B

def matadd3(A, B, C):
    return A + B + C

def matmul(A, B):
    return A @ B

def matmul3(A, B, C):
    return A @ B @ C

def bmm(A, B):
    return jax.lax.batch_matmul(A, B)