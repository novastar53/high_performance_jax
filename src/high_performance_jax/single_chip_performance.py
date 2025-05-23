import jax.numpy as jnp



def measure_tpu_hbm_memory_transfer(op, dim, dtype):
    bytes = jnp.dtype(dtype).itemsize

    match op:
        case "matadd":
            return 3 * (dim**2) * bytes 
        case "matadd3":
            return 6 * (dim**2) * bytes 
        case "matmul":
            return 3 * (dim**2) * bytes
        case _:
            return 0

def measure_tpu_flops(op, dim):
    match op:
        case "matadd":
            return dim**2
        case "matadd3":
            return dim**3
        case "matmul":
            return (dim**2)*(2*dim - 1)
        case _:
            return 0


def matadd(A, B):
    return A + B

def matadd3(A, B, C):
    return A + B + C

def matmul(A, B):
    return A @ B


