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
            return 3 * A.size * bytes 
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


