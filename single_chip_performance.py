from datetime import datetime
import random
import string
import jax
import jax.numpy as jnp


from deepkit.utils import timeit


def matadd(A, B):
    return A + B


def matmul(A, B):
    return A @ B


def main():
    dim = 2**14
    A = jnp.ones((dim, dim))
    B = jnp.ones((dim, dim))
    task = "matadd"

    average_time_ms, trace_dir = timeit(matadd, A, B, task=task)
    print(f"trace {trace_dir}: average time milliseconds: {average_time_ms:.2f}")

    matadd_jit = jax.jit(matadd)

    average_time_ms, trace_dir = timeit(matadd_jit, A, B, task=task)
    print(f"trace {trace_dir}: average time milliseconds: {average_time_ms:.2f}")

if __name__ == "__main__":
    main()
