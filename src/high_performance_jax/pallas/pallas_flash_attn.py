import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

@jax.jit
def mha_reference(q, k, v):
    logits = jnp.einsum('bhqc,bhkc->bhqk', q, k)
    d = q.shape[-1]
    logits = logits / jnp.sqrt(d)
    probs = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum('bhqk,bhkc->bhqc', probs, v)
    return o

B = 8
N = 16
H = 4
C = 128
key = jax.random.key(0)
q = jax.random.normal(key, (B, H, N, C), dtype=jnp.float32)
k = jax.random.normal(key, (B, H, N, C), dtype=jnp.float32)
v = jax.random.normal(key, (B, H, N, C), dtype=jnp.float32)
o = mha_reference(q, k, v)

