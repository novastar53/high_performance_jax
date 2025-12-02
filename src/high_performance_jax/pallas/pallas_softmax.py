import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plgpu

D = 256
key = jax.random.key(0)
logits = jax.random.normal(shape=(D, D), key=key)


# Basic softmax
probs_ref = jax.nn.softmax(logits, axis=-1)


# Manual softmax (jax)
max_rows = jnp.max(logits, axis=-1)
s = jnp.exp(logits - max_rows[..., None])
l = jnp.sum(s, axis=-1)
l = l[..., None]
probs_manual = s / l 

assert(jnp.allclose(probs_ref, probs_manual))

# Pallas softmax
G = 8 # Num groups

def softmax_kernel(x_ref, o_ref):
    x_reg = x_ref[...]
    o_reg = jnp.exp(x_reg)
    o_ref[...] = o_reg


@jax.jit
def softmax(logits):
    result = pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(logits.shape, logits.dtype),
        grid=(G, G),
        in_specs=[pl.BlockSpec((logits.shape[0] // G, logits.shape[1] // G), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((logits.shape[0] // G, logits.shape[1] // G), lambda i, j: (i, j))
    )(logits)
    return result


probs_pl = softmax(logits)
s = jnp.exp(logits)
assert(jnp.allclose(probs_pl, s))
