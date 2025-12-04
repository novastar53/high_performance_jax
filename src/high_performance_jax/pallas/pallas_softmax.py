import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
#from jax.experimental.pallas import triton as plgpu

D = 8
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
G = 2 # Num groups

def softmax_kernel(x_ref, mi_ref, o_ref, mo_ref):
    x_reg = x_ref[...]
    x_max_reg = jnp.max(x_reg, axis=-1)[..., None]
    mo_reg = jnp.maximum(mi_ref[...], x_max_reg)
    o_reg = jnp.exp(x_reg)
    o_ref[...] = o_reg
    mi_ref[...] = mo_reg
    mo_ref[...] = mo_reg


@jax.jit
def softmax(logits, m):
    result = pl.pallas_call(
        softmax_kernel,
        out_shape=[jax.ShapeDtypeStruct(logits.shape, logits.dtype), jax.ShapeDtypeStruct(m.shape, m.dtype)],
        grid=(G, G),
        in_specs=[pl.BlockSpec((logits.shape[0] // G, logits.shape[1] // G), lambda i, j: (i, j)),
                  pl.BlockSpec((m.shape[0] // G, 1), lambda i, j: (i, 0))],
        out_specs=[pl.BlockSpec((logits.shape[0] // G, logits.shape[1] // G), lambda i, j: (i, j)),
                   pl.BlockSpec((m.shape[0] // G, 1), lambda i, j: (i, 0))],
        interpret=True
    )(logits, m)
    return result


m = jnp.ones((logits.shape[0], 1)) * float('inf') * (-1)
probs_pl, m_pl = softmax(logits, m)
s = jnp.exp(logits)
m_gt = jnp.max(logits, axis=-1)
assert(jnp.allclose(probs_pl, s))
assert(jnp.allclose(jnp.squeeze(m_pl),m_gt))
