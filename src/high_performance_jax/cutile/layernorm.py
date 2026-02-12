import jax  
import jax.dlpack  
from jax import custom_vjp  
try:  
    from jax.extend import core  
except ImportError:  
    from jax import core
import cuda.tile as ct  
import math  
  
# DLPack helpers  
def from_jax(x):  
    return x.__dlpack__() 
  
def to_jax(capsule, shape, dtype):  
    return jax.dlpack.from_dlpack(capsule)  
  
# Primitive definition  
cutile_ln_p = core.Primitive('cutile_layer_norm')  
  
def cutile_ln_impl(x_cap, w_cap, b_cap, y_cap, mean_cap, rstd_cap,  
                   *, eps, TILE_N, stream, grid_fwd):  
    ct.launch(stream, grid_fwd, cutile_layer_norm_jax,  
              (x_cap, w_cap, b_cap, y_cap, mean_cap, rstd_cap, eps, TILE_N))  
    return []  
  
cutile_ln_p.def_impl(cutile_ln_impl)  
  
def cutile_ln_abstract(*avals, **kwargs):  
    return []  # outputs are in-place buffers  
  
cutile_ln_p.def_abstract_eval(cutile_ln_abstract)  
  
# Forward wrapper (allocates buffers, converts to DLPack, launches)  
def cutile_layer_norm_jax(x, weight, bias, eps=1e-5):  
    M, N = x.shape  
    TILE_N = 1024  
    y = jax.numpy.empty_like(x)  
    mean = jax.numpy.empty((M,), dtype=jax.numpy.float32, device=x.device)  
    rstd = jax.numpy.empty((M,), dtype=jax.numpy.float32, device=x.device)  
  
    x_cap, w_cap, b_cap, y_cap, mean_cap, rstd_cap = map(  
        from_jax, (x, weight, bias, y, mean, rstd))  
  
    stream = jax.devices()[0]  # JAX default stream  
    grid_fwd = (M,)  
  
    cutile_ln_p.bind(x_cap, w_cap, b_cap, y_cap, mean_cap, rstd_cap,  
                     eps=eps, TILE_N=TILE_N, stream=stream, grid_fwd=grid_fwd)  
  
    return y, mean, rstd  
  
# VJP definition  
@custom_vjp  
def cutile_layer_norm_vjp(x, weight, bias, eps=1e-5):  
    return cutile_layer_norm_jax(x, weight, bias, eps)  
  
# Forward for VJP (same as above, returns residuals)  
def cutile_fwd(x, weight, bias, eps):  
    y, mean, rstd = cutile_layer_norm_jax(x, weight, bias, eps)  
    return y, (x, weight, bias, mean, rstd, eps)  
  
# Backward: launches the two cuTile backward kernels  
def cutile_bwd(res, grad_y_cap):  
    x, weight, bias, mean, rstd, eps = res  
    M, N = x.shape  
    TILE_N = 1024  
    GROUP_SIZE_M = 64  
    TILE_M = 32  
  
    # Allocate backward buffers in JAX  
    dx = jax.numpy.empty_like(x)  
    dw_partial = jax.numpy.zeros((GROUP_SIZE_M, N), dtype=jax.numpy.float32, device=weight.device)  
    db_partial = jax.numpy.zeros((GROUP_SIZE_M, N), dtype=jax.numpy.float32, device=bias.device)  
    locks = jax.numpy.zeros(GROUP_SIZE_M, dtype=jax.numpy.int32, device=weight.device)  
    final_dw = jax.numpy.empty((N,), dtype=weight.dtype, device=weight.device)  
    final_db = jax.numpy.empty((N,), dtype=bias.dtype, device=bias.device)  
  
    # Convert to DLPack capsules  
    args1 = map(from_jax, (dx, grad_y_cap, dw_partial, db_partial,  
                           x, weight, mean, rstd, locks))  
    args2 = map(from_jax, (dw_partial, db_partial, final_dw, final_db))  
  
    stream = jax.devices()[0].device  
    grid_bwd1 = (M,)  
    grid_bwd2 = (math.ceil(N / TILE_N),)  
  
    ct.launch(stream, grid_bwd1, layer_norm_bwd_dx_partial_dwdb,  
              (*args1, TILE_N))  
    ct.launch(stream, grid_bwd2, layer_norm_bwd_dwdb,  
              (*args2, TILE_M, TILE_N))  
  
    return dx, final_dw, final_db, None  
  
cutile_layer_norm_vjp.defvjp(cutile_fwd, cutile_bwd)


import jax.numpy as jnp  
import numpy as np  
import pytest  
  
# Assume the corrected VJP implementation is available as cutile_layer_norm_vjp  
# from your module (e.g., from cutile_layernorm import cutile_layer_norm_vjp)  
  
def test_cutile_layer_norm_jax():  
    # Device check  
    print(jax.devices())
  
    # Simple test shape  
    M, N = 128, 1024  
    eps = 1e-5  
    dtype = jnp.float32  
  
    key = jax.random.PRNGKey(0)  
    x = jax.random.normal(key, (M, N), dtype=dtype)  
    weight = jax.random.normal(jax.random.PRNGKey(1), (N,), dtype=dtype)  
    bias = jax.random.normal(jax.random.PRNGKey(2), (N,), dtype=dtype)  
  
    # Forward pass via custom VJP  
    y_cutile, _ = cutile_layer_norm_vjp(x, weight, bias, eps)  
  
    # Reference JAX LayerNorm  
    def jax_layernorm(x, weight, bias, eps):  
        mean = jnp.mean(x, axis=-1, keepdims=True)  
        var = jnp.var(x, axis=-1, keepdims=True)  
        rstd = jnp.rsqrt(var + eps)  
        return (x - mean) * rstd * weight + bias  
  
    y_ref = jax_layernorm(x, weight, bias, eps)  
  
    # Validate forward  
    np.testing.assert_allclose(y_cutile, y_ref, rtol=1e-4, atol=1e-4)  
  
    # Gradient check via VJP  
    def loss_fn(y):  
        return jnp.sum(y)  
  
    grad_cutile = jax.grad(lambda x, w, b: jnp.sum(cutile_layer_norm_vjp(x, w, b)[0]))(x, weight, bias)  
    grad_ref = jax.grad(lambda x, w, b: jnp.sum(jax_layernorm(x, w, b, eps)))(x, weight, bias)  
  
    np.testing.assert_allclose(grad_cutile[0], grad_ref[0], rtol=1e-3, atol=1e-3)  
    np.testing.assert_allclose(grad_cutile[1], grad_ref[1], rtol=1e-3, atol=1e-3)  
    np.testing.assert_allclose(grad_cutile[2], grad_ref[2], rtol=1e-3, atol=1e-3)  
  
if __name__ == '__main__':  
    test_cutile_layer_norm_jax()  
    print('✅ JAX cuTile LayerNorm VJP test passed')
