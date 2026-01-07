from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plgpu
import flax.nnx as nnx


INTERPRET_MODE = True # Set to False on GPU

# Pallas softmax
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 3


# Manual softmax (jax)
def manual_softmax(logits):
    max_rows = jnp.max(logits, axis=-1)
    s = jnp.exp(logits - max_rows[..., None])
    l = jnp.sum(s, axis=-1)
    l = l[..., None]
    return s / l 


def manual_softmax_backward(g, y):
    g_dot_y =  g @ y.T
    g_diff = g - g_dot_y[:, None]
    dx = y * g_diff
    return dx


def online_softmax(logits):
    out = jnp.zeros_like(logits)
    m = jnp.full((logits.shape[0],), -jnp.inf)
    l = jnp.zeros((logits.shape[0],))
    for i in range(0, logits.shape[0], BLOCK_M):  # This axis can be tiled in parallel blocks.
        for j in range(0, logits.shape[1], BLOCK_N):  # This axis cannot be tiled in parallel, so it is tiled sequentially
            block = logits[i:i+BLOCK_M, j:j+BLOCK_N] # Load a block
            block_max = jnp.max(block, axis=-1) # Get the max across the block
            curr_max = m[i:i+BLOCK_M] # Retrieve the previous computed max for the rows
            new_max = jnp.maximum(curr_max, block_max) # Update the max for all the rows
            m = m.at[i:i+BLOCK_M].set(new_max)  
            l_block = l[i:i+BLOCK_M] # Get the denominator for the rows in the block
            l_block = l_block * jnp.exp(curr_max - new_max) + jnp.sum( # Correct and update the denominator based on the current block
                jnp.exp(block - new_max[:, None]), axis=-1
            )
            l = l.at[i:i+BLOCK_M].set(l_block)
        for j in range(0, logits.shape[1], BLOCK_N):  # Loop over the column blocks and generate the output values 
            out_block = jnp.exp(logits[i:i+BLOCK_M, j:j+BLOCK_N] - m[i:i+BLOCK_M][:, None]) / l[i:i+BLOCK_M][:, None]
            out = out.at[i:i+BLOCK_M, j:j+BLOCK_N].set(out_block)
    
    return out


def softmax_kernel(x_ref, out_ref, *, n_col_blocks, n_rows, n_cols):
    max_reg = jnp.full((BLOCK_M,), -jnp.inf, dtype=jnp.float32) 
    l_reg = jnp.zeros((BLOCK_M,), dtype=jnp.float32) 
    row_ids = pl.program_id(0) * BLOCK_M + jnp.arange(BLOCK_M)
    row_mask = row_ids < n_rows

    def stats_body(t, args):
        max_reg, l_reg = args
        idx = pl.dslice(t * BLOCK_N, BLOCK_N)
        col_ids = t * BLOCK_N + jnp.arange(BLOCK_N)
        cols_mask = col_ids < n_cols
        mask = row_mask[:, None] & cols_mask[None, :]

        x_tile = plgpu.load(
            x_ref.at[:, idx],
            mask=mask,
            other=-jnp.inf,
        ).astype(jnp.float32)
        max_tile = jnp.max(x_tile, axis=-1)
        max_new = jnp.maximum(max_reg, max_tile)
        l_update = l_reg * jnp.exp(max_reg - max_new) + jnp.sum(
            jnp.exp(x_tile - max_new[:, None]), axis=-1
        )
        return max_new, l_update
        
    max_reg, l_reg = jax.lax.fori_loop(0, n_col_blocks, stats_body, (max_reg, l_reg))

    def out_body(t, _):
        idx = pl.dslice(t * BLOCK_N, BLOCK_N)
        col_ids = t * BLOCK_N + jnp.arange(BLOCK_N)
        cols_mask = col_ids < n_cols
        mask = row_mask[:, None] & cols_mask[None, :]

        x_tile = plgpu.load(
            x_ref.at[:, idx],
            mask=mask,
            other=-jnp.inf,
        ).astype(jnp.float32)
        out_tile = jnp.exp(x_tile - max_reg[:, None]) / l_reg[:, None]
        plgpu.store(out_ref.at[:, idx], out_tile.astype(jnp.float32), mask=mask)

    _ = jax.lax.fori_loop(0, n_col_blocks, out_body, None)


@jax.jit
def softmax(logits):
    n_row_blocks = pl.cdiv(logits.shape[0], BLOCK_M)
    n_col_blocks = pl.cdiv(logits.shape[1], BLOCK_N)
    return pl.pallas_call(
        partial(softmax_kernel, n_col_blocks=n_col_blocks, n_rows=logits.shape[0], n_cols=logits.shape[1]),
        out_shape=jax.ShapeDtypeStruct(logits.shape, jnp.float32),
        grid=(n_row_blocks,),
        in_specs=[pl.BlockSpec((BLOCK_M, logits.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((BLOCK_M, logits.shape[1]), lambda i: (i, 0)),
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        ),
    )(logits)


def softmax_backward_kernel(y_ref, dy_ref, dx_ref):
    # compute the inner product <g_ref, y_ref>
    dy_reg = plgpu.load(dy_ref)
    y_reg = plgpu.load(y_ref)
    g_dot_y = jnp.sum(dy_reg * y_reg, axis=1)

    # Compute the output block
    output_reg = y_reg * ( dy_reg - g_dot_y[:, None] )
    plgpu.store(dx_ref, output_reg)


@jax.jit
def softmax_backward(y, dy):
    M, N = y.shape

    grid = (pl.cdiv(M, BLOCK_M),)
    out_shape = jax.ShapeDtypeStruct((M, N), y.dtype)

    return pl.pallas_call(
        softmax_backward_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),  # y
            pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),  # dy 
        ],
        out_specs=pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),  # dx
        interpret=INTERPRET_MODE,
        compiler_params=plgpu.CompilerParams(
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        ),
    )(y, dy)


@jax.custom_vjp
def softmax_pallas(x):
    return softmax(x)


def softmax_fwd(x):
    y = softmax(x)
    return y, y


def softmax_bwd(saved_y, dy):
    (y,) = (saved_y,)
    dx = softmax_backward(y, dy)
    return (dx,)


softmax_pallas.defvjp(softmax_fwd, softmax_bwd)


@dataclass
class ModelConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int


class Model(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config
        self.l1 = nnx.Linear(config.in_dim, config.hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(config.hidden_dim, config.out_dim, rngs=rngs)

    def forward_logits(self, x):
        x = x.reshape(-1, x.shape[-1])
        x = self.l1(x)
        x = jax.nn.relu(x)
        x = self.l2(x)
        return x

    def __call__(self, x):
        return self.forward_logits(x)

if __name__ == "__main__":
    import jax
    import optax
    import jax.numpy as jnp

    def loss_fn(model, x, y):
        logits = model.forward_logits(x)
        probs = softmax_pallas(logits)
        labels = y.reshape(-1)
        one_hot = jax.nn.one_hot(labels, probs.shape[-1], dtype=probs.dtype)
        loss = -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-9), axis=-1))
        return loss, (probs, logits)

    def loss_from_logits_pallas(logits, y):
        probs = softmax_pallas(logits)
        labels = y.reshape(-1)
        one_hot = jax.nn.one_hot(labels, probs.shape[-1], dtype=probs.dtype)
        loss = -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-9), axis=-1))
        return loss

    def loss_from_logits_gt(logits, y):
        probs = jax.nn.softmax(logits)
        labels = y.reshape(-1)
        one_hot = jax.nn.one_hot(labels, probs.shape[-1], dtype=probs.dtype)
        loss = -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-9), axis=-1))
        return loss


    @nnx.jit
    def step(state, x, y):
        (loss, (y_pred, logits)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True)(state.model, x, y)
        state.update(grads)
        d_logits_pallas = jax.grad(loss_from_logits_pallas)(logits, y)
        d_logits_gt = jax.grad(loss_from_logits_gt)(logits, y)
        return loss, d_logits_pallas, d_logits_gt

    B, E = 256, 24

    default = jax.random.key(69)
    gate_noise = jax.random.key(42)
    rngs = nnx.Rngs(default=default, gate_noise=gate_noise)

    num_classes = 2
    config = ModelConfig(in_dim=E, hidden_dim=E * 4, out_dim=num_classes)
    model = Model(config, rngs)
    model.train(add_noise=False)
    tx = optax.adam(1e-1)
    state = nnx.Optimizer(model, tx)

    x = jax.random.normal(jax.random.key(1000), (B, E))
    class_ids = (x[:, 0] > 0).astype(jnp.int32)
    y = class_ids

    iters = 10
    for i in range(iters):
        loss, d_logits_pallas, d_logits_gt = step(state, x, y)
        #print(d_logits_pallas)
        #print(d_logits_gt)
        assert(jnp.allclose(d_logits_pallas, d_logits_gt))
        print(f"iter {i}: loss={loss}")
