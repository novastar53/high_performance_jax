import jax
import jax.numpy as jnp

def print_grid(N, filled_cells):
    """
    Prints an NxN grid.
    
    Parameters:
    - N: size of the grid (int)
    - filled_cells: list of tuples (row, col) representing filled cells
    """
    # Convert filled_cells to a set for O(1) lookups
    filled = set(filled_cells)

    for i in range(N):
        row = ""
        for j in range(N):
            if (i, j) in filled:
                row += "X "   # filled cell
            else:
                row += ". "   # empty cell
        print(row.strip())

N, D = 24, 8

q = jax.random.normal(
    jax.random.key(0),
    (N, D)
)
k = jax.random.normal(
    jax.random.key(1),
    (N, D)
)
v = jax.random.normal(
    jax.random.key(2),
    (N, D)
)

# regular attention
a = jnp.einsum('nd,md->nm', q, k)
a = jax.nn.softmax(a, axis=-1)
o = jnp.einsum('nm,md->nd', a, v)
print(o.shape)

# flash attention (python)
block_size = 4
n_blocks = N // block_size

computed_blocks = []
result = jnp.empty((N, N))
for row in range(n_blocks):
    for col in range(n_blocks):
        q_blk = q[row*block_size:(row+1)*block_size, :]
        k_blk = k.T[:, col*block_size:(col+1)*block_size] 
        result = result.at[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size].set(q_blk @ k_blk)
        computed_blocks.append((row, col))
        print_grid(n_blocks, computed_blocks)
        print("-----")