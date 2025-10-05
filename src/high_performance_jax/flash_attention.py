import jax
import jax.numpy as jnp

# online softmax
x = jnp.array([1, 5, 2, 8])
z = 0
max_x = float("-inf")
for i in range(len(x)):
    new_max_x = max(x[0], max_x)
    z = z*jnp.exp(max_x - new_max_x) + jnp.exp(x[i] - new_max_x)
    max_x = new_max_x

result = jnp.exp(x - max_x) / z
standard_result = jax.nn.softmax(x)
print(jnp.allclose(result, standard_result))


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

q = jax.random.randint(
    jax.random.key(0),
    (N, D), -5, 5
)
k = jax.random.randint(
    jax.random.key(1),
    (N, D), -5, 5
)
v = jax.random.randint(
    jax.random.key(2),
    (N, D), -5, 5
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
result = jnp.zeros_like(q, dtype=jnp.float32)
z = jnp.zeros((N,), dtype=jnp.float32)
m = jnp.full((N,), float('-inf'))
for row in range(n_blocks):
    q_blk = q[row*block_size:(row+1)*block_size, :]
    m_blk = m[row*block_size:(row+1)*block_size]
    z_blk = z[row*block_size:(row+1)*block_size]
    result_blk = result[row*block_size:(row+1)*block_size, :]
    for col in range(n_blocks):
        k_blk = k.T[:, col*block_size:(col+1)*block_size] 
        v_blk = v[col*block_size:(col+1)*block_size, :]
        s_blk = q_blk @ k_blk
        m_blk_old = m_blk
        m_blk = jnp.maximum(m_blk, jnp.max(s_blk, axis=-1))
        p_blk = jnp.exp(s_blk - m_blk)
        z_blk = z_blk*jnp.exp(m_blk_old - m_blk) + jnp.sum(p_blk, axis=-1)
        p = p_blk @ v_blk
        result_blk = (jnp.eye(block_size,) * jnp.exp(m_blk - m_blk_old)) @ result_blk + p
        computed_blocks.append((row, col))
        #print_grid(n_blocks, computed_blocks)
        #print("-----")
    m = m.at[row*block_size:(row+1)*block_size].set(m_blk)
    z = z.at[row*block_size:(row+1)*block_size].set(z_blk)
    result = result.at[row*block_size:(row+1)*block_size, :].set(result_blk)

#result = jnp.exp(result - jnp.eye(N, N) * jnp.max(result, axis=-1))
#z = jnp.sum(result, axis=-1)
result /= z[..., None]

s = q @ k.T
p = jax.nn.softmax(s, axis=-1)
test_result = p @ v
print(jnp.allclose(result, test_result))