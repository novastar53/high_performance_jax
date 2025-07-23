import os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

devices = jax.devices()

mesh = Mesh(devices, ("devices", ))
spec = PartitionSpec("devices",)
sharding = NamedSharding(mesh, spec)


x = jax.random.normal(
    jax.random.key(0),
    (16, 3)
)

#x = jax.device_put(x, sharding)
x = jax.lax.with_sharding_constraint(x, sharding)

fn = jax.jit(jax.vmap(lambda x_i: x_i + 2))

y = fn(x)
jax.debug.visualize_array_sharding(y)

assert(jnp.allclose(x + 2, y))