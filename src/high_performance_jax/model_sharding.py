import dataclasses
import os 

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

print(jax.device_count())

mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 4)),
    ('data', 'model')
)


# A helper function to quickly create a NamedSharding object
# using the globally defined 'mesh'.
def named_sharding(*names: str | None) -> NamedSharding:
  # P(*names) creates a PartitionSpec, e.g., P('data', None)
  # NamedSharding binds this PartitionSpec to the 'mesh'.
  return NamedSharding(mesh, P(*names))


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None # Sharding rule for embedding-like dimensions
  mlp: str | None = None   # Sharding rule for MLP layers dimensions
  data: str | None = None  # Sharding rule for the data batch dimension

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(
  embed=None,
  mlp='model',
  data='data',
)


# Define the MLP using Flax NNX API.
class MLP(nnx.Module):
  # Constructor takes input/hidden/output dimensions and an NNX Rngs object.
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    # Define the first weight matrix as an nnx.Param.
    self.w1 = nnx.Param(
      # Initialize with lecun_normal initializer using a key from rngs.
      nnx.initializers.lecun_normal()(rngs.params(), (din, dmid)),
      # CRITICAL: Specify the desired sharding using MeshRules.
      # ('embed', 'mlp') -> (None, 'model') -> Replicate dim 0, shard dim 1 along 'model' axis.
      sharding=mesh_rules('embed', 'mlp'),
    )
    # Define the first bias vector as an nnx.Param.
    self.b1 = nnx.Param(
      jnp.zeros((dmid,)), # Initialize with zeros.
      # Sharding: ('mlp',) -> ('model',) -> Shard dim 0 along 'model' axis.
      sharding=mesh_rules('mlp'),
    )
    # Define the second weight matrix as an nnx.Param.
    self.w2 = nnx.Param(
      nnx.initializers.lecun_normal()(rngs.params(), (dmid, dout)),
       # Sharding: ('embed', 'mlp') -> (None, 'model') -> Replicate dim 0, shard dim 1 along 'model' axis.
      sharding=mesh_rules('embed', 'mlp'),
    )
    # Note: No second bias b2 is defined in this simple example.

  # The forward pass of the MLP.
  def __call__(self, x: jax.Array):
    # Standard MLP calculation: (x @ W1 + b1) -> ReLU -> @ W2
    # NNX automatically accesses the .value attribute of nnx.Param objects.
    return nnx.relu(x @ self.w1 + self.b1) @ self.w2


@nnx.jit
def create_model():
  # Instantiate the MLP model. rngs=nnx.Rngs(0) provides PRNG keys.
  model = MLP(8, 32, 16, rngs=nnx.Rngs(0))

  # === Explicit Sharding Application ===
  # 1. Extract ALL state (model params + optimizer momentum) into a flat State pytree.
  graphdef, state = nnx.split(model)

  # 2. Define the target sharding for the state pytree.
  # This function maps state paths to NamedSharding objects based on stored metadata.
  def get_named_shardings(path: tuple, value: nnx.VariableState):
    # Assumes params and momentum use the sharding defined in their metadata.
    return value.replace(NamedSharding(mesh, P(*value.sharding)))
  # Create the pytree of NamedSharding objects.
  named_shardings = state.map(get_named_shardings)
  print(named_shardings)

  # 3. Apply sharding constraint. This tells JAX how the 'state' pytree
  # SHOULD be sharded when computations involving it are run under jit/pjit.
  # It doesn't immediately move data but sets up the constraint for the compiler.
  sharded_state = jax.lax.with_sharding_constraint(state, named_shardings)

  # 4. Update the original objects (model params, optimizer momentum)
  # with the constrained state values. This step makes the sharding
  # "stick" to the objects themselves for subsequent use outside this function.
  model = nnx.merge(graphdef, sharded_state)

  return model


if __name__ == "__main__":
    model = MLP(8, 32, 16, rngs=nnx.Rngs(0))
    jax.debug.visualize_array_sharding(model.b1.value)
    model = create_model()
    print(model.b1.value.shape)
    for shard in model.b1.value.addressable_shards:
      print(shard.data.shape)
    jax.debug.visualize_array_sharding(model.b1.value)
