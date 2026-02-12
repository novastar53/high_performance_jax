import jax  
import jax.dlpack  
from jax import core  
import cuda.tile as ct  
  
def from_jax(x: jax.Array) -> Any:  
    """Return a DLPack capsule for cuTile compatibility."""  
    return jax.dlpack.to_dlpack(x)  
  
def to_jax(capsule: Any, shape: tuple, dtype: jax.numpy.dtype) -> jax.Array:  
    """Convert a DLPack capsule back to a JAX array."""  
    return jax.dlpack.from_dlpack(capsule)
