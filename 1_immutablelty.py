import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, pmap
from jax import lax
from jax import make_jaxpr
from jax import random
from jax import device_put
import matplotlib.pyplot as plt


# Fact 1: JAX's syntax is remarkably similar to NumPy's 
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)

size = 10
index = 0
value = 23

# In NumPy arrays are mutable
x = np.arange(size)
print(x)
x[index] = value
print(x)

# In JAX we have to deal with immutable arrays
x = jnp.arange(size)
print(x)
y = x.at[index].set(value)
print(x)
print(y)



