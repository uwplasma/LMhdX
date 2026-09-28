import time

import jax
import jax.numpy as jnp

from lmhdx.core3d import duct_problem
from lmhdx.design import channel_flow_response

jax.config.update("jax_enable_x64", True)

p = duct_problem(hartmann=300.0, cells=48)
scales = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])          # Ha 60..300 samples along a duct
weights = jnp.full(5, 0.2)                              # segment-length fractions


def q(s):
    return channel_flow_response(p, magnetic_field_scale=s).flow_per_unit_drive


batched = jax.jit(jax.vmap(q))
t = time.perf_counter()
qb = batched(scales).block_until_ready()
cold = time.perf_counter() - t
t = time.perf_counter()
batched(scales).block_until_ready()
warm = time.perf_counter() - t
loop = jnp.array([float(jax.jit(q)(s)) for s in scales])
print("vmap == loop:", float(jnp.max(jnp.abs(qb / loop - 1))), f"cold {cold:.1f}s warm {warm*1e3:.0f}ms for 5 fields")
# objective: segment-weighted resistance sum_k w_k A/q_k at a common size multiplier m (traced a)
obj = jax.jit(jax.value_and_grad(lambda m: jnp.sum(weights * 4.0 / jax.vmap(q)(m * scales))))
v, g = obj(0.9)
h = 1e-4
fd = (float(obj(0.9 + h)[0]) - float(obj(0.9 - h)[0])) / (2 * h)
print(f"multi-field objective {float(v):.6f}, grad {float(g):.8e}, fd {fd:.8e}, rel {abs(float(g)-fd)/abs(fd):.1e}")
