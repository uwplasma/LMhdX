"""Throwaway probe: which stage-2 building blocks run on LMX v1.4.0 (no lmhdx.design channel_* functions).

The flow integral is done locally (cell areas dy x dz times the axial face velocity), which is what
PR #156's channel_flow_rate does. Checks: rectangular duct, traced field_scale value + gradient,
the fixed-area optimum at H = 100, the exact 1/R cross-duct field (Case P), and one 3-D periodic
modulated-field solve (Case R validity).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmhdx.bc import PERIODIC, BoundaryCondition  # noqa: E402
from lmhdx.core3d import ChannelProblem, ImposedField  # noqa: E402
from lmhdx.grid import Grid, uniform_faces, wall_resolving_faces  # noqa: E402
from lmhdx.steady import solve_steady_state  # noqa: E402

WALL = BoundaryCondition("neumann")


def flow(problem, s=1.0):
    u = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), field_scale=s).velocity[0].data[0]
    _, dy, dz = problem.grid.widths
    return jnp.sum(jnp.asarray(dy)[:, None] * jnp.asarray(dz)[None, :] * u)


def rect(beta, ha, cells=48, field=None, nx=1, length=1.0):
    y = wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=6, max_ratio=None)
    side = min(1.0 / np.sqrt(ha), 0.25 * beta)
    z = wall_resolving_faces(cells, -beta, beta, layer_thickness=side, cells_in_layer=6, max_ratio=None)
    g = Grid(uniform_faces(nx, 0.0, length), y, z)
    return ChannelProblem(grid=g, conditions=(BoundaryCondition(PERIODIC), WALL, WALL), conductivity=1.0,
                          magnetic_field=(0.0, ha, 0.0) if field is None else field(g), forcing=(1.0, 0.0, 0.0), dt=1.0)


# 1. rectangle + traced field scale, value and gradient
p = rect(1.0, 300.0)
q = jax.jit(lambda s: flow(p, s))
t = time.perf_counter()
v = float(q(0.5))
cold = time.perf_counter() - t
g = float(jax.jit(jax.grad(lambda s: flow(p, s)))(0.5))
fd = (float(q(0.5 + 1e-4)) - float(q(0.5 - 1e-4))) / 2e-4
print(f"1. rect + field_scale: q {v:.6e} (cold {cold:.1f} s); grad {g:.8e} vs fd {fd:.8e} rel {abs(g-fd)/abs(fd):.1e}")

# 2. fixed-area optimum at H = 100 (compare with the branch: 62.35 / 59.86 / 65.32)
vals = []
for beta in (0.2, 0.1, 0.07):
    a = 1 / np.sqrt(beta)
    vals.append((4 * beta / float(flow(rect(beta, 100 * a)))) / a**2)
print("2. fixed-area dp/dx at beta 0.2/0.1/0.07:", [round(x, 4) for x in vals])


# 3. Case P: exact 1/R toroidal field across the duct, Rc/a = 10
def toroidal(g, rc=10.0, ha=100.0):
    yv, zv = np.asarray(g.y_faces), np.asarray(g.z_faces)
    A = ha * rc * 0.5 * np.log((rc + zv[None, :]) ** 2 + yv[:, None] ** 2)
    by, bz = np.diff(A, axis=1) / g.widths[2][None, :], -np.diff(A, axis=0) / g.widths[1][:, None]
    nx = g.shape[0]
    f = (np.zeros(g.face_shape(0)), np.repeat(by[None], nx, 0), np.repeat(bz[None], nx, 0))
    c = (np.zeros(g.shape), 0.5 * (f[1][:, :-1] + f[1][:, 1:]), 0.5 * (f[2][:, :, :-1] + f[2][:, :, 1:]))
    return ImposedField(g, c, f)


q_uniform = float(flow(rect(1.0, 100.0)))
q_pol = float(flow(rect(1.0, 100.0, field=toroidal)))
print(f"3. Case P 1/R field (Rc/a 10): q/q_uniform {q_pol / q_uniform:.6f}  (branch: 1.001957)")


# 4. Case R validity: one 3-D periodic modulated solve (lambda/a 25), 24 cells, 16 stations
def modulated(g, ha=50.0, delta=0.1, lam=25.0):
    k = 2 * np.pi / lam
    xf, yf = np.asarray(g.x_faces)[:, None], np.asarray(g.y_faces)[None, :]
    A = -ha * (xf + (delta / k) * np.sin(k * xf) * np.cosh(k * yf))
    fx, fy = np.diff(A, axis=1) / g.widths[1][None, :], -np.diff(A, axis=0) / g.widths[0][:, None]
    f = tuple(np.repeat(v[:, :, None], g.shape[2], axis=2) for v in (fx, fy))
    c = (0.5 * (f[0][:-1] + f[0][1:]), 0.5 * (f[1][:, :-1] + f[1][:, 1:]))
    return ImposedField(g, (*c, np.zeros(g.shape)), (*f, np.zeros(g.face_shape(2))))


t = time.perf_counter()
p3 = rect(1.0, 50.0 * 1.1, cells=24, field=modulated, nx=16, length=25.0)
sol = solve_steady_state(p3, forcing=(1.0, 0.0, 0.0), linear_max_restarts=600)
_, dy, dz = p3.grid.widths
g3 = float(np.sum(np.asarray(dy)[:, None] * np.asarray(dz)[None, :] * np.asarray(sol.velocity[0].data[0])))
print(f"4. 3-D modulated duct: G {g3:.6e}, residual {float(sol.residual_norm):.1e}, {time.perf_counter() - t:.0f} s")
