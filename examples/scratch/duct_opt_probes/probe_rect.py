"""Throwaway feasibility probe for the duct-optimization plan (not part of LMX).

Checks, on a rectangular fully developed duct built directly as a ChannelProblem:
1. beta = 1 reproduces duct_problem bit for bit (same faces, same flow response);
2. the Hartmann number can be swept through the traced field scale on one mesh
   (no rebuild), with accuracy held against the spectral reference;
3. the field-scale gradient matches central differences;
4. cold/warm cost and CG iterations as the field scale moves away from 1;
5. how the dimensionless flow response q changes with aspect ratio beta.
"""

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmx.bc import PERIODIC, BoundaryCondition  # noqa: E402
from lmx.core3d import ChannelProblem, duct_problem  # noqa: E402
from lmx.design import channel_flow_response  # noqa: E402
from lmx.grid import Grid, uniform_faces, wall_resolving_faces  # noqa: E402
from lmx.steady import solve_steady_state  # noqa: E402

from validation.shercliff import flow_rate  # noqa: E402

HA_MAX = 300.0


def rect(beta, ha_max=HA_MAX, ny=48, nz=48, cells_in_layer=6):
    """Half-widths 1 (along B, y) and beta (z); dimensionless like duct_problem."""
    y = wall_resolving_faces(ny, -1.0, 1.0, layer_thickness=1.0 / ha_max, cells_in_layer=cells_in_layer, max_ratio=None)
    z = wall_resolving_faces(
        nz, -beta, beta, layer_thickness=1.0 / np.sqrt(ha_max), cells_in_layer=cells_in_layer, max_ratio=None
    )
    insulating = BoundaryCondition("neumann")
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=(BoundaryCondition(PERIODIC), insulating, insulating),
        conductivity=1.0,
        magnetic_field=(0.0, ha_max, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )


def q_of(problem):
    return jax.jit(lambda s: channel_flow_response(problem, magnetic_field_scale=s).flow_per_unit_drive)


square = rect(1.0)
ref = duct_problem(hartmann=HA_MAX, cells=48)
same_faces = all(np.array_equal(a, b) for a, b in zip(square.grid.faces, ref.grid.faces))
print(f"1. beta=1 faces identical to duct_problem(300, 48): {same_faces}")

q = q_of(square)
t0 = time.perf_counter()
q1 = float(q(1.0))
cold = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(10):
    q(1.0).block_until_ready()
warm = (time.perf_counter() - t0) / 10
print(f"4. cold {cold:.2f} s, warm {warm * 1e3:.1f} ms (square, Ha 300, 48^2)")

print("2. Ha swept through the traced field scale on the Ha-300 mesh (square), against the spectral reference:")
for ha in (30.0, 100.0, 200.0, 300.0):
    s = ha / HA_MAX
    value = float(q(s)) / 4.0  # mean velocity = Q / area, area 4 for the square
    spectral = flow_rate(ha, 60)
    steps = int(solve_steady_state(square, forcing=(1.0, 0.0, 0.0), field_scale=s).steps)
    print(f"   Ha {ha:5.0f}: rel err {value / spectral - 1:+.3e}  CG steps {steps}")

grad = jax.jit(jax.grad(lambda s: channel_flow_response(square, magnetic_field_scale=s).flow_per_unit_drive))
s, h = 0.5, 1e-4
g = float(grad(s))
fd = (float(q(s + h)) - float(q(s - h))) / (2 * h)
print(f"3. dq/ds at s=0.5: adjoint {g:.10e}  central diff {fd:.10e}  rel {abs(g - fd) / abs(fd):.2e}")

print("5. dimensionless resistance per unit mean velocity, R = area / q, versus beta (Ha 100, same cells):")
for beta in (0.5, 1.0, 2.0, 4.0):
    problem = rect(beta)
    t0 = time.perf_counter()
    qb = float(q_of(problem)(100.0 / HA_MAX))
    build = time.perf_counter() - t0
    area = 4.0 * beta
    print(f"   beta {beta:3.1f}: q {qb:.6e}  R=A/q {area / qb:.4e}  (build+compile+solve {build:.1f} s)")
