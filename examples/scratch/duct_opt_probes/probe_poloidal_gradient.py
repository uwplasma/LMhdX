"""Throwaway probe: a poloidal duct in a tokamak toroidal field, B ~ 1/R across the duct.

Cross-section axes: y along the toroidal direction (the field, half-width a = 1), z along the
major radius (half-width b = beta), duct centre at major radius Rc (in units of a). The field is
the exact vacuum toroidal field on the flat cross-section, B = B0 Rc (x_R, -y) / (x_R^2 + y^2)
with x_R = Rc + z: curl and divergence free, from the flux function A = B0 Rc ln|r|
(B_y = dA/dz, B_z = -dA/dy). Faces hold differences of A, so the discrete divergence is
round-off, as lmhdx.core3d.fringe_field does for the ANL fringe.
Compares the flow per unit drive and the flow centroid against the uniform field B0.
"""

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmhdx.core3d import ChannelProblem, ImposedField  # noqa: E402
from lmhdx.fully_developed import channel_cross_section_weights, channel_flow_rate  # noqa: E402
from lmhdx.grid import (  # noqa: E402  # noqa: E402  # noqa: E402
    CENTER,
    FACE,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.ops import divergence  # noqa: E402
from lmhdx.steady import solve_steady_state  # noqa: E402

HA = 100.0
BETA = 1.0
CELLS = 48


def grid():
    y = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / HA, cells_in_layer=6, max_ratio=None)
    z = wall_resolving_faces(
        CELLS, -BETA, BETA, layer_thickness=1.0 / np.sqrt(HA), cells_in_layer=6, max_ratio=None
    )
    return Grid(uniform_faces(1, 0.0, 1.0), y, z)


def toroidal_field(g, rc):
    y, z = np.asarray(g.y_faces), np.asarray(g.z_faces)
    corners = HA * rc * 0.5 * np.log((rc + z[None, :]) ** 2 + y[:, None] ** 2)  # A on (y, z) corners
    b_y = np.diff(corners, axis=1) / g.widths[2][None, :]  # on y faces: (ny+1, nz)
    b_z = -np.diff(corners, axis=0) / g.widths[1][:, None]  # on z faces: (ny, nz+1)
    nx = g.shape[0]
    faces = (np.zeros(g.face_shape(0)), np.repeat(b_y[None], nx, axis=0), np.repeat(b_z[None], nx, axis=0))
    centres = (np.zeros(g.shape), 0.5 * (faces[1][:, :-1] + faces[1][:, 1:]), 0.5 * (faces[2][:, :, :-1] + faces[2][:, :, 1:]))
    return ImposedField(g, centres, faces)


def solve(field):
    g = grid()
    wall = BoundaryCondition("neumann")
    problem = ChannelProblem(
        grid=g,
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=field(g),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    t0 = time.perf_counter()
    solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0))
    u = solution.velocity[0].data[0]
    q = float(channel_flow_rate(problem, u))
    elapsed = time.perf_counter() - t0
    w = np.asarray(channel_cross_section_weights(problem))
    zc = np.asarray(g.centers[2])[None, :]
    centroid = float(np.sum(w * np.asarray(u) * zc) / np.sum(w * np.asarray(u)))
    return q, centroid, float(solution.residual_norm), elapsed, problem


q0, c0, r0, t_uniform, _ = solve(lambda g: (0.0, HA, 0.0))
print(f"uniform Ha {HA:.0f}: q {q0:.6e}, centroid z {c0:+.2e}, residual {r0:.1e}, {t_uniform:.1f} s")
for rc in (500.0, 50.0, 10.0):
    q, c, r, t, problem = solve(lambda g, rc=rc: toroidal_field(g, rc))
    field = problem.magnetic_field
    div = divergence(
        tuple(Field(np.asarray(f), tuple(FACE if i == ax else CENTER for i in range(3)), problem.grid) for ax, f in enumerate(field.faces))
    )
    spread = 2.0 * BETA / rc
    print(
        f"Rc/a {rc:5.0f} (dB/B across duct ~{spread:.3f}): q/q_uniform {q / q0:.6f}, "
        f"flow centroid z {c:+.4f} (outboard +), residual {r:.1e}, max|div B| {float(np.max(np.abs(div.data))):.1e}, {t:.1f} s"
    )
