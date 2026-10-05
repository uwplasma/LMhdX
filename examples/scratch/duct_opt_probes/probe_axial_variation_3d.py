"""Throwaway probe: is the locally fully developed sum accurate for a slowly varying axial field?

Square insulating duct (half-width 1), field B_y + i B_x = B0 (1 + delta cos k(x + i y)): the
curl- and divergence-free pair of a cosine modulation along the axis, periodic over one
wavelength lam, from the harmonic flux function A = -B0 (x + (delta/k) sin(kx) cosh(ky)).
Solves the 3-D steady Stokes problem over one period, then compares its flow per unit mean
drive with the quasi fully developed estimate 1 / mean_x(1 / G_FD(B_y(x, 0))) from 2-D solves.
Usage: python probe_axial_variation_3d.py LAMBDA [NX]
"""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmhdx.core3d import ChannelProblem, ImposedField, duct_problem  # noqa: E402
from lmhdx.fully_developed import channel_flow_rate, channel_flow_response  # noqa: E402
from lmhdx.grid import (  # noqa: E402  # noqa: E402
    PERIODIC,
    BoundaryCondition,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.steady import solve_steady_state  # noqa: E402

HA, DELTA, CELLS = 50.0, 0.10, 24
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 16
k = 2.0 * np.pi / LAM

peak = HA * (1 + DELTA * np.cosh(k))
y = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / peak, cells_in_layer=4, max_ratio=None)
z = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(HA), cells_in_layer=4, max_ratio=None)
grid = Grid(uniform_faces(NX, 0.0, LAM), y, z)

xf, yf = np.asarray(grid.x_faces)[:, None], np.asarray(grid.y_faces)[None, :]
corners = -HA * (xf + (DELTA / k) * np.sin(k * xf) * np.cosh(k * yf))  # A on (x, y) corners
along_x = np.diff(corners, axis=1) / grid.widths[1][None, :]  # B_x = dA/dy on x faces
along_y = -np.diff(corners, axis=0) / grid.widths[0][:, None]  # B_y = -dA/dx on y faces
faces = tuple(np.repeat(v[:, :, None], grid.shape[2], axis=2) for v in (along_x, along_y))
centres = (0.5 * (faces[0][:-1] + faces[0][1:]), 0.5 * (faces[1][:, :-1] + faces[1][:, 1:]))
field = ImposedField(grid, (*centres, np.zeros(grid.shape)), (*faces, np.zeros(grid.face_shape(2))))

wall = BoundaryCondition("neumann")
problem = ChannelProblem(
    grid=grid,
    conditions=(BoundaryCondition(PERIODIC), wall, wall),
    conductivity=1.0,
    magnetic_field=field,
    forcing=(1.0, 0.0, 0.0),
    dt=1.0,
)
t0 = time.perf_counter()
solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), linear_max_restarts=600)
g3 = float(channel_flow_rate(problem, solution.velocity[0].data[0]))
t3 = time.perf_counter() - t0

# Quasi fully developed: 2-D solves at each station's midplane field, on the same cross-section cells.
reference = duct_problem(hartmann=peak, cells=CELLS, cells_in_layer=4)
square = ChannelProblem(
    grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
    conditions=reference.conditions,
    conductivity=1.0,
    magnetic_field=(0.0, peak, 0.0),
    forcing=(1.0, 0.0, 0.0),
    dt=1.0,
)
respond = jax.jit(lambda s: channel_flow_response(square, magnetic_field_scale=s).flow_per_unit_drive)
xc = np.asarray(grid.centers[0])
stations = HA * (1 + DELTA * np.cos(k * xc)) / peak
g_fd = np.array([float(respond(s)) for s in stations])
g_quasi = 1.0 / np.mean(1.0 / g_fd)
print(
    f"lambda/a {LAM:6.1f}  NX {NX}  3-D G {g3:.6e}  quasi-FD G {g_quasi:.6e}  "
    f"3-D excess pressure drop {g_quasi / g3 - 1:+.3e}  residual {float(solution.residual_norm):.1e}  3-D solve {t3:.0f} s"
)
