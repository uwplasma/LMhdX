"""Throwaway probe: pressure gradient at fixed flow rate and fixed cross-section area, versus aspect ratio.

Dimensionless: mu = sigma = 1, area A = 4 (the unit square at beta = 1), mean velocity V = 1,
and B * sqrt(sigma/mu) = HA_UNIT, so Ha = HA_UNIT * a. With half-widths a (along B) and
b = beta * a, 4ab = A gives a = 1/sqrt(beta). The solve runs on the half-width-1 mesh
[-1,1] x [-beta,beta]; dp/dx = V * mu / a^2 * (4 beta / q), q = dimensionless Q per unit drive.
Walls: insulating, or thin walls of fixed thickness, c = C0 / a on all four.
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmhdx.core3d import ChannelProblem  # noqa: E402
from lmhdx.fully_developed import channel_flow_response  # noqa: E402
from lmhdx.grid import (  # noqa: E402  # noqa: E402
    PERIODIC,
    BoundaryCondition,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)

HA_UNIT = 100.0
CELLS = 48


def problem(beta, ha, c):
    y = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=6, max_ratio=None)
    z = wall_resolving_faces(
        CELLS, -beta, beta, layer_thickness=1.0 / np.sqrt(ha), cells_in_layer=6, max_ratio=None
    )
    wall = BoundaryCondition("neumann")
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, ha, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_conductance=(0.0, c, c),
    )


c0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
print(
    f"walls: {'insulating' if c0 == 0 else f'thin, c = {c0}/a'}   (fixed area 4, fixed V = 1, Ha = {HA_UNIT:.0f} a)"
)
for beta in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
    a = 1.0 / np.sqrt(beta)
    ha, c = HA_UNIT * a, c0 / a
    q = float(channel_flow_response(problem(beta, ha, c)).flow_per_unit_drive)
    gradient = (4.0 * beta / q) / a**2
    print(
        f"  beta {beta:4.2f}  a {a:5.3f}  b {beta * a:5.3f}  Ha {ha:6.1f}  c {c:6.4f}  dp/dx {gradient:9.4f}"
    )
