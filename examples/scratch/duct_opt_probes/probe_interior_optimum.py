"""Throwaway probe: does the insulating duct have an interior optimal aspect ratio at fixed area and flow?

Dimensionless as probe_landscape.py: area 4, V = 1, mu = sigma = 1, Ha = H * a with
a = 1/sqrt(beta) the half-width along B and b = beta * a across it. dp/dx = (4 beta / q) / a^2.
Usage: python probe_interior_optimum.py H CELLS
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmx.bc import PERIODIC, BoundaryCondition  # noqa: E402
from lmx.core3d import ChannelProblem  # noqa: E402
from lmx.design import channel_flow_response  # noqa: E402
from lmx.grid import Grid, uniform_faces, wall_resolving_faces  # noqa: E402

H = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 48


def problem(beta, ha):
    y = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=6, max_ratio=None)
    side = min(1.0 / np.sqrt(ha), 0.25 * beta)  # the side layer can be wider than a slender duct
    z = wall_resolving_faces(CELLS, -beta, beta, layer_thickness=side, cells_in_layer=6, max_ratio=None)
    wall = BoundaryCondition("neumann")
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, ha, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )


print(f"H = {H:.0f} (Ha at the square), cells {CELLS}")
for beta in (0.2, 0.1, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012):
    a = 1.0 / np.sqrt(beta)
    ha = H * a
    q = float(channel_flow_response(problem(beta, ha)).flow_per_unit_drive)
    print(f"  beta {beta:5.3f}  Ha {ha:6.1f}  b*sqrt(Ha)/a {beta * np.sqrt(ha):5.2f}  dp/dx {(4 * beta / q) / a**2:9.4f}")
