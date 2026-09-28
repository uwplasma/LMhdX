"""Throwaway probe: where is the insulating duct's optimal aspect ratio, at fixed area and flow, versus field?

Same scaling as probe_interior_optimum.py (area 4, V = 1, Ha = H a, a = 1/sqrt(beta)). For one H,
samples beta around the guess beta* = (1.8 / sqrt(H))^(4/3), fits a parabola in ln(beta) through the
three lowest points, and reports beta*, Ha*, beta* sqrt(Ha*) and dp*/dp(square).
Usage: python probe_design_law.py H [CELLS]
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from lmx.bc import PERIODIC, BoundaryCondition  # noqa: E402
from lmx.core3d import ChannelProblem  # noqa: E402
from lmx.design import channel_flow_response  # noqa: E402
from lmx.grid import Grid, uniform_faces, wall_resolving_faces  # noqa: E402

H = float(sys.argv[1])
CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 64


def dpdx(beta):
    a = 1.0 / np.sqrt(beta)
    ha = H * a
    y = wall_resolving_faces(CELLS, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=6, max_ratio=None)
    side = min(1.0 / np.sqrt(ha), 0.25 * beta)
    z = wall_resolving_faces(CELLS, -beta, beta, layer_thickness=side, cells_in_layer=6, max_ratio=None)
    wall = BoundaryCondition("neumann")
    problem = ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, ha, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    q = float(channel_flow_response(problem).flow_per_unit_drive)
    return (4.0 * beta / q) / a**2


guess = min((1.8 / np.sqrt(H)) ** (4.0 / 3.0), 1.0)
betas = guess * np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.45, 2.0])
values = np.array([dpdx(b) for b in betas])
i = int(np.clip(np.argmin(values), 1, len(betas) - 2))
x, y = np.log(betas[i - 1 : i + 2]), values[i - 1 : i + 2]
c2, c1, c0 = np.polyfit(x, y, 2)
beta_star = float(np.exp(-c1 / (2 * c2)))
dp_star = float(c0 - c1**2 / (4 * c2))
ha_star = H / np.sqrt(beta_star)
square = dpdx(1.0)
interior = 0 < int(np.argmin(values)) < len(betas) - 1
print(
    f"H {H:5.0f}  beta* {beta_star:.4f}  Ha* {ha_star:6.1f}  beta* sqrt(Ha*) {beta_star * np.sqrt(ha_star):.3f}  "
    f"dp*/dp(square) {dp_star / square:.3f}  interior {interior}  samples {np.round(values, 3).tolist()}"
)
