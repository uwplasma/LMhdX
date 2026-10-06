"""Smolentsev et al. 2015 Table I solved on the staggered core, not only stored.

Table I lists the flow rate of a square duct for unit ``-dP/dx`` with half-width,
density, viscosity and conductivity one, which is the normalisation of
``duct_problem``: the flow rate is the area integral of the axial velocity. A1 is
Shercliff's insulating duct and A2 is Hunt's duct with conducting Hartmann walls
(c 0.01) and insulating side walls. Every row, Ha 500 to 15,000, is gated against
the analytic column the package ships (validation row 27).
"""

import math
from importlib.resources import files

import numpy as np
import pytest

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_steady_state

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.validation

_TABLE = tomllib.loads(
    files("lmhdx").joinpath("data/benchmarks/references/samper-table-i.toml").read_text(encoding="utf-8")
)
ROWS = {
    ("A1" if case["hartmann_wall_conductance"] == 0.0 else "A2", case["hartmann_number"]): case
    for case in _TABLE["cases"]
}

# Cells and cells inside the layer, along the field (Hartmann layer 1/Ha) then across
# it (side layer 1/sqrt(Ha)), each series scaled by 1.5: one fitted mapping at three
# spacings. Above Ha 500 the Hartmann direction gets twice the cells.
RATIO = 1.5
MESHES = {
    500: ((32, 4, 32, 4), (48, 6, 48, 6), (72, 9, 72, 9)),
    "high": ((64, 8, 32, 4), (96, 12, 48, 6), (144, 18, 72, 9)),
}


def _flow_rate(case: dict, cells: int, layer: int, cells_across: int, layer_across: int) -> float:
    hartmann = float(case["hartmann_number"])
    enable_x64()
    wall = BoundaryCondition("neumann")
    problem = ChannelProblem(
        grid=Grid(
            uniform_faces(1, 0.0, 1.0),
            wall_resolving_faces(
                cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=layer, max_ratio=None
            ),
            wall_resolving_faces(
                cells_across,
                -1.0,
                1.0,
                layer_thickness=hartmann**-0.5,
                cells_in_layer=layer_across,
                max_ratio=None,
            ),
        ),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_conductance=(0.0, case["hartmann_wall_conductance"], 0.0),
    )
    velocity = np.asarray(solve_steady_state(problem).velocity[0].data)[0]
    return float((velocity * np.asarray(problem.grid.cell_volumes())[0]).sum())


# Ha 500 runs on pull requests. The higher rows take 15-35 s each from a cold cache,
# more than the channel shard's pull-request budget leaves, so they run on main.
@pytest.mark.parametrize(
    "row, hartmann",
    [
        pytest.param(row, hartmann, marks=() if hartmann == 500 else pytest.mark.slow)
        for row, hartmann in sorted(ROWS, key=lambda key: (key[1], key[0]))
    ],
)
def test_table_i_converges_at_second_order_to_the_analytic_flow_rate(row, hartmann):
    """Three meshes give the observed order and a Richardson value to compare with Table I.

    Measured on the floor stack, the error of the finest mesh is 0.16-0.43 %, the
    observed order 1.96-2.02 and the Richardson value within 2.2e-4 of the table in
    every row; the table's four digits round at 6.5e-5 to 3.6e-4. Row 27 asks for
    1e-3 up to Ha 5,000 and 5e-3 above. A1 at Ha 15,000 needs the Neumann null
    eigenvalue held at exactly zero (#183): with the computed one, which carries
    round-off, the finest mesh reads -0.65 % and 128:16:64:8 +3.16 %.
    """
    case = ROWS[(row, hartmann)]
    reference = case["analytical_flow_rate"]
    coarse, medium, fine = (_flow_rate(case, *mesh) for mesh in MESHES.get(hartmann, MESHES["high"]))
    errors = [(rate - reference) / reference for rate in (coarse, medium, fine)]
    # The discretisation overestimates the flow rate and approaches it monotonically.
    assert errors[0] > errors[1] > errors[2] > 0.0
    order = math.log((coarse - medium) / (medium - fine)) / math.log(RATIO)
    assert 1.8 < order < 2.2
    extrapolated = fine + (fine - medium) / (RATIO**order - 1.0)
    assert abs(extrapolated - reference) / reference < (1.0e-3 if hartmann <= 5000 else 5.0e-3)
