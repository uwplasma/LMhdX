"""Smolentsev et al. 2015 Table I solved on the staggered core, not only stored.

Table I lists the flow rate of a square duct for unit ``-dP/dx`` with half-width,
density, viscosity and conductivity one, which is the normalisation of
``duct_problem``: the flow rate is the area integral of the axial velocity. A1 is
Shercliff's insulating duct and A2 is Hunt's duct with conducting Hartmann walls
(c 0.01) and insulating side walls. Only the Ha 500 rows are gated here; the
Ha 5,000-15,000 rows are reported in docs/validation. A1 at Ha 15,000 does not
converge under refinement, which points at the float64 potential solve (plan
step 2b.3).
"""

import math

import numpy as np
import pytest

from lmx.core3d import duct_problem
from lmx.steady import solve_steady_state

pytestmark = pytest.mark.validation

# Analytic column of Smolentsev et al. 2015, Table I, at Ha 500. Four digits, so the
# rounding is at most 6.5e-5 relative for A1 and 3.6e-4 for A2.
TABLE_I = {"A1": (0.0, 7.680e-3), "A2": (0.01, 1.405e-3)}

# Cells per direction and cells inside each layer, scaled together by 1.5 so each
# mesh is the same fitted geometric mapping at a finer spacing.
MESHES = ((32, 4), (48, 6), (72, 9))
RATIO = 1.5


def _flow_rate(row: str, cells: int, cells_in_layer: int) -> float:
    conductance, _ = TABLE_I[row]
    problem = duct_problem(
        hartmann=500.0, cells=cells, cells_in_layer=cells_in_layer, wall_conductance=conductance
    )
    velocity = solve_steady_state(problem).velocity[0].data
    areas = np.asarray(problem.grid.cell_volumes())[0]
    return float((np.asarray(velocity)[0] * areas).sum())


@pytest.mark.parametrize("row", sorted(TABLE_I))
def test_table_i_at_ha_500_converges_at_second_order_to_the_analytic_flow_rate(row):
    """Three meshes give the observed order and a Richardson value to compare with Table I.

    Measured on the floor stack: A1 order 2.00, Richardson -1.5e-5 relative to the
    table; A2 order 1.96, +2.2e-4. Both sit inside the table's own rounding, and the
    gate is validation row 27's 1e-3 at Ha <= 5,000.
    """
    _, reference = TABLE_I[row]
    coarse, medium, fine = (_flow_rate(row, *mesh) for mesh in MESHES)
    errors = [(rate - reference) / reference for rate in (coarse, medium, fine)]
    # The discretisation overestimates the flow rate and approaches it monotonically.
    assert errors[0] > errors[1] > errors[2] > 0.0
    order = math.log((coarse - medium) / (medium - fine)) / math.log(RATIO)
    assert 1.8 < order < 2.2
    extrapolated = fine + (fine - medium) / (RATIO**order - 1.0)
    assert abs(extrapolated - reference) / reference < 1.0e-3
