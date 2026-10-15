"""Smolentsev et al. 2015 Table I solved on the staggered core, not only stored.

Table I lists the flow rate of a square duct for unit ``-dP/dx`` with half-width,
density, viscosity and conductivity one, which is the normalisation of
``duct_problem``: the flow rate is the area integral of the axial velocity. A1 is
Shercliff's insulating duct and A2 is Hunt's duct with conducting Hartmann walls
(c 0.01) and insulating side walls. Every row, Ha 500 to 15,000, is gated against
the analytic column the package ships (validation row 27).
"""

from importlib.resources import files

import pytest

from validation.gci import TABLE_I_MESHES, TABLE_I_RATIO, grid_convergence, table_i_flow_rate

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


def test_the_grid_convergence_index_reproduces_celik_2008_table_1():
    """Celik et al. 2008, Table 1, phi_p (N 18,000 / 8,000 / 4,500 on a 2-D mesh, so r 1.5 and 4/3).

    The paper gives p 1.53, phi_ext 6.1685, e_a 1.5 %, e_ext 1.7 % and GCI_fine 2.2 %.
    """
    result = grid_convergence(5.863, 5.972, 6.063, 1.5, 4.0 / 3.0)
    assert result.monotone
    assert round(result.order, 2) == 1.53
    assert round(result.extrapolated, 4) == 6.1685
    assert (round(100 * result.relative_change, 1), round(100 * result.extrapolated_error, 1)) == (1.5, 1.7)
    assert round(100 * result.gci_fine, 1) == 2.2
    # Equal ratios solve in closed form.
    exact = grid_convergence(1.0 + 4.0, 1.0 + 1.0, 1.0 + 0.25, 2.0)
    assert exact.order == pytest.approx(2.0, abs=1e-12)
    assert exact.extrapolated == pytest.approx(1.0, abs=1e-12)


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
    rates = [
        table_i_flow_rate(hartmann, case["hartmann_wall_conductance"], *mesh)
        for mesh in TABLE_I_MESHES.get(hartmann, TABLE_I_MESHES["high"])
    ]
    errors = [(rate - reference) / reference for rate in rates]
    # The discretisation overestimates the flow rate and approaches it monotonically.
    assert errors[0] > errors[1] > errors[2] > 0.0
    convergence = grid_convergence(*rates, TABLE_I_RATIO)
    assert 1.8 < convergence.order < 2.2
    assert abs(convergence.extrapolated - reference) / reference < (1.0e-3 if hartmann <= 5000 else 5.0e-3)
    # The analytic value lies inside the fine mesh's uncertainty band (Celik 2008, safety factor 1.25).
    assert abs(rates[2] - reference) / reference < convergence.gci_fine
