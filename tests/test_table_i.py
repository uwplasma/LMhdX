"""Smolentsev et al. 2015 Table I solved on the staggered core, not only stored.

Table I lists the flow rate of a square duct for unit ``-dP/dx`` with half-width,
density, viscosity and conductivity one, which is the normalisation of
``duct_problem``: the flow rate is the area integral of the axial velocity. A1 is
Shercliff's insulating duct and A2 is Hunt's duct with conducting Hartmann walls
(c 0.01) and insulating side walls. Every row, Ha 500 to 15,000, is gated against
the analytic column the package ships (validation row 27).
"""

import tomllib
from importlib.resources import files

import pytest

from validation.gci import TABLE_I_MESHES, TABLE_I_RATIO, grid_convergence, table_i_flow_rate
from validation.shercliff import quadrant_flow_rate

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


# Ni, Munipalli, Huang, Morley and Abdou 2007 Part I (J. Comput. Phys. 227, 174): their published tables.
#
# Section 4 of the paper validates its consistent and conservative current scheme on
# fully developed duct flow. LMhdX's staggered scheme conserves charge exactly, so it
# is a different scheme with the same property, and the paper's numbers are a
# code-to-code band, not an exact target. Three anchors, all run from the paper's data:
#
# * Table 2: mass flow rate of Shercliff's duct (insulating walls) at Ha 50, 500 and
#   1000 on three meshes of 38^2, 46^2 and 62^2 cells (4, 8 and 16 cells across 2/Ha
#   and 2/sqrt(Ha), 30 in the core), the pressure gradient chosen so that the exact
#   rate is 4. LMhdX meshes follow that recipe with ``wall_resolving_faces``; the
#   paper does not give the stretching, so the match is of layer resolution.
# * Tables 3 and 4: the analytic pressure gradient of Hunt's duct (Ha 300, c 0.1 and
#   Ha 500, c 0.05, Hartmann walls conducting), quoted as ``Re * dp/dx``. The
#   independent spectral solve of ``validation.shercliff`` is compared with them.
# * Table 3's own 3-D error, 0.06 %, is not reproduced at 47^2: see the documentation.

# Table 2, Mesh-1 to Mesh-3: 38^2, 46^2 and 62^2 cells (4, 8 and 16 layer cells), coarse to fine.
NI_TABLE_2 = {
    50: (4.13534, 4.03574, 4.01075),
    500: (4.14244, 4.04400, 4.01862),
    1000: (4.13525, 4.03927, 4.01476),
}
# Tables 3 and 4: Re x analytic pressure gradient, equal in every column of a table.
NI_TABLE_3_4 = {(300, 0.1): 5854.302, (500, 0.05): 9531.4475}


@pytest.mark.parametrize("hartmann", [50, 500, 1000])
def test_table_2_flow_rates_of_shercliffs_duct_on_the_papers_layer_resolutions(hartmann):
    """LMhdX errors are 0.3-4 % and monotone, below Ni's on the coarsest mesh, within 1.4x on the middle one and 3.5x on the finest.

    Measured (error of the flow rate, LMhdX against Ni): Ha 50 2.7/3.4, 0.77/0.89, 0.35/0.27 %;
    Ha 500 2.6/3.6, 1.1/1.1, 0.86/0.47 %; Ha 1000 2.6/3.4, 1.3/0.98, 1.1/0.37 %.
    """
    reference = quadrant_flow_rate(hartmann)
    ours, theirs = [], []
    for layer_cells, published in zip((4, 8, 16), NI_TABLE_2[hartmann], strict=True):
        count = 30 + 2 * layer_cells
        rate = table_i_flow_rate(hartmann, 0.0, count, layer_cells, count, layer_cells, 2.0) / reference
        ours.append(rate / 4.0 - 1.0)
        theirs.append(published / 4.0 - 1.0)
    assert ours[0] > ours[1] > ours[2] > 0.0
    assert ours[0] < theirs[0] and ours[1] < 1.4 * theirs[1]
    assert all(0.3 < mine / paper < 3.5 for mine, paper in zip(ours, theirs, strict=True) if paper > 0.0)


@pytest.mark.parametrize(("hartmann", "wall"), sorted(NI_TABLE_3_4))
def test_tables_3_and_4_analytic_pressure_gradient_of_hunts_duct(hartmann, wall):
    """The paper's analytic ``Re dp/dx`` agrees with the independent spectral solve to 6e-4.

    Measured: the spectral values are 5857.20 and 9535.10 (converged to 1e-5 with 80 points),
    against Ni's 5854.302 and 9531.4475: +4.9e-4 and +4.2e-4. These are the size of Ni's own
    reported 3-D errors (0.016-0.075 %), so their error is partly the reference.
    """
    # With unit mean velocity the dimensionless gradient is 1 / (flow rate per unit forcing / area).
    spectral = 1.0 / quadrant_flow_rate(hartmann, 64, hartmann_wall=wall)
    assert spectral / NI_TABLE_3_4[(hartmann, wall)] - 1.0 == pytest.approx(4.5e-4, abs=2e-4)
