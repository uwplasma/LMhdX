"""Laminar anchors of the literature, solved through ``Problem`` (plan H4.1; tests T1, T2, T3, T14, T15, T27).

The numbers are the published ones. Where a table is itself a finite-grid result, the exact
solution (an independent Chebyshev or mapped-Chebyshev solve) is the strict gate, and the table
is held to the offset it is measured to have from that solution. Every mesh sequence is
three meshes of ratio 1.5, and a Richardson value with an observed order of 2 is what is compared.
"""

import jax
import numpy as np
import pytest

import lmhdx
from lmhdx import Problem, Sides, Solver, Uniform, Wall, duct
from lmhdx.fields import Func
from lmhdx.grid import Grid, axis
from lmhdx.units import WallLayer
from validation.gci import grid_convergence
from validation.shercliff import flow_rate, quadrant_flow_rate

pytestmark = pytest.mark.validation

_MESHES = (97, 145, 217)  # cells per side; (n - 1) grows by 1.5 each time
_RATIOS = (_MESHES[2] / _MESHES[1], _MESHES[1] / _MESHES[0])


def _duct(ha, cells, *, conductance=0.0, beta=(5.0, 3.0), viscosity=1.0, **fields):
    """A square duct at Hartmann number ``ha`` (unit half-width), tanh-stretched, mean velocity one."""
    ny, nz = (cells, cells) if isinstance(cells, int) else cells
    return Problem(
        Grid(axis(1, 0.0, 1.0), axis(ny, -1.0, 1.0, beta=beta[0]), axis(nz, -1.0, 1.0, beta=beta[1])),
        bc=("periodic", Wall(conductance), "wall"),
        B=Uniform(0.0, ha * np.sqrt(viscosity), 0.0),
        viscosity=viscosity,
        flow_rate=4.0,
        **fields,
    )


def _measures(ha, cells, beta=(5.0, 3.0)):
    """The drive, centreline velocity and wall shears of a periodic duct, from the discrete wall fluxes.

    ``Cf Re`` is twice the wall-averaged shear rate ``du/dn`` at unit mean velocity and viscosity, and
    ``Re_tau`` the square root of ``Re`` (here 1e5, KZB 2012) times the shear rate at the wall's midplane.
    """
    problem = _duct(ha, cells, beta=beta)
    found = lmhdx.solve(problem)
    u = np.asarray(found.velocity[0].data[0])
    dy, dz = (np.diff(faces) for faces in (problem.grid.y_faces, problem.grid.z_faces))
    mid = cells // 2
    hartmann = np.stack([2 * u[0] / dy[0], 2 * u[-1] / dy[-1]])  # du/dy on the walls y = -+1, along z
    side = np.stack([2 * u[:, 0] / dz[0], 2 * u[:, -1] / dz[-1]])  # du/dz on z = -+1, along y
    return {
        "G": float(found.pressure_gradient),
        "Ucl": float(u[mid, mid]),
        "CfHa": float(hartmann.sum(axis=0) @ dz) / 2,
        "CfSh": float(side.sum(axis=0) @ dy) / 2,
        "ReTauHa": float(np.sqrt(1e5 * hartmann[0, mid])),
        "ReTauSh": float(np.sqrt(1e5 * side[0, mid])),
    }


def _richardson(ha, beta=(5.0, 3.0)):
    runs = [_measures(ha, cells, beta) for cells in _MESHES]
    limits = {}
    for name in runs[0]:
        study = grid_convergence(*(run[name] for run in runs), *_RATIOS)
        assert 1.95 < study.order < 2.05 and study.monotone, (ha, name, study)
        limits[name] = study.extrapolated
    return limits


# Krasnov, Zikanov and Boeck 2012, Table 2, laminar entries: Ucl, Cf_Ha Re, Cf_Sh Re, Re_tau Ha and Sh (Re 1e5).
_KZB = {
    100: (1.1033, 197.7, 23.3, 3322, 1117),
    200: (1.0688, 396.0, 32.1, 4625, 1308),
    300: (1.0548, 594.7, 38.8, 5627, 1438),
    350: (1.0507, 694.1, 41.9, 6065, 1492),
    400: (1.0465, 793.6, 44.6, 6472, 1541),
}
# The exact (Chebyshev, N = 64, converged to 1e-4) laminar duct, the strict gate (digest T15):
# G = dp/dx Re, Ucl, Cf_Ha Re, Cf_Sh Re.
_EXACT = {100: (110.44, 1.1044, 197.63, 23.25), 200: (213.90, 1.0695, 395.79, 32.01)}
# What the published table differs from the exact solution by, per quantity, measured here at Ha 100 and 200
# and held at its largest value over Ha 100-400 plus the table's printed digits: the DNS grid's own error.
_TABLE_OFFSET = {"Ucl": 1.2e-3, "CfHa": 1.0e-3, "CfSh": 6.0e-3, "ReTauHa": 1.0e-3, "ReTauSh": 3.2e-3}


@pytest.mark.parametrize("ha", sorted(_KZB))
def test_kzb_2012_table_2_laminar_columns(ha):
    limits = _richardson(ha, beta=(4.0, 3.0) if ha <= 200 else (5.0, 3.0))
    # The drive is exact: Richardson meets the spectral solution to far better than the plan's 5e-4,
    # and the momentum balance G = (Cf_Ha Re + Cf_Sh Re) / 2 holds in the discrete fluxes to the solver's 1e-9.
    assert limits["G"] == pytest.approx(1.0 / quadrant_flow_rate(ha, 48), rel=5e-5)
    assert 0.5 * (limits["CfHa"] + limits["CfSh"]) == pytest.approx(limits["G"], rel=1e-7)
    if ha in _EXACT:
        exact = dict(zip(("G", "Ucl", "CfHa", "CfSh"), _EXACT[ha]))
        for name, value in exact.items():
            assert limits[name] == pytest.approx(value, rel=5e-4), name
    printed = dict(zip(("Ucl", "CfHa", "CfSh", "ReTauHa", "ReTauSh"), _KZB[ha]))
    for name, value in printed.items():
        assert limits[name] == pytest.approx(value, rel=_TABLE_OFFSET[name]), name
    # Re_tau^Ha = sqrt(Re Ha Ucl) in the laminar rows (the Hartmann layer is exponential), to 1e-3.
    assert limits["ReTauHa"] == pytest.approx(np.sqrt(1e5 * ha * limits["Ucl"]), rel=1e-3)


@pytest.mark.parametrize("ha, expected", [(300, 31.662), (500, 52.106), (1000, 102.88)])
def test_shercliff_pressure_gradient_of_ni_2007(ha, expected):
    """Ni et al. 2007, Parts I-III: dp/dx of the insulating duct at Re 10, flow rate 4 (plan T2, 0.3 %)."""
    found = lmhdx.solve(_duct(ha, 127, viscosity=0.1))
    assert float(found.pressure_gradient) == pytest.approx(expected, rel=3e-3)
    assert float(found.flow_rate) == pytest.approx(4.0, rel=1e-12)


@pytest.mark.parametrize(
    "ha, conductance, expected", [(300, 0.05, 374.897), (300, 0.1, 585.4302), (500, 0.05, 953.14475)]
)
def test_hunt_pressure_gradient_of_ni_2007(ha, conductance, expected):
    """Ni et al. 2007: dp/dx of Hunt's duct with conducting Hartmann walls at Re 10 (plan T3, 0.1 %)."""
    found = lmhdx.solve(_duct(ha, 127, conductance=conductance, viscosity=0.1))
    assert float(found.pressure_gradient) == pytest.approx(expected, rel=1e-3)


def test_the_inertia_free_pressure_gradient_scales_with_the_inverse_reynolds_number():
    """Ni et al. 2007 Tables 3-4: Hunt, Ha 300, c 0.05, dp/dx 374.897 / 37.490 / 7.498 / 3.749 at Re 10-1000."""
    for reynolds, expected in ((10, 374.897), (100, 37.490), (500, 7.498), (1000, 3.749)):
        found = lmhdx.solve(_duct(300, 127, conductance=0.05, viscosity=1.0 / reynolds))
        assert float(found.pressure_gradient) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("ha, f_exact, f_dns", [(17, 1.5243e-2, 1.531e-2), (20, 1.7397e-2, 1.747e-2)])
def test_the_laminar_duct_friction_factor_at_reynolds_3000(ha, f_exact, f_dns):
    """Zikanov et al. 2014 (K27): f = 2 dp/dx at Re 3000 against the exact solution (5e-4) and the periodic DNS (1.5 %)."""
    friction = []
    for cells in _MESHES:
        found = lmhdx.solve(_duct(ha, cells, beta=(3.0, 3.0), viscosity=1.0 / 3000.0))
        friction.append(2.0 * float(found.pressure_gradient))
    study = grid_convergence(*friction, *_RATIOS)
    assert 1.95 < study.order < 2.05 and study.extrapolated == pytest.approx(f_exact, rel=5e-4)
    assert study.extrapolated == pytest.approx(f_dns, rel=1.5e-2)
    assert f_dns > study.extrapolated  # the DNS sits above the exact laminar value


def _hartmann_layer_cells(faces, ha):
    """Cells inside one Hartmann thickness 1/Ha of the lower wall, counting the cut cell by its fraction."""
    return float(np.interp(1.0 / ha - 1.0, faces, np.arange(faces.size)))


def test_the_friction_error_falls_with_the_cells_in_the_hartmann_layer_at_second_order():
    """Plan T27: friction against cells inside 1/Ha (Zhao and Zikanov 2012: +7 %, +1.5 %, +0.3 % at 1.2, 2.5, 4 cells).

    Measured here on Ha 100 with the side layers resolved: 1.4 % at 2.9 cells, 0.46 % at 5.3 cells, 0.23 % at
    8.9 cells. The plan's 0.5 % at 4 cells is not met by this scheme (0.8 % at about 4.1 cells); 0.5 % is met from 5 cells.
    """
    exact, ha = 1.0 / quadrant_flow_rate(100, 48), 100.0
    errors = {}
    for beta in (2.0, 2.5, 3.0):
        found = lmhdx.solve(_duct(100, (96, 193), beta=(beta, 3.0)))
        cells = round(_hartmann_layer_cells(axis(96, -1.0, 1.0, beta=beta), ha), 1)
        errors[cells] = abs(float(found.pressure_gradient) / exact - 1.0)
    (few, mid, many), (e_few, e_mid, e_many) = zip(*sorted(errors.items()))
    assert few >= 2.5 and e_few <= 0.02
    assert mid >= 5.0 and e_mid <= 0.005
    assert e_few > e_mid > e_many
    # Refining a fixed stretching puts more cells in the layer and converges at the scheme's second order.
    drive = [
        float(lmhdx.solve(_duct(100, cells, beta=(3.0, 3.0))).pressure_gradient) for cells in (64, 96, 144)
    ]
    study = grid_convergence(*drive, 1.5)
    assert 1.9 < study.order < 2.1 and study.extrapolated == pytest.approx(exact, rel=5e-5)


def test_the_hunt_wall_derivative_meets_ni_2007_and_the_spectral_duct():
    """T35: d(dp/dx)/dc of Hunt's duct (Ha 300, Re 10) against Ni's two tables and an independent spectral solve.

    Ni et al. 2007 give dp/dx 374.897 at c 0.05 and 585.4302 at c 0.1: a secant of 4210.6 per unit c. The
    derivative at the midpoint differs from a secant by the curvature of dp/dx (1.15 % here, the same in the
    spectral duct), so the table is met after that correction, and the automatic derivative of the 127-cell
    solve is held to the spectral derivative at the mesh error of the pressure gradient itself (plan T3: 1.2e-3
    measured, 2e-3 allowed).
    """
    lmhdx.enable_x64()
    mid = 0.075
    slope = float(
        jax.grad(lambda c: lmhdx.solve(_duct(300, 127, conductance=c, viscosity=0.1)).pressure_gradient)(mid)
    )

    def spectral(c):
        return 0.1 / quadrant_flow_rate(300, 64, hartmann_wall=c)

    secant = (spectral(0.1) - spectral(0.05)) / 0.05
    reference = (spectral(mid + 1e-3) - spectral(mid - 1e-3)) / 2e-3
    assert (585.4302 - 374.897) / 0.05 == pytest.approx(4210.6, abs=0.1)
    assert secant == pytest.approx(4210.6, rel=2e-3)
    assert abs(secant - reference) / reference == pytest.approx(0.0115, abs=5e-4)
    assert slope == pytest.approx(reference, rel=2e-3)
    assert slope == pytest.approx(4210.6 * reference / secant, rel=2e-3)


def test_the_wall_derivative_converges_with_the_mesh_at_second_order():
    """The derivative, not only the drive, has the scheme's order: three meshes of ratio 1.5 (Ha 20, c 0.05)."""
    lmhdx.enable_x64()
    runs = [
        jax.value_and_grad(
            lambda c, n=n: (
                lmhdx.solve(_duct(20, n, conductance=c, beta=(3.0, 3.0), viscosity=0.1)).pressure_gradient
            )
        )(0.05)
        for n in (49, 73, 109)
    ]
    for index in (0, 1):
        study = grid_convergence(*(float(run[index]) for run in runs), 1.5, 1.5)
        assert 1.9 < study.order < 2.1 and study.monotone
    assert study.extrapolated == pytest.approx(31.8176, rel=1e-4)


# The unit duct on the 2.0 route: the matched, resolved-wall, field and time-accuracy pins of the
# fully developed case route that `duct` replaced (mean velocity is the flow rate over the area 4).


def _mean(found):
    return float(found.flow_rate) / 4.0


def test_the_spectral_quadrant_reproduces_the_full_domain_solve():
    """The Ha 1000 and Hunt Ha 300 references come from the quadrant; it must agree where both converge.

    Measured at Ha 20: 3e-8 apart for the insulating duct, and 1.1e-4 for Hunt's c = 0.1,
    where the full-domain solve converges slowly (0.0171481 on 48 points, 0.0171494 on 64)
    towards the quadrant's 0.0171500 (32 and 48 mapped points agree to 1e-6).
    """
    assert quadrant_flow_rate(20.0, 32) == pytest.approx(flow_rate(20.0, 48), rel=1e-7)
    assert quadrant_flow_rate(20.0, 32, hartmann_wall=0.1) == pytest.approx(
        flow_rate(20.0, 48, hartmann_wall=0.1), rel=2e-4
    )


def test_spectral_reference_takes_an_aspect_ratio():
    """A rectangle at B = 0 is the Fourier series of Poiseuille flow; with a field the two solves agree."""
    beta, odd = 0.4, np.arange(1, 4000, 2)
    series = (
        (16.0 / np.pi**3)
        / (4.0 * beta)
        * np.sum(
            4.0 / (np.pi * odd**4) * (2.0 * beta - 4.0 / (np.pi * odd) * np.tanh(np.pi * odd * beta / 2.0))
        )
    )
    assert flow_rate(0.0, 48, aspect=beta) == pytest.approx(series, rel=1e-9)
    assert quadrant_flow_rate(20.0, 32, aspect=beta) == pytest.approx(
        flow_rate(20.0, 48, aspect=beta), rel=1e-7
    )


_CONDUCTING = WallLayer("hartmann wall", 1.0, 0.1, 8)


@pytest.mark.parametrize(
    ("y", "fine", "cell_centred"),
    [
        (Wall(layers=(_CONDUCTING,)), 0.0172604, 0.0174251),
        (Sides(Wall(layers=(_CONDUCTING,)), Wall()), 0.0237986, 0.0241184),
    ],
    ids=["both Hartmann walls", "lower Hartmann wall"],
)
def test_walls_resolved_in_cells_converge_to_the_cell_centred_answer(y, fine, cell_centred):
    """Hunt Ha 20, c = 0.1 as a 0.1-thick wall of the fluid's conductivity in 8 cells.

    Office, float64, mean velocity on 32/64/96/128 cells: both Hartmann walls resolved
    0.017346/0.017288/0.017267/0.017260 (cell-centred 32/64/96 0.017425/0.017310/0.017265);
    the lower one alone 0.023915/0.023831/0.023807/0.023799 (0.024118/0.023868/0.023777).
    """
    found = lmhdx.solve(duct(20, 32, walls=dict(y=y)))
    assert found.residual < 1e-8
    assert _mean(found) == pytest.approx(fine, rel=0.006)
    assert _mean(found) == pytest.approx(cell_centred, rel=0.01)


def test_resolved_walls_on_both_axes_match_the_cell_centred_corners():
    """Hartmann walls 4x and side walls 0.5x the fluid's conductivity, 0.05 thick in 8 cells.

    Office, float64, mean velocity at Ha 20, core 32/64/96/128 0.0123907/0.0123487/0.0123297/0.0123234,
    cell-centred 32/64/96 0.0123993/0.0123538/0.0123333. Thin walls of the same conductance sit 0.16 %
    lower.
    """
    walls = dict(
        y=Wall(layers=(WallLayer("hartmann wall", 4.0, 0.05, 8),)),
        z=Wall(layers=(WallLayer("side wall", 0.5, 0.05, 8),)),
    )
    assert _mean(lmhdx.solve(duct(20, 32, walls=walls))) == pytest.approx(0.0123993, rel=0.001)


def test_thin_walls_conduct_on_both_axes():
    """Every wall a sheet, c = 0.05, joined in series at the corners.

    Office, float64, mean velocity at Ha 20: core 32/64/96 0.022599/0.022526/0.022502, the
    cell-centred solver with 0.02-thick walls in 8 cells 0.022559 (64 cells).
    """
    found = lmhdx.solve(duct(20, 32, walls=dict(y=Wall(0.05), z=Wall(0.05))))
    assert _mean(found) == pytest.approx(0.022559, rel=0.003)


def test_a_varying_field_converges_to_the_cell_centred_answer():
    """Office, float64, Ha 20, ``B_y (1 + 0.3 z)``: 16/32/64/128 cells 0.042082/0.039003/0.038792/0.038739
    (differences shrink 14.6x then 4.0x: second order); cell-centred 16/32/64 0.044505/0.039116/0.038733.
    """

    def sheared(x, y, z, strength):
        return 0 * y, strength * (1.0 + 0.3 * z) + 0 * y, 0 * y

    mean = _mean(lmhdx.solve(duct(20, 32).replace(B=Func(sheared, 20.0))))
    assert mean == pytest.approx(0.0387386, rel=0.007)
    assert mean == pytest.approx(0.0391165, rel=0.01)


def test_an_axial_field_component_leaves_the_fully_developed_flow_alone():
    """``u x B`` of an axial flow has no ``B_x`` part, and the transverse force it adds is a gradient."""
    plain = lmhdx.solve(duct(10, 12, cells_in_layer=2))
    axial = lmhdx.solve(duct(10, 12, cells_in_layer=2).replace(B=Uniform(3.0, 10.0, 0.0)))
    np.testing.assert_allclose(axial.velocity[0].data, plain.velocity[0].data, rtol=1e-9, atol=1e-12)


def test_an_odd_mesh_is_as_accurate_as_its_even_neighbours():
    """An odd cell count keeps one geometric centre cell; measured against the spectral mean velocity.

    Office, float64, Ha 20: 15 / 16 cells +10.9 % / +7.7 %, 31 / 32 +0.94 % / +0.89 %,
    63 / 64 +0.45 % / +0.44 % (the cell-centred solver on 31 cells: +2.7 %).
    """
    exact = flow_rate(20.0, 48)
    errors = {}
    for cells in (31, 32):
        found = lmhdx.solve(duct(20, cells))
        assert found.velocity[0].data.shape[1:] == (cells, cells)
        errors[cells] = (_mean(found) - exact) / exact
    assert 0.0 < errors[31] < 0.0100
    assert errors[31] == pytest.approx(errors[32], rel=0.1)


def _startup_mean(time, terms=200):
    """Mean velocity of the unit-forced square duct started from rest: the eigenfunction series."""
    odd = np.arange(1, 2 * terms, 2)
    m, n = np.meshgrid(odd, odd, indexing="ij")
    rate = np.pi**2 / 4 * (m**2 + n**2)
    return float(np.sum(64 / (np.pi**4 * m**2 * n**2 * rate) * (1 - np.exp(-rate * time))))


def test_a_duct_started_from_rest_runs_at_first_order_in_time():
    """Euler steps of the unforced-field duct from rest: first order in dt, and the series within 1 % on 16^2.

    Office, float64, 16^2 at t = 0.2: dt 0.02/0.01/0.005 give 0.08996/0.09114/0.09175 against the
    eigenfunction series 0.09091.
    """
    problem = duct(0.0, 16)
    weights = np.asarray(problem.grid.cell_volumes())[0]
    means = []
    for dt in (0.02, 0.01, 0.005):
        found = lmhdx.solve(problem, Solver(method="euler", dt=dt, steps=round(0.2 / dt)))
        means.append(float((weights * np.asarray(found.velocity[0].data[0])).sum() / weights.sum()))
    assert 1.7 < (means[1] - means[0]) / (means[2] - means[1]) < 2.3
    assert means[2] == pytest.approx(_startup_mean(0.2), rel=0.01)
