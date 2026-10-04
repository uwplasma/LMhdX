"""The fully developed CaseSpec route onto the staggered core (plan step 1.10b)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx.cases import solve_fully_developed_fields as cell_centred_fields
from lmhdx.core3d import duct_problem
from lmhdx.design import fluid_cell_areas
from lmhdx.fully_developed import (
    case_mesh,
    channel_problem,
    core_applies,
    solve_fully_developed,
    solve_fully_developed_fields,
)
from lmhdx.specs import BoundaryCondition, MagneticFieldSpec, RegionSpec
from validation.shercliff import flow_rate, quadrant_flow_rate

pytestmark = pytest.mark.numerical

# The ten matched cases of #141: (Ha, wall conductance ratio, cells on the staggered core).
MATCHED = [
    (20.0, 0.0, 32),
    (100.0, 0.0, 48),
    (300.0, 0.0, 48),
    (1000.0, 0.0, 64),
    (20.0, 0.027, 32),
    (100.0, 0.027, 32),
    (300.0, 0.027, 32),
    (20.0, 0.1, 32),
    (100.0, 0.1, 32),
    (300.0, 0.1, 48),
]


def _case(hartmann, conductance, cells, **overrides):
    if conductance:
        return lmhdx.make_hunt_case(
            ha=hartmann, ny=cells, nz=cells, wall_conductance_ratio=conductance, **overrides
        )
    return lmhdx.make_shercliff_case(ha=hartmann, ny=cells, nz=cells, **overrides)


def _mean_velocity(solution):
    return float(solution.diagnostics.mean_velocity_history[-1])


@pytest.mark.parametrize(("hartmann", "conductance", "cells"), MATCHED)
def test_a_case_is_the_duct_problem_the_parity_was_measured_on(hartmann, conductance, cells):
    """Parity (#141) was measured on ``duct_problem``; the route has to build that problem."""
    routed = channel_problem(_case(hartmann, conductance, cells))
    reference = duct_problem(hartmann=hartmann, cells=cells, wall_conductance=conductance)
    assert routed.grid == reference.grid
    assert routed.conditions == reference.conditions
    assert routed.magnetic_field == pytest.approx(reference.magnetic_field, rel=1e-12)
    assert routed.wall_conductance == pytest.approx(reference.wall_conductance, rel=1e-12)
    assert (routed.density, routed.viscosity, routed.conductivity, routed.dt) == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("hartmann", "conductance", "cells", "exact"),
    [
        (20.0, 0.0, 32, 0.0383217842),
        (300.0, 0.027, 32, 0.00041667),
        pytest.param(1000.0, 0.0, 64, 0.000972103, marks=pytest.mark.slow),
        pytest.param(300.0, 0.1, 48, 0.00017073, marks=pytest.mark.slow),
    ],
)
def test_lmx_solve_of_a_case_meets_the_matched_accuracy(hartmann, conductance, cells, exact):
    """#141 measured +0.89 %, +0.89 %, +0.71 % and +0.71 %: each within the 1 % rule."""
    solution = lmhdx.solve(_case(hartmann, conductance, cells))
    assert solution.status == "converged" and solution.converged
    assert solution.residual < 1e-8
    assert solution.state.u.shape == (cells, cells)
    assert 0.0 < (_mean_velocity(solution) - exact) / exact < 0.01


def test_the_report_is_consistent_with_its_fields():
    case = _case(20.0, 0.027, 16)
    solution = solve_fully_developed(case)
    mesh = case_mesh(case)
    areas = np.asarray(mesh.dy)[:, None] * np.asarray(mesh.dz)[None, :]
    u, phi, jy, jz, lorentz = (np.asarray(value) for value in solve_fully_developed_fields(case))
    np.testing.assert_allclose(np.asarray(solution.state.u), u, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(solution.state.lorentz_x), lorentz, rtol=1e-12, atol=1e-15)
    diagnostics = solution.diagnostics
    assert float(diagnostics.volumetric_flow_rate_history[-1]) == pytest.approx((areas * u).sum(), rel=1e-12)
    assert float(diagnostics.applied_forcing_history[-1]) == case.forcing
    # Charge balance per cell: the flux imbalance of the smallest cell against the current.
    imbalance = float(diagnostics.div_current_max_history[-1]) * float(min(np.min(mesh.dy), np.min(mesh.dz)))
    assert imbalance < 1e-8 * float(np.abs(jy).max())
    assert abs(float(np.sum(areas * phi))) < 1e-12 * float(np.abs(phi).max())
    # The Lorentz force carries the drive in the core, so the power it absorbs is
    # what the drive puts in less the viscous dissipation: negative, and smaller.
    lorentz_power = float(diagnostics.lorentz_power_history[-1])
    assert -case.forcing * float((areas * u).sum()) < lorentz_power < 0.0
    assert float(diagnostics.ohmic_power[-1]) > 0.0


def test_a_physical_duct_follows_its_hartmann_number():
    """Width 4, rho 2, nu 3, sigma 5: the flow is the unit duct's, scaled by f a^2 / (rho nu)."""
    half, density, viscosity, conductivity, hartmann, drive = 2.0, 2.0, 3.0, 5.0, 20.0, 7.0
    case = lmhdx.make_shercliff_case(
        ha=hartmann,
        width=2 * half,
        height=2 * half,
        ny=32,
        nz=32,
        conductivity=conductivity,
        density=density,
        viscosity=viscosity,
    )
    solution = lmhdx.solve(dataclasses.replace(case, forcing=drive))
    unit = lmhdx.solve(_case(hartmann, 0.0, 32))
    scale = drive * half**2 / (density * viscosity)
    assert _mean_velocity(solution) == pytest.approx(scale * _mean_velocity(unit), rel=1e-7)
    assert channel_problem(case).grid.y_faces == pytest.approx(
        half * duct_problem(hartmann=20, cells=32).grid.y_faces
    )


def test_a_field_along_z_clusters_the_z_walls_to_the_hartmann_layer():
    case = lmhdx.make_shercliff_case(ha=20.0, ny=32, nz=32)
    swapped = dataclasses.replace(case, magnetic_field=MagneticFieldSpec("constant", (0.0, 0.0, 20.0)))
    along_y, along_z = channel_problem(case).grid, channel_problem(swapped).grid
    np.testing.assert_array_equal(along_z.y_faces, along_y.z_faces)
    np.testing.assert_array_equal(along_z.z_faces, along_y.y_faces)
    assert _mean_velocity(lmhdx.solve(swapped)) == pytest.approx(_mean_velocity(lmhdx.solve(case)), rel=1e-9)


def test_a_prescribed_flow_rate_is_met_exactly():
    case = _case(20.0, 0.027, 16)
    unit = lmhdx.solve(case)
    target = 0.3
    fixed = dataclasses.replace(
        case,
        forcing=0.0,
        boundary_conditions=(
            *case.boundary_conditions,
            BoundaryCondition("inlet", "inlet_flow_rate", target),
        ),
    )
    solution = lmhdx.solve(fixed)
    assert float(solution.diagnostics.volumetric_flow_rate_history[-1]) == pytest.approx(target, rel=1e-12)
    expected = target / float(unit.diagnostics.volumetric_flow_rate_history[-1])
    assert float(solution.diagnostics.applied_forcing_history[-1]) == pytest.approx(expected, rel=1e-10)
    u = solve_fully_developed_fields(fixed)[0]
    np.testing.assert_allclose(np.asarray(u), np.asarray(solution.state.u), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("conductance", [0.0, 0.027])
def test_the_field_and_drive_gradients_match_central_differences(conductance):
    case = _case(20.0, conductance, 12)

    def throughput(drive, scale):
        u = solve_fully_developed_fields(case, forcing=drive, magnetic_field_scale=scale)[0]
        return jnp.mean(u)

    value, gradient = jax.jit(jax.value_and_grad(throughput, argnums=(0, 1)))(1.0, 1.0)
    eager = jax.grad(throughput, argnums=(0, 1))(1.0, 1.0)
    np.testing.assert_allclose(eager, gradient, rtol=1e-10)
    step = 1e-4
    difference = (float(throughput(1.0, 1.0 + step)) - float(throughput(1.0, 1.0 - step))) / (2 * step)
    assert float(gradient[1]) == pytest.approx(difference, rel=1e-6)
    assert float(gradient[1]) < 0.0  # a stronger field brakes the flow
    # Linear in the drive, to the CG tolerance of the two solves: measured 5e-10.
    assert float(gradient[0]) == pytest.approx(float(value), rel=1e-8)


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


def _with(case, **changes):
    return dataclasses.replace(case, **changes)


def test_what_the_core_does_not_model_is_refused():
    shercliff, hunt = _case(5.0, 0.0, 8), _case(5.0, 0.05, 8)
    refused = [
        _with(shercliff, magnetic_field=MagneticFieldSpec("constant", (1.0, 5.0, 0.0))),
        _with(shercliff, magnetic_field=MagneticFieldSpec("analytic", fn=lambda y, z: y)),
        _with(shercliff, geometry=dataclasses.replace(shercliff.geometry, kind="pipe_ogrid")),
        _with(shercliff, regions=(*shercliff.regions, RegionSpec("second", "fluid", 1.0, 1.0, 1.0))),
        _with(shercliff, boundary_conditions=(BoundaryCondition("j", "imposed_current_density", 1.0),)),
        _with(shercliff, boundary_conditions=(BoundaryCondition("w", "conducting_wall", side="left_right"),)),
        # One conducting Hartmann wall: the two walls of an axis differ.
        _with(
            hunt,
            boundary_conditions=(
                BoundaryCondition("wall", "conducting_wall", region="conducting_wall", side="left"),
                BoundaryCondition("side", "insulating", region="insulating_wall", side="top_bottom"),
            ),
        ),
        # A conducting layer that no boundary names is a thick wall.
        _with(hunt, boundary_conditions=(BoundaryCondition("walls", "no_slip"),)),
    ]
    for case in refused:
        with pytest.raises(NotImplementedError):
            channel_problem(case)
    with pytest.raises(ValueError, match="unramped"):
        channel_problem(
            _with(shercliff, magnetic_field=MagneticFieldSpec("constant", (0.0, 5.0, 0.0), ramp_duration=1.0))
        )


def test_an_odd_mesh_is_as_accurate_as_its_even_neighbours():
    """An odd cell count keeps one geometric centre cell; measured against the spectral mean velocity.

    Office, float64, Ha 20: 15 / 16 cells +10.9 % / +7.7 %, 31 / 32 +0.94 % / +0.89 %,
    63 / 64 +0.45 % / +0.44 % (the cell-centred solver on 31 cells: +2.7 %).
    """
    exact = flow_rate(20.0, 48)
    errors = {}
    for ny, nz in ((31, 31), (32, 32), (31, 32)):
        solution = lmhdx.solve(lmhdx.make_shercliff_case(ha=20.0, ny=ny, nz=nz))
        assert solution.state.u.shape == (ny, nz) and solution.residual < 1e-8
        errors[ny, nz] = (_mean_velocity(solution) - exact) / exact
    assert 0.0 < errors[31, 31] < 0.0100 and 0.0 < errors[31, 32] < 0.0100
    assert errors[31, 31] == pytest.approx(errors[32, 32], rel=0.1)
    case = lmhdx.make_shercliff_case(ha=20.0, ny=11, nz=13)
    assert core_applies(case)

    def throughput(scale):
        return jnp.mean(solve_fully_developed_fields(case, magnetic_field_scale=scale)[0])

    gradient = jax.grad(throughput)(1.0)
    step = 1e-4
    difference = (float(throughput(1.0 + step)) - float(throughput(1.0 - step))) / (2 * step)
    assert float(gradient) == pytest.approx(difference, rel=1e-6) and float(gradient) < 0.0


def test_a_case_the_core_does_not_represent_keeps_the_cell_centred_solve():
    case = lmhdx.make_shercliff_case(ha=5.0, ny=7, nz=8)
    case = _with(case, regions=(*case.regions, RegionSpec("second", "fluid", 1.0, 1.0, 1.0)))
    assert not core_applies(case)
    velocity = lmhdx.solve_fully_developed_fields(case)[0]
    reference = cell_centred_fields(case)[0]
    assert bool(jnp.array_equal(velocity, reference))
    assert fluid_cell_areas(case).shape == velocity.shape


def test_a_derivative_in_the_drive_differentiates_no_solve():
    """At a fixed field the solve runs once, outside any trace; the drive only scales it."""
    case = lmhdx.make_hartmann_case(ha=2, ny=8, nz=8)

    def objective(drive):
        return jnp.mean(lmhdx.solve_fully_developed_fields(case, forcing=drive)[0])

    def loops(function):
        return str(jax.make_jaxpr(function)(1.0)).count("while[")

    assert loops(objective) == loops(jax.grad(objective)) == 0
    assert loops(lambda x: jax.jvp(objective, (x,), (1.0,))[1]) == 0
    traced = str(
        jax.make_jaxpr(lambda s: lmhdx.solve_fully_developed_fields(case, magnetic_field_scale=s)[0])(1.0)
    )
    assert traced.count("while[") > 0
    value, gradient = jax.value_and_grad(objective)(2.0)
    assert float(gradient) == pytest.approx(float(value) / 2.0, rel=1e-12)


def test_a_sweep_inside_a_trace_runs_the_shape_program():
    """2b.1 stage 5: an objective traced in the drive runs each Hartmann number's solve on concrete arrays.

    From the third Hartmann number on the mesh shape the solve is the shape's program on
    arrays built on the host, which stay concrete inside the enclosing trace.
    """
    for hartmann in (2.0, 3.0, 4.5):
        case = lmhdx.make_hartmann_case(ha=hartmann, ny=8, nz=8)

        def objective(drive, case=case):
            return jnp.mean(lmhdx.solve_fully_developed_fields(case, forcing=drive)[0])

        assert float(jax.jit(objective)(2.0)) == pytest.approx(float(objective(2.0)), rel=1e-12)
