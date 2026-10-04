"""The fully developed CaseSpec route onto the staggered core (plan step 1.10b)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx.cases import solve_fully_developed_fields as cell_centred_fields
from lmhdx.cases import solve_steady as cell_centred_solve
from lmhdx.core3d import ImposedField, duct_problem
from lmhdx.design import fluid_cell_areas
from lmhdx.fully_developed import (
    case_mesh,
    channel_problem,
    core_applies,
    solve_fully_developed,
    solve_fully_developed_fields,
    solve_fully_developed_transient,
)
from lmhdx.mesh import write_tabulated_field_npz
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


def _sheared(case, axial=0.0):
    """The case's field times ``1 + 0.3 z``, a divergence-free shear, with an optional axial component."""
    strength = case.magnetic_field.value[1]

    def field(y, z):
        return jnp.stack([axial + 0 * y, strength * (1.0 + 0.3 * z) + 0 * y, 0 * y], axis=-1)

    return _with(case, magnetic_field=MagneticFieldSpec("analytic", fn=field))


def test_an_axial_field_component_leaves_the_fully_developed_flow_alone():
    """``u x B`` of an axial flow has no ``B_x`` part, and the transverse force it adds is a gradient."""
    case = _case(10.0, 0.0, 12)
    axial = _with(case, magnetic_field=MagneticFieldSpec("constant", (3.0, *case.magnetic_field.value[1:])))
    assert channel_problem(axial).magnetic_field == (3.0, 10.0, 0.0)
    reference = solve_fully_developed_fields(case)[0]
    np.testing.assert_allclose(solve_fully_developed_fields(axial)[0], reference, rtol=1e-9, atol=1e-12)


def test_a_varying_field_runs_on_the_core_and_converges_to_the_cell_centred_answer(tmp_path):
    """Office, float64, Ha 20, ``B_y (1 + 0.3 z)``: core 16/32/64/128 cells 0.042082/0.039003/0.038792/0.038739
    (differences shrink 14.6x then 4.0x: second order); cell-centred 16/32/64 0.044505/0.039116/0.038733.
    """
    case = _sheared(_case(20.0, 0.0, 32))
    assert isinstance(channel_problem(case).magnetic_field, ImposedField)
    mean = _mean_velocity(lmhdx.solve(case))
    assert mean == pytest.approx(0.0387386, rel=0.007)
    assert mean == pytest.approx(_mean_velocity(cell_centred_solve(case)), rel=0.01)
    section = case_mesh(case)
    y, z = (np.asarray(centres) for centres in (section.y_centers, section.z_centers))
    yy, zz = np.meshgrid(y, z, indexing="ij")
    path = write_tabulated_field_npz(
        tmp_path / "field.npz", y=y, z=z, bx=0 * yy, by=20.0 * (1 + 0.3 * zz), bz=0 * yy
    )
    tabulated = _with(case, magnetic_field=MagneticFieldSpec("tabulated", table_path=str(path)))
    sampled, analytic = (channel_problem(c).magnetic_field.components for c in (tabulated, case))
    np.testing.assert_allclose(sampled, analytic, rtol=1e-12, atol=1e-12)
    small = _sheared(_case(20.0, 0.0, 12), axial=2.0)

    def throughput(scale):
        return jnp.mean(solve_fully_developed_fields(small, magnetic_field_scale=scale)[0])

    gradient = jax.grad(throughput)(1.0)
    difference = (float(throughput(1.0 + 1e-4)) - float(throughput(1.0 - 1e-4))) / 2e-4
    assert float(gradient) == pytest.approx(difference, rel=1e-6) and float(gradient) < 0.0


def _one_wall(case):
    """The Hunt duct with only its lower Hartmann wall conducting."""
    return _with(
        case,
        boundary_conditions=(
            BoundaryCondition("wall", "conducting_wall", region="conducting_wall", side="left"),
            BoundaryCondition("side", "insulating", region="insulating_wall", side="right,top,bottom"),
        ),
    )


def _resolved(case):
    return _with(case, geometry=dataclasses.replace(case.geometry, wall_model="resolved"))


@pytest.mark.parametrize(
    ("walls", "fine", "cell_centred"),
    [(_resolved, 0.0172604, 0.0174251), (_one_wall, 0.0237986, 0.0241184)],
)
def test_walls_resolved_in_cells_converge_to_the_cell_centred_answer(walls, fine, cell_centred):
    """Hunt Ha 20, c = 0.1 as a 0.1-thick wall of the fluid's conductivity in 8 cells.

    Office, float64, mean velocity on 32/64/96/128 cells: both Hartmann walls resolved
    0.017346/0.017288/0.017267/0.017260 (cell-centred 32/64/96 0.017425/0.017310/0.017265);
    the lower one alone 0.023915/0.023831/0.023807/0.023799 (0.024118/0.023868/0.023777).
    At Ha 100 the cell-centred solver is still 2.0 % low on 96 cells where the core is within 0.05 %.
    """
    case = walls(_case(20.0, 0.1, 32))
    problem = channel_problem(case)
    assert problem.wall_conductance == (0.0, 0.0, 0.0) and problem.wall_layers[1] is not None
    assert (problem.wall_layers[1][1] is None) == (walls is _one_wall)
    solution = lmhdx.solve(case)
    assert solution.residual < 1e-8
    assert _mean_velocity(solution) == pytest.approx(fine, rel=0.006)
    assert _mean_velocity(solution) == pytest.approx(cell_centred, rel=0.01)


def test_resolved_walls_on_both_axes_match_the_cell_centred_corners():
    """Hartmann walls 4x and side walls 0.5x the fluid's conductivity, 0.05 thick in 8 cells.

    Office, float64, mean velocity at Ha 20, core 32/64/96/128 0.0123907/0.0123487/0.0123297/0.0123234,
    cell-centred 32/64/96 0.0123993/0.0123538/0.0123333; Ha 100 core 0.00078544/0.00078017/0.00077984/
    0.00077940, cell-centred 0.00073014/0.00075151/0.00076372 (converging from below). Thin walls
    of the same conductance sit 0.16 % (Ha 20) and 0.45 % (Ha 100) lower.
    """
    base = lmhdx.make_hunt_case(ha=20.0, ny=32, nz=32, wall_thickness=0.05, wall_cells=8)
    case = _resolved(
        _with(
            base,
            regions=(
                base.regions[0],
                RegionSpec("hart", "solid", 4.0, 1.0, 1.0, 0.05),
                RegionSpec("side", "solid", 0.5, 1.0, 1.0, 0.05),
            ),
            boundary_conditions=(
                BoundaryCondition("h", "conducting_wall", region="hart", side="left_right"),
                BoundaryCondition("s", "conducting_wall", region="side", side="top_bottom"),
            ),
        )
    )
    assert all(channel_problem(case).wall_layers[1:])
    assert _mean_velocity(lmhdx.solve(case)) == pytest.approx(0.0123993, rel=0.001)


def test_a_resolved_wall_differentiates_in_the_field():
    case = _one_wall(_case(20.0, 0.1, 12))

    def throughput(scale):
        return jnp.mean(solve_fully_developed_fields(case, magnetic_field_scale=scale)[0])

    gradient = jax.grad(throughput)(1.0)
    difference = (float(throughput(1.0 + 1e-4)) - float(throughput(1.0 - 1e-4))) / 2e-4
    assert float(gradient) == pytest.approx(difference, rel=1e-6) and float(gradient) < 0.0


def test_thin_walls_conduct_on_both_axes():
    """Every wall a sheet, c = 0.05, joined in series at the corners.

    Office, float64, mean velocity: Ha 20 core 32/64/96 0.022599/0.022526/0.022502, the
    cell-centred solver with 0.02-thick walls in 8 cells 0.022559 (64 cells); Ha 100 core
    0.0018350/0.0018283/0.0018280, cell-centred 0.0017987 (0.02 thick) and 0.0018113 (0.01
    thick, 96 cells), approaching the core as the wall thins. The spectral side-wall reference
    gives 0.0015121 there, 21 % lower, so it is not used as a gate.
    """
    case = _with(
        _case(20.0, 0.05, 32),
        boundary_conditions=(
            BoundaryCondition(
                "walls", "conducting_wall", region="conducting_wall", side="left,right,top,bottom"
            ),
        ),
    )
    assert channel_problem(case).wall_conductance == pytest.approx((0.0, 0.05, 0.05), rel=1e-12)
    assert _mean_velocity(lmhdx.solve(case)) == pytest.approx(0.022559, rel=0.003)


def test_what_the_core_does_not_model_is_refused():
    shercliff, hunt = _case(5.0, 0.0, 8), _case(5.0, 0.05, 8)
    refused = [
        _with(shercliff, geometry=dataclasses.replace(shercliff.geometry, kind="pipe_ogrid")),
        _with(shercliff, regions=(*shercliff.regions, RegionSpec("second", "fluid", 1.0, 1.0, 1.0))),
        _with(shercliff, boundary_conditions=(BoundaryCondition("j", "imposed_current_density", 1.0),)),
        _with(shercliff, boundary_conditions=(BoundaryCondition("w", "conducting_wall", side="left_right"),)),
        # Thin walls must be equal on an axis.
        _with(_one_wall(hunt), geometry=dataclasses.replace(hunt.geometry, wall_model="thin")),
    ]
    for case in refused:
        with pytest.raises(NotImplementedError):
            channel_problem(case)
    with pytest.raises(ValueError, match="unramped"):
        channel_problem(
            _with(shercliff, magnetic_field=MagneticFieldSpec("constant", (0.0, 5.0, 0.0), ramp_duration=1.0))
        )
    with pytest.raises(NotImplementedError, match="even cell count"):
        channel_problem(lmhdx.make_shercliff_case(ha=20.0, ny=15, nz=16))


def test_a_case_the_core_does_not_represent_keeps_the_cell_centred_solve():
    case = lmhdx.make_shercliff_case(ha=5.0, ny=7, nz=8)
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


def _transient(case, dt, t_final, stride=0, **changes):
    return _with(
        case,
        solver=dataclasses.replace(case.solver, mode="transient"),
        time_stepper=dataclasses.replace(case.time_stepper, dt=dt, t_final=t_final, max_steps=10**6),
        output=dataclasses.replace(case.output, history_stride=stride),
        **changes,
    )


def _startup_mean(time, terms=200):
    """Mean velocity of the unit-forced square duct started from rest: the eigenfunction series."""
    odd = np.arange(1, 2 * terms, 2)
    m, n = np.meshgrid(odd, odd, indexing="ij")
    rate = np.pi**2 / 4 * (m**2 + n**2)
    return float(np.sum(64 / (np.pi**4 * m**2 * n**2 * rate) * (1 - np.exp(-rate * time))))


@pytest.mark.parametrize("hartmann", [1e-9, 20.0])
def test_a_transient_case_runs_on_the_core_at_first_order_in_time(hartmann):
    """Implicit Euler, the Lorentz force inside the CG solve: first order whatever ``dt sigma B^2 / rho``.

    Office, float64. Started from rest without a field, 64² at t = 0.5: dt 0.01/0.005/0.0025 give
    -0.44/-0.17/-0.04 % against the eigenfunction series. Ha 20 on 32², t = 0.1 (dt sigma B^2 = 0.8):
    dt 0.002/0.001 give 0.034498/0.034597, the cell-centred loop 0.034937 at dt 0.002 on its own mesh;
    Ha 100, t = 0.02: 0.0080035/0.0080156 against 0.0079559. 200 steps on 32² at Ha 20/100 take
    6.4-6.9 s cold and 0.35-0.54 s warm on the core, 98-102 s in the cell-centred loop.
    """
    duct = lmhdx.make_shercliff_case(ha=hartmann, ny=16, nz=16)
    means = []
    for dt in (0.02, 0.01, 0.005):
        solution = lmhdx.solve(_transient(duct, dt, 0.2))
        assert solution.status == "completed" and solution.steps == round(0.2 / dt)
        means.append(_mean_velocity(solution))
    assert 1.7 < (means[1] - means[0]) / (means[2] - means[1]) < 2.3
    if hartmann < 1.0:
        assert means[2] == pytest.approx(_startup_mean(0.2), rel=0.01)


def test_a_transient_run_restarts_holds_its_flow_rate_and_settles_on_the_steady_state():
    case = _transient(_case(10.0, 0.0, 12), 0.05, 1.0, stride=4)
    straight = solve_fully_developed_transient(case)
    assert straight.diagnostics.time_history.shape == (6,) and straight.state.time == pytest.approx(1.0)
    half = solve_fully_developed_transient(
        _with(case, time_stepper=dataclasses.replace(case.time_stepper, t_final=0.5))
    )
    resumed = solve_fully_developed_transient(case, initial_state=half.state)
    np.testing.assert_allclose(resumed.state.u, straight.state.u, rtol=1e-10, atol=1e-14)
    settled = solve_fully_developed_transient(
        _with(case, time_stepper=dataclasses.replace(case.time_stepper, t_final=6.0))
    )
    assert _mean_velocity(settled) == pytest.approx(
        _mean_velocity(lmhdx.solve(_case(10.0, 0.0, 12))), rel=1e-4
    )
    fixed = _transient(
        _case(10.0, 0.0, 12),
        0.05,
        0.5,
        stride=2,
        forcing=0.0,
        boundary_conditions=(BoundaryCondition("in", "inlet_flow_rate", 0.1),),
    )
    rates = np.asarray(solve_fully_developed_transient(fixed).diagnostics.volumetric_flow_rate_history)
    np.testing.assert_allclose(rates, 0.1, rtol=1e-12)
    ramped = _with(
        case,
        magnetic_field=MagneticFieldSpec("constant", (0.0, 10.0, 0.0), ramp_start=1.0, ramp_duration=1.0),
    )
    ramped_mean = _mean_velocity(solve_fully_developed_transient(ramped))
    assert ramped_mean > _mean_velocity(straight)
