"""Fully developed duct design: linearity, exact drive elimination and gradients."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx.core3d import ChannelProblem, duct_problem
from lmhdx.design import (
    DuctResponse,
    channel_cross_section_weights,
    channel_drive_for_flow_rate,
    channel_fixed_flow_hydraulic_power,
    channel_flow_rate,
    channel_flow_response,
    drive_for_flow_rate,
    fixed_flow_hydraulic_power,
    fluid_cell_areas,
    hydraulic_power,
    linear_flow_response,
    pressure_drop,
    volumetric_flow_rate,
)
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
from lmhdx.steady import solve_steady_state
from validation.shercliff import flow_rate

pytestmark = pytest.mark.unit

LENGTH = 2.5


def test_documented_profile_fit_recovers_drive_and_field():
    tutorial = Path(__file__).resolve().parents[1] / "docs/tutorials/fully_developed.md"
    code = tutorial.read_text().split("## Fit a measured velocity profile", 1)[1]
    code = code.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, str(tutorial), "exec"), namespace)
    fit, evaluate = namespace["fit"], namespace["evaluate"]
    np.testing.assert_allclose(fit.x, namespace["truth"], rtol=0, atol=1e-6)
    assert fit.fun < 1e-14
    initial = np.array([1.0, 1.0])
    value, gradient = evaluate(initial)
    step = 1e-4
    finite = np.array(
        [
            (evaluate(initial + step * axis)[0] - evaluate(initial - step * axis)[0]) / (2 * step)
            for axis in np.eye(2)
        ]
    )
    assert fit.fun < value
    np.testing.assert_allclose(gradient, finite, rtol=1e-5, atol=1e-8)
    # Symmetric drive/field controls cannot fit an antisymmetric target component.
    target, areas = namespace["target"], namespace["areas"]
    odd = 0.1 * jnp.sqrt(jnp.mean(target**2)) * jnp.linspace(-1, 1, target.shape[0])[:, None]
    namespace["target"] = target + odd
    namespace["normalization"] = jnp.sum(areas * (target + odd) ** 2)
    namespace["value_and_gradient"] = jax.jit(jax.value_and_grad(namespace["loss"]))
    incompatible = namespace["minimize"](
        evaluate, fit.x, jac=True, method="L-BFGS-B", bounds=[(0.1, 3.0), (0.5, 2.0)]
    )
    floor = float(jnp.sum(areas * odd**2) / namespace["normalization"])
    assert incompatible.success and incompatible.fun > 1e-4
    assert incompatible.fun == pytest.approx(floor, rel=1e-7)


def _case(ha: float = 5.0, ny: int = 12, nz: int = 12):
    return lmhdx.make_hartmann_case(ha=ha, ny=ny, nz=nz)


def _flow(case, drive, scale=1.0):
    velocity, *_ = lmhdx.solve_fully_developed_fields(case, forcing=drive, magnetic_field_scale=scale)
    return volumetric_flow_rate(case, velocity)


@pytest.mark.parametrize("factory", [lmhdx.make_hartmann_case, lmhdx.make_hunt_case])
def test_fluid_areas_are_the_cross_section_the_case_is_solved_on(factory):
    """A Hunt duct's walls are thin sheets on the core, so its mesh is the fluid alone."""
    from lmhdx.fully_developed import case_mesh

    case = factory(ha=5, ny=12, nz=12)
    areas = np.asarray(fluid_cell_areas(case))
    mesh = case_mesh(case)
    np.testing.assert_array_equal(areas, np.asarray(mesh.dy)[:, None] * np.asarray(mesh.dz)[None, :])
    assert areas.shape == (12, 12) and np.all(areas > 0.0)
    assert float(np.sum(areas)) == pytest.approx(case.geometry.width * case.geometry.height, rel=1e-12)
    velocity = jnp.ones(areas.shape)
    integrate = jax.jit(jax.value_and_grad(lambda u: volumetric_flow_rate(case, u)))
    value, gradient = integrate(velocity)
    assert float(value) == pytest.approx(float(areas.sum()), rel=1e-12)
    np.testing.assert_array_equal(gradient, areas)


def test_a_case_and_its_channel_problem_have_one_response():
    """``linear_flow_response`` of a case is ``channel_flow_response`` of the problem it routes to."""
    from lmhdx.fully_developed import channel_problem

    case = lmhdx.make_hunt_case(ha=20.0, ny=16, nz=16, wall_conductance_ratio=0.027)
    for scale in (1.0, 1.5):
        routed = float(linear_flow_response(case, magnetic_field_scale=scale).flow_per_unit_drive)
        native = channel_flow_response(channel_problem(case), magnetic_field_scale=scale).flow_per_unit_drive
        assert routed == pytest.approx(float(native), rel=1e-8)


def test_channel_cross_section_weights_sum_to_the_full_area():
    """Every transverse cell is fluid, on a uniform mesh and on a wall-resolving one alike."""
    for hartmann in (0.0, 20.0):
        problem = duct_problem(hartmann=hartmann, cells=32)
        weights = np.asarray(channel_cross_section_weights(problem))
        assert weights.shape == problem.grid.shape[1:]
        assert np.all(weights > 0.0)
        extent = problem.grid.extent
        assert float(weights.sum()) == pytest.approx(extent[1] * extent[2], rel=1e-12)


def test_flow_rate_is_linear_in_the_drive():
    case = _case()
    single = _flow(case, 1.0)
    double = _flow(case, 2.0)
    assert float(double) == pytest.approx(2.0 * float(single), rel=1e-10)
    assert abs(float(_flow(case, 0.0))) < 1e-12 * abs(float(single))


def test_eliminated_drive_hits_the_requested_throughput_exactly():
    case, target = _case(), 0.35
    drive = drive_for_flow_rate(case, target)
    assert float(_flow(case, drive)) == pytest.approx(target, rel=1e-10)


def test_the_analytic_drive_derivative_matches_automatic_differentiation():
    """``df/dQ = 1/G`` because the problem is linear in the drive."""
    case, target = _case(), 0.35
    response = linear_flow_response(case)
    analytic = 1.0 / float(response.flow_per_unit_drive)
    automatic = float(jax.grad(lambda value: drive_for_flow_rate(case, value))(target))
    assert automatic == pytest.approx(analytic, rel=1e-10)
    # And the forward relation is the same number seen from the other side.
    forward = float(jax.grad(lambda drive: _flow(case, drive))(1.0))
    assert forward == pytest.approx(float(response.flow_per_unit_drive), rel=1e-10)


def test_pressure_drop_and_power_follow_the_drive():
    drive, flow = 1.4, 0.6
    assert float(pressure_drop(drive, LENGTH)) == pytest.approx(drive * LENGTH, rel=1e-12)
    assert float(hydraulic_power(drive, flow, LENGTH)) == pytest.approx(drive * LENGTH * flow, rel=1e-12)
    with pytest.raises(ValueError, match="length must be positive"):
        pressure_drop(drive, 0.0)


def test_holding_throughput_costs_more_power_in_a_stronger_field():
    """Magnetic drag: the same throughput needs a larger drive as Ha grows."""
    case, target = _case(), 0.2
    powers = [
        float(fixed_flow_hydraulic_power(case, target, LENGTH, magnetic_field_scale=scale))
        for scale in (0.5, 1.0, 2.0)
    ]
    assert powers[0] < powers[1] < powers[2]
    assert all(value > 0.0 for value in powers)


def test_fixed_flow_power_is_differentiable_in_the_field_scale():
    case, target = _case(), 0.2

    def power(scale):
        return fixed_flow_hydraulic_power(case, target, LENGTH, magnetic_field_scale=scale)

    gradient = float(jax.grad(power)(1.0))
    step = 1.0e-4
    difference = (float(power(1.0 + step)) - float(power(1.0 - step))) / (2.0 * step)
    assert gradient == pytest.approx(difference, rel=1e-5)
    # Stronger field, more drag, more power.
    assert gradient > 0.0


def test_response_object_reports_its_field_and_converts_targets():
    case = _case()
    response = linear_flow_response(case, magnetic_field_scale=1.5)
    assert isinstance(response, DuctResponse)
    assert response.magnetic_field_scale == pytest.approx(1.5)
    assert float(response.drive_for(2.0)) == pytest.approx(
        2.0 / float(response.flow_per_unit_drive), rel=1e-12
    )


@pytest.mark.parametrize("route", ["case", "channel"])
def test_flow_rate_rejects_a_mismatched_velocity(route):
    problem, rate = (
        (_case(), volumetric_flow_rate)
        if route == "case"
        else (duct_problem(hartmann=0.0, cells=8), channel_flow_rate)
    )
    with pytest.raises(ValueError, match="does not match the mesh"):
        rate(problem, jnp.zeros((3, 3)))


def test_a_conducting_wall_costs_more_power_than_an_insulating_one():
    """Hunt versus Shercliff at fixed throughput: wall currents add drag."""
    target = 0.05
    insulating = lmhdx.make_shercliff_case(ha=5.0, ny=12, nz=12)
    conducting = lmhdx.make_hunt_case(ha=5.0, ny=12, nz=12, wall_cells=2, insulator_cells=2)
    insulating_power = float(fixed_flow_hydraulic_power(insulating, target, LENGTH))
    conducting_power = float(fixed_flow_hydraulic_power(conducting, target, LENGTH))
    assert conducting_power > insulating_power


def _uniform_duct(hartmann: float, cells: int) -> ChannelProblem:
    """A square insulating duct matching ``lmhdx.make_hartmann_case``'s ``[-1, 1]^2`` grid."""
    faces = uniform_faces(cells, -1.0, 1.0)
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), faces, faces),
        conditions=(BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN), BoundaryCondition(NEUMANN)),
        conductivity=1.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )


def test_channel_drive_round_trips_to_its_target_flow_rate():
    problem = duct_problem(hartmann=5.0, cells=16)
    target = 0.02
    drive = channel_drive_for_flow_rate(problem, target)
    solution = solve_steady_state(problem, forcing=(float(drive), 0.0, 0.0))
    achieved = channel_flow_rate(problem, solution.velocity[0].data[0])
    assert float(achieved) == pytest.approx(target, rel=1e-8)


def test_channel_flow_response_rejects_advection():
    problem = duct_problem(hartmann=0.0, cells=8, advection="central")
    with pytest.raises(ValueError, match="requires advection='off'"):
        channel_flow_response(problem)


@pytest.mark.parametrize(("hartmann", "cells", "bound"), [(20.0, 32, 0.02), (100.0, 48, 0.01)])
def test_channel_flow_rate_matches_the_spectral_reference_on_a_wall_resolving_mesh(hartmann, cells, bound):
    """Measured 0.887% at Ha 20/32 cells, 0.684% at Ha 100/48 cells."""
    problem = duct_problem(hartmann=hartmann, cells=cells)
    solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0))
    total = channel_flow_rate(problem, solution.velocity[0].data[0])
    extent = problem.grid.extent
    mean = float(total) / (extent[1] * extent[2])
    exact = flow_rate(hartmann, 40)
    assert abs(mean - exact) / exact < bound


def test_channel_flow_rate_is_exactly_linear_in_the_drive():
    """Measured Q(2f)/2Q(f) - 1 = 0.0, Q(0) = 0.0: exact on this CG solve."""
    problem = duct_problem(hartmann=20.0, cells=32)

    def flow_at(drive):
        solution = solve_steady_state(problem, forcing=(drive, 0.0, 0.0))
        return channel_flow_rate(problem, solution.velocity[0].data[0])

    single, double, zero = (float(flow_at(drive)) for drive in (1.0, 2.0, 0.0))
    assert double / (2.0 * single) - 1.0 == 0.0
    assert zero == 0.0


def test_channel_fixed_flow_power_matches_finite_differences_in_the_field_scale():
    problem = duct_problem(hartmann=20.0, cells=32)
    target = 0.01

    def power(scale):
        return channel_fixed_flow_hydraulic_power(problem, target, LENGTH, magnetic_field_scale=scale)

    gradient = float(jax.grad(power)(1.0))
    step = 1.0e-4
    difference = (float(power(1.0 + step)) - float(power(1.0 - step))) / (2.0 * step)
    assert gradient == pytest.approx(difference, rel=1e-5)
    # Stronger field, more drag, more power.
    assert gradient > 0.0
