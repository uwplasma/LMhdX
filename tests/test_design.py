"""Fully developed duct design: linearity, exact drive elimination and gradients."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx.core3d import ChannelProblem, duct_problem, zero_velocity
from lmhdx.fully_developed import (
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
from lmhdx.grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, uniform_faces
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


# Crediting a heat load (thermal review corrections, part 6a): load-on less load-off on a held flow.


def _held_plug(speed: float, conductivity: float, field: float = 0.0, cells: int = 32):
    """A held plug flow along x on [0, 1], ``C = 1``, from an inlet at 0 with no conduction out of the outlet.

    A uniform unit load then delivers the steady fraction ``1 - (1 - exp(-lambda)) / lambda`` of itself through
    the outlet, ``lambda = C u / k``, and conducts the rest back out of the inlet. The two transverse axes are
    periodic, so a field along y drives the uniform current ``sigma u B`` and the Joule heat ``sigma u^2 B^2``.
    """
    from lmhdx.thermal import ThermalProblem, build_thermal_step, initial_thermal_state, thermal_boundary

    grid = Grid(uniform_faces(cells, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0))
    wrap = BoundaryCondition(PERIODIC)
    opening = BoundaryCondition(DIRICHLET, lower=speed * np.ones((1, 1)), upper_kind=NEUMANN)
    flow = ChannelProblem(
        grid, (opening, wrap, wrap), magnetic_field=(0.0, field, 0.0), dt=0.5 / (cells * speed)
    )
    ends = thermal_boundary("temperature", 0.0, upper_kind="heat_flux", conductivity=conductivity)
    thermal = ThermalProblem(flow, (ends, wrap, wrap), 1.0, conductivity, 0.0)
    rest = zero_velocity(flow)
    state = initial_thermal_state(
        thermal, 0.0, (rest[0].replace_data(jnp.full_like(rest[0].data, speed)), *rest[1:])
    )
    return build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True), state


def _uniform(state, value: float) -> Field:
    grid = state.temperature.grid
    return Field(jnp.full(grid.shape, value), (CENTER,) * 3, grid)


def _credit(speed: float, conductivity: float, field: float = 0.0):
    """The unit load's attribution and the load-on run's last diagnostics, at steady state (30 residence times)."""
    from lmhdx.thermal import load_attribution

    thermal_step, state = _held_plug(speed, conductivity, field)
    attribution = load_attribution(thermal_step, state, _uniform(state, 0.0), _uniform(state, 1.0), 1920)
    return attribution, jax.tree_util.tree_map(float, attribution.loaded)


def _leaving(lam: float) -> float:
    return 1.0 - (1.0 - np.exp(-lam)) / lam


def test_a_wall_heat_flux_enters_its_cells_exactly_as_a_source():
    """What ``load_attribution`` asks of a wall load: a flux ``q`` in at y = -1 and the source ``q A / V`` in the
    cells along that wall, with the wall insulated, give the same step to 1e-12 (16 cells, one flowing step)."""
    from lmhdx.thermal import ThermalProblem, build_thermal_step, initial_thermal_state, thermal_boundary

    faces = uniform_faces(16, -1.0, 1.0)
    flow = ChannelProblem(
        Grid(uniform_faces(1, 0.0, 1.0), faces, faces),
        (BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN), BoundaryCondition(NEUMANN)),
        magnetic_field=(0.0, 4.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    insulated, heated = (thermal_boundary("heat_flux", q, 0.0, conductivity=0.5) for q in (0.0, -3.0))
    wrap = BoundaryCondition(PERIODIC)
    walls, bare = (ThermalProblem(flow, (wrap, y, insulated), 2.0, 0.5, 300.0) for y in (heated, insulated))
    grid = flow.grid
    rest = zero_velocity(flow)
    moving = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])
    noise = 300.0 + np.random.default_rng(23).standard_normal(grid.shape)
    source = np.zeros(grid.shape)
    source[:, 0, :] = 3.0 * grid.face_areas(1)[:, 0, :] / grid.cell_volumes()[:, 0, :]

    def stepped(problem, heat):
        state = initial_thermal_state(problem, noise, moving)
        return build_thermal_step(problem, dtype=jnp.float64)(
            state, Field(jnp.asarray(heat), (CENTER,) * 3, grid)
        )[0]

    flux, cells = stepped(walls, np.zeros(grid.shape)), stepped(bare, source)
    assert float(jnp.max(jnp.abs(flux.temperature.data - cells.temperature.data))) <= 1e-12 * 300.0


def test_a_step_that_moves_its_flow_cannot_be_credited():
    """``load_attribution`` refuses a step built without ``hold_flow=True``, whose flow the load could move (#250)."""
    from lmhdx.thermal import (
        ThermalProblem,
        build_thermal_step,
        initial_thermal_state,
        load_attribution,
        thermal_boundary,
    )

    faces = uniform_faces(6, -1.0, 1.0)
    wrap, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
    flow = ChannelProblem(
        Grid(uniform_faces(1, 0.0, 1.0), faces, faces), (wrap, wall, wall), forcing=(1.0, 0.0, 0.0)
    )
    insulated = thermal_boundary("heat_flux", conductivity=1.0)
    thermal = ThermalProblem(flow, (wrap, insulated, insulated), 1.0, 1.0, 0.0)
    state = initial_thermal_state(thermal, 0.0)
    for held in (False, True):
        thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=held)
        assert thermal_step.hold_flow is held
    with pytest.raises(ValueError, match="hold_flow=True"):
        load_attribution(
            build_thermal_step(thermal, dtype=jnp.float64),
            state,
            _uniform(state, 0.0),
            _uniform(state, 1.0),
            4,
        )


def test_a_load_is_credited_by_difference_and_its_heat_balances():
    """O01: at ``lambda = 5`` what the load sends out of the outlet and back out of the inlet is the load, to 1e-12,
    with nothing left in storage; the delivered fraction is within 0.03 of the exact steady 0.801 on 32 cells."""
    attribution, _ = _credit(1.0, 0.2)
    imposed, delivered, conducted, stored, residual = (float(value) for value in attribution[:5])
    assert imposed == pytest.approx(1.0, rel=1e-12) and abs(stored) <= 1e-9
    assert abs(residual) <= 1e-12 and delivered + conducted == pytest.approx(1.0, rel=1e-12)
    assert conducted > 0.0 and abs(delivered - _leaving(5.0)) < 0.03


def test_more_flow_credits_neither_joule_heat_nor_gross_outlet_enthalpy():
    """O02: a field adds the Joule heat ``sigma u^2 B^2``, 9 and then 36 at twice the flow, and the gross outlet
    enthalpy grows with it to many times the load; the load's own credit stays below the load, rising only as less
    of it leaks back out of the inlet."""
    (slow, slow_run), (fast, fast_run) = (_credit(speed, 0.2, field=3.0) for speed in (1.0, 2.0))
    assert (slow_run.joule_power, fast_run.joule_power) == pytest.approx((9.0, 36.0), rel=1e-12)
    assert (
        slow_run.advective_heat_out > 5.0 and fast_run.advective_heat_out > 4.0 * slow_run.advective_heat_out
    )
    for credit in (slow, fast):
        assert float(credit.imposed) == pytest.approx(1.0, rel=1e-12) and 0.0 < float(credit.delivered) < 1.0
    assert float(slow.delivered) < float(fast.delivered) < _leaving(10.0)
    fieldless = _credit(2.0, 0.2)[0]
    for credit in (slow, fast):  # the whole balance, not only the delivered part
        assert abs(float(credit.residual)) <= 1e-12 and abs(float(credit.stored)) <= 1e-9
        assert float(credit.delivered + credit.conducted) == pytest.approx(1.0, rel=1e-12)
    for name in ("delivered", "conducted"):
        assert float(getattr(fast, name)) == pytest.approx(float(getattr(fieldless, name)), rel=1e-12)


def test_neither_the_background_nor_the_start_is_credited():
    """The load's response is the same with a background source of 2 and an uneven start as with neither, to 1e-12,
    and also while it is still filling the duct (50 steps, storing heat): the difference removes both exactly."""
    from lmhdx.thermal import load_attribution

    thermal_step, state = _held_plug(1.0, 0.2, field=3.0)
    uneven = state._replace(
        temperature=state.temperature.replace_data(
            jnp.asarray(np.random.default_rng(31).uniform(0.0, 2.0, state.temperature.shape))
        )
    )
    plain, busy = (
        load_attribution(thermal_step, start, _uniform(state, background), _uniform(state, 1.0), 50)
        for start, background in ((state, 0.0), (uneven, 2.0))
    )
    assert float(plain.stored) > 0.1  # still filling
    for name in ("imposed", "delivered", "conducted", "stored"):
        assert float(getattr(busy, name)) == pytest.approx(float(getattr(plain, name)), rel=1e-12, abs=1e-13)
    assert float(busy.loaded.external_power) == pytest.approx(3.0, rel=1e-12)  # the background is in the run


def test_heat_leaking_out_of_the_inlet_does_not_count_as_delivered_duty():
    """A conductive fluid (``lambda = 0.5``) keeps its peak under a limit of 0.5, which the ``lambda = 20`` duct
    exceeds, only by sending most of the load back out of the thermostatted inlet: it fails a 90 % delivery
    requirement, so it is not a feasible cooling design. In this fixture (``u = C = A = 1``) the outlet temperature is
    the delivered fraction, so the two limits are tied; the point is that only the delivery check names the leak."""
    accepted = []
    for conductivity in (2.0, 0.05):  # lambda 0.5 and 20
        credit, run = _credit(1.0, conductivity)
        accepted.append((run.temperature_max <= 0.5, float(credit.delivered) >= 0.9 * float(credit.imposed)))
    assert accepted == [(True, False), (False, True)]


# A thermal duct design (thermal review corrections, part 6b): the tutorial's PbLi workflow, run from the page.


def test_the_thermal_duct_design_runs_as_written_and_keeps_its_contracts():
    """The square reference reproduces the spectral flow on the screening mesh (-1.48 %, second order toward 0) and its
    Joule heat; a feasible design meets every limit and a rejected one says why; the cost is ``Q dp``, which pressure
    alone would misrank across flows, and the candidate is the cheapest feasible design evaluated, at its active wall
    limit; invalid designs are rejected, not ranked; and the refinement, open-duct, capacity and fixed-flux results
    hold."""
    from lmhdx.core3d import energy_budget

    page = Path(__file__).resolve().parents[1] / "docs/tutorials/duct_design_law.md"
    section = page.read_text().split("## A thermal design", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    run = {"__name__": "thermal_design"}
    exec(compile(code, str(page), "exec"), run)
    square, candidate, limit = run["square"], run["candidate"], run["T_MAX"]
    a, square_duct = np.sqrt(run["AREA"] / 4.0), run["ducts"][-1]
    hartmann, speed = square_duct["hartmann"], square["flow_rate"] / run["AREA"]
    spectral = run["LENGTH"] * run["MU"] * speed / (a**2 * flow_rate(hartmann, 40))
    assert -0.016 < square["pressure_drop"] / spectral - 1.0 < -0.013
    unloaded = run["trial"](square_duct, square["flow_rate"], 0.0)  # the Joule heat alone: no cancellation
    joule = float(energy_budget(square["velocity"], square_duct["flow"]).joule)
    assert unloaded["joule"] == pytest.approx(joule, rel=1e-9) and 0.0 < joule < square["hydraulic_power"]
    records = (
        [record for row in run["sweep"] for record in row] + run["edges"] + list(run["evaluated"].values())
    )
    for record in records:
        if record["rejected"]:
            assert all(isinstance(reason, str) and reason for reason in record["rejected"])
        else:
            assert (
                max(record["wall_peak"], record["fluid_peak"]) <= limit
                and record["re_over_ha"] <= run["LAMINAR"]
            )
    feasible = [record for record in records if not record["rejected"]]
    assert candidate["hydraulic_power"] == min(record["hydraulic_power"] for record in feasible)
    assert (
        limit - 0.1 < candidate["wall_peak"] <= limit
        and candidate["hydraulic_power"] < square["hydraulic_power"]
    )
    assert (
        candidate["wall_peak"] - candidate["fluid_peak"] > 0.05
    )  # a surface flux: the wall face above every cell
    assert any(  # across flows the lower pressure drop is not always the lower power
        one["pressure_drop"] < other["pressure_drop"] and one["hydraulic_power"] > other["hydraulic_power"]
        for one in feasible
        for other in feasible
    )
    for stopped in (0.0, np.nan):
        assert run["trial"](square_duct, stopped)["rejected"][0].startswith("invalid")
    assert run["largest_duty"](square_duct, 1e-3) is None  # Re/Ha fails it with no duty at all
    assert abs(run["refined"]["hydraulic_power"] / candidate["hydraulic_power"] - 1.0) < 0.02
    assert min(run["delivered"]) >= 0.99 and max(run["peaks"]) <= limit
    assert (
        abs(run["delivered"][1] - run["delivered"][0]) < 1e-4
        and abs(run["peaks"][1] - run["peaks"][0]) < 0.05
    )
    duties = dict(run["capacity"])
    assert (
        duties[1.0] == pytest.approx(run["DUTY"], rel=2e-3)
        and max(d for d in duties.values() if d is not None) > 1.5 * run["DUTY"]
    )
    for record in (square, candidate):  # buoyancy is not small in either: both are outside the held flow (#251)
        assert max(record["mixed_convection"].values()) > 0.1
    widths = dict(run["per_width"])  # at a fixed flux the slender end loses its advantage
    assert min(widths.values()) < 0.7 * widths[1.0] < widths[min(widths)]  # b/a 0.08
