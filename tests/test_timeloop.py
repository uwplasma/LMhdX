"""The compiled trajectory: equivalence with the host loop, and no host synchronisation."""

import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx import core3d
from lmhdx.core3d import (
    ChannelProblem,
    advance,
    energy_budget,
    kinetic_energy,
    step,
    trajectory_diagnostics,
    zero_velocity,
)
from lmhdx.grid import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
from lmhdx.ops import divergence

pytestmark = pytest.mark.unit

PERIODIC_AXIS = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _problem(cells: int = 8, **overrides) -> ChannelProblem:
    grid = Grid(uniform_faces(3, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_AXIS, WALL, WALL),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def test_the_compiled_trajectory_reproduces_the_host_loop():
    problem = _problem()
    factorization = problem.factorization()
    steps = 12

    velocity = zero_velocity(problem)
    for _ in range(steps):
        velocity, _, _ = step(velocity, problem, factorization)

    run = advance(problem, steps, factorization=factorization)
    for scanned, looped in zip(run.velocity, velocity, strict=True):
        assert np.max(np.abs(np.asarray(scanned.data) - np.asarray(looped.data))) < 1e-13


def test_the_trajectory_reports_a_history_without_synchronising():
    problem = _problem()
    run = advance(problem, 10)
    assert run.steps == 10
    assert run.divergence_residual.shape == (10,)
    assert run.kinetic_energy.shape == (10,)
    # A driven flow starting from rest gains energy monotonically at first.
    energy = np.asarray(run.kinetic_energy)
    assert np.all(np.diff(energy) > 0.0)
    assert float(np.max(np.asarray(run.divergence_residual))) < 1e-11


def test_the_end_state_carries_the_final_pressure_and_potential():
    problem = _problem()
    run = advance(problem, 6)
    assert run.pressure.shape == problem.grid.shape
    assert run.potential.shape == problem.grid.shape
    assert np.all(np.isfinite(np.asarray(run.pressure.data)))
    assert float(np.max(np.abs(np.asarray(divergence(run.velocity).data)))) < 1e-11


def test_the_time_loop_holds_no_host_synchronisation():
    """A single `float(...)` inside the loop would serialise the device queue."""
    source = inspect.getsource(core3d)
    for pattern in (r"\bfloat\(\s*[a-z_]+\.data", r"\bbool\(", r"device_get", r"\.item\(\)"):
        assert not re.search(pattern, source), pattern


def test_the_trajectory_compiles_once_and_differentiates():
    problem = _problem(cells=6)
    factorization = problem.factorization()

    def kinetic(drive):
        driven = ChannelProblem(
            grid=problem.grid,
            conditions=problem.conditions,
            forcing=(drive, 0.0, 0.0),
            dt=problem.dt,
        )
        return advance(driven, 8, factorization=factorization).kinetic_energy[-1]

    value, gradient = jax.jit(jax.value_and_grad(kinetic))(1.0)
    assert float(value) > 0.0
    size = 1.0e-5
    difference = (kinetic(1.0 + size) - kinetic(1.0 - size)) / (2.0 * size)
    assert float(gradient) == pytest.approx(float(difference), rel=1e-6)


def test_checkpointing_does_not_change_the_answer():
    """Rematerialisation is a memory trade, not a numerical one."""
    problem = _problem(cells=6)
    factorization = problem.factorization()
    plain = advance(problem, 8, factorization=factorization, checkpoint=False)
    saved = advance(problem, 8, factorization=factorization, checkpoint=True)
    for first, second in zip(plain.velocity, saved.velocity, strict=True):
        assert np.max(np.abs(np.asarray(first.data) - np.asarray(second.data))) < 1e-14


def test_a_longer_trajectory_costs_no_extra_live_state():
    """Scan length is static, so the traced graph does not grow with the run."""
    problem = _problem(cells=6)
    factorization = problem.factorization()
    short = jax.make_jaxpr(lambda: advance(problem, 4, factorization=factorization).velocity[0].data)()
    long = jax.make_jaxpr(lambda: advance(problem, 64, factorization=factorization).velocity[0].data)()
    # The scan body is compiled once; only its trip count differs.
    assert abs(len(str(long)) - len(str(short))) < 0.05 * len(str(short))
    # Only scalar diagnostics are stacked per step; the fields ride in the carry. Stacking the
    # pressure and potential held 3.1 GiB over 100 steps at 128^3 to return the last pair.
    scan = next(equation for equation in long.eqns if equation.primitive.name == "scan")
    assert all(var.aval.shape == (64,) for var in scan.outvars[scan.params["num_carry"] :])


def test_diagnostics_match_a_direct_evaluation():
    problem = _problem(cells=6)
    velocity = zero_velocity(problem)
    velocity, _, _ = step(velocity, problem)
    residual, energy = trajectory_diagnostics(velocity, problem)
    assert float(residual) == pytest.approx(float(jnp.max(jnp.abs(divergence(velocity).data))), rel=1e-12)
    assert float(energy) >= 0.0


def test_advance_validates_its_length():
    with pytest.raises(ValueError, match="steps must be positive"):
        advance(_problem(cells=4), 0)


def _uniform_duct(cells: int = 12, hartmann: float = 10.0, conductance: float = 0.0, **overrides):
    """A duct on a uniform mesh."""
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_AXIS, WALL, WALL),
        conductivity=1.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
        wall_conductance=(0.0, conductance, 0.0),
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def test_the_lorentz_force_does_exactly_minus_the_joule_dissipation():
    """A discrete identity, not an approximation: it holds at any state, to round-off."""
    problem = _uniform_duct()
    run = advance(problem, 20)
    budget = energy_budget(run.velocity, problem)
    assert float(jnp.abs(budget.ohmic_defect / budget.scale)) < 1e-12
    assert float(budget.joule) > 0.0
    assert float(budget.lorentz) < 0.0


@pytest.mark.parametrize(("hartmann", "cells"), [(20.0, 24), (100.0, 32)])
def test_the_ohmic_identity_is_exact_on_a_layer_mesh(hartmann, cells):
    """Stretched cells too: the force interpolation is the transpose of the electromotive one.

    Discretely, Lorentz work plus Joule dissipation is ``<div J, phi>``, which the
    potential solve makes zero to its own round-off. With the distance-weighted
    interpolation the two sides differed by 1e-4 to 3e-3 on these meshes. The
    defect is measured against the Joule dissipation itself, not the largest term
    of the budget, so a small current cannot hide it.

    A random divergence-free velocity carries currents as large as its
    electromotive force, and the identity closes with nothing subtracted. In a
    developed state the current is a small difference between the motional and
    the potential terms, so the round-off of the potential solve is amplified by
    the Hartmann number (1e-10 of the Joule term here); the charge term is then
    subtracted, which leaves the interpolations alone under test.
    """
    import dataclasses

    from lmhdx.core3d import duct_problem, face_currents, project, velocity_offset
    from lmhdx.grid import Field
    from lmhdx.ops import cell_inner_product

    problem = dataclasses.replace(duct_problem(hartmann=hartmann, cells=cells), dt=1.0e-2)
    marched = advance(problem, 10, viscous=problem.viscous_factorizations()).velocity
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    random = project(
        tuple(
            Field(
                jax.random.normal(
                    keys[axis], problem.grid.offset_shape(velocity_offset(axis)), dtype=jnp.float64
                ),
                velocity_offset(axis),
                problem.grid,
            )
            for axis in range(3)
        ),
        problem,
    )[0]
    for state in (marched, random):
        budget = energy_budget(state, problem)
        potential, currents, _ = face_currents(state, problem)
        charge = cell_inner_product(divergence(currents), potential)
        assert float(budget.joule) > 0.0
        assert float(jnp.abs((budget.ohmic_defect - charge) / budget.joule)) < 1e-12
    assert float(jnp.abs(budget.ohmic_defect / budget.joule)) < 1e-12


def test_a_conducting_wall_takes_power_out_through_the_boundary():
    """Leaving the boundary work out is a large error; with it the ohmic identity closes to round-off.

    The sheet takes its current across the half cell and dissipates it at its own potential. The adjacent
    cell value as the wall potential left a first-order defect: 2.7e-3 and 1.3e-3 on these meshes.
    """
    budgets = {}
    for cells in (12, 24):
        problem = _uniform_duct(cells, conductance=0.05)
        budgets[cells] = energy_budget(
            advance(problem, 20, viscous=problem.viscous_factorizations()).velocity, problem
        )
    budget = budgets[12]
    assert float(budget.wall) > 0.0
    # Dropping the boundary work is a percent-level error.
    without = float(jnp.abs((budget.joule + budget.lorentz) / budget.scale))
    assert without > 1.0e-2
    for budget in budgets.values():
        assert abs(float(budget.ohmic_defect / budget.scale)) < 1e-12


@pytest.mark.parametrize("conductance", [0.0, 0.05])
def test_the_ohmic_identity_holds_in_a_varying_field(conductance):
    """Plan step 1.9a: the ANL fringe with its solenoidal pair, on a layer mesh, with and without a thin wall.

    The field is a coefficient on the current faces of both paths, so the identity stays exact. A thin
    wall's half-cell current carries no electromotive force and so exerts no force; with the force taken
    from it, the fringe's ``B_x`` at the Hartmann walls left a defect of 2.3e-6 of the Joule term.
    """
    import dataclasses

    from lmhdx.core3d import duct_problem, fringe_field, project, velocity_offset
    from lmhdx.grid import Field

    duct = duct_problem(hartmann=20.0, cells=24, wall_conductance=conductance)
    grid = Grid(uniform_faces(8, -6.0, 6.0), duct.grid.y_faces, duct.grid.z_faces)
    problem = dataclasses.replace(duct, grid=grid, magnetic_field=fringe_field(grid, strength=20.0))
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    offsets = [velocity_offset(axis) for axis in range(3)]
    random = tuple(
        Field(
            jax.random.normal(keys[axis], grid.offset_shape(offsets[axis]), dtype=jnp.float64),
            offsets[axis],
            grid,
        )
        for axis in range(3)
    )
    budget = energy_budget(project(random, problem)[0], problem)
    assert float(budget.joule) > 0.0
    assert (float(budget.wall) > 0.0) == (conductance > 0.0)
    assert float(jnp.abs(budget.ohmic_defect / budget.joule)) < 1e-12


def test_the_budget_is_the_rate_of_change_of_kinetic_energy():
    """`defect` is dE/dt, so a step of half the size halves the error against it."""
    errors = []
    for step_size in (4.0e-3, 2.0e-3):
        problem = _uniform_duct(dt=step_size)
        start = advance(problem, 10).velocity
        budget = energy_budget(start, problem)
        after, _, _ = step(start, problem)
        rate = (kinetic_energy(after, problem) - kinetic_energy(start, problem)) / step_size
        errors.append(abs(float((rate - budget.defect) / budget.scale)))
    assert errors[1] < 0.6 * errors[0], errors


def test_a_mixed_precision_trajectory_follows_the_float64_one(true_float32_matmuls):
    """Float32 solves with float64 corrections, twenty implicit steps on a layer-resolving duct."""
    from lmhdx.grid import wall_resolving_faces

    hartmann = 20.0
    transverse, spanwise = (
        wall_resolving_faces(24, -1.0, 1.0, layer_thickness=thickness, cells_in_layer=6, max_ratio=None)
        for thickness in (1.0 / hartmann, 1.0 / np.sqrt(hartmann))
    )
    grid = Grid(uniform_faces(4, 0.0, 1.0), transverse, spanwise)
    runs = {}
    for precision in ("state", "mixed"):
        problem = _uniform_duct(hartmann=hartmann, grid=grid, precision=precision)
        runs[precision] = advance(
            problem, 20, factorization=problem.factorization(), viscous=problem.viscous_factorizations()
        )
    reference, mixed = runs["state"], runs["mixed"]
    scale = max(float(jnp.max(jnp.abs(field.data))) for field in reference.velocity)
    difference = max(
        float(jnp.max(jnp.abs(first.data - second.data)))
        for first, second in zip(mixed.velocity, reference.velocity, strict=True)
    )
    assert difference < 1e-9 * scale
    potential = reference.potential.data
    assert float(jnp.max(jnp.abs(mixed.potential.data - potential))) < 1e-9 * float(
        jnp.max(jnp.abs(potential))
    )
    # An axially uniform drive leaves no divergence to project: both pressures are round-off.
    assert float(jnp.max(jnp.abs(mixed.pressure.data - reference.pressure.data))) < 1e-12
    with pytest.raises(ValueError, match="precision must be"):
        _uniform_duct(precision="half")


def test_a_run_can_be_taken_in_chunks():
    """Restart is the same physics: the state is the whole of it."""
    problem = _problem()
    factorization = problem.factorization()
    whole = advance(problem, 12, factorization=factorization)
    first = advance(problem, 6, factorization=factorization)
    second = advance(problem, 6, first.velocity, factorization=factorization)
    for chunked, complete in zip(second.velocity, whole.velocity, strict=True):
        assert np.max(np.abs(np.asarray(chunked.data) - np.asarray(complete.data))) < 1e-14


def _thermal_of(problem: ChannelProblem):
    """The flow with insulated thermal walls, for its Joule heat; plan step 5.1d."""
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    insulated = thermal_boundary("heat_flux", conductivity=1.0)
    conditions = tuple(PERIODIC_AXIS if c.is_periodic else insulated for c in problem.conditions)
    return ThermalProblem(
        problem, conditions, specific_heat=1.0, thermal_conductivity=1.0, reference_temperature=0.0
    )


def _layered_duct() -> ChannelProblem:
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(12, -1.0, 1.0), uniform_faces(12, -1.0, 1.0))
    end = ((2.0, 2.0), (0.1, 0.1))
    return ChannelProblem(
        grid,
        (PERIODIC_AXIS, WALL, WALL),
        magnetic_field=(0.0, 10.0, 0.0),
        wall_layers=(None, (end, end), None),
    )


@pytest.mark.parametrize("walls", ["insulating-uniform", "insulating-stretched", "thin", "layers"])
def test_the_deposited_joule_heat_integrates_to_the_budget_joule_power(walls):
    """On one state, the fluid's deposited heat is energy_budget's ``joule`` to 1e-11 (ADR 0008, V12).

    Interior faces share their dissipation between two half cells, a wall face
    gives the fluid only its own, and a conducting wall's dissipation stays out.
    """
    from lmhdx.core3d import duct_problem, face_currents, project, velocity_offset
    from lmhdx.grid import CENTER, Field
    from lmhdx.ops import cell_inner_product
    from lmhdx.thermal import _joule_heating

    problem = {
        "insulating-uniform": lambda: _uniform_duct(12),
        "insulating-stretched": lambda: duct_problem(hartmann=20.0, cells=24),
        "thin": lambda: duct_problem(hartmann=20.0, cells=24, wall_conductance=0.05),
        "layers": _layered_duct,
    }[walls]()
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    noise = tuple(
        Field(
            jax.random.normal(keys[axis], problem.grid.offset_shape(velocity_offset(axis))),
            velocity_offset(axis),
            problem.grid,
        )
        for axis in range(3)
    )
    state = project(noise, problem)[0]
    heat = _joule_heating(face_currents(state, problem)[1], _thermal_of(problem))
    budget = energy_budget(state, problem)
    deposited = float(
        cell_inner_product(heat, Field(jnp.ones(problem.grid.shape), (CENTER,) * 3, problem.grid))
    )
    assert float(jnp.min(heat.data)) >= 0.0 and float(budget.joule) > 0.0
    assert abs(deposited - float(budget.joule)) <= 1e-11 * float(budget.joule)
    assert (float(budget.wall) > 0.0) == (walls in ("thin", "layers"))


def test_no_field_or_a_field_along_the_flow_deposits_no_heat():
    from lmhdx.core3d import face_currents
    from lmhdx.thermal import _joule_heating

    for field in ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)):
        problem = _uniform_duct(8, magnetic_field=field)
        rest = zero_velocity(problem)
        along = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])
        heat = _joule_heating(face_currents(along, problem)[1], _thermal_of(problem))
        assert float(jnp.max(jnp.abs(heat.data))) <= 1e-24


# The coupled thermal step and its heat ledger (plan step 5.1f).


def _heated(flow: ChannelProblem, conditions=None, **settings):
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    insulated = thermal_boundary("heat_flux", conductivity=0.5)
    conditions = conditions or tuple(PERIODIC_AXIS if c.is_periodic else insulated for c in flow.conditions)
    return ThermalProblem(flow, conditions, 2.0, 0.5, 300.0, **settings)


def _cells(thermal, values):
    from lmhdx.grid import CENTER, Field

    grid = thermal.flow.grid
    return Field(jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.shape), (CENTER,) * 3, grid)


@pytest.mark.parametrize("gravity", [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0)])
def test_a_passive_thermal_step_moves_the_flow_exactly_as_step_does(monkeypatch, gravity):
    """With one potential solve: the Lorentz force and the Joule heat share it. A stored ``g`` with ``beta = 0`` is passive."""
    from lmhdx.thermal import build_thermal_step, initial_thermal_state

    flow = _uniform_duct(8, conductance=0.05)
    viscous = flow.viscous_factorizations()
    velocity = advance(flow, 3, viscous=viscous).velocity
    thermal = _heated(flow, gravity=gravity)
    state = initial_thermal_state(thermal, 300.0, velocity)
    solves, solve = [], core3d._solve_potential
    monkeypatch.setattr(core3d, "_solve_potential", lambda *args: solves.append(1) or solve(*args))
    moved = build_thermal_step(thermal, dtype=jnp.float64)(state, _cells(thermal, 0.0))[0].velocity
    assert len(solves) == 1
    expected = step(velocity, flow, flow.factorization(), viscous)[0]
    assert all(np.array_equal(a.data, b.data) for a, b in zip(moved, expected))


@pytest.mark.parametrize("case", ["heated", "fluxes", "walls", "flowing"])
def test_the_heat_ledger_closes_at_its_rounding_floor(case):
    """Storage plus conduction out is Joule plus external heat, measured at ``max(1e-10 P, 64 eps C V T / dt)``."""
    from lmhdx.thermal import build_thermal_step, initial_thermal_state, thermal_boundary

    flow = _uniform_duct(8, hartmann=10.0 if case == "flowing" else 0.0)
    insulated = thermal_boundary("heat_flux", conductivity=0.5)
    walls = {
        "fluxes": thermal_boundary("heat_flux", 2.0, 5.0, conductivity=0.5),
        "walls": thermal_boundary("temperature", 290.0, 310.0),
    }.get(case, insulated)
    thermal = _heated(flow, (PERIODIC_AXIS, walls, insulated))
    noise = 0.1 * np.random.default_rng(17).standard_normal(flow.grid.shape)
    velocity = advance(flow, 5, viscous=flow.viscous_factorizations()).velocity if case == "flowing" else None
    state = initial_thermal_state(thermal, 300.0 + noise, velocity)
    source = _cells(thermal, 7.0 if case in ("heated", "flowing") else 0.0)
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64)
    floor = 64 * np.finfo(float).eps * thermal.heat_capacity * 4.0 * 310.0 / flow.dt
    for _ in range(3):
        state, record = thermal_step(state, source)
        terms = (record.joule_power, record.external_power, record.conductive_heat_out, record.storage_rate)
        power = max(abs(float(value)) for value in terms)
        assert abs(float(record.heat_balance_residual)) <= max(1e-10 * power, floor)
        assert float(record.heat_solve_residual) <= 1e-12 * 310.0 / flow.dt and bool(record.finite)
    if case == "fluxes":
        assert float(record.conductive_heat_out) == pytest.approx((2.0 + 5.0) * 2.0, rel=1e-12)
    if case == "walls":  # the wall cells keep evolving: no wall value is written into them
        assert float(jnp.max(state.temperature.data[:, [0, -1], :])) < 309.0
    assert (float(record.joule_power) > 0.0) == (case == "flowing")
    assert float(record.external_power) == pytest.approx(28.0 if case in ("heated", "flowing") else 0.0)


def test_the_thermal_step_keeps_its_structure_and_compute_dtype():
    """Five array leaves in, the same out, whole-output jit; a foreign dtype is refused, true float32 runs."""
    import subprocess
    import sys

    from lmhdx.thermal import build_thermal_step, initial_thermal_state

    thermal = _heated(_uniform_duct(6, hartmann=4.0), expansion_coefficient=1.0e-3, gravity=(0.0, 0.0, -1.0))
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64)
    state = initial_thermal_state(thermal, 300.0)
    leaves, tree = jax.tree_util.tree_flatten(state)
    new, record = jax.jit(thermal_step)(state, _cells(thermal, 1.0))
    assert len(leaves) == 5 and jax.tree_util.tree_structure(new) == tree and len(record) == 19
    flags = ("finite", "backflow_boundary_valid")
    values = [value for name, value in record._asdict().items() if name not in flags]
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves((new, values)))
    assert jax.tree_util.tree_structure(jax.eval_shape(thermal_step, state, _cells(thermal, 1.0))[0]) == tree
    with pytest.raises(ValueError, match="compute dtype"):
        build_thermal_step(thermal, dtype=jnp.float32)
    code = """
import jax.numpy as jnp
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
from lmhdx.thermal import ThermalProblem, build_thermal_step, initial_thermal_state, thermal_boundary
walls = BoundaryCondition(NEUMANN)
grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(6, -1.0, 1.0), uniform_faces(6, -1.0, 1.0))
flow = ChannelProblem(grid, (BoundaryCondition(PERIODIC), walls, walls), magnetic_field=(0.0, 4.0, 0.0))
insulated = thermal_boundary("heat_flux", conductivity=0.5)
thermal = ThermalProblem(flow, (BoundaryCondition(PERIODIC), insulated, insulated), 2.0, 0.5, 300.0)
state = initial_thermal_state(thermal, 300.0)
new, record = build_thermal_step(thermal, dtype=jnp.float32)(state, state.temperature)
print(new.temperature.dtype, record.sensible_energy.dtype, bool(record.finite))
"""
    environment = {**__import__("os").environ, "JAX_ENABLE_X64": "false"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["float32", "float32", "True"]


def test_thermal_inputs_are_checked_before_a_run():
    from lmhdx.thermal import initial_thermal_state, validate_thermal_inputs

    thermal = _heated(_uniform_duct(6))
    state, source = initial_thermal_state(thermal, 300.0), _cells(thermal, 0.0)
    validate_thermal_inputs(thermal, state, source, divergence_tolerance=1e-12)
    velocity = state.velocity
    leaky = (velocity[0], velocity[1].replace_data(velocity[1].data.at[:, 0, :].set(1.0)), velocity[2])
    swirl = (velocity[0], velocity[1].replace_data(velocity[1].data.at[:, 3, :].set(1.0)), velocity[2])
    for bad_state, bad_source, tolerance in (
        (state, source, -1.0),
        (state._replace(temperature=velocity[0]), source, 0.0),
        (
            state._replace(
                temperature=state.temperature.replace_data(state.temperature.data.at[0, 0, 0].set(np.nan))
            ),
            source,
            0.0,
        ),
        (
            state._replace(
                temperature=state.temperature.replace_data(state.temperature.data.astype(jnp.float32))
            ),
            source,
            0.0,
        ),
        (state._replace(velocity=velocity[:2]), source, 0.0),
        (state._replace(time=jnp.asarray(np.inf)), source, 0.0),
        (state._replace(velocity=leaky), source, 1e9),
        (state._replace(velocity=swirl), source, 1e-12),
        (state, _cells(_heated(_uniform_duct(8)), 0.0), 0.0),
    ):
        with pytest.raises(ValueError):
            validate_thermal_inputs(thermal, bad_state, bad_source, divergence_tolerance=tolerance)


# The compiled thermal run: scan, derivatives and compilation (plan step 5.1g).


def _buoyant_duct():
    from lmhdx.thermal import thermal_boundary

    insulated = thermal_boundary("heat_flux", conductivity=0.5)
    walls = (PERIODIC_AXIS, insulated, thermal_boundary("temperature", 301.0, 299.0))
    return _heated(
        _uniform_duct(6, hartmann=4.0), walls, expansion_coefficient=1.0e-2, gravity=(0.0, 0.0, -1.0)
    )


def _pattern(thermal, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(thermal.flow.grid.shape)


def test_the_thermal_scan_reproduces_the_stepped_loop_and_stacks_only_scalars():
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal = _buoyant_duct()
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64)
    state, source = initial_thermal_state(thermal, 300.0 + 0.3 * _pattern(thermal, 18)), _cells(thermal, 1.0)
    looped = state
    for _ in range(6):
        looped, _ = thermal_step(looped, source)
    final, history = jax.jit(lambda start, heat: advance_thermal(thermal_step, start, heat, 6))(state, source)
    for scanned, stepped in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(looped)):
        assert float(jnp.max(jnp.abs(scanned - stepped))) <= 1e-12 * max(
            1.0, float(jnp.max(jnp.abs(stepped)))
        )
    assert all(np.shape(value) == (6,) for value in history) and bool(jnp.all(history.finite))

    def traced(steps):
        return jax.make_jaxpr(lambda start: advance_thermal(thermal_step, start, source, steps))(state)

    short, long = traced(4), traced(128)
    assert abs(len(str(long)) - len(str(short))) < 0.01 * len(str(short))
    scan = next(equation for equation in long.eqns if equation.primitive.name == "scan")
    assert all(var.aval.shape == (128,) for var in scan.outvars[scan.params["num_carry"] :])
    with pytest.raises(ValueError, match="steps must be positive"):
        advance_thermal(thermal_step, state, source, 0)


def test_new_values_reuse_the_compiled_run_and_a_new_problem_rebuilds():
    import dataclasses

    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal = _buoyant_duct()
    thermal_step, traces = build_thermal_step(thermal, dtype=jnp.float64), []

    def run(start, heat):
        traces.append(1)
        return advance_thermal(thermal_step, start, heat, 3)[0].temperature.data

    compiled = jax.jit(run)
    first = compiled(initial_thermal_state(thermal, 300.0), _cells(thermal, 1.0))
    second = compiled(
        initial_thermal_state(thermal, 300.5 + 0.1 * _pattern(thermal, 19)), _cells(thermal, 2.0)
    )
    assert len(traces) == 1 and not np.allclose(first, second)
    for problem, same in ((dataclasses.replace(thermal, thermal_conductivity=2.0), False), (thermal, True)):
        rebuilt = build_thermal_step(problem, dtype=jnp.float64)
        again = advance_thermal(rebuilt, initial_thermal_state(thermal, 300.0), _cells(thermal, 1.0), 3)[0]
        assert np.allclose(again.temperature.data, first, rtol=1e-13, atol=0.0) == same


def _objective(thermal, target: str):
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal_step = build_thermal_step(thermal, dtype=jnp.float64)
    base, direction = 300.0 + 0.5 * _pattern(thermal, 20), _pattern(thermal, 21)

    def objective(amplitude, checkpoint=True):
        heated = target == "heat source"
        start = initial_thermal_state(thermal, base + (0.0 if heated else amplitude) * direction)
        heat = _cells(thermal, 1.0 + (amplitude if heated else 0.0))
        final, _ = advance_thermal(thermal_step, start, heat, 6, checkpoint=checkpoint)
        if target == "buoyant velocity":
            return sum(jnp.sum(component.data**2) for component in final.velocity)
        return jnp.sum((final.temperature.data - 300.0) ** 2)

    return objective


def test_checkpointing_changes_neither_the_thermal_run_nor_its_gradient():
    objective = _objective(_buoyant_duct(), "initial temperature")
    saved, plain = (jax.value_and_grad(lambda a, c=c: objective(a, c))(0.2) for c in (True, False))
    assert float(saved[0]) == pytest.approx(float(plain[0]), rel=1e-13)
    assert float(saved[1]) == pytest.approx(float(plain[1]), rel=1e-12)


@pytest.mark.parametrize("target", ["initial temperature", "heat source", "buoyant velocity"])
def test_thermal_gradients_match_central_differences(target):
    """The best of three difference sizes within 1e-5 (ADR 0008, V17); the velocity feels the temperature only by buoyancy."""
    objective = jax.jit(_objective(_buoyant_duct(), target))
    gradient = float(jax.jit(jax.grad(objective))(0.0))
    errors = [
        abs(gradient - float(objective(size) - objective(-size)) / (2.0 * size)) / abs(gradient)
        for size in (1.0e-2, 1.0e-3, 1.0e-4)
    ]
    assert abs(gradient) > 0.0 and min(errors) < 1e-5, (gradient, errors)


def test_the_thermal_page_workflow_runs_as_written():
    """The worked example of docs/physics/thermal.md, executed from the page (plan step 5.1h)."""
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "docs" / "physics" / "thermal.md"
    section = page.read_text().split("## A worked example", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"__name__": "thermal_page"}
    exec(compile(code, str(page), "exec"), namespace)
    assert namespace["history"].temperature_max.shape == (200,)
    assert float(namespace["problem"].expansion_coefficient) > 0.0  # both variants ran


# Heat carried into and out of an open duct (plan step 5.3).


def _plug_duct(cells: int, inlet: float = 0.0, conductivity: float = 0.2):
    """A duct open along x on [0, 1], one periodic cell across, and the held unit plug flow it carries."""
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    grid = Grid(uniform_faces(cells, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0))
    opening = BoundaryCondition(DIRICHLET, lower=np.ones((1, 1)), upper_kind=NEUMANN)
    flow = ChannelProblem(
        grid, (opening, PERIODIC_AXIS, PERIODIC_AXIS), forcing=(0.0, 0.0, 0.0), dt=0.5 / cells
    )
    ends = thermal_boundary("temperature", inlet, upper_kind="heat_flux", conductivity=conductivity)
    thermal = ThermalProblem(flow, (ends, PERIODIC_AXIS, PERIODIC_AXIS), 1.0, conductivity, 0.0)
    rest = zero_velocity(flow)
    return thermal, (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])


def test_an_open_duct_carries_its_heat_out_at_first_order():
    """Plug flow from an inlet at 0 with a source ``q``: the exact steady profile, at first order in ``h``.

    ``C u T' = k T'' + q`` with ``T(0) = 0`` and ``T'(1) = 0`` gives
    ``T = (q / C u) [x - (exp(lambda (x - 1)) - exp(-lambda)) / lambda]``, ``lambda = C u / k = 5``.
    """
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    errors = []
    for cells in (16, 32, 64):
        thermal, plug = _plug_duct(cells)
        thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
        start = initial_thermal_state(thermal, 0.0, plug)
        state, history = advance_thermal(thermal_step, start, _cells(thermal, 2.0), 20 * cells)  # to t = 10
        x = np.asarray(thermal.flow.grid.centers[0])
        exact = 2.0 * (x - (np.exp(5.0 * (x - 1.0)) - np.exp(-5.0)) / 5.0)
        errors.append(float(np.max(np.abs(np.asarray(state.temperature.data)[:, 0, 0] - exact))))
        assert float(jnp.max(jnp.abs(history.heat_balance_residual))) <= 1e-10 * 2.0
        assert abs(float(history.storage_rate[-1])) <= 1e-3 * 2.0 < float(history.advective_heat_out[-1])
    orders = np.log2(np.divide(errors[:-1], errors[1:]))
    assert np.all((orders > 0.8) & (orders < 1.2)), (errors, orders)


def test_an_inflow_at_the_duct_temperature_leaves_it_unchanged():
    """No source and an inlet at the duct's own 300: nothing changes, and what flows in flows out."""
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal, plug = _plug_duct(16, inlet=300.0)
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    start = initial_thermal_state(thermal, 300.0, plug)
    state, history = advance_thermal(thermal_step, start, _cells(thermal, 0.0), 40)
    assert float(jnp.max(jnp.abs(state.temperature.data - 300.0))) <= 1e-12 * 300.0
    assert float(jnp.max(jnp.abs(history.advective_heat_out))) <= 1e-12 * 300.0
    assert float(jnp.max(jnp.abs(history.heat_balance_residual))) <= 1e-10 * 300.0


def test_an_open_duct_holds_its_flow_and_takes_an_inlet_temperature():
    import dataclasses

    from lmhdx.thermal import (
        build_thermal_step,
        initial_thermal_state,
        thermal_boundary,
        validate_thermal_inputs,
    )

    thermal, plug = _plug_duct(8)
    with pytest.raises(ValueError, match="hold_flow"):
        build_thermal_step(thermal, dtype=jnp.float64)
    buoyant = dataclasses.replace(
        thermal, expansion_coefficient=1.0e-3, gravity=(-1.0, 0.0, 0.0)
    )  # along the open x
    with pytest.raises(ValueError, match="buoyancy"):
        build_thermal_step(buoyant, dtype=jnp.float64, hold_flow=True)
    for ends in (
        thermal_boundary("heat_flux", conductivity=0.2),
        thermal_boundary("temperature", 0.0, 1.0, upper_kind="heat_flux", conductivity=0.2),
    ):
        with pytest.raises(ValueError, match="inlet temperature"):
            dataclasses.replace(thermal, thermal_conditions=(ends, *thermal.thermal_conditions[1:]))
    state = initial_thermal_state(thermal, 0.0, plug)
    validate_thermal_inputs(thermal, state, _cells(thermal, 0.0), divergence_tolerance=1e-12)


# Flow returning through an outlet (review corrections, part 2): the incoming enthalpy is boundary data.


def _returning_duct(
    amplitude: float = 2.0, shape: str = "outlet", field: float = 0.0, conductance: float = 0.0
):
    """A duct open along x on [0, 2], walls at y = +-1, carrying a held discrete curl: divergence free exactly.

    ``psi = (y + 1) + amplitude s(x) sin(pi (y + 1))`` at the cell corners, with ``s = (x/2)^2`` (``"outlet"``:
    a band returns through the outlet while the net flow is 2) or ``s = sin(pi x / 2)`` (``"inside"``: it
    returns inside only). ``psi`` is constant on each wall, so no flow crosses one.
    """
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    x, y = uniform_faces(8, 0.0, 2.0), uniform_faces(6, -1.0, 1.0)
    ramp = (x / 2.0) ** 2 if shape == "outlet" else np.sin(np.pi * x / 2.0)
    wave = np.sin(np.pi * (y + 1.0))
    wave[[0, -1]] = 0.0  # exactly: sin(2 pi) is not zero in floating point, and no flow may cross a wall
    psi = (y + 1.0)[None, :] + amplitude * ramp[:, None] * wave[None, :]
    along = (np.diff(psi, axis=1) / np.diff(y)[None, :])[:, :, None]
    across = (-np.diff(psi, axis=0) / np.diff(x)[:, None])[:, :, None]
    grid = Grid(x, y, uniform_faces(1, 0.0, 1.0))
    opening = BoundaryCondition(DIRICHLET, lower=along[0], upper_kind=NEUMANN)
    rest = zero_velocity(ChannelProblem(grid, (opening, WALL, PERIODIC_AXIS)))
    velocity = (rest[0].replace_data(jnp.asarray(along)), rest[1].replace_data(jnp.asarray(across)), rest[2])
    from lmhdx.thermal import _outgoing_rate

    dt = 0.5 / float(jnp.max(_outgoing_rate(velocity)))
    flow = ChannelProblem(
        grid,
        (opening, WALL, PERIODIC_AXIS),
        magnetic_field=(0.0, 0.0, field),
        wall_conductance=(0.0, conductance, 0.0),
        dt=dt,
    )
    walls = (
        thermal_boundary("temperature", 300.0, upper_kind="heat_flux", conductivity=0.05),
        thermal_boundary("heat_flux", conductivity=0.05),
        PERIODIC_AXIS,
    )
    thermal = ThermalProblem(flow, walls, 1.0, 0.05, 0.0, outlet_backflow="reservoir_flux")
    return thermal, velocity


def test_the_outlet_face_brings_in_the_reservoir_temperature_where_the_flow_returns():
    """Two outlet faces, areas 2 and 3, velocities 1 and -0.25, cells at 310 and 330, reservoir at 290 and 280, C = 5.

    Rates 2 out and 0.75 back; 3100 leaves with the cell's temperature, 1050 comes in with the reservoir's, 2050 net.
    The second cell's 330 takes no part.
    """
    import dataclasses

    from lmhdx.grid import CENTER, Field
    from lmhdx.thermal import ThermalProblem, _carried_out, _outlet_exchange, _upwind_fluxes, thermal_boundary

    grid = Grid(uniform_faces(1, 0.0, 1.0), np.array([0.0, 2.0, 5.0]), uniform_faces(1, 0.0, 1.0))
    opening = BoundaryCondition(DIRICHLET, lower=np.zeros((2, 1)), upper_kind=NEUMANN)
    flow = ChannelProblem(grid, (opening, WALL, PERIODIC_AXIS))
    walls = (
        thermal_boundary("temperature", 300.0, upper_kind="heat_flux", conductivity=1.0),
        thermal_boundary("heat_flux", conductivity=1.0),
        PERIODIC_AXIS,
    )
    thermal = ThermalProblem(flow, walls, 5.0, 1.0, 0.0, outlet_backflow="reservoir_flux")
    rest = zero_velocity(flow)
    velocity = (rest[0].replace_data(rest[0].data.at[1, :, 0].set(jnp.array([1.0, -0.25]))), *rest[1:])
    temperature = Field(jnp.array([[[310.0], [330.0]]]), (CENTER,) * 3, grid)
    reservoir = jnp.array([[290.0], [280.0]])
    fluxes = _upwind_fluxes(temperature, velocity, thermal, reservoir)
    back, out, brought, valid = _outlet_exchange(velocity, reservoir, thermal, jnp.float64)
    assert (float(back), float(out), float(brought), bool(valid)) == (0.75, 2.0, 1050.0, True)
    assert float(_carried_out(fluxes, thermal)) == pytest.approx(2050.0, rel=1e-14)
    assert float(5.0 * 2.0 * fluxes[0].data[1, 0, 0]) == pytest.approx(3100.0, rel=1e-14)
    unchecked = dataclasses.replace(thermal, outlet_backflow="error")
    assert not bool(_outlet_exchange(velocity, None, unchecked, jnp.float64)[3])


@pytest.mark.parametrize("case", ["uniform", "reservoir", "joule"])
def test_a_returning_outlet_balances_its_heat_and_stays_bounded(case):
    """Positive net flow with a band returning through the outlet: each ledger closes to max(1e-10 P, 64 eps C V T/dt).

    ``uniform``: inlet, reservoir and duct at 300 stay at 300. ``reservoir``: a 280 reservoir cools the duct from the
    outlet, within [280, 300]. ``joule``: a field and thin conducting walls add Joule heat, and the ledger still closes.
    """
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal, velocity = _returning_duct(**(dict(field=3.0, conductance=0.05) if case == "joule" else {}))
    reservoir = 300.0 if case == "uniform" else 280.0
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    start = initial_thermal_state(thermal, 300.0, velocity)
    state, history = advance_thermal(
        thermal_step, start, _cells(thermal, 0.0), 30, backflow_temperature=reservoir
    )
    terms = (
        history.storage_rate,
        history.conductive_heat_out,
        history.advective_heat_out,
        history.joule_power,
    )
    power = jnp.max(jnp.stack([jnp.abs(term) for term in terms]), axis=0)
    floor = 64 * np.finfo(float).eps * thermal.heat_capacity * 4.0 * 300.0 / thermal.flow.dt  # volume 4
    assert bool(jnp.all(jnp.abs(history.heat_balance_residual) <= jnp.maximum(1e-10 * power, floor)))
    assert float(history.outlet_backflow_rate[0]) > 0.0 and bool(history.backflow_boundary_valid.all())
    if case == "uniform":
        assert float(jnp.max(jnp.abs(state.temperature.data - 300.0))) <= 1e-12 * 300.0
    elif case == "reservoir":
        assert 280.0 - 1e-12 <= float(jnp.min(state.temperature.data)) < 300.0
        assert float(jnp.max(state.temperature.data)) <= 300.0 + 1e-12
        assert float(history.outlet_backflow_enthalpy_in[-1]) == pytest.approx(
            thermal.heat_capacity * float(history.outlet_backflow_rate[-1]) * 280.0, rel=1e-14
        )
    else:
        assert float(history.joule_power[-1]) > 0.0


def test_the_outlet_condition_is_the_total_flux_one_and_converges_at_first_order():
    """``T = 1 + x^2`` against flow entering at the outlet: the Danckwerts datum converges, a switched Dirichlet does not.

    With ``u = -1`` the outlet face is incoming, and ``C u T_res = C u T - k dT/dn`` there gives
    ``T_res = T(1) + k T'(1) / C = 2.4``; the source is ``q = C u T' - k T''``. The inlet stays at ``T(0) = 1``.
    Replacing the datum by ``T(1) = 2`` solves a different problem: its error stays at least five times larger.
    """
    from lmhdx.thermal import (
        ThermalProblem,
        advance_thermal,
        build_thermal_step,
        initial_thermal_state,
        thermal_boundary,
    )

    errors = {2.4: [], 2.0: []}
    for cells in (16, 32, 64):
        grid = Grid(uniform_faces(cells, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0))
        opening = BoundaryCondition(DIRICHLET, lower=-np.ones((1, 1)), upper_kind=NEUMANN)
        flow = ChannelProblem(grid, (opening, PERIODIC_AXIS, PERIODIC_AXIS), dt=0.5 / cells)
        ends = thermal_boundary("temperature", 1.0, upper_kind="heat_flux", conductivity=0.2)
        thermal = ThermalProblem(
            flow, (ends, PERIODIC_AXIS, PERIODIC_AXIS), 1.0, 0.2, 0.0, outlet_backflow="reservoir_flux"
        )
        rest = zero_velocity(flow)
        reversed_flow = (rest[0].replace_data(-jnp.ones_like(rest[0].data)), *rest[1:])
        x = np.asarray(grid.centers[0])
        source = _cells(thermal, (-2.0 * x - 0.4)[:, None, None])
        thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
        for datum, found in errors.items():
            start = initial_thermal_state(thermal, 1.0, reversed_flow)
            state, _ = advance_thermal(thermal_step, start, source, 20 * cells, backflow_temperature=datum)
            found.append(float(np.max(np.abs(np.asarray(state.temperature.data)[:, 0, 0] - (1.0 + x**2)))))
    orders = np.log2(np.divide(errors[2.4][:-1], errors[2.4][1:]))
    assert np.all((orders > 0.8) & (orders < 1.2)), (errors, orders)
    assert errors[2.0][-1] >= 5.0 * errors[2.4][-1], errors


def test_an_outlet_needs_its_returning_temperature_and_refuses_an_unknown_one():
    import dataclasses

    from lmhdx.thermal import build_thermal_step, initial_thermal_state, validate_thermal_inputs

    thermal, velocity = _returning_duct()
    state, heat = initial_thermal_state(thermal, 300.0, velocity), _cells(thermal, 0.0)
    validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-10, backflow_temperature=280.0)
    validate_thermal_inputs(
        thermal, state, heat, divergence_tolerance=1e-10, backflow_temperature=np.full((6, 1), 280.0)
    )
    unchecked = dataclasses.replace(thermal, outlet_backflow="error")
    for problem, datum in (
        (thermal, None),
        (thermal, np.full((1, 6), 280.0)),
        (thermal, np.inf),
        (thermal, 280.0 + 1.0j),
        (unchecked, None),  # positive net flow, but a band returns through the outlet
        (unchecked, 280.0),
    ):
        with pytest.raises(ValueError):
            validate_thermal_inputs(
                problem, state, heat, divergence_tolerance=1e-10, backflow_temperature=datum
            )
    inside, recirculating = _returning_duct(shape="inside")
    validate_thermal_inputs(  # recirculation that never reaches the outlet is the flow's own business
        dataclasses.replace(inside, outlet_backflow="error"),
        initial_thermal_state(inside, 300.0, recirculating),
        _cells(inside, 0.0),
        divergence_tolerance=1e-10,
    )
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    with pytest.raises(ValueError, match="needs backflow_temperature"):
        thermal_step(state, heat)
    closed, _ = _plug_duct(4)
    for change in (
        dict(outlet_backflow="reservoir"),
        dict(advection="off", outlet_backflow="reservoir_flux"),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(closed, **change)
    with pytest.raises(ValueError, match="inflow-outflow"):
        dataclasses.replace(_heated(_uniform_duct(4)), outlet_backflow="reservoir_flux")


# Host checks and compiled contracts (review corrections, part 3).


def test_a_run_past_its_courant_limit_is_refused_before_it_starts():
    """``dt max(sum_f max(u_f . n_f, 0) A_f / V)``, the transport's own rate: refused above the limit, 0.8 by default.

    A held plug flow at Courant 0.5 passes and fails a 0.4 limit; at Courant 2 it is refused unless the heat only
    conducts. A closed duct's evolving flow is checked at its start: 300 across one unit cell at ``dt = 2e-3`` is 0.6.
    """
    import contextlib
    import dataclasses

    from lmhdx.thermal import initial_thermal_state, validate_thermal_inputs

    thermal, plug = _plug_duct(8)
    state, heat = initial_thermal_state(thermal, 0.0, plug), _cells(thermal, 0.0)
    validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-12, advective_cfl_limit=0.5)
    with pytest.raises(ValueError, match="Courant number 0.5 exceeds"):
        validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-12, advective_cfl_limit=0.4)
    for limit in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="advective_cfl_limit"):
            validate_thermal_inputs(
                thermal, state, heat, divergence_tolerance=1e-12, advective_cfl_limit=limit
            )
    fast = dataclasses.replace(thermal, flow=dataclasses.replace(thermal.flow, dt=2.0 / 8))
    with pytest.raises(ValueError, match="Courant number 2 exceeds"):
        validate_thermal_inputs(fast, state, heat, divergence_tolerance=1e-12)
    validate_thermal_inputs(
        dataclasses.replace(fast, advection="off"), state, heat, divergence_tolerance=1e-12
    )
    closed = _heated(_uniform_duct(6))
    rest = initial_thermal_state(closed, 300.0)
    for speed in (300.0, 500.0):
        along = rest.velocity[0].replace_data(jnp.full_like(rest.velocity[0].data, speed))
        moving = rest._replace(velocity=(along, *rest.velocity[1:]))
        refused = pytest.raises(ValueError, match="Courant number 1 exceeds")
        with refused if speed > 400.0 else contextlib.nullcontext():
            validate_thermal_inputs(closed, moving, _cells(closed, 0.0), divergence_tolerance=1e-12)
    plug_reservoir = dataclasses.replace(thermal, outlet_backflow="reservoir_flux")
    with pytest.raises(ValueError, match="needs backflow_temperature"):  # even with no face returning
        validate_thermal_inputs(plug_reservoir, state, heat, divergence_tolerance=1e-12)


def _primitives(jaxpr) -> set[str]:
    """Every primitive of a jaxpr and of the jaxprs inside it."""
    from jax.extend.core import ClosedJaxpr, Jaxpr

    names = set()
    for equation in jaxpr.eqns:
        names.add(equation.primitive.name)
        for value in equation.params.values():
            for inner in value if isinstance(value, (tuple, list)) else (value,):
                if isinstance(inner, ClosedJaxpr):
                    names |= _primitives(inner.jaxpr)
                elif isinstance(inner, Jaxpr):
                    names |= _primitives(inner)
    return names


def test_a_reservoir_run_compiles_whole_and_takes_new_values_without_retracing():
    """The reservoir is a numerical input like the heat source: one trace for new values, a loop-invariant scan input.

    The jitted step and scan match the eager steps to 1e-12 relative, keep the state's leaves, grids and offsets, stack
    only scalars, and hold no host callback; the scan's text grows by under 1% from 4 to 128 steps.
    """
    from lmhdx.thermal import advance_thermal, build_thermal_step, initial_thermal_state

    thermal, velocity = _returning_duct()
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    state, heat = initial_thermal_state(thermal, 300.0, velocity), _cells(thermal, 1.0)
    eager, tree = state, jax.tree_util.tree_structure(state)
    for _ in range(4):
        eager, record = thermal_step(eager, heat, backflow_temperature=280.0)
    one, _ = jax.jit(thermal_step)(state, heat, backflow_temperature=jnp.asarray(280.0))
    assert jax.tree_util.tree_structure(one) == tree and len(record) == 19
    traces = []

    def run(start, source, reservoir):
        traces.append(1)
        return advance_thermal(thermal_step, start, source, 4, backflow_temperature=reservoir)

    compiled = jax.jit(run)
    final, history = compiled(state, heat, jnp.asarray(280.0))
    warmer = compiled(state, heat, jnp.asarray(290.0))[0]
    assert len(traces) == 1 and float(jnp.min(warmer.temperature.data - final.temperature.data)) >= -1e-10
    assert not np.allclose(warmer.temperature.data, final.temperature.data)
    profile = compiled(state, heat, jnp.full((6, 1), 280.0))[0]  # a new shape is a new trace, the same run
    assert len(traces) == 2 and np.array_equal(profile.temperature.data, final.temperature.data)
    assert jax.tree_util.tree_structure(final) == tree and all(np.shape(value) == (4,) for value in history)
    for scanned, stepped in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(eager)):
        assert scanned.dtype == stepped.dtype == jnp.float64
        assert float(jnp.max(jnp.abs(scanned - stepped))) <= 1e-12 * float(jnp.max(jnp.abs(stepped)))

    def traced(steps):
        return jax.make_jaxpr(
            lambda start, reservoir: advance_thermal(
                thermal_step, start, heat, steps, backflow_temperature=reservoir
            )
        )(state, jnp.asarray(280.0))

    short, long = traced(4), traced(128)
    assert abs(len(str(long)) - len(str(short))) < 0.01 * len(str(short))
    scan = next(equation for equation in long.eqns if equation.primitive.name == "scan")
    assert len(scan.invars) == scan.params["num_consts"] + scan.params["num_carry"]  # nothing indexed by step
    assert all(var.aval.shape == (128,) for var in scan.outvars[scan.params["num_carry"] :])
    assert not any("callback" in name for name in _primitives(long.jaxpr))


def test_the_reservoir_derivative_lives_on_the_returning_faces_only():
    """``d(u T_face)/d T_res`` is ``u`` on a returning face and zero on a leaving one, exactly; a run's derivative in a
    reservoir amplitude matches central differences to 1e-5 (best of three sizes), checkpointed or not.

    The held flow does not depend on the reservoir, so no face changes sign under the perturbation. Across a face
    whose velocity changes sign, upwinding is only piecewise differentiable, and nothing is smoothed to hide it.
    """
    from lmhdx.thermal import _upwind_fluxes, advance_thermal, build_thermal_step, initial_thermal_state

    thermal, velocity = _returning_duct()
    state = initial_thermal_state(thermal, 300.0 + 2.0 * _pattern(thermal, 22), velocity)

    def outlet(reservoir):
        return _upwind_fluxes(state.temperature, velocity, thermal, reservoir)[0].data[-1]

    jacobian = np.asarray(jax.jacfwd(outlet)(jnp.full((6, 1), 280.0))).reshape(6, 6)
    normal = np.asarray(velocity[0].data[-1, :, 0])
    assert np.any(normal < 0.0) and np.any(normal > 0.0)
    assert np.array_equal(jacobian, np.diag(np.where(normal < 0.0, normal, 0.0)))
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    profile = jnp.asarray(np.linspace(-1.0, 1.0, 6)[:, None])

    def objective(amplitude, checkpoint=True):
        reservoir = 280.0 + amplitude * (1.0 + profile)
        final, _ = advance_thermal(
            thermal_step,
            state,
            _cells(thermal, 0.0),
            6,
            checkpoint=checkpoint,
            backflow_temperature=reservoir,
        )
        return jnp.mean(final.temperature.data)

    saved, plain = (jax.value_and_grad(lambda a, c=c: objective(a, c))(0.5) for c in (True, False))
    assert float(saved[0]) == pytest.approx(float(plain[0]), rel=1e-13)
    assert float(saved[1]) == pytest.approx(float(plain[1]), rel=1e-12)
    compiled = jax.jit(objective)
    gradient = float(jax.jit(jax.grad(objective))(0.5))
    errors = [
        abs(gradient - float(compiled(0.5 + size) - compiled(0.5 - size)) / (2.0 * size)) / abs(gradient)
        for size in (1.0e-1, 1.0e-2, 1.0e-3)
    ]
    assert gradient > 0.0 and min(errors) < 1e-5, (gradient, errors)  # a warmer reservoir warms the duct


# Review corrections, part 4: conductivity, electrical walls and preparation.


def test_a_heat_flux_wall_keeps_its_flux_only_when_rebuilt_with_a_new_conductivity():
    """A heat-flux condition stores ``q / k``: rebuilt at each ``k`` it lets out 2 and 5 over area 2, 14, both times;
    kept from ``k = 0.5`` under ``k = 2`` it lets out four times as much (#232)."""
    import dataclasses

    from lmhdx.thermal import build_thermal_step, initial_thermal_state, thermal_boundary

    flow, insulated = _uniform_duct(8, hartmann=0.0), thermal_boundary("heat_flux", conductivity=0.5)
    outs = []
    for k, built_with in ((0.5, 0.5), (2.0, 2.0), (2.0, 0.5)):
        walls = (PERIODIC_AXIS, thermal_boundary("heat_flux", 2.0, 5.0, conductivity=built_with), insulated)
        thermal = dataclasses.replace(_heated(flow, walls), thermal_conductivity=k)
        record = build_thermal_step(thermal, dtype=jnp.float64)(
            initial_thermal_state(thermal, 300.0), _cells(thermal, 0.0)
        )[1]
        outs.append(float(record.conductive_heat_out))
    assert outs == pytest.approx([14.0, 14.0, 56.0], rel=1e-12)


def test_resolved_electrical_walls_leave_the_temperature_on_the_fluid_cells():
    """``wall_layers`` widen only the electric solve (#235): the temperature keeps the fluid's cells and volume, 4,
    and the step deposits the fluid's budget Joule power to 1e-11 while the walls dissipate their own."""
    import dataclasses

    from lmhdx.core3d import project, velocity_offset
    from lmhdx.grid import Field
    from lmhdx.thermal import build_thermal_step, initial_thermal_state

    layered = _layered_duct()
    for problem in (layered, dataclasses.replace(layered, wall_layers=(None, None, None))):
        thermal = _thermal_of(problem)
        thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
        rest, record = thermal_step(initial_thermal_state(thermal, 1.0), _cells(thermal, 0.0))
        assert rest.temperature.grid == problem.grid and rest.temperature.shape == (1, 12, 12)
        assert float(record.sensible_energy) == pytest.approx(4.0, rel=1e-13)
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    noise = tuple(
        Field(
            jax.random.normal(keys[a], layered.grid.offset_shape(velocity_offset(a))),
            velocity_offset(a),
            layered.grid,
        )
        for a in range(3)
    )
    velocity, thermal = project(noise, layered)[0], _thermal_of(layered)
    record = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)(
        initial_thermal_state(thermal, 1.0, velocity), _cells(thermal, 0.0)
    )[1]
    budget = energy_budget(velocity, layered)
    assert float(budget.wall) > 0.0
    assert abs(float(record.joule_power) - float(budget.joule)) <= 1e-11 * float(budget.joule)


def test_a_held_flow_prepares_no_viscous_inverse(monkeypatch):
    from lmhdx.thermal import build_thermal_step

    built, prepare = [], ChannelProblem.viscous_factorizations
    monkeypatch.setattr(
        ChannelProblem, "viscous_factorizations", lambda self: built.append(1) or prepare(self)
    )
    closed = _heated(_uniform_duct(6))
    for thermal in (_plug_duct(8)[0], closed):
        build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    assert not built
    build_thermal_step(closed, dtype=jnp.float64)
    assert built == [1]
