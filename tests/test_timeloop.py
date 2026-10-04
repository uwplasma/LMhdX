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
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
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
