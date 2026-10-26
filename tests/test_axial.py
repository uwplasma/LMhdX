"""The inflow-outflow axis (plan 1.9b, D26): boundary data, symmetry, balances, buffers and the adjoint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx.axial import (
    CONVERGED,
    STAGNATED,
    UNSTEADY,
    _lift,
    _watch,
    axial_faces,
    charge_balance,
    fringe_duct,
    fully_developed_inlet,
    mass_balance,
    open_duct,
    pressure_drop,
    solve_open_duct,
    station_flow_rates,
    station_pressure,
)
from lmhdx.core3d import EULER, RK3, ChannelProblem, _step, velocity_offset, zero_velocity
from lmhdx.grid import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Grid, pad, uniform_faces
from lmhdx.poisson import fast_diagonal_helmholtz, fast_diagonal_poisson
from lmhdx.steady import _face_weights, _orthogonal_projection, steady_residual, with_inflow

pytestmark = pytest.mark.numerical

_COMPILES: list = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, duration, **_: (
        _COMPILES.append(event) if event.endswith("backend_compile_duration") else None
    )
)
_OPEN = BoundaryCondition(DIRICHLET, upper_kind=NEUMANN)
_WALL = BoundaryCondition(NEUMANN)


def _small_fringe(**overrides):
    settings = dict(hartmann=10.0, wall_conductance=0.02, upstream=3.0, downstream=3.0, spacing=0.5)
    settings.update(overrides)
    return fringe_duct(cells=12, cells_in_layer=3, **settings)


def test_array_boundary_data_is_frozen_hashable_and_padded_per_end():
    profile = np.arange(6.0).reshape(2, 3)
    condition = BoundaryCondition(DIRICHLET, lower=profile, upper=1.5, upper_kind=NEUMANN)
    profile[0, 0] = 99.0
    assert condition == BoundaryCondition(
        DIRICHLET, lower=np.arange(6.0).reshape(2, 3), upper=1.5, upper_kind=NEUMANN
    )
    assert hash(condition) == hash(
        BoundaryCondition(DIRICHLET, lower=condition.lower, upper=1.5, upper_kind=NEUMANN)
    )
    assert condition.is_mixed and not condition.is_homogeneous and condition.homogeneous() == _OPEN
    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(2, 0.0, 1.0), uniform_faces(3, 0.0, 1.0))
    data = jnp.ones(grid.shape)
    padded = np.asarray(pad(data, 0, condition, grid=grid))
    np.testing.assert_array_equal(padded[0], 2.0 * np.arange(6.0).reshape(2, 3) - 1.0)
    np.testing.assert_allclose(padded[-1], 1.0 + 1.5 * 0.25)
    with pytest.raises(ValueError, match="periodic"):
        BoundaryCondition(PERIODIC, upper_kind=NEUMANN)
    with pytest.raises(ValueError, match="two dimensional"):
        BoundaryCondition(DIRICHLET, lower=np.ones(3))
    with pytest.raises(ValueError, match="inhomogeneous"):
        fast_diagonal_poisson(grid, (condition, _WALL, _WALL))


def test_the_mixed_factorizations_invert_their_operators():
    """Pressure: Neumann inlet, p = 0 outlet; velocity: the outlet face free with zero gradient."""
    grid = Grid(
        axial_faces(-3.0, 2.0, (-1.0, 1.0), 0.4), uniform_faces(5, -1.0, 1.0), uniform_faces(4, -1.0, 1.0)
    )
    rng = np.random.default_rng(3)
    pressure = fast_diagonal_poisson(grid, (BoundaryCondition(NEUMANN, upper_kind=DIRICHLET), _WALL, _WALL))
    assert not pressure.singular
    rhs = zero_velocity(ChannelProblem(grid=grid, conditions=(_OPEN, _WALL, _WALL)))
    from lmhdx.grid import CENTER, Field
    from lmhdx.ops import laplacian

    source = Field(jnp.asarray(rng.standard_normal(grid.shape)), (CENTER,) * 3, grid)
    solved = pressure.solve(source)
    assert float(jnp.max(jnp.abs(laplacian(solved, pressure.conditions).data - source.data))) < 1e-10
    wall = BoundaryCondition(DIRICHLET)
    helmholtz = fast_diagonal_helmholtz(
        grid, velocity_offset(0), (_OPEN, wall, wall), shift=1.0, coefficient=0.3
    )
    assert helmholtz.slices[0] == slice(1, grid.shape[0] + 1)
    axial = rhs[0].replace_data(jnp.asarray(rng.standard_normal(rhs[0].shape)))
    from lmhdx.ops import staggered_laplacian

    result = helmholtz.solve(axial)
    applied = result.data - 0.3 * staggered_laplacian(result, (_OPEN, wall, wall)).data
    np.testing.assert_allclose(np.asarray(applied)[1:], np.asarray(axial.data)[1:], atol=1e-10)
    assert float(jnp.max(jnp.abs(result.data[0]))) == 0.0


def test_the_open_stokes_operator_is_symmetric_in_the_face_volume_inner_product():
    """The two end faces own half a cell each, which is what makes the outlet face a symmetric unknown."""
    problem = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0)
    factorization, weights = problem.factorization(), _face_weights(problem)
    rng = np.random.default_rng(0)

    def draw():
        velocity = tuple(
            v.replace_data(jnp.asarray(rng.standard_normal(v.shape))) for v in zero_velocity(problem)
        )
        return _orthogonal_projection(velocity, problem, factorization)

    def dot(first, second):
        return sum(jnp.sum(w.data * a.data * b.data) for w, a, b in zip(weights, first, second, strict=True))

    first, second = draw(), draw()
    applied = [steady_residual(v, problem, factorization, forcing=(0.0, 0.0, 0.0)) for v in (first, second)]
    left, right = float(dot(first, applied[1])), float(dot(second, applied[0]))
    assert abs(left - right) < 1e-12 * abs(left)
    assert float(dot(first, applied[0])) < 0.0
    assert float(jnp.max(jnp.abs(first[0].data[0]))) == 0.0, "the correction has no inlet flux"


def test_a_uniform_field_duct_is_its_fully_developed_flow_to_round_off():
    """With nothing to develop, the open duct reproduces the inlet profile and its gradient everywhere."""
    base = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0)
    uniform = ChannelProblem(
        grid=base.grid,
        conditions=(BoundaryCondition(PERIODIC), _WALL, _WALL),
        conductivity=1.0,
        magnetic_field=(0.0, 10.0, 0.0),
        wall_conductance=(0.0, 0.02, 0.02),
        dt=1.0,
    )
    problem = open_duct(uniform, 4.0)
    _, gradient = fully_developed_inlet(problem, 4.0)
    solution = solve_open_duct(problem)
    axial = np.asarray(solution.velocity[0].data)
    np.testing.assert_allclose(
        axial, np.broadcast_to(axial[0], axial.shape), rtol=0, atol=1e-10 * np.max(axial)
    )
    centres, means = station_pressure(solution.pressure)
    np.testing.assert_allclose(np.diff(np.asarray(means)) / np.diff(centres), gradient, rtol=1e-9)
    # p = 0 at the outlet face: the last centre sits half a cell upstream of it.
    width = float(problem.grid.widths[0][-1])
    assert float(means[-1]) == pytest.approx(-0.5 * width * gradient, rel=1e-9)


def test_the_fringe_duct_balances_mass_charge_and_flow_and_develops_upstream():
    """1.9b exit: mass, charge and flow rate to 1e-12; the uniform region's gradient within 0.5 %."""
    problem = _small_fringe(upstream=6.0)
    _, gradient = fully_developed_inlet(problem, 4.0)
    solution = solve_open_duct(problem)
    assert float(solution.residual_norm) <= 1e-9 * float(solution.initial_residual_norm)
    assert float(jnp.max(jnp.abs(station_flow_rates(solution.velocity) - 4.0))) < 1e-12 * 4.0
    assert float(mass_balance(solution.velocity)) < 1e-12
    assert float(charge_balance(solution, problem)) < 1e-12
    np.testing.assert_array_equal(np.asarray(solution.currents[0].data)[[0, -1]], 0.0)
    centres, means = station_pressure(solution.pressure)
    upstream = (centres > -8.0) & (centres < -5.0)
    slope = np.polyfit(centres[upstream], np.asarray(means)[upstream], 1)[0]
    assert slope == pytest.approx(gradient, rel=5e-3)
    # lmhdx.solve routes an open duct here rather than solving it with no inflow (#226).
    np.testing.assert_array_equal(
        np.asarray(lmhdx.solve(problem).pressure.data), np.asarray(solution.pressure.data)
    )


def test_doubled_buffers_change_neither_the_drop_nor_the_window_current():
    """Row 28: twice the buffers move the drop over [-6, 2] and the axial current at its ends by <= 0.5 %."""

    def measure(scale):
        problem = _small_fringe(upstream=15.0 * scale, downstream=10.0 * scale, spacing=0.5)
        solution = solve_open_duct(problem)
        faces = problem.grid.x_faces
        areas = problem.grid.face_areas(0)
        currents = [
            float(np.sum(np.abs(np.asarray(solution.currents[0].data)[i]) * areas[i]))
            for i in (int(np.argmin(np.abs(faces + 6.0))), int(np.argmin(np.abs(faces - 2.0))))
        ]
        return float(pressure_drop(solution.pressure, -6.0, 2.0)), currents

    (drop, currents), (doubled_drop, doubled_currents) = measure(1.0), measure(2.0)
    assert doubled_drop == pytest.approx(drop, rel=5e-3)
    np.testing.assert_allclose(doubled_currents, currents, rtol=5e-3)


def test_the_pressure_drop_adjoint_matches_central_differences():
    problem = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0)

    def drop(scale):
        return pressure_drop(solve_open_duct(problem, field_scale=scale).pressure, -4.0, 2.0)

    gradient = float(jax.grad(drop)(1.0))
    step = 1e-4
    central = (float(drop(1.0 + step)) - float(drop(1.0 - step))) / (2.0 * step)
    assert gradient == pytest.approx(central, rel=1e-6)


def test_building_and_solving_an_open_duct_compiles_a_few_programs_and_a_repeat_none():
    """H0.1: the inlet profile and the solve are compiled programs, not hundreds of eagerly dispatched operations."""
    del _COMPILES[:]
    problem = _small_fringe(hartmann=9.0, spacing=1.0)
    first = solve_open_duct(problem)
    built = len(_COMPILES)
    second = solve_open_duct(problem, field_scale=1.01)
    assert built < 40 and len(_COMPILES) == built
    assert float(second.residual_norm) <= 1e-9 * float(second.initial_residual_norm)
    assert float(first.residual_norm) <= 1e-9 * float(first.initial_residual_norm)


def test_invalid_open_axes_are_refused():
    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(4, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    with pytest.raises(ValueError, match="one inflow-outflow axis"):
        ChannelProblem(grid=grid, conditions=(_OPEN, _OPEN, _WALL))
    with pytest.raises(ValueError, match="has no wall"):
        ChannelProblem(grid=grid, conditions=(_OPEN, _WALL, _WALL), wall_conductance=(0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="first axis"):
        solve_open_duct(ChannelProblem(grid=grid, conditions=(_WALL, _OPEN, _WALL)))
    from lmhdx.core3d import fringe_field

    varying = ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), _WALL, _WALL),
        magnetic_field=fringe_field(grid, half_length=0.5, centre=0.0, solenoidal=True),
    )
    with pytest.raises(ValueError, match="uniform over the inlet"):
        fully_developed_inlet(varying, 1.0)
    with pytest.raises(ValueError, match="no inflow-outflow axis"):
        with_inflow(zero_velocity(varying), varying)
    with pytest.raises(ValueError, match="one inflow-outflow axis"):
        ChannelProblem(grid=grid, conditions=(BoundaryCondition(NEUMANN, upper_kind=DIRICHLET), _WALL, _WALL))
    with pytest.raises(ValueError, match="unknown boundary kind"):
        BoundaryCondition(DIRICHLET, upper_kind="outflow")
    with pytest.raises(ValueError, match="lower <= core"):
        axial_faces(0.0, 1.0, (0.5, 2.0), 0.1)
    # Transport needs buffers at the core spacing (1.9d): growth 1 keeps every axial cell at it.
    assert np.allclose(np.diff(axial_faces(-4.0, 3.0, (-1.0, 1.0), 0.5, growth=1.0)), 0.5)
    assert np.allclose(np.diff(_small_fringe(growth=1.0).grid.faces[0]), 0.5)


def test_the_inertial_fringe_is_solved_balanced_and_differentiated_by_newton_and_by_the_march():
    """H3: Ha 10, Re 5 (N 20), central transport; both methods against the pinned Newton root of `main` and each other (1e-6), the march's adjoint against central differences."""
    problem = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0, flow_rate=20.0, advection="central")
    found = {}
    for method in ("newton", "march"):

        def drop(scale, method=method):
            solution = solve_open_duct(problem, field_scale=scale, method=method)
            return pressure_drop(solution.pressure, -4.0, 2.0), solution

        evaluate = jax.jit(jax.value_and_grad(drop, has_aux=True))
        (value, solution), gradient = evaluate(1.0)
        assert int(solution.status) == CONVERGED
        assert float(solution.residual_norm) <= 1e-9 * float(solution.initial_residual_norm)
        assert float(jnp.max(jnp.abs(station_flow_rates(solution.velocity) - 20.0))) < 1e-12 * 20.0
        assert float(mass_balance(solution.velocity)) < 1e-12
        assert float(charge_balance(solution, problem)) < 1e-12
        stokes = 413.5619471240624  # the Stokes-limit drop of the same duct
        assert 1e-3 < abs(float(value) - stokes) / stokes < 0.2
        # The root of the Newton solve of `main` before H3 (same operator, residual 1e-12): the plan's 1e-6 gate.
        assert float(value) == pytest.approx(420.27523399266335, rel=1e-6)
        assert float(gradient) == pytest.approx(331.1104370772331, rel=1e-6)
        found[method] = (float(value), float(gradient))
    assert found["newton"] == pytest.approx(found["march"], rel=1e-6)
    # Newton's root is differentiated by the march's rule; its Taylor and central-difference checks are the march's.
    h2 = [float(evaluate(1.0 + h)[0][0]) for h in (0.0, 0.04, 0.02, 0.01)]
    errors = [abs(h2[i] - h2[0] - h * found["march"][1]) for i, h in ((1, 0.04), (2, 0.02), (3, 0.01))]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(2.0, abs=0.15)
    assert np.log2(errors[1] / errors[2]) == pytest.approx(2.0, abs=0.15)
    step = 1e-4
    central = (float(evaluate(1.0 + step)[0][0]) - float(evaluate(1.0 - step)[0][0])) / (2.0 * step)
    assert found["march"][1] == pytest.approx(central, rel=1e-6)


def test_the_rk3_step_leaves_a_stokes_flow_of_an_open_duct_where_it_is_at_any_step_and_field_scale():
    """The stage on an inflow-outflow axis: the lift, the mixed pressure solve and the scaled shifts keep the fixed point."""
    problem = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0)
    factorization, viscous = problem.factorization(), problem.viscous_factorizations(RK3)
    lift = with_inflow(_lift(problem), problem)
    for scale, dt in ((1.0, 0.05), (1.0, 3.0), (1.3, 0.4)):
        state = solve_open_duct(problem, field_scale=scale).velocity
        moved, _, _, (iterations, residual) = _step(
            state, problem, factorization, viscous, RK3, None, 0.0, None, (dt, scale, lift)
        )
        size = max(float(jnp.max(jnp.abs(f.data))) for f in state)
        change = max(float(jnp.max(jnp.abs(a.data - b.data))) for a, b in zip(moved, state, strict=True))
        assert change < 1e-9 * size and float(residual) < 1e-12 and 3 <= int(iterations) < 600
        assert float(mass_balance(moved)) < 1e-11 * size


def test_a_march_that_has_not_converged_is_classified_stagnated_or_unsteady():
    """H3.4: a rotation (a limit cycle) and a growing perturbation have no steady state; a slow decay is only stagnant."""
    start = jnp.array([1.0, 0.0, 0.5])
    turn = jnp.array([[jnp.cos(0.3), -jnp.sin(0.3), 0.0], [jnp.sin(0.3), jnp.cos(0.3), 0.0], [0.0, 0.0, 1.0]])
    for chunk, status in (
        (lambda x: turn @ x, UNSTEADY),
        (lambda x: 1.2 * x, UNSTEADY),
        (lambda x: x * (1.0 - 1e-3), STAGNATED),
        (lambda x: x + jnp.array([1e-3, 0.0, 0.0]), STAGNATED),
    ):
        assert int(_watch(chunk, lambda x, chunk=chunk: jnp.linalg.norm(chunk(x) - x), start)) == status


def test_a_step_of_an_open_duct_at_a_traced_field_scale_and_step_passes_the_derivative_standard():
    """H.10.6 on the dynamic step: tangent and pullback pair, and the Taylor remainder in the field scale falls as h^2."""
    problem = _small_fringe(spacing=1.0, upstream=2.0, downstream=2.0, flow_rate=20.0, advection="central")
    factorization, viscous = problem.factorization(), problem.viscous_factorizations()
    lift = with_inflow(_lift(problem), problem)
    state = solve_open_duct(problem).velocity

    @jax.jit
    def moved(scale):
        found = _step(state, problem, factorization, viscous, EULER, None, 0.0, None, (0.01, scale, lift))[0]
        return jnp.concatenate([f.data.ravel() for f in found])

    value, tangent = jax.jvp(moved, (1.0,), (1.0,))
    cotangent = jnp.asarray(np.random.default_rng(1).standard_normal(value.shape))
    assert float(jax.vjp(moved, 1.0)[1](cotangent)[0]) == pytest.approx(float(cotangent @ tangent), rel=1e-8)
    errors = [float(jnp.linalg.norm(moved(1.0 + h) - value - h * tangent)) for h in (0.04, 0.02, 0.01)]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(2.0, abs=0.1)
    assert np.log2(errors[1] / errors[2]) == pytest.approx(2.0, abs=0.1)
