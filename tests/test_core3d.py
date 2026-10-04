"""The projection step: incompressibility, the damping treatment, and duct flow.

The strongest check here is the last one. A periodic duct driven by a constant
body force reaches the fully developed state, which LMhdX already solves by an
entirely different route in :func:`lmhdx.solve_fully_developed_fields`: a
two-dimensional cross-section solve on `lmhdx.mesh`, with its own operators and its
own linear algebra. Agreement between the two exercises the new staggered core,
its electric coupling and its projection at once, against code that shares none
of them.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.bc import NEUMANN, PERIODIC, BoundaryCondition
from lmhdx.cases import make_hartmann_case, solve_fully_developed_fields
from lmhdx.core3d import (
    ChannelProblem,
    duct_problem,
    enforce_face_constraints,
    project,
    step,
    velocity_condition,
    velocity_offset,
    zero_velocity,
)
from lmhdx.grid import CENTER, Field, Grid, tanh_faces, uniform_faces, wall_resolving_faces
from lmhdx.ops import divergence
from lmhdx.timeloop import advance
from validation.shercliff import flow_rate

# Physics validation rather than unit checks: the channel cases integrate to a
# steady state, which the tier system runs in the regression lane.
pytestmark = pytest.mark.numerical

PERIODIC_X = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _duct(ny: int = 12, nz: int = 12, nx: int = 4) -> Grid:
    return Grid(uniform_faces(nx, 0.0, 1.0), uniform_faces(ny, -1.0, 1.0), uniform_faces(nz, -1.0, 1.0))


def _problem(grid: Grid, **overrides) -> ChannelProblem:
    settings = dict(
        grid=grid,
        conditions=(PERIODIC_X, WALL, WALL),
        density=1.0,
        viscosity=1.0,
        conductivity=1.0,
        magnetic_field=(0.0, 0.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    settings.update(overrides)
    return ChannelProblem(**settings)


def _advance(problem: ChannelProblem, steps: int) -> tuple[Field, Field, Field]:
    factorization = problem.factorization()
    velocity = zero_velocity(problem)
    for _ in range(steps):
        velocity, _, _ = step(velocity, problem, factorization)
    return velocity


def test_velocity_positions_and_conditions_follow_the_layout():
    assert velocity_offset(0) == (0.0, CENTER, CENTER)
    assert velocity_offset(2) == (CENTER, CENTER, 0.0)
    conditions = (PERIODIC_X, WALL, WALL)
    assert velocity_condition(conditions, 0).is_periodic
    assert velocity_condition(conditions, 1).kind == "dirichlet"


def test_face_constraints_close_the_boundary_flux():
    grid = _duct(ny=6, nz=6)
    problem = _problem(grid)
    key = jax.random.PRNGKey(0)
    velocity = tuple(
        Field(
            jax.random.normal(fold, grid.offset_shape(velocity_offset(component)), dtype=jnp.float64),
            velocity_offset(component),
            grid,
        )
        for component, fold in enumerate(jax.random.split(key, 3))
    )
    constrained = enforce_face_constraints(velocity, problem)
    # The periodic axis must agree across its duplicated face, and the walls
    # must be impermeable; together these make the net boundary flux vanish.
    assert np.allclose(np.asarray(constrained[0].data)[0], np.asarray(constrained[0].data)[-1], atol=0.0)
    assert np.all(np.asarray(constrained[1].data)[:, 0, :] == 0.0)
    assert np.all(np.asarray(constrained[2].data)[:, :, -1] == 0.0)
    volumes = grid.cell_volumes()
    net = float(np.sum(volumes * np.asarray(divergence(constrained).data)))
    assert abs(net) < 1e-12


def test_projection_removes_the_divergence_and_is_idempotent():
    grid = _duct(ny=8, nz=8)
    problem = _problem(grid)
    key = jax.random.PRNGKey(1)
    velocity = tuple(
        Field(
            jax.random.normal(fold, grid.offset_shape(velocity_offset(component)), dtype=jnp.float64),
            velocity_offset(component),
            grid,
        )
        for component, fold in enumerate(jax.random.split(key, 3))
    )
    projected, _ = project(velocity, problem)
    scale = float(np.max(np.abs(np.asarray(divergence(enforce_face_constraints(velocity, problem)).data))))
    assert np.max(np.abs(np.asarray(divergence(projected).data))) < 1e-11 * scale

    twice, _ = project(projected, problem)
    for first, second in zip(projected, twice, strict=True):
        assert np.max(np.abs(np.asarray(first.data) - np.asarray(second.data))) < 1e-11


def test_the_step_keeps_the_velocity_divergence_free():
    problem = _problem(_duct(ny=8, nz=8))
    residual = np.asarray(divergence(_advance(problem, 20)).data)
    assert np.max(np.abs(residual)) < 1e-11


def test_a_driven_duct_develops_a_symmetric_forward_profile():
    problem = _problem(_duct(ny=10, nz=10))
    axial = np.asarray(_advance(problem, 40)[0].data)
    assert np.all(axial >= -1e-12)
    assert np.max(np.abs(axial[:, ::-1, :] - axial)) < 1e-11
    assert np.max(np.abs(axial[:, :, ::-1] - axial)) < 1e-11
    # The profile peaks in the middle and vanishes towards the walls.
    middle = axial[0, axial.shape[1] // 2, axial.shape[2] // 2]
    assert middle > axial[0, 0, axial.shape[2] // 2]


def test_damping_rates_follow_the_field_direction():
    problem = _problem(_duct(ny=4, nz=4), magnetic_field=(0.0, 3.0, 0.0), conductivity=2.0, density=4.0)
    rates = problem.damping_rates
    # A field along y damps the two transverse components and leaves its own free.
    assert rates[1] == pytest.approx(0.0)
    assert rates[0] == pytest.approx(2.0 * 9.0 / 4.0)
    assert rates[2] == pytest.approx(2.0 * 9.0 / 4.0)


def test_implicit_damping_stays_stable_past_the_explicit_limit():
    """The damping rate here is far beyond what an explicit step could carry."""
    problem = _problem(_duct(ny=8, nz=8), magnetic_field=(0.0, 30.0, 0.0), dt=2.0e-3)
    assert problem.dt * problem.damping_rates[0] > 1.0, "the test needs a step past the explicit limit"
    axial = np.asarray(_advance(problem, 40)[0].data)
    assert np.all(np.isfinite(axial))
    assert np.max(np.abs(axial)) < 10.0


def test_a_transverse_field_slows_the_flow_it_is_driven_through():
    """Magnetic drag: the same drive reaches a smaller throughput at higher Ha."""
    grid = _duct(ny=10, nz=10)
    throughputs = []
    for strength in (0.0, 3.0, 10.0):
        problem = _problem(grid, magnetic_field=(0.0, strength, 0.0), dt=2.0e-3)
        throughputs.append(float(np.mean(np.asarray(_advance(problem, 60)[0].data))))
    assert throughputs[0] > throughputs[1] > throughputs[2] > 0.0


def test_the_diffusive_step_limit_is_reported_and_binding():
    """Diffusion is explicit, so the limit is real; the field imposes none."""
    problem = _problem(_duct(ny=12, nz=12, nx=3), viscosity=1.0)
    limit = problem.diffusive_step_limit
    assert limit == pytest.approx(1.0 / (2.0 * (9.0 + 36.0 + 36.0)), rel=1e-12)
    # Halving the viscosity doubles the allowed step; the magnetic field, being
    # treated implicitly, does not enter at all.
    assert _problem(_duct(ny=12, nz=12, nx=3), viscosity=0.5).diffusive_step_limit == pytest.approx(
        2.0 * limit, rel=1e-12
    )
    strong = _problem(_duct(ny=12, nz=12, nx=3), magnetic_field=(0.0, 50.0, 0.0))
    assert strong.diffusive_step_limit == pytest.approx(limit, rel=1e-12)


def _channel(cells: int) -> Grid:
    """A plane channel: periodic along x and z, walls at y = +/- 1."""
    return Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(2, 0.0, 1.0))


def _poiseuille_error(cells: int, *, drive: float = 1.0, viscosity: float = 1.0) -> float:
    """Return the relative L2 error against the analytic plane-channel profile."""
    grid = _channel(cells)
    base = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, PERIODIC_X),
        viscosity=viscosity,
        forcing=(drive, 0.0, 0.0),
    )
    problem = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, PERIODIC_X),
        viscosity=viscosity,
        forcing=(drive, 0.0, 0.0),
        dt=0.4 * base.diffusive_step_limit,
    )
    steps = int(round(4.0 / problem.dt))
    computed = np.asarray(_advance(problem, steps)[0].data)[0, :, 0]
    y = np.asarray(grid.centers[1])
    exact = drive / (2.0 * viscosity) * (1.0 - y**2)
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact))


def test_plane_channel_matches_the_analytic_parabola():
    """Steady Stokes flow between plates has the exact profile f(1-y^2)/(2 nu).

    Sixteen cells across the channel give a relative L2 error near 5e-3, which
    is what a second-order scheme gives on a mesh this coarse; the rate itself
    is checked by the refinement test below.
    """
    assert _poiseuille_error(16) < 8.0e-3


def test_the_plane_channel_converges_at_second_order():
    """Refinement must remove the error at the rate the discretization claims."""
    coarse, fine = _poiseuille_error(8), _poiseuille_error(16)
    assert fine < coarse
    order = np.log2(coarse / fine)
    assert 1.7 < order < 2.3, (coarse, fine, order)


def test_a_transverse_field_reduces_the_channel_throughput():
    """The same drive moves less fluid once the field is switched on."""
    grid = _channel(12)
    throughputs = []
    for strength in (0.0, 4.0):
        base = ChannelProblem(
            grid=grid,
            conditions=(PERIODIC_X, WALL, PERIODIC_X),
            magnetic_field=(0.0, strength, 0.0),
            forcing=(1.0, 0.0, 0.0),
        )
        problem = ChannelProblem(
            grid=grid,
            conditions=(PERIODIC_X, WALL, PERIODIC_X),
            magnetic_field=(0.0, strength, 0.0),
            forcing=(1.0, 0.0, 0.0),
            dt=0.4 * base.diffusive_step_limit,
        )
        steps = int(round(2.0 / problem.dt))
        throughputs.append(float(np.mean(np.asarray(_advance(problem, steps)[0].data))))
    assert throughputs[0] > throughputs[1] > 0.0


def test_channel_problem_validates_its_inputs():
    grid = _duct(ny=4, nz=4)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        ChannelProblem(grid=grid, conditions=(WALL, WALL))
    with pytest.raises(ValueError, match="viscosity must be positive"):
        ChannelProblem(grid=grid, conditions=(WALL,) * 3, viscosity=0.0)
    with pytest.raises(ValueError, match="conductivity must not be negative"):
        ChannelProblem(grid=grid, conditions=(WALL,) * 3, conductivity=-1.0)


def test_a_constant_field_given_as_arrays_is_the_uniform_field_bit_for_bit():
    """Plan step 1.9a: arrays take the same arithmetic as three numbers, down to the implicit shifts."""
    grid = Grid(uniform_faces(3, 0.0, 1.0), tanh_faces(10, -1.0, 1.0, 1.5), tanh_faces(8, -1.0, 1.0, 1.3))
    values = (0.3, 20.0, -1.7)
    uniform = _problem(grid, magnetic_field=values, forcing=(1.0, 0.2, 0.0), dt=1.0e-2)
    arrays = dataclasses.replace(
        uniform, magnetic_field=tuple(np.full(grid.shape, value) for value in values)
    )
    assert arrays.damping_rates == uniform.damping_rates
    assert arrays.peak_field_squared == uniform.peak_field_squared
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    velocity = tuple(
        Field(
            jax.random.normal(keys[axis], grid.offset_shape(velocity_offset(axis))),
            velocity_offset(axis),
            grid,
        )
        for axis in range(3)
    )
    runs = [
        step(velocity, problem, problem.factorization(), problem.viscous_factorizations())
        for problem in (uniform, arrays)
    ]
    for first, second in zip(*(jax.tree.leaves(run) for run in runs), strict=True):
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_a_uniform_field_and_conductivity_are_broadcast_not_captured():
    """Under tracing both are broadcast scalars: captured, they were six cell-sized constants per step."""
    from lmhdx.core3d import _attribute_value, _constant, _imposed_field
    from lmhdx.em import face_conductivity, face_electromotive_force

    problem = _problem(_duct(), magnetic_field=(0.0, 20.0, 0.0))
    scalar = problem.scalar_conditions

    def electric(velocity):
        conductivity = _constant(problem, _attribute_value, "conductivity")
        return [
            (face_electromotive_force(velocity, _imposed_field(problem), axis, scalar).data, conductivity)
            for axis in range(3)
        ] + [face_conductivity(conductivity, axis, scalar[axis]).data for axis in range(3)]

    captured = jax.make_jaxpr(electric)(zero_velocity(problem)).consts
    assert all(np.size(value) < np.prod(problem.grid.shape) for value in captured)


def test_a_varying_field_is_validated_and_keeps_the_problem_static():
    from lmhdx.core3d import ImposedField

    grid = _duct(ny=4, nz=4)
    along = np.linspace(1.0, 2.0, grid.shape[0])[:, None, None] * np.ones(grid.shape)
    problem = _problem(grid, magnetic_field=(0.5, along, 0.0), conductivity=2.0, density=4.0)
    assert isinstance(problem.magnetic_field, ImposedField)
    twin = _problem(grid, magnetic_field=(0.5, along.copy(), 0.0), conductivity=2.0, density=4.0)
    assert problem == twin and hash(problem) == hash(twin)
    assert problem != dataclasses.replace(twin, magnetic_field=(0.5, 2.0 * along, 0.0))
    with pytest.raises(ValueError, match="read-only"):
        problem.magnetic_field.components[1][0, 0, 0] = 0.0
    # One shift per component: the largest rate over the cells, sigma (|B|^2 - B_c^2) / rho.
    assert problem.damping_rates == pytest.approx((2.0 * 4.0 / 4.0, 2.0 * 0.25 / 4.0, 2.0 * 4.25 / 4.0))
    assert problem.peak_field_squared == pytest.approx(4.25)
    for field, message in (
        ((0.0, along[:, :2], 0.0), "do not match"),
        ((0.0, np.where(along > 1.5, np.nan, along), 0.0), "must be finite"),
        ((0.0, np.inf, 0.0), "must be finite"),
        ((1.0, 2.0), "three magnetic field components"),
    ):
        with pytest.raises(ValueError, match=message):
            _problem(grid, magnetic_field=field)
    with pytest.raises(ValueError, match="different grid"):
        _problem(_duct(ny=6, nz=6), magnetic_field=problem.magnetic_field)


def test_the_step_differentiates_and_jits():
    grid = _duct(ny=6, nz=6, nx=2)
    base = _problem(grid, dt=5.0e-3)
    factorization = base.factorization()

    def kinetic(drive):
        problem = _problem(grid, forcing=(drive, 0.0, 0.0), dt=5.0e-3)
        velocity = zero_velocity(problem)
        for _ in range(3):
            velocity, _, _ = step(velocity, problem, factorization)
        return jnp.sum(velocity[0].data ** 2)

    gradient = float(jax.jit(jax.grad(kinetic))(1.0))
    size = 1.0e-5
    difference = (kinetic(1.0 + size) - kinetic(1.0 - size)) / (2.0 * size)
    assert gradient == pytest.approx(float(difference), rel=1e-6)


def _implicit_channel_error(cells: int, *, step_multiple: float) -> tuple[float, float]:
    """Integrate the plane channel with implicit viscosity at a chosen step size.

    ``step_multiple`` is the step as a multiple of the explicit stability limit,
    so a value above one is a step no explicit scheme could take.
    """
    grid = _channel(cells)
    conditions = (PERIODIC_X, WALL, PERIODIC_X)
    base = ChannelProblem(grid=grid, conditions=conditions, forcing=(1.0, 0.0, 0.0))
    problem = ChannelProblem(
        grid=grid,
        conditions=conditions,
        forcing=(1.0, 0.0, 0.0),
        dt=step_multiple * base.diffusive_step_limit,
    )
    factorization, viscous = problem.factorization(), problem.viscous_factorizations()
    velocity = zero_velocity(problem)
    for _ in range(int(round(4.0 / problem.dt))):
        velocity, _, _ = step(velocity, problem, factorization, viscous)
    computed = np.asarray(velocity[0].data)[0, :, 0]
    y = np.asarray(grid.centers[1])
    exact = 0.5 * (1.0 - y**2)
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact)), problem.dt


def test_implicit_viscosity_is_stable_far_past_the_explicit_limit():
    """Twenty times the explicit limit: an explicit step would diverge here."""
    error, used = _implicit_channel_error(16, step_multiple=20.0)
    reference = ChannelProblem(
        grid=_channel(16), conditions=(PERIODIC_X, WALL, PERIODIC_X), forcing=(1.0, 0.0, 0.0)
    )
    assert used > 10.0 * reference.diffusive_step_limit
    assert np.isfinite(error) and error < 1.0e-2


def test_implicit_and_explicit_viscosity_reach_the_same_steady_state():
    """The treatment changes the path, not the state the path ends at."""
    explicit = _poiseuille_error(16)
    implicit, _ = _implicit_channel_error(16, step_multiple=0.4)
    assert implicit == pytest.approx(explicit, rel=2.0e-2)


def test_implicit_viscosity_keeps_second_order_convergence():
    coarse, _ = _implicit_channel_error(8, step_multiple=5.0)
    fine, _ = _implicit_channel_error(16, step_multiple=5.0)
    order = np.log2(coarse / fine)
    assert 1.7 < order < 2.3, (coarse, fine, order)


def _manufactured_velocity(point):
    """A divergence-free, no-slip 3-D field: the curl of ``(chi, 0, psi)`` plus an axial flow.

    ``g = (1 - s^2)^2`` and its derivative vanish on the walls ``y, z = +-1``, so every
    component does; ``x`` is periodic. Written once, pointwise, and differentiated by JAX,
    so the source shares nothing with the discrete operators.
    """
    x, y, z = point
    wave = 2.0 * jnp.pi

    def g(s):
        return (1.0 - s**2) ** 2

    psi = jax.grad(lambda *p: g(p[1]) * g(p[2]) * jnp.sin(wave * p[0]), argnums=(0, 1, 2))(x, y, z)
    chi = jax.grad(lambda *p: 0.5 * g(p[1]) * g(p[2]) * jnp.cos(wave * p[0]), argnums=(0, 1, 2))(x, y, z)
    return jnp.stack([psi[1] + g(y) * g(z), chi[2] - psi[0], -chi[1]])


def _manufactured_error(cells: int, advection: str, viscosity: float = 0.2) -> float:
    """March the full step to its steady state under the manufactured source; return the max error."""

    def source(point):
        # Steady momentum with zero pressure: (u . grad) u - nu lap u.
        velocity = _manufactured_velocity(point)
        gradient = jax.jacfwd(_manufactured_velocity)(point)
        laplacian = jnp.trace(jax.hessian(_manufactured_velocity)(point), axis1=1, axis2=2)
        return gradient @ velocity - viscosity * laplacian

    grid = Grid(
        uniform_faces(cells, 0.0, 1.0), tanh_faces(cells, -1.0, 1.0, 1.5), tanh_faces(cells, -1.0, 1.0, 1.2)
    )

    def sample(function, component):
        axes = [grid.faces[a] if a == component else grid.centers[a] for a in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = jnp.stack([jnp.asarray(c.ravel()) for c in mesh], axis=1)
        return jax.vmap(function)(points)[:, component].reshape(mesh[0].shape)

    # `step` adds the forcing pointwise, so a face array per component is a body force.
    problem = _problem(
        grid,
        viscosity=viscosity,
        conductivity=0.0,
        forcing=tuple(sample(source, component) for component in range(3)),
        dt=0.05,
        advection=advection,
    )
    factorization, viscous = problem.factorization(), problem.viscous_factorizations()

    @jax.jit
    def march(velocity):
        return jax.lax.fori_loop(0, 1500, lambda _, v: step(v, problem, factorization, viscous)[0], velocity)

    velocity = march(zero_velocity(problem))
    again = step(velocity, problem, factorization, viscous)[0]
    assert max(float(jnp.max(jnp.abs(a.data - b.data))) for a, b in zip(again, velocity, strict=True)) < 1e-12
    return max(
        float(jnp.max(jnp.abs(field.data - sample(_manufactured_velocity, component))))
        for component, field in enumerate(velocity)
    )


@pytest.mark.parametrize("advection", ["central", "limited"])
def test_a_manufactured_flow_converges_at_second_order_with_convection_on(advection):
    """Plan step 1.5b, validation row 8: the full step, convection on, a stretched 3-D mesh.

    Both wall-normal axes are tanh-stretched and every velocity component varies in all
    three directions; the Reynolds number is about 5, so convection is not a perturbation
    (switched off, the error is 4.3 and does not fall). Measured: central 0.246 -> 0.0625
    (order 1.98), limited 0.183 -> 0.0444 (order 2.04) from 12^3 to 24^3.
    """
    coarse, fine = _manufactured_error(12, advection), _manufactured_error(24, advection)
    order = np.log2(coarse / fine)
    assert 1.8 < order < 2.3, (coarse, fine, order)


# --- Reconciliation with the production solver and an independent reference ---
#
# `lmhdx.solve_fully_developed_fields` solves the same duct on a two-dimensional
# cross-section mesh with its own operators. Its conventions had to be matched
# before the two could be compared at all: `GeometrySpec.width` and `height` are
# the *full* transverse extents, so a `width=height=2` case is the grid
# `[-1, 1]^2` used here; `CaseSpec.forcing` is the axial pressure gradient
# `-dp/dx` and enters the momentum equation divided by the density, exactly as
# `ChannelProblem.forcing` does; and both read `viscosity` as the kinematic one.
#
# `validation.shercliff` closes the loop from outside the package: a Chebyshev
# collocation solve of the governing system, converged to eight digits, sharing
# no operator with either route.


def _duct_mean_velocity(cells: int, hartmann: float, *, resolve_layers: bool = False) -> float:
    """Return the volume-averaged axial velocity of a steady insulating duct."""
    if resolve_layers:
        transverse = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=6, max_ratio=1.35
        )
        spanwise = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(hartmann), cells_in_layer=6, max_ratio=1.35
        )
    else:
        transverse = spanwise = uniform_faces(cells, -1.0, 1.0)
    grid = Grid(uniform_faces(1, 0.0, 1.0), transverse, spanwise)
    problem = ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_X, WALL, WALL),
        conductivity=1.0 if hartmann else 0.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=0.02,
    )
    velocity = advance(
        problem,
        400,
        factorization=problem.factorization(),
        viscous=problem.viscous_factorizations(),
    ).velocity
    volumes = np.asarray(grid.cell_volumes())[0]
    return float((np.asarray(velocity[0].data)[0] * volumes).sum() / volumes.sum())


def _production_duct(cells: int, hartmann: float) -> np.ndarray:
    """Solve the same duct through the production fully developed route."""
    case = make_hartmann_case(ha=hartmann, width=2.0, height=2.0, ny=cells, nz=cells)
    uniform = dataclasses.replace(case.geometry, target_ha=None)
    return np.asarray(solve_fully_developed_fields(dataclasses.replace(case, geometry=uniform))[0])


def test_the_hydrodynamic_duct_reconciles_with_the_production_solver():
    """Without a field the two routes agree to well inside their shared truncation error."""
    cells = 32
    production = _production_duct(cells, 0.0)
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
    problem = ChannelProblem(
        grid=grid, conditions=(PERIODIC_X, WALL, WALL), conductivity=0.0, forcing=(1.0, 0.0, 0.0), dt=0.02
    )
    velocity = advance(
        problem, 400, factorization=problem.factorization(), viscous=problem.viscous_factorizations()
    ).velocity
    new = np.asarray(velocity[0].data)[0]
    assert np.linalg.norm(new - production) / np.linalg.norm(production) < 5e-3
    # Both sit within their own discretisation error of the independent reference.
    exact = flow_rate(0.0, 40)
    for mean in (float(new.mean()), float(production.mean())):
        assert abs(mean - exact) / exact < 5e-3


def test_the_duct_converges_to_the_spectral_reference_with_a_field():
    """Second order in the mean velocity, against a reference that shares no code."""
    exact = flow_rate(5.0, 40)
    coarse = abs(_duct_mean_velocity(16, 5.0) - exact) / exact
    fine = abs(_duct_mean_velocity(32, 5.0) - exact) / exact
    assert fine < 0.02
    assert np.log2(coarse / fine) > 1.8


def test_a_wall_resolving_mesh_reaches_the_reference_flow_rate_at_hartmann_20():
    """The layers carry the physics, so the mesh has to resolve them rather than be fine."""
    exact = flow_rate(20.0, 40)
    resolved = _duct_mean_velocity(32, 20.0, resolve_layers=True)
    assert abs(resolved - exact) / exact < 0.02
    # The same cell count spread uniformly cannot resolve the a/Ha layer.
    assert abs(_duct_mean_velocity(32, 20.0) - exact) / exact > 0.1


def test_the_duct_helper_resolves_the_layers_it_names():
    """The mesh follows the physics: a/Ha against the field, a/sqrt(Ha) across it."""
    problem = duct_problem(hartmann=100.0, cells=40)
    transverse, spanwise = (np.diff(problem.grid.faces[axis]) for axis in (1, 2))
    assert transverse.min() < 1.0 / 100.0
    assert spanwise.min() < 1.0 / np.sqrt(100.0)
    # The gentlest stretching that spans the duct, not the finest one available.
    assert transverse.max() / transverse.min() < spanwise.max() / spanwise.min() * 1.0e3
    assert problem.wall_conductance == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="hartmann must not be negative"):
        duct_problem(hartmann=-1.0)


def test_the_public_solve_reaches_the_new_core():
    """`lmhdx.solve` dispatches a ChannelProblem to the certified steady solve of :mod:`lmhdx.steady`.

    The certificate is relative: the final residual is within ten times the
    tolerance of the residual at rest. An insulating Stokes duct is one CG solve,
    which stops at the tolerance rather than overshooting it as Newton does, so
    an absolute bound would test the route instead of the contract.
    """
    import lmhdx
    from lmhdx.core3d import zero_velocity
    from lmhdx.steady import steady_residual

    problem = lmhdx.duct_problem(hartmann=5.0, cells=16)
    solution = lmhdx.solve(problem)
    at_rest = steady_residual(zero_velocity(problem), problem)
    scale = max(float(np.sqrt(sum(np.sum(np.asarray(field.data) ** 2) for field in at_rest))), 1.0)
    assert float(np.mean(np.asarray(solution.velocity[0].data))) > 0.0
    assert float(solution.residual_norm) <= 10.0 * 1.0e-9 * scale


# Thin conducting walls (plan step 1.3b): each wall is a sheet with its own potential, reached across
# the half cell and conducting along itself (Walker's condition), solved directly as separable nodes.


def _mean_free(problem: ChannelProblem, seed: int) -> Field:
    volumes = jnp.asarray(problem.grid.cell_volumes())
    data = jax.random.normal(jax.random.PRNGKey(seed), problem.grid.shape, dtype=jnp.float64)
    return Field(data - jnp.sum(volumes * data) / jnp.sum(volumes), (CENTER,) * 3, problem.grid)


def _charge(problem: ChannelProblem, potential: Field, walls):
    """Charge balance at rest and sheet residuals; ``walls=None`` is the replaced first-order closure."""
    from lmhdx.em import face_conductivity, face_current, thin_wall_current, thin_wall_flux, wall_insulated

    scalar = problem.scalar_conditions
    sigma = Field(jnp.full(problem.grid.shape, float(problem.conductivity)), (CENTER,) * 3, problem.grid)
    currents, sheets = [], []
    for axis in range(3):
        conductivity = face_conductivity(sigma, axis, scalar[axis])
        rest = conductivity.replace_data(jnp.zeros_like(conductivity.data))
        current = wall_insulated(
            face_current(potential, conductivity, rest, axis, scalar[axis]), axis, scalar[axis]
        )
        conduction = float(problem.wall_conductance[axis] * problem.conductivity)
        if conduction and walls is None:
            layer = rest.replace_data(rest.data.at[:, [0, -1]].set(potential.data[:, [0, -1]]))
            current = current.replace_data(
                current.data + thin_wall_flux(layer, axis, scalar[axis], conduction, scalar).data
            )
        elif conduction:
            inflow = thin_wall_current(potential, walls[axis], conductivity, axis)
            carried = thin_wall_flux(walls[axis], axis, scalar[axis], conduction, scalar)
            sheets.append(float(jnp.max(jnp.abs(inflow.data - carried.data))))
            current = current.replace_data(current.data + inflow.data)
        currents.append(current)
    return divergence(tuple(currents)), sheets


@pytest.mark.parametrize("stretch", [None, 1.6])
def test_the_wall_potential_converges_at_second_order(stretch):
    """``phi = (y^2 + a) cos(pi z)``, ``a = -1 - 2/(c pi^2)``, meets Walker's ``d_n phi = c d_tt phi`` exactly.

    The adjacent cell value as the wall potential was first order; the sheet unknown is second order.
    """
    shift = -1.0 - 2.0 / (0.1 * np.pi**2)
    errors = []
    for cells in (8, 16, 32):
        faces = uniform_faces(cells, -1.0, 1.0) if stretch is None else tanh_faces(cells, -1.0, 1.0, stretch)
        problem = _problem(Grid(uniform_faces(1, 0.0, 1.0), faces, faces), wall_conductance=(0.0, 0.1, 0.0))
        y, z = problem.grid.centers[1][None, :, None], np.cos(np.pi * problem.grid.centers[2][None, None, :])
        source = Field(jnp.asarray((2.0 - np.pi**2 * (y**2 + shift)) * z), (CENTER,) * 3, problem.grid)
        potential, walls = problem.potential_factorization().solve_with_walls(source)
        exact, volumes = (y**2 + shift) * z, problem.grid.cell_volumes()
        gauge = np.sum(volumes * exact) / np.sum(volumes)
        sheet = np.asarray(walls[1].data)[:, [0, -1]] - ((1.0 + shift) * z - gauge)
        errors.append([np.max(np.abs(sheet)), np.max(np.abs(np.asarray(potential.data) - exact + gauge))])
    # The bound of the other order gates, over the last two levels (tanh wall potential: 1.49, then 1.90).
    assert np.all(np.log2(np.divide(errors[1], errors[2])) > 1.8), errors


def test_a_thin_wall_conserves_charge_and_keeps_the_solve_symmetric():
    """Per cell and per sheet element, with a solve map self-adjoint in the cell volumes."""
    from lmhdx.core3d import _solve_potential
    from lmhdx.ops import cell_inner_product

    problem = duct_problem(hartmann=20.0, cells=24, wall_conductance=0.05)
    factorization = problem.factorization()
    source, first, second = (_mean_free(problem, seed) for seed in (1, 2, 3))

    def solve(rhs, case=problem):
        return _solve_potential(rhs, case, factorization)

    potential, walls = solve(source)
    balance, sheets = _charge(problem, potential, walls)
    scale = float(jnp.max(jnp.abs(source.data)))
    assert max(float(jnp.max(jnp.abs(balance.data + source.data))), *sheets) < 1e-10 * scale
    forward, backward = (
        float(cell_inner_product(a, solve(b)[0])) for a, b in ((first, second), (second, first))
    )
    assert abs(forward - backward) <= 1e-12 * abs(forward)
    # Mixed precision solves the same sheets, and a vanishing conductance approaches the insulating wall.
    mixed = solve(source, dataclasses.replace(problem, precision="mixed"))[0].data
    assert float(jnp.max(jnp.abs(mixed - potential.data))) < 1e-10 * float(jnp.max(jnp.abs(potential.data)))
    insulating = duct_problem(hartmann=20.0, cells=24).factorization().solve(source).data
    faint = solve(source, dataclasses.replace(problem, wall_conductance=(0.0, 1.0e-8, 0.0)))[0].data
    assert float(jnp.max(jnp.abs(faint - insulating))) < 1e-6 * float(jnp.max(jnp.abs(insulating)))


def test_two_conducting_walls_meet_in_a_charge_conserving_corner():
    """What one sheet delivers to the corner the other receives, and the corner edge conducts nothing."""
    from lmhdx.core3d import _solve_potential

    faces = (uniform_faces(5, 0.0, 1.0), tanh_faces(12, -1.0, 1.0, 1.3), tanh_faces(10, -1.0, 1.0, 1.3))
    problem = _problem(Grid(*faces), wall_conductance=(0.0, 0.05, 0.08), conductivity=2.0)
    source = _mean_free(problem, 4)
    potential, walls = _solve_potential(source, problem, problem.factorization())
    balance, _ = _charge(problem, potential, walls)
    assert float(jnp.max(jnp.abs(balance.data + source.data))) < 1e-10 * float(jnp.max(jnp.abs(source.data)))
    factorization = problem.potential_factorization()
    augmented = factorization._corrected(jnp.pad(source.data / 2.0, ((0, 0), (1, 1), (1, 1))))
    inner = augmented[:, 1:-1, 1:-1]
    np.testing.assert_allclose(inner - inner.mean(), potential.data - potential.data.mean(), atol=1e-12)
    assert float(jnp.max(jnp.abs(factorization._apply(augmented)[:, 1:-1, 1:-1] - source.data / 2.0))) < 1e-10
    heights, widths = problem.grid.widths[1:]
    potential = float(jnp.max(jnp.abs(augmented)))
    for row, column in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        corner = augmented[:, row, column]
        along_y = 0.05 * (augmented[:, row, 1 if column == 0 else -2] - corner) / (0.5 * widths[column])
        along_z = 0.08 * (augmented[:, 1 if row == 0 else -2, column] - corner) / (0.5 * heights[row])
        # Each flux is a difference of sheet potentials about 2,000 times larger than it, so
        # round-off scales with the potential over the half widths, not with the flux: measured
        # 1e-16 to 6e-16 of that scale (and 3e-14 to 1.2e-12 of the flux, varying with the platform).
        scale = potential * (0.05 / (0.5 * widths[column]) + 0.08 / (0.5 * heights[row]))
        assert float(jnp.max(jnp.abs(along_y + along_z))) < 1e-14 * scale


def test_the_thin_wall_solve_is_ten_times_faster_than_the_krylov_route():
    """Warm and compiled at Ha 100 on 48^2, against GMRES on the first-order closure it replaced."""
    import time

    import solvax

    from lmhdx.core3d import _solve_potential

    problem = duct_problem(hartmann=100.0, cells=48, wall_conductance=0.05)
    factorization, source = problem.factorization(), _mean_free(problem, 0)
    problem.potential_factorization()

    def first_order(potential):
        balance = _charge(problem, potential, None)[0]
        return balance.replace_data(-balance.data)

    def krylov(data, inverse=factorization.solve):
        rhs = source.replace_data(data)
        return solvax.gmres(first_order, rhs, precond=inverse, rtol=1e-12, max_restarts=20).x.data

    routes = (
        jax.jit(krylov),
        jax.jit(lambda data: _solve_potential(source.replace_data(data), problem, factorization)[0].data),
    )
    times = ([], [])
    for _ in range(10):
        for route, samples in zip(routes, times, strict=True):
            start = time.perf_counter()
            np.asarray(route(source.data))
            samples.append(time.perf_counter() - start)
    # The first call of each route compiles it.
    ratio = np.median(times[0][1:]) / np.median(times[1][1:])
    assert ratio > 10.0, ratio


def test_resolved_wall_layers_are_frozen_and_checked():
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(4, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    conditions = (PERIODIC_X, WALL, WALL)
    problem = ChannelProblem(grid, conditions, wall_layers=(None, ([2, [0.1, 0.1]], None), None))
    assert problem.wall_layers == (None, (((2.0, 2.0), (0.1, 0.1)), None), None) and problem.conducting_walls
    assert hash(problem) == hash(ChannelProblem(grid, conditions, wall_layers=problem.wall_layers))
    assert ChannelProblem(
        grid, conditions, wall_layers=(None, *problem.wall_layers[1:2] * 2)
    ).conducting_walls
    deep = Grid(uniform_faces(2, 0.0, 1.0), grid.faces[1], grid.faces[2])
    with pytest.raises(ValueError, match="one cell along the third"):
        ChannelProblem(deep, conditions, wall_layers=(None, *problem.wall_layers[1:2] * 2))
    for layers in ((((1.0, (0.1,)), None), None, None), (None, ((0.0, (0.1,)), None), None)):
        with pytest.raises(ValueError):
            ChannelProblem(grid, conditions, wall_layers=layers)
    with pytest.raises(ValueError):
        ChannelProblem(grid, conditions, wall_conductance=(0.0, 0.1, 0.0), wall_layers=problem.wall_layers)
