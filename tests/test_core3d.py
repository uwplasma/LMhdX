"""The projection step: incompressibility, the damping treatment, and duct flow.

The strongest check here is the last one. A periodic duct driven by a constant
body force reaches the fully developed state, which
:func:`lmhdx.solve_fully_developed_fields` solves in one CG solve; the time loop
reaching it by projection steps checks the step against the steady solve, and
`validation.shercliff` checks both against code that shares none of them.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.cases import make_hartmann_case
from lmhdx.core3d import (
    ChannelProblem,
    advance,
    duct_problem,
    enforce_face_constraints,
    project,
    step,
    velocity_condition,
    velocity_offset,
    zero_velocity,
)
from lmhdx.fully_developed import solve_fully_developed_fields
from lmhdx.grid import (
    CENTER,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    tanh_faces,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.ops import divergence
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
    from lmhdx.ops import face_conductivity, face_electromotive_force

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


# The step's stages (plan step 5.1a): a caller that composes them, as the thermal step does, advances
# the same flow. Insulating with transport, thin walls on a stretched one-cell periodic duct, resolved layers.


def _stage_cases() -> tuple[tuple[ChannelProblem, bool], ...]:
    stretched = Grid(uniform_faces(1, 0.0, 1.0), tanh_faces(8, -1.0, 1.0, 1.3), tanh_faces(8, -1.0, 1.0, 1.3))
    thin = dict(magnetic_field=(0.0, 0.5, 4.0), wall_conductance=(0.0, 0.05, 0.08), density=1.3)
    layers = (((2.0, 2.0), (0.1, 0.1)),) * 2
    return (
        (_problem(_duct(6, 6, 2), magnetic_field=(0.0, 0.0, 4.0), advection="limited", density=1.3), False),
        (_problem(stretched, **thin), True),
        (_problem(_duct(6, 6, 1), magnetic_field=(0.0, 0.0, 3.0), wall_layers=(None, layers, layers)), True),
    )


def _random_velocity(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    rng = np.random.default_rng(7)
    noise = (f.replace_data(jnp.asarray(rng.standard_normal(f.data.shape))) for f in zero_velocity(problem))
    return enforce_face_constraints(tuple(noise), problem)


def _identical(first, second) -> bool:
    return all(np.array_equal(a.data, b.data) for a, b in zip(first, second, strict=True))


def test_the_step_is_the_force_and_momentum_stages_it_is_built_from():
    from lmhdx.core3d import (
        _advance_momentum,
        _lorentz_from_currents,
        electric_state,
        face_currents,
        face_lorentz_force,
    )

    for problem, implicit in _stage_cases():
        velocity, factorization = _random_velocity(problem), problem.factorization()
        viscous = problem.viscous_factorizations() if implicit else None
        potential, currents, field = face_currents(velocity, problem, factorization)
        force = _lorentz_from_currents(currents, field, problem)
        body = face_lorentz_force(force, problem)
        moved, pressure = _advance_momentum(velocity, body, problem, factorization, viscous)
        stepped, stepped_pressure, stepped_potential = step(velocity, problem, factorization, viscous)
        assert _identical(force, electric_state(velocity, problem, factorization)[1])
        assert _identical((*moved, pressure, potential), (*stepped, stepped_pressure, stepped_potential))


def test_an_extra_force_density_enters_beside_the_forcing():
    """Zero is the step bit for bit; a uniform one is more forcing, divided by the same density."""
    from lmhdx.core3d import _advance_momentum, electric_state, face_lorentz_force

    problem = _stage_cases()[1][0]
    velocity, factorization = _random_velocity(problem), problem.factorization()
    viscous = problem.viscous_factorizations()
    body = face_lorentz_force(electric_state(velocity, problem, factorization)[1], problem)
    extra = (0.4, -0.3, 0.2)

    def advance(case, force=None):
        moved, pressure = _advance_momentum(velocity, body, case, factorization, viscous, extra_force=force)
        return (*moved, pressure)

    zero = tuple(f.replace_data(jnp.zeros_like(f.data)) for f in velocity)
    uniform = tuple(f.replace_data(jnp.full_like(f.data, value)) for f, value in zip(velocity, extra))
    more = dataclasses.replace(problem, forcing=tuple(a + b for a, b in zip(problem.forcing, extra)))
    assert _identical(advance(problem, zero), advance(problem))
    # Only the order of one addition differs, so each array agrees to its own rounding.
    for pushed, forced in zip(advance(problem, uniform), advance(more)):
        error, scale = (float(jnp.max(jnp.abs(data))) for data in (pushed.data - forced.data, forced.data))
        assert error <= 1e-13 * scale


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
# `lmhdx.solve_fully_developed_fields` solves the same duct as one steady CG
# solve of a `CaseSpec`, whose conventions are these: `GeometrySpec.width` and `height` are
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
    return np.asarray(solve_fully_developed_fields(case)[0])


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
    from lmhdx.ops import face_conductivity, face_current, thin_wall_current, thin_wall_flux, wall_insulated

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


# Solid walls in the charge balance (plan step 5.4b, ADR 0009 D36): the potential on the extended grid.


_SYMMETRIC = (((4.0, 4, 0.1), (4.0, 4, 0.1)), ((0.5, 4, 0.1), (0.5, 4, 0.1)))
# Unequal cells and thicknesses per wall, and one end insulated, so no corner depth ties.
_UNEQUAL = (((4.0, 4, 0.1), (4.0, 3, 0.06)), ((0.5, 2, 0.08), None))


def _sections(ends=_SYMMETRIC, cells: int = 12, conductivity: float = 3.0):
    """One periodic cell of a square duct, its walls resolved on both axes as ``wall_layers`` and as the same
    cells of ``solid_walls``: each end ``(ratio, cells, thickness)`` or ``None``, the ratio over the fluid's
    ``conductivity``, which the solid walls take as an absolute value."""
    from lmhdx.core3d import SolidWall

    base = dataclasses.replace(
        duct_problem(hartmann=20.0, cells=cells, cells_in_layer=3), conductivity=conductivity
    )

    def layer(end):
        return None if end is None else ((end[0],) * end[1], (end[2] / end[1],) * end[1])

    def solid(end):
        return None if end is None else SolidWall(end[2], end[1], 1.0, 1.0, end[0] * conductivity)

    layered = dataclasses.replace(base, wall_layers=(None, *(tuple(map(layer, pair)) for pair in ends)))
    return layered, dataclasses.replace(
        base, solid_walls=((None, None), *(tuple(map(solid, pair)) for pair in ends))
    )


def _noise(problem: ChannelProblem, seed: int = 54):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    shapes = [problem.grid.offset_shape(velocity_offset(axis)) for axis in range(3)]
    noise = tuple(
        Field(jax.random.normal(keys[axis], shapes[axis]), velocity_offset(axis), problem.grid)
        for axis in range(3)
    )
    return project(noise, problem)[0]


@pytest.mark.parametrize("ends", [_SYMMETRIC, _UNEQUAL], ids=["symmetric", "unequal"])
def test_solid_walls_carry_the_currents_of_the_same_walls_resolved_in_cells(ends):
    """A 2-D section with Hartmann walls 4x and side walls 0.5x the fluid's conductivity of 3, equal on every end
    or of unequal cells with one end insulated: PCG on the extended grid gives the resolved path's potential,
    currents and powers, corners included (both take the nearer wall's material), and the Lorentz work is minus
    the dissipation in fluid and walls."""
    from lmhdx.core3d import energy_budget, face_currents

    problems = _sections(ends)
    velocity = _noise(problems[0])
    (phi, expected, _), (psi, currents, _) = (face_currents(velocity, problem) for problem in problems)
    volumes = problems[0].grid.cell_volumes()
    centred = [
        np.asarray(p.data) - np.sum(volumes * np.asarray(p.data)) / np.sum(volumes) for p in (phi, psi)
    ]
    assert centred[1] == pytest.approx(centred[0], abs=1e-9 * np.max(np.abs(centred[0])))
    scale = max(float(jnp.max(jnp.abs(current.data))) for current in expected)
    for want, got in zip(expected, currents):
        assert np.asarray(got.data) == pytest.approx(np.asarray(want.data), abs=1e-9 * scale)
    budgets = [energy_budget(velocity, problem) for problem in problems]
    for name in ("lorentz", "joule", "wall"):
        assert float(getattr(budgets[1], name)) == pytest.approx(float(getattr(budgets[0], name)), rel=1e-9)
    work = float(budgets[1].lorentz + budgets[1].joule + budgets[1].wall)
    assert abs(work) <= 1e-9 * float(budgets[1].joule) and float(budgets[1].wall) > 0.0


def test_a_section_in_solid_walls_develops_the_flow_of_its_resolved_walls():
    from lmhdx.fully_developed import channel_flow_rate
    from lmhdx.steady import solve_steady_state

    rates = [
        float(channel_flow_rate(problem, solve_steady_state(problem).velocity[0].data[0]))
        for problem in _sections()
    ]
    assert rates[1] == pytest.approx(rates[0], rel=1e-8)


def test_thinner_solid_walls_tend_to_the_thin_wall_model_at_their_conductance():
    """Hunt's duct at Ha 20: Hartmann walls alone, so no corners, of conductance ratio ``c = sigma_w t / (sigma a)
    = 0.05``, thinned from a/10 to a/40 at that ``c``, approach the thin-wall model's flow rate at second order in
    ``t``: 4.3e-3, 1.1e-3 and 2.8e-4."""
    from lmhdx.core3d import SolidWall
    from lmhdx.fully_developed import channel_flow_rate
    from lmhdx.steady import solve_steady_state

    def rate(problem):
        return float(channel_flow_rate(problem, solve_steady_state(problem).velocity[0].data[0]))

    thin = duct_problem(hartmann=20.0, cells=24, wall_conductance=0.05)
    reference, errors = rate(thin), []
    for thickness in (0.1, 0.05, 0.025):
        wall = SolidWall(thickness, 2, 1.0, 1.0, 0.05 / thickness)
        walls = ((None, None), (wall, wall), (None, None))
        errors.append(
            abs(
                rate(dataclasses.replace(thin, wall_conductance=(0.0,) * 3, solid_walls=walls)) / reference
                - 1
            )
        )
    assert errors == pytest.approx([4.3e-3, 1.1e-3, 2.8e-4], rel=0.05)


def test_the_charge_balance_closes_in_the_walls_and_none_leaves_them():
    """The currents through the fluid and its solid walls, on the extended grid. The charge right-hand side's volume
    mean, which the gauge would remove, is rounding: no real source is absorbed. The divergence vanishes in every
    cell, fluid and wall, to the solve's 1e-13; the walls carry current; and the insulated outer faces carry none."""
    from lmhdx.core3d import SolidWall, _motional, _solid_wall_solve

    wall = SolidWall(0.1, 2, 1.0, 1.0, 2.0)
    problem = ChannelProblem(
        _duct(10, 10, 4),
        (PERIODIC_X, WALL, WALL),
        magnetic_field=(0.0, 10.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=0.01,
        solid_walls=((None, None), (wall, wall), (wall, wall)),
    )
    *_, motional = _motional(_noise(problem), problem)
    conjugate, charge = _solid_wall_solve(
        problem.grid, problem.scalar_conditions, problem.solid_walls, 1.0, problem.precision
    )
    padded = tuple(Field(conjugate.padded(flux.data), flux.offset, conjugate.grid) for flux in motional)
    rhs = -divergence(padded).data
    assert abs(float(charge.defect(rhs))) <= 1e-15 * float(jnp.max(jnp.abs(rhs)))
    potential, _ = charge.solve(rhs)
    currents = tuple(
        flux.replace_data(flux.data - inward.data)
        for flux, inward in zip(padded, charge.face_fluxes(potential))
    )
    scale = max(float(jnp.max(jnp.abs(current.data))) for current in currents)
    assert float(jnp.max(jnp.abs(divergence(currents).data))) <= 1e-10 * scale
    solid = np.ones(conjugate.grid.shape, dtype=bool)
    solid[conjugate.fluid] = False
    axial = np.asarray(currents[0].data)[:-1]  # the periodic axis repeats its first face
    assert float(np.max(np.abs(axial[solid]))) > 1e-2 * scale, "the walls carry no current"
    for axis in (1, 2):
        ends = np.moveaxis(np.asarray(currents[axis].data), axis, 0)[[0, -1]]
        assert float(np.max(np.abs(ends))) <= 1e-12 * scale


def test_a_solid_wall_s_electrical_conductivity_is_a_leaf_of_the_charge_solve():
    """Two solid walls' electrical conductivities, put into the cells in the trace, are leaves of the charge solve:
    the derivative of a weighted sum of the fluid's face currents meets a central-difference sweep, its
    Taylor remainder falls at second order, it satisfies the JVP/VJP identity, it meets its own central difference
    on a finer mesh, and a new value does not recompile. The steady solve builds its walls as constants: a gradient
    of its pressure drop in them is planned with the wall-conductance work, and a traced conductivity is refused."""
    from lmhdx.core3d import SolidWall, _motional, _solid_wall_currents

    traces = []

    def objective_on(cells):
        wall = SolidWall(0.1, 2, 1.0, 1.0, 2.0)
        problem = ChannelProblem(
            _duct(cells, cells, 1),
            (PERIODIC_X, WALL, WALL),
            magnetic_field=(0.0, 10.0, 0.0),
            forcing=(1.0, 0.0, 0.0),
            dt=0.01,
            solid_walls=((None, None), (wall, wall), (wall, wall)),
        )
        *_, motional = _motional(_noise(problem), problem)
        rng = np.random.default_rng(cells)
        weights = [jnp.asarray(rng.standard_normal(flux.data.shape)) for flux in motional]

        def objective(values):
            traces.append(None)
            _, currents = _solid_wall_currents(motional, problem, {(1, 0): values[0], (2, 1): values[1]})
            return sum(jnp.vdot(weight, current.data) for weight, current in zip(weights, currents))

        return objective

    objective, sigma, direction = objective_on(8), jnp.asarray([2.0, 2.0]), jnp.asarray([1.0, -0.5])
    slope = float(jax.grad(objective)(sigma) @ direction)
    errors = [
        abs(
            float(objective(sigma + h * direction) - objective(sigma - h * direction)) / (2.0 * h) / slope
            - 1.0
        )
        for h in (1e-2, 1e-3)
    ]
    assert errors[1] < 1e-7 and errors[1] < errors[0] / 50.0, errors  # the central difference's own h^2
    base = objective(sigma)
    remainders = [abs(float(objective(sigma + h * direction) - base) - h * slope) for h in (0.1, 0.05, 0.025)]
    assert all(3.5 < coarse / fine < 4.5 for coarse, fine in zip(remainders, remainders[1:])), remainders
    tangent = float(jax.jvp(objective, (sigma,), (direction,))[1])
    assert float(jax.vjp(objective, sigma)[1](1.0)[0] @ direction) == pytest.approx(tangent, rel=1e-10)
    traces.clear()
    jitted = jax.jit(objective)
    jitted(sigma), jitted(1.2 * sigma)
    assert len(traces) == 1, "a new wall conductivity recompiled"
    finer = objective_on(12)
    central = (finer(sigma + 1e-3 * direction) - finer(sigma - 1e-3 * direction)) / 2e-3
    assert float(jax.grad(finer)(sigma) @ direction) == pytest.approx(float(central), rel=1e-6)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.grad(lambda s: SolidWall(0.1, 2, 1.0, 1.0, s).electrical_conductivity)(2.0)


@pytest.mark.slow
def test_solid_walls_of_conductance_0_05_meet_hunt_s_duct_at_ha_300():
    """A published number for the solid wall. Hunt's duct at Ha 300 has conducting Hartmann walls of
    c = sigma_w t / (sigma a) = 0.05 and insulating side walls. Rogerio's digest attributes dp/dx = 374.897 at Re 10
    to Ni et al. (2007), not checked against the paper: that is (Ha^2/Re) K for K = dp/dx / (sigma u B^2) =
    0.0416552, and the spectral solve of validation.shercliff gives 0.0416622, 0.017 % away. On 48 cells the
    thin-wall model is 0.48 % below it, converging from 1.1 % on 32. Solid walls of that c, 0.1 and 0.05 of the
    half-width thick, sit 2.7 % and 0.74 % below the thin-wall model, approaching it at second order in the
    thickness."""
    from lmhdx.core3d import SolidWall
    from lmhdx.fully_developed import channel_flow_rate
    from lmhdx.steady import solve_steady_state

    hartmann, digest = 300.0, 374.897 / (300.0**2 / 10.0)

    def pressure_gradient(problem):
        rate = float(channel_flow_rate(problem, solve_steady_state(problem).velocity[0].data[0]))
        return 1.0 / (hartmann**2 * rate / 4.0)

    thin = duct_problem(hartmann=hartmann, cells=48, wall_conductance=0.05)
    reference = pressure_gradient(thin)
    assert abs(reference / digest - 1.0) < 6e-3
    below = []
    for thickness in (0.1, 0.05):
        wall = SolidWall(thickness, 3, 1.0, 1.0, 0.05 / thickness)
        solid = dataclasses.replace(
            thin, wall_conductance=(0.0,) * 3, solid_walls=((None, None), (wall, wall), (None, None))
        )
        below.append(1.0 - pressure_gradient(solid) / reference)
    assert below == pytest.approx([0.0267, 0.0074], rel=0.05)
    assert 3.3 < below[0] / below[1] < 4.3


def test_solid_walls_lift_the_one_axial_cell_limit_and_conserve_charge_in_3d():
    """Walls resolved in cells on two axes need one axial cell; solid walls take any. In a 3-D duct with a field
    varying along it, the fluid's charge balance closes near the direct solves' rounding, the walls dissipate what
    the Lorentz work does not leave in the fluid, the Lorentz force differentiates in the field's scale both
    ways, and steps run."""
    from lmhdx.core3d import SolidWall, electric_state, energy_budget, face_currents

    grid = _duct(10, 10, 4)
    x = np.asarray(grid.centers[0])[:, None, None] * np.ones(grid.shape)
    field = (0.0, 10.0 * (1.0 + 0.5 * np.cos(2.0 * np.pi * x)), 0.0)
    layer = ((2.0, 2.0), (0.05, 0.05))
    with pytest.raises(ValueError, match="one cell along the third"):
        ChannelProblem(grid, (PERIODIC_X, WALL, WALL), wall_layers=(None, (layer, layer), (layer, layer)))
    wall = SolidWall(0.1, 2, 1.0, 1.0, 2.0)
    problem = ChannelProblem(
        grid,
        (PERIODIC_X, WALL, WALL),
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        dt=0.01,
        solid_walls=((None, None), (wall, wall), (wall, wall)),
    )
    velocity = _noise(problem)
    _, currents, _ = face_currents(velocity, problem)
    scale = max(float(jnp.max(jnp.abs(current.data))) for current in currents)
    assert float(jnp.max(jnp.abs(divergence(currents).data))) <= 1e-10 * scale
    budget = energy_budget(velocity, problem)
    assert float(budget.wall) > 0.0
    assert abs(float(budget.lorentz + budget.joule + budget.wall)) <= 1e-9 * float(budget.joule)

    def force(scale):
        return electric_state(velocity, problem, field_scale=scale)[1][0].data

    # Quadratic in the scale, so the central difference is exact; the gradient runs the adjoint solve.
    tangent, difference = jax.jvp(force, (1.0,), (1.0,))[1], force(1.5) - force(0.5)
    assert np.asarray(tangent) == pytest.approx(
        np.asarray(difference), abs=1e-8 * float(jnp.max(jnp.abs(difference)))
    )
    weights = jnp.asarray(np.random.default_rng(8).standard_normal(difference.shape))
    gradient = float(jax.grad(lambda scale: jnp.vdot(weights, force(scale)))(1.0))
    assert gradient == pytest.approx(float(jnp.vdot(weights, difference)), rel=1e-8)
    moved = velocity
    for _ in range(3):
        moved = step(moved, problem, problem.factorization(), problem.viscous_factorizations())[0]
    assert all(bool(jnp.all(jnp.isfinite(component.data))) for component in moved)


def test_solid_walls_are_checked_and_the_benchmark_s_charge_takes_few_iterations():
    """What a flow's solid walls refuse, how a thermal problem takes them, and the benchmark's charge: steel at
    the pinned 1.2457 times the PbLi's conductivity, 2/23 of the half-width thick on four sides, converges to
    1e-13 in 10 PCG iterations on a random compatible right-hand side, its solution of zero mean."""
    from lmhdx.core3d import SolidWall, _solid_wall_solve
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    base = duct_problem(hartmann=20.0, cells=12, cells_in_layer=3)
    bare, wall = SolidWall(0.1, 2, 1.0, 1.0), SolidWall(0.1, 2, 1.0, 1.0, 2.0)
    pair = ((None, None), (wall, wall), (None, None))
    layer = ((2.0, 2.0), (0.05, 0.05))
    across = ChannelProblem(base.grid, (PERIODIC_X, BoundaryCondition(PERIODIC), WALL))
    for problem, overrides, message in (
        (base, {"solid_walls": ((None, None), (bare, bare), (None, None))}, "electrical_conductivity"),
        (base, {"solid_walls": ((bare, None), (None, None), (None, None))}, "transverse"),
        (base, {"wall_conductance": (0.0, 0.05, 0.0), "solid_walls": pair}, "beside them"),
        (base, {"wall_layers": (None, None, (layer, layer)), "solid_walls": pair}, "beside them"),
        (
            base,
            {"conductivity": 0.0, "magnetic_field": (0.0, 0.0, 0.0), "solid_walls": pair},
            "conducting fluid",
        ),
        (across, {"conductivity": 1.0, "solid_walls": pair}, "periodic"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(problem, **overrides)
    assert dataclasses.replace(base, solid_walls=None).solid_walls == ((None, None),) * 3
    steel = SolidWall(2.0 / 23.0, 4, 18.3, 4.27, 1.2457)
    walls = ((None, None), (steel, steel), (steel, steel))
    walled = dataclasses.replace(base, solid_walls=walls)
    insulated = thermal_boundary("heat_flux", conductivity=18.3)
    thermal = ThermalProblem(walled, (PERIODIC_X, insulated, insulated), 1.0, 1.0, 0.0)  # the flow's walls
    assert thermal.walls == walls and thermal.thermal_grid.shape == (1, 20, 20)
    with pytest.raises(ValueError, match="only among the flow's solid_walls"):
        ThermalProblem(base, (PERIODIC_X, insulated, insulated), 1.0, 1.0, 0.0, walls=walls)
    with pytest.raises(ValueError, match="give the same, or none"):
        ThermalProblem(
            walled,
            (PERIODIC_X, insulated, insulated),
            1.0,
            1.0,
            0.0,
            walls=((None, None), (bare, bare), (None, None)),
        )
    conjugate, charge = _solid_wall_solve(walled.grid, walled.scalar_conditions, walls, 1.0, "state")
    rhs = jnp.asarray(np.random.default_rng(5).standard_normal(conjugate.grid.shape))
    potential, iterations = charge.solve(rhs - charge.defect(rhs))  # compatible: a net source is refused
    assert int(iterations) <= 12
    assert abs(float(jnp.sum(jnp.asarray(conjugate.grid.cell_volumes()) * potential))) < 1e-10


# Buoyancy in the momentum stages (plan step 5.1e, ADR 0008 D32).


def _buoyant(grid: Grid, flow_conditions, *, density: float = 1.0, field=(0.0, 0.0, 0.0)):
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    flow = ChannelProblem(
        grid, flow_conditions, density=density, forcing=(0.0, 0.0, 0.0), dt=1.0e-2, magnetic_field=field
    )
    insulated = thermal_boundary("heat_flux", conductivity=1.0)
    conditions = tuple(c if c.is_periodic else insulated for c in flow_conditions)
    return ThermalProblem(
        flow, conditions, 1.0, 1.0, 300.0, expansion_coefficient=2.0e-3, gravity=(0.0, 0.0, -9.8)
    )


def _buoyant_step(thermal, temperature):
    from lmhdx.core3d import _advance_momentum
    from lmhdx.thermal import _buoyancy

    flow = thermal.flow
    rest = zero_velocity(flow)
    body = tuple(field.replace_data(jnp.zeros_like(field.data)) for field in rest)
    extra = _buoyancy(Field(jnp.asarray(temperature), (CENTER,) * 3, flow.grid), thermal)
    return _advance_momentum(
        rest, body, flow, flow.factorization(), flow.viscous_factorizations(), extra_force=extra
    )


def test_a_temperature_stratified_along_gravity_stays_at_rest():
    """The buoyancy of ``T(z)`` is a pure gradient, so the projection takes all of it into the pressure."""
    grid = Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0), tanh_faces(8, -1.0, 1.0, 1.3))
    thermal = _buoyant(grid, (PERIODIC_X, PERIODIC_X, WALL))
    z = np.asarray(grid.centers[2])[None, None, :]
    velocity, pressure = _buoyant_step(thermal, np.broadcast_to(300.0 + 3.0 * z**2, grid.shape))
    scale = thermal.flow.dt * 2.0e-3 * 9.8 * 3.0
    assert max(float(jnp.max(jnp.abs(component.data))) for component in velocity) <= 1e-12 * scale
    assert float(jnp.max(jnp.abs(pressure.data))) > 0.0


def test_a_horizontal_temperature_difference_drives_a_circulation():
    """``T(x)`` under vertical gravity is not a gradient: the projection keeps a circulation of its size."""
    grid = Grid(uniform_faces(8, 0.0, 2.0), uniform_faces(1, 0.0, 1.0), uniform_faces(8, -1.0, 1.0))
    thermal = _buoyant(grid, (PERIODIC_X, PERIODIC_X, WALL))
    x = np.asarray(grid.centers[0])[:, None, None]
    velocity, _ = _buoyant_step(thermal, np.broadcast_to(300.0 + np.cos(np.pi * x), grid.shape))
    scale = thermal.flow.dt * 2.0e-3 * 9.8
    assert float(jnp.max(jnp.abs(velocity[2].data))) >= 0.1 * scale
    assert float(jnp.max(jnp.abs(velocity[0].data))) >= 0.1 * scale
    assert float(jnp.max(jnp.abs(divergence(velocity).data))) <= 1e-10 * scale


def test_buoyancy_accelerates_the_flow_independently_of_its_density():
    """The force density scales with ``rho_0`` and is divided by it; the pressure keeps the factor."""
    grid = Grid(uniform_faces(8, 0.0, 2.0), uniform_faces(1, 0.0, 1.0), tanh_faces(8, -1.0, 1.0, 1.3))
    x = np.asarray(grid.centers[0])[:, None, None]
    temperature = np.broadcast_to(300.0 + np.cos(np.pi * x), grid.shape)
    light = _buoyant_step(_buoyant(grid, (PERIODIC_X, PERIODIC_X, WALL)), temperature)
    heavy = _buoyant_step(_buoyant(grid, (PERIODIC_X, PERIODIC_X, WALL), density=2.5), temperature)
    for first, second in zip(light[0], heavy[0]):
        assert float(jnp.max(jnp.abs(first.data - second.data))) <= 1e-13 * float(
            jnp.max(jnp.abs(first.data))
        )
    assert np.allclose(np.asarray(heavy[1].data), 2.5 * np.asarray(light[1].data), rtol=1e-12, atol=1e-12)


def _benard_growth(rayleigh: float, dt: float, chandrasekhar: float = 0.0) -> float:
    """Growth rate of the critical roll between rigid isothermal plates, heated below, in units of ``alpha/d^2``.

    ``d = nu = alpha = 1``, so ``g beta Delta T = Ra``; a vertical field of ``sigma B^2 = Q``. The rate is
    taken between ``t = 2`` and ``3``, after the faster modes have decayed and long before saturation.
    """
    from lmhdx.thermal import (
        ThermalProblem,
        advance_thermal,
        build_thermal_step,
        initial_thermal_state,
        thermal_boundary,
    )

    wavenumber = 3.117
    grid = Grid(
        uniform_faces(16, 0.0, 2.0 * np.pi / wavenumber),
        uniform_faces(1, 0.0, 1.0),
        uniform_faces(16, 0.0, 1.0),
    )
    field = (0.0, 0.0, float(np.sqrt(chandrasekhar)))
    flow = ChannelProblem(grid, (PERIODIC_X, PERIODIC_X, WALL), magnetic_field=field, dt=dt)
    plates = thermal_boundary("temperature", 1.0, 0.0)
    thermal = ThermalProblem(
        flow,
        (PERIODIC_X, PERIODIC_X, plates),
        1.0,
        1.0,
        0.0,
        expansion_coefficient=1.0,
        gravity=(0.0, 0.0, -rayleigh),
    )
    x, z = np.asarray(grid.centers[0])[:, None, None], np.asarray(grid.centers[2])[None, None, :]
    state = initial_thermal_state(
        thermal, np.broadcast_to(1.0 - z + 1e-8 * np.cos(wavenumber * x) * np.sin(np.pi * z), grid.shape)
    )
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64)
    heat = state.temperature.replace_data(jnp.zeros(grid.shape))
    run = jax.jit(
        lambda start: advance_thermal(thermal_step, start, heat, round(1.0 / dt), checkpoint=False)[0]
    )
    amplitudes = []
    for _ in range(3):
        state = run(state)
        amplitudes.append(float(jnp.max(jnp.abs(state.velocity[2].data))))
    return float(np.log(amplitudes[2] / amplitudes[1]))


def test_benard_convection_sets_in_at_the_critical_rayleigh_number():
    """Rigid plates: onset at ``Ra_c = 1707.76``, ``k_c = 3.117`` (Reid & Harris 1958; Chandrasekhar 1961, Ch. II).

    The partitioned step lags buoyancy and transport by one step, so the onset converges at first order in
    ``dt`` from below; Richardson in ``dt`` on 16 x 16 cells lands within 4 %. A vertical field of
    ``Q = Ha^2 = 100`` lifts onset to ``Ra_c = 3757`` (Chandrasekhar 1961, Ch. IV), so ``Ra = 2500`` grows
    without it and decays with it.
    """
    onsets = []
    for dt in (2.5e-3, 1.25e-3):
        below, above = (_benard_growth(rayleigh, dt) for rayleigh in (1600.0, 1800.0))
        onsets.append(1600.0 - below * 200.0 / (above - below))
    assert onsets[0] < onsets[1] < 1707.76, onsets
    assert abs((2.0 * onsets[1] - onsets[0]) / 1707.76 - 1.0) < 0.04, onsets
    assert _benard_growth(2500.0, 1.25e-3) > 0.0 > _benard_growth(2500.0, 1.25e-3, chandrasekhar=100.0)


# The coupled thermal step's accuracy (plan step 5.1f).


def _thermal_run(thermal, temperature, velocity, source: float, steps: int):
    from lmhdx.thermal import build_thermal_step, initial_thermal_state

    thermal_step = jax.jit(build_thermal_step(thermal, dtype=jnp.float64))
    state = initial_thermal_state(thermal, temperature, velocity)
    heat = state.temperature.replace_data(jnp.full(thermal.flow.grid.shape, source))
    for _ in range(steps):
        state, record = thermal_step(state, heat)
    assert bool(record.finite)
    return state


def test_a_carried_and_diffused_wave_converges_at_first_order():
    """At uniform speed ``U`` in a periodic box ``exp(-alpha k^2 t) cos k(x - Ut) + q t / C`` is exact.

    Upwinding at Courant 0.5 and backward Euler both err by O(h) at a fixed Courant number.
    """
    from lmhdx.thermal import ThermalProblem

    errors = []
    for cells in (32, 64, 128):
        grid = Grid(uniform_faces(cells, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0))
        flow = ChannelProblem(grid, (PERIODIC_X,) * 3, forcing=(0.0, 0.0, 0.0), dt=0.5 / cells)
        thermal = ThermalProblem(flow, (PERIODIC_X,) * 3, 1.0, 0.05, 0.0)
        rest = zero_velocity(flow)
        carrier = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])
        x = np.asarray(grid.centers[0])[:, None, None]
        state = _thermal_run(thermal, np.cos(2.0 * np.pi * x), carrier, 2.0, cells // 2)
        exact = np.exp(-0.05 * (2.0 * np.pi) ** 2 * 0.25) * np.cos(2.0 * np.pi * (x - 0.25)) + 2.0 * 0.25
        errors.append(float(np.max(np.abs(np.asarray(state.temperature.data) - exact))))
    orders = np.log2(np.divide(errors[:-1], errors[1:]))
    assert np.all((orders > 0.8) & (orders < 1.2)), (errors, orders)


def test_the_coupled_step_converges_at_first_order_in_time():
    """Buoyancy, Joule heat, a drive and heated walls together: temperature and velocity err by O(dt).

    Against a run at dt/32 on the same mesh, to t = 0.2 (ADR 0008, V16).
    """
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(8, -1.0, 1.0), uniform_faces(8, -1.0, 1.0))
    y = np.asarray(grid.centers[1])[None, :, None]
    start = np.broadcast_to(300.0 + 0.5 * np.sin(np.pi * y), grid.shape)
    walls = (
        PERIODIC_X,
        thermal_boundary("heat_flux", conductivity=0.2),
        thermal_boundary("temperature", 301.0, 299.0),
    )
    results = {}
    for dt in (0.02, 0.01, 0.005, 0.000625):
        flow = ChannelProblem(
            grid, (PERIODIC_X, WALL, WALL), magnetic_field=(0.0, 5.0, 0.0), forcing=(1.0, 0.0, 0.0), dt=dt
        )
        thermal = ThermalProblem(
            flow, walls, 1.0, 0.2, 300.0, expansion_coefficient=0.05, gravity=(0.0, 0.0, -1.0)
        )
        results[dt] = _thermal_run(thermal, start, None, 0.0, round(0.2 / dt))
    reference = results.pop(0.000625)
    errors = [
        [
            float(jnp.max(jnp.abs(state.temperature.data - reference.temperature.data))),
            max(float(jnp.max(jnp.abs(a.data - b.data))) for a, b in zip(state.velocity, reference.velocity)),
        ]
        for state in results.values()
    ]
    orders = np.log2(np.divide(errors[:-1], errors[1:]))
    assert np.all((orders > 0.8) & (orders < 1.5)), (errors, orders)
