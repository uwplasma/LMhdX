"""Charge-conservative face currents, Ohm's law and the face Lorentz force."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.grid import (
    CENTER,
    DIRICHLET,
    FACE,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    geometric_faces,
    tanh_faces,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.ops import (
    cell_average,
    cell_inner_product,
    charge_residual,
    face_average,
    face_average_adjoint,
    face_conductivity,
    face_current,
    face_electromotive_force,
    face_inner_product,
    lorentz_force,
    thin_wall_current,
    thin_wall_flux,
    wall_insulated,
)
from lmhdx.poisson import pipe_grid

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
WRAP = BoundaryCondition(PERIODIC)
WALLS = (WALL, WALL, WALL)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)
UNIFORM = Grid(uniform_faces(4, 0.0, 2.0), uniform_faces(5, -1.0, 1.0), uniform_faces(3, -0.5, 0.5))
# Hartmann and side layers of a Ha 100 duct, with the cross-stream axis stretched as well.
LAYERED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    wall_resolving_faces(24, -1.0, 1.0, layer_thickness=0.01, cells_in_layer=6, max_ratio=None),
    wall_resolving_faces(16, -1.0, 1.0, layer_thickness=0.1, cells_in_layer=6, max_ratio=None),
)


def _cells(grid: Grid, function) -> Field:
    x, y, z = (np.asarray(values) for values in grid.centers)
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    return Field(jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.shape), (CENTER,) * 3, grid)


def _faces(grid: Grid, axis: int, function) -> Field:
    coordinates = [np.asarray(values) for values in grid.centers]
    coordinates[axis] = np.asarray(grid.faces[axis])
    x, y, z = coordinates
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    offset = tuple(FACE if position == axis else CENTER for position in range(3))
    return Field(
        jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.face_shape(axis)), offset, grid
    )


def _uniform_velocity(grid: Grid, components) -> tuple[Field, Field, Field]:
    return tuple(
        _faces(grid, axis, lambda x, y, z, value=components[axis]: value * jnp.ones_like(x * y * z))
        for axis in range(3)
    )


def _uniform_field(grid: Grid, components) -> tuple[Field, Field, Field]:
    return tuple(
        _cells(grid, lambda x, y, z, value=components[axis]: value * jnp.ones_like(x * y * z))
        for axis in range(3)
    )


def test_cell_average_of_a_linear_face_field_returns_the_cell_value():
    grid, axis = STRETCHED, 0
    faces = _faces(grid, axis, lambda x, y, z: 2.0 * x + 1.0)
    averaged = np.asarray(cell_average(faces, axis).data)
    expected = 2.0 * np.asarray(grid.centers[axis]) + 1.0
    assert np.max(np.abs(averaged - expected[:, None, None])) < 1e-13


def test_uniform_conductivity_survives_the_face_average():
    sigma = _cells(STRETCHED, lambda x, y, z: 3.5 * jnp.ones_like(x * y * z))
    for axis in range(3):
        faces = face_conductivity(sigma, axis, WALL)
        assert faces.shape == STRETCHED.face_shape(axis)
        assert np.max(np.abs(np.asarray(faces.data) - 3.5)) < 1e-13


def test_face_conductivity_across_a_jump_is_the_harmonic_mean():
    grid, axis = UNIFORM, 1
    low, high = 1.0e-3, 5.0
    sigma = _cells(grid, lambda x, y, z: jnp.where(y < 0.0, low, high) * jnp.ones_like(x * z))
    faces = np.asarray(face_conductivity(sigma, axis, WALL).data)
    centers = np.asarray(grid.centers[axis])
    interface = int(np.searchsorted(centers, 0.0))
    harmonic = 2.0 * low * high / (low + high)
    assert faces[:, interface, :] == pytest.approx(harmonic, rel=1e-12)
    # Away from the interface each side keeps its own value.
    assert faces[:, 1, :] == pytest.approx(low, rel=1e-12)
    assert faces[:, -2, :] == pytest.approx(high, rel=1e-12)
    # The harmonic mean is dominated by the poor conductor, unlike the arithmetic one.
    assert harmonic < 0.5 * (low + high)


def test_electromotive_force_matches_the_analytic_cross_product():
    grid = STRETCHED
    velocity_values, field_values = (0.3, -1.1, 2.0), (0.7, 0.2, -0.5)
    velocity = _uniform_velocity(grid, velocity_values)
    magnetic_field = _uniform_field(grid, field_values)
    expected = np.cross(np.asarray(velocity_values), np.asarray(field_values))
    for axis in range(3):
        emf = face_electromotive_force(velocity, magnetic_field, axis, WALLS)
        assert emf.shape == grid.face_shape(axis)
        assert np.max(np.abs(np.asarray(emf.data) - expected[axis])) < 1e-13


def test_ohms_law_reduces_to_each_of_its_two_terms():
    grid, axis, sigma_value = STRETCHED, 2, 2.5
    sigma = _cells(grid, lambda x, y, z: sigma_value * jnp.ones_like(x * y * z))
    conductivity = face_conductivity(sigma, axis, WALL)
    zero_emf = conductivity.replace_data(jnp.zeros_like(conductivity.data))
    slope = 1.7
    faces = np.asarray(grid.faces[axis])
    potential = _cells(grid, lambda x, y, z: slope * z)
    condition = BoundaryCondition(DIRICHLET, lower=slope * faces[0], upper=slope * faces[-1])

    conduction = face_current(potential, conductivity, zero_emf, axis, condition)
    assert np.max(np.abs(np.asarray(conduction.data) + sigma_value * slope)) < 1e-12

    flat = _cells(grid, lambda x, y, z: jnp.zeros_like(x * y * z))
    emf_value = 0.9
    emf = conductivity.replace_data(emf_value * jnp.ones_like(conductivity.data))
    motional = face_current(flat, conductivity, emf, axis, WALL)
    assert np.max(np.abs(np.asarray(motional.data) - sigma_value * emf_value)) < 1e-12


def test_uniform_current_in_a_uniform_field_gives_the_analytic_lorentz_force():
    grid = UNIFORM
    current_values, field_values = (0.5, -2.0, 1.25), (0.3, 0.8, -1.4)
    currents = _uniform_velocity(grid, current_values)
    magnetic_field = _uniform_field(grid, field_values)
    expected = np.cross(np.asarray(current_values), np.asarray(field_values))
    force = lorentz_force(currents, magnetic_field, WALLS)
    for axis in range(3):
        assert force[axis].shape == grid.shape
        assert np.max(np.abs(np.asarray(force[axis].data) - expected[axis])) < 1e-12


def test_lorentz_force_is_orthogonal_to_a_uniform_field():
    grid = STRETCHED
    field_values = (0.0, 1.0, 0.0)
    magnetic_field = _uniform_field(grid, field_values)
    currents = tuple(
        _faces(grid, axis, lambda x, y, z, axis=axis: jnp.sin(x + 2.0 * y - z + axis)) for axis in range(3)
    )
    force = lorentz_force(currents, magnetic_field, WALLS)
    projection = sum(np.asarray(force[axis].data) * field_values[axis] for axis in range(3))
    assert np.max(np.abs(projection)) < 1e-13


def _random_faces(grid: Grid, axis: int, key, condition: BoundaryCondition) -> Field:
    """Random face data, zero on the wall faces of a non-periodic axis."""
    data = jax.random.normal(key, grid.face_shape(axis), dtype=jnp.float64)
    if not condition.is_periodic:
        data = data.at[(slice(None),) * axis + (0,)].set(0.0).at[(slice(None),) * axis + (-1,)].set(0.0)
    return Field(data, tuple(FACE if position == axis else CENTER for position in range(3)), grid)


@pytest.mark.parametrize(
    ("grid", "conditions"),
    [
        (LAYERED, (WALL, WALL, WALL)),
        (LAYERED, (WRAP, WALL, WRAP)),
        (pipe_grid(16, 24, 100.0), (WALL, WRAP, WRAP)),
    ],
    ids=["layered-walls", "layered-periodic", "polar-pipe"],
)
def test_the_face_average_and_its_adjoint_are_exact_transposes(grid, conditions):
    """The interpolation pair of the electromotive and Lorentz forces, under the volume weights."""
    keys = jax.random.split(jax.random.PRNGKey(11), 6)
    for axis in range(3):
        condition = conditions[axis]
        cells = Field(jax.random.normal(keys[axis], grid.shape, dtype=jnp.float64), (CENTER,) * 3, grid)
        faces = _random_faces(grid, axis, keys[3 + axis], condition)
        left = float(face_inner_product(faces, face_average(cells, axis, condition), axis, condition))
        right = float(cell_inner_product(face_average_adjoint(faces, axis, condition), cells))
        assert abs(left - right) <= 1e-13 * abs(left)

        # The same statement through JAX's own transpose of the interpolation.
        weights = jax.grad(
            lambda data, faces=faces, condition=condition: face_inner_product(
                faces.replace_data(data), faces.replace_data(jnp.ones_like(data)), axis, condition
            )
        )(jnp.zeros_like(faces.data))
        (transposed,) = jax.linear_transpose(
            lambda data, condition=condition: face_average(cells.replace_data(data), axis, condition).data,
            cells.data,
        )(weights * faces.data)
        expected = transposed / jnp.asarray(grid.cell_volumes())
        adjoint = face_average_adjoint(faces, axis, condition).data
        assert float(jnp.max(jnp.abs(adjoint - expected))) <= 1e-13 * float(jnp.max(jnp.abs(expected)))

        # Both halves are averages, so constants survive either direction.
        ones = faces.replace_data(jnp.ones_like(faces.data))
        assert float(jnp.max(jnp.abs(face_average_adjoint(ones, axis, condition).data - 1.0))) < 1e-12
        flat = cells.replace_data(jnp.ones_like(cells.data))
        assert float(jnp.max(jnp.abs(face_average(flat, axis, condition).data - 1.0))) < 1e-14


def test_the_face_average_is_the_distance_weighted_interpolation_on_uniform_cells():
    """The two differ only where neighbouring cells differ in width."""
    from lmhdx.ops import face_interpolate

    field = _cells(UNIFORM, lambda x, y, z: jnp.sin(x + 2.0 * y) * jnp.cos(3.0 * z))
    for axis in range(3):
        averaged = np.asarray(face_average(field, axis, WALL).data)
        interpolated = np.asarray(face_interpolate(field, axis, WALL).data)
        assert np.max(np.abs(averaged - interpolated)) < 1e-15
    stretched = _cells(STRETCHED, lambda x, y, z: x + y + z)
    assert (
        np.max(
            np.abs(
                np.asarray(face_average(stretched, 0, WALL).data - face_interpolate(stretched, 0, WALL).data)
            )
        )
        > 1e-2
    )


def test_the_lorentz_force_is_minus_the_adjoint_of_the_electromotive_force():
    """The work the face force does on a velocity is minus the current dotted with its electromotive force.

    That is the discrete form of `u.(J x B) = -J.(u x B)`, and it holds exactly on a
    stretched mesh with a field that varies in space. It is what makes the Lorentz
    force do exactly minus the Joule dissipation and the steady Stokes operator
    symmetric, which the conjugate-gradient solve of plan step 1.7d depends on.
    """
    grid = LAYERED
    conditions = (WRAP, WALL, WALL)
    magnetic_field = (
        _cells(grid, lambda x, y, z: 0.3 + 0.1 * jnp.sin(y) + 0.0 * x * z),
        _cells(grid, lambda x, y, z: 1.0 + 0.2 * jnp.cos(z) * jnp.sin(x)),
        _cells(grid, lambda x, y, z: -0.4 + 0.1 * y * z + 0.0 * x),
    )
    keys = jax.random.split(jax.random.PRNGKey(7), 6)
    velocity = tuple(_random_faces(grid, axis, keys[axis], conditions[axis]) for axis in range(3))
    currents = tuple(_random_faces(grid, axis, keys[3 + axis], conditions[axis]) for axis in range(3))

    power = sum(
        float(
            face_inner_product(
                currents[axis],
                face_electromotive_force(velocity, magnetic_field, axis, conditions),
                axis,
                conditions[axis],
            )
        )
        for axis in range(3)
    )
    force = lorentz_force(currents, magnetic_field, conditions)
    work = sum(
        float(
            face_inner_product(
                velocity[axis], face_average(force[axis], axis, conditions[axis]), axis, conditions[axis]
            )
        )
        for axis in range(3)
    )
    assert abs(work + power) <= 1e-13 * abs(power)


def test_the_anl_fringe_is_divergence_free_on_the_grid():
    """Plan step 1.9a: face means of one flux function cancel in the discrete divergence.

    Measured: 0 on uniform cells and 2.6e-16 on this tanh mesh. The analytic pair has a divergence and a
    curl of round-off (6e-17). Away from the two joins, where ``B_y`` jumps by ``(cosh(ky) - 1)/2``
    (0.070 of ``B0`` at ``|y| = 1``), the cells hold it to the midpoint error of averaging two faces,
    ``h^2 k^2 B0 cosh(k) / 8``: 1.2e-3 measured here against that bound of 4.9e-3.
    """
    from lmhdx.core3d import fringe_field
    from lmhdx.ops import divergence

    grid = Grid(uniform_faces(48, -6.0, 6.0), tanh_faces(24, -1.0, 1.0, 2.0), uniform_faces(3, -1.0, 1.0))
    k, x, y = np.pi / 6.0, grid.centers[0][:, None], grid.centers[1][None, :]
    for solenoidal in (True, False):
        field = fringe_field(grid, strength=2.0, solenoidal=solenoidal)
        offsets = [tuple(FACE if position == axis else CENTER for position in range(3)) for axis in range(3)]
        faces = tuple(Field(jnp.asarray(data), offsets[axis], grid) for axis, data in enumerate(field.faces))
        assert float(jnp.max(jnp.abs(divergence(faces).data))) <= 1e-12
        rise = np.cosh(k * y) if solenoidal else 1.0
        inside = np.clip(x, -3.0, 3.0)
        expected = (
            -np.cos(k * inside) * np.sinh(k * y) * float(solenoidal) * (np.abs(x) < 3.0),
            1.0 - np.sin(k * inside) * rise + np.where(np.abs(x) < 3.0, 0.0, np.sign(-x) * (1.0 - rise)),
        )
        away = np.abs(np.abs(grid.centers[0]) - 3.0) > 0.3
        bound = 0.25**2 * k**2 * 2.0 * np.cosh(k) / 8.0
        for component, values in zip(field.components, expected, strict=False):
            assert np.max(np.abs(component[away, :, 0] - values[away])) <= bound
        assert np.all(field.components[2] == 0.0)

    def pair(point):
        return jnp.stack(
            [
                -jnp.cos(k * point[0]) * jnp.sinh(k * point[1]),
                1.0 - jnp.sin(k * point[0]) * jnp.cosh(k * point[1]),
            ]
        )

    points = jnp.asarray(np.random.default_rng(0).uniform([-2.99, -1.0], [2.99, 1.0], size=(64, 2)))
    gradient = jax.vmap(jax.jacfwd(pair))(points)
    assert float(jnp.max(jnp.abs(gradient[:, 0, 0] + gradient[:, 1, 1]))) <= 1e-14
    assert float(jnp.max(jnp.abs(gradient[:, 1, 0] - gradient[:, 0, 1]))) <= 1e-14
    with pytest.raises(ValueError, match="Cartesian"):
        fringe_field(pipe_grid(16, 24, 100.0))


def test_a_divergence_free_current_leaves_no_charge_residual():
    grid = STRETCHED
    currents = (
        _faces(grid, 0, lambda x, y, z: 2.0 * x),
        _faces(grid, 1, lambda x, y, z: -3.0 * y),
        _faces(grid, 2, lambda x, y, z: 1.0 * z),
    )
    residual = np.asarray(charge_residual(currents).data)
    assert np.max(np.abs(residual)) < 1e-12


def test_insulated_conduction_current_conserves_charge_when_the_potential_solves_its_equation():
    """The potential that solves the discrete equation makes the face fluxes close."""
    from lmhdx.grid import PERIODIC
    from lmhdx.ops import laplacian
    from lmhdx.poisson import fast_diagonal_poisson

    grid = STRETCHED
    conditions = (BoundaryCondition(PERIODIC), WALL, WALL)
    sigma = _cells(grid, lambda x, y, z: jnp.ones_like(x * y * z))
    velocity = _uniform_velocity(grid, (0.0, 0.0, 1.3))
    magnetic_field = _uniform_field(grid, (0.0, 0.7, 0.0))
    conductivities = [face_conductivity(sigma, axis, conditions[axis]) for axis in range(3)]
    emfs = [face_electromotive_force(velocity, magnetic_field, axis, conditions) for axis in range(3)]

    # With uniform conductivity the potential equation is the plain Laplacian.
    source = charge_residual(
        tuple(
            conductivities[axis].replace_data(conductivities[axis].data * emfs[axis].data)
            for axis in range(3)
        )
    )
    potential = fast_diagonal_poisson(grid, conditions).solve(source)
    assert float(jnp.max(jnp.abs(laplacian(potential, conditions).data - source.data))) < 1e-11

    currents = tuple(
        face_current(potential, conductivities[axis], emfs[axis], axis, conditions[axis]) for axis in range(3)
    )
    residual = np.asarray(charge_residual(currents).data)
    scale = float(np.max(np.abs(np.asarray(source.data)))) or 1.0
    assert np.max(np.abs(residual)) < 1e-12 * scale


def test_coupling_is_differentiable_and_jits():
    grid = UNIFORM
    sigma = _cells(grid, lambda x, y, z: 1.0 + 0.1 * jnp.sin(x))
    magnetic_field = _uniform_field(grid, (0.0, 1.0, 0.0))

    def work(values):
        potential = Field(values, (CENTER,) * 3, grid)
        velocity = _uniform_velocity(grid, (1.0, 0.0, 0.0))
        currents, forces = [], None
        for axis in range(3):
            conductivity = face_conductivity(sigma, axis, WALL)
            emf = face_electromotive_force(velocity, magnetic_field, axis, WALLS)
            currents.append(face_current(potential, conductivity, emf, axis, WALL))
        forces = lorentz_force(tuple(currents), magnetic_field, WALLS)
        return jnp.sum(forces[0].data ** 2 + forces[2].data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(3), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(work))(values)
    step = 1.0e-6
    direction = jax.random.normal(jax.random.PRNGKey(5), grid.shape, dtype=jnp.float64)
    difference = (work(values + step * direction) - work(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_coupling_preserves_single_precision():
    grid = UNIFORM
    sigma = Field(jnp.ones(grid.shape, dtype=jnp.float32), (CENTER,) * 3, grid)
    conductivity = face_conductivity(sigma, 0, WALL)
    assert conductivity.dtype == jnp.float32


def test_electric_helpers_validate_their_inputs():
    grid = UNIFORM
    cell = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="expected a field on faces"):
        cell_average(cell, 0)
    with pytest.raises(ValueError, match="conductivity must be cell centred"):
        face_conductivity(face, 0, WALL)
    with pytest.raises(ValueError, match="does not match grid"):
        face_conductivity(Field(jnp.zeros((2, 2, 2)), (CENTER,) * 3, grid), 0, WALL)
    with pytest.raises(ValueError, match="conductivity must live on the same faces"):
        face_current(cell, cell, face, 0, WALL)
    with pytest.raises(ValueError, match="electromotive force must live on the same faces"):
        face_current(cell, face, cell, 0, WALL)


def test_an_insulating_wall_carries_no_current():
    """`J.n = 0` is the wall condition, so the closure has to zero the wall faces."""
    grid = UNIFORM
    potential = _cells(grid, lambda x, y, z: x + y)
    conductivity = face_conductivity(_cells(grid, lambda x, y, z: 1.0 + 0.0 * x), 0, WALL)
    electromotive = _faces(grid, 0, lambda x, y, z: 1.0 + y)
    current = wall_insulated(face_current(potential, conductivity, electromotive, 0, WALL), 0, WALL)
    assert float(jnp.max(jnp.abs(current.data[0]))) == 0.0
    assert float(jnp.max(jnp.abs(current.data[-1]))) == 0.0
    assert float(jnp.max(jnp.abs(current.data[1:-1]))) > 0.0


def test_only_an_insulating_wall_is_closed_off():
    """A periodic axis has no wall, and a prescribed potential is a conducting one."""
    grid = UNIFORM
    flux = _faces(grid, 1, lambda x, y, z: 1.0 + 0.0 * x)
    assert wall_insulated(flux, 1, BoundaryCondition("periodic")) is flux
    assert wall_insulated(flux, 1, BoundaryCondition(DIRICHLET)) is flux
    insulated = wall_insulated(flux, 1, WALL)
    assert float(jnp.max(jnp.abs(insulated.data[:, 0]))) == 0.0
    assert float(jnp.max(jnp.abs(insulated.data[:, 1:-1]))) == 1.0


def _sheets(grid: Grid, axis: int, function) -> Field:
    """A wall potential: ``function`` on the two walls normal to ``axis``, zero on the interior faces."""
    data = np.array(_faces(grid, axis, function).data)
    data[(slice(None),) * axis + (slice(1, -1),)] = 0.0
    offset = tuple(FACE if position == axis else CENTER for position in range(3))
    return Field(jnp.asarray(data), offset, grid)


def test_a_thin_wall_conducts_only_what_the_tangential_potential_drives():
    """The wall current is the surface Laplacian of the sheet's own potential."""
    grid = UNIFORM
    conditions = (WALL, WALL, WALL)
    uniform = _sheets(grid, 1, lambda x, y, z: 3.0 + 0.0 * x)
    assert float(jnp.max(jnp.abs(thin_wall_flux(uniform, 1, WALL, 0.05, conditions).data))) < 1e-14

    curved = _sheets(grid, 1, lambda x, y, z: z**2 + 0.0 * x)
    doubled = thin_wall_flux(curved, 1, WALL, 0.10, conditions)
    driven = thin_wall_flux(curved, 1, WALL, 0.05, conditions)
    assert float(jnp.max(jnp.abs(driven.data[:, 0]))) > 0.0
    # The stored value points along the axis, so the two walls carry opposite signs.
    assert np.allclose(np.asarray(driven.data[:, 0]), -np.asarray(driven.data[:, -1]))
    assert float(jnp.max(jnp.abs(driven.data[:, 1:-1]))) == 0.0
    # A wall twice as conductive carries twice the current.
    assert np.allclose(np.asarray(doubled.data), 2.0 * np.asarray(driven.data))
    # Away from the sheet's edges the five-point Laplacian of z^2 is exactly 2.
    np.testing.assert_allclose(np.asarray(driven.data[:, 0, 1:-1]), 0.05 * 2.0, rtol=1e-12)
    # A separate conductance per wall.
    lower_only = thin_wall_flux(curved, 1, WALL, (0.05, 0.0), conditions)
    assert np.array_equal(np.asarray(lower_only.data[:, 0]), np.asarray(driven.data[:, 0]))
    assert float(jnp.max(jnp.abs(lower_only.data[:, -1]))) == 0.0


def test_no_wall_conducts_without_a_conductance_or_without_a_wall():
    grid = UNIFORM
    conditions = (WALL, WALL, WALL)
    curved = _sheets(grid, 1, lambda x, y, z: z**2 + 0.0 * x)
    assert float(jnp.max(jnp.abs(thin_wall_flux(curved, 1, WALL, 0.0, conditions).data))) == 0.0
    periodic = BoundaryCondition("periodic")
    assert float(jnp.max(jnp.abs(thin_wall_flux(curved, 1, periodic, 0.05, conditions).data))) == 0.0
    with pytest.raises(ValueError, match="the wall potential must live on the faces normal to axis 1"):
        thin_wall_flux(_cells(grid, lambda x, y, z: z**2), 1, WALL, 0.05, conditions)


def test_a_thin_wall_draws_the_half_cell_current_of_its_own_potential():
    """Ohm's law across the half cell against the wall, exact for a potential linear in the normal."""
    grid, sigma, slope = STRETCHED, 2.5, 0.7
    potential = _cells(grid, lambda x, y, z: 1.0 + slope * y + 0.3 * z + 0.0 * x)
    walls = _sheets(grid, 1, lambda x, y, z: 1.0 + slope * y + 0.3 * z + 0.0 * x)
    conductivity = face_conductivity(_cells(grid, lambda x, y, z: sigma + 0.0 * x), 1, WALL)
    current = thin_wall_current(potential, walls, conductivity, 1)
    # The stretched wall cells differ in width, and the difference is still exact.
    np.testing.assert_allclose(np.asarray(current.data[:, [0, -1]]), -sigma * slope, rtol=1e-12)
    assert float(jnp.max(jnp.abs(current.data[:, 1:-1]))) == 0.0
    with pytest.raises(ValueError, match="the wall potential must live on the faces normal to axis 1"):
        thin_wall_current(potential, potential, conductivity, 1)


def test_a_sheet_on_a_pipe_wall_takes_the_azimuthal_metric_of_the_wall():
    """The surface Laplacian of a radial wall divides by the wall radius, not the adjacent centre's."""
    grid = pipe_grid(6, 16, 0.0)
    angles, step = np.asarray(grid.centers[1]), float(grid.widths[1][0])
    data = np.zeros(grid.face_shape(0))
    data[-1] = np.cos(angles)[:, None]
    flux = thin_wall_flux(
        Field(jnp.asarray(data), (FACE, CENTER, CENTER), grid), 0, WALL, 0.1, (WALL, WRAP, WRAP)
    )
    expected = 0.1 * 4.0 * np.sin(0.5 * step) ** 2 / step**2 * np.cos(angles)
    np.testing.assert_allclose(np.asarray(flux.data[-1, :, 0]), expected, atol=1e-14)
