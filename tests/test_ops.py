"""Boundary padding and the compatible gradient, divergence and Laplacian."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.core3d import (
    ChannelProblem,
    SolidWall,
    conjugate_grid,
    enforce_face_constraints,
    project,
    step,
    velocity_offset,
    zero_velocity,
)
from lmhdx.grid import (
    CENTER,
    DIRICHLET,
    FACE,
    NEUMANN,
    PERIODIC,
    POLAR,
    BoundaryCondition,
    Field,
    Grid,
    geometric_faces,
    pad,
    tanh_faces,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.ops import (
    advective_step_limit,
    cell_inner_product,
    divergence,
    face_average_adjoint,
    face_distances,
    face_gradient,
    face_inner_product,
    face_interpolate,
    laplacian,
    momentum_advection,
    outgoing_rate,
    staggered_laplacian,
)
from lmhdx.poisson import fast_diagonal_helmholtz, variable_helmholtz
from lmhdx.thermal import (
    ThermalProblem,
    _buoyancy,
    _joule_heating,
    _upwind_fluxes,
    thermal_boundary,
)

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
WRAP = BoundaryCondition(PERIODIC)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)


def _cells(grid: Grid, function) -> Field:
    x, y, z = (np.asarray(values) for values in grid.centers)
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    data = jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.shape)
    return Field(data, (CENTER,) * 3, grid)


def _faces(grid: Grid, axis: int, function) -> Field:
    coordinates = [np.asarray(values) for values in grid.centers]
    coordinates[axis] = np.asarray(grid.faces[axis])
    x, y, z = coordinates
    values = function(x[:, None, None], y[None, :, None], z[None, None, :])
    data = jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float64), grid.face_shape(axis))
    offset = tuple(FACE if position == axis else CENTER for position in range(3))
    return Field(data, offset, grid)


def test_constant_field_has_zero_gradient_on_every_interior_face():
    field = _cells(STRETCHED, lambda x, y, z: jnp.full(jnp.broadcast_shapes(x.shape, y.shape, z.shape), 2.5))
    for axis in range(3):
        gradient = np.asarray(face_gradient(field, axis, WALL).data)
        interior = gradient[(slice(None),) * axis + (slice(1, -1),)]
        assert np.max(np.abs(interior)) < 1e-15
        assert np.max(np.abs(gradient)) < 1e-15  # a homogeneous Neumann wall is flat too


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_field_gradient_is_exact_including_the_walls(axis):
    slope, grid = 1.7, STRETCHED
    field = _cells(grid, lambda x, y, z: slope * (x, y, z)[axis])
    faces = np.asarray(grid.faces[axis])
    condition = BoundaryCondition(DIRICHLET, lower=slope * faces[0], upper=slope * faces[-1])
    gradient = np.asarray(face_gradient(field, axis, condition).data)
    assert np.max(np.abs(gradient - slope)) < 1e-13


def test_divergence_of_a_linear_flux_matches_its_analytic_value():
    grid = STRETCHED
    faces = (
        _faces(grid, 0, lambda x, y, z: 2.0 * x),
        _faces(grid, 1, lambda x, y, z: -3.0 * y),
        _faces(grid, 2, lambda x, y, z: 0.5 * z),
    )
    result = np.asarray(divergence(faces).data)
    assert np.max(np.abs(result - (2.0 - 3.0 + 0.5))) < 1e-12


UNIFORM = Grid(uniform_faces(6, 0.0, 3.0), uniform_faces(8, -1.0, 1.0), uniform_faces(5, -0.5, 0.5))


def _quadratic_conditions(grid, axis):
    faces = np.asarray(grid.faces[axis])
    return tuple(
        BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
        if position == axis
        else WALL
        for position in range(3)
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_laplacian_of_a_quadratic_is_exact_away_from_the_walls(axis):
    # One axis at a time keeps the wall value constant along the wall, which a
    # scalar Dirichlet condition can represent exactly.
    grid = UNIFORM
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    interior = result[(slice(None),) * axis + (slice(1, -1),)]
    assert np.max(np.abs(interior - 2.0)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_two_point_wall_flux_leaves_its_known_defect_in_the_wall_cell(axis):
    """Pin the accuracy of the symmetric two-point Dirichlet flux at the wall.

    The wall face carries ``(p_0 - g) / (dx_0 / 2)``, a difference centred a
    quarter cell inside the wall, so for a quadratic the wall cell's Laplacian
    returns 1.5 rather than 2. Keeping that stencil is what makes the operator
    symmetric and conservative; a three-point wall stencil would raise the wall
    order at the cost of that symmetry and must change this test deliberately.
    """
    grid = UNIFORM
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    for wall in (0, -1):
        assert np.max(np.abs(result[(slice(None),) * axis + (wall,)] - 1.5)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1])
def test_stretched_truncation_error_is_bounded_by_the_mesh_nonuniformity(axis):
    """A stretched two-point gradient is centred between cell centres, not on the face.

    The resulting Laplacian truncation error is first order in the spacing
    change, which is expected: solution order on stretched meshes is a
    manufactured-solution question and is verified where that study lives.
    """
    grid = STRETCHED
    field = _cells(grid, lambda x, y, z: (x, y, z)[axis] ** 2)
    result = np.asarray(laplacian(field, _quadratic_conditions(grid, axis)).data)
    interior = result[(slice(None),) * axis + (slice(1, -1),)]
    widths = np.asarray(grid.widths[axis])
    bound = np.max(np.abs(np.diff(widths))) / np.min(widths)
    assert np.max(np.abs(interior - 2.0)) <= 2.0 * bound


def test_gradient_and_divergence_are_exact_discrete_adjoints():
    grid, key = STRETCHED, jax.random.PRNGKey(0)
    pressure_key, flux_key = jax.random.split(key)
    pressure = Field(jax.random.normal(pressure_key, grid.shape, dtype=jnp.float64), (CENTER,) * 3, grid)
    faces, keys = [], jax.random.split(flux_key, 3)
    for axis in range(3):
        data = jax.random.normal(keys[axis], grid.face_shape(axis), dtype=jnp.float64)
        # A wall-impermeable flux removes the boundary term from the identity.
        data = data.at[(slice(None),) * axis + (0,)].set(0.0)
        data = data.at[(slice(None),) * axis + (-1,)].set(0.0)
        offset = tuple(FACE if position == axis else CENTER for position in range(3))
        faces.append(Field(data, offset, grid))

    left = cell_inner_product(pressure, divergence(tuple(faces)))
    right = sum(
        face_inner_product(faces[axis], face_gradient(pressure, axis, WALL), axis, WALL) for axis in range(3)
    )
    assert float(jnp.abs(left + right)) <= 1e-14 * float(jnp.abs(left))


def test_periodic_gradient_wraps_and_its_divergence_sums_to_zero():
    grid = Grid(uniform_faces(8, 0.0, 2.0 * np.pi), uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0))
    periodic = BoundaryCondition(PERIODIC)
    field = _cells(grid, lambda x, y, z: jnp.sin(x) + 0.0 * (y + z))
    gradient = face_gradient(field, 0, periodic)
    values = np.asarray(gradient.data)
    assert np.allclose(values[0], values[-1], atol=1e-15)
    faces = (gradient, _faces(grid, 1, lambda x, y, z: 0.0 * x), _faces(grid, 2, lambda x, y, z: 0.0 * x))
    volumes = grid.cell_volumes()
    assert abs(float(np.sum(volumes * np.asarray(divergence(faces).data)))) < 1e-13


def test_face_interpolation_is_exact_for_a_linear_field():
    grid, axis = STRETCHED, 1
    field = _cells(grid, lambda x, y, z: 3.0 * y - 1.0)
    faces = np.asarray(grid.faces[axis])
    condition = BoundaryCondition(DIRICHLET, lower=3.0 * faces[0] - 1.0, upper=3.0 * faces[-1] - 1.0)
    interpolated = np.asarray(face_interpolate(field, axis, condition).data)
    expected = 3.0 * faces - 1.0
    assert np.max(np.abs(interpolated - expected.reshape(1, -1, 1))) < 1e-13


def test_neumann_padding_reproduces_a_prescribed_wall_gradient():
    grid, axis, flux = STRETCHED, 0, 0.75
    field = _cells(grid, lambda x, y, z: 0.0 * x)
    condition = BoundaryCondition(NEUMANN, lower=flux, upper=flux)
    gradient = np.asarray(face_gradient(field, axis, condition).data)
    assert np.allclose(gradient[0], flux, atol=1e-14)
    assert np.allclose(gradient[-1], flux, atol=1e-14)


def test_dirichlet_padding_places_the_prescribed_value_on_the_wall():
    grid, axis = STRETCHED, 2
    field = _cells(grid, lambda x, y, z: 1.0 + 0.0 * z)
    condition = BoundaryCondition(DIRICHLET, lower=-2.0, upper=4.0)
    padded = np.asarray(pad(field.data, axis, condition, grid=grid))
    assert np.allclose(0.5 * (padded[:, :, 0] + padded[:, :, 1]), -2.0, atol=1e-15)
    assert np.allclose(0.5 * (padded[:, :, -1] + padded[:, :, -2]), 4.0, atol=1e-15)


def test_periodic_padding_wraps_both_ends():
    data = jnp.arange(24.0).reshape(2, 3, 4)
    padded = np.asarray(pad(data, 2, BoundaryCondition(PERIODIC)))
    assert np.array_equal(padded[:, :, 0], np.asarray(data)[:, :, -1])
    assert np.array_equal(padded[:, :, -1], np.asarray(data)[:, :, 0])


def test_face_distances_span_the_axis_and_wrap_when_periodic():
    grid, axis = STRETCHED, 0
    widths = np.asarray(grid.widths[axis])
    wall = face_distances(grid, axis, WALL)
    assert wall.size == grid.shape[axis] + 1
    assert wall[0] == pytest.approx(widths[0]) and wall[-1] == pytest.approx(widths[-1])
    wrapped = face_distances(grid, axis, BoundaryCondition(PERIODIC))
    assert wrapped[0] == wrapped[-1] == pytest.approx(0.5 * (widths[0] + widths[-1]))


def test_operators_preserve_single_precision():
    grid = STRETCHED
    field = Field(jnp.ones(grid.shape, dtype=jnp.float32), (CENTER,) * 3, grid)
    gradient = face_gradient(field, 0, WALL)
    assert gradient.dtype == jnp.float32
    assert face_interpolate(field, 0, WALL).dtype == jnp.float32


def test_operators_compose_under_jit_and_differentiation():
    grid = STRETCHED
    conditions = (WALL, WALL, WALL)

    def energy(values):
        field = Field(values, (CENTER,) * 3, grid)
        return jnp.sum(np.asarray(grid.cell_volumes()) * laplacian(field, conditions).data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(1), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    assert gradient.shape == grid.shape and np.all(np.isfinite(np.asarray(gradient)))


def test_boundary_conditions_validate_their_inputs():
    with pytest.raises(ValueError, match="unknown boundary kind"):
        BoundaryCondition("sponge")
    with pytest.raises(ValueError, match="do not take prescribed values"):
        BoundaryCondition(PERIODIC, lower=1.0)
    with pytest.raises(ValueError, match="requires the grid"):
        pad(jnp.zeros((2, 2, 2)), 0, BoundaryCondition(NEUMANN, lower=1.0))
    with pytest.raises(ValueError, match="three-dimensional"):
        pad(jnp.zeros((2, 2)), 0, WALL)
    with pytest.raises(ValueError, match="out of range"):
        pad(jnp.zeros((2, 2, 2)), 3, WALL)


def test_operators_reject_mismatched_positions_and_grids():
    grid = STRETCHED
    cell = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="expected a cell-centred field"):
        face_gradient(face, 0, WALL)
    with pytest.raises(ValueError, match="expected a field on faces"):
        divergence((cell, cell, cell))
    with pytest.raises(ValueError, match="does not match grid"):
        face_gradient(Field(jnp.zeros((2, 2, 2)), (CENTER,) * 3, grid), 0, WALL)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        laplacian(cell, (WALL, WALL))
    other = Grid(uniform_faces(6, 0.0, 3.0), *grid.faces[1:])
    with pytest.raises(ValueError, match="share one grid"):
        cell_inner_product(cell, Field(jnp.zeros(other.shape), (CENTER,) * 3, other))
    with pytest.raises(ValueError, match="share one grid"):
        divergence((face, Field(jnp.zeros(other.face_shape(1)), (CENTER, FACE, CENTER), other), face))


def test_homogeneous_neumann_padding_needs_no_grid():
    data = jnp.arange(8.0).reshape(2, 2, 2)
    padded = np.asarray(pad(data, 1, WALL))
    assert np.array_equal(padded[:, 0], np.asarray(data)[:, 0])
    assert np.array_equal(padded[:, -1], np.asarray(data)[:, -1])


def test_face_inner_product_rejects_mismatched_grids_and_shapes():
    grid = STRETCHED
    other = Grid(uniform_faces(6, 0.0, 3.0), *grid.faces[1:])
    face = Field(jnp.zeros(grid.face_shape(0)), (FACE, CENTER, CENTER), grid)
    with pytest.raises(ValueError, match="share one grid"):
        face_inner_product(
            face, Field(jnp.zeros(other.face_shape(0)), (FACE, CENTER, CENTER), other), 0, WALL
        )
    with pytest.raises(ValueError, match="does not match grid"):
        face_inner_product(face, Field(jnp.zeros((2, 2, 2)), (FACE, CENTER, CENTER), grid), 0, WALL)


def _polar(radial: int, azimuthal: int) -> Grid:
    return Grid(
        uniform_faces(radial, 0.0, 1.0),
        uniform_faces(azimuthal, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )


@pytest.mark.parametrize("grid", [STRETCHED, _polar(6, 8)], ids=["stretched", "polar"])
def test_the_stencils_capture_no_cell_sized_metric(grid):
    """Volumes, areas and half-cell weights are broadcast factors, not constants captured per call.

    Captured, they were 1.0 of the 1.1 GiB of constants in a compiled 64^3 steady duct solve.
    """
    grid = Grid(*grid.faces[:2], uniform_faces(3, 0.0, 1.0), geometry=grid.geometry)
    faces = tuple(_faces(grid, axis, lambda x, y, z: x + y * z) for axis in range(3))
    cells = _cells(grid, lambda x, y, z: x * y + z)

    def stencils(faces, cells):
        return (
            divergence(faces),
            cell_inner_product(cells, cells),
            *(face_average_adjoint(face, axis, WRAP) for axis, face in enumerate(faces)),
            *(face_inner_product(face, face, axis, WRAP) for axis, face in enumerate(faces)),
        )

    captured = jax.make_jaxpr(stencils)(faces, cells).consts
    assert max(np.size(value) for value in captured) < np.prod(grid.shape)


def _polar_cells(grid: Grid, function) -> Field:
    radius, angle, axial = (np.asarray(values) for values in grid.centers)
    r, theta, z = np.meshgrid(radius, angle, axial, indexing="ij")
    return Field(jnp.asarray(function(r, theta)), (CENTER, CENTER, CENTER), grid)


def test_the_flux_form_laplacian_is_exact_on_a_paraboloid():
    """`1 - r^2` has Laplacian `-4` everywhere, and the flux form reproduces it exactly.

    Everywhere except the wall cell: the two-point wall flux is first order, and
    a cell whose volume is also first order therefore carries an order-one error
    in the Laplacian. That is the same closure the Cartesian operator uses, and
    it is why the Poisson *solution* stays second order while this pointwise
    reading of the operator does not.
    """
    grid = _polar(16, 32)
    field = _polar_cells(grid, lambda r, theta: 1.0 - r**2)
    values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
    assert np.max(np.abs(values[:-1] + 4.0)) < 1e-12
    assert abs(values[-1, 0, 0] + 4.0) > 0.1


def test_the_polar_laplacian_is_second_order_away_from_the_axis():
    """`r cos(theta)` is harmonic; the radial and azimuthal terms cancel at order `1/r`."""
    errors = []
    for count in (16, 32, 64):
        grid = _polar(count, 2 * count)
        field = _polar_cells(grid, lambda r, theta: r * np.cos(theta))
        values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
        radius = np.asarray(grid.centers[0])
        inside = (radius > 0.2) & (radius < 0.95)
        errors.append(float(np.max(np.abs(values[inside]))))
    orders = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert min(orders) > 1.8, orders
    # Against the axis the two terms are each of order 1/r, so their cancellation
    # loses an order. A pipe resolves its wall layers, not its centre.
    grid = _polar(32, 64)
    field = _polar_cells(grid, lambda r, theta: r * np.cos(theta))
    values = np.asarray(laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
    assert np.max(np.abs(values[0])) > 2.0 * errors[1]


def test_the_staggered_laplacian_carries_the_polar_metric():
    """It is a flux balance, so `(1/r) d/dr (r d/dr)` comes out of the same stencil."""
    grid = _polar(16, 32)
    field = _polar_cells(grid, lambda r, theta: 1.0 - r**2)
    values = np.asarray(staggered_laplacian(field, (BoundaryCondition(DIRICHLET), WRAP, WRAP)).data)
    assert np.max(np.abs(values[:-1] + 4.0)) < 1e-12


def test_the_fast_diagonalization_assembly_refuses_a_polar_grid():
    """The `1/r^2` azimuthal term does not separate; `fast_diagonal_polar_poisson` does it instead."""
    from lmhdx.poisson import assemble_axis_laplacian

    with pytest.raises(ValueError, match="fast diagonalization assumes"):
        assemble_axis_laplacian(_polar(4, 8), 0, BoundaryCondition(DIRICHLET))


# Conservative momentum transport: conservation, order on a stretched mesh, boundedness.

PERIODIC_AXIS = BoundaryCondition(PERIODIC)
NO_SLIP = BoundaryCondition(DIRICHLET)
BOX = (PERIODIC_AXIS,) * 3


def _periodic_faces(count: int, amplitude: float = 0.0) -> np.ndarray:
    """Faces on ``[0, 2 pi]``; a nonzero amplitude stretches them smoothly and periodically."""
    fraction = np.linspace(0.0, 1.0, count + 1)
    return 2.0 * np.pi * (fraction + amplitude * np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi))


def _coordinates(grid: Grid, component: int) -> tuple[np.ndarray, ...]:
    axes = [grid.faces[axis] if axis == component else grid.centers[axis] for axis in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def _taylor_green(grid: Grid, component: int) -> tuple[Field, np.ndarray]:
    """A divergence-free field and the exact ``div(u u)`` of that field.

    With ``u = (sin x cos y cos z, -cos x sin y cos z, 0)`` the divergence
    vanishes and the flux divergence reduces to ``(sin x cos x cos^2 z,
    sin y cos y cos^2 z, 0)``, derived by hand rather than from any operator here.
    """
    x, y, z = _coordinates(grid, component)
    values = (
        np.sin(x) * np.cos(y) * np.cos(z),
        -np.cos(x) * np.sin(y) * np.cos(z),
        np.zeros_like(x),
    )
    exact = (
        np.sin(x) * np.cos(x) * np.cos(z) ** 2,
        np.sin(y) * np.cos(y) * np.cos(z) ** 2,
        np.zeros_like(x),
    )
    field = Field(jnp.asarray(values[component]), velocity_offset(component), grid)
    return field, exact[component]


def _uniform(grid: Grid, values: tuple[float, float, float]) -> tuple[Field, Field, Field]:
    return tuple(
        Field(
            jnp.full(grid.offset_shape(velocity_offset(component)), value), velocity_offset(component), grid
        )
        for component, value in enumerate(values)
    )


def _error(count: int, amplitude: float) -> float:
    faces = _periodic_faces(count, amplitude)
    grid = Grid(faces, faces, faces)
    fields, exact = zip(*(_taylor_green(grid, component) for component in range(3)))
    transported = momentum_advection(fields, BOX, limited=False)
    return max(
        float(np.sqrt(np.mean((np.asarray(transported[component].data) - exact[component]) ** 2)))
        for component in range(2)
    )


@pytest.mark.parametrize("limited", [False, True])
def test_a_uniform_flow_is_not_transported(limited):
    """Galilean invariance: a constant velocity has no flux divergence, on any mesh."""
    faces = _periodic_faces(8, 0.4)
    grid = Grid(faces, faces, faces)
    transported = momentum_advection(_uniform(grid, (1.3, -0.7, 0.2)), BOX, limited=limited)
    for field in transported:
        assert float(jnp.max(jnp.abs(field.data))) < 1e-13


@pytest.mark.parametrize("limited", [False, True])
def test_the_flux_form_conserves_momentum(limited):
    """Telescoping fluxes: the transported momentum of a periodic box sums to zero."""
    faces = _periodic_faces(10, 0.3)
    grid = Grid(faces, faces, faces)
    fields = tuple(_taylor_green(grid, component)[0] for component in range(3))
    transported = momentum_advection(fields, BOX, limited=limited)
    for component, field in enumerate(transported):
        interior = np.asarray(field.data)[(slice(None),) * component + (slice(None, -1),)]
        scale = float(np.max(np.abs(interior))) + 1.0
        assert abs(float(np.sum(interior))) < 1e-11 * scale * interior.size


@pytest.mark.parametrize("amplitude", [0.0, 0.5])
def test_transport_is_second_order_including_on_a_stretched_mesh(amplitude):
    coarse, medium, fine = (_error(count, amplitude) for count in (16, 32, 64))
    orders = (np.log2(coarse / medium), np.log2(medium / fine))
    assert min(orders) > 1.9, orders


def _carried_in_y(grid: Grid, profile: np.ndarray) -> tuple[Field, Field, Field]:
    """Transport ``profile(y)`` as ``u_x`` on a uniform ``u_y = 1``.

    The state is divergence free, so this is linear scalar advection along ``y``
    and the limiter has to satisfy the usual bound on it.
    """
    return (
        Field(jnp.asarray(profile), velocity_offset(0), grid),
        Field(jnp.ones(grid.offset_shape(velocity_offset(1))), velocity_offset(1), grid),
        Field(jnp.zeros(grid.offset_shape(velocity_offset(2))), velocity_offset(2), grid),
    )


def test_the_limiter_reduces_to_the_central_flux_on_a_uniform_gradient():
    """Where successive gradients agree the van Leer weight is one, so nothing is clipped."""
    faces = _periodic_faces(16, 0.0)
    grid = Grid(faces, faces, faces)
    _, y, _ = _coordinates(grid, 0)
    fields = _carried_in_y(grid, 0.3 * y)
    central = np.asarray(momentum_advection(fields, BOX, limited=False)[0].data)
    limited = np.asarray(momentum_advection(fields, BOX, limited=True)[0].data)
    # The ramp is linear everywhere except across the periodic wrap; compare away from it.
    interior = (slice(None), slice(2, -2), slice(None))
    assert np.max(np.abs(central[interior] - limited[interior])) < 1e-14


def test_the_limiter_does_not_overshoot_a_step():
    """A discontinuity stays inside its own bounds; the central flux does not."""
    faces = _periodic_faces(32, 0.0)
    grid = Grid(faces, faces, faces)
    _, y, _ = _coordinates(grid, 0)
    profile = np.where((y > 2.0) & (y < 4.0), 1.0, 0.0)
    fields = _carried_in_y(grid, profile)
    dt = 0.5 * float(np.min(grid.widths[1]))
    stepped = {
        name: profile - dt * np.asarray(momentum_advection(fields, BOX, limited=name)[0].data)
        for name in (False, True)
    }
    assert np.max(stepped[True]) <= 1.0 + 1e-12
    assert np.min(stepped[True]) >= -1e-12
    assert np.max(stepped[False]) > 1.0 + 1e-3


def test_the_step_limit_reports_the_convective_bound():
    faces = _periodic_faces(8, 0.0)
    grid = Grid(faces, faces, faces)
    limit = advective_step_limit(_uniform(grid, (2.0, 1.0, 0.0)))
    widths = [float(np.min(grid.widths[axis])) for axis in range(3)]
    assert float(limit) == pytest.approx(1.0 / (2.0 / widths[0] + 1.0 / widths[1]))


def test_advection_needs_one_condition_per_axis():
    faces = _periodic_faces(4, 0.0)
    grid = Grid(faces, faces, faces)
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        momentum_advection(_uniform(grid, (1.0, 0.0, 0.0)), (PERIODIC_AXIS, PERIODIC_AXIS))


def _channel(advection: str) -> ChannelProblem:
    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(8, -1.0, 1.0), uniform_faces(8, -1.0, 1.0))
    return ChannelProblem(
        grid=grid,
        conditions=(PERIODIC_AXIS, NO_SLIP, NO_SLIP),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
        advection=advection,
    )


def test_the_channel_rejects_an_unknown_advection_choice():
    with pytest.raises(ValueError, match="advection must be one of"):
        _channel("upwind")


@pytest.mark.parametrize("advection", ["central", "limited"])
def test_a_step_with_transport_stays_divergence_free(advection):
    """Transport enters the predictor, so the projection still has to clean up after it."""
    from lmhdx.ops import divergence

    problem = _channel(advection)
    velocity = zero_velocity(problem)
    for _ in range(4):
        velocity, _, _ = step(velocity, problem)
    assert float(jnp.max(jnp.abs(divergence(velocity).data))) < 1e-11
    assert np.all(np.isfinite(np.asarray(velocity[0].data)))


def test_transport_off_reproduces_the_stokes_step():
    problem = _channel("off")
    stokes, _, _ = step(zero_velocity(problem), problem)
    transported, _, _ = step(zero_velocity(problem), _channel("central"))
    # From rest the first step has nothing to transport, so the two agree exactly.
    for left, right in zip(stokes, transported, strict=True):
        assert float(jnp.max(jnp.abs(left.data - right.data))) == 0.0


# The Laplacian of a staggered field, which a marker-and-cell velocity needs.

LAPLACIAN_BOX = Grid(uniform_faces(8, 0.0, 2.0), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -0.5, 0.5))
STRETCHED_LAPLACIAN_BOX = Grid(
    geometric_faces(8, 0.0, 2.0, 1.2), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -0.5, 0.5)
)


def _positions(grid: Grid, offset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = []
    for axis in range(3):
        values = grid.faces[axis] if offset[axis] == FACE else grid.centers[axis]
        coordinates.append(np.asarray(values))
    x, y, z = coordinates
    return x[:, None, None], y[None, :, None], z[None, None, :]


def _field(grid: Grid, offset, function) -> Field:
    x, y, z = _positions(grid, offset)
    values = np.broadcast_to(function(x, y, z), grid.offset_shape(offset))
    return Field(jnp.asarray(values, dtype=jnp.float64), offset, grid)


AXIAL = (FACE, CENTER, CENTER)


def test_linear_field_has_zero_laplacian_everywhere_it_is_defined():
    grid = STRETCHED_LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: 2.0 * x - 3.0 * y + z)
    conditions = (WALL, BoundaryCondition(DIRICHLET), WALL)
    result = np.asarray(staggered_laplacian(field, conditions).data)
    # Cell-centred axes carry a wall condition, so only the interior is defined.
    assert np.max(np.abs(result[1:-1, 1:-1, 1:-1])) < 1e-12


def test_quadratic_along_the_face_axis_is_exact_on_interior_faces():
    grid = LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WALL, WALL, WALL)).data)
    assert np.max(np.abs(result[1:-1] - 2.0)) < 1e-12


def test_boundary_faces_of_a_walled_face_axis_are_returned_as_zero():
    """Their value is prescribed, not evolved, so no one-sided stencil is invented."""
    grid = LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WALL, WALL, WALL)).data)
    assert np.all(result[0] == 0.0)
    assert np.all(result[-1] == 0.0)


def test_periodic_face_axis_wraps_and_matches_the_analytic_second_derivative():
    grid = Grid(uniform_faces(32, 0.0, 2.0 * np.pi), uniform_faces(3, 0.0, 1.0), uniform_faces(3, 0.0, 1.0))
    field = _field(grid, AXIAL, lambda x, y, z: np.sin(x) + 0.0 * (y + z))
    result = np.asarray(staggered_laplacian(field, (WRAP, WALL, WALL)).data)
    expected = -np.asarray(np.sin(grid.faces[0]))[:, None, None]
    # Second-order accuracy on 32 cells over a full period.
    assert np.max(np.abs(result - expected)) < 5e-3
    assert np.allclose(result[0], result[-1], atol=1e-14)


def test_quadratic_along_a_cell_centred_axis_is_exact_in_the_interior():
    grid = LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: y**2 + 0.0 * (x + z))
    faces = np.asarray(grid.faces[1])
    condition = BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
    result = np.asarray(staggered_laplacian(field, (WALL, condition, WALL)).data)
    assert np.max(np.abs(result[1:-1, 1:-1, :] - 2.0)) < 1e-12


def test_all_three_axes_add_up():
    grid = LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: x**2 + y**2 + z**2)
    conditions = []
    for axis in range(3):
        faces = np.asarray(grid.faces[axis])
        conditions.append(
            BoundaryCondition(DIRICHLET, lower=float(faces[0]) ** 2, upper=float(faces[-1]) ** 2)
        )
    result = np.asarray(staggered_laplacian(field, tuple(conditions)).data)
    assert np.max(np.abs(result[1:-1, 1:-1, 1:-1] - 6.0)) < 1e-12


@pytest.mark.parametrize("offset", [(FACE, CENTER, CENTER), (CENTER, FACE, CENTER), (CENTER, CENTER, FACE)])
def test_every_velocity_position_is_supported(offset):
    grid = LAPLACIAN_BOX
    field = _field(grid, offset, lambda x, y, z: x + y + z)
    result = staggered_laplacian(field, (WALL, WALL, WALL))
    assert result.offset == offset
    assert result.shape == grid.offset_shape(offset)


def test_the_interior_operator_is_symmetric():
    """An implicit viscous solve needs symmetry; this checks it on the free faces."""
    grid = LAPLACIAN_BOX
    conditions = (WALL, BoundaryCondition(DIRICHLET), BoundaryCondition(DIRICHLET))
    shape = grid.offset_shape(AXIAL)
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1] = True
    indices = np.argwhere(interior)
    operator = np.zeros((len(indices), len(indices)))
    for column, position in enumerate(indices):
        unit = np.zeros(shape)
        unit[tuple(position)] = 1.0
        applied = np.asarray(staggered_laplacian(Field(jnp.asarray(unit), AXIAL, grid), conditions).data)
        operator[:, column] = applied[interior]
    asymmetry = np.max(np.abs(operator - operator.T)) / np.max(np.abs(operator))
    assert asymmetry < 1e-12


def test_the_staggered_laplacian_differentiates_and_jits():
    grid = LAPLACIAN_BOX
    conditions = (WALL, WALL, WALL)

    def energy(values):
        field = Field(values, AXIAL, grid)
        return jnp.sum(staggered_laplacian(field, conditions).data ** 2)

    values = jax.random.normal(jax.random.PRNGKey(0), grid.offset_shape(AXIAL), dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    step = 1.0e-6
    direction = jax.random.normal(jax.random.PRNGKey(1), grid.offset_shape(AXIAL), dtype=jnp.float64)
    difference = (energy(values + step * direction) - energy(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_the_staggered_laplacian_validates_its_inputs():
    grid = LAPLACIAN_BOX
    field = _field(grid, AXIAL, lambda x, y, z: x + 0.0 * (y + z))
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        staggered_laplacian(field, (WALL, WALL))
    with pytest.raises(ValueError, match="does not match its offset"):
        staggered_laplacian(Field(jnp.zeros((2, 2, 2)), AXIAL, grid), (WALL, WALL, WALL))


# Heat conduction (plan step 5.1b, ADR 0008): thermal conditions, the boundary lift and the implicit inverse.

CELL = (CENTER, CENTER, CENTER)


def _thermal(grid: Grid, conditions, *, dt: float = 1.0e-2, k: float = 2.0) -> ThermalProblem:
    flow = ChannelProblem(
        grid, tuple(WRAP if c.is_periodic else WALL for c in conditions), density=1.5, dt=dt
    )
    return ThermalProblem(
        flow, conditions, specific_heat=3.0, thermal_conductivity=k, reference_temperature=0.0
    )


def _advance_heat(problem: ThermalProblem, temperature: Field, heating: Field, steps: int = 1) -> Field:
    factorization, lift = problem.diffusion_factorization(), problem.boundary_lift()
    for _ in range(steps):  # (I - dt alpha L_0) T' = T + dt (q / (rho_0 c_p) + alpha b)
        rhs = temperature.data + float(problem.flow.dt) * (
            heating.data / problem.heat_capacity + problem.diffusivity * lift.data
        )
        temperature = factorization.solve(temperature.replace_data(rhs))
    return temperature


def _mixed_walls(rng) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
    """Mixed kinds, scalar and array data on two wall-bounded axes of STRETCHED, so the corners take both."""
    return (
        thermal_boundary(
            "temperature", rng.standard_normal((8, 5)), 0.3, upper_kind="heat_flux", conductivity=2.0
        ),
        thermal_boundary(
            "heat_flux", rng.standard_normal((6, 5)), rng.standard_normal((6, 5)), conductivity=2.0
        ),
        WRAP,
    )


def test_neumann_data_is_the_derivative_along_the_axis_at_both_ends():
    """The ghost arithmetic thermal_boundary relies on: ``first - d h`` below and ``last + d h`` above."""
    data = jnp.asarray(np.random.default_rng(1).standard_normal(STRETCHED.shape))
    padded = np.asarray(pad(data, 1, BoundaryCondition(NEUMANN, 0.7, -1.3), grid=STRETCHED))
    widths = np.asarray(STRETCHED.widths[1])
    assert np.allclose(padded[:, 0], data[:, 0] - 0.7 * widths[0], rtol=0.0, atol=1e-14)
    assert np.allclose(padded[:, -1], data[:, -1] - 1.3 * widths[-1], rtol=0.0, atol=1e-14)


def test_thermal_boundary_turns_outward_heat_flux_into_the_axis_derivative():
    assert thermal_boundary("heat_flux", 2.0, 3.0, conductivity=4.0) == BoundaryCondition(NEUMANN, 0.5, -0.75)
    mixed = thermal_boundary("temperature", 300.0, 5.0, upper_kind="heat_flux", conductivity=2.0)
    assert mixed.kinds == (DIRICHLET, NEUMANN) and (mixed.lower, mixed.upper) == (300.0, -2.5)
    same = thermal_boundary("temperature", 1.0, 2.0, upper_kind="temperature")
    assert same == BoundaryCondition(DIRICHLET, 1.0, 2.0) and thermal_boundary("periodic") == WRAP
    profile = np.array([[1.0, 2.0], [3.0, 4.0]])
    condition = thermal_boundary("heat_flux", profile, conductivity=2.0)
    profile[0, 0] = 99.0
    assert np.array_equal(condition.lower, [[0.5, 1.0], [1.5, 2.0]])
    for bad in (
        dict(kind="adiabatic"),
        dict(kind="heat_flux"),
        dict(kind="heat_flux", conductivity=0.0),
        dict(kind="temperature", lower=np.inf),
        dict(kind="periodic", lower=1.0),
    ):
        with pytest.raises(ValueError):
            thermal_boundary(**bad)


def test_a_thermal_problem_checks_its_inputs():
    grid = Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(3, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    walls = (WRAP, thermal_boundary("temperature", 1.0, 2.0), thermal_boundary("heat_flux", conductivity=1.0))
    problem = ThermalProblem(ChannelProblem(grid, (WRAP, WALL, WALL)), walls, 2.0, 0.5, 300.0)
    assert problem.heat_capacity == 2.0 and problem.diffusivity == 0.25
    tangential = thermal_boundary("temperature", np.ones((2, 3)), np.zeros((2, 3)))
    assert dataclasses.replace(problem, thermal_conditions=(*walls[:2], tangential)).flow is problem.flow
    for change in (
        dict(thermal_conditions=(WALL, *walls[1:])),
        dict(thermal_conditions=(*walls[:2], thermal_boundary("temperature", np.ones((3, 2))))),
        dict(thermal_conditions=walls[:2]),
        dict(specific_heat=0.0),
        dict(thermal_conductivity=np.nan),
        dict(reference_temperature=np.nan),
        dict(expansion_coefficient=-1.0e-4),
        dict(gravity=(0.0, -9.8)),
        dict(gravity=(0.0, 0.0, np.inf)),
        dict(flow=dataclasses.replace(problem.flow, dt=np.inf)),
        dict(flow=ChannelProblem(_polar(4, 6), (WALL, WRAP, WRAP))),
        dict(
            flow=ChannelProblem(grid, (BoundaryCondition(DIRICHLET, upper_kind=NEUMANN), WALL, WALL)),
            thermal_conditions=(thermal_boundary("heat_flux", conductivity=0.5), *walls[1:]),
        ),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(problem, **change)
    for change in (dict(flow=None), dict(thermal_conditions=(WRAP, walls[1], "insulated"))):
        with pytest.raises(TypeError):
            dataclasses.replace(problem, **change)


def test_the_boundary_lift_makes_the_laplacian_affine():
    """``L(T) = L_0(T) + L(0)`` for any temperature, to rounding (1e-12 of ``L(T)``)."""
    rng = np.random.default_rng(3)
    conditions = _mixed_walls(rng)
    temperature = Field(jnp.asarray(rng.standard_normal(STRETCHED.shape)), CELL, STRETCHED)
    full = laplacian(temperature, conditions).data
    homogeneous = laplacian(temperature, tuple(c.homogeneous() for c in conditions)).data
    lift = _thermal(STRETCHED, conditions).boundary_lift().data
    assert float(jnp.max(jnp.abs(full - homogeneous - lift))) <= 1e-12 * float(jnp.max(jnp.abs(full)))


def test_the_diffusion_inverse_solves_the_production_laplacian():
    """Its residual under ``ops.laplacian`` is within 1e-10 of the right-hand side, not just its own operator's."""
    rng = np.random.default_rng(4)
    conditions = tuple(c.homogeneous() for c in _mixed_walls(rng))
    problem = _thermal(STRETCHED, conditions, dt=0.05)
    rhs = Field(jnp.asarray(rng.standard_normal(STRETCHED.shape)), CELL, STRETCHED)
    solution = problem.diffusion_factorization().solve(rhs)
    operator = laplacian(solution, conditions).data
    residual = solution.data - problem.flow.dt * problem.diffusivity * operator - rhs.data
    assert float(jnp.max(jnp.abs(residual))) <= 1e-10 * float(jnp.max(jnp.abs(rhs.data)))
    staggered = staggered_laplacian(solution, conditions).data
    assert float(jnp.max(jnp.abs(staggered - operator))) <= 1e-13 * float(jnp.max(jnp.abs(operator)))


def test_insulated_conduction_keeps_stored_heat_and_heats_uniformly():
    """A uniform heat rate raises every cell by ``dt q / (rho c_p)`` a step; no heat leaves or is removed."""
    insulated = thermal_boundary("heat_flux", conductivity=2.0)
    problem = _thermal(STRETCHED, (insulated, insulated, WRAP))
    start = Field(jnp.full(STRETCHED.shape, 300.0), CELL, STRETCHED)
    heated = _advance_heat(problem, start, start.replace_data(jnp.full(STRETCHED.shape, 4.0)), steps=3)
    rise = 3 * problem.flow.dt * 4.0 / problem.heat_capacity
    # Cells hold about 300, so rounding is relative to the temperature, not to the rise.
    assert float(jnp.max(jnp.abs(heated.data - (300.0 + rise)))) <= 1e-13 * 300.0
    noise = start.replace_data(jnp.asarray(np.random.default_rng(5).standard_normal(STRETCHED.shape)))
    spread = _advance_heat(problem, noise, noise.replace_data(jnp.zeros(STRETCHED.shape)), steps=3)
    before, after = (
        float(cell_inner_product(field, noise.replace_data(jnp.ones(STRETCHED.shape))))
        for field in (noise, spread)
    )
    assert abs(after - before) <= 1e-13 * float(np.sum(STRETCHED.cell_volumes()))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_outward_heat_flux_cools_through_both_ends(sign):
    """A positive outward flux takes ``q A dt`` out through each end and cools beside both; negative heats."""
    lower, upper = 2.0 * sign, 5.0 * sign
    conditions = (
        thermal_boundary("heat_flux", conductivity=2.0),
        thermal_boundary("heat_flux", lower, upper, conductivity=2.0),
        WRAP,
    )
    problem = _thermal(STRETCHED, conditions)
    zero = Field(jnp.zeros(STRETCHED.shape), CELL, STRETCHED)
    after = np.asarray(_advance_heat(problem, zero, zero).data)
    stored = problem.heat_capacity * float(np.sum(STRETCHED.cell_volumes() * after))
    area = float(np.sum(STRETCHED.face_areas(1)[:, 0, :]))
    assert stored == pytest.approx(-problem.flow.dt * (lower + upper) * area, rel=1e-12)
    profile = sign * after.mean(axis=(0, 2))
    middle = len(profile) // 2
    assert profile[0] < profile[middle] and profile[-1] < profile[middle]


def test_a_periodic_mode_decays_at_its_discrete_rate():
    """``cos(2 pi m x / L)`` is an eigenvector: one step divides it by ``1 + dt alpha (4/h^2) sin^2(pi m h / L)``."""
    cells, length, mode = 16, 2.0, 3
    grid = Grid(uniform_faces(cells, 0.0, length), uniform_faces(1, -1.0, 1.0), uniform_faces(1, -1.0, 1.0))
    insulated = thermal_boundary("heat_flux", conductivity=2.0)
    problem = _thermal(grid, (WRAP, insulated, insulated))
    wave = _cells(grid, lambda x, y, z: np.cos(2.0 * np.pi * mode * x / length) + 0.0 * (y + z))
    after = _advance_heat(problem, wave, wave.replace_data(jnp.zeros(grid.shape)))
    width = length / cells
    rate = (
        problem.flow.dt * problem.diffusivity * (4.0 / width**2) * np.sin(np.pi * mode * width / length) ** 2
    )
    assert float(jnp.max(jnp.abs(after.data - wave.data / (1.0 + rate)))) <= 1e-13


@pytest.mark.parametrize("stretch", [None, 1.3], ids=["uniform", "tanh"])
def test_a_heated_slab_converges_at_second_order(stretch):
    """``k T'' + q = 0`` between two wall temperatures: the parabola, at second order.

    The wall heat flux is exact to rounding on any face distribution. The cell
    balances fix every difference of neighbouring face fluxes, and the flux times
    the centre-to-centre distance sums to the wall-temperature difference: off the
    midpoints the interior terms telescope to ``-(h_N^2 - h_1^2)/8`` and the two
    half-cell wall terms return exactly that, so a uniform source is carried
    exactly, wall fluxes included.
    """
    k, q, low, high = 2.0, 3.0, 1.0, 2.0
    errors = []
    for cells in (8, 16, 32):
        faces = uniform_faces(cells, -1.0, 1.0) if stretch is None else tanh_faces(cells, -1.0, 1.0, stretch)
        grid = Grid(uniform_faces(1, 0.0, 1.0), faces, uniform_faces(1, 0.0, 1.0))
        insulated = thermal_boundary("heat_flux", conductivity=k)
        problem = _thermal(grid, (WRAP, thermal_boundary("temperature", low, high), insulated), dt=1.0e3, k=k)
        zero = Field(jnp.zeros(grid.shape), CELL, grid)
        # dt alpha lambda_min ~ 1e3, so six implicit steps reach the steady state to rounding.
        steady = np.asarray(_advance_heat(problem, zero, zero.replace_data(jnp.full(grid.shape, q)), 6).data)
        y = np.asarray(grid.centers[1])
        exact = low + (high - low) * (y + 1.0) / 2.0 + q / (2.0 * k) * (y + 1.0) * (1.0 - y)
        flux = k * (steady[0, 0, 0] - low) / (0.5 * grid.widths[1][0])
        wall = k * (high - low) / 2.0 + q
        assert abs(flux - wall) <= 1e-11 * wall
        errors.append(np.max(np.abs(steady[0, :, 0] - exact)))
    assert np.log2(errors[1] / errors[2]) > 1.8, errors


# Heat transport (plan step 5.1c): conservative first-order upwind fluxes and the outgoing-flux Courant number.

INSULATED = thermal_boundary("heat_flux", conductivity=2.0)


def _carrier(problem: ThermalProblem, seed: int, *, solenoidal: bool = False) -> tuple[Field, Field, Field]:
    rng = np.random.default_rng(seed)
    noise = tuple(
        f.replace_data(jnp.asarray(rng.standard_normal(f.data.shape))) for f in zero_velocity(problem.flow)
    )
    velocity = enforce_face_constraints(noise, problem.flow)
    return project(velocity, problem.flow)[0] if solenoidal else velocity


def _transport(temperature: Field, velocity, problem: ThermalProblem) -> jnp.ndarray:
    return divergence(_upwind_fluxes(temperature, velocity, problem)).data


def test_the_upwind_flux_takes_the_upstream_cell_and_nothing_through_a_wall():
    """On STRETCHED, periodic along its geometric axis, whose two end cells differ in width.

    The carrier is shifted by one, so its wall faces carry a velocity the walls must not pass on.
    """
    problem = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    temperature = Field(
        jnp.asarray(np.random.default_rng(6).standard_normal(STRETCHED.shape)), CELL, STRETCHED
    )
    velocity = tuple(f.replace_data(f.data + 1.0) for f in _carrier(problem, 7))
    for axis, flux in enumerate(_upwind_fluxes(temperature, velocity, problem)):
        t, u, f = (np.moveaxis(np.asarray(a.data), axis, 0) for a in (temperature, velocity[axis], flux))
        assert np.array_equal(f[1:-1], u[1:-1] * np.where(u[1:-1] >= 0.0, t[:-1], t[1:]))
        if axis:
            assert np.all(u[[0, -1]]) and not np.any(f[[0, -1]])
        else:
            assert np.array_equal(f[0], f[-1]) and np.array_equal(
                f[0], u[0] * np.where(u[0] >= 0.0, t[-1], t[0])
            )


def test_upwind_transport_conserves_heat_and_moves_a_constant_only_by_the_divergence():
    problem = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    velocity = _carrier(problem, 8, solenoidal=True)
    temperature = Field(
        jnp.asarray(np.random.default_rng(9).standard_normal(STRETCHED.shape)), CELL, STRETCHED
    )
    transport = temperature.replace_data(_transport(temperature, velocity, problem))
    ones = temperature.replace_data(jnp.ones(STRETCHED.shape))
    absolute = float(cell_inner_product(transport.replace_data(jnp.abs(transport.data)), ones))
    assert abs(float(cell_inner_product(transport, ones))) <= 1e-13 * absolute
    constant = _transport(temperature.replace_data(jnp.full(STRETCHED.shape, 300.0)), velocity, problem)
    scale = 300.0 * float(jnp.max(outgoing_rate(velocity)))
    assert float(jnp.max(jnp.abs(constant - 300.0 * divergence(velocity).data))) <= 1e-12 * scale


def test_upwind_transport_makes_no_new_extremum_at_courant_0_8():
    problem = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    velocity = _carrier(problem, 10, solenoidal=True)
    start = np.random.default_rng(11).uniform(0.0, 1.0, STRETCHED.shape)
    temperature = Field(jnp.asarray(start), CELL, STRETCHED)
    dt = 0.8 / float(jnp.max(outgoing_rate(velocity)))
    for _ in range(10):
        temperature = temperature.replace_data(
            temperature.data - dt * _transport(temperature, velocity, problem)
        )
    assert (
        start.min() - 1e-12
        <= float(jnp.min(temperature.data))
        <= float(jnp.max(temperature.data))
        <= start.max() + 1e-12
    )


def test_the_van_leer_face_value_is_exact_for_a_linear_temperature_on_a_stretched_grid():
    """``T = 2 + 3 y`` across the tanh-stretched walls: every face whose two neighbouring faces are interior reads
    ``2 + 3 y_f`` from either side, to 1e-12; the physical distances make the slopes exact."""
    from lmhdx.thermal import _limited

    problem = dataclasses.replace(_thermal(STRETCHED, (WRAP, INSULATED, INSULATED)), advection="van_leer")
    y = np.asarray(STRETCHED.centers[1])
    temperature = Field(
        jnp.asarray(np.broadcast_to(2.0 + 3.0 * y[None, :, None], STRETCHED.shape)), CELL, STRETCHED
    )
    padded = pad(temperature.data, 1, problem.thermal_conditions[1], grid=STRETCHED)
    left, right = padded[:, :-1, :], padded[:, 1:, :]
    from_left, from_right = (np.asarray(value) for value in _limited(padded, 1, problem, left, right))
    exact = np.broadcast_to(2.0 + 3.0 * np.asarray(STRETCHED.y_faces)[None, :, None], from_left.shape)
    inner = slice(2, -2)
    assert np.allclose(from_left[:, inner], exact[:, inner], rtol=1e-12, atol=0.0)
    assert np.allclose(from_right[:, inner], exact[:, inner], rtol=1e-12, atol=0.0)


# Stretched as far as the duct pages go: a Hartmann-like layer with no ratio limit, and a periodic axis whose first
# and last widths, side by side at the seam, are 27 to 1.
AGGRESSIVE = Grid(
    geometric_faces(8, 0.0, 3.0, 1.6),
    wall_resolving_faces(16, -1.0, 1.0, layer_thickness=0.02, cells_in_layer=4, max_ratio=None),
    uniform_faces(4, -0.5, 0.5),
)


@pytest.mark.parametrize("grid", [STRETCHED, AGGRESSIVE], ids=["tanh", "aggressive"])
def test_van_leer_transport_conserves_heat_and_makes_no_new_extremum_at_courant_0_5(grid):
    """With a random solenoidal carrier: the face fluxes cancel in the total, and ten steps at the scheme's tested
    Courant limit, 0.5, stay within the start's range, with nothing clipped after a step."""
    problem = dataclasses.replace(_thermal(grid, (WRAP, INSULATED, INSULATED)), advection="van_leer")
    velocity = _carrier(problem, 10, solenoidal=True)
    start = np.random.default_rng(11).uniform(0.0, 1.0, grid.shape)
    temperature = Field(jnp.asarray(start), CELL, grid)
    transport = temperature.replace_data(_transport(temperature, velocity, problem))
    ones = temperature.replace_data(jnp.ones(grid.shape))
    absolute = float(cell_inner_product(transport.replace_data(jnp.abs(transport.data)), ones))
    assert abs(float(cell_inner_product(transport, ones))) <= 1e-13 * absolute
    dt = 0.5 / float(jnp.max(outgoing_rate(velocity)))
    for _ in range(10):
        temperature = temperature.replace_data(
            temperature.data - dt * _transport(temperature, velocity, problem)
        )
    assert (
        start.min() - 1e-12
        <= float(jnp.min(temperature.data))
        <= float(jnp.max(temperature.data))
        <= start.max() + 1e-12
    )


@pytest.mark.parametrize("scheme", ["upwind", "van_leer"])
@pytest.mark.parametrize("cells", [1, 2])
def test_a_short_periodic_axis_carries_heat_through_its_wrap_face(cells, scheme):
    """A one-cell axis carries the same flux in and out, so its transport vanishes; heat is conserved either way,
    and the wrap face's two copies, the first face and the last, carry the same flux."""
    grid = Grid(uniform_faces(cells, 0.0, 1.0), tanh_faces(6, -1.0, 1.0, 1.3), uniform_faces(4, -1.0, 1.0))
    problem = dataclasses.replace(_thermal(grid, (WRAP, INSULATED, INSULATED)), advection=scheme)
    velocity = _carrier(problem, 12)
    temperature = Field(jnp.asarray(np.random.default_rng(13).standard_normal(grid.shape)), CELL, grid)
    seam = np.asarray(_upwind_fluxes(temperature, velocity, problem)[0].data)
    assert np.array_equal(seam[0], seam[-1])
    transport = _transport(temperature, velocity, problem)
    assert abs(float(jnp.sum(jnp.asarray(grid.cell_volumes()) * transport))) <= 1e-13 * float(
        jnp.sum(jnp.abs(transport))
    )
    if cells == 1:
        still = (velocity[0].replace_data(jnp.zeros_like(velocity[0].data)), *velocity[1:])
        assert np.allclose(transport, _transport(temperature, still, problem), rtol=0.0, atol=1e-13)


def test_the_limited_face_value_differentiates_where_it_is_smooth():
    """Under ``jax.jit``, on STRETCHED's walled y axis. A uniform field has every slope zero, where the van Leer mean
    is 0/0, guarded: its Jacobian is finite. A field increasing along y, with no extremum, has slopes of one sign:
    the directional derivative matches central differences to 1e-7. The face value is only piecewise smooth, with
    kinks where a slope vanishes or changes sign and where the value is held between its cells; the transport has
    one more, where a face velocity changes sign."""
    from lmhdx.thermal import _limited

    problem = dataclasses.replace(_thermal(STRETCHED, (WRAP, INSULATED, INSULATED)), advection="van_leer")

    @jax.jit
    def faces(data):
        padded = pad(data, 1, problem.thermal_conditions[1], grid=STRETCHED)
        return jnp.stack(_limited(padded, 1, problem, padded[:, :-1, :], padded[:, 1:, :]))

    jacobian = jax.jacfwd(faces)(jnp.full(STRETCHED.shape, 3.0))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    y = np.asarray(STRETCHED.centers[1])[None, :, None]
    rising = jnp.asarray(np.broadcast_to(2.0 + y + 0.3 * y**2, STRETCHED.shape))
    direction = jnp.asarray(np.random.default_rng(16).standard_normal(STRETCHED.shape))
    tangent = jax.jvp(faces, (rising,), (direction,))[1]
    size = 1e-6
    central = (faces(rising + size * direction) - faces(rising - size * direction)) / (2.0 * size)
    assert float(jnp.max(jnp.abs(tangent - central))) <= 1e-7 * float(jnp.max(jnp.abs(tangent)))


def test_the_outgoing_rate_counts_only_what_leaves_each_cell():
    """Against ``sum_f max(u.n, 0) A_f / V`` from the grid's own areas and volumes; zero at rest."""
    problem = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    velocity = _carrier(problem, 14)
    expected = np.zeros(STRETCHED.shape)
    for axis, carrier in enumerate(velocity):
        flux = np.moveaxis(np.asarray(carrier.data) * STRETCHED.face_areas(axis), axis, 0)
        leaving = np.maximum(flux[1:], 0.0) + np.maximum(-flux[:-1], 0.0)
        expected += np.moveaxis(leaving, 0, axis)
    expected /= STRETCHED.cell_volumes()
    assert np.allclose(np.asarray(outgoing_rate(velocity)), expected, rtol=1e-13, atol=0.0)
    assert not np.any(np.asarray(outgoing_rate(zero_velocity(problem.flow))))


def test_upwind_transport_converges_at_first_order():
    """A cosine carried at Courant 0.5 to t = 1/4: upwinding diffuses by ``u h (1 - C)/2``, so the error is O(h)."""
    errors = []
    for cells in (32, 64, 128):
        grid = Grid(uniform_faces(cells, 0.0, 1.0), uniform_faces(1, -1.0, 1.0), uniform_faces(1, -1.0, 1.0))
        problem = _thermal(grid, (WRAP, INSULATED, INSULATED))
        rest = zero_velocity(problem.flow)
        velocity = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])
        temperature = _cells(grid, lambda x, y, z: np.cos(2.0 * np.pi * x) + 0.0 * (y + z))
        dt = 0.5 / cells
        for _ in range(cells // 2):
            temperature = temperature.replace_data(
                temperature.data - dt * _transport(temperature, velocity, problem)
            )
        exact = np.cos(2.0 * np.pi * (np.asarray(grid.centers[0]) - 0.25))
        errors.append(float(np.max(np.abs(np.asarray(temperature.data)[:, 0, 0] - exact))))
    orders = np.log2(np.divide(errors[:-1], errors[1:]))
    assert np.all((orders > 0.8) & (orders < 1.2)), (errors, orders)


# Joule heat (plan step 5.1d): each face current squared, then shared between the half cells beside it.


def _face_fields(grid: Grid, values) -> tuple[Field, Field, Field]:
    offsets = [tuple(FACE if position == axis else CENTER for position in range(3)) for axis in range(3)]
    return tuple(Field(jnp.asarray(value), offsets[axis], grid) for axis, value in enumerate(values))


def test_joule_heat_squares_each_face_current_before_sharing_it():
    """Uniform currents give ``|J|^2 / sigma``; alternating ones, whose cell average is zero, still heat."""
    base = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    problem = dataclasses.replace(base, flow=dataclasses.replace(base.flow, conductivity=2.0))
    shapes = [STRETCHED.face_shape(axis) for axis in range(3)]
    uniform = _face_fields(
        STRETCHED, [np.full(shape, value) for shape, value in zip(shapes, (0.5, -1.5, 2.0))]
    )
    assert np.allclose(np.asarray(_joule_heating(uniform, problem).data), 6.5 / 2.0, rtol=1e-14, atol=0.0)
    sign = np.broadcast_to((-1.0) ** np.arange(shapes[0][0])[:, None, None], shapes[0])
    alternating = _face_fields(STRETCHED, [sign, np.zeros(shapes[1]), np.zeros(shapes[2])])
    assert np.allclose(np.asarray(_joule_heating(alternating, problem).data), 0.5, rtol=1e-14, atol=0.0)
    rng = np.random.default_rng(15)
    noise = _face_fields(STRETCHED, [rng.standard_normal(shape) for shape in shapes])
    assert float(jnp.min(_joule_heating(noise, problem).data)) >= 0.0
    for off in (
        dataclasses.replace(base, joule_heating=False),
        dataclasses.replace(base, flow=dataclasses.replace(base.flow, conductivity=0.0)),
    ):
        assert not np.any(np.asarray(_joule_heating(noise, off).data))
    with pytest.raises(TypeError):
        dataclasses.replace(base, joule_heating=1)


# Buoyancy (plan step 5.1e, ADR 0008 D32): -rho_0 beta (T - T_r) g on the velocity faces.


def test_the_buoyancy_force_pushes_a_hot_cell_against_gravity():
    """A hot cell under ``g = -9.8 z`` pushes up on its two z faces by the face-average weight; nothing else moves."""
    base = _thermal(STRETCHED, (WRAP, INSULATED, INSULATED))
    problem = dataclasses.replace(
        base, expansion_coefficient=2.0e-4, gravity=(0.0, 0.0, -9.8), reference_temperature=1.0
    )
    hot = np.ones(STRETCHED.shape)
    hot[3, 4, 2] = 6.0
    forces = _buoyancy(Field(jnp.asarray(hot), CELL, STRETCHED), problem)
    expected = np.zeros(STRETCHED.face_shape(2))
    expected[3, 4, [2, 3]] = 0.5 * 1.5 * 2.0e-4 * 5.0 * 9.8  # uniform z cells share the force equally
    assert np.allclose(np.asarray(forces[2].data), expected, rtol=1e-14, atol=1e-18)
    assert not np.any(np.asarray(forces[0].data)) and not np.any(np.asarray(forces[1].data))
    assert all(force.offset == velocity_offset(axis) for axis, force in enumerate(forces))
    sideways = dataclasses.replace(problem, gravity=(0.0, -2.0, 3.0))
    tilted = _buoyancy(Field(jnp.asarray(hot), CELL, STRETCHED), sideways)
    upward = np.asarray(forces[2].data) / -9.8
    assert np.allclose(np.asarray(tilted[2].data), 3.0 * upward, rtol=1e-14, atol=1e-18)
    # Warm everywhere, so the cells beside the walls push too; the walls still take nothing.
    warm = _buoyancy(Field(jnp.asarray(hot + 1.0), CELL, STRETCHED), sideways)
    for axis in (1, 2):
        faces = np.moveaxis(np.asarray(warm[axis].data), axis, 0)
        assert not np.any(faces[[0, -1]]) and np.all(faces[1:-1])
    for passive in (
        dataclasses.replace(problem, expansion_coefficient=0.0),
        dataclasses.replace(problem, expansion_coefficient=0.0, gravity=(1.0, -2.0, 3.0)),
        dataclasses.replace(problem, gravity=(0.0, 0.0, 0.0)),
    ):
        assert _buoyancy(Field(jnp.asarray(hot), CELL, STRETCHED), passive) is None
    # Along the periodic x no pressure balances the mean push (ADR 0008, D32): refused while beta > 0.
    with pytest.raises(ValueError, match="periodic axis"):
        dataclasses.replace(problem, gravity=(1.0, -2.0, 3.0))
    reference = _buoyancy(Field(jnp.ones(STRETCHED.shape), CELL, STRETCHED), problem)
    assert not any(np.any(np.asarray(force.data)) for force in reference)


# Conduction through solid walls (plan step 5.4b, ADR 0009 D36): the extended grid and the PCG solve.

# 316 steel at 573 K: k, and rho c_p over 1e6. The PbLi it lines is 20.4 and 1.867; t_w/a is 2/23.
STEEL = (18.3, 4.27)


def _walled(cells: int = 6, wall_cells: int = 2):
    """A duct walled in steel on four sides, its PbLi cells and the cells' ``k`` and ``rho c_p``."""
    fluid = Grid(uniform_faces(4, 0.0, 2.0), *(tanh_faces(cells, -1.0, 1.0, 2.0) for _ in range(2)))
    wall = SolidWall(2.0 / 23.0, wall_cells, *STEEL)
    walled = conjugate_grid(fluid, ((None, None), (wall, wall), (wall, wall)))
    return walled, walled.cells("thermal_conductivity", 20.4), walled.cells("heat_capacity", 1.867)


@pytest.mark.parametrize("lower", ["temperature", "heat_flux"])
def test_two_materials_in_series_conduct_exactly(lower):
    """Steady conduction through steel, fluid and steel held at 1 above, and at 0 or heated at 0.75 below: the
    harmonic face conductance makes the cell temperatures the exact piecewise-linear profile and carries the same
    flux through every face, the two interfaces among them. A heat flux built with the wall's ``k`` delivers it."""
    fluid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(8, -1.0, 1.0))
    wall = SolidWall(0.5, 4, 3.0, 1.0)
    walled = conjugate_grid(fluid, ((None, None), (None, None), (wall, wall)))
    k = walled.cells("thermal_conductivity", 1.0)
    z = np.asarray(walled.grid.centers[2])
    below = np.where(
        z < -1.0, (z + 1.5) / 3.0, np.where(z < 1.0, 0.5 / 3.0 + z + 1.0, 13 / 6 + (z - 1.0) / 3.0)
    )
    if lower == "temperature":  # the resistance below z, over the total 7/3
        ends, exact, flux = thermal_boundary("temperature", 0.0, 1.0), below * 3.0 / 7.0, 3.0 / 7.0
    else:  # 0.75 enters below, a negative outward flux, and leaves through the top
        ends = thermal_boundary("heat_flux", -0.75, 1.0, upper_kind="temperature", conductivity=3.0)
        exact, flux = 1.0 + 0.75 * (7.0 / 3.0 - below), -0.75
    conditions = (WRAP, WALL, ends)
    steady = variable_helmholtz(walled.grid, conditions, k, 1e-12, 1.0)  # no storage to speak of
    temperature, _ = steady.solve(steady.lift(conditions))
    assert np.asarray(temperature)[0, 0] == pytest.approx(exact, abs=1e-9)
    crossing = np.asarray(steady.face_fluxes(temperature, conditions)[2].data)[0, 0]
    assert crossing == pytest.approx(np.full(17, flux), abs=1e-9)


def test_two_materials_brought_into_contact_follow_the_exact_transient():
    """Steel at 1 against fluid at 0, each deep: the contact temperature is the mean weighted by the effusivities
    ``sqrt(k rho c_p)``, constant in time, with an erf profile on either side. Implicit steps on the increment
    reach it at second order with ``dt`` refined as ``h^2``."""
    from scipy.special import erf

    errors, (k, c), time = [], (0.9, 2.3), 0.02
    for cells, steps in ((32, 50), (64, 200)):
        fluid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(1, 0.0, 1.0), uniform_faces(cells, 0.0, 1.0))
        walled = conjugate_grid(fluid, ((None, None), (None, None), (SolidWall(1.0, cells, k, c), None)))
        dt, z = time / steps, np.asarray(walled.grid.centers[2])
        conduction = variable_helmholtz(
            walled.grid,
            (WRAP, WALL, WALL),
            walled.cells("thermal_conductivity", 1.0),
            walled.cells("heat_capacity", 1.0),
            dt,
        )
        advance = jax.jit(lambda t: t + conduction.solve(dt * conduction.divergence(t))[0])
        temperature = jnp.asarray(np.broadcast_to(np.where(z < 0.0, 1.0, 0.0), walled.grid.shape))
        for _ in range(steps):
            temperature = advance(temperature)
        contact = np.sqrt(k * c) / (np.sqrt(k * c) + 1.0)
        steel = contact + (1.0 - contact) * erf(-z / (2.0 * np.sqrt(k / c * time)))
        exact = np.where(z < 0.0, steel, contact * (1.0 - erf(z / (2.0 * np.sqrt(time)))))
        errors.append(float(np.max(np.abs(np.asarray(temperature)[0, 0] - exact))))
    order = np.log2(errors[0] / errors[1])
    assert errors[1] < 1e-3 and 1.8 < order < 2.2, (errors, order)


@pytest.mark.parametrize("cells", [6, 24])
@pytest.mark.parametrize("dt", [0.05, 100.0])
def test_walled_conduction_is_symmetric_conserves_heat_and_converges_in_few_iterations(cells, dt):
    """A duct walled on four sides with the benchmark's steel and PbLi, insulated outside: the operator is
    symmetric, a step stores the heat put in, and the volume-mean preconditioner leaves under ten iterations,
    on either mesh and at either step."""
    walled, k, c = _walled(cells)
    conduction = variable_helmholtz(walled.grid, (WRAP, WALL, WALL), k, c, dt)
    rng = np.random.default_rng(54)
    u, v = (jnp.asarray(rng.standard_normal(walled.grid.shape)) for _ in range(2))
    assert float(jnp.vdot(conduction.apply(u), v)) == pytest.approx(
        float(jnp.vdot(u, conduction.apply(v))), rel=1e-12
    )
    source = np.zeros(walled.grid.shape)
    source[walled.fluid] = 1.0
    increment, iterations = conduction.solve(dt * jnp.asarray(source))  # from a uniform temperature
    stored = np.sum(walled.grid.cell_volumes() * c * np.asarray(increment))
    assert stored == pytest.approx(dt * np.sum(walled.grid.cell_volumes() * source), rel=1e-10)
    assert int(iterations) < 10


def test_walled_conduction_at_a_high_contrast_converges_in_the_documented_iterations():
    """Copper-like walls, ``k`` 19 times the fluid's, on 24 cells across: the volume-mean preconditioner takes more
    iterations (33 measured, against 8 for steel), within the 40 the docstring's range allows."""
    fluid = Grid(uniform_faces(4, 0.0, 2.0), *(tanh_faces(24, -1.0, 1.0, 2.0) for _ in range(2)))
    wall = SolidWall(2.0 / 23.0, 2, 390.0, 3.45)
    walled = conjugate_grid(fluid, ((None, None), (wall, wall), (wall, wall)))
    k, c = walled.cells("thermal_conductivity", 20.4), walled.cells("heat_capacity", 1.867)
    source = np.zeros(walled.grid.shape)
    source[walled.fluid] = 1.0
    conduction = variable_helmholtz(walled.grid, (WRAP, WALL, WALL), k, c, 0.05)
    increment, iterations = conduction.solve(0.05 * source)
    assert 10 < int(iterations) <= 40 and np.all(np.isfinite(np.asarray(increment)))


def test_uniform_conduction_is_the_fast_diagonal_inverse_and_walled_conduction_differentiates():
    """One material: PCG is the fast-diagonal inverse itself, one iteration. With walls, forward and reverse
    derivatives in the right-hand side match central differences, the tangent and adjoint solves starting from
    zero."""
    conditions = (WRAP, WALL, WALL)
    uniform = variable_helmholtz(STRETCHED, conditions, 2.0, 3.0, 0.01)
    rhs = jnp.asarray(np.random.default_rng(5).standard_normal(STRETCHED.shape))
    temperature, iterations = uniform.solve(rhs)
    direct = fast_diagonal_helmholtz(STRETCHED, CELL, conditions, shift=3.0, coefficient=0.02).solve(
        Field(rhs, CELL, STRETCHED)
    )
    assert np.asarray(temperature) == pytest.approx(np.asarray(direct.data), rel=1e-9, abs=1e-12)
    assert int(iterations) == 1
    walled, k, c = _walled()
    conduction = variable_helmholtz(walled.grid, conditions, k, c, 0.05)
    rng = np.random.default_rng(6)
    rhs, weights = (jnp.asarray(1e-3 * rng.standard_normal(walled.grid.shape)) for _ in range(2))

    def objective(scale):
        return jnp.vdot(weights, conduction.solve(scale * rhs)[0])

    difference = float(objective(1.5) - objective(0.5))
    assert float(jax.grad(objective)(1.0)) == pytest.approx(difference, rel=1e-9)
    assert float(jax.jvp(objective, (1.0,), (1.0,))[1]) == pytest.approx(difference, rel=1e-9)


def test_a_wall_conductivity_is_a_leaf_of_the_conduction_solve():
    """A wall's conductivity, put into the cells in the trace (``ConjugateGrid.cells`` with ``walls``), is a leaf of
    the PCG solve. In two walls at once, the gradient meets central differences at two steps, its Taylor remainder
    falls at second order, it repeats at a tighter tolerance, it meets its own central difference on a finer mesh,
    and it satisfies the JVP/VJP identity; a new value does not recompile. Unconverged, value and gradient are NaN."""

    def walled_solve(cells):
        walled, k, c = _walled(cells=cells)
        y, z = (np.asarray(walled.grid.centers[axis]) for axis in (1, 2))
        heat = np.exp(-((y[:, None] - 0.3) ** 2 + (z[None, :] + 0.5) ** 2))
        weight = np.where(z[None, :] < 0.0, 1.0, 0.2) * (1.0 + y[:, None])
        rhs, weights = (jnp.asarray(np.broadcast_to(a, walled.grid.shape)) for a in (heat, weight))
        conduction = variable_helmholtz(walled.grid, (WRAP, WALL, WALL), k, c, 0.05)

        def objective(values, operator=conduction):
            traces.append(None)
            k = walled.cells("thermal_conductivity", 20.4, {(2, 0): values[0], (1, 1): values[1]})
            return jnp.vdot(weights, operator.with_coefficients(conductivity=k).solve(rhs)[0])

        return conduction, objective

    traces, steel = [], jnp.asarray([STEEL[0]] * 2)
    conduction, objective = walled_solve(6)
    direction, gradient = jnp.asarray([1.0, -0.5]), jax.grad(objective)(steel)
    slope = float(gradient @ direction)
    for h in (1e-2, 1e-3):
        central = (objective(steel + h * direction) - objective(steel - h * direction)) / (2.0 * h)
        assert float(central) == pytest.approx(slope, rel=2e-7)
    base = objective(steel)
    remainders = [abs(float(objective(steel + h * direction) - base) - h * slope) for h in (4.0, 2.0, 1.0)]
    assert all(3.5 < coarse / fine < 4.5 for coarse, fine in zip(remainders, remainders[1:])), remainders
    tight = jax.grad(lambda values: objective(values, dataclasses.replace(conduction, tolerance=1e-13)))(
        steel
    )
    assert np.asarray(tight) == pytest.approx(np.asarray(gradient), rel=1e-9)
    tangent = float(jax.jvp(objective, (steel,), (direction,))[1])
    assert float(jax.vjp(objective, steel)[1](1.0)[0] @ direction) == pytest.approx(tangent, rel=1e-12)
    traces.clear()
    jitted = jax.jit(objective)
    jitted(steel), jitted(1.1 * steel)
    assert len(traces) == 1, "a new wall conductivity recompiled"
    unconverged = dataclasses.replace(conduction, max_steps=1)
    assert np.isnan(float(objective(steel, unconverged)))
    assert np.all(np.isnan(np.asarray(jax.grad(lambda values: objective(values, unconverged))(steel))))
    _, finer = walled_solve(10)
    central = (finer(steel + 1e-3 * direction) - finer(steel - 1e-3 * direction)) / 2e-3
    assert float(jax.grad(finer)(steel) @ direction) == pytest.approx(float(central), rel=1e-7)


def test_highly_conducting_walls_hold_the_fluid_at_their_temperature():
    """An independent check at a high contrast. Walls 1000 times the fluid's conductivity, 0.1 thick and held at 0
    outside, around a square fluid heated uniformly: as the contrast grows the fluid sees its boundary at 0, and its
    mean temperature tends to the Dirichlet Poisson series ``256/pi^6 sum 1/(m^2 n^2 (m^2 + n^2))``, 0.140577, raised
    by the walls' own drop ``q a t / (2 k_w)``. On 24 and 48 cells the error falls at about second order (0.70 % and
    0.20 %), and the Richardson value meets the series and the drop within 1e-4 of the mean, in 39 and 52 PCG
    iterations."""
    m = np.arange(1, 4001, 2)[:, None]
    series = float(np.sum(256.0 / (np.pi**6 * m**2 * m.T**2 * (m**2 + m.T**2))))
    held, means = BoundaryCondition(DIRICHLET), []
    for cells in (24, 48):
        fluid = Grid(uniform_faces(1, 0.0, 1.0), *(tanh_faces(cells, -1.0, 1.0, 1.5) for _ in range(2)))
        walled = conjugate_grid(fluid, ((None, None),) + ((SolidWall(0.1, 4, 1000.0, 1.0),) * 2,) * 2)
        heat = np.zeros(walled.grid.shape)
        heat[walled.fluid] = 1.0
        conduction = variable_helmholtz(
            walled.grid, (WRAP, held, held), walled.cells("thermal_conductivity", 1.0), 1e-9, 1.0
        )
        temperature, iterations = dataclasses.replace(conduction, max_steps=400).solve(jnp.asarray(heat))
        volumes = walled.grid.cell_volumes()[walled.fluid]
        means.append(float(np.sum(np.asarray(temperature)[walled.fluid] * volumes) / np.sum(volumes)))
        assert int(iterations) < 80
    errors = [mean / series - 1.0 for mean in means]
    assert 3.0 < errors[0] / errors[1] < 4.5, errors
    richardson = means[1] + (means[1] - means[0]) / 3.0
    assert richardson == pytest.approx(series + 0.5 * 0.1 / 1000.0, abs=1e-4 * series)


def test_walls_build_the_extended_grid_and_its_corners_take_the_nearer_wall():
    """The fluid's faces are kept exactly; a corner cell takes the nearer wall's material, the first axis's when
    equally near to rounding, as resolved walls do, so equal walls give a mirror-symmetric map on an off-centre fluid
    too (rounding split three of its ties before); and what cannot be a wall is refused."""
    walled, k, c = _walled(cells=6, wall_cells=2)
    assert walled.grid.shape == (4, 10, 10) and walled.fluid == (slice(0, 4), slice(2, 8), slice(2, 8))
    inside = Grid(
        *(
            np.asarray(faces)[block.start : block.stop + 1]
            for faces, block in zip(walled.grid.faces, walled.fluid)
        )
    )
    assert inside == Grid(uniform_faces(4, 0.0, 2.0), *(tanh_faces(6, -1.0, 1.0, 2.0) for _ in range(2)))
    assert np.asarray(walled.grid.faces[1])[0] == pytest.approx(-1.0 - 2.0 / 23.0)
    assert np.all(k[walled.fluid] == 20.4) and k[0, 0, 5] == STEEL[0] and c[0, 5, 0] == STEEL[1]
    fluid = Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(4, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    steel, copper = SolidWall(0.2, 2, *STEEL), SolidWall(0.2, 2, 400.0, 3.4)
    mixed = conjugate_grid(fluid, [[None, None], [copper, copper], (steel, steel)])  # lists are pairs too
    k = mixed.cells("thermal_conductivity", 1.0)
    assert (k[0, 0, 0], k[0, 1, 0], k[0, 0, 1], k[0, 6, 7]) == (400.0, 400.0, STEEL[0], 400.0)
    assert np.array_equal(k, k[:, ::-1]) and np.array_equal(k, k[:, :, ::-1])  # every corner alike
    off_centre = Grid(uniform_faces(1, 0.0, 1.0), *(uniform_faces(5, -0.7, 0.3) for _ in range(2)))
    k = conjugate_grid(off_centre, ((None, None), (copper, copper), (steel, steel))).cells(
        "thermal_conductivity", 1
    )
    assert np.array_equal(k, k[:, ::-1]) and np.array_equal(k, k[:, :, ::-1]), "ties split by rounding"
    for walls, message in (
        (((steel, None), (None, None), (None, None)), "transverse"),
        (((None, None), (0.1, None), (None, None)), "SolidWall"),
    ):
        with pytest.raises(ValueError, match=message):
            conjugate_grid(fluid, walls)
    with pytest.raises(ValueError, match="Cartesian"):
        conjugate_grid(_polar(2, 4), ((None, None),) * 3)
    with pytest.raises(ValueError, match="whole"):
        SolidWall(0.3, 1.5, *STEEL)
    with pytest.raises(ValueError, match="positive and finite"):
        variable_helmholtz(fluid, (WRAP, WALL, WALL), np.nan, 1.0, 1.0)
    conduction = variable_helmholtz(fluid, (WRAP, WALL, WALL), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="own kinds"):
        conduction.lift((WRAP, thermal_boundary("temperature", 1.0), WALL))


def test_without_a_shift_one_material_is_the_singular_fast_diagonal_inverse():
    """The charge balance of one conductivity, insulated outside: PCG is the fast-diagonal Poisson inverse in one
    iteration, the right-hand side's incompatible mean removed and the solution's mean zero."""
    from lmhdx.poisson import fast_diagonal_poisson

    charge = variable_helmholtz(STRETCHED, (WRAP, WALL, WALL), 2.0, None, 1.0)
    rhs = jnp.asarray(np.random.default_rng(7).standard_normal(STRETCHED.shape))
    potential, iterations = charge.solve(rhs)
    direct = fast_diagonal_poisson(STRETCHED, (WRAP, WALL, WALL)).solve(Field(-rhs / 2.0, CELL, STRETCHED))
    assert np.asarray(potential) == pytest.approx(
        np.asarray(direct.data), abs=1e-10 * float(jnp.max(jnp.abs(direct.data)))
    )
    assert int(iterations) == 1
    with pytest.raises(ValueError, match="insulated or periodic"):
        variable_helmholtz(STRETCHED, (WRAP, WALL, thermal_boundary("temperature")), 2.0, None, 1.0)


def test_a_float32_fast_diagonal_helmholtz_stays_definite_at_a_large_coefficient():
    """Probed in float32, an insulated axis's null mode came out +3e-5, so at a large coefficient the shifted
    operator turned indefinite and PCG broke down at once; the eigenvalues are clamped to the Laplacian's sign."""
    import subprocess
    import sys

    code = """
import numpy as np
from lmhdx.core3d import SolidWall, conjugate_grid
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, tanh_faces, uniform_faces
from lmhdx.poisson import variable_helmholtz
fluid = Grid(uniform_faces(2, 0.0, 1.0), *(tanh_faces(24, -1.0, 1.0, 2.0) for _ in range(2)))
wall = SolidWall(2.0 / 23.0, 4, 18.3, 4.27)
walled = conjugate_grid(fluid, ((None, None), (wall, wall), (wall, wall)))
k, c = walled.cells("thermal_conductivity", 20.4), walled.cells("heat_capacity", 1.867)
walls = (BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN), BoundaryCondition(NEUMANN))
conduction = variable_helmholtz(walled.grid, walls, k, c, 1.0e4)
assert max(float(np.max(values)) for values in conduction.preconditioner.values) <= 0.0
increment, iterations = conduction.solve(np.ones(walled.grid.shape, np.float32))
assert increment.dtype == np.float32 and np.all(np.isfinite(increment)), int(iterations)
"""
    env = {key: value for key, value in __import__("os").environ.items() if key != "JAX_ENABLE_X64"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=300)
