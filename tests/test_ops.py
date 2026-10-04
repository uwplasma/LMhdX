"""Boundary padding and the compatible gradient, divergence and Laplacian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.core3d import ChannelProblem, step, velocity_offset, zero_velocity
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
    staggered_laplacian,
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
