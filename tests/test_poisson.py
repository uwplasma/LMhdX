"""Direct Poisson solves against dense factorizations and the assembled operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition
from lmhdx.grid import CENTER, Field, Grid, geometric_faces, tanh_faces, uniform_faces
from lmhdx.ops import laplacian
from lmhdx.poisson import assemble_axis_laplacian, fast_diagonal_poisson

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
FIXED = BoundaryCondition(DIRICHLET)
WRAPPED = BoundaryCondition(PERIODIC)
STRETCHED = Grid(
    geometric_faces(6, 0.0, 3.0, 1.25),
    tanh_faces(8, -1.0, 1.0, 1.6),
    uniform_faces(5, -0.5, 0.5),
)
SMALL = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))


def _random_cells(grid: Grid, seed: int = 0) -> Field:
    data = jax.random.normal(jax.random.PRNGKey(seed), grid.shape, dtype=jnp.float64)
    return Field(data, (CENTER,) * 3, grid)


def _dense_operator(grid: Grid, conditions) -> np.ndarray:
    """Assemble the full Laplacian by applying it to unit vectors."""
    size = int(np.prod(grid.shape))
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        field = Field(jnp.asarray(unit.reshape(grid.shape)), (CENTER,) * 3, grid)
        columns.append(np.asarray(laplacian(field, conditions).data).reshape(size))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize(
    "conditions",
    [
        (FIXED, FIXED, FIXED),
        (FIXED, WALL, FIXED),
        (WRAPPED, FIXED, FIXED),
        (WRAPPED, WRAPPED, FIXED),
    ],
    ids=["dirichlet", "mixed-neumann", "one-periodic", "two-periodic"],
)
def test_direct_solve_inverts_the_assembled_operator(conditions):
    grid = STRETCHED
    factorization = fast_diagonal_poisson(grid, conditions)
    assert not factorization.singular
    exact = _random_cells(grid, seed=3)
    rhs = laplacian(exact, conditions)
    solution = factorization.solve(rhs)
    assert float(jnp.max(jnp.abs(solution.data - exact.data))) < 1e-11
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_direct_solve_matches_a_dense_factorization_on_a_small_grid():
    grid, conditions = SMALL, (FIXED, FIXED, WALL)
    dense = _dense_operator(grid, conditions)
    rhs = _random_cells(grid, seed=5)
    expected = np.linalg.solve(dense, np.asarray(rhs.data).reshape(-1)).reshape(grid.shape)
    solution = fast_diagonal_poisson(grid, conditions).solve(rhs)
    assert np.max(np.abs(np.asarray(solution.data) - expected)) < 1e-12


def test_axis_operator_reproduces_the_three_dimensional_stencil():
    grid, axis = STRETCHED, 1
    operator = assemble_axis_laplacian(grid, axis, FIXED)
    assert operator.shape == (grid.shape[axis],) * 2
    profile = jax.random.normal(jax.random.PRNGKey(7), (grid.shape[axis],), dtype=jnp.float64)
    field = Field(jnp.broadcast_to(profile[None, :, None], grid.shape), (CENTER,) * 3, grid)
    conditions = tuple(FIXED if position == axis else WALL for position in range(3))
    expected = np.asarray(laplacian(field, conditions).data)[0, :, 0]
    assert np.max(np.abs(operator @ np.asarray(profile) - expected)) < 1e-12


@pytest.mark.parametrize(
    "conditions",
    [(WALL, WALL, WALL), (WRAPPED, WRAPPED, WRAPPED), (WALL, WRAPPED, WALL)],
    ids=["all-neumann", "all-periodic", "neumann-periodic"],
)
def test_singular_problems_solve_up_to_a_constant(conditions):
    grid = STRETCHED
    factorization = fast_diagonal_poisson(grid, conditions)
    assert factorization.singular
    volumes = np.asarray(grid.cell_volumes())
    exact = _random_cells(grid, seed=11)
    centred = exact.replace_data(
        exact.data - jnp.sum(jnp.asarray(volumes) * exact.data) / jnp.sum(jnp.asarray(volumes))
    )
    rhs = laplacian(centred, conditions)
    solution = factorization.solve(rhs)
    assert abs(float(jnp.sum(jnp.asarray(volumes) * solution.data))) < 1e-12
    assert float(jnp.max(jnp.abs(solution.data - centred.data))) < 1e-11
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_singular_solve_discards_the_incompatible_component():
    grid, conditions = SMALL, (WALL, WALL, WALL)
    factorization = fast_diagonal_poisson(grid, conditions)
    rhs = _random_cells(grid, seed=13)
    solution = factorization.solve(rhs)
    # The constant part of the right-hand side cannot be represented; the rest is.
    assert float(factorization.residual_norm(solution, rhs)) < 1e-11


def test_solution_is_differentiable_and_jits():
    grid, conditions = SMALL, (FIXED, FIXED, FIXED)
    factorization = fast_diagonal_poisson(grid, conditions)

    def energy(values):
        solution = factorization.solve(Field(values, (CENTER,) * 3, grid))
        return jnp.sum(solution.data**2)

    values = jax.random.normal(jax.random.PRNGKey(17), grid.shape, dtype=jnp.float64)
    gradient = jax.jit(jax.grad(energy))(values)
    step, direction = 1.0e-6, jax.random.normal(jax.random.PRNGKey(19), grid.shape, dtype=jnp.float64)
    difference = (energy(values + step * direction) - energy(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.sum(gradient * direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_single_precision_solves_at_single_precision_accuracy():
    grid, conditions = SMALL, (FIXED, FIXED, FIXED)
    factorization = fast_diagonal_poisson(grid, conditions)
    exact = Field(
        jnp.asarray(np.random.default_rng(0).normal(size=grid.shape), dtype=jnp.float32), (CENTER,) * 3, grid
    )
    rhs = laplacian(exact, conditions)
    solution = factorization.solve(rhs)
    assert solution.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(solution.data - exact.data))) < 1e-3


def test_factorization_refuses_inhomogeneous_boundary_data():
    with pytest.raises(ValueError, match="inhomogeneous boundary data"):
        fast_diagonal_poisson(SMALL, (BoundaryCondition(DIRICHLET, lower=1.0), FIXED, FIXED))


def test_factorization_validates_its_inputs():
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        fast_diagonal_poisson(SMALL, (FIXED, FIXED))
    with pytest.raises(ValueError, match="precision must be one of"):
        fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED), precision="half")
    with pytest.raises(ValueError, match="refinements must be a positive integer"):
        fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED), precision="mixed", refinements=0)
    factorization = fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED))
    other = Grid(uniform_faces(5, 0.0, 1.0), *SMALL.faces[1:])
    with pytest.raises(ValueError, match="share the factorized grid"):
        factorization.solve(_random_cells(other))
    with pytest.raises(ValueError, match="cell centred"):
        factorization.solve(Field(jnp.zeros(SMALL.face_shape(0)), (0.0, CENTER, CENTER), SMALL))


def test_factorization_refuses_an_operator_that_is_not_symmetric(monkeypatch):
    """A wall stencil that broke symmetry would invalidate the factorization."""
    import lmhdx.poisson as poisson

    def asymmetric(grid, axis, condition):
        operator = assemble_axis_laplacian(grid, axis, condition)
        operator[0, 1] += 1.0
        return operator

    monkeypatch.setattr(poisson, "assemble_axis_laplacian", asymmetric)
    with pytest.raises(ValueError, match="not symmetric under the cell widths"):
        poisson.fast_diagonal_poisson(SMALL, (FIXED, FIXED, FIXED))


@pytest.mark.parametrize(
    ("offset", "conditions", "name"),
    [
        ((0.0, CENTER, CENTER), (WRAPPED, WALL, WRAPPED), "periodic-face"),
        ((CENTER, 0.0, CENTER), (WRAPPED, WALL, WRAPPED), "walled-face"),
        ((CENTER, CENTER, CENTER), (WRAPPED, WALL, WRAPPED), "cell-centred"),
    ],
    ids=["periodic-face", "walled-face", "cell-centred"],
)
def test_the_helmholtz_factorization_inverts_its_operator(offset, conditions, name):
    """A staggered implicit viscous solve, checked against the stencil it factorizes."""
    from lmhdx.ops import staggered_laplacian
    from lmhdx.poisson import fast_diagonal_helmholtz, free_slice

    del name
    grid = Grid(uniform_faces(4, 0.0, 1.0), geometric_faces(8, -1.0, 1.0, 1.15), uniform_faces(4, -1.0, 1.0))
    coefficient = 0.05
    factorization = fast_diagonal_helmholtz(grid, offset, conditions, shift=1.0, coefficient=coefficient)
    generator = np.random.default_rng(0)
    rhs = Field(jnp.asarray(generator.normal(size=grid.offset_shape(offset))), offset, grid)
    solution = factorization.solve(rhs)
    applied = solution.data - coefficient * staggered_laplacian(solution, conditions).data
    free = tuple(free_slice(grid, axis, offset[axis], conditions[axis]) for axis in range(3))
    residual = np.max(np.abs(np.asarray(applied)[free] - np.asarray(rhs.data)[free]))
    assert residual < 1e-12


def test_the_helmholtz_solve_leaves_prescribed_entries_at_zero():
    """Wall faces are boundary data the caller owns, not unknowns to set."""
    from lmhdx.poisson import fast_diagonal_helmholtz

    grid = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(6, -1.0, 1.0), uniform_faces(4, -1.0, 1.0))
    offset, conditions = (CENTER, 0.0, CENTER), (WRAPPED, WALL, WRAPPED)
    factorization = fast_diagonal_helmholtz(grid, offset, conditions, shift=1.0, coefficient=0.1)
    rhs = Field(jnp.ones(grid.offset_shape(offset)), offset, grid)
    solution = np.asarray(factorization.solve(rhs).data)
    assert np.all(solution[:, 0, :] == 0.0)
    assert np.all(solution[:, -1, :] == 0.0)
    assert np.any(solution[:, 1:-1, :] != 0.0)


def test_the_helmholtz_factorization_validates_its_inputs():
    from lmhdx.poisson import fast_diagonal_helmholtz

    grid = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))
    with pytest.raises(ValueError, match="one boundary condition per axis"):
        fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL))
    with pytest.raises(ValueError, match="inhomogeneous boundary data"):
        fast_diagonal_helmholtz(grid, (CENTER,) * 3, (BoundaryCondition(DIRICHLET, lower=1.0), WALL, WALL))
    factorization = fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL, WALL))
    with pytest.raises(ValueError, match="match the factorized position"):
        factorization.solve(Field(jnp.zeros(grid.face_shape(0)), (0.0, CENTER, CENTER), grid))


def test_the_helmholtz_factorization_refuses_an_asymmetric_operator(monkeypatch):
    """The same tripwire as the scalar case, for the staggered assembly."""
    import lmhdx.poisson as poisson

    grid = Grid(*(uniform_faces(4, 0.0, 1.0) for _ in range(3)))

    original = poisson.assemble_staggered_axis_operator

    def asymmetric(grid_, axis, offset, condition):
        operator = original(grid_, axis, offset, condition)
        operator[0, 1] += 5.0
        return operator

    monkeypatch.setattr(poisson, "assemble_staggered_axis_operator", asymmetric)
    with pytest.raises(ValueError, match="not symmetric under its cell weights"):
        poisson.fast_diagonal_helmholtz(grid, (CENTER,) * 3, (WALL, WALL, WALL))


def test_a_wall_resolving_mesh_still_factorizes():
    """Eight orders of magnitude in the operator is a graded mesh, not a broken stencil."""
    from lmhdx.grid import wall_resolving_faces
    from lmhdx.poisson import fast_diagonal_poisson

    faces = wall_resolving_faces(48, -1.0, 1.0, layer_thickness=1.0 / 300.0, cells_in_layer=6, max_ratio=1.45)
    grid = Grid(uniform_faces(1, 0.0, 1.0), faces, faces)
    widths = np.diff(faces)
    assert widths.max() / widths.min() > 1.0e3
    factorization = fast_diagonal_poisson(grid, (WRAPPED, WALL, WALL))
    source = _random_cells(grid)
    solved = factorization.solve(source.replace_data(source.data - jnp.mean(source.data)))
    assert bool(jnp.all(jnp.isfinite(solved.data)))


def _polar_grid(radial: int, azimuthal: int, axial: int = 1) -> Grid:
    from lmhdx.grid import POLAR

    return Grid(
        uniform_faces(radial, 0.0, 1.0),
        uniform_faces(azimuthal, 0.0, 2.0 * np.pi),
        uniform_faces(axial, 0.0, 1.0),
        geometry=POLAR,
    )


@pytest.mark.parametrize("radial_condition", [FIXED, WALL], ids=["dirichlet", "neumann"])
def test_the_polar_factorization_inverts_its_own_operator(radial_condition):
    """One radial eigendecomposition per azimuthal mode, and the mode zero null space removed."""
    from lmhdx.ops import laplacian
    from lmhdx.poisson import fast_diagonal_polar_poisson

    grid = _polar_grid(12, 16, 3)
    conditions = (radial_condition, WRAPPED, WRAPPED)
    factorization = fast_diagonal_polar_poisson(grid, conditions)
    volumes = np.asarray(grid.cell_volumes())
    source = np.asarray(_random_cells(grid).data)
    if factorization.singular:
        source = source - (source * volumes).sum() / volumes.sum()
    field = Field(jnp.asarray(source), (CENTER, CENTER, CENTER), grid)
    difference = np.asarray(laplacian(factorization.solve(field), conditions).data) - source
    if factorization.singular:
        difference = difference - (difference * volumes).sum() / volumes.sum()
    assert np.max(np.abs(difference)) / np.max(np.abs(source)) < 1e-11


def test_the_polar_solve_is_second_order_on_a_paraboloid():
    """`lap phi = -4` with `phi(1) = 0` is `1 - r^2`, wall closure and axis included."""
    from lmhdx.poisson import fast_diagonal_polar_poisson

    errors = []
    for count in (16, 32, 64):
        grid = _polar_grid(count, 2 * count)
        factorization = fast_diagonal_polar_poisson(grid, (FIXED, WRAPPED, WRAPPED))
        source = Field(jnp.full(grid.shape, -4.0), (CENTER, CENTER, CENTER), grid)
        radius = np.asarray(grid.centers[0])[:, None, None]
        errors.append(float(np.max(np.abs(np.asarray(factorization.solve(source).data) - (1.0 - radius**2)))))
    orders = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert min(orders) > 1.9, orders


def test_the_polar_factorization_states_what_it_needs():
    from lmhdx.poisson import fast_diagonal_polar_poisson

    cartesian = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0))
    with pytest.raises(ValueError, match="use fast_diagonal_poisson"):
        fast_diagonal_polar_poisson(cartesian, (WALL, WRAPPED, WRAPPED))
    grid = _polar_grid(4, 8)
    with pytest.raises(ValueError, match="azimuth of a polar grid is periodic"):
        fast_diagonal_polar_poisson(grid, (WALL, WALL, WRAPPED))
    from lmhdx.grid import POLAR
    from lmhdx.poisson import azimuthal_eigenvalues

    stretched = Grid(
        uniform_faces(4, 0.0, 1.0),
        np.array([0.0, 1.0, 3.0, 2.0 * np.pi]),
        uniform_faces(2, 0.0, 1.0),
        geometry=POLAR,
    )
    with pytest.raises(ValueError, match="needs a uniform azimuth"):
        azimuthal_eigenvalues(stretched)


# --- Mixed precision: float32 contractions, float64 residual corrections (D16) ---


def _layer_grid(cells: int, layer: float, axial: int = 4) -> Grid:
    """A duct cross-section resolving a wall layer of ``layer``; 1e-3 at 64 cells is width ratio 2,831."""
    from lmhdx.grid import wall_resolving_faces

    faces = wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=layer, cells_in_layer=6, max_ratio=None)
    return Grid(uniform_faces(axial, 0.0, 1.0), faces, faces)


def _relative(candidate, reference) -> float:
    return float(jnp.max(jnp.abs(candidate - reference)) / jnp.max(jnp.abs(reference)))


@pytest.mark.parametrize(("cells", "layer"), [(32, 0.05), (64, 1.0e-3)], ids=["ha20-layer", "ha1000-layer"])
def test_mixed_precision_poisson_reaches_the_float64_solve(true_float32_matmuls, cells, layer):
    """The singular pressure operator of a duct, periodic along it and insulating across."""
    grid, conditions = _layer_grid(cells, layer), (WRAPPED, WALL, WALL)
    rhs = _random_cells(grid, seed=23)
    reference = fast_diagonal_poisson(grid, conditions).solve(rhs)
    mixed = fast_diagonal_poisson(grid, conditions, precision="mixed").solve(rhs)
    assert mixed.dtype == jnp.float64
    assert _relative(mixed.data, reference.data) < 1e-10
    volumes = jnp.asarray(grid.cell_volumes())
    assert abs(float(jnp.sum(volumes * mixed.data))) < 1e-14 * float(jnp.sum(volumes * jnp.abs(mixed.data)))


def test_mixed_precision_is_as_accurate_as_float64_where_float64_is_not_exact(true_float32_matmuls):
    """At width ratio 3e5 the float64 solve is itself 1e-7 off; mixed lands on its float64 correction."""
    grid, conditions = _layer_grid(24, 1.0e-3), (WRAPPED, WALL, WALL)
    exact = fast_diagonal_poisson(grid, conditions)
    rhs = _random_cells(grid, seed=37)
    first = exact.solve(rhs)
    volumes = jnp.asarray(grid.cell_volumes())
    defect = rhs.data - laplacian(first, conditions).data
    corrected = first.data + exact.solve(rhs.replace_data(defect)).data
    corrected = corrected - jnp.sum(volumes * corrected) / jnp.sum(volumes)
    assert _relative(corrected, first.data) > 1e-9
    mixed = fast_diagonal_poisson(grid, conditions, precision="mixed").solve(rhs)
    assert _relative(mixed.data, corrected) < 1e-12


@pytest.mark.parametrize("component", [0, 1, 2], ids=["u", "v", "w"])
def test_mixed_precision_helmholtz_reaches_the_float64_solve(true_float32_matmuls, component):
    """The implicit viscous solve of each velocity component on the Ha 1000 layer mesh."""
    from lmhdx.core3d import velocity_offset
    from lmhdx.poisson import fast_diagonal_helmholtz

    grid, conditions = _layer_grid(64, 1.0e-3), (WRAPPED, FIXED, FIXED)
    offset = velocity_offset(component)
    # Ha 20 across y at dt 2e-3: the shift carries the damping of the u and w components.
    settings = dict(shift=1.0 + 2.0e-3 * (0.0 if component == 1 else 400.0), coefficient=2.0e-3)
    rhs = Field(
        jnp.asarray(np.random.default_rng(component).normal(size=grid.offset_shape(offset))), offset, grid
    )
    reference = fast_diagonal_helmholtz(grid, offset, conditions, **settings).solve(rhs)
    mixed = fast_diagonal_helmholtz(grid, offset, conditions, precision="mixed", **settings).solve(rhs)
    assert _relative(mixed.data, reference.data) < 1e-10
    assert np.all(np.asarray(mixed.data)[reference.data == 0.0] == 0.0)


@pytest.mark.parametrize("radial_condition", [WALL, FIXED], ids=["neumann", "dirichlet"])
def test_mixed_precision_polar_poisson_reaches_the_float64_solve(true_float32_matmuls, radial_condition):
    """A pipe section resolving the Ha 1000 layer, singular under the insulating wall."""
    from lmhdx.pipe import pipe_grid
    from lmhdx.poisson import fast_diagonal_polar_poisson

    grid, conditions = pipe_grid(48, 32, 1000.0), (radial_condition, WRAPPED, WRAPPED)
    rhs = _random_cells(grid, seed=41)
    reference = fast_diagonal_polar_poisson(grid, conditions).solve(rhs)
    mixed = fast_diagonal_polar_poisson(grid, conditions, precision="mixed").solve(rhs)
    assert _relative(mixed.data, reference.data) < 1e-10


def test_mixed_precision_differentiates_like_float64(true_float32_matmuls):
    """Reverse mode runs the transposed corrections, which converge like the forward ones."""
    grid, conditions = _layer_grid(64, 1.0e-3), (WRAPPED, WALL, WALL)
    weights = _random_cells(grid, seed=43).data

    def functional(factorization):
        def value(values):
            return jnp.sum(weights * factorization.solve(Field(values, (CENTER,) * 3, grid)).data ** 2)

        return value

    values = _random_cells(grid, seed=47).data
    reference = jax.grad(functional(fast_diagonal_poisson(grid, conditions)))(values)
    mixed = jax.jit(jax.grad(functional(fast_diagonal_poisson(grid, conditions, precision="mixed"))))(values)
    assert _relative(mixed, reference) < 1e-8


def test_mixed_precision_leaves_float32_states_alone():
    """A float32 right-hand side takes the plain float32 solve, bit for bit."""
    from lmhdx.pipe import pipe_grid
    from lmhdx.poisson import fast_diagonal_helmholtz, fast_diagonal_polar_poisson

    grid, polar = _layer_grid(24, 0.05), pipe_grid(12, 8, 20.0)
    builders = [
        (lambda **option: fast_diagonal_poisson(grid, (WRAPPED, WALL, WALL), **option), grid, (CENTER,) * 3),
        (
            lambda **option: fast_diagonal_helmholtz(
                grid, (CENTER, 0.0, CENTER), (WRAPPED, FIXED, FIXED), **option
            ),
            grid,
            (CENTER, 0.0, CENTER),
        ),
        (
            lambda **option: fast_diagonal_polar_poisson(polar, (WALL, WRAPPED, WRAPPED), **option),
            polar,
            (CENTER,) * 3,
        ),
    ]
    for build, where, offset in builders:
        data = np.random.default_rng(3).normal(size=where.offset_shape(offset))
        rhs = Field(jnp.asarray(data, dtype=jnp.float32), offset, where)
        state, mixed = build().solve(rhs), build(precision="mixed").solve(rhs)
        assert mixed.dtype == state.dtype
        assert np.array_equal(np.asarray(mixed.data), np.asarray(state.data))


def test_walls_resolved_in_cells_solve_the_dense_conductivity_jump():
    """The Kronecker sum with wall nodes is the five-point operator of the fluid and wall cells together."""
    from lmhdx.poisson import fast_diagonal_thin_wall_poisson

    ny, nz = 10, 8
    fy, fz = geometric_faces(ny, -1.0, 1.0, 1.2), uniform_faces(nz, -1.0, 1.0)
    grid = Grid(uniform_faces(1, 0.0, 1.0), fy, fz)
    lower, upper = (5.0, (0.03, 0.04, 0.05)), (0.5, (0.1, 0.1))
    periodic, insulating = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
    solver = fast_diagonal_thin_wall_poisson(
        grid, (periodic, insulating, insulating), (0.0, 0.0, 0.0), layers=(None, (lower, upper), None)
    )
    volumes = np.diff(fy)[:, None] * np.diff(fz)[None, :]
    rhs = np.random.default_rng(0).standard_normal((ny, nz))
    rhs -= np.sum(rhs * volumes) / np.sum(volumes)
    potential, walls = solver.solve_with_walls(Field(jnp.asarray(rhs[None]), (CENTER,) * 3, grid))
    widths = np.concatenate([lower[1][::-1], np.diff(fy), upper[1]])
    sigma = np.concatenate([[lower[0]] * 3, [1.0] * ny, [upper[0]] * 2])
    rows, hz = widths.size, np.diff(fz)
    matrix = np.zeros((rows * nz, rows * nz))
    for i in range(rows):
        for k in range(nz):
            links = []
            if i + 1 < rows:
                links.append(
                    (
                        (i + 1) * nz + k,
                        hz[k] / (0.5 * widths[i] / sigma[i] + 0.5 * widths[i + 1] / sigma[i + 1]),
                    )
                )
            if k + 1 < nz:
                links.append((i * nz + k + 1, sigma[i] * widths[i] / (0.5 * (hz[k] + hz[k + 1]))))
            for other, link in links:
                for a, b in ((i * nz + k, other), (other, i * nz + k)):
                    matrix[a, a] -= link
                    matrix[a, b] += link
    source = np.zeros((rows, nz))
    source[3 : 3 + ny] = rhs * volumes
    dense = np.linalg.lstsq(matrix, source.ravel(), rcond=None)[0].reshape(rows, nz)
    dense -= np.sum(dense[3 : 3 + ny] * volumes) / np.sum(volumes)
    np.testing.assert_allclose(potential.data[0], dense[3 : 3 + ny], atol=1e-11 * np.max(np.abs(dense)))
    # The reported interface potential carries the series current of the two half cells.
    series = 1.0 / (0.5 * np.diff(fy)[0] + 0.5 * 0.03 / 5.0)
    reported = (walls[1].data[0, 0] - potential.data[0, 0]) / (0.5 * np.diff(fy)[0])
    np.testing.assert_allclose(reported, series * (dense[2] - dense[3]), rtol=1e-10)
    assert walls[0] is None and walls[2] is None
    with pytest.raises(ValueError, match="no thin wall"):
        fast_diagonal_thin_wall_poisson(
            grid, (periodic, insulating, insulating), (0.0, 0.0, 0.1), layers=(None, (lower, None), None)
        )


def test_resolved_walls_on_two_axes_give_each_corner_cell_its_nearer_wall():
    """The Woodbury corner correction against the five-point operator of fluid, walls and corners."""
    from lmhdx.poisson import fast_diagonal_thin_wall_poisson

    ny, nz = 10, 8
    fy, fz = geometric_faces(ny, -1.0, 1.0, 1.2), uniform_faces(nz, -1.0, 1.0)
    grid = Grid(uniform_faces(1, 0.0, 0.7), fy, fz)
    walls_y = (((5.0, 0.2, 0.2), (0.03, 0.04, 0.05)), (0.5, (0.1, 0.1)))
    walls_z = ((3.0, (0.05, 0.05)), ((2.0, 0.1), (0.02, 0.06)))
    periodic, insulating = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
    solver = fast_diagonal_thin_wall_poisson(
        grid, (periodic, insulating, insulating), (0.0, 0.0, 0.0), layers=(None, walls_y, walls_z)
    )
    volumes = np.diff(fy)[:, None] * np.diff(fz)[None, :]
    rhs = np.random.default_rng(0).standard_normal((ny, nz))
    rhs -= np.sum(rhs * volumes) / np.sum(volumes)
    potential = solver.solve(Field(jnp.asarray(rhs[None]), (CENTER,) * 3, grid)).data[0]

    def axis(walls, fluid):
        (low_ratio, low_widths), (high_ratio, high_widths) = walls
        widths = np.concatenate([low_widths[::-1], fluid, high_widths])
        ratios = np.concatenate(
            [
                np.broadcast_to(low_ratio, len(low_widths))[::-1],
                np.ones(fluid.size),
                np.broadcast_to(high_ratio, len(high_widths)),
            ]
        )
        centres = np.cumsum(widths) - 0.5 * widths
        lower, upper = (
            centres[len(low_widths)] - 0.5 * fluid[0],
            centres[-len(high_widths) - 1] + 0.5 * fluid[-1],
        )
        depth = np.maximum(lower - centres, 0.0) + np.maximum(centres - upper, 0.0)
        return widths, ratios, depth, len(low_widths)

    hy, sy, dy, ly = axis(walls_y, np.diff(fy))
    hz, sz, dz, lz = axis(walls_z, np.diff(fz))
    wy, wz = dy[:, None] > 0, dz[None, :] > 0
    sigma = np.where(wy & wz, np.where(dy[:, None] <= dz[None, :], sy[:, None], sz[None, :]), 1.0)
    sigma = np.where(wy & ~wz, sy[:, None], np.where(wz & ~wy, sz[None, :], sigma))
    rows, columns = hy.size, hz.size
    matrix = np.zeros((rows * columns, rows * columns))
    for i in range(rows):
        for k in range(columns):
            for j, m, area, h0, h1 in (
                (i + 1, k, hz[k], hy[i], hy[min(i + 1, rows - 1)]),
                (i, k + 1, hy[i], hz[k], hz[min(k + 1, columns - 1)]),
            ):
                if j < rows and m < columns:
                    link = area / (0.5 * h0 / sigma[i, k] + 0.5 * h1 / sigma[j, m])
                    for a, b in ((i * columns + k, j * columns + m), (j * columns + m, i * columns + k)):
                        matrix[a, a] -= link
                        matrix[a, b] += link
    source = np.zeros((rows, columns))
    source[ly : ly + ny, lz : lz + nz] = rhs * volumes
    dense = np.linalg.lstsq(matrix, source.ravel(), rcond=None)[0].reshape(rows, columns)[
        ly : ly + ny, lz : lz + nz
    ]
    dense -= np.sum(dense * volumes) / np.sum(volumes)
    np.testing.assert_allclose(potential, dense, atol=1e-12 * np.max(np.abs(dense)))
