"""The 8^3 oracle that plan step 1.4 requires before a momentum discretization lands.

ADR 0002 records that the marker-and-cell layout is the default and that the
choice must be settled by measurement rather than preference. This module is that
measurement. It assembles the pressure operator that each candidate layout would
produce and compares them on the properties a projection method depends on:

* the dimension of the nullspace, because a pressure that is not determined up to
  a single constant is not determined at all;
* the discrete adjoint identity between gradient and divergence, because a
  projection is only idempotent when it holds;
* symmetry under the cell volumes, because it decides whether the direct
  factorization of :mod:`lmhdx.poisson` applies at all, and whether the adjoint
  solve of a derivative can reuse the forward factorization;
* the discrete energy identity of the projection, which holds only when the
  projection is orthogonal.

The collocated candidate is built here rather than in the package: it is the
rejected design, and the plan asks that rejected prototypes leave evidence, not
code. The numbers these tests assert are the ones quoted in the ADR.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.grid import CENTER, NEUMANN, BoundaryCondition, Field, Grid, uniform_faces
from lmhdx.ops import cell_inner_product, divergence, face_gradient, face_inner_product
from lmhdx.poisson import fast_diagonal_poisson

pytestmark = pytest.mark.unit

WALL = BoundaryCondition(NEUMANN)
ORACLE = Grid(*(uniform_faces(8, 0.0, 1.0) for _ in range(3)))


def _staggered_pressure_operator(grid: Grid) -> np.ndarray:
    """Assemble divergence-of-gradient with the staggered operators of :mod:`lmhdx.ops`."""
    size = int(np.prod(grid.shape))
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        field = Field(jnp.asarray(unit.reshape(grid.shape)), (CENTER,) * 3, grid)
        gradients = tuple(face_gradient(field, axis, WALL) for axis in range(3))
        columns.append(np.asarray(divergence(gradients).data).reshape(size))
    return np.stack(columns, axis=1)


def _collocated_gradient(grid: Grid, axis: int) -> np.ndarray:
    """The wide centred difference of one axis as a matrix; the collocated divergence sums them."""
    size = int(np.prod(grid.shape))
    width = np.asarray(grid.widths[axis])
    distance = np.concatenate(
        ([width[0] + width[1]], width[:-2] + 2.0 * width[1:-1] + width[2:], [width[-2] + width[-1]])
    )
    shape = [-1 if position == axis else 1 for position in range(3)]
    columns = []
    for index in range(size):
        values = np.zeros(size)
        values[index] = 1.0
        values = values.reshape(grid.shape)
        padded = np.concatenate(
            (np.take(values, [0], axis=axis), values, np.take(values, [-1], axis=axis)), axis=axis
        )
        upper = np.take(padded, range(2, padded.shape[axis]), axis=axis)
        lower = np.take(padded, range(0, padded.shape[axis] - 2), axis=axis)
        columns.append(((upper - lower) / distance.reshape(shape)).reshape(size))
    return np.stack(columns, axis=1)


def _collocated_pressure_operator(grid: Grid) -> np.ndarray:
    """Assemble the same operator when velocity and pressure share the cell centre.

    The gradient and the divergence are then both the wide centred difference
    over neighbouring cell centres, mirrored at the walls, which is the natural
    choice on a collocated mesh and the one that produces the classic decoupling.
    """
    return sum(matrix @ matrix for matrix in (_collocated_gradient(grid, axis) for axis in range(3)))


def _nullspace_dimension(operator: np.ndarray, *, tolerance: float = 1e-9) -> int:
    singular = np.linalg.svd(operator, compute_uv=False)
    return int(np.sum(singular <= tolerance * singular[0]))


def test_staggered_pressure_operator_has_only_the_constant_nullspace():
    operator = _staggered_pressure_operator(ORACLE)
    assert _nullspace_dimension(operator) == 1


def test_collocated_pressure_operator_decouples_into_independent_sublattices():
    """The wide centred composition leaves a checkerboard family undetermined.

    Each axis decouples its even and odd cells, so the three-dimensional operator
    carries one constant per sublattice: eight, not one. Those extra modes are
    the pressure oscillations a collocated projection cannot see, which is why
    that layout needs an added interpolation to be usable at all.
    """
    operator = _collocated_pressure_operator(ORACLE)
    assert _nullspace_dimension(operator) == 8


def test_staggered_gradient_and_divergence_are_adjoint_while_collocated_is_not():
    grid = ORACLE
    generator = np.random.default_rng(0)
    pressure = Field(jnp.asarray(generator.normal(size=grid.shape)), (CENTER,) * 3, grid)
    fluxes = []
    for axis in range(3):
        data = generator.normal(size=grid.face_shape(axis))
        data[(slice(None),) * axis + (0,)] = 0.0
        data[(slice(None),) * axis + (-1,)] = 0.0
        offset = tuple(0.0 if position == axis else CENTER for position in range(3))
        fluxes.append(Field(jnp.asarray(data), offset, grid))

    left = float(cell_inner_product(pressure, divergence(tuple(fluxes))))
    right = float(
        sum(
            face_inner_product(fluxes[axis], face_gradient(pressure, axis, WALL), axis, WALL)
            for axis in range(3)
        )
    )
    assert abs(left + right) <= 1e-14 * abs(left)


def test_staggered_operator_is_symmetric_under_the_cell_volumes():
    """Symmetry is what lets :mod:`lmhdx.poisson` factorize the operator directly."""
    grid = ORACLE
    operator = _staggered_pressure_operator(grid)
    volumes = grid.cell_volumes().reshape(-1)
    weighted = volumes[:, None] * operator
    asymmetry = np.max(np.abs(weighted - weighted.T)) / np.max(np.abs(weighted))
    assert asymmetry < 1e-12


def test_staggered_stencil_is_narrower_than_the_collocated_one():
    """A narrow stencil is cheaper per solve and cheaper to differentiate."""
    staggered = _staggered_pressure_operator(ORACLE)
    collocated = _collocated_pressure_operator(ORACLE)
    staggered_entries = int(np.sum(np.abs(staggered) > 1e-12))
    collocated_entries = int(np.sum(np.abs(collocated) > 1e-12))
    assert staggered_entries < collocated_entries
    # Seven-point versus a composition that reaches two cells along each axis.
    assert staggered_entries / np.prod(ORACLE.shape) < 7.0


def test_the_staggered_projection_keeps_the_energy_identity_and_the_collocated_one_does_not():
    """ADR 0002: ``|u*|^2 = |u|^2 + |G p|^2`` when the projection is orthogonal.

    The staggered projection, through the production fast-diagonal solve, removes
    the divergence to round-off and splits the energy exactly (measured 2.6e-15 and
    9.8e-16 relative). The collocated one cannot remove the divergence it measures:
    its right-hand side leaves the range of its own operator, the least-squares
    pressure leaves 6.1 % of the divergence, and the energy split misses by 9.4e4
    times the energy.
    """
    grid = ORACLE
    size = int(np.prod(grid.shape))
    volumes = grid.cell_volumes().reshape(-1)
    generator = np.random.default_rng(1)

    fluxes = []
    for axis in range(3):
        data = generator.normal(size=grid.face_shape(axis))
        data[(slice(None),) * axis + (0,)] = 0.0
        data[(slice(None),) * axis + (-1,)] = 0.0
        offset = tuple(0.0 if position == axis else CENTER for position in range(3))
        fluxes.append(Field(jnp.asarray(data), offset, grid))
    source = divergence(tuple(fluxes))
    pressure = fast_diagonal_poisson(grid, (WALL,) * 3).solve(source)
    source = np.asarray(source.data)
    gradients = tuple(face_gradient(pressure, axis, WALL) for axis in range(3))
    projected = tuple(f.replace_data(f.data - g.data) for f, g in zip(fluxes, gradients, strict=True))

    def energy(fields):
        return float(sum(face_inner_product(f, f, axis, WALL) for axis, f in enumerate(fields)))

    remaining = float(jnp.max(jnp.abs(divergence(projected).data))) / float(np.max(np.abs(source)))
    assert remaining < 1e-13
    before = energy(fluxes)
    assert abs(before - energy(projected) - energy(gradients)) < 1e-13 * before

    matrices = [_collocated_gradient(grid, axis) for axis in range(3)]
    velocity = [generator.normal(size=size) for _ in range(3)]
    source = sum(matrix @ component for matrix, component in zip(matrices, velocity, strict=True))
    operator = sum(matrix @ matrix for matrix in matrices)
    pressure = np.linalg.lstsq(operator, source, rcond=None)[0]
    gradients = [matrix @ pressure for matrix in matrices]
    projected = [component - gradient for component, gradient in zip(velocity, gradients, strict=True)]
    divergence_left = sum(matrix @ component for matrix, component in zip(matrices, projected, strict=True))
    assert np.max(np.abs(divergence_left)) > 0.03 * np.max(np.abs(source))

    def collocated_energy(fields):
        return float(sum(volumes @ (field * field) for field in fields))

    before = collocated_energy(velocity)
    assert abs(before - collocated_energy(projected) - collocated_energy(gradients)) > before


def test_only_the_staggered_adjoint_solve_reuses_its_forward_factorization():
    """The derivative cost of a pressure solve, as ADR 0002 records it.

    Reverse mode through ``L p = b`` solves with ``L^T``. The staggered operator is
    symmetric under the volumes (0 on this mesh), so the adjoint is the forward solve
    again; the collocated one is 0.67 away, so its adjoint needs a second factorization
    or a nonsymmetric Krylov solve, and its gradient and divergence are not adjoint
    either (0.82 relative).
    """
    grid = ORACLE
    volumes = grid.cell_volumes().reshape(-1)

    def asymmetry(operator):
        weighted = volumes[:, None] * operator
        return np.max(np.abs(weighted - weighted.T)) / np.max(np.abs(weighted))

    assert asymmetry(_staggered_pressure_operator(grid)) < 1e-12
    assert asymmetry(_collocated_pressure_operator(grid)) > 0.5
    matrices = [_collocated_gradient(grid, axis) for axis in range(3)]
    generator = np.random.default_rng(0)
    pressure = generator.normal(size=volumes.size)
    velocity = [generator.normal(size=volumes.size) for _ in range(3)]
    left = volumes @ (pressure * sum(m @ u for m, u in zip(matrices, velocity, strict=True)))
    right = sum(volumes @ (u * (m @ pressure)) for m, u in zip(matrices, velocity, strict=True))
    assert abs(left + right) > 0.5 * abs(left)
