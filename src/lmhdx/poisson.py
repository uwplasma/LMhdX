"""Direct Poisson solves by fast diagonalization on a tensor-product grid.

The finite-volume Laplacian of :mod:`lmhdx.ops` separates: on a tensor-product
grid it is the Kronecker sum of three one-dimensional operators. Each of those is
symmetric once the cell widths are folded in, so diagonalizing them on the host
turns a Poisson solve into three tensor contractions and one elementwise divide.
The cost is a handful of matrix multiplies rather than an iteration whose count
depends on the Hartmann number, the result is exact to round-off rather than to a
tolerance, and the solve is a linear map, so it differentiates for free.

The one-dimensional operators are read out of :mod:`lmhdx.ops` itself, by applying
the assembled Laplacian to unit vectors on a grid that is one cell wide across
the other two axes. The factorization therefore cannot drift away from the
stencil the rest of the code uses.

A pure Neumann or fully periodic problem determines its solution only up to a
constant. The compatible component is removed from the right-hand side and the
returned field has zero volume-weighted mean.

The factorization represents the homogeneous operator. Inhomogeneous boundary
data is affine, not linear, so it belongs in the right-hand side; passing a
condition that carries a value is refused rather than silently linearized.

``precision="mixed"`` runs the contractions of a float64 solve in float32 and
recovers float64 accuracy by defect correction: the residual against the
assembled operator is formed in float64 and solved again in float32,
``refinements`` times (:func:`solvax.iterative_refinement`). The orthogonal
transforms keep their relative accuracy mode by mode, so the shifted viscous
operator reaches the float64 floor after one correction; the singular
Laplacian on a layer-resolving mesh contracts more slowly and takes two, which
are the factory defaults. The contraction assumes true float32
matmuls; on Ampere GPUs pin ``jax_default_matmul_precision`` to ``"float32"``,
because TensorFloat-32 stalls the correction near 1e-4. Float32 states are
solved in float32 as before.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from ._programs import attribute, grid_program, host_array
from .bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition
from .grid import CENTER, FACE, POLAR, Field, Grid, uniform_faces
from .ops import foldable, laplacian, staggered_laplacian

__all__ = [
    "FastDiagonalPolarPoisson",
    "assemble_radial_laplacian",
    "azimuthal_eigenvalues",
    "fast_diagonal_polar_poisson",
    "FastDiagonalHelmholtz",
    "FastDiagonalPoisson",
    "FastDiagonalThinWallPoisson",
    "assemble_axis_laplacian",
    "assemble_staggered_axis_operator",
    "assemble_thick_wall_operator",
    "assemble_thin_wall_operator",
    "fast_diagonal_helmholtz",
    "fast_diagonal_poisson",
    "fast_diagonal_thin_wall_poisson",
    "free_slice",
]

_SINGULAR_TOLERANCE = 1.0e-9
_AXIS_SHAPES = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
_PRECISIONS = ("state", "mixed")


def _check_precision(precision: str, refinements: int) -> None:
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {list(_PRECISIONS)}, got {precision!r}")
    if int(refinements) != refinements or refinements < 1:
        raise ValueError("refinements must be a positive integer")


def _single(array: np.ndarray) -> np.ndarray:
    """Cast a host array to float32 once, at factorization, rather than per solve."""
    return np.ascontiguousarray(array, dtype=np.float32)


def _refined(factorization, direct, matvec, data: jnp.ndarray, volumes: tuple | None) -> jnp.ndarray:
    """Return ``direct(data)`` in the precision the factorization asks for.

    A float64 right-hand side under ``precision="mixed"`` is solved in float32
    and corrected against ``matvec`` in float64; anything else is solved
    directly in its own precision. ``volumes`` marks a singular operator: the
    incompatible mean is removed from the right-hand side, from every residual
    and from the result in float64, where the float32 solve would lose it:
    ``(owner, build)``, the host builder of the volumes (:func:`lmhdx._programs.host_array`).
    """
    if factorization.precision != "mixed" or data.dtype != jnp.float64:
        return direct(data)
    operator = matvec
    if volumes is not None:
        data = _volume_mean_removed(data, volumes)

        def matvec(values):
            return _volume_mean_removed(operator(values), volumes)

    def single(values):
        return direct(values, single=True)

    solution, _ = solvax.iterative_refinement(
        matvec,
        data,
        solvax.as_low_precision(single, jnp.float32),
        iterations=int(factorization.refinements),
        residual_dtype=data.dtype,
    )
    return solution if volumes is None else _volume_mean_removed(solution, volumes)


def _corrected(factorization, direct, matvec, data, result, volumes: tuple | None) -> jnp.ndarray:
    """Apply ``factorization.corrections`` float64 defect corrections to a ``"state"`` solve.

    The lowest mode of a long inflow-outflow axis sits nine decades below the largest
    across a Hartmann layer, and the transforms lose that ratio in round-off: at Ha 100 the
    pressure drop drifted at 1e-8 and the charge balance at 1e-9 without it.
    """
    if factorization.precision != "state":
        return result
    for _ in range(factorization.corrections):
        defect = data - matvec(result)
        result = result + direct(defect if volumes is None else _volume_mean_removed(defect, volumes))
    return (
        result if volumes is None or not factorization.corrections else _volume_mean_removed(result, volumes)
    )


def _reciprocal(values: np.ndarray) -> np.ndarray:
    """``1 / values`` on the host, zero where a singular mode's value is (its entry is overwritten)."""
    values = np.asarray(values)
    return np.divide(1.0, values, out=np.zeros_like(values), where=values != 0.0).astype(values.dtype)


def _volume_mean_removed(values: jnp.ndarray, volumes: tuple) -> jnp.ndarray:
    weights = host_array(*volumes, dtype=values.dtype)
    return values - jnp.sum(weights * values) / jnp.sum(weights)


def _cell_volumes(grid: Grid) -> np.ndarray:
    return grid.cell_volumes()


def _own_measure(factorization) -> np.ndarray:
    return factorization._measure()


def _normalized_measure(factorization) -> np.ndarray:
    measure = factorization._measure()
    return measure / np.sum(measure)


def _low_entry(factorization, name: str, axis: int | None = None) -> np.ndarray:
    entry = factorization._low[name]
    return entry if axis is None else entry[axis]


def _inverse_denominator(factorization, single: bool) -> np.ndarray:
    """The stored reciprocal of the eigenvalue denominator of a fast-diagonal solve."""
    denominator = np.asarray(factorization._low["denominator"]) if single else factorization._denominator()
    return _reciprocal(denominator)


def _scale_product(factorization, inverse: bool) -> np.ndarray:
    factors = [
        (1.0 / scale if inverse else scale).reshape(shape)
        for scale, shape in zip(factorization.scales, _AXIS_SHAPES)
    ]
    return factors[0] * factors[1] * factors[2]


def _axis_scale(factorization, axis: int, inverse: bool) -> np.ndarray:
    scale = factorization.scales[axis]
    factor = 1.0 / scale if inverse else scale
    return factor.reshape([-1 if position == axis else 1 for position in range(3)])


def _basis(factorization, axis: int, transpose: bool) -> np.ndarray:
    vector = factorization.vectors[axis]
    return vector.T if transpose else vector


def _single_bases(vectors, scales, denominator: np.ndarray) -> dict:
    """Return the float32 copies a mixed-precision Cartesian solve contracts with."""
    shapes = [tuple(-1 if position == axis else 1 for position in range(3)) for axis in range(3)]
    return {
        "vectors": tuple(_single(vector) for vector in vectors),
        "transposed": tuple(_single(vector.T) for vector in vectors),
        "scales": tuple(_single(scale.reshape(shape)) for scale, shape in zip(scales, shapes, strict=True)),
        "inverse": tuple(
            _single((1.0 / scale).reshape(shape)) for scale, shape in zip(scales, shapes, strict=True)
        ),
        "denominator": _single(denominator),
    }


def _scaled(factorization, data: jnp.ndarray, single: bool, *, inverse: bool) -> jnp.ndarray:
    if not single and foldable(data.shape):
        # One stored factor: the product of the three axis scales, formed on the host.
        return data * host_array(factorization, _scale_product, inverse, dtype=data.dtype)
    for axis in range(3):
        if not single:
            data = data * host_array(factorization, _axis_scale, axis, inverse, dtype=data.dtype)
        else:
            data = data * host_array(factorization, _low_entry, "inverse" if inverse else "scales", axis)
    return data


def _contracted(factorization, data: jnp.ndarray, single: bool, *, transpose: bool) -> jnp.ndarray:
    for axis in range(3):
        if not single:
            matrix = host_array(factorization, _basis, axis, transpose, dtype=data.dtype)
        else:
            matrix = host_array(factorization, _low_entry, "transposed" if transpose else "vectors", axis)
        data = jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)
    return data


def _relative_asymmetry(operator: np.ndarray) -> float:
    """Return the asymmetry of an operator against its own diagonal scale.

    A wall-resolving mesh can span four orders of magnitude in cell width, and
    the entries of the operator span eight. Measured against the largest entry,
    the round-off of the small rows looks like a defect; measured against
    ``sqrt(|d_i d_j|)``, the natural scale of the entry itself, it does not.
    A stencil that is genuinely not symmetric is wrong by an order one fraction
    of its own entries, so this separates the two cleanly.
    """
    diagonal = np.sqrt(np.abs(np.diag(operator)))
    scale = np.outer(diagonal, diagonal)
    floor = max(float(np.max(scale)), 1.0) * float(np.finfo(operator.dtype).eps)
    return float(np.max(np.abs(operator - operator.T) / np.maximum(scale, floor)))


def _symmetry_tolerance(operator: np.ndarray) -> float:
    """Return the relative asymmetry a correct assembly may still show.

    The one-dimensional operators are read out through the production stencil,
    so they are assembled at whatever precision the session runs in. A bound
    tied to float64 would reject a perfectly good float32 assembly, so the
    tolerance follows the dtype.
    """
    return max(1.0e-8, 1.0e11 * float(np.finfo(operator.dtype).eps))


def _require_separable(grid: Grid) -> None:
    if grid.is_polar:
        raise ValueError(
            "fast diagonalization assumes the Laplacian separates into a sum of one-dimensional "
            "operators, which the 1/r^2 azimuthal term of a polar grid does not"
        )


def _probe(
    apply, key: tuple, line: Grid, offset: tuple[float, float, float], positions, selection=slice(None)
) -> np.ndarray:
    """Return the matrix of a linear stencil, one column per unit vector at ``positions``.

    All unit vectors go through the stencil in one batched, compiled call on the
    host CPU. Probing them one by one as eager operations dispatched hundreds of
    small kernels, which was most of the cold start (4 s on a CPU, 20 s on a GPU
    host, for a 48-cell duct). ``key`` names the stencil ``apply`` applies.
    """
    units = np.zeros((len(positions),) + line.offset_shape(offset))
    for column, position in enumerate(positions):
        units[(column,) + tuple(position)] = 1.0

    def build(grid: Grid):
        return jax.vmap(lambda data: apply(Field(data, offset, grid)).data)

    program = (key, line.shape, line.geometry, offset, tuple(map(tuple, positions)), "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        # One program per stencil shape, the line's metric its arguments: a new mesh is not traced (2b.1).
        applied = grid_program(program, build, line, jax.ShapeDtypeStruct(units.shape, units.dtype))(units)
    return np.asarray(applied).reshape(len(positions), -1)[:, selection].T


def assemble_axis_laplacian(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    """Return the dense one-dimensional Laplacian :mod:`lmhdx.ops` applies along ``axis``.

    The other two axes are collapsed to a single cell with a homogeneous
    Neumann condition, which contributes nothing, so the result is exactly the
    stencil the three-dimensional operator uses along ``axis``.
    """
    _require_separable(grid)
    faces = [uniform_faces(1, 0.0, 1.0)] * 3
    faces[axis] = np.asarray(grid.faces[axis])
    return _axis_laplacian(Grid(*faces), axis, condition).copy()


@functools.lru_cache(maxsize=64)
def _axis_laplacian(line: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    conditions = tuple(
        condition if position == axis else BoundaryCondition("neumann") for position in range(3)
    )
    positions = [(0,) * axis + (index,) + (0,) * (2 - axis) for index in range(line.shape[axis])]
    return _probe(
        lambda field: laplacian(field, conditions), ("laplacian", conditions), line, (CENTER,) * 3, positions
    )


def assemble_radial_laplacian(grid: Grid, condition: BoundaryCondition) -> np.ndarray:
    """Return the dense radial Laplacian the polar flux form applies.

    The azimuth is collapsed to a single periodic cell, which contributes
    nothing because its two faces carry the same value, and the axial direction
    to a single cell with a homogeneous Neumann condition. The metric factors of
    the azimuth and the axis cancel between the face areas and the cell volume,
    so what is left is exactly ``(1/r) d/dr (r d/dr)`` as the production stencil
    discretizes it, including the zero-area face on the axis.
    """
    line = Grid(
        np.asarray(grid.x_faces),
        uniform_faces(1, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )
    return _radial_laplacian(line, condition).copy()


@functools.lru_cache(maxsize=16)
def _radial_laplacian(line: Grid, condition: BoundaryCondition) -> np.ndarray:
    conditions = (condition, BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN))
    positions = [(index, 0, 0) for index in range(line.shape[0])]
    return _probe(
        lambda field: laplacian(field, conditions), ("laplacian", conditions), line, (CENTER,) * 3, positions
    )


def azimuthal_eigenvalues(grid: Grid) -> np.ndarray:
    """Return the eigenvalue of the azimuthal second difference for every mode.

    The discrete Fourier basis diagonalizes a uniform periodic second
    difference, so mode ``m`` contributes ``-4 sin^2(pi m / N) / dtheta^2``
    divided by ``r^2``. That last division is what stops the polar Laplacian
    separating into a sum of one-dimensional operators -- and what makes it
    separate again once the azimuth is transformed, one radial operator per mode.
    """
    widths = np.asarray(grid.widths[1])
    if not np.allclose(widths, widths[0]):
        raise ValueError("the azimuthal transform needs a uniform azimuth")
    count = grid.shape[1]
    modes = np.arange(count)
    return -4.0 * np.sin(np.pi * modes / count) ** 2 / float(widths[0]) ** 2


@dataclass(frozen=True)
class FastDiagonalPolarPoisson:
    """A factorized polar Laplacian: one radial eigendecomposition per azimuthal mode."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    radial_vectors: np.ndarray
    radial_values: np.ndarray
    radial_scale: np.ndarray
    axial_vectors: np.ndarray
    axial_values: np.ndarray
    axial_scale: np.ndarray
    singular: bool
    shift: float = 0.0
    coefficient: float = -1.0
    precision: str = "state"
    refinements: int = 2
    wall_conductance: float = 0.0
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.wall_conductance and self.precision == "mixed":
            raise ValueError("the polar thin-wall factorization solves in the precision of its state")
        if self.precision == "mixed":
            total = self.radial_values[:, :, None] + self.axial_values[None, None, :]
            denominator = np.moveaxis(self.shift - self.coefficient * total, 0, 1)
            if self.singular:
                denominator[0, 0, 0] = 1.0
            low = {"radial": _single(self.radial_vectors), "axial": _single(self.axial_vectors)}
            object.__setattr__(self, "_low", low | {"denominator": _single(denominator)})

    def solve(self, rhs: Field) -> Field:
        """Return the field this operator maps to ``rhs``.

        The operator is ``shift*I - coefficient*laplacian``; the Poisson factory
        passes ``(0, -1)``, which leaves the Laplacian itself.
        """
        if rhs.grid != self.grid:
            raise ValueError("right-hand side must share the factorized grid")
        if rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("the polar factorization solves for a cell-centred field")

        def operator(values):
            return (
                self.shift * values
                - self.coefficient * laplacian(rhs.replace_data(values), self.conditions).data
            )

        if self.wall_conductance:
            return self.solve_with_wall(rhs)[0]
        volumes = (self.grid, _cell_volumes) if self.singular else None
        return rhs.replace_data(_refined(self, self._direct, operator, rhs.data, volumes))

    def solve_with_wall(self, rhs: Field) -> tuple[Field, Field]:
        """Return the cell potential and, on the radial faces, the outer thin wall's potential.

        As :func:`lmhdx.em.thin_wall_current` reads it: the outer entry is the sheet,
        the inner entry repeats the adjacent cell so it carries no current.
        """
        if not self.wall_conductance or rhs.grid != self.grid or rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("a thin-wall solve needs a wall and a cell-centred field on the factorized grid")
        solution = self._direct(jnp.pad(rhs.data, ((0, 1), (0, 0), (0, 0))))
        volumes = jnp.asarray(self.grid.cell_volumes(), dtype=solution.dtype)
        solution = solution - jnp.sum(volumes * solution[:-1]) / jnp.sum(volumes)
        inner = jnp.zeros_like(solution)[2:]
        wall = jnp.concatenate((solution[:1], inner, solution[-1:]), axis=0)
        return rhs.replace_data(solution[:-1]), Field(wall, (FACE, CENTER, CENTER), self.grid)

    def _measure(self) -> np.ndarray:
        """Return the weights of the unknowns: cell volumes, and ``c`` times the wall area on the sheet."""
        if not self.wall_conductance:
            return self.grid.cell_volumes()
        return self.radial_scale[:, None, None] ** 2 * np.outer(*self.grid.widths[1:])[None]

    def _direct(self, data: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        dtype = data.dtype
        low = self._low if single else None
        radial_vectors = jnp.asarray(self.radial_vectors if low is None else low["radial"])
        axial_vectors = jnp.asarray(self.axial_vectors if low is None else low["axial"])
        volumes = jnp.asarray(self._measure(), dtype=dtype)
        if self.singular:
            data = data - jnp.sum(volumes * data) / jnp.sum(volumes)
        data = data * jnp.asarray(self.radial_scale[:, None, None], dtype=dtype)
        data = data * jnp.asarray(self.axial_scale[None, None, :], dtype=dtype)
        transformed = jnp.fft.fft(data, axis=1)
        transformed = jnp.einsum("mji,jmz->imz", radial_vectors, transformed)
        transformed = jnp.tensordot(axial_vectors.T, transformed, axes=([1], [2]))
        transformed = jnp.moveaxis(transformed, 0, 2)
        if low is None:
            total = (
                jnp.asarray(self.radial_values)[:, :, None] + jnp.asarray(self.axial_values)[None, None, :]
            )
            denominator = jnp.moveaxis(self.shift - self.coefficient * total, 0, 1)
            if self.singular:
                denominator = denominator.at[0, 0, 0].set(1.0)
        else:
            denominator = jnp.asarray(low["denominator"])
        if self.singular:
            transformed = transformed.at[0, 0, 0].set(0.0)
        transformed = transformed / denominator
        restored = jnp.tensordot(axial_vectors, transformed, axes=([1], [2]))
        restored = jnp.moveaxis(restored, 0, 2)
        restored = jnp.einsum("mji,imz->jmz", radial_vectors, restored)
        solution = jnp.real(jnp.fft.ifft(restored, axis=1))
        solution = solution / jnp.asarray(self.radial_scale[:, None, None], dtype=dtype)
        solution = solution / jnp.asarray(self.axial_scale[None, None, :], dtype=dtype)
        if self.singular:
            solution = solution - jnp.sum(volumes * solution) / jnp.sum(volumes)
        return solution


def fast_diagonal_polar_poisson(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    shift: float = 0.0,
    coefficient: float = -1.0,
    precision: str = "state",
    refinements: int = 2,
    wall_conductance: float = 0.0,
) -> FastDiagonalPolarPoisson:
    """Factorize ``shift*I - coefficient*laplacian``, one eigendecomposition per azimuthal mode.

    The default leaves the Laplacian itself. A positive shift with a positive
    coefficient is the damped operator that preconditions a pipe at large
    Hartmann number. ``precision="mixed"`` solves float64 right-hand sides in
    float32 with ``refinements`` float64 corrections (module docstring).

    ``wall_conductance`` closes the outer radius of the potential Laplacian with
    a thin conducting wall: a sheet node on the radial operator
    (:func:`assemble_thin_wall_operator`) that sees the azimuthal eigenvalue at
    the wall radius, so each mode is still one eigendecomposition.
    """
    if not grid.is_polar:
        raise ValueError("this factorization is for a polar grid; use fast_diagonal_poisson")
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    if not conditions[1].is_periodic:
        raise ValueError("the azimuth of a polar grid is periodic by construction")
    radial = assemble_radial_laplacian(grid, conditions[0])
    radial_weights = np.asarray(grid.centers[0]) * np.asarray(grid.widths[0])
    inverse_square = 1.0 / np.asarray(grid.centers[0]) ** 2
    if wall_conductance:
        if wall_conductance < 0.0 or conditions[0].kind != NEUMANN or (shift, coefficient) != (0.0, -1.0):
            raise ValueError("a thin wall closes the potential Laplacian of an insulating radial condition")
        radial, radial_weights = assemble_thin_wall_operator(grid, 0, (0.0, float(wall_conductance)))
        inverse_square = np.append(inverse_square, 1.0 / float(grid.x_faces[-1]) ** 2)
    radial_root = np.sqrt(radial_weights)
    symmetric = radial_root[:, None] * radial / radial_root[None, :]
    asymmetry = _relative_asymmetry(symmetric)
    if asymmetry > _symmetry_tolerance(symmetric):
        raise ValueError(
            f"the radial operator is not symmetric under the annular volumes "
            f"(relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
        )
    symmetric = 0.5 * (symmetric + symmetric.T)
    azimuthal = azimuthal_eigenvalues(grid)
    vectors, values = [], []
    for eigenvalue in azimuthal:
        operator = symmetric + np.diag(eigenvalue * inverse_square)
        mode_values, mode_vectors = np.linalg.eigh(operator)
        # Descending, so the zero eigenvalue of a singular mode is first, as the
        # Cartesian factorization also arranges it.
        vectors.append(mode_vectors[:, ::-1])
        values.append(mode_values[::-1])
    axial = assemble_axis_laplacian(
        Grid(*(uniform_faces(1, 0.0, 1.0),) * 2, np.asarray(grid.z_faces)), 2, conditions[2]
    )
    axial_weights = np.asarray(grid.widths[2])
    axial_root = np.sqrt(axial_weights)
    axial_symmetric = axial_root[:, None] * axial / axial_root[None, :]
    axial_values, axial_vectors = np.linalg.eigh(0.5 * (axial_symmetric + axial_symmetric.T))
    axial_values, axial_vectors = axial_values[::-1], axial_vectors[:, ::-1]
    singular = shift == 0.0 and conditions[0].kind == NEUMANN and conditions[2].is_periodic
    return FastDiagonalPolarPoisson(
        grid,
        tuple(conditions),
        np.stack(vectors),
        np.stack(values),
        radial_root,
        axial_vectors,
        axial_values,
        axial_root,
        singular,
        float(shift),
        float(coefficient),
        precision,
        refinements,
        float(wall_conductance),
    )


@dataclass(frozen=True)
class FastDiagonalPoisson:
    """A factorized Laplacian that solves ``laplacian(u) = rhs`` in three contractions."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
    scales: tuple[np.ndarray, np.ndarray, np.ndarray]
    singular: bool
    precision: str = "state"
    refinements: int = 2
    corrections: int = 0
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.precision == "mixed":
            object.__setattr__(
                self, "_low", _single_bases(self.vectors, self.scales, self._eigenvalue_total())
            )

    def solve(self, rhs: Field) -> Field:
        """Return the field whose Laplacian is ``rhs``."""
        if rhs.grid != self.grid:
            raise ValueError("right-hand side must share the factorized grid")
        if rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("right-hand side must be cell centred")

        def operator(values):
            return laplacian(rhs.replace_data(values), self.conditions).data

        volumes = (self.grid, _cell_volumes) if self.singular else None
        result = _refined(self, self._direct, operator, rhs.data, volumes)
        result = _corrected(self, self._direct, operator, rhs.data, result, volumes)
        return Field(result, (CENTER, CENTER, CENTER), self.grid)

    def _direct(self, data: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        dtype = data.dtype
        # Stored factors, not quotients: normalized volumes and reciprocal eigenvalue sums.
        if self.singular:
            weights = host_array(self, _normalized_measure, dtype=dtype)
            data = data - jnp.sum(weights * data)
        scaled = _scaled(self, data, single, inverse=False)
        transformed = _contracted(self, scaled, single, transpose=True)
        solution = transformed * host_array(self, _inverse_denominator, single, dtype=transformed.dtype)
        if self.singular:
            solution = solution.at[0, 0, 0].set(0.0)
        restored = _contracted(self, solution, single, transpose=False)
        result = _scaled(self, restored, single, inverse=True)
        if self.singular:
            result = result - jnp.sum(weights * result)
        return result

    def residual_norm(self, solution: Field, rhs: Field) -> jnp.ndarray:
        """Return the maximum absolute residual of a candidate solution."""
        applied = laplacian(solution, self.conditions)
        difference = applied.data - rhs.data
        if self.singular:
            difference = _volume_mean_removed(difference, (self.grid, _cell_volumes))
        return jnp.max(jnp.abs(difference))

    def _eigenvalue_total(self) -> np.ndarray:
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        if self.singular:
            total[0, 0, 0] = 1.0
        return total

    def _eigenvalue_sum(self, dtype) -> jnp.ndarray:
        return jnp.asarray(self._eigenvalue_total(), dtype=dtype)

    def _denominator(self) -> np.ndarray:
        return self._eigenvalue_total()

    def _measure(self) -> np.ndarray:
        """Return the weights of the unknowns the contractions act on: the cell volumes."""
        return self.grid.cell_volumes()


def fast_diagonal_poisson(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    precision: str = "state",
    refinements: int = 2,
) -> FastDiagonalPoisson:
    """Factorize the separable Laplacian for ``grid`` under ``conditions``.

    ``precision="mixed"`` solves float64 right-hand sides in float32 with
    ``refinements`` float64 corrections (module docstring).
    """
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    for axis, condition in enumerate(conditions):
        if not condition.is_homogeneous:
            raise ValueError(
                f"axis {axis} carries inhomogeneous boundary data; factorize the homogeneous "
                "operator and move the boundary contribution into the right-hand side"
            )
    vectors, values, scales = [], [], []
    for axis, condition in enumerate(conditions):
        operator = assemble_axis_laplacian(grid, axis, condition)
        eigenvalues, eigenvectors, root = _symmetric_eigen(
            operator,
            np.asarray(grid.widths[axis]),
            f"axis {axis} operator is not symmetric under the cell widths",
        )
        vectors.append(eigenvectors)
        values.append(eigenvalues)
        scales.append(root)
    singular = _is_singular(values)
    corrections = int(any(condition.is_mixed for condition in conditions))
    if singular:
        # Order the constant mode first so a single entry carries the nullspace.
        vectors, values = _promote_null_mode(vectors, values)
    return FastDiagonalPoisson(
        grid,
        tuple(conditions),
        tuple(vectors),
        tuple(values),
        tuple(scales),
        singular,
        precision,
        refinements,
        corrections,
    )


def _is_singular(values: list[np.ndarray]) -> bool:
    """Return whether the Kronecker sum has a zero eigenvalue: every axis has one of its own.

    Each axis is judged against its own spectrum. Against the largest eigenvalue of
    all three, the lowest mode of a long inflow-outflow axis (0.04 against 7e7 across
    a Hartmann layer at Ha 100) passed for a null mode, and the solve removed it.
    """
    return all(
        float(np.min(np.abs(value))) <= _SINGULAR_TOLERANCE * max(float(np.max(np.abs(value))), 1.0)
        for value in values
    )


def _promote_null_mode(
    vectors: list[np.ndarray], values: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Move each axis's smallest eigenvalue to index zero."""
    ordered_vectors, ordered_values = [], []
    for vector, value in zip(vectors, values, strict=True):
        order = np.argsort(np.abs(value), kind="stable")
        ordered_vectors.append(vector[:, order])
        ordered_values.append(value[order])
    return ordered_vectors, ordered_values


def _symmetric_eigen(operator: np.ndarray, weights: np.ndarray, message: str):
    """Return the eigenvalues, eigenvectors and weight roots of an operator symmetric under ``weights``."""
    root = np.sqrt(weights)
    symmetric = root[:, None] * operator / root[None, :]
    asymmetry = _relative_asymmetry(symmetric)
    if asymmetry > _symmetry_tolerance(symmetric):
        raise ValueError(
            f"{message} (relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
    if np.max(np.abs(operator @ np.ones(len(weights)))) <= _SINGULAR_TOLERANCE * np.max(np.abs(operator)):
        # Constants are in the null space, so that eigenvalue is zero. Computed, it carries round-off
        # of the largest one (5e-10 across a Hartmann layer at Ha 100), which the lowest mode of a
        # long inflow-outflow axis (0.04) cannot absorb: the pressure drop then drifts at 1e-8.
        eigenvalues[np.argmin(np.abs(eigenvalues))] = 0.0
    return eigenvalues, eigenvectors, root


def assemble_thin_wall_operator(
    grid: Grid, axis: int, conductance: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-dimensional potential operator with a sheet node on each conducting wall, and its weights.

    A thin wall of conductance ratio ``c`` has a potential of its own: the fluid
    reaches it across the half cell, ``j_n = (phi_P - phi_w)/(h_P/2)``, and the
    sheet carries that current along itself, ``j_n = -c lap_t phi_w`` (Walker's
    condition). The sheet is one more node on the wall-normal axis, weighted by
    ``c`` times the wall's face measure where a cell is weighted by its own
    measure, so the tangential axes act on it as on a cell and the operator stays
    a Kronecker sum. The rows are read out of the production stencil: the coupling
    is the difference between the prescribed-value and the insulating wall cell.
    An end with zero conductance gets no node.
    """
    if grid.is_polar and axis != 0:
        raise ValueError("a thin wall on a polar grid is normal to the radius")

    def assemble(kind: str) -> np.ndarray:
        condition = BoundaryCondition(kind)
        if grid.is_polar:
            return assemble_radial_laplacian(grid, condition)
        return assemble_axis_laplacian(grid, axis, condition)

    insulating, prescribed = assemble(NEUMANN), assemble(DIRICHLET)
    face, cell = (np.asarray(measure) for measure in grid.axis_measures(axis))
    lower, upper = (float(value) for value in conductance)
    count, lead = insulating.shape[0], int(lower > 0.0)
    size = count + lead + int(upper > 0.0)
    operator, weights = np.zeros((size, size)), np.zeros(size)
    operator[lead : lead + count, lead : lead + count] = insulating
    weights[lead : lead + count] = cell
    for index, value, node in ((0, lower, 0), (count - 1, upper, size - 1)):
        if value <= 0.0:
            continue
        weights[node] = value * face[0 if index == 0 else -1]
        if weights[node] <= 0.0:
            raise ValueError("a thin wall needs a wall face of nonzero area")
        row, coupling = lead + index, insulating[index, index] - prescribed[index, index]
        operator[row, row] -= coupling
        operator[row, node] = coupling
        operator[node, row] = cell[index] * coupling / weights[node]
        operator[node, node] = -operator[node, row]
    return operator, weights


def assemble_thick_wall_operator(
    grid: Grid, axis: int, layers: tuple
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Return the one-dimensional potential operator with a wall resolved in cells at each end, and its weights.

    ``layers`` is ``(lower, upper)``: ``None`` for an insulating wall, else
    ``(ratios, widths)``, each cell's conductivity over the fluid's (one number
    for all) and its width, from the fluid outwards, insulated outside; adjacent
    cells join through their half cells in series. A wall cell is weighted by
    the ratio times its width, so the tangential axes conduct through it at the
    wall's conductivity and the operator stays a Kronecker sum, as for
    :func:`assemble_thin_wall_operator`; the wall must span the fluid's
    tangential extent. The fluid reaches the first wall cell through the two
    half cells in series. Also returned, per end, is the fraction of the
    potential difference across the fluid's half cell, which
    :meth:`FastDiagonalThinWallPoisson.solve_with_walls` uses to report the
    interface potential :func:`lmhdx.em.thin_wall_current` reads.
    """
    if grid.is_polar:
        raise ValueError("a wall resolved in cells needs a Cartesian grid")
    insulating = assemble_axis_laplacian(grid, axis, BoundaryCondition(NEUMANN))
    prescribed = assemble_axis_laplacian(grid, axis, BoundaryCondition(DIRICHLET))
    cell = np.asarray(grid.widths[axis], dtype=float)
    ends = [_wall_cells(layer) if layer else None for layer in layers]
    counts = [0 if end is None else end[1].size for end in ends]
    count, lead = cell.size, counts[0]
    size = count + sum(counts)
    operator, weights, fractions = np.zeros((size, size)), np.zeros(size), [1.0, 1.0]
    operator[lead : lead + count, lead : lead + count] = insulating
    weights[lead : lead + count] = cell
    for side, end in enumerate(ends):
        if end is None:
            continue
        ratio, widths = end
        index = 0 if side == 0 else count - 1
        # The production stencil's half-cell conductance, then the wall's half cell in series.
        half = cell[index] * (insulating[index, index] - prescribed[index, index])
        fractions[side] = 1.0 / (1.0 + half * 0.5 * widths[0] / ratio[0])
        nodes = [lead + index] + [lead - 1 - k if side == 0 else lead + count + k for k in range(widths.size)]
        weights[nodes[1:]] = ratio * widths
        resistance = 0.5 * widths / ratio
        links = [half * fractions[side]] + list(1.0 / (resistance[:-1] + resistance[1:]))
        for (first, second), link in zip(zip(nodes[:-1], nodes[1:]), links, strict=True):
            for row, column in ((first, second), (second, first)):
                operator[row, column] += link / weights[row]
                operator[row, row] -= link / weights[row]
    return operator, weights, tuple(fractions)


def _wall_cells(layer) -> tuple[np.ndarray, np.ndarray]:
    """Return a resolved wall's per-cell conductivity ratios and widths, checked."""
    widths = np.asarray(layer[1], dtype=float)
    ratios = np.broadcast_to(np.asarray(layer[0], dtype=float), widths.shape).copy()
    if widths.ndim != 1 or widths.size == 0 or np.any(widths <= 0.0) or np.any(ratios <= 0.0):
        raise ValueError("a resolved wall needs positive conductivity ratios and positive widths")
    return ratios, widths


@dataclass(frozen=True)
class FastDiagonalThinWallPoisson(FastDiagonalPoisson):
    """The potential Laplacian closed by thin conducting walls, factorized exactly.

    Each conducting axis carries a sheet node at both walls
    (:func:`assemble_thin_wall_operator`), so the solve is still three contractions
    and a divide, symmetric in the cell volumes extended by ``c`` times the wall
    area. Where two conducting walls meet, the corner node joins the two sheets in
    series, so the charge one delivers is what the other receives (Hua et al. 1988
    split the corner the same way). The Kronecker sum would also let that node
    conduct along the edge, which has no sheet area; a rank-four Woodbury
    correction per mode of the third axis removes it, and vanishes when that axis
    does not vary.
    """

    conductance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    operators: tuple[np.ndarray, ...] = ()
    weights: tuple[np.ndarray, ...] = ()
    corner_gain: np.ndarray | None = None
    # Wall nodes below and above each axis, and the interface fractions of resolved walls.
    pads: tuple[tuple[int, int], ...] | None = None
    fractions: tuple[tuple[float, float], ...] | None = None
    # Resolved walls on two axes: the corner cells' indices, and the Woodbury basis and capacitance.
    corner_fix: tuple | None = None

    def _pads(self) -> tuple[tuple[int, int], ...]:
        if self.pads is not None:
            return self.pads
        return tuple((1, 1) if float(value) > 0.0 else (0, 0) for value in self.conductance)

    def _measure(self) -> np.ndarray:
        return (
            self.weights[0][:, None, None] * self.weights[1][None, :, None] * self.weights[2][None, None, :]
        )

    def solve(self, rhs: Field) -> Field:
        """Return the cell potential whose charge balance is ``rhs``."""
        return self.solve_with_walls(rhs)[0]

    def solve_with_walls(self, rhs: Field) -> tuple[Field, tuple[Field | None, Field | None, Field | None]]:
        """Return the cell potential, with zero volume mean, and each conducting axis's sheet potentials.

        Sheet potentials come back on the faces normal to their axis, as
        :func:`lmhdx.em.thin_wall_current` reads them: wall entries set, interior zero.
        """
        if rhs.grid != self.grid or rhs.offset != (CENTER, CENTER, CENTER):
            raise ValueError("right-hand side must be cell centred on the factorized grid")
        pads = self._pads()
        data = jnp.pad(rhs.data, pads)
        solution = _refined(self, self._corrected, self._apply, data, (self, _own_measure))
        solution = _corrected(self, self._corrected, self._apply, data, solution, (self, _own_measure))
        cells = tuple(slice(low, size - high) for (low, high), size in zip(pads, solution.shape))
        volumes = host_array(self.grid, _cell_volumes, dtype=solution.dtype)
        solution = solution - jnp.sum(volumes * solution[cells]) / jnp.sum(volumes)
        walls = []
        for axis, (low, high) in enumerate(pads):
            if not low + high:
                walls.append(None)
                continue
            lines = solution[cells[:axis] + (slice(None),) + cells[axis + 1 :]]
            ends = []
            for side, (node, fluid, present) in enumerate(((low - 1, low, low), (-high, -high - 1, high))):
                at = (slice(None),) * axis
                adjacent = lines[at + (slice(fluid, fluid + 1 or None),)]
                wall = lines[at + (slice(node, node + 1 or None),)] if present else adjacent
                fraction = 1.0 if self.fractions is None else self.fractions[axis][side]
                # A resolved wall reports the potential on the interface, which the half-cell current reads.
                ends.append(wall if fraction == 1.0 else adjacent + fraction * (wall - adjacent))
            sheets = lines[(slice(None),) * axis + (slice(low, lines.shape[axis] - high),)]
            inner = jnp.zeros_like(sheets)[(slice(None),) * axis + (slice(1, None),)]
            offset = tuple(FACE if other == axis else CENTER for other in range(3))
            walls.append(Field(jnp.concatenate((ends[0], inner, ends[1]), axis=axis), offset, self.grid))
        return Field(solution[cells], (CENTER, CENTER, CENTER), self.grid), tuple(walls)

    def _apply(self, values: jnp.ndarray) -> jnp.ndarray:
        """The assembled operator, for the float64 residual of a mixed-precision solve."""
        matrices = [
            host_array(self, _thin_wall_entry, "operators", axis, dtype=values.dtype) for axis in range(3)
        ]
        total = sum(
            jnp.moveaxis(jnp.tensordot(matrix, values, axes=([1], [axis])), 0, axis)
            for axis, matrix in enumerate(matrices)
        )
        if self.corner_fix is not None:
            indices = self.corner_fix[0]
            difference = host_array(self, _thin_wall_entry, "corner_fix", 3, dtype=total.dtype)
            inverse_weights = host_array(self, _corner_inverse_weights, dtype=total.dtype)
            local = (
                jnp.zeros(values.size, total.dtype).at[indices].set(difference @ values.reshape(-1)[indices])
            )
            return total + local.reshape(values.shape) * inverse_weights
        if self.corner_gain is None:
            return total
        edge = self._edge()
        along = matrices[edge] @ _corners(values, edge)
        return total - _corners(values, edge, along)

    def _corrected(self, data: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        solution = self._direct(data, single)
        if self.corner_fix is not None:
            # Woodbury: the corner cells' links as they are, not as the Kronecker sum makes them.
            dtype = solution.dtype
            basis = host_array(self, _thin_wall_entry, "corner_fix", 1, dtype=dtype)
            capacitance = host_array(self, _thin_wall_entry, "corner_fix", 2, dtype=dtype)
            inverse_weights = host_array(self, _corner_inverse_weights, dtype=dtype)
            indices = self.corner_fix[0]
            modes = capacitance @ (basis.T @ solution.reshape(-1)[indices])
            source = jnp.zeros(solution.size, dtype).at[indices].set(basis @ modes)
            return solution - self._direct(source.reshape(solution.shape) * inverse_weights, single)
        if self.corner_gain is None:
            return solution
        edge, dtype = self._edge(), solution.dtype
        root = host_array(self, _thin_wall_entry, "scales", edge, dtype=dtype)[:, None]
        vectors = host_array(self, _thin_wall_entry, "vectors", edge, dtype=dtype)
        gain = host_array(self, _thin_wall_entry, "corner_gain", None, dtype=dtype)
        modes = jnp.einsum("mpq,mq->mp", gain, vectors.T @ (root * _corners(solution, edge)))
        return solution - self._direct(_corners(solution, edge, (vectors @ modes) / root), single)

    def _edge(self) -> int:
        return next(axis for axis, value in enumerate(self.conductance) if not float(value) > 0.0)


def _corner_inverse_weights(factorization) -> np.ndarray:
    return 1.0 / factorization._measure()


def _thin_wall_entry(factorization, name: str, axis: int | None) -> np.ndarray:
    entry = getattr(factorization, name)
    return entry if axis is None else entry[axis]


_CORNER_ROWS, _CORNER_COLUMNS = np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])


def _corners(data: jnp.ndarray, edge: int, values: jnp.ndarray | None = None) -> jnp.ndarray:
    """Gather the four corner edges along ``edge`` as ``(length, 4)``, or scatter ``values`` there into zeros."""
    axes = (edge, *(axis for axis in range(3) if axis != edge))
    ordered = jnp.moveaxis(data, axes, (0, 1, 2))
    if values is None:
        return ordered[:, _CORNER_ROWS, _CORNER_COLUMNS]
    return jnp.moveaxis(
        jnp.zeros_like(ordered).at[:, _CORNER_ROWS, _CORNER_COLUMNS].set(values), (0, 1, 2), axes
    )


def fast_diagonal_thin_wall_poisson(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    conductance: tuple[float, float, float],
    *,
    precision: str = "state",
    refinements: int = 2,
    layers: tuple | None = None,
) -> FastDiagonalThinWallPoisson:
    """Factorize the potential Laplacian with a thin wall of ratio ``conductance[axis]`` on both walls of an axis.

    Zero keeps the insulating closure of ``conditions``; a conducting axis must
    have insulating (Neumann) walls to replace, and one axis at least must not
    conduct. ``layers`` gives axes walls resolved in cells instead
    (:func:`assemble_thick_wall_operator`), ``(lower, upper)`` per axis or
    ``None``, with no thin wall. Two axes may have them when the third has one
    cell: the Kronecker sum would give a corner cell the product of its two
    walls' ratios, which :func:`_corner_fix` corrects exactly.
    """
    _require_separable(grid)
    layers = (None, None, None) if layers is None else tuple(layers)
    resolved = [axis for axis, pair in enumerate(layers) if pair and any(pair)]
    conducting = [axis for axis, value in enumerate(conductance) if float(value) > 0.0]
    if resolved and conducting:
        raise ValueError("walls resolved in cells take no thin wall on another axis")
    if len(resolved) > 1 and (len(resolved) > 2 or grid.shape[3 - sum(resolved)] != 1):
        raise ValueError("walls resolved in cells on two axes need one cell along the third")
    conducting = conducting + resolved
    pads, fractions = [], []
    if len(conditions) != 3 or len(conductance) != 3 or len(conducting) == 3 or min(conductance) < 0.0:
        raise ValueError(
            "thin walls need one condition and one non-negative conductance per axis, one axis without"
        )
    operators, vectors, values, scales, weights = [], [], [], [], []
    for axis, condition in enumerate(conditions):
        if not condition.is_homogeneous or (axis in conducting and condition.kind != NEUMANN):
            raise ValueError(
                f"axis {axis} needs a homogeneous condition, and insulating walls if it conducts"
            )
        pads.append((0, 0))
        fractions.append((1.0, 1.0))
        if axis in resolved:
            operator, weight, fractions[axis] = assemble_thick_wall_operator(grid, axis, layers[axis])
            pads[axis] = tuple(0 if layer is None else len(layer[1]) for layer in layers[axis])
        elif axis in conducting:
            ratio = float(conductance[axis])
            operator, weight = assemble_thin_wall_operator(grid, axis, (ratio, ratio))
            pads[axis] = (1, 1)
        else:
            operator, weight = assemble_axis_laplacian(grid, axis, condition), np.asarray(grid.widths[axis])
        message = f"axis {axis} thin-wall operator is not symmetric under its weights"
        eigenvalues, eigenvectors, root = _symmetric_eigen(operator, weight, message)
        for store, item in zip(
            (operators, vectors, values, scales, weights), (operator, eigenvectors, eigenvalues, root, weight)
        ):
            store.append(item)
    singular = _is_singular(values)
    if singular:
        vectors, values = _promote_null_mode(vectors, values)
    factorization = FastDiagonalThinWallPoisson(
        grid,
        tuple(conditions),
        tuple(vectors),
        tuple(values),
        tuple(scales),
        singular,
        precision,
        refinements,
        conductance=tuple(float(value) for value in conductance),
        operators=tuple(operators),
        weights=tuple(weights),
        corner_gain=_corner_gain(vectors, values, scales, 3 - sum(conducting))
        if len(conducting) == 2
        else None,
        pads=tuple(pads) if resolved else None,
        fractions=tuple(fractions) if resolved else None,
    )
    if len(resolved) == 2:
        factorization = replace(factorization, corner_fix=_corner_fix(factorization, grid, resolved, layers))
    return factorization


def _corner_fix(factorization: FastDiagonalThinWallPoisson, grid: Grid, axes: list[int], layers) -> tuple:
    """Return the Woodbury correction that gives the corner cells of two resolved walls their own links.

    In the Kronecker sum a corner cell conducts at the product of its two walls'
    ratios, and joins its neighbours accordingly. Here a corner cell takes the
    ratio of the nearer wall at its depth (the cell-centred solver's
    nearest-side rule, ties to the first axis), and every link touching it is
    the two half cells in series. The difference ``D`` of the symmetric flux
    matrices lives on those links; with ``D = U L U^T`` on their cells, the solve
    becomes ``x - K^-1 W^-1 U C U^T x`` for ``C = (L^-1 + U^T K^-1 U)^-1``, one
    more fast solve. Returned: the flat indices of those cells, ``U``, ``C`` and
    ``D`` restricted to them (for the assembled operator).
    """
    first, second = axes
    edge = 3 - first - second
    shape = tuple(len(weight) for weight in factorization.weights)
    geometry, ratio, depth, wall = [], [], [], []
    for axis in axes:
        pads = factorization.pads[axis]
        widths = np.asarray(grid.widths[axis], dtype=float)
        cells = [_wall_cells(layer) if layer else (np.zeros(0), np.zeros(0)) for layer in layers[axis]]
        lower, upper = cells
        geometry.append(np.concatenate([lower[1][::-1], widths, upper[1]]))
        ratio.append(np.concatenate([lower[0][::-1], np.ones(widths.size), upper[0]]))
        inside = np.zeros(geometry[-1].size, dtype=bool)
        inside[: pads[0]] = inside[geometry[-1].size - pads[1] :] = True
        wall.append(inside)
        centres = np.cumsum(geometry[-1]) - 0.5 * geometry[-1]
        edges = (centres[pads[0]] - 0.5 * widths[0], centres[pads[0] + widths.size - 1] + 0.5 * widths[-1])
        depth.append(
            np.where(
                centres < edges[0], edges[0] - centres, np.where(centres > edges[1], centres - edges[1], 0.0)
            )
        )
    corner = wall[0][:, None] & wall[1][None, :]
    nearer_first = depth[0][:, None] <= depth[1][None, :]
    sigma = np.where(
        corner,
        np.where(nearer_first, ratio[0][:, None], ratio[1][None, :]),
        np.where(wall[0][:, None], ratio[0][:, None], np.where(wall[1][None, :], ratio[1][None, :], 1.0)),
    )
    weight = [np.asarray(factorization.weights[axis]) for axis in axes]
    stiffness = [
        weight[k][:, None] * np.asarray(factorization.operators[axis]) for k, axis in enumerate(axes)
    ]
    plane = (shape[first], shape[second])
    links = []
    for i, k in zip(*np.nonzero(corner)):
        for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j, m = i + di, k + dk
            if not (0 <= j < plane[0] and 0 <= m < plane[1]) or (corner[j, m] and (j, m) < (i, k)):
                continue
            along = 0 if di else 1
            a, b = ((i, j), (k, m)) if along == 0 else ((k, m), (i, j))
            across = k if along == 0 else i
            kron = stiffness[along][a[0], a[1]] * weight[1 - along][across]
            h = geometry[along]
            area = geometry[1 - along][across]
            true = area / (0.5 * h[a[0]] / sigma[i, k] + 0.5 * h[a[1]] / sigma[j, m])
            links.append(((i, k), (j, m), true - kron))
    cells = sorted({cell for link in links for cell in link[:2]})
    position = {cell: index for index, cell in enumerate(cells)}
    difference = np.zeros((len(cells), len(cells)))
    for p, q, change in links:
        a, b = position[p], position[q]
        difference[a, b] += change
        difference[b, a] += change
        difference[a, a] -= change
        difference[b, b] -= change
    # The third axis has one cell: its width scales every in-plane flux.
    difference *= float(np.asarray(factorization.weights[edge])[0])
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    keep = np.abs(eigenvalues) > 1e-12 * np.max(np.abs(eigenvalues))
    eigenvalues, basis = eigenvalues[keep], eigenvectors[:, keep]
    full = [0, 0, 0]
    indices = []
    for i, k in cells:
        full[first], full[second], full[edge] = i, k, 0
        indices.append(int(np.ravel_multi_index(tuple(full), shape)))
    indices = np.asarray(indices)
    inverse_weights = 1.0 / factorization._measure()
    columns = np.zeros((basis.shape[1], *shape))
    columns.reshape(basis.shape[1], -1)[:, indices] = basis.T
    columns *= inverse_weights[None]
    solved = np.asarray(jax.vmap(factorization._direct)(jnp.asarray(columns)))
    projected = solved.reshape(basis.shape[1], -1)[:, indices] @ basis
    capacitance = np.linalg.inv(np.diag(1.0 / eigenvalues) + projected)
    return indices, basis, capacitance, difference


def _corner_gain(vectors, values, scales, edge: int) -> np.ndarray | None:
    """Return the per-mode Woodbury gain that removes conduction along the corner edges.

    In the eigenbasis of the edge axis that conduction is ``lambda_m`` on the four
    corner nodes of mode ``m``, so the correction is
    ``-lambda_m (I - lambda_m C_m)^-1`` with ``C_m`` the corner block of the
    uncorrected inverse; ``None`` when nothing varies along the edge.
    """
    first, second = (axis for axis in range(3) if axis != edge)
    tolerance = _SINGULAR_TOLERANCE * max(1.0, *(float(np.max(np.abs(value))) for value in values))
    if not np.any(np.abs(values[edge]) > tolerance):
        return None
    corners = ((first, _CORNER_ROWS), (second, _CORNER_COLUMNS))
    left = [vectors[axis][rows] / scales[axis][rows, None] for axis, rows in corners]
    right = [vectors[axis][rows] * scales[axis][rows, None] for axis, rows in corners]
    total = values[first][:, None] + values[second][None, :]
    gain = np.zeros((values[edge].size, 4, 4))
    for mode, eigenvalue in enumerate(values[edge]):
        if abs(eigenvalue) > tolerance:
            block = np.einsum("pi,pj,ij,qi,qj->pq", *left, 1.0 / (total + eigenvalue), *right)
            gain[mode] = -eigenvalue * np.linalg.inv(np.eye(4) - eigenvalue * block)
    return gain


def free_slice(grid: Grid, axis: int, offset_value: float, condition: BoundaryCondition) -> slice:
    """Return the entries of a staggered axis that a solve may change.

    A cell-centred axis is free everywhere. A face-centred axis with a wall has
    its two boundary faces prescribed, and a periodic one carries a duplicate of
    its first face at the end. Solving on anything else would either invent a
    value for a prescribed face or treat one face as two unknowns.
    """
    if offset_value == CENTER:
        return slice(None)
    if condition.is_periodic:
        return slice(0, grid.shape[axis])
    if condition.is_mixed:
        # Only a Dirichlet end is prescribed; the other face of an inflow-outflow axis is free.
        return slice(
            int(condition.kinds[0] == DIRICHLET), grid.shape[axis] + int(condition.kinds[1] != DIRICHLET)
        )
    return slice(1, grid.shape[axis])


def assemble_staggered_axis_operator(
    grid: Grid, axis: int, offset: tuple[float, float, float], condition: BoundaryCondition
) -> np.ndarray:
    """Return the dense one-dimensional operator :mod:`lmhdx.ops` applies along ``axis``.

    As with :func:`assemble_axis_laplacian`, the other axes are collapsed to a
    single cell under a homogeneous Neumann condition so they contribute
    nothing, and the operator is read out of the production stencil rather than
    written a second time.
    """
    faces = [uniform_faces(1, 0.0, 1.0)] * 3
    faces[axis] = np.asarray(grid.faces[axis])
    line_offset = tuple(offset[axis] if position == axis else CENTER for position in range(3))
    return _staggered_axis_operator(Grid(*faces), axis, line_offset, condition).copy()


@functools.lru_cache(maxsize=64)
def _staggered_axis_operator(
    line: Grid, axis: int, line_offset: tuple[float, float, float], condition: BoundaryCondition
) -> np.ndarray:
    conditions = tuple(
        condition if position == axis else BoundaryCondition("neumann") for position in range(3)
    )
    selection = free_slice(line, axis, line_offset[axis], condition)
    free = range(*selection.indices(line.offset_shape(line_offset)[axis]))
    positions = [tuple(index if position == axis else 0 for position in range(3)) for index in free]
    return _probe(
        lambda field: staggered_laplacian(field, conditions),
        ("staggered", conditions),
        line,
        line_offset,
        positions,
        selection,
    )


@dataclass(frozen=True)
class FastDiagonalHelmholtz:
    """A factorized ``shift * I - coefficient * laplacian`` for one staggered position.

    This is what an implicit viscous solve needs. The operator separates exactly
    as the pressure Laplacian does, so the same host-side eigendecomposition
    turns the solve into three contractions and a divide, and the step is no
    longer bounded by the mesh.
    """

    grid: Grid
    offset: tuple[float, float, float]
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    shift: float
    coefficient: float
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
    scales: tuple[np.ndarray, np.ndarray, np.ndarray]
    slices: tuple[slice, slice, slice]
    precision: str = "state"
    refinements: int = 1
    _low: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_precision(self.precision, self.refinements)
        if self.precision == "mixed":
            object.__setattr__(self, "_low", _single_bases(self.vectors, self.scales, self._denominator()))

    def solve(self, rhs: Field) -> Field:
        """Return the field this operator maps to ``rhs``.

        Prescribed entries are returned as zero: they are boundary data the
        caller owns, not unknowns this solve may set.
        """
        if rhs.grid != self.grid or rhs.offset != self.offset:
            raise ValueError("right-hand side must match the factorized position")

        def operator(values):
            embedded = rhs.replace_data(jnp.zeros(rhs.shape, dtype=values.dtype).at[self.slices].set(values))
            laplacian_ = staggered_laplacian(embedded, self.conditions).data
            shift, coefficient = attribute(self, "shift"), attribute(self, "coefficient")
            return (shift * embedded.data - coefficient * laplacian_)[self.slices]

        solution = _refined(self, self._direct, operator, rhs.data[self.slices], None)
        return rhs.replace_data(jnp.zeros_like(rhs.data).at[self.slices].set(solution))

    def _denominator(self) -> np.ndarray:
        total = self.values[0][:, None, None] + self.values[1][None, :, None] + self.values[2][None, None, :]
        return self.shift - self.coefficient * total

    def _direct(self, interior: jnp.ndarray, single: bool = False) -> jnp.ndarray:
        scaled = _scaled(self, interior, single, inverse=False)
        transformed = _contracted(self, scaled, single, transpose=True)
        inverse = host_array(self, _inverse_denominator, single, dtype=transformed.dtype)
        restored = _contracted(self, transformed * inverse, single, transpose=False)
        return _scaled(self, restored, single, inverse=True)


def fast_diagonal_helmholtz(
    grid: Grid,
    offset: tuple[float, float, float],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    shift: float = 1.0,
    coefficient: float = 1.0,
    precision: str = "state",
    refinements: int = 1,
) -> FastDiagonalHelmholtz:
    """Factorize ``shift * I - coefficient * laplacian`` at one staggered position.

    ``precision="mixed"`` solves float64 right-hand sides in float32 with
    ``refinements`` float64 corrections (module docstring).
    """
    if len(conditions) != 3:
        raise ValueError("a factorization needs one boundary condition per axis")
    for axis, condition in enumerate(conditions):
        if not condition.is_homogeneous:
            raise ValueError(
                f"axis {axis} carries inhomogeneous boundary data; factorize the homogeneous "
                "operator and move the boundary contribution into the right-hand side"
            )
    vectors, values, scales, slices = [], [], [], []
    for axis, condition in enumerate(conditions):
        operator = assemble_staggered_axis_operator(grid, axis, offset, condition)
        selection = free_slice(grid, axis, offset[axis], condition)
        weights = _axis_weights(grid, axis, offset[axis], condition)[selection]
        root = np.sqrt(weights)
        symmetric = root[:, None] * operator / root[None, :]
        asymmetry = _relative_asymmetry(symmetric)
        if asymmetry > _symmetry_tolerance(symmetric):
            raise ValueError(
                f"axis {axis} operator is not symmetric under its cell weights "
                f"(relative asymmetry {asymmetry:.3e}); fast diagonalization does not apply"
            )
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
        vectors.append(eigenvectors)
        values.append(eigenvalues)
        scales.append(root)
        slices.append(selection)
    return FastDiagonalHelmholtz(
        grid,
        tuple(offset),
        tuple(conditions),
        float(shift),
        float(coefficient),
        tuple(vectors),
        tuple(values),
        tuple(scales),
        tuple(slices),
        precision,
        refinements,
    )


def _axis_weights(grid: Grid, axis: int, offset_value: float, condition: BoundaryCondition) -> np.ndarray:
    """Return the measure that makes the one-dimensional operator symmetric.

    A cell-centred value owns its cell; a face-centred one owns the dual cell
    spanning the two half-cells on either side of the face.
    """
    widths = np.asarray(grid.widths[axis])
    if offset_value == CENTER:
        return widths
    if condition.is_periodic:
        wrap = 0.5 * (widths[0] + widths[-1])
        return np.concatenate(([wrap], 0.5 * (widths[:-1] + widths[1:]), [wrap]))
    ends = (0.5 * widths[0], 0.5 * widths[-1]) if condition.is_mixed else (widths[0], widths[-1])
    return np.concatenate(([ends[0]], 0.5 * (widths[:-1] + widths[1:]), [ends[1]]))
