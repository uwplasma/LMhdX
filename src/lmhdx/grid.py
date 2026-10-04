"""Tensor-product grid, staggered field container and wall-resolving coordinates.

The grid holds host-side face coordinates and is hashable static metadata: it is
never a traced value, so eigendecompositions and stencil bookkeeping happen once
at trace time. A :class:`Field` carries the traced array plus its staggered
position, following the marker-and-cell convention where an offset of ``0.5``
along an axis means cell-centred and ``0.0`` means the lower face.

Coordinate families cluster cells near walls, which high-Hartmann flow requires:
the Hartmann layer scales as ``a/Ha`` and the side layer as ``a/sqrt(Ha)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import numpy as np

__all__ = [
    "CARTESIAN",
    "POLAR",
    "CENTER",
    "FACE",
    "Field",
    "Grid",
    "geometric_faces",
    "tanh_faces",
    "uniform_faces",
    "wall_resolving_faces",
]

CENTER = 0.5
FACE = 0.0
_AXES = ("x", "y", "z")


def _validated(faces: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(faces, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array of at least two faces")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


CARTESIAN = "cartesian"
POLAR = "polar"
_GEOMETRIES = (CARTESIAN, POLAR)


@dataclass(frozen=True)
class Grid:
    """Immutable tensor-product grid defined by strictly increasing face coordinates.

    ``geometry`` names what the three coordinates mean. :data:`CARTESIAN` is
    ``(x, y, z)``. :data:`POLAR` is ``(r, theta, z)``, which is a tensor product
    in the coordinates but not in the metric: an azimuthal face keeps its area
    while a radial one grows with ``r``, and the azimuthal distance between two
    cell centres is ``r*dtheta``. Every stencil in :mod:`lmhdx.ops` reads the
    metric through :meth:`face_areas`, :meth:`cell_volumes` and
    :func:`lmhdx.ops.face_distances`, so putting it here is enough to make the
    same operators solve a pipe.

    The axis is not a special case in flux form. The face at ``r = 0`` has zero
    area, so nothing flows through it and no regularity condition has to be
    imposed by hand -- which is the reason to write the divergence as a flux
    balance rather than as a differentiated product.
    """

    x_faces: np.ndarray
    y_faces: np.ndarray
    z_faces: np.ndarray
    geometry: str = CARTESIAN

    def __post_init__(self) -> None:
        for axis in _AXES:
            object.__setattr__(self, f"{axis}_faces", _validated(getattr(self, f"{axis}_faces"), name=axis))
        if self.geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {list(_GEOMETRIES)}, got {self.geometry!r}")
        if self.geometry == POLAR:
            if self.x_faces[0] < 0.0:
                raise ValueError("a polar grid needs a non-negative radius")
            span = float(self.y_faces[-1] - self.y_faces[0])
            if not np.isclose(span, 2.0 * np.pi):
                raise ValueError(f"a polar grid must span 2*pi in the azimuth, got {span:.6g}")

    def __hash__(self) -> int:
        return hash((self.geometry, tuple(faces.tobytes() for faces in self.faces)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.geometry == other.geometry and all(
            np.array_equal(a, b) for a, b in zip(self.faces, other.faces, strict=True)
        )

    @property
    def is_polar(self) -> bool:
        return self.geometry == POLAR

    @property
    def faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.x_faces, self.y_faces, self.z_faces)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of cells along each axis."""
        return tuple(int(faces.size - 1) for faces in self.faces)

    @property
    def widths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell widths along each axis."""
        return tuple(np.diff(faces) for faces in self.faces)

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell-centre coordinates along each axis."""
        return tuple(0.5 * (faces[:-1] + faces[1:]) for faces in self.faces)

    @property
    def extent(self) -> tuple[float, float, float]:
        """Domain length along each axis."""
        return tuple(float(faces[-1] - faces[0]) for faces in self.faces)

    def axis_index(self, axis: str | int) -> int:
        """Resolve ``"x"``/``"y"``/``"z"`` or an integer to an axis index."""
        if isinstance(axis, int):
            if axis not in (0, 1, 2):
                raise ValueError(f"axis index {axis} is out of range")
            return axis
        try:
            return _AXES.index(axis)
        except ValueError as error:
            raise ValueError(f"unknown axis {axis!r}") from error

    def volume_factors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one factor per axis whose product, left to right, is :meth:`cell_volumes`.

        Stencils multiply them on the device, where XLA fuses the product into its
        consumer, rather than capturing a full-size array at every call.
        """
        dx, dy, dz = self.widths
        return (self._annular() if self.is_polar else dx)[:, None, None], dy[None, :, None], dz[None, None, :]

    def cell_volumes(self) -> np.ndarray:
        """Volume of every cell, shaped like the cell-centred field."""
        first, second, third = self.volume_factors()
        return first * second * third

    def area_factors(self, axis: str | int) -> tuple[np.ndarray, ...]:
        """Return broadcastable factors whose product, left to right, is :meth:`face_areas`.

        A radial face is r*dtheta*dz, an azimuthal one dr*dz, an axial one the
        annular sector; only the first two follow from the widths alone.
        """
        index = self.axis_index(axis)
        factors = tuple(
            width.reshape([-1 if axes == other else 1 for axes in range(3)])
            for other, width in enumerate(self.widths)
            if other != index
        )
        if not self.is_polar or index == 1:
            return factors
        radial = self.x_faces if index == 0 else self._annular() / self.widths[0]
        return (*factors, radial[:, None, None])

    def face_areas(self, axis: str | int) -> np.ndarray:
        """Area of the faces normal to ``axis``, shaped like that face field."""
        area = np.ones(self.face_shape(axis))
        for factor in self.area_factors(axis):
            area = area * factor
        return area

    def _annular(self) -> np.ndarray:
        return 0.5 * (self.x_faces[1:] ** 2 - self.x_faces[:-1] ** 2)

    def axis_measures(self, axis: str | int) -> tuple[np.ndarray, np.ndarray]:
        """Return the face and cell measures of one axis, with the other two divided out.

        A one-dimensional stencil along ``axis`` needs the flux area and the
        control volume only up to whatever the other two directions contribute,
        because that part cancels between them. Cartesian returns ones and the
        cell widths; polar returns the radius on the radial faces and the
        annular measure on the cells, which is what turns a plain second
        difference into ``(1/r) d/dr (r d/dr)``.
        """
        index = self.axis_index(axis)
        widths = np.asarray(self.widths[index])
        if not self.is_polar or index == 2:
            return np.ones(self.shape[index] + 1), widths
        if index == 0:
            return np.asarray(self.x_faces), np.asarray(self.centers[0]) * widths
        radius = np.asarray(self.centers[0])[:, None, None]
        return np.ones(self.shape[1] + 1), radius * widths[None, :, None]

    def face_shape(self, axis: str | int) -> tuple[int, int, int]:
        """Shape of a field living on the faces normal to ``axis`` (walls included)."""
        index = self.axis_index(axis)
        shape = list(self.shape)
        shape[index] += 1
        return tuple(shape)

    def offset_shape(self, offset: tuple[float, float, float]) -> tuple[int, int, int]:
        """Shape implied by a staggered ``offset``; ``FACE`` adds one entry on that axis."""
        return tuple(n + (1 if o == FACE else 0) for n, o in zip(self.shape, offset, strict=True))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Field:
    """A traced array together with its staggered position on a :class:`Grid`."""

    data: jax.Array
    offset: tuple[float, float, float]
    grid: Grid

    def tree_flatten(self):
        return (self.data,), (self.offset, self.grid)

    @classmethod
    def tree_unflatten(cls, static, children):
        offset, grid = static
        return cls(children[0], offset, grid)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    def replace_data(self, data: jax.Array) -> "Field":
        """Return the same staggered position carrying ``data``."""
        return Field(data, self.offset, self.grid)


def uniform_faces(count: int, lower: float, upper: float) -> np.ndarray:
    """Return ``count`` equal cells spanning ``[lower, upper]``."""
    if count < 1:
        raise ValueError("count must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    return np.linspace(lower, upper, count + 1)


def geometric_faces(
    count: int, lower: float, upper: float, ratio: float, *, both_ends: bool = True
) -> np.ndarray:
    """Return faces whose successive cell widths grow by ``ratio`` away from the wall.

    With ``both_ends`` the distribution is mirrored about the midpoint, which
    clusters cells at both walls of a duct.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    if both_ends:
        if count % 2:
            raise ValueError("count must be even when clustering at both ends")
        half = geometric_faces(count // 2, 0.0, 0.5, ratio, both_ends=False)
        normalized = np.concatenate([half[:-1], 1.0 - half[::-1]])
    else:
        steps = np.ones(count) if np.isclose(ratio, 1.0) else ratio ** np.arange(count)
        edges = np.concatenate([[0.0], np.cumsum(steps)])
        normalized = edges / edges[-1]
    return lower + (upper - lower) * normalized


def tanh_faces(count: int, lower: float, upper: float, beta: float) -> np.ndarray:
    """Return symmetric hyperbolic-tangent faces; larger ``beta`` clusters harder at both walls."""
    if count < 1:
        raise ValueError("count must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    uniform = np.linspace(-1.0, 1.0, count + 1)
    stretched = np.tanh(beta * uniform) / np.tanh(beta)
    normalized = 0.5 * (stretched + 1.0)
    normalized = (normalized - normalized[0]) / (normalized[-1] - normalized[0])
    return lower + (upper - lower) * normalized


def wall_resolving_faces(
    count: int,
    lower: float,
    upper: float,
    *,
    layer_thickness: float,
    cells_in_layer: int = 8,
    max_ratio: float | None = 1.15,
) -> np.ndarray:
    """Return faces resolving a wall layer of ``layer_thickness`` at both ends.

    The smallest ``cells_in_layer`` cells fall inside the layer while successive
    widths grow by at most ``max_ratio``, the stretching limit reported for
    high-Hartmann duct meshes. Raise when the request cannot be met with
    ``count`` cells so a caller never silently runs an unresolved layer.

    ``max_ratio=None`` fits the gentlest ratio that still spans the domain. The
    widths are rescaled to fill the half-width either way, so a ratio larger
    than the cell count needs buys no resolution -- it buys an operator whose
    entries span orders of magnitude for nothing. An odd ``count`` puts one
    centre cell, the next width of the progression, on the midplane.
    """
    if layer_thickness <= 0.0:
        raise ValueError("layer_thickness must be positive")
    if cells_in_layer < 1:
        raise ValueError("cells_in_layer must be positive")
    if max_ratio is None:
        return wall_resolving_faces(
            count,
            lower,
            upper,
            layer_thickness=layer_thickness,
            cells_in_layer=cells_in_layer,
            max_ratio=_fitted_ratio(count, 0.5 * (upper - lower), layer_thickness, cells_in_layer),
        )
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least one")
    half_cells, half_width = count // 2, 0.5 * (upper - lower)
    if layer_thickness > half_width:
        raise ValueError("layer_thickness must not exceed the half-width")
    first = layer_thickness / _geometric_sum(cells_in_layer, max_ratio)
    # An odd count keeps one centre cell, the next term of the progression, split across the midplane.
    widths = first * max_ratio ** np.arange(half_cells + count % 2)
    reach = float(np.sum(widths[:half_cells]) + 0.5 * np.sum(widths[half_cells:]))
    if reach < half_width:
        raise ValueError(
            f"{count} cells reach {reach:.4g} of the required {half_width:.4g}; "
            "increase count, cells_in_layer or max_ratio"
        )
    # Scaling down to fill the half-width only thins the inner cells further,
    # so the layer request is still satisfied after this rescaling.
    widths *= half_width / reach
    half = np.concatenate([[0.0], np.cumsum(widths[:half_cells])])
    return np.concatenate([lower + half[: len(half) - 1 + count % 2], upper - half[::-1]])


def _fitted_ratio(count: int, half_width: float, layer_thickness: float, cells_in_layer: int) -> float:
    """Return the smallest growth ratio whose widths still reach ``half_width``."""

    def reaches(ratio: float) -> bool:
        first = layer_thickness / _geometric_sum(cells_in_layer, ratio)
        terms = first * ratio ** np.arange(count // 2 + count % 2)
        return float(np.sum(terms[: count // 2]) + 0.5 * np.sum(terms[count // 2 :])) >= half_width

    low, high = 1.0, 4.0
    if not reaches(high):
        raise ValueError(
            f"{count} cells cannot span {half_width:.4g} while resolving a layer of "
            f"{layer_thickness:.4g}; increase count or cells_in_layer"
        )
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if reaches(middle) else (middle, high)
    return high


def _geometric_sum(terms: int, ratio: float) -> float:
    """Return ``sum(ratio**k)`` for ``k`` below ``terms``."""
    if np.isclose(ratio, 1.0):
        return float(terms)
    return float((ratio**terms - 1.0) / (ratio - 1.0))
