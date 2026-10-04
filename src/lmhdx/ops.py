"""Finite-volume stencils on the staggered grid.

Gradients map a cell-centred field to the faces normal to one axis; the
divergence maps face-normal fields back to cells as the net flux through the
cell's faces divided by its volume. Written this way the two are exact discrete
adjoints under the volume and face weights of :func:`cell_inner_product` and
:func:`face_inner_product`, which is the identity a compatible pressure or
potential solve depends on: it is what makes the projection idempotent and keeps
the Lorentz force from doing spurious work in the core of a high-Hartmann duct.

Ghost cells carry the wall condition (see :mod:`lmhdx.bc`), so a wall face is
differenced with the same expression as an interior face. The ghost cell mirrors
the width of the wall cell, which places its centre one wall-cell width outside
and makes that single expression reproduce both the prescribed-value and the
prescribed-gradient condition.

Accuracy at a prescribed-value wall is deliberate. The wall flux is the two-point
difference ``(p_0 - g) / (dx_0 / 2)``, which is centred a quarter cell inside the
wall and so is first order there, while every interior face is second order. This
is the stencil that keeps the assembled Laplacian symmetric and the fluxes
conservative, which the pressure and potential solves depend on; a three-point
wall stencil would raise the wall order at the cost of that symmetry. The choice
is pinned by test, not left implicit.

On a stretched mesh the interior two-point gradient is centred between the two
cell centres rather than on the face, so its truncation error is first order in
the spacing change. Solution order there is a manufactured-solution question,
verified in the step that owns that study rather than asserted here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._programs import host_array, host_scalar
from .bc import BoundaryCondition, pad
from .grid import CENTER, FACE, Field, Grid

__all__ = [
    "axis_divergence",
    "cell_inner_product",
    "divergence",
    "face_average",
    "face_average_adjoint",
    "face_distances",
    "face_gradient",
    "face_inner_product",
    "face_interpolate",
    "laplacian",
    "staggered_laplacian",
]


def face_distances(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    """Return the centre-to-centre distance across every face normal to ``axis``.

    One dimensional where the metric is, and three dimensional where it is not:
    the azimuthal spacing of a polar grid is ``r*dtheta``, so it varies across
    the radius and cannot be described by a single array along its own axis.
    """
    widths = np.asarray(grid.widths[axis])
    interior = 0.5 * (widths[:-1] + widths[1:])
    if condition.is_periodic:
        wrap = 0.5 * (widths[0] + widths[-1])
        spacing = np.concatenate(([wrap], interior, [wrap]))
    else:
        spacing = np.concatenate(([widths[0]], interior, [widths[-1]]))
    if grid.is_polar and axis == 1:
        return _length_scale(grid, axis) * spacing[None, :, None]
    return spacing


def _length_scale(grid: Grid, axis: int) -> np.ndarray | float:
    """Return the factor turning a coordinate increment along ``axis`` into a length.

    One everywhere except the azimuth of a polar grid, where it is the radius.
    """
    if grid.is_polar and axis == 1:
        return np.asarray(grid.centers[0])[:, None, None]
    return 1.0


def _metric(values: np.ndarray, axis: int) -> np.ndarray:
    """Broadcast a spacing that may already carry its own shape."""
    return values if values.ndim == 3 else _shaped(values, axis)


def _inverse_distances(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    return _metric(1.0 / face_distances(grid, axis, condition), axis)


def face_gradient(field: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Differentiate a cell-centred field onto the faces normal to ``axis``."""
    grid = field.grid
    index = grid.axis_index(axis)
    _require_cell_centred(field)
    padded = pad(field.data, index, condition, grid=grid)
    difference = _take(padded, index, slice(1, None)) - _take(padded, index, slice(None, -1))
    factor = host_array(grid, _inverse_distances, index, condition, dtype=field.dtype)
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(difference * factor, offset, grid)


def axis_divergence(face: Field, axis: int | str) -> Field:
    """Return what one face-normal field contributes to the divergence."""
    grid = face.grid
    index = grid.axis_index(axis)
    _require_face(face, index)
    flux = _product(grid, _area_factors, (index,), face.dtype) * face.data
    contribution = _take(flux, index, slice(1, None)) - _take(flux, index, slice(None, -1))
    inverse = _product(grid, _inverse_volume_factors, (), face.dtype)
    return Field(contribution * inverse, (CENTER, CENTER, CENTER), grid)


def _area_factors(grid: Grid, axis: int) -> tuple[np.ndarray, ...]:
    return grid.area_factors(axis)


def _inverse_volume_factors(grid: Grid) -> tuple[np.ndarray, ...]:
    return tuple(1.0 / factor for factor in grid.volume_factors())


def _volume_factors(grid: Grid) -> tuple[np.ndarray, ...]:
    return grid.volume_factors()


def divergence(faces: tuple[Field, Field, Field]) -> Field:
    """Return the net outward flux per unit volume of three face-normal fields."""
    grid = faces[0].grid
    total = None
    for index, face in enumerate(faces):
        if face.grid != grid:
            raise ValueError("face fields must share one grid")
        contribution = axis_divergence(face, index).data
        total = contribution if total is None else total + contribution
    return Field(total, (CENTER, CENTER, CENTER), grid)


def laplacian(field: Field, conditions: tuple[BoundaryCondition, ...]) -> Field:
    """Return the divergence of the gradient of a cell-centred field."""
    if len(conditions) != 3:
        raise ValueError("laplacian needs one boundary condition per axis")
    gradients = tuple(face_gradient(field, index, conditions[index]) for index in range(3))
    return divergence(gradients)


def face_interpolate(field: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Interpolate a cell-centred field onto faces, weighted by the distance to each centre.

    Distance weighting keeps the interpolation second order on a stretched mesh,
    where the arithmetic mean is only first order.
    """
    grid = field.grid
    index = grid.axis_index(axis)
    _require_cell_centred(field)
    padded = pad(field.data, index, condition, grid=grid)
    weight = host_array(grid, _interpolation_weight, index, dtype=field.dtype)
    lower = _take(padded, index, slice(None, -1))
    upper = _take(padded, index, slice(1, None))
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(weight * lower + (1.0 - weight) * upper, offset, grid)


def _interpolation_weight(grid: Grid, axis: int) -> np.ndarray:
    widths = np.asarray(grid.widths[axis])
    ghosted = np.concatenate(([widths[0]], widths, [widths[-1]]))
    return _shaped(ghosted[1:] / (ghosted[:-1] + ghosted[1:]), axis)


def face_average(field: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Average a cell-centred field over the control volume straddling each face.

    That control volume is half of each neighbouring cell, so the face value is
    ``(h_L c_L + h_R c_R) / (h_L + h_R)``: weighted by the cell's own width, where
    :func:`face_interpolate` weights by the opposite one. The two agree on a
    uniform mesh. On a stretched one this average is not exact for a linear
    field -- it is off by ``(h_R - h_L)/2`` times the slope -- but it is the one
    two-point interpolation whose transpose, :func:`face_average_adjoint`, is
    itself an average. The electromotive force and the Lorentz force need exactly
    that pair, since each must be the adjoint of the other.

    A polar metric scales both halves of the control volume by the same face
    area and length scale, so the weights stay one dimensional.
    """
    grid = field.grid
    index = grid.axis_index(axis)
    _require_cell_centred(field)
    padded = pad(field.data, index, condition, grid=grid)
    lower = _take(padded, index, slice(None, -1))
    upper = _take(padded, index, slice(1, None))
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    weights = tuple(
        host_array(grid, _average_weight, index, condition, side, dtype=field.dtype) for side in (0, 1)
    )
    return Field(weights[0] * lower + weights[1] * upper, offset, grid)


def _average_weight(grid: Grid, axis: int, condition: BoundaryCondition, side: int) -> np.ndarray:
    """The share of the lower (``side`` 0) or upper cell in :func:`face_average`."""
    ghosted = _ghosted_widths(np.asarray(grid.widths[axis]), condition)
    share = ghosted[:-1] / (ghosted[:-1] + ghosted[1:])
    return _shaped(1.0 - share if side else share, axis)


def face_average_adjoint(face: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Return the transpose of :func:`face_average` under the volume-weighted inner products.

    Each cell reads its two faces, weighted by the half of the cell each face's
    control volume owns: ``m^-/V`` and ``m^+/V`` with ``m = A * l * h/2``, the face
    area ``A``, the length scale ``l`` of the axis and the cell width ``h``. The
    two halves add to the cell volume on Cartesian and polar grids alike, so the
    result is a true average. With the face current and the face field as the
    operand it is the face form of the Lorentz force of Ni et al.,
    ``(1/V) sum_f J_f A_f (r_f - r_c) x B_f``.

    The identity

    .. math:: \\langle g, \\mathrm{face\\_average}(c)\\rangle_{\\mathrm{face}}
       = \\langle \\mathrm{face\\_average\\_adjoint}(g), c\\rangle_{\\mathrm{cell}}

    under :func:`face_inner_product` and :func:`cell_inner_product` holds to
    round-off for every ``g`` that vanishes on the wall faces of a non-periodic
    axis -- an impermeable velocity or an insulated current, which is what either
    operator is applied to. On a periodic axis the duplicated wrap face is read as
    the mean of its two copies, which is what the halved weight of each copy in
    :func:`face_inner_product` transposes to, so there it holds for every ``g``.
    """
    grid = face.grid
    index = grid.axis_index(axis)
    _require_face(face, index)
    data = face.data
    if condition.is_periodic:
        wrap = 0.5 * (_take(data, index, slice(None, 1)) + _take(data, index, slice(-1, None)))
        data = jnp.concatenate((wrap, _take(data, index, slice(1, -1)), wrap), axis=index)
    lower_weight, upper_weight = (
        host_array(grid, _half_cell_weight, index, side, dtype=face.dtype) for side in (0, 1)
    )
    lower = _take(data, index, slice(None, -1))
    upper = _take(data, index, slice(1, None))
    return Field(lower_weight * lower + upper_weight * upper, (CENTER, CENTER, CENTER), grid)


def _half_cell_weight(grid: Grid, axis: int, side: int) -> np.ndarray:
    return _half_cell_weights(grid, axis)[side]


def _ghosted_widths(widths: np.ndarray, condition: BoundaryCondition) -> np.ndarray:
    """Cell widths with the ghost cell on each end: the wrapped cell, or the mirrored wall cell."""
    if condition.is_periodic:
        return np.concatenate(([widths[-1]], widths, [widths[0]]))
    return np.concatenate(([widths[0]], widths, [widths[-1]]))


def _half_cell_weights(grid: Grid, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the fraction of each cell owned by the control volumes of its lower and upper face.

    The widths across the axis cancel between the face measure and the cell volume, and so does the
    axis's own width: one half each on a Cartesian grid, a function of the radius on a polar one. So
    they are formed on one radial line and broadcast, not captured per cell and per call.
    """
    grid = Grid(grid.x_faces, grid.y_faces[[0, -1]], grid.z_faces[[0, -1]], grid.geometry)
    measure = grid.face_areas(axis) * _length_scale(grid, axis)
    half = _shaped(0.5 * np.asarray(grid.widths[axis]), axis)
    volumes = grid.cell_volumes()
    lower = np.take(measure, np.arange(grid.shape[axis]), axis=axis) * half / volumes
    upper = np.take(measure, np.arange(1, grid.shape[axis] + 1), axis=axis) * half / volumes
    return lower, upper


def cell_inner_product(left: Field, right: Field) -> jnp.ndarray:
    """Return the volume-weighted inner product of two cell-centred fields."""
    _require_cell_centred(left)
    _require_cell_centred(right)
    if left.grid != right.grid:
        raise ValueError("fields must share one grid")
    return jnp.sum(_product(left.grid, _volume_factors, (), left.dtype) * left.data * right.data)


def face_inner_product(
    left: Field, right: Field, axis: int | str, condition: BoundaryCondition
) -> jnp.ndarray:
    """Return the face-weighted inner product of two face-normal fields.

    The weight is the face area times the centre-to-centre distance, the volume
    of the control cell straddling that face, which is the weight under which
    :func:`divergence` and :func:`face_gradient` are exact adjoints.

    A periodic axis stores its wrap face twice, once at each end, so each copy
    takes half the weight. Counting both in full would double the measure of
    that control volume, which an adjoint identity would not notice -- both
    sides carry the same error -- but an energy budget would, and did. The two
    end faces of an inflow-outflow axis carry values, and own half a cell each.
    """
    grid = left.grid
    index = grid.axis_index(axis)
    _require_face(left, index)
    _require_face(right, index)
    if left.grid != right.grid:
        raise ValueError("fields must share one grid")
    weights = _product(grid, _face_measures, (index, condition), left.dtype)
    return jnp.sum(weights * left.data * right.data)


def _face_measures(grid: Grid, axis: int, condition: BoundaryCondition) -> tuple[np.ndarray, ...]:
    """The factors of the control volume straddling each face normal to ``axis``."""
    distances = face_distances(grid, axis, condition)
    if condition.is_periodic or condition.is_mixed:
        # A wrap face is stored twice; an inflow-outflow face owns only the half cell inside.
        distances = distances.copy()
        wrap = (slice(None),) * axis + ([0, -1],) if distances.ndim == 3 else ([0, -1],)
        distances[wrap] *= 0.5
    shaped = distances if distances.ndim == 3 else _shaped(distances, axis)
    return (*grid.area_factors(axis), shaped)


def _require_cell_centred(field: Field) -> None:
    if field.offset != (CENTER, CENTER, CENTER):
        raise ValueError(f"expected a cell-centred field, got offset {field.offset}")
    if field.shape != field.grid.shape:
        raise ValueError(f"cell field shape {field.shape} does not match grid {field.grid.shape}")


def _require_face(field: Field, index: int) -> None:
    expected = tuple(FACE if position == index else CENTER for position in range(3))
    if field.offset != expected:
        raise ValueError(f"expected a field on faces normal to axis {index}, got offset {field.offset}")
    if field.shape != field.grid.face_shape(index):
        raise ValueError(f"face field shape {field.shape} does not match grid {field.grid.face_shape(index)}")


def _take(data: jnp.ndarray, axis: int, selection: slice) -> jnp.ndarray:
    return data[(slice(None),) * axis + (selection,)]


def _shaped(values: np.ndarray, axis: int) -> np.ndarray:
    return values.reshape([-1 if position == axis else 1 for position in range(3)])


def _broadcast(values: np.ndarray, axis: int, dtype) -> jnp.ndarray:
    return jnp.asarray(_shaped(values, axis), dtype=dtype)


def foldable(shape: tuple[int, ...]) -> bool:
    """Whether a metric product of this shape is formed on the host: a line or a plane, never a volume.

    A stored plane costs what one slab of the field does; a volume would bring back the
    cell-sized captured constants that dominated a 64^3 compile, so those stay broadcast
    factors multiplied on the device. A cross-section (one axial cell) is always a plane.
    """
    return sum(extent > 1 for extent in shape) <= 2


def _product(grid: Grid, build, static: tuple, dtype) -> jnp.ndarray:
    """Multiply the metric factors ``build(grid, *static)`` left to right: on the host when small, else on the device."""
    factors = build(grid, *static)
    if foldable(np.broadcast_shapes(*(np.shape(factor) for factor in factors))):
        return host_array(grid, _folded, build, static, dtype=dtype)
    total = host_array(grid, _factor, build, static, 0, dtype=dtype)
    for index in range(1, len(factors)):
        total = total * host_array(grid, _factor, build, static, index, dtype=dtype)
    return total


def _folded(grid: Grid, build, static: tuple) -> np.ndarray:
    factors = build(grid, *static)
    total = np.asarray(factors[0])
    for factor in factors[1:]:
        total = total * factor
    return total


def _factor(grid: Grid, build, static: tuple, index: int) -> np.ndarray:
    return build(grid, *static)[index]


def staggered_laplacian(
    field: Field, conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
) -> Field:
    """Return the Laplacian of a field at any staggered position.

    A velocity component in the marker-and-cell layout is cell-centred along two
    axes and face-centred along the third, so its Laplacian needs both stencils.
    Along a cell-centred axis the wall condition supplies a ghost value and the
    scalar path applies unchanged. Along a face-centred axis the field already
    sits on the wall, so no ghost exists and none is invented: the value is
    differenced to the cell centres with the cell widths and back to the faces
    with the centre-to-centre distances.

    On a face-centred axis with a wall the two boundary faces are returned as
    zero. Their value is prescribed by the boundary condition, not evolved, and
    returning zero keeps a caller that updates them from silently using a
    one-sided stencil that does not exist. A periodic axis wraps and has no such
    face.
    """
    if len(conditions) != 3:
        raise ValueError("a staggered Laplacian needs one boundary condition per axis")

    grid = field.grid
    if field.shape != grid.offset_shape(field.offset):
        raise ValueError(f"field shape {field.shape} does not match its offset {field.offset}")
    total = None
    for axis, condition in enumerate(conditions):
        if field.offset[axis] == CENTER:
            contribution = _centred_axis_laplacian(field, axis, condition)
        else:
            contribution = _face_axis_laplacian(field, axis, condition)
        total = contribution if total is None else total + contribution
    return field.replace_data(total)


def _centred_axis_laplacian(field: Field, axis: int, condition: BoundaryCondition) -> jnp.ndarray:
    """Flux balance along an axis on which the field is cell centred.

    Written as a flux difference rather than a second difference, so the metric
    of the axis enters through :meth:`lmhdx.grid.Grid.axis_measures` and the same
    stencil is ``(1/r) d/dr (r d/dr)`` on a polar grid. The two forms are
    identical where the measures are one and the widths, which is Cartesian.
    """
    grid = field.grid
    padded = pad(field.data, axis, condition, grid=grid)
    # One stored factor per face, measure over distance, and one reciprocal per cell.
    flux = (_take(padded, axis, slice(1, None)) - _take(padded, axis, slice(None, -1))) * host_array(
        grid, _conductance, axis, condition, dtype=field.dtype
    )
    difference = _take(flux, axis, slice(1, None)) - _take(flux, axis, slice(None, -1))
    return difference * host_array(grid, _inverse_cell_measure, axis, dtype=field.dtype)


def _conductance(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    distances = face_distances(grid, axis, condition)
    face_measure = grid.axis_measures(axis)[0]
    return _metric((face_measure if distances.ndim == 1 else _shaped(face_measure, axis)) / distances, axis)


def _inverse_cell_measure(grid: Grid, axis: int) -> np.ndarray:
    return _metric(1.0 / grid.axis_measures(axis)[1], axis)


def _face_axis_laplacian(field: Field, axis: int, condition: BoundaryCondition) -> jnp.ndarray:
    """Second difference along an axis on which the field sits on the faces."""
    grid = field.grid
    data = field.data
    inverse_widths = host_array(grid, _inverse_widths, axis, dtype=field.dtype)
    inverse_distances = host_array(
        grid, _inverse_centre_distances, axis, condition.is_periodic, dtype=field.dtype
    )
    if condition.is_periodic:
        # The first and last faces coincide; drop the duplicate before wrapping.
        interior = _take(data, axis, slice(None, -1))
        gradient = (_roll(interior, axis, -1) - interior) * inverse_widths
        difference = gradient - _roll(gradient, axis, 1)
        result = difference * inverse_distances
        return jnp.concatenate((result, _take(result, axis, slice(None, 1))), axis=axis)
    gradient = (_take(data, axis, slice(1, None)) - _take(data, axis, slice(None, -1))) * inverse_widths
    inner = (
        _take(gradient, axis, slice(1, None)) - _take(gradient, axis, slice(None, -1))
    ) * inverse_distances
    if not condition.is_mixed:
        # Zero at the wall faces; a pad, as zeros built under ``vmap`` would be a constant of the
        # batch that a program shared across grids cannot hold (2b.1).
        return jnp.pad(inner, [(1, 1) if position == axis else (0, 0) for position in range(inner.ndim)])
    zeros = jnp.zeros_like(_take(data, axis, slice(None, 1)))
    ends = [zeros, zeros]
    # A Neumann end of an inflow-outflow axis is an unknown face with zero normal gradient:
    # the flux beyond it vanishes over the half cell it owns, which keeps the operator symmetric.
    for end, (kind, sign, at) in enumerate(zip(condition.kinds, (1.0, -1.0), (0, -1), strict=True)):
        if kind == "neumann":
            edge = _take(gradient, axis, slice(at, None) if at else slice(None, 1))
            ends[end] = sign * edge * host_scalar(grid, _inverse_half_width, axis, at)
    return jnp.concatenate((ends[0], inner, ends[1]), axis=axis)


def _inverse_widths(grid: Grid, axis: int) -> np.ndarray:
    return _shaped(1.0 / np.asarray(grid.widths[axis]), axis)


def _inverse_centre_distances(grid: Grid, axis: int, periodic: bool) -> np.ndarray:
    widths = np.asarray(grid.widths[axis])
    distances = 0.5 * (widths + np.roll(widths, 1)) if periodic else 0.5 * (widths[:-1] + widths[1:])
    return _shaped(1.0 / distances, axis)


def _inverse_half_width(grid: Grid, axis: int, at: int):
    return 2.0 / np.asarray(grid.widths[axis])[at]


def _roll(data: jnp.ndarray, axis: int, shift: int) -> jnp.ndarray:
    return jnp.roll(data, shift, axis=axis)
