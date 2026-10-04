r"""Finite-volume stencils on the staggered grid.

Gradients map a cell-centred field to the faces normal to one axis; the
divergence maps face-normal fields back to cells as the net flux through the
cell's faces divided by its volume. Written this way the two are exact discrete
adjoints under the volume and face weights of :func:`cell_inner_product` and
:func:`face_inner_product`, which is the identity a compatible pressure or
potential solve depends on: it is what makes the projection idempotent and keeps
the Lorentz force from doing spurious work in the core of a high-Hartmann duct.

Ghost cells carry the wall condition (see :mod:`lmhdx.grid`), so a wall face is
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

Conservative momentum transport on the staggered grid.

The momentum equation for a duct at finite interaction parameter carries
:math:`\nabla\cdot(\mathbf u\mathbf u)`, and how that term is discretized
decides two things that matter more than its formal order. It must be
*conservative*, so that the momentum leaving one control volume is exactly the
momentum entering its neighbour and a long run cannot drift, and it must be
*bounded*, so that the thin side layers of a high-Hartmann-number duct do not
seed oscillations that a central difference would happily grow.

Both follow from writing the term as the divergence of a flux through the faces
of the staggered control volume of each component. For component :math:`i` the
control volume is the cell shifted by half a width along :math:`i`; its faces
normal to :math:`i` sit at cell centres, where the transported and transporting
velocity are the same average of the two neighbouring :math:`u_i`, and its faces
normal to :math:`j\ne i` sit on the edges, where :math:`u_j` is averaged along
:math:`i` and :math:`u_i` along :math:`j`. Differencing those fluxes telescopes:
the discrete momentum of a periodic box is conserved to round-off, which is the
property the tests pin rather than the truncation order.

The interpolation is distance weighted, so it stays second order on a stretched
mesh where the arithmetic mean is not. ``limited`` replaces it with the van Leer
blend between that central value and the upwind one; the blend is the identity
wherever the profile is smooth and monotone and collapses to first-order upwind
at an extremum, which is the usual and deliberate price of boundedness. Leave it
off to measure order on a smooth manufactured field, on for a real run.

Advection is explicit, so it reintroduces a step limit -- but a convective one,
:math:`\Delta t\le(\sum_j\max|u_j|/\Delta_j)^{-1}`, which is set by the flow
rather than by the mesh squared or by :math:`Ha^2`.

Charge-conservative electric coupling on the staggered grid.

For inductionless flow the current density is

.. math:: \mathbf J = \sigma(-\nabla\phi + \mathbf u\times\mathbf B),
   \qquad \nabla\cdot\mathbf J = 0,

and the momentum equation is driven by :math:`\mathbf J\times\mathbf B`.

The discretization follows Ni et al., *J. Comput. Phys.* **227** (2007) 174 and
205, which is what the codes that reproduce the ALEX ducts use. Its rule is that
one face-normal current

.. math:: J_{n,f} = \sigma_f\left[-\frac{\phi_N-\phi_P}{d_{PN}}
   + (\mathbf u_f\times\mathbf B_f)\cdot\mathbf n_f\right]

is the single source of truth: the potential equation is the divergence of
exactly that flux, and the Lorentz force is rebuilt from the same numbers. The
reason is quantitative. In the core of a duct the momentum balance is
:math:`-\nabla p + \mathbf J\times\mathbf B = 0` to :math:`O(Ha^{-2})`, so an
:math:`O(\Delta)` inconsistency between the two parts of :math:`\mathbf J` is
amplified by :math:`Ha^2` and appears as a spurious core current. Computing the
potential from one stencil and the force from another is what that rule forbids.

Face conductivity is the harmonic mean, which is the series resistance of the two
half-cells and therefore the right average across a fluid-wall jump; the
arithmetic mean would overstate the current entering a poorly conducting wall.

Velocities live on their own faces, so a transverse component is averaged to the
cell centre and then carried to the face where the electromotive force is
needed. The Lorentz force travels the same path backwards: each face current is
averaged to the cell centre, and the force is carried from there to the velocity
faces. Both paths use :func:`lmhdx.ops.face_average` and its transpose
:func:`lmhdx.ops.face_average_adjoint`, so the force map is exactly minus the
adjoint of the electromotive map in the volume-weighted inner products. That is
the discrete form of :math:`\int \mathbf u\cdot(\mathbf J\times\mathbf B)
= -\int \mathbf J\cdot(\mathbf u\times\mathbf B)`: the Lorentz force does
exactly minus the Joule dissipation, and the steady Stokes operator is symmetric,
on a stretched mesh as on a uniform one. The distance-weighted interpolation
would be exact for a linear field on a stretched mesh, but its transpose is not
an average there, and a force built from it does work the currents never
dissipated.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._programs import host_array, host_scalar
from .grid import CENTER, FACE, NEUMANN, BoundaryCondition, Field, Grid, pad

__all__ = [
    "advective_step_limit",
    "axis_divergence",
    "cell_average",
    "cell_inner_product",
    "charge_residual",
    "divergence",
    "face_average",
    "face_average_adjoint",
    "face_conductivity",
    "face_current",
    "face_distances",
    "face_electromotive_force",
    "face_gradient",
    "face_inner_product",
    "face_interpolate",
    "laplacian",
    "lorentz_force",
    "momentum_advection",
    "staggered_laplacian",
    "thin_wall_current",
    "thin_wall_flux",
    "wall_insulated",
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


# Conservative momentum transport on the staggered grid (formerly ``lmhdx.ops``).


def momentum_advection(
    velocity: tuple[Field, Field, Field],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    *,
    limited: bool = True,
) -> tuple[Field, Field, Field]:
    """Return :math:`\\nabla\\cdot(\\mathbf u\\mathbf u)` for each velocity component.

    The sign is that of the flux divergence, so a momentum equation subtracts it.
    ``conditions`` are the conditions the velocity itself sees, one per axis.
    """
    if len(conditions) != 3:
        raise ValueError("momentum advection needs one boundary condition per axis")
    if velocity[0].grid.is_polar:
        raise ValueError("momentum transport on a polar grid needs the curvature terms; not implemented")
    return tuple(_component(velocity, component, conditions, limited) for component in range(3))


def advective_step_limit(velocity: tuple[Field, Field, Field]) -> jnp.ndarray:
    """Return the convective step limit of a state, ``1 / sum_j max|u_j| / dx_j``."""
    grid = velocity[0].grid
    rate = sum(
        jnp.max(jnp.abs(field.data)) / float(np.min(grid.widths[axis])) for axis, field in enumerate(velocity)
    )
    return 1.0 / rate


def _component(
    velocity: tuple[Field, Field, Field],
    component: int,
    conditions: tuple[BoundaryCondition, ...],
    limited: bool,
) -> Field:
    field = velocity[component]
    total = _along_own_axis(field, component, conditions[component], limited)
    for axis in range(3):
        if axis != component:
            total = total + _across_axis(velocity, component, axis, conditions, limited)
    return field.replace_data(total)


def _along_own_axis(field: Field, axis: int, condition: BoundaryCondition, limited: bool) -> jnp.ndarray:
    """Difference the flux at the two cell centres bounding one velocity face."""
    data = field.data
    carrier = 0.5 * (_take(data, axis, slice(None, -1)) + _take(data, axis, slice(1, None)))
    stencil = _wide(data, axis, condition, field.grid, layers=1, duplicated=True)
    flux = carrier * _face_value(stencil, carrier, axis, 0.5, limited)
    difference = _take(flux, axis, slice(1, None)) - _take(flux, axis, slice(None, -1))
    if condition.is_periodic:
        wrap = _take(flux, axis, slice(None, 1)) - _take(flux, axis, slice(-1, None))
        difference = jnp.concatenate((wrap, difference, wrap), axis=axis)
    else:
        edge = jnp.zeros_like(_take(difference, axis, slice(None, 1)))
        difference = jnp.concatenate((edge, difference, edge), axis=axis)
    distances = face_distances(field.grid, axis, condition)
    return difference / _broadcast(distances, axis, field.dtype)


def _across_axis(
    velocity: tuple[Field, Field, Field],
    component: int,
    axis: int,
    conditions: tuple[BoundaryCondition, ...],
    limited: bool,
) -> jnp.ndarray:
    """Difference the flux at the two edges bounding one velocity face along ``axis``."""
    field = velocity[component]
    grid = field.grid
    carrier = _interpolate(velocity[axis].data, component, conditions[component], grid)
    stencil = _wide(field.data, axis, conditions[axis], grid, layers=2, duplicated=False)
    weight = _lower_weight(np.asarray(grid.widths[axis]))
    flux = carrier * _face_value(stencil, carrier, axis, _broadcast(weight, axis, field.dtype), limited)
    difference = _take(flux, axis, slice(1, None)) - _take(flux, axis, slice(None, -1))
    return difference / _broadcast(np.asarray(grid.widths[axis]), axis, field.dtype)


def _face_value(stencil, carrier, axis: int, weight, limited: bool):
    """Return the transported value at every flux point of ``axis``.

    ``stencil`` holds the four values each flux point needs, so it is three
    entries longer along ``axis`` than the number of flux points.
    """
    far_lower = _take(stencil, axis, slice(None, -3))
    lower = _take(stencil, axis, slice(1, -2))
    upper = _take(stencil, axis, slice(2, -1))
    far_upper = _take(stencil, axis, slice(3, None))
    central = weight * lower + (1.0 - weight) * upper
    if not limited:
        return central
    ahead = upper - lower
    behind = jnp.where(carrier >= 0.0, lower - far_lower, far_upper - upper)
    upwind = jnp.where(carrier >= 0.0, lower, upper)
    safe = jnp.where(ahead == 0.0, 1.0, ahead)
    ratio = jnp.where(ahead == 0.0, 0.0, behind / safe)
    return upwind + _van_leer(ratio) * (central - upwind)


def _wide(data, axis: int, condition: BoundaryCondition, grid, *, layers: int, duplicated: bool):
    """Return ``data`` with ``layers`` ghost entries at each end of ``axis``.

    A velocity field on a periodic axis repeats its first face at the end, so the
    wrap has to skip that duplicate; :func:`lmhdx.grid.pad` assumes the cell-centred
    layout and cannot tell the two apart. Away from a periodic axis the second
    ghost is the mirror of the first, and only the limiter ratio ever reads it.
    """
    if condition.is_periodic:
        shift = 1 if duplicated else 0
        below = _take(data, axis, slice(-layers - shift, -shift or None))
        above = _take(data, axis, slice(shift, layers + shift))
        return jnp.concatenate((below, data, above), axis=axis)
    padded = data
    for _ in range(layers):
        padded = pad(padded, axis, condition, grid=grid)
    return padded


def _van_leer(ratio):
    """The van Leer limiter: one where the profile is smooth, zero at an extremum."""
    magnitude = jnp.abs(ratio)
    return (ratio + magnitude) / (1.0 + magnitude)


def _interpolate(data, axis: int, condition: BoundaryCondition, grid) -> jnp.ndarray:
    """Move a field from cell centres to faces along ``axis``, weighted by distance."""
    padded = pad(data, axis, condition, grid=grid)
    weight = _broadcast(_lower_weight(np.asarray(grid.widths[axis])), axis, data.dtype)
    return weight * _take(padded, axis, slice(None, -1)) + (1.0 - weight) * _take(
        padded, axis, slice(1, None)
    )


def _lower_weight(widths: np.ndarray) -> np.ndarray:
    ghosted = np.concatenate(([widths[0]], widths, [widths[-1]]))
    return ghosted[1:] / (ghosted[:-1] + ghosted[1:])


# Charge-conservative electric coupling on the staggered grid (formerly ``lmhdx.ops``).

# (normal axis, first transverse axis, second transverse axis) in cyclic order.
_CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def cell_average(field: Field, axis: int | str) -> Field:
    """Average a face-normal field onto cell centres along ``axis``."""
    grid = field.grid
    index = grid.axis_index(axis)
    expected = tuple(FACE if position == index else CENTER for position in range(3))
    if field.offset != expected:
        raise ValueError(f"expected a field on faces normal to axis {index}, got offset {field.offset}")
    lower = field.data[(slice(None),) * index + (slice(None, -1),)]
    upper = field.data[(slice(None),) * index + (slice(1, None),)]
    return Field(0.5 * (lower + upper), (CENTER, CENTER, CENTER), grid)


def face_conductivity(sigma: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Return the harmonic-mean conductivity on the faces normal to ``axis``.

    The harmonic mean is the series resistance of the two half-cells, so it is
    the average that reproduces the current through a fluid-wall interface. Its
    arithmetic counterpart lets a poorly conducting wall draw too much current.
    """
    grid = sigma.grid
    index = grid.axis_index(axis)
    if sigma.offset != (CENTER, CENTER, CENTER):
        raise ValueError("conductivity must be cell centred")
    if jnp.ndim(sigma.data) != 3 or sigma.shape != grid.shape:
        raise ValueError(f"conductivity shape {sigma.shape} does not match grid {grid.shape}")
    padded = pad(sigma.data, index, condition, grid=grid)
    lower = padded[(slice(None),) * index + (slice(None, -1),)]
    upper = padded[(slice(None),) * index + (slice(1, None),)]
    lower_half, upper_half = (
        host_array(grid, _half_widths, index, side, dtype=sigma.dtype) for side in (0, 1)
    )
    total = lower_half + upper_half
    resistance = lower_half / lower + upper_half / upper
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(total / resistance, offset, grid)


def _half_widths(grid: Grid, axis: int, side: int) -> np.ndarray:
    """Half the width of the cell below (``side`` 0) or above each face, the wall cell mirrored."""
    widths = np.asarray(grid.widths[axis])
    ghosted = np.concatenate(([widths[0]], widths, [widths[-1]]))
    half = 0.5 * (ghosted[1:] if side else ghosted[:-1])
    return half.reshape([-1 if position == axis else 1 for position in range(3)])


def face_electromotive_force(
    velocity: tuple[Field, Field, Field],
    magnetic_field: tuple[Field, Field, Field],
    axis: int | str,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> Field:
    """Return ``(u x B).n`` on the faces normal to ``axis``.

    ``velocity`` holds the three face-normal components in the marker-and-cell
    layout and ``magnetic_field`` the three cell-centred components. Both
    transverse velocity components are averaged to cell centres with
    :func:`lmhdx.ops.face_average_adjoint` and carried to the target face with
    :func:`lmhdx.ops.face_average`, the transpose of the path
    :func:`lorentz_force` and the momentum update take back, so the
    electromotive force is evaluated where the current flux needs it.
    """
    grid = velocity[0].grid
    index = grid.axis_index(axis)
    _, first, second = _CYCLIC[index]
    velocity_first = face_average(
        face_average_adjoint(velocity[first], first, conditions[first]), index, conditions[index]
    )
    velocity_second = face_average(
        face_average_adjoint(velocity[second], second, conditions[second]), index, conditions[index]
    )
    field_first = face_interpolate(magnetic_field[first], index, conditions[index])
    field_second = face_interpolate(magnetic_field[second], index, conditions[index])
    emf = velocity_first.data * field_second.data - velocity_second.data * field_first.data
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(emf, offset, grid)


def wall_insulated(flux: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Zero a face-normal flux on the two boundary faces of an insulating wall.

    An insulating wall carries no current, :math:`\\mathbf J\\cdot\\mathbf n = 0`,
    and both sides of the potential equation have to say so. The homogeneous
    Neumann Laplacian already drops the wall faces; unless the motional term is
    dropped there too, the equation asks the potential to absorb a boundary flux
    that the operator cannot produce, and the resulting current -- and the
    Lorentz force built from it -- is wrong wherever a side wall cuts across
    :math:`\\mathbf u\\times\\mathbf B`.

    Only a Neumann condition names an insulating wall. A prescribed potential
    is a perfectly conducting one and does carry current, so it is returned
    untouched, as is a periodic axis, which has no wall at all.
    """
    index = flux.grid.axis_index(axis)
    if condition.kind != NEUMANN:
        return flux
    selection = (slice(None),) * index
    data = flux.data.at[selection + (0,)].set(0.0).at[selection + (-1,)].set(0.0)
    return flux.replace_data(data)


def thin_wall_current(potential: Field, wall_potential: Field, conductivity: Field, axis: int | str) -> Field:
    """Return the current the fluid sends into a thin conducting wall, on its two wall faces.

    A thin wall is a sheet with a potential of its own, reached across the half
    cell against it: Ohm's law over that half cell, ``sigma (phi_P - phi_w)/(h_P/2)``
    outward, with no electromotive force because the wall does not move. It is the
    prescribed-value wall of :func:`face_current` with the sheet potential as the
    value, second order where the adjacent cell value was first. Only the two wall
    entries of ``wall_potential`` (on the faces normal to ``axis``) are read; one
    equal to the adjacent cell carries no current. Interior faces are zero, so the
    result adds to an insulated :func:`face_current`.
    """
    grid = potential.grid
    index = grid.axis_index(axis)
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    if wall_potential.offset != offset or wall_potential.shape != grid.face_shape(index):
        raise ValueError(f"the wall potential must live on the faces normal to axis {index}")
    if conductivity.offset != offset or conductivity.shape != grid.face_shape(index):
        raise ValueError("conductivity must live on the wall faces")
    current = jnp.zeros(grid.face_shape(index), dtype=potential.dtype)
    # The stored value points along the axis: outward on the upper wall, inward on the lower.
    for face, cell, sign in ((0, 0, 1.0), (-1, -1, -1.0)):
        at = (slice(None),) * index + (face,)
        difference = wall_potential.data[at] - potential.data[(slice(None),) * index + (cell,)]
        # The half width is read from a constant, not written as a literal (2b.1).
        half = host_array(grid, _wall_half_width, index, cell, dtype=potential.dtype)[0]
        current = current.at[at].set(sign * conductivity.data[at] * difference / half)
    return Field(current, offset, grid)


def _wall_half_width(grid: Grid, axis: int, cell: int) -> np.ndarray:
    widths = np.asarray(grid.widths[axis])
    return 0.5 * widths[cell : cell + 1 if cell >= 0 else None][:1]


def thin_wall_flux(
    wall_potential: Field,
    axis: int | str,
    condition: BoundaryCondition,
    conductance: float | tuple[float, float],
    tangential: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> Field:
    """Return the current a thin conducting wall carries away along itself, on its two wall faces.

    A wall of conductance :math:`\\sigma_w t_w` conducts along itself, with a
    surface current :math:`\\mathbf K = -c\\nabla_\\tau\\varphi_w` for
    :math:`c = \\sigma_w t_w/(\\sigma a)`. Charge conservation in the sheet,
    :math:`\\nabla_\\tau\\cdot\\mathbf K = \\mathbf J\\cdot\\mathbf n`, makes
    the wall-normal current :math:`-c\\nabla_\\tau^2\\varphi_w` -- Walker's thin-wall
    condition, and the reason a Hunt duct carries a fraction of its current
    through the wall instead of through the Hartmann layer. The sheet balance
    holds where this equals :func:`thin_wall_current`.

    ``wall_potential`` is read as in :func:`thin_wall_current`, and the five-point
    surface Laplacian is taken on the sheet with the wall's own metric: on a polar
    grid the azimuthal term of a radial wall divides by the wall radius, not by the
    radius of the cells beside it. ``tangential`` closes the sheet's edges.
    ``conductance`` is one value or a ``(lower, upper)`` pair; zero conducts nothing.
    """
    grid = wall_potential.grid
    index = grid.axis_index(axis)
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    if wall_potential.offset != offset or wall_potential.shape != grid.face_shape(index):
        raise ValueError(f"the wall potential must live on the faces normal to axis {index}")
    pair = (conductance, conductance) if np.ndim(conductance) == 0 else conductance
    flux = jnp.zeros(grid.face_shape(index), dtype=wall_potential.dtype)
    for position, value, sign in ((0, float(pair[0]), 1.0), (-1, float(pair[1]), -1.0)):
        if condition.is_periodic or not value:
            continue
        at = (slice(None),) * index + (slice(position, position + 1) if position == 0 else slice(-1, None),)
        sheet, total = wall_potential.data[at], 0.0
        for other in (other for other in range(3) if other != index):
            # The azimuth of a polar grid is an angle: lengths along it scale with the sheet's radius.
            scale = float(grid.faces[index][position]) if grid.is_polar and (index, other) == (0, 1) else 1.0
            gradient = jnp.diff(pad(sheet, other, tangential[other]), axis=other) / _broadcast(
                scale * np.asarray(face_distances(Grid(*grid.faces), other, tangential[other])),
                other,
                sheet.dtype,
            )
            total = total + jnp.diff(gradient, axis=other) / _broadcast(
                scale * grid.widths[other], other, sheet.dtype
            )
        # The stored value points along the axis: outward on the upper wall, inward on the lower.
        flux = flux.at[at].set(sign * value * total)
    return Field(flux, offset, grid)


def face_current(
    potential: Field,
    conductivity: Field,
    electromotive_force: Field,
    axis: int | str,
    condition: BoundaryCondition,
) -> Field:
    """Return the face-normal current density of Ohm's law on one face set.

    This is the only place a current is formed. The potential equation and the
    Lorentz force are both built from its result, which is what keeps them
    consistent at high Hartmann number.
    """
    gradient = face_gradient(potential, axis, condition)
    if conductivity.offset != gradient.offset or conductivity.shape != gradient.shape:
        raise ValueError("conductivity must live on the same faces as the potential gradient")
    if electromotive_force.offset != gradient.offset or electromotive_force.shape != gradient.shape:
        raise ValueError("electromotive force must live on the same faces as the potential gradient")
    return gradient.replace_data(conductivity.data * (electromotive_force.data - gradient.data))


def charge_residual(currents: tuple[Field, Field, Field]) -> Field:
    """Return the net current leaving each cell per unit volume."""
    return divergence(currents)


def lorentz_force(
    currents: tuple[Field, Field, Field],
    magnetic_field: tuple[Field, Field, Field],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> tuple[Field, Field, Field]:
    """Return the cell-centred Lorentz force built from the face currents.

    Ni's face form is used,

    .. math:: (\\mathbf J\\times\\mathbf B)_c
       = \\frac{1}{\\Omega_c}\\sum_f J_{n,f}\\,s_f\\,(\\mathbf r_f-\\mathbf r_c)
         \\times\\mathbf B_f,

    which never forms a cell-centred current vector and evaluates the magnetic
    field on the faces, so it stays correct where the field varies. The
    face-to-centre arm is half a cell, so the sum over the two faces of one axis
    is :func:`lmhdx.ops.face_average_adjoint` of the current times the face field:
    the transpose of the interpolation :func:`face_electromotive_force` uses.
    """
    grid = currents[0].grid
    components = [jnp.zeros(grid.shape, dtype=currents[0].dtype) for _ in range(3)]
    for index, current in enumerate(currents):
        _, first, second = _CYCLIC[index]
        field_first = face_interpolate(magnetic_field[first], index, conditions[index])
        field_second = face_interpolate(magnetic_field[second], index, conditions[index])
        # With the current stored along the positive axis, the outward flux and the
        # face-to-centre arm change sign together on the lower face, so the two
        # faces add rather than cancel. The cross product e_k x B contributes
        # +B_first to the second transverse component and -B_second to the first.
        for target, source, sign in ((second, field_first, 1.0), (first, field_second, -1.0)):
            averaged = face_average_adjoint(
                current.replace_data(current.data * source.data), index, conditions[index]
            )
            components[target] = components[target] + sign * averaged.data
    return tuple(Field(component, (CENTER, CENTER, CENTER), grid) for component in components)
