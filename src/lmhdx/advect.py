"""Conservative momentum transport on the staggered grid.

The momentum equation for a duct at finite interaction parameter carries
:math:`\\nabla\\cdot(\\mathbf u\\mathbf u)`, and how that term is discretized
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
normal to :math:`j\\ne i` sit on the edges, where :math:`u_j` is averaged along
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
:math:`\\Delta t\\le(\\sum_j\\max|u_j|/\\Delta_j)^{-1}`, which is set by the flow
rather than by the mesh squared or by :math:`Ha^2`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .bc import BoundaryCondition, pad
from .grid import Field
from .ops import _broadcast, _take, face_distances

__all__ = ["advective_step_limit", "momentum_advection"]


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
    wrap has to skip that duplicate; :func:`lmhdx.bc.pad` assumes the cell-centred
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
