"""Charge-conservative electric coupling on the staggered grid.

For inductionless flow the current density is

.. math:: \\mathbf J = \\sigma(-\\nabla\\phi + \\mathbf u\\times\\mathbf B),
   \\qquad \\nabla\\cdot\\mathbf J = 0,

and the momentum equation is driven by :math:`\\mathbf J\\times\\mathbf B`.

The discretization follows Ni et al., *J. Comput. Phys.* **227** (2007) 174 and
205, which is what the codes that reproduce the ALEX ducts use. Its rule is that
one face-normal current

.. math:: J_{n,f} = \\sigma_f\\left[-\\frac{\\phi_N-\\phi_P}{d_{PN}}
   + (\\mathbf u_f\\times\\mathbf B_f)\\cdot\\mathbf n_f\\right]

is the single source of truth: the potential equation is the divergence of
exactly that flux, and the Lorentz force is rebuilt from the same numbers. The
reason is quantitative. In the core of a duct the momentum balance is
:math:`-\\nabla p + \\mathbf J\\times\\mathbf B = 0` to :math:`O(Ha^{-2})`, so an
:math:`O(\\Delta)` inconsistency between the two parts of :math:`\\mathbf J` is
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
the discrete form of :math:`\\int \\mathbf u\\cdot(\\mathbf J\\times\\mathbf B)
= -\\int \\mathbf J\\cdot(\\mathbf u\\times\\mathbf B)`: the Lorentz force does
exactly minus the Joule dissipation, and the steady Stokes operator is symmetric,
on a stretched mesh as on a uniform one. The distance-weighted interpolation
would be exact for a linear field on a stretched mesh, but its transpose is not
an average there, and a force built from it does work the currents never
dissipated.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._programs import host_array
from .bc import NEUMANN, BoundaryCondition, pad
from .grid import CENTER, FACE, Field, Grid
from .ops import (
    _broadcast,
    divergence,
    face_average,
    face_average_adjoint,
    face_distances,
    face_gradient,
    face_interpolate,
)

__all__ = [
    "cell_average",
    "charge_residual",
    "face_conductivity",
    "face_current",
    "face_electromotive_force",
    "lorentz_force",
    "thin_wall_current",
    "thin_wall_flux",
    "wall_insulated",
]

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
