"""One projection step for inductionless duct flow on the staggered grid.

The step is the classical fractional one, assembled from the pieces earlier plan
steps settled: the potential is solved first, the face currents of :mod:`lmhdx.ops`
supply the Lorentz force, the momentum is advanced, and a pressure Poisson solve
projects the velocity back onto the discretely divergence-free space.

Two details are specific to magnetohydrodynamics and matter more than the rest of
the step put together.

*Stiffness from the field.* The velocity part of the Lorentz force is a damping,
:math:`-\\sigma(B^2 I - \\mathbf B\\mathbf B^{\\mathsf T})\\mathbf u`, with rate
:math:`\\sigma B^2/\\rho`. Treating it explicitly forces
:math:`\\Delta t \\propto Ha^{-2}`, which is why explicit codes stall at
blanket-scale Hartmann numbers. It is applied here as a correction to the
conservative face force, so the stage reads ``H u + (dt/rho) G p = b``,
``D u = 0``, with ``H = (1 + dt * lambda)`` per component and ``b`` the old
velocity plus the forces. The pressure gradient meets the same ``H`` as every other
force, so a pressure force in balance with the forcing leaves the fluid at rest and
the stage's fixed point is the physical steady state. The pressure solves the
matched Schur operator ``D H^-1 G``, exactly, by SOLVAX's constrained solve. The
correction vanishes when the right-hand side does, and the conservative force keeps
its own discretization.

*Consistency.* The force comes from the same face currents the potential equation
is built on, never from a separately differenced potential. In the core the
balance is :math:`-\\nabla p + \\mathbf J\\times\\mathbf B = 0` to
:math:`O(Ha^{-2})`, so any inconsistency between the two is amplified by
:math:`Ha^2`. The insulating wall is part of that consistency: the motional term
is dropped on wall faces by :func:`lmhdx.ops.wall_insulated` before its divergence
is taken, because the operator that receives it has no wall flux either. Leaving
it in makes the potential absorb a boundary current the wall cannot carry, and a
square duct at :math:`Ha=20` then runs at less than half its correct flow rate.

Two constraints must hold before the divergence of a face velocity means
anything, and both are imposed rather than assumed. A wall-normal component is
set to zero on its wall faces, and on a periodic axis the duplicated first and
last face are made equal. Without either the discrete divergence carries a net
flux through the boundary, and with every axis periodic or Neumann the pressure
has no way to remove that constant: the projection would return a field that is
still not divergence free.

*Stiffness from viscosity.* Diffusion may be taken either way. Left explicit it
bounds the step by :attr:`ChannelProblem.diffusive_step_limit`, reported rather
than enforced so a caller sweeping a parameter sees the constraint instead of a
silently clipped step. Passing the factorizations from
:meth:`ChannelProblem.viscous_factorizations` to :func:`step` solves

.. math:: \\left[(1+\\Delta t\\,\\lambda)I-\\Delta t\\,\\nu\\nabla^2\\right]\\mathbf u^{*}
   =(1+\\Delta t\\,\\lambda)\\mathbf u+\\Delta t\\,(\\mathbf F+\\mathbf f-\\nabla p)/\\rho,

backward Euler on the viscous term with the damping correction folded into the
shift, ``H`` of the stage. Next to a wall ``H`` does not commute with the gradient, so
``D H^-1 G`` is solved by conjugate gradients preconditioned with its spectral
multiplier (one iteration on periodic axes, a handful near walls), and the step is
bounded by accuracy rather than by the mesh. Both stiffnesses are gone at that point: the
magnetic one because it grows as :math:`Ha^2`, the viscous one because it grows as the
mesh is refined.

*Convective transport.* ``advection`` selects it: ``"off"`` is the Stokes limit,
appropriate at the large interaction parameters of a blanket channel and the
default so that no run acquires a convective step limit by accident;
``"central"`` is the conservative flux form of :mod:`lmhdx.ops`, second order on
a stretched mesh; ``"limited"`` adds the van Leer blend that keeps the thin side
layers bounded. Transport is explicit, so switching it on bounds the step by
:func:`lmhdx.ops.advective_step_limit`.

*Time stepping.* ``scheme`` is :data:`EULER`, one stage, or :data:`RK3`, the low-storage
IMEX scheme of Spalart, Moser and Rogers with Crank-Nicolson viscous and damping terms:
each stage is the constrained stage above with the weights of its :class:`Tableau`, so
the rest state and the divergence constraint hold at every stage. Forward Euler with
central transport is unstable at every Courant number; the RK3 stage map is stable to
``sqrt 3`` (:func:`courant_step`). RK3 is third order for transport and second order
for the implicit terms, which is what the manufactured and linear-growth tests measure.

Run the projection step as one compiled trajectory.

A Python loop around :func:`lmhdx.core3d.step` dispatches every operation from the
host. On an accelerator that is the difference between a kernel queue the device
can run ahead on and a round trip per step, and it is the reason the audit that
opened this plan found one GPU slower than a laptop CPU on a small duct. The
whole trajectory is compiled here instead, with :func:`jax.lax.scan`.

Reverse-mode differentiation of a long trajectory is the other half of the
problem. Storing every intermediate state is what exhausts device memory, so the
step is wrapped in :func:`jax.checkpoint`: the forward pass keeps one state per
step and the backward pass recomputes the interior of each step. SOLVAX's
``checkpointed_fori_loop`` offers the square-root schedule for the deeper case;
per-step rematerialisation is the simpler contract and is what the momentum step
needs, since its expensive parts are the two Poisson solves rather than a long
chain of cheap operations.

Diagnostics come out as scan outputs rather than through a host callback, so a
run reports its history without ever synchronising mid-trajectory.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from . import _pin_matmul_precision
from ._programs import attribute, bound, host_array, host_scalar
from .grid import (
    CENTER,
    DIRICHLET,
    FACE,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from .ops import (
    advective_step_limit,
    cell_force_to_faces,
    divergence,
    face_conductivity,
    face_current,
    face_electromotive_force,
    face_gradient,
    face_inner_product,
    lorentz_force,
    momentum_advection,
    staggered_laplacian,
    thin_wall_current,
    wall_insulated,
)
from .poisson import (
    FastDiagonalHelmholtz,
    FastDiagonalPoisson,
    FastDiagonalThinWallPoisson,
    _nearer_first_wall,
    _wall_cells,
    fast_diagonal_helmholtz,
    fast_diagonal_poisson,
    fast_diagonal_thin_wall_poisson,
)

__all__ = [
    "ChannelProblem",
    "ConjugateGrid",
    "EULER",
    "EnergyBudget",
    "ImposedField",
    "RK3",
    "SolidWall",
    "Tableau",
    "Trajectory",
    "advance",
    "conjugate_grid",
    "courant_step",
    "duct_problem",
    "electric_state",
    "energy_budget",
    "enforce_face_constraints",
    "face_currents",
    "face_lorentz_force",
    "fringe_field",
    "kinetic_energy",
    "project",
    "step",
    "trajectory_diagnostics",
    "velocity_condition",
    "velocity_offset",
    "zero_velocity",
]

_NO_SLIP = BoundaryCondition(DIRICHLET)
_INSULATING = BoundaryCondition("neumann")
# The pressure of an inflow-outflow axis: no correction through the inlet, p = 0 at the outlet (D26).
_OUTLET_PRESSURE = BoundaryCondition(NEUMANN, upper_kind=DIRICHLET)
_ADVECTION = ("off", "central", "limited")


class Tableau(NamedTuple):
    """An IMEX Runge-Kutta scheme in the low-storage form of Spalart, Moser and Rogers (1991).

    A stage advances ``u`` by ``dt (explicit R(u_old) + carried R(u_prev) + (explicit + carried -
    implicit) L(u_old) + implicit L(u_new))``: ``R`` is transport and the force that is not damping,
    ``L`` the damping and, if implicit, the viscous term. ``limit`` is the largest Courant number
    ``dt sum_a |u_a|/dx_a`` that keeps central transport stable (``0`` for Euler, which never does).
    """

    explicit: tuple[float, ...]
    carried: tuple[float, ...]
    implicit: tuple[float, ...]
    limit: float


EULER = Tableau((1.0,), (0.0,), (1.0,), 0.0)
RK3 = Tableau((8 / 15, 5 / 12, 3 / 4), (0.0, -17 / 60, -5 / 12), (4 / 15, 1 / 15, 1 / 6), float(np.sqrt(3.0)))


def velocity_offset(component: int) -> tuple[float, float, float]:
    """Return the staggered position of one velocity component."""
    return tuple(FACE if axis == component else CENTER for axis in range(3))


def velocity_condition(
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition], axis: int
) -> BoundaryCondition:
    """Return the condition a velocity component sees along ``axis``.

    A periodic axis stays periodic. At a wall every component is homogeneous:
    the tangential ones by no slip, the normal one because it sits on the wall
    and is prescribed there by :func:`enforce_face_constraints`. An
    inflow-outflow axis keeps its kinds with the inlet data removed: the solve
    carries the inlet as a lift (:mod:`lmhdx.axial`), so every component is zero
    at the inlet and has zero axial gradient at the outlet.
    """
    condition = conditions[axis]
    if condition.is_mixed:
        return condition.homogeneous()
    return condition if condition.is_periodic else _NO_SLIP


@dataclass(frozen=True, eq=False)
class ImposedField:
    """An imposed magnetic field that varies in space, at the cell centres of one grid.

    ``components`` are ``(B_x, B_y, B_z)`` of ``grid.shape``; the optional ``faces``
    are the face-normal components whose discrete divergence :func:`fringe_field`
    controls. The arrays are copied read-only and compared by value, as
    :class:`lmhdx.grid.Grid` is, so a problem carrying the field stays static. Both
    the electromotive force and the Lorentz force multiply a component at the
    current face, so the force stays exactly minus the adjoint of the other.
    """

    grid: Grid
    components: tuple[np.ndarray, np.ndarray, np.ndarray]
    faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        shapes = {"components": (self.grid.shape,) * 3, "faces": tuple(map(self.grid.face_shape, range(3)))}
        for name, expected in shapes.items():
            arrays = getattr(self, name)
            if arrays is None:
                continue
            if len(arrays) != 3:
                raise ValueError(f"an imposed field needs three {name}")
            frozen = tuple(np.array(array, dtype=np.float64) for array in arrays)
            for array, shape in zip(frozen, expected, strict=True):
                if array.shape != shape:
                    raise ValueError(f"imposed field {name} of shape {array.shape} do not match {shape}")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"imposed field {name} must be finite")
                array.setflags(write=False)
            object.__setattr__(self, name, frozen)

    def _key(self) -> tuple:
        arrays = self.components + (self.faces or ())
        return (self.grid, self.faces is None, *(array.tobytes() for array in arrays))

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return self._key() == other._key() if isinstance(other, ImposedField) else NotImplemented


@dataclass(frozen=True)
class SolidWall:
    """A solid wall on one end of a transverse axis: ``thickness`` in ``cells`` equal cells of one material.

    ``thermal_conductivity`` and ``heat_capacity``, its ``rho c_p``, are what it conducts and stores heat with
    (plan step 5.4b, ADR 0009 D36).
    """

    thickness: float
    cells: int
    thermal_conductivity: float
    heat_capacity: float

    def __post_init__(self) -> None:
        if isinstance(self.cells, bool) or int(self.cells) != self.cells or self.cells < 1:
            raise ValueError("a solid wall has a whole, positive number of cells")
        object.__setattr__(self, "cells", int(self.cells))
        for name in ("thickness", "thermal_conductivity", "heat_capacity"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"a solid wall's {name} must be positive and finite")
            object.__setattr__(self, name, value)


class ConjugateGrid(NamedTuple):
    """A grid with solid walls added as cells (:func:`conjugate_grid`): the grid, the fluid's block, the walls."""

    grid: Grid
    fluid: tuple[slice, slice, slice]
    walls: tuple

    def cells(self, name: str, fluid, walls: dict | None = None):
        """Each cell's value of the walls' property ``name``, with ``fluid`` in the fluid's cells.

        ``walls`` maps a wall, ``(axis, side)``, to a value in place of its own. The values may then be traced: the
        result is a ``jnp`` array and which cell is which stays static, so a wall's conductivity or heat capacity can
        be a leaf of a solve and a new value does not recompile. A corner cell lies in a wall of both transverse axes
        and takes the nearer wall's value, the first axis's when they are equally near, as the corner cells of
        resolved walls do (:func:`lmhdx.poisson.fast_diagonal_thin_wall_poisson`).
        """
        slots = [(axis, side) for axis in (1, 2) for side in (0, 1)]
        own = [
            fluid if self.walls[axis][side] is None else getattr(self.walls[axis][side], name)
            for axis, side in slots
        ]
        if not walls:
            return np.asarray([fluid, *own], dtype=float)[self._labels()]
        if any(slot not in slots or self.walls[slot[0]][slot[1]] is None for slot in walls):
            raise ValueError("a wall's value is given by (axis, side) of a solid wall of the grid")
        table = [walls.get(slot, value) for slot, value in zip(slots, own)]
        return jnp.stack([jnp.asarray(value, dtype=float) for value in (fluid, *table)])[self._labels()]

    def _labels(self) -> np.ndarray:
        """Each cell's material: 0 the fluid's, ``1 + 2 (axis - 1) + side`` a wall's; a corner the nearer wall's."""
        labels, depths = [], []
        for axis in (1, 2):
            block = self.fluid[axis]
            faces, centres = np.asarray(self.grid.faces[axis]), np.asarray(self.grid.centers[axis])
            label = np.zeros(centres.size, dtype=int)
            label[: block.start], label[block.stop :] = 2 * axis - 1, 2 * axis
            labels.append(label)
            inside = (faces[block.start], faces[block.stop])
            depths.append(np.maximum(inside[0] - centres, 0.0) + np.maximum(centres - inside[1], 0.0))
        first, second = labels[0][:, None], labels[1][None, :]
        takes_first = (first > 0) & ((second == 0) | _nearer_first_wall(depths[0], depths[1]))
        return np.broadcast_to(np.where(takes_first, first, second), self.grid.shape)


def conjugate_grid(grid: Grid, walls) -> ConjugateGrid:
    """Return ``grid`` with solid walls added as cells beyond its faces on the transverse axes 1 and 2.

    ``walls`` gives each axis ``(lower, upper)``, each ``None`` or a :class:`SolidWall`; axis 0 must be the
    flow's and takes none. A wall spans the extended tangential extent, so walls on both axes meet in corner
    cells. The fluid's faces are kept exactly, so the fluid's block of the new grid is ``grid`` itself.
    """
    if grid.is_polar:
        raise ValueError("solid walls need a Cartesian grid")
    walls = tuple((None, None) if pair is None else tuple(pair) for pair in walls)
    if len(walls) != 3 or any(
        len(pair) != 2 or any(not (wall is None or isinstance(wall, SolidWall)) for wall in pair)
        for pair in walls
    ):
        raise ValueError("solid walls are a (lower, upper) pair per axis, each None or a SolidWall")
    if any(walls[0]):
        raise ValueError("solid walls go on the transverse axes 1 and 2; axis 0 is the flow's")
    faces, fluid = [np.asarray(grid.faces[0])], [slice(0, grid.shape[0])]
    for axis in (1, 2):
        edges, (lower, upper) = np.asarray(grid.faces[axis]), walls[axis]
        below = (
            [] if lower is None else edges[0] - lower.thickness * (1.0 - np.arange(lower.cells) / lower.cells)
        )
        above = (
            [] if upper is None else edges[-1] + upper.thickness * np.arange(1, upper.cells + 1) / upper.cells
        )
        faces.append(np.concatenate([below, edges, above]))
        fluid.append(slice(len(below), len(below) + grid.shape[axis]))
    return ConjugateGrid(Grid(*faces), tuple(fluid), walls)


@dataclass(frozen=True)
class ChannelProblem:
    """A duct segment with uniform material properties and an imposed field.

    ``magnetic_field`` is three numbers for a uniform field, or a varying one: an
    :class:`ImposedField`, or three arrays of ``grid.shape`` at the cell centres
    (scalars broadcast), which become one.

    One axis may be an inflow-outflow axis (plan D26): a condition of kind
    ``dirichlet`` with ``upper_kind="neumann"``, whose ``lower`` is the inlet
    profile of the velocity normal to it. :mod:`lmhdx.axial` builds and solves
    such a duct, in the Stokes limit or, with advection, by Newton's method.

    ``wall_conductance`` closes an axis with thin conducting walls, the same on
    both. ``wall_layers`` resolves walls in cells instead: per axis
    ``(lower, upper)``, each ``None`` (insulating) or ``(ratios, widths)``, each
    cell's conductivity over the fluid's (one number for all) and its width, from
    the fluid outwards, insulated outside; stored per cell. A wall carries
    potential and current only and spans the fluid's tangential extent. Two axes
    may have them when the third has one cell; a corner cell then takes the
    nearer wall's layer (:func:`lmhdx.poisson.fast_diagonal_thin_wall_poisson`).

    The wall fields are the static pattern and the values the host factorization is built from. A
    wall parameter under a derivative rides as an argument instead: :meth:`with_walls` attaches the
    factorization built from it in the trace as ``potential_walls``, which is neither hashed nor
    compared, so a problem that carries one belongs to one trace and is never a cache key.
    """

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    density: float = 1.0
    viscosity: float = 1.0
    conductivity: float = 1.0
    magnetic_field: tuple[float, float, float] | ImposedField = (0.0, 0.0, 0.0)
    forcing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dt: float = 1.0e-3
    advection: str = "off"
    wall_conductance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    precision: str = "state"
    wall_layers: tuple = (None, None, None)
    potential_walls: object = dataclasses.field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.conditions) != 3:
            raise ValueError("a channel needs one boundary condition per axis")
        for name in ("density", "viscosity", "dt"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.conductivity) < 0.0:
            raise ValueError("conductivity must not be negative")
        if self.advection not in _ADVECTION:
            raise ValueError(f"advection must be one of {sorted(_ADVECTION)}, got {self.advection!r}")
        if len(self.wall_conductance) != 3:
            raise ValueError("a channel needs one wall conductance per axis")
        if any(float(value) < 0.0 for value in self.wall_conductance):
            raise ValueError("wall conductance must not be negative")
        if self.precision not in ("state", "mixed"):
            raise ValueError(f"precision must be 'state' or 'mixed', got {self.precision!r}")
        self._freeze_wall_layers()
        self._check_open_axis()
        field = self.magnetic_field
        if not isinstance(field, ImposedField):
            if len(field) != 3:
                raise ValueError("a channel needs three magnetic field components")
            if any(np.ndim(value) for value in field):
                shape = self.grid.shape
                field = ImposedField(
                    self.grid, tuple(np.broadcast_to(v, shape) if np.ndim(v) == 0 else v for v in field)
                )
                object.__setattr__(self, "magnetic_field", field)
            elif not np.all(np.isfinite(np.asarray(field, dtype=float))):
                raise ValueError("the magnetic field must be finite")
        if isinstance(field, ImposedField) and field.grid != self.grid:
            raise ValueError("the imposed field is sampled on a different grid")
        # True float32 contractions unless the user chose a precision; see lmhdx.enable_x64.
        _pin_matmul_precision()

    def _freeze_wall_layers(self) -> None:
        """Store ``wall_layers`` as per-cell ``(ratios, widths)`` tuples, so the problem stays hashable."""
        if len(self.wall_layers) != 3:
            raise ValueError("a channel needs one wall-layer entry per axis")
        frozen = []
        for axis, pair in enumerate(self.wall_layers):
            if not pair or not any(pair):
                frozen.append(None)
                continue
            if len(pair) != 2 or self.conditions[axis].is_periodic or float(self.wall_conductance[axis]):
                raise ValueError(
                    f"axis {axis} takes resolved walls as a (lower, upper) pair on a wall-bounded axis"
                )
            cells = [None if end is None else _wall_cells(end) for end in pair]
            frozen.append(
                tuple(
                    None if end is None else tuple(tuple(map(float, part)) for part in end) for end in cells
                )
            )
        resolved = [axis for axis, pair in enumerate(frozen) if pair]
        if len(resolved) > 1 and (len(resolved) > 2 or self.grid.shape[3 - sum(resolved)] != 1):
            raise ValueError("walls resolved in cells on two axes need one cell along the third")
        if resolved and any(float(value) for value in self.wall_conductance):
            raise ValueError("walls resolved in cells take no thin wall on another axis")
        object.__setattr__(self, "wall_layers", tuple(frozen))

    def _check_open_axis(self) -> None:
        mixed = [axis for axis, condition in enumerate(self.conditions) if condition.is_mixed]
        if not mixed:
            return
        condition = self.conditions[mixed[0]]
        if len(mixed) > 1 or condition.kinds != (DIRICHLET, NEUMANN) or np.any(condition.upper):
            raise ValueError("one inflow-outflow axis at most: a Dirichlet inlet below, a free outlet above")
        if float(self.wall_conductance[mixed[0]]) or self.wall_layers[mixed[0]]:
            raise ValueError("an inflow-outflow axis has no wall")

    @property
    def open_axis(self) -> int | None:
        """The inflow-outflow axis, or ``None``."""
        return next((axis for axis, condition in enumerate(self.conditions) if condition.is_mixed), None)

    @property
    def pressure_conditions(self) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
        """The pressure's conditions: :attr:`scalar_conditions`, except ``p = 0`` at an outlet."""
        return tuple(
            _OUTLET_PRESSURE if condition.is_mixed else scalar
            for condition, scalar in zip(self.conditions, self.scalar_conditions, strict=True)
        )

    @property
    def conducting_walls(self) -> bool:
        """Whether any wall carries current along itself."""
        return any(float(value) > 0.0 for value in self.wall_conductance) or any(self.wall_layers)

    @property
    def scalar_conditions(self) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
        """Conditions for the potential: periodic, else a homogeneous Neumann wall.

        The ends of an inflow-outflow axis are insulating too: no normal current,
        ``dphi/dn = (u x B).n`` (D26), with the potential's mean removed.
        """
        return tuple(condition if condition.is_periodic else _INSULATING for condition in self.conditions)

    @property
    def peak_field_squared(self) -> float:
        """Return the largest :math:`|\\mathbf B|^2` over the cells: a uniform field's own."""
        field = self.magnetic_field
        if isinstance(field, ImposedField):
            return float(np.max(sum(np.square(component) for component in field.components)))
        return float(np.dot(field, field))

    @property
    def damping_rates(self) -> tuple[float, float, float]:
        """Return the implicit damping rate of each velocity component.

        The diagonal of :math:`\\sigma(B^2 I - \\mathbf B\\mathbf B^{\\mathsf T})/\\rho`
        is used. Off-diagonal coupling stays in the explicit conservative force,
        which is exact for a field aligned with an axis and a documented
        approximation otherwise.

        A varying field takes the largest rate over the cells, since the fast solve
        needs one shift: a cell of rate ``r`` then updates by ``1 - dt r/(1 + dt lambda)``
        in ``(0, 1]``, where the volume mean overshoots once ``dt r > 2 (1 + dt lambda)``.
        """
        field = self.magnetic_field
        if isinstance(field, ImposedField):
            squares = [np.square(component) for component in field.components]
            total = sum(squares)
            return tuple(
                float(self.conductivity) * float(np.max(total - square)) / float(self.density)
                for square in squares
            )
        squared = float(np.dot(field, field))
        return tuple(
            float(self.conductivity) * (squared - float(component) ** 2) / float(self.density)
            for component in self.magnetic_field
        )

    @property
    def diffusive_step_limit(self) -> float:
        """Return the largest stable step for the explicit viscous term.

        The usual bound for an explicit centred Laplacian,
        ``dt <= 1 / (2 * nu * sum(1 / dx**2))``, evaluated on the finest cell of
        each axis. The magnetic damping imposes no such limit here because it is
        treated implicitly, so this is normally the binding constraint.
        """
        inverse = sum(1.0 / float(np.min(widths)) ** 2 for widths in self.grid.widths)
        return 1.0 / (2.0 * float(self.viscosity) * inverse)

    def viscous_factorizations(self, scheme: Tableau = EULER) -> tuple:
        """Factorize the implicit viscous operator for each velocity component.

        Build these once on the host and pass them to :func:`step` to take
        diffusion implicitly. The shift carries the magnetic damping, so one
        solve removes both stiff terms. :attr:`precision` selects the float32
        solve with float64 correction of :mod:`lmhdx.poisson` for float64 states.
        A multi-stage ``scheme`` gets one tuple of components per stage, each at its implicit weight.
        """
        conditions = tuple(velocity_condition(self.conditions, axis) for axis in range(3))
        stages = tuple(
            tuple(
                fast_diagonal_helmholtz(
                    self.grid,
                    velocity_offset(component),
                    conditions,
                    shift=1.0 + weight * float(self.dt) * self.damping_rates[component],
                    coefficient=weight * float(self.dt) * float(self.viscosity),
                    precision=self.precision,
                )
                for component in range(3)
            )
            for weight in scheme.implicit
        )
        return stages[0] if len(stages) == 1 else stages

    def factorization(self) -> FastDiagonalPoisson:
        """Factorize the scalar Laplacian shared by the pressure and the potential.

        Both see the same homogeneous conditions, so one factorization serves
        both. Build it once outside a traced function and pass it to
        :func:`step`; the assembly reads concrete arrays and cannot be traced.
        ``precision="mixed"`` solves float64 states in float32 with a float64
        correction; float32 states are solved in float32 either way.

        A thin conducting wall changes the potential's operator but not the
        pressure's; the potential then uses :meth:`potential_factorization`.
        """
        return fast_diagonal_poisson(self.grid, self.pressure_conditions, precision=self.precision)

    def potential_factorization(self) -> FastDiagonalPoisson:
        """Factorize the charge operator, with a sheet of potential unknowns on each conducting wall.

        :meth:`factorization` when no wall conducts; otherwise
        :func:`lmhdx.poisson.fast_diagonal_thin_wall_poisson`, built once per grid,
        conditions, conductances and precision, and reused under tracing.
        """
        if self.potential_walls is not None:
            return self.potential_walls
        if not self.conducting_walls:
            if self.open_axis is None:
                return self.factorization()
            return _neumann_factorization(self.grid, self.scalar_conditions, self.precision)
        return _thin_wall_factorization(
            self.grid,
            self.scalar_conditions,
            tuple(float(value) for value in self.wall_conductance),
            self.precision,
            int(self.open_axis is not None),
            self.wall_layers,
        )

    def with_walls(self, conductance: tuple, layers: tuple = (None, None, None)) -> ChannelProblem:
        """Return this problem with its potential factorization built, in the trace, from traced wall values.

        ``conductance`` is one thin-wall ratio per axis and ``layers`` the ``(lower, upper)`` per-cell
        ``(ratios, widths)`` of resolved walls, as :func:`lmhdx.poisson.fast_diagonal_thin_wall_poisson`
        takes them. Which ends conduct, and the cell counts, are the problem's own static pattern: an
        argument that crossed a ``jit`` boundary has lost the difference between a number and a tracer.
        """
        conductance = tuple(c if float(k) > 0.0 else 0.0 for c, k in zip(conductance, self.wall_conductance))
        layers = tuple(
            pair and tuple(end if kept else None for end, kept in zip(pair, static))
            for pair, static in zip(layers, self.wall_layers)
        )
        built = _build_thin_wall(
            self.grid,
            self.scalar_conditions,
            conductance,
            self.precision,
            int(self.open_axis is not None),
            layers,
        )
        return dataclasses.replace(self, potential_walls=built)


def face_currents(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    field_scale: float | jnp.ndarray = 1.0,
) -> tuple[Field, tuple[Field, Field, Field], tuple[Field, Field, Field]]:
    """Return the potential, the face-normal currents and the imposed field.

    One place forms the face currents, so the projection step, the steady solve
    and the energy budget cannot drift apart in how they close the wall or scale
    the potential. ``field_scale`` multiplies the imposed field and may be traced.
    """
    factorization = problem.factorization() if factorization is None else factorization
    scalar = problem.scalar_conditions
    field = tuple(
        component.replace_data(field_scale * component.data) for component in _imposed_field(problem)
    )
    conductivities = [
        face_conductivity(_constant(problem, _attribute_value, "conductivity"), axis, scalar[axis])
        for axis in range(3)
    ]
    emfs = [face_electromotive_force(velocity, field, axis, scalar) for axis in range(3)]
    motional = tuple(
        wall_insulated(
            conductivities[axis].replace_data(conductivities[axis].data * emfs[axis].data),
            axis,
            scalar[axis],
        )
        for axis in range(3)
    )
    potential, walls = _solve_potential(divergence(motional), problem, factorization)
    currents = tuple(
        _closed_current(potential, walls[axis], conductivities[axis], emfs[axis], axis, problem)
        for axis in range(3)
    )
    return potential, currents, field


def electric_state(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    field_scale: float | jnp.ndarray = 1.0,
) -> tuple[Field, tuple[Field, Field, Field]]:
    """Return the induced potential and the Lorentz force it carries.

    A thin wall's half-cell current carries no electromotive force, so it exerts no
    force: with it, conducting side walls in a uniform field left the Stokes operator
    asymmetric by 1.1e-3 (Ha 20, 24 cells); without, round-off, and the flow rate
    moves by at most 2.2e-4 of itself on 24 and 48 cells.
    """
    potential, currents, field = face_currents(velocity, problem, factorization, field_scale)
    return potential, _lorentz_from_currents(currents, field, problem)


def _lorentz_from_currents(
    currents: tuple[Field, Field, Field], field: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[Field, Field, Field]:
    """Return the cell-centred Lorentz force of the face currents :func:`face_currents` returns.

    The force leaves out a thin wall's half-cell current; a caller that needs the
    full currents, as the Joule power does, keeps them from :func:`face_currents`.
    """
    scalar = problem.scalar_conditions
    closed = tuple(wall_insulated(current, axis, scalar[axis]) for axis, current in enumerate(currents))
    return lorentz_force(closed, field, scalar)


def face_lorentz_force(
    force: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[Field, Field, Field]:
    """Carry the cell-centred Lorentz force onto the velocity faces.

    :func:`lmhdx.ops.face_average` is the transpose of the cell average that
    :func:`lmhdx.ops.face_electromotive_force` applies to the velocity, so the work
    this force does on any impermeable velocity is exactly minus the face
    current dotted with that velocity's electromotive force. The step, the
    steady residual and the energy budget all take the force from here.
    """
    return cell_force_to_faces(force, problem.scalar_conditions)


def _closed_current(
    potential: Field, wall: Field | None, conductivity: Field, emf: Field, axis: int, problem: ChannelProblem
) -> Field:
    """Ohm's law inside, and the half-cell current into a thin conducting wall on its wall faces."""
    scalar = problem.scalar_conditions
    ohmic = wall_insulated(face_current(potential, conductivity, emf, axis, scalar[axis]), axis, scalar[axis])
    if wall is None:
        return ohmic
    return ohmic.replace_data(ohmic.data + thin_wall_current(potential, wall, conductivity, axis).data)


def _solve_potential(
    source: Field, problem: ChannelProblem, factorization: FastDiagonalPoisson
) -> tuple[Field, tuple[Field | None, Field | None, Field | None]]:
    """Solve the charge equation for the potential and each conducting wall's sheet potential.

    A thin conducting wall is a sheet of unknowns joined to the fluid by the
    half-cell flux, and the operator stays a Kronecker sum, so either closure is
    three contractions: a fixed linear map that differentiates with no iteration.
    """
    scaled = source.replace_data(host_scalar(problem, _resistivity) * source.data)
    if not problem.conducting_walls:
        if problem.open_axis is not None:
            factorization = bound(problem, _potential_factorization)
        return factorization.solve(scaled), (None, None, None)
    return bound(problem, _potential_factorization).solve_with_walls(scaled)


def _resistivity(problem: ChannelProblem) -> float:
    return 1.0 / float(problem.conductivity) if float(problem.conductivity) else 0.0


def _factorization(problem: ChannelProblem) -> FastDiagonalPoisson:
    return problem.factorization()


def _potential_factorization(problem: ChannelProblem) -> FastDiagonalPoisson:
    return problem.potential_factorization()


def duct_problem(
    *,
    hartmann: float,
    cells: int = 48,
    wall_conductance: float = 0.0,
    forcing: float = 1.0,
    advection: str = "off",
    cells_in_layer: int = 6,
) -> ChannelProblem:
    """Build a square insulating or Hunt duct at a given Hartmann number.

    Non-dimensionalised the way the analytic solutions are: half width, density,
    kinematic viscosity and conductivity all one, the field along ``y`` with
    magnitude ``hartmann``, and ``forcing`` the axial pressure gradient. The two
    transverse meshes resolve the layers that actually exist -- ``a/Ha`` against
    the walls normal to the field and ``a/sqrt(Ha)`` against the others -- with
    the gentlest stretching that spans the duct, which is the difference between
    a converged flow rate and a fine mesh in the wrong place.

    The duct is solved in float64: JAX's float64 mode is turned on if it is
    off, since the steady solve at these tolerances does not converge in
    float32. A float32 channel is built with :class:`ChannelProblem` directly.
    """
    if hartmann < 0.0:
        raise ValueError("hartmann must not be negative")
    from . import enable_x64

    enable_x64()
    if hartmann:
        transverse = wall_resolving_faces(
            cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=cells_in_layer, max_ratio=None
        )
        spanwise = wall_resolving_faces(
            cells,
            -1.0,
            1.0,
            layer_thickness=1.0 / np.sqrt(hartmann),
            cells_in_layer=cells_in_layer,
            max_ratio=None,
        )
    else:
        transverse = spanwise = uniform_faces(cells, -1.0, 1.0)
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), transverse, spanwise),
        conditions=(BoundaryCondition(PERIODIC), _INSULATING, _INSULATING),
        conductivity=1.0 if hartmann else 0.0,
        magnetic_field=(0.0, float(hartmann), 0.0),
        forcing=(float(forcing), 0.0, 0.0),
        dt=1.0,
        advection=advection,
        wall_conductance=(0.0, float(wall_conductance), 0.0),
    )


def zero_velocity(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """Return a velocity at rest in the staggered layout."""
    return tuple(
        Field(
            jnp.zeros(problem.grid.offset_shape(velocity_offset(component)), dtype=jnp.result_type(float)),
            velocity_offset(component),
            problem.grid,
        )
        for component in range(3)
    )


def enforce_face_constraints(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[Field, Field, Field]:
    """Impose impermeability at walls, face agreement across a periodic axis, and a zero inlet.

    Both are prerequisites for the discrete divergence to represent a flux
    balance; see the module docstring.
    """
    constrained = []
    for component, field in enumerate(velocity):
        data = field.data
        selection = (slice(None),) * component
        if problem.conditions[component].is_periodic:
            data = data.at[selection + (-1,)].set(data[selection + (0,)])
        elif problem.conditions[component].is_mixed:
            data = data.at[selection + (0,)].set(0.0)
        else:
            data = data.at[selection + (0,)].set(0.0)
            data = data.at[selection + (-1,)].set(0.0)
        constrained.append(field.replace_data(data))
    return tuple(constrained)


def project(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
) -> tuple[tuple[Field, Field, Field], Field]:
    """Remove the divergence from a velocity field and return the pressure that did it."""
    factorization = problem.factorization() if factorization is None else factorization
    velocity = enforce_face_constraints(velocity, problem)
    source = divergence(velocity)
    density, dt = attribute(problem, "density"), attribute(problem, "dt")
    pressure = factorization.solve(source.replace_data((density / dt) * source.data))
    scalar = problem.pressure_conditions
    corrected = tuple(
        field.replace_data(
            field.data - (dt / density) * face_gradient(pressure, component, scalar[component]).data
        )
        for component, field in enumerate(velocity)
    )
    return enforce_face_constraints(corrected, problem), pressure


def courant_step(velocity: tuple[Field, Field, Field], scheme: Tableau = RK3, cfl: float = 0.9) -> float:
    """Return ``cfl`` times the largest step at which ``scheme`` keeps central transport of ``velocity`` stable."""
    if not scheme.limit:
        raise ValueError("central transport is unstable under this scheme at every step; use RK3")
    return cfl * scheme.limit * float(advective_step_limit(velocity))


def step(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    viscous: tuple | None = None,
    scheme: Tableau = EULER,
    force: Callable[[jnp.ndarray], tuple[Field, Field, Field]] | None = None,
    time: float = 0.0,
    pressure: Field | None = None,
) -> tuple[tuple[Field, Field, Field], Field, Field]:
    """Advance one step of ``scheme`` and return velocity, pressure and potential.

    Passing ``viscous`` takes diffusion implicitly (Crank-Nicolson in :data:`RK3`) and lifts the step
    off :attr:`ChannelProblem.diffusive_step_limit`; a multi-stage ``scheme`` takes the per-stage
    tuple of :meth:`ChannelProblem.viscous_factorizations`. ``force(t)`` is a further force density
    on the velocity faces, evaluated at the time of each stage's input state, starting from ``time``.
    ``pressure``, the last step's, starts the first stage's solve (each later stage starts from the one before).
    """
    return _step(velocity, problem, factorization, viscous, scheme, force, time, pressure)[:3]


def _step(velocity, problem, factorization, viscous, scheme, force, time, pressure=None):
    factorization = problem.factorization() if factorization is None else factorization
    stages = (viscous,) if len(scheme.explicit) == 1 else viscous
    carried, solves = None, []
    for k, weights in enumerate(zip(scheme.explicit, scheme.carried, scheme.implicit, strict=True)):
        potential, current = electric_state(velocity, problem, factorization)
        velocity, pressure, info, carried = _advance_momentum(
            velocity,
            face_lorentz_force(current, problem),
            problem,
            None if stages is None else stages[k],
            extra_force=None if force is None else force(time),
            weights=weights,
            carried=carried,
            pressure=pressure,
        )
        solves.append(info)
        time = time + (weights[0] + weights[1]) * problem.dt
    iterations, residual = (
        jnp.stack([getattr(i, name) for i in solves]) for name in ("iterations", "residual")
    )
    return velocity, pressure, potential, (jnp.sum(iterations), jnp.max(residual))


@functools.lru_cache(maxsize=16)
def _stage(grid, conditions, scalar, shifts, coefficient, viscous):
    """``H^-1`` per component, ``G``, the face constraints and a jitted pressure solve, built once per stage."""
    inverse = tuple(
        (lambda f, axis=axis: f / shifts[axis])
        if viscous is None
        else (lambda f, axis=axis: viscous[axis].solve(Field(f, velocity_offset(axis), grid)).data)
        for axis in range(3)
    )

    def gradient(pressure):
        return [
            face_gradient(Field(pressure, (CENTER,) * 3, grid), axis, scalar[axis]).data for axis in range(3)
        ]

    def constrained(arrays):
        faces = (Field(a, velocity_offset(axis), grid) for axis, a in enumerate(arrays))
        return enforce_face_constraints(tuple(faces), SimpleNamespace(conditions=conditions))

    with jax.ensure_compile_time_eval():
        schur = solvax.fast_diagonal_schur(grid.widths, [c.kind for c in scalar], shifts, coefficient)
    widths = grid.volume_factors()

    def solve(rhs, **options):
        # The cell volumes are formed in the trace from the three width vectors, not captured as a cell-sized constant.
        volumes = functools.reduce(jnp.multiply, (jnp.asarray(w, schur.volumes.dtype) for w in widths))
        # The right-hand side's mean is the wall flux, zero by construction: no relative compatibility test.
        return dataclasses.replace(schur, volumes=volumes).solve_exact(
            rhs,
            gradient=gradient,
            divergence=lambda fields: divergence(constrained(fields)).data,
            helmholtz=inverse,
            defect_rtol=1.0,
            **options,
        )

    return inverse, gradient, constrained, jax.jit(solve, static_argnames="rtol")


def _advance_momentum(
    velocity: tuple[Field, Field, Field],
    body: tuple[Field, Field, Field],
    problem: ChannelProblem,
    viscous: tuple[FastDiagonalHelmholtz, ...] | None = None,
    *,
    extra_force: tuple[Field, Field, Field] | None = None,
    weights: tuple[float, float, float] = (1.0, 0.0, 1.0),
    carried: tuple[jnp.ndarray, ...] | None = None,
    pressure: Field | None = None,
) -> tuple[tuple[Field, Field, Field], Field, solvax.SpdSolveInfo, tuple[jnp.ndarray, ...]]:
    """Solve one stage ``H u + c (dt/rho) G p = b``, ``D u = 0``; return velocity, pressure, status, ``R``.

    ``weights`` are the ``(explicit, carried, implicit)`` of the stage in :class:`Tableau`, ``c`` the sum
    of the first two, and ``carried`` the ``R`` the previous stage returned. ``R(u) = f + lambda u`` holds
    the face Lorentz force ``body``, the other forces and minus transport, with the damping returned to
    it, and the viscous term when that is explicit. ``b = u + dt (explicit R + carried R_prev + (c -
    implicit) L u)``, ``L = -lambda`` (plus the viscous term when implicit), and ``H = 1 + implicit dt
    lambda`` per component, minus ``implicit dt nu`` times the Laplacian when implicit. A pressure
    force meets the same ``H`` as every other force, so a gradient in balance with the forcing leaves
    the fluid at rest. With ``w = H^-1 b`` and ``q = c dt p / rho``, SOLVAX's
    :meth:`~solvax.FastDiagonalSchur.solve_exact` solves ``-D H^-1 G q = D w`` exactly by
    preconditioned conjugate gradients (one iteration on periodic axes, a few near walls) and ``u =
    w - H^-1 G q``. The status holds the iterations and relative Schur residual; a failure is NaN.

    ``extra_force`` is a further force density on the velocity faces (buoyancy, say), added to the
    forcing before the division by density. ``pressure``, the last stage's or step's, starts the solve:
    0-1 iterations in a steady flow against 13-15 from zero.
    """
    if any(condition.is_mixed for condition in problem.pressure_conditions):
        raise NotImplementedError("an inflow-outflow axis needs mixed end kinds in SOLVAX (issue 172)")
    explicit, previous, implicit = weights
    dt, grid, rates = float(problem.dt), problem.grid, problem.damping_rates
    shifts = tuple(1.0 + implicit * dt * rate for rate in rates)
    coefficient = implicit * dt * float(problem.viscosity) * (viscous is not None)
    inverse, gradient, constrained, solve = _stage(
        grid, problem.conditions, problem.pressure_conditions, shifts, coefficient, viscous
    )
    velocity_conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    transport = (
        None
        if problem.advection == "off"
        else momentum_advection(velocity, velocity_conditions, limited=problem.advection == "limited")
    )
    predicted, rate = [], []
    for axis, field in enumerate(velocity):
        force = body[axis].data + problem.forcing[axis]
        if extra_force is not None:
            force = force + extra_force[axis].data
        damping = rates[axis] * field.data
        viscous_term = problem.viscosity * staggered_laplacian(field, velocity_conditions).data
        now = force / problem.density - (0.0 if transport is None else transport[axis].data) + damping
        now = now + (viscous_term if viscous is None else 0.0)
        old = -damping + (0.0 if viscous is None else viscous_term)
        before = 0.0 if carried is None else previous * carried[axis]
        rate.append(now)
        change = explicit * now + before + (explicit + previous - implicit) * old
        predicted.append(inverse[axis](field.data + dt * change))
    predicted = constrained(predicted)
    source = -divergence(predicted).data
    # A divergence under the field's round-off is noise, and CG started on noise can break down.
    scale = sum(jnp.max(jnp.abs(w.data)) / np.min(grid.widths[a]) for a, w in enumerate(predicted))
    source = jnp.where(jnp.abs(source) > 16 * jnp.finfo(source.dtype).eps * scale, source, 0.0)
    stage = (explicit + previous) * dt
    # A right-hand side of exactly zero has the tolerance rtol |rhs| = 0, which no guess meets: drop it there.
    guess = (
        None
        if pressure is None
        else jnp.where(jnp.any(source != 0), stage * pressure.data / problem.density, 0.0)
    )
    flux, info = solve(source, x0=guess, rtol=1e-13 if velocity[0].dtype == jnp.float64 else None)
    pull = [inverse[axis](g) for axis, g in enumerate(gradient(flux))]
    corrected = constrained([w.data - g for w, g in zip(predicted, pull, strict=True)])
    pressure = Field(problem.density * flux / stage, (CENTER,) * 3, grid)
    return corrected, pressure, info, tuple(rate)


@functools.lru_cache(maxsize=16)
def _neumann_factorization(
    grid: Grid, conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition], precision: str
) -> FastDiagonalPoisson:
    """The potential's factorization where it differs from the pressure's: an inflow-outflow duct.

    It keeps one float64 defect correction, as the pressure's does (:mod:`lmhdx.poisson`).
    """
    with jax.ensure_compile_time_eval():
        return dataclasses.replace(
            fast_diagonal_poisson(grid, conditions, precision=precision), corrections=1
        )


def _build_thin_wall(
    grid, conditions, conductance, precision, corrections, layers
) -> FastDiagonalThinWallPoisson:
    """The thin-wall factorization; a periodic axis has no wall, so its conductance is ignored."""
    walls = tuple(
        0.0 if condition.is_periodic else value for condition, value in zip(conditions, conductance)
    )
    factorization = fast_diagonal_thin_wall_poisson(
        grid, conditions, walls, precision=precision, layers=layers
    )
    return dataclasses.replace(factorization, corrections=corrections)


@functools.lru_cache(maxsize=16)
def _thin_wall_factorization(grid, conditions, conductance, precision, corrections=0, layers=(None,) * 3):
    """Build the thin-wall factorization of concrete walls once, on the host."""
    with jax.ensure_compile_time_eval():
        return _build_thin_wall(grid, conditions, conductance, precision, corrections, layers)


def fringe_field(
    grid: Grid,
    *,
    half_length: float = 3.0,
    centre: float = 0.0,
    strength: float = 1.0,
    solenoidal: bool = True,
) -> ImposedField:
    """Return the ANL fringe (ANL/FPP/TM-228; plan Section 3.1 items 7-8), divergence free on the grid.

    With ``s = x - centre``, ``x0 = half_length``, ``B0 = strength`` and
    ``k = pi / (2 x0)``, the midplane field falls as ``B_y = B0 (1 - sin(ks)) / 2``
    over ``|s| <= x0`` and is uniform outside. ``solenoidal=False`` keeps that
    profile at every ``y``; ``True`` adds the companion Votyakov et al. (2009) ask
    for, ``B_y + i B_x = B0 (1 - sin k(s + iy)) / 2``, curl and divergence free in
    and outside the fringe. The profile is not analytic at ``s = +-x0``: there
    ``B_x = 0`` keeps the divergence continuous and ``B_y`` jumps by
    ``B0 (cosh(ky) - 1) / 2``, 7.0 % of ``B0`` at ``|y| = 1`` for ``x0 = 3``.
    Both come from ``A = -B0 [s/2 + cos(ks) cosh(ky) / (2k)]`` (``cosh`` replaced
    by one when not solenoidal), with ``B_x = dA/dy``, ``B_y = -dA/dx``: each face
    holds the mean of its normal component, a difference of ``A``, so
    :func:`lmhdx.ops.divergence` of the faces is round-off, and the cells average
    their two faces. ``y`` is measured from the magnet midplane.
    """
    if grid.is_polar:
        raise ValueError("the fringe field is built on a Cartesian grid")
    if half_length <= 0.0:
        raise ValueError("half_length must be positive")
    k = np.pi / (2.0 * half_length)
    s = np.asarray(grid.x_faces)[:, None] - centre
    inside = np.clip(s, -half_length, half_length)
    rise = np.cosh(k * np.asarray(grid.y_faces) * (1.0 if solenoidal else 0.0))[None, :]
    corners = -strength * (
        inside / 2.0 + np.cos(k * inside) * rise / (2.0 * k) + np.minimum(s + half_length, 0.0)
    )
    along_x = np.diff(corners, axis=1) / grid.widths[1][None, :]
    along_y = -np.diff(corners, axis=0) / grid.widths[0][:, None]
    faces = tuple(np.repeat(values[:, :, None], grid.shape[2], axis=2) for values in (along_x, along_y))
    components = (0.5 * (faces[0][:-1] + faces[0][1:]), 0.5 * (faces[1][:, :-1] + faces[1][:, 1:]))
    return ImposedField(grid, (*components, np.zeros(grid.shape)), (*faces, np.zeros(grid.face_shape(2))))


def _imposed_field(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """Return the imposed field at the cell centres: three constants, or the arrays of a varying one."""
    field = problem.magnetic_field
    if not isinstance(field, ImposedField):
        return tuple(_constant(problem, _field_component, component) for component in range(3))
    dtype = jnp.result_type(float)
    return tuple(
        Field(host_array(problem, _imposed_component, component, dtype=dtype), (CENTER,) * 3, problem.grid)
        for component in range(3)
    )


def _field_component(problem: ChannelProblem, component: int) -> float:
    return problem.magnetic_field[component]


def _imposed_component(problem: ChannelProblem, component: int) -> np.ndarray:
    return problem.magnetic_field.components[component]


def _attribute_value(problem: ChannelProblem, name: str) -> float:
    return getattr(problem, name)


def _one_element(problem: ChannelProblem, build, static: tuple) -> np.ndarray:
    return np.full(1, build(problem, *static), dtype=jnp.result_type(float))


def _constant(problem: ChannelProblem, build, *static) -> Field:
    """A broadcast scalar under tracing, which XLA fuses into its consumers, never a captured array.

    The scalar, ``build(problem, *static)``, is read from a one-element constant rather than
    written as a literal, so programs that differ only in the value share one executable (2b.1).
    """
    dtype = jnp.result_type(float)
    scalar = host_array(problem, _one_element, build, static, dtype=dtype)[0]
    return Field(jnp.broadcast_to(scalar, problem.grid.shape), (CENTER,) * 3, problem.grid)


# Run the projection step as one compiled trajectory (formerly ``lmhdx.core3d``).


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    """The end state of a run and the per-step history of its diagnostics."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    divergence_residual: jnp.ndarray
    kinetic_energy: jnp.ndarray
    stage_iterations: jnp.ndarray | None = None
    stage_residual: jnp.ndarray | None = None

    @property
    def steps(self) -> int:
        return int(self.divergence_residual.shape[0])


def kinetic_energy(velocity: tuple[Field, Field, Field], problem: ChannelProblem) -> jnp.ndarray:
    """Return the kinetic energy of a staggered velocity.

    Each component is weighted by the control volume of its own faces, the same
    measure under which the divergence and the gradient are exact adjoints. That
    is what makes the energy budget close: a cell-centred average would leave a
    defect of the order of the interpolation error rather than of round-off.
    """
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    return (
        0.5
        * float(problem.density)
        * sum(face_inner_product(field, field, axis, conditions[axis]) for axis, field in enumerate(velocity))
    )


def trajectory_diagnostics(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the divergence residual and kinetic energy of one state."""
    return jnp.max(jnp.abs(divergence(velocity).data)), kinetic_energy(velocity, problem)


@dataclass(frozen=True)
class EnergyBudget:
    """The mechanical power crossing one control volume: the whole duct.

    The pressure does no work on a discretely divergence-free velocity, because
    the divergence and the gradient are exact adjoints under this measure, so
    the balance is between what the drive and the field put in and what
    viscosity takes out. Two independent statements come out of it, and they are
    worth keeping apart. ``defect`` is what the steady solve promises: the
    residual, projected onto the velocity, is zero. ``ohmic_defect`` is a
    property of the discretization instead -- continuously the Lorentz force
    does exactly minus the Joule dissipation, and how nearly the discrete face
    currents reproduce that is a measurement, not a guarantee.
    """

    drive: jnp.ndarray
    lorentz: jnp.ndarray
    viscous: jnp.ndarray
    joule: jnp.ndarray
    wall: jnp.ndarray

    @property
    def defect(self) -> jnp.ndarray:
        """Net power into the fluid, which is the rate of change of kinetic energy."""
        return self.drive + self.lorentz - self.viscous

    @property
    def ohmic_defect(self) -> jnp.ndarray:
        """How far the discrete Lorentz work is from the dissipation it should equal.

        A conducting wall takes current out of the fluid and dissipates it in the
        sheet, so the fluid's own Joule term is not the whole of it; ``wall`` is
        the electrical power crossing the boundary, and leaving it out of the
        comparison is a 26 % error at a wall conductance of 0.027.
        """
        return self.joule + self.wall + self.lorentz

    @property
    def scale(self) -> jnp.ndarray:
        """The largest term, for turning either defect into a relative number."""
        return jnp.max(jnp.abs(jnp.stack([self.drive, self.lorentz, self.viscous, self.joule, self.wall])))


def energy_budget(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
) -> EnergyBudget:
    """Return the mechanical power balance of one state."""
    factorization = problem.factorization() if factorization is None else factorization
    scalar = problem.scalar_conditions
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    potential, currents, field = face_currents(velocity, problem, factorization)
    force = _lorentz_from_currents(currents, field, problem)
    density = float(problem.density)
    drive = sum(
        float(problem.forcing[axis])
        * face_inner_product(
            component, component.replace_data(jnp.ones_like(component.data)), axis, conditions[axis]
        )
        for axis, component in enumerate(velocity)
    )
    body = face_lorentz_force(force, problem)
    lorentz = sum(
        face_inner_product(component, body[axis], axis, conditions[axis])
        for axis, component in enumerate(velocity)
    )
    viscous = (
        -density
        * float(problem.viscosity)
        * sum(
            face_inner_product(component, staggered_laplacian(component, conditions), axis, conditions[axis])
            for axis, component in enumerate(velocity)
        )
    )
    conductivity = float(problem.conductivity)
    half_cells, wall = _wall_power(potential, currents, problem)
    joule = (
        (
            sum(
                face_inner_product(current, current, axis, scalar[axis])
                for axis, current in enumerate(currents)
            )
            - half_cells
        )
        / conductivity
        if conductivity
        else jnp.zeros(())
    )
    return EnergyBudget(
        jnp.asarray(drive),
        jnp.asarray(lorentz),
        jnp.asarray(viscous),
        jnp.asarray(joule),
        jnp.asarray(wall),
    )


def _wall_power(potential: Field, currents: tuple[Field, Field, Field], problem: ChannelProblem):
    """Return the excess Joule weight on conducting wall faces and the power the sheets dissipate.

    A current into a thin wall crosses only the half cell against it, so its
    Joule term is weighted by ``h/2`` where :func:`lmhdx.ops.face_inner_product`
    weights the whole cell; the sheets take the outward current at their own
    potential, ``phi_w = phi_P - J h / (2 sigma)``. With both, the Lorentz work
    is exactly minus the dissipation. Outward is negative on the lower wall.
    """
    grid = problem.grid
    scalar = problem.scalar_conditions
    conductivity = float(problem.conductivity)
    half_cells = jnp.zeros((), dtype=potential.dtype)
    total = jnp.zeros((), dtype=potential.dtype)
    if not conductivity:
        return half_cells, total
    for axis in range(3):
        if scalar[axis].is_periodic or not (
            float(problem.wall_conductance[axis]) or problem.wall_layers[axis]
        ):
            continue
        area = jnp.asarray(grid.face_areas(axis), dtype=potential.dtype)
        widths = grid.widths[axis]
        for position, sign in ((0, -1.0), (-1, 1.0)):
            selection = (slice(None),) * axis + (position,)
            half = 0.5 * float(widths[position])
            outward = sign * currents[axis].data[selection]
            sheet = potential.data[selection] - outward * half / conductivity
            half_cells = half_cells + jnp.sum(outward**2 * area[selection]) * half
            total = total + jnp.sum(sheet * outward * area[selection])
    return half_cells, total


def advance(
    problem: ChannelProblem,
    steps: int,
    velocity: tuple[Field, Field, Field] | None = None,
    *,
    factorization: FastDiagonalPoisson | None = None,
    viscous: tuple | None = None,
    scheme: Tableau = EULER,
    force: Callable[[jnp.ndarray], tuple[Field, Field, Field]] | None = None,
    time: float = 0.0,
    checkpoint: bool = True,
) -> Trajectory:
    """Run ``steps`` steps of ``scheme`` as one compiled scan, from ``time``.

    ``steps`` is static: it fixes the length of the compiled trajectory. The
    factorizations are built once on the host, outside the trace, because their
    assembly reads concrete arrays; passing ``viscous`` takes diffusion
    implicitly, which is what lets a long run choose its step for accuracy.
    ``viscous`` and ``force(t)`` are those of :func:`step`.

    The compiled run is kept per problem, ``steps``, factorizations, ``scheme`` and
    ``checkpoint``: a restart in chunks compiles once, and only the state and the time are
    arguments. A problem with a traced parameter, or a ``force``, is traced into the caller's
    program instead (a force may close over the caller's tracers, which a kept program must not).
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    velocity = zero_velocity(problem) if velocity is None else velocity
    try:
        hash(problem)
    except TypeError:
        return _trajectory(problem, steps, factorization, viscous, scheme, force, checkpoint)(velocity, time)
    if force is not None:
        return _trajectory(problem, steps, factorization, viscous, scheme, force, checkpoint)(velocity, time)
    return _compiled_trajectory(problem, steps, factorization, viscous, scheme, checkpoint)(velocity, time)


def _trajectory(problem, steps, factorization, viscous, scheme, force, checkpoint):
    """The run of ``steps`` steps as a function of the starting velocity and time."""
    factorization = problem.factorization() if factorization is None else factorization

    def single(carry, _):
        *state, now = carry
        updated, pressure, potential, solves = _step(
            state[0], problem, factorization, viscous, scheme, force, now, state[1]
        )
        history = (*trajectory_diagnostics(updated, problem), *solves)
        return (updated, pressure, potential, now + problem.dt), history

    def run(velocity, time):
        # The last pressure and potential ride in the carry: stacked as scan outputs they cost two
        # cell fields per step (3.1 GiB over 100 steps at 128^3 in float64) to return the last one.
        cell = Field(jnp.zeros(problem.grid.shape, velocity[0].dtype), (CENTER,) * 3, problem.grid)
        start = (velocity, cell, cell, jnp.asarray(time, velocity[0].dtype))
        body = jax.checkpoint(single) if checkpoint else single
        (final, pressure, potential, _), history = jax.lax.scan(body, start, xs=None, length=steps)
        return Trajectory(final, pressure, potential, *history)

    return run


@functools.lru_cache(maxsize=8)
def _compiled_trajectory(problem, steps, factorization, viscous, scheme, checkpoint):
    return jax.jit(_trajectory(problem, steps, factorization, viscous, scheme, None, checkpoint))
