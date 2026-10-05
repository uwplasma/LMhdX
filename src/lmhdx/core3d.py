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
conservative face force, so the update reads
``u* = u + dt * rhs / (1 + dt * lambda)``. The correction vanishes when the
right-hand side does, so it changes the path to a steady state but not the steady
state itself, and the conservative force keeps its own discretization.

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
   =(1+\\Delta t\\,\\lambda)\\mathbf u+\\Delta t\\,(\\mathbf F+\\mathbf f)/\\rho,

backward Euler on the viscous term with the damping correction folded into the
shift. That operator separates exactly as the pressure Laplacian does, so it
costs three contractions and a divide, and the step is then bounded by accuracy
rather than by the mesh. Both stiffnesses are gone at that point: the magnetic
one because it grows as :math:`Ha^2`, the viscous one because it grows as the
mesh is refined.

*Convective transport.* ``advection`` selects it: ``"off"`` is the Stokes limit,
appropriate at the large interaction parameters of a blanket channel and the
default so that no run acquires a convective step limit by accident;
``"central"`` is the conservative flux form of :mod:`lmhdx.ops`, second order on
a stretched mesh; ``"limited"`` adds the van Leer blend that keeps the thin side
layers bounded. Transport is explicit, so switching it on bounds the step by
:func:`lmhdx.ops.advective_step_limit`.

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
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

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
    divergence,
    face_average,
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
    _wall_cells,
    fast_diagonal_helmholtz,
    fast_diagonal_poisson,
    fast_diagonal_thin_wall_poisson,
)

__all__ = [
    "ChannelProblem",
    "EnergyBudget",
    "ImposedField",
    "Trajectory",
    "advance",
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
class ChannelProblem:
    """A duct segment with uniform material properties and an imposed field.

    ``magnetic_field`` is three numbers for a uniform field, or a varying one: an
    :class:`ImposedField`, or three arrays of ``grid.shape`` at the cell centres
    (scalars broadcast), which become one.

    One axis may be an inflow-outflow axis (plan D26): a condition of kind
    ``dirichlet`` with ``upper_kind="neumann"``, whose ``lower`` is the inlet
    profile of the velocity normal to it. :mod:`lmhdx.axial` builds and solves
    such a duct, in the Stokes limit only.

    ``wall_conductance`` closes an axis with thin conducting walls, the same on
    both. ``wall_layers`` resolves walls in cells instead: per axis
    ``(lower, upper)``, each ``None`` (insulating) or ``(ratios, widths)``, each
    cell's conductivity over the fluid's (one number for all) and its width, from
    the fluid outwards, insulated outside; stored per cell. A wall carries
    potential and current only and spans the fluid's tangential extent. Two axes
    may have them when the third has one cell; a corner cell then takes the
    nearer wall's layer (:func:`lmhdx.poisson.fast_diagonal_thin_wall_poisson`).
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

    def viscous_factorizations(self) -> tuple[FastDiagonalHelmholtz, ...]:
        """Factorize the implicit viscous operator for each velocity component.

        Build these once on the host and pass them to :func:`step` to take
        diffusion implicitly. The shift carries the magnetic damping, so one
        solve removes both stiff terms. :attr:`precision` selects the float32
        solve with float64 correction of :mod:`lmhdx.poisson` for float64 states.
        """
        conditions = tuple(velocity_condition(self.conditions, axis) for axis in range(3))
        return tuple(
            fast_diagonal_helmholtz(
                self.grid,
                velocity_offset(component),
                conditions,
                shift=1.0 + float(self.dt) * self.damping_rates[component],
                coefficient=float(self.dt) * float(self.viscosity),
                precision=self.precision,
            )
            for component in range(3)
        )

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
    scalar = problem.scalar_conditions
    closed = tuple(wall_insulated(current, axis, scalar[axis]) for axis, current in enumerate(currents))
    return potential, lorentz_force(closed, field, scalar)


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
    scalar = problem.scalar_conditions
    return tuple(face_average(component, axis, scalar[axis]) for axis, component in enumerate(force))


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


def step(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    viscous: tuple[FastDiagonalHelmholtz, ...] | None = None,
) -> tuple[tuple[Field, Field, Field], Field, Field]:
    """Advance one projection step and return velocity, pressure and potential.

    Passing ``viscous`` takes diffusion implicitly and lifts the step off
    :attr:`ChannelProblem.diffusive_step_limit`.
    """
    factorization = problem.factorization() if factorization is None else factorization
    potential, force = electric_state(velocity, problem, factorization)
    body = face_lorentz_force(force, problem)

    velocity_conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    transport = (
        None
        if problem.advection == "off"
        else momentum_advection(velocity, velocity_conditions, limited=problem.advection == "limited")
    )
    predicted = []
    for component, component_field in enumerate(velocity):
        drive = (body[component].data + problem.forcing[component]) / problem.density
        if transport is not None:
            drive = drive - transport[component].data
        rate = problem.damping_rates[component]
        shift = 1.0 + problem.dt * rate
        if viscous is None:
            diffusion = staggered_laplacian(component_field, velocity_conditions)
            updated = component_field.data + problem.dt * (problem.viscosity * diffusion.data + drive) / shift
        else:
            source = component_field.replace_data(shift * component_field.data + problem.dt * drive)
            updated = viscous[component].solve(source).data
        predicted.append(component_field.replace_data(updated))

    corrected, pressure = project(tuple(predicted), problem, factorization)
    return corrected, pressure, potential


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


@functools.lru_cache(maxsize=16)
def _thin_wall_factorization(
    grid: Grid,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
    conductance: tuple[float, float, float],
    precision: str,
    corrections: int = 0,
    layers: tuple = (None, None, None),
) -> FastDiagonalThinWallPoisson:
    """Build the thin-wall factorization once; a periodic axis has no wall, so its conductance is ignored."""
    walls = tuple(
        0.0 if condition.is_periodic else value for condition, value in zip(conditions, conductance)
    )
    with jax.ensure_compile_time_eval():
        factorization = fast_diagonal_thin_wall_poisson(
            grid, conditions, walls, precision=precision, layers=layers
        )
    return dataclasses.replace(factorization, corrections=corrections)


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


@dataclass(frozen=True)
class Trajectory:
    """The end state of a run and the per-step history of its diagnostics."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    divergence_residual: jnp.ndarray
    kinetic_energy: jnp.ndarray

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
    force = lorentz_force(
        tuple(wall_insulated(c, axis, scalar[axis]) for axis, c in enumerate(currents)), field, scalar
    )
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
    viscous: tuple[FastDiagonalHelmholtz, ...] | None = None,
    checkpoint: bool = True,
) -> Trajectory:
    """Run ``steps`` projection steps as one compiled scan.

    ``steps`` is static: it fixes the length of the compiled trajectory. The
    factorizations are built once on the host, outside the trace, because their
    assembly reads concrete arrays; passing ``viscous`` takes diffusion
    implicitly, which is what lets a long run choose its step for accuracy.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    factorization = problem.factorization() if factorization is None else factorization
    velocity = zero_velocity(problem) if velocity is None else velocity

    def single(carry, _):
        updated, pressure, potential = step(carry[0], problem, factorization, viscous)
        return (updated, pressure, potential), trajectory_diagnostics(updated, problem)

    # The last pressure and potential ride in the carry: stacked as scan outputs they cost two
    # cell fields per step (3.1 GiB over 100 steps at 128^3 in float64) to return the last one.
    cell = Field(jnp.zeros(problem.grid.shape, velocity[0].dtype), (CENTER,) * 3, problem.grid)
    start = (velocity, cell, cell)
    body = jax.checkpoint(single) if checkpoint else single
    (final, pressure, potential), (residual, energy) = jax.lax.scan(body, start, xs=None, length=steps)
    return Trajectory(final, pressure, potential, residual, energy)
