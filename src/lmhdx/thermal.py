"""Heat transport in the duct fluid: the temperature equation on the Cartesian core (ADR 0008).

The equation is

.. math:: \\rho_0 c_p\\left[\\partial_t T + \\nabla\\cdot(\\mathbf u T)\\right]
   = \\nabla\\cdot(k\\nabla T) + |\\mathbf J|^2/\\sigma + q,

with constant properties, the temperature at the cell centres of the staggered
grid, and its own boundary conditions. The fluid is the only medium: a wall
enters through a prescribed temperature or heat flux, not through its own
conduction, so electrical power entering a conducting wall is not fluid heat.
Viscous heating is omitted.

*Boundary conditions.* An axis is periodic, or each end prescribes a temperature
or an outward conductive heat flux, independently of the electrical and
mechanical conditions: an electrically insulating wall need not be adiabatic. A
flux enters :class:`lmhdx.grid.BoundaryCondition` as the derivative along the
axis that :func:`lmhdx.grid.pad` applies, ``q/k`` at the lower end and ``-q/k``
at the upper, so a positive outward flux cools through both;
:func:`thermal_boundary` makes that conversion.

*Conduction.* Implicit, through one cell-centred factorization of
``I - dt alpha L_0``, ``alpha = k/(rho_0 c_p)``, under the homogeneous
conditions. The boundary data enter as the fixed lift ``b = L(0)``, since
:func:`lmhdx.grid.pad` is affine and ``L(T) = L_0(T) + b``. The unit shift keeps
the operator nonsingular under insulated and periodic walls, so the solve keeps
the mean: stored heat is never removed as a gauge.

*Transport.* Conservative and first-order upwind: each velocity face carries
``u_f T_up``, the temperature of the cell the flow comes from, and the transport
term is minus the divergence of those fluxes. A wall face carries none, whatever
its thermal condition; a periodic axis reads its wrap face from the last and the
first cell, so a one-cell axis carries the same flux in and out. Forward in time
on a discretely divergence-free carrier, the update makes no new extremum while
``dt`` times every cell's outgoing volume-flux rate stays below one; 0.8 is the
documented envelope. Upwinding adds a numerical diffusivity of about ``|u| h/2``,
which is measured against ``alpha`` before a heat-transfer number is quoted.

*Buoyancy.* With expansion coefficient ``beta`` and gravity ``g`` the force
density ``-rho_0 beta (T - T_r) g`` joins the Lorentz force before the division
by density and the projection (ADR 0008, D32). It reaches the velocity faces by
the same average as the Lorentz force and is zeroed on wall faces. Without
``beta`` or ``g`` there is no force term at all, so the flow is the isothermal
one. The fractional step does not hold every hydrostatic state exactly; the
claim is convergence in the time step.

*The developed duct.* Along a duct periodic in its flow direction, a uniform
axial heat input makes ``T = gamma x + theta``, and :func:`developed_heat`
solves ``theta`` once on the cross-section (ADR 0008, D33): the bulk, mean and
peak wall temperatures and the Nusselt number of a station, at fixed flow and
heat load, with derivatives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .core3d import (
    ChannelProblem,
    ImposedField,
    _advance_momentum,
    _lorentz_from_currents,
    enforce_face_constraints,
    face_currents,
    face_lorentz_force,
    velocity_offset,
    zero_velocity,
)
from .grid import (
    CENTER,
    DIRICHLET,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    pad,
)
from .ops import (
    _broadcast,
    _take,
    cell_average,
    cell_force_to_faces,
    cell_inner_product,
    divergence,
    face_average_adjoint,
    face_gradient,
    laplacian,
    neighbours,
    outgoing_rate,
    padded_gradient,
    wall_insulated,
)
from .poisson import FastDiagonalHelmholtz, fast_diagonal_helmholtz, fast_diagonal_poisson
from .steady import with_inflow

__all__ = [
    "DevelopedHeat",
    "LoadAttribution",
    "ThermalDiagnostics",
    "ThermalProblem",
    "ThermalRun",
    "ThermalState",
    "advance_thermal",
    "build_thermal_step",
    "developed_heat",
    "initial_thermal_state",
    "load_attribution",
    "load_thermal_record",
    "thermal_boundary",
    "validate_developed_inputs",
    "validate_thermal_inputs",
    "write_thermal_record",
]

_KINDS = {"temperature": DIRICHLET, "heat_flux": NEUMANN, "periodic": PERIODIC}
# Each transport's default Courant limit: inside upwinding's bound of one for no new extremum, and van Leer's a
# tested envelope, not a derived bound; conduction alone has none.
_SCHEMES = {"upwind": 0.8, "van_leer": 0.5, "off": None}
_CELL = (CENTER, CENTER, CENTER)


def thermal_boundary(
    kind: str,
    lower: float | np.ndarray = 0.0,
    upper: float | np.ndarray = 0.0,
    *,
    upper_kind: str | None = None,
    conductivity: float | None = None,
) -> BoundaryCondition:
    """Return the temperature condition of one axis from physical data.

    ``kind`` names both ends, or the lower end only when ``upper_kind`` differs:
    ``"temperature"`` prescribes the wall temperature, ``"heat_flux"`` the
    conductive heat flux leaving the fluid through that end (positive cools),
    and ``"periodic"`` wraps the axis. A flux is stored as the derivative along
    the axis, ``q/k`` below and ``-q/k`` above, so ``conductivity``, the fluid's
    ``k``, is required with any flux end, and a condition built for one ``k``
    does not hold the same flux under another. Either value may be an array over
    the two other axes, in their order; it is copied.
    """
    kinds = (kind, kind if upper_kind is None else upper_kind)
    for name in kinds:
        if name not in _KINDS:
            raise ValueError(f"unknown thermal boundary kind {name!r}; expected one of {sorted(_KINDS)}")
    if "heat_flux" in kinds and not (
        conductivity is not None and np.isfinite(conductivity) and float(conductivity) > 0.0
    ):
        raise ValueError("a heat-flux end needs the fluid's thermal conductivity, positive and finite")
    values = []
    for end, data in enumerate((lower, upper)):
        value = np.array(data, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("thermal boundary data must be finite")
        if kinds[end] == "heat_flux":
            value = (1.0 if end == 0 else -1.0) * value / float(conductivity)
        values.append(value if value.ndim else float(value))
    mixed = kinds[0] != kinds[1]
    return BoundaryCondition(_KINDS[kinds[0]], *values, upper_kind=_KINDS[kinds[1]] if mixed else None)


@dataclass(frozen=True)
class ThermalProblem:
    """A flow problem with the fluid's heat capacity, conductivity and temperature conditions.

    ``flow`` is unchanged: its grid, time step, mechanics and electrics carry
    over, and ``flow.density`` is the ``rho_0`` of the heat capacity.
    ``thermal_conditions`` holds one :class:`lmhdx.grid.BoundaryCondition` per
    axis, best built with :func:`thermal_boundary`; an axis is periodic exactly
    where the flow's is. The flow must be on a Cartesian grid. Along an
    inflow-outflow axis the condition is the inlet temperature, a number or a
    profile, with no conductive flux at the outlet:
    ``thermal_boundary("temperature", T_in, upper_kind="heat_flux", conductivity=k)``.
    ``joule_heating`` deposits the fluid's Joule heat.

    ``reference_temperature`` is the ``T_r`` of the Boussinesq density
    ``rho_0 [1 - beta (T - T_r)]`` and of the reported sensible heat.
    ``expansion_coefficient`` is ``beta``, zero for passive transport, and
    ``gravity`` the gravitational acceleration, pointing down, along the grid's
    three axes; while ``beta > 0`` it has no component along a periodic axis,
    where no periodic pressure balances its mean (ADR 0008, D32). ``advection`` is ``"upwind"``,
    first order; ``"van_leer"``, a van Leer-limited second-order face value, bounded by the two cells it lies
    between; or ``"off"`` for conduction alone, whatever the flow's own momentum transport.

    ``outlet_backflow`` says what enters where the flow returns through the
    outlet of an inflow-outflow axis: ``"error"`` refuses such a run, and
    ``"reservoir_flux"`` brings in the reservoir temperature each step is given
    as ``backflow_temperature``, a prescribed total heat flux ``C u T_res``.
    """

    flow: ChannelProblem
    thermal_conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    specific_heat: float
    thermal_conductivity: float
    reference_temperature: float
    joule_heating: bool = True
    expansion_coefficient: float = 0.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    advection: str = "upwind"
    outlet_backflow: str = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.flow, ChannelProblem):
            raise TypeError("flow must be a lmhdx.core3d.ChannelProblem")
        if self.flow.grid.is_polar:
            raise ValueError("heat transport needs a Cartesian flow")
        for name in ("specific_heat", "thermal_conductivity"):
            if not (np.isfinite(getattr(self, name)) and float(getattr(self, name)) > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("density", "dt"):
            if not np.isfinite(getattr(self.flow, name)):
                raise ValueError(f"the flow's {name} must be finite")
        if not np.isfinite(self.reference_temperature):
            raise ValueError("reference_temperature must be finite")
        if not (np.isfinite(self.expansion_coefficient) and float(self.expansion_coefficient) >= 0.0):
            raise ValueError("expansion_coefficient must be nonnegative and finite")
        if len(self.gravity) != 3 or not np.all(np.isfinite(np.asarray(self.gravity, dtype=float))):
            raise ValueError("gravity must be three finite components")
        object.__setattr__(self, "gravity", tuple(float(component) for component in self.gravity))
        if self.advection not in _SCHEMES:
            raise ValueError(f"advection must be one of {sorted(_SCHEMES)}, got {self.advection!r}")
        if self.outlet_backflow not in ("error", "reservoir_flux"):
            raise ValueError(
                f"outlet_backflow must be 'error' or 'reservoir_flux', got {self.outlet_backflow!r}"
            )
        if self.outlet_backflow == "reservoir_flux" and (
            self.flow.open_axis is None or self.advection == "off"
        ):
            raise ValueError("outlet_backflow='reservoir_flux' needs an inflow-outflow axis and transport")
        if not isinstance(self.joule_heating, bool):
            raise TypeError("joule_heating must be True or False")
        if len(self.thermal_conditions) != 3:
            raise ValueError("heat transport needs one boundary condition per axis")
        shape = self.flow.grid.shape
        for axis, (condition, flow) in enumerate(zip(self.thermal_conditions, self.flow.conditions)):
            if not isinstance(condition, BoundaryCondition):
                raise TypeError(f"axis {axis} needs a lmhdx.grid.BoundaryCondition")
            if condition.is_periodic != flow.is_periodic:
                raise ValueError(f"axis {axis} must be periodic for heat exactly where it is for the flow")
            if flow.is_mixed and (condition.kinds != (DIRICHLET, NEUMANN) or np.any(condition.upper)):
                raise ValueError(
                    f"axis {axis} takes an inlet temperature and no conductive flux at the outlet"
                )
            tangential = shape[:axis] + shape[axis + 1 :]
            for value in (condition.lower, condition.upper):
                if np.ndim(value) and np.shape(value) != tangential:
                    raise ValueError(f"axis {axis} boundary data must have shape {tangential}")
        if float(self.expansion_coefficient) and any(
            component and flow.is_periodic for component, flow in zip(self.gravity, self.flow.conditions)
        ):
            raise ValueError("gravity along a periodic axis is refused while expansion_coefficient > 0")

    @property
    def heat_capacity(self) -> float:
        """The volumetric heat capacity ``rho_0 c_p``."""
        return float(self.flow.density) * float(self.specific_heat)

    @property
    def diffusivity(self) -> float:
        """The thermal diffusivity ``k / (rho_0 c_p)``."""
        return float(self.thermal_conductivity) / self.heat_capacity

    def diffusion_factorization(self) -> FastDiagonalHelmholtz:
        """Factorize ``I - dt alpha L_0`` at the cell centres under the homogeneous conditions.

        Build it once on the host, as :meth:`lmhdx.core3d.ChannelProblem.viscous_factorizations`;
        the flow's ``precision`` selects the mixed-precision solve.
        """
        return fast_diagonal_helmholtz(
            self.flow.grid,
            _CELL,
            tuple(condition.homogeneous() for condition in self.thermal_conditions),
            shift=1.0,
            coefficient=float(self.flow.dt) * self.diffusivity,
            precision=self.flow.precision,
        )

    def boundary_lift(self) -> Field:
        """Return ``b = L(0)``, what the boundary data add to the Laplacian of any temperature."""
        grid = self.flow.grid
        with jax.ensure_compile_time_eval():
            zero = Field(jnp.zeros(grid.shape, dtype=jnp.result_type(float)), _CELL, grid)
            return laplacian(zero, self.thermal_conditions)


def _conduct(
    temperature: Field,
    heating: Field,
    problem: ThermalProblem,
    factorization: FastDiagonalHelmholtz,
    lift: Field,
) -> Field:
    """Advance conduction and volumetric heating by one implicit step.

    ``(I - dt alpha L_0) T' = T + dt (q / (rho_0 c_p) + alpha b)``, with ``q`` the
    heat per unit volume and time in ``heating``.
    """
    dt, alpha = float(problem.flow.dt), problem.diffusivity
    rhs = temperature.data + dt * (heating.data / problem.heat_capacity + alpha * lift.data)
    return factorization.solve(temperature.replace_data(rhs))


def _upwind_fluxes(
    temperature: Field,
    velocity: tuple[Field, Field, Field],
    problem: ThermalProblem,
    reservoir: jnp.ndarray | None = None,
) -> tuple[Field, Field, Field]:
    """Return ``u_f T_up`` on the velocity faces; the transport term is minus their divergence.

    ``T_up`` is the temperature of the cell the face velocity comes from, or with
    ``advection="van_leer"`` its limited face value (:func:`_limited`), from the
    one array padded by the thermal conditions; the ghost of a wall is never
    used, since a wall face carries no advective flux. The
    inlet face brings in the inlet temperature, or takes out the first cell's
    against a backflow. The outlet face takes out the last cell's; where the flow
    returns there it brings in ``reservoir``, the outlet's incoming temperature.
    Without one, only an unvalidated step under ``outlet_backflow="error"``
    reaches a returning face; it takes the last cell's temperature and reports the
    boundary invalid.
    """
    scalar, opening, grid = problem.flow.scalar_conditions, problem.flow.open_axis, temperature.grid
    fluxes = []
    for axis, carrier in enumerate(velocity):
        padded = pad(temperature.data, axis, problem.thermal_conditions[axis], grid=grid)
        lower, upper = neighbours(padded, axis)
        if problem.advection == "van_leer" and grid.shape[axis] > 1:
            lower, upper = _limited(padded, axis, problem, lower, upper)
        flux = carrier.data * jnp.where(carrier.data >= 0.0, lower, upper)
        if axis != opening:
            fluxes.append(wall_insulated(carrier.replace_data(flux), axis, scalar[axis]))
            continue
        ends = [(slice(None),) * axis + (end,) for end in (0, -1)]
        inlet = jnp.asarray(problem.thermal_conditions[axis].lower, dtype=flux.dtype)
        entering = carrier.data[ends[0]] * jnp.where(
            carrier.data[ends[0]] >= 0.0, inlet, temperature.data[ends[0]]
        )
        leaving = carrier.data[ends[1]]
        returning = temperature.data[ends[1]] if reservoir is None else reservoir
        flux = flux.at[ends[0]].set(entering)
        flux = flux.at[ends[1]].set(leaving * jnp.where(leaving >= 0.0, temperature.data[ends[1]], returning))
        fluxes.append(carrier.replace_data(flux))
    return tuple(fluxes)


def _limited(padded, axis: int, problem: ThermalProblem, left, right):
    """Return each face's van Leer value seen from the cell on its left and on its right, on physical distances.

    ``padded`` is the temperature padded by the axis's thermal condition. From a cell of width ``w`` the face
    value is ``T + (w/2) s`` toward the face, with ``s`` the van Leer mean ``(a |b| + |a| b) / (|a| + |b|)`` of
    the face gradients on either side of the cell, zero where they differ in sign; it is then held between
    the two cells the face lies between, which keeps it bounded on a stretched grid.
    """
    grid, condition = problem.flow.grid, problem.thermal_conditions[axis]
    slopes, count = padded_gradient(padded, grid, axis, condition), grid.shape[axis]
    edges = (count - 1, 1) if condition.is_periodic else (0, count)  # faces -1 and n + 1: the wrap, or a copy
    outer = [jax.lax.slice_in_dim(slopes, index, index + 1, axis=axis) for index in edges]
    before, at = neighbours(jnp.concatenate([outer[0], slopes], axis=axis), axis)
    after = jnp.concatenate([_take(slopes, axis, slice(1, None)), outer[1]], axis=axis)
    widths = np.asarray(grid.widths[axis])
    beside = (widths[-1], widths[0]) if condition.is_periodic else (widths[0], widths[-1])
    left_width, right_width = (
        _broadcast(values, axis, slopes.dtype)
        for values in (np.concatenate([[beside[0]], widths]), np.concatenate([widths, [beside[1]]]))
    )

    def van_leer(a, b):
        size = jnp.abs(a) + jnp.abs(b)
        return (a * jnp.abs(b) + jnp.abs(a) * b) / jnp.where(size > 0.0, size, 1.0)

    low, high = jnp.minimum(left, right), jnp.maximum(left, right)
    from_left = jnp.clip(left + 0.5 * left_width * van_leer(before, at), low, high)
    from_right = jnp.clip(right - 0.5 * right_width * van_leer(at, after), low, high)
    return from_left, from_right


def _joule_heating(currents: tuple[Field, Field, Field], problem: ThermalProblem) -> Field:
    """Return the fluid's Joule heat per unit volume, ``|J|^2 / sigma``, from the full face currents.

    Each face's ``J_f^2 / sigma`` is shared between the half cells on either side
    by :func:`lmhdx.ops.face_average_adjoint`, squared before it is shared so no
    sub-cell dissipation is lost. A wall face heats only the fluid's half cell;
    what the current dissipates beyond it, in a conducting wall, is the wall's.
    The volume integral is the ``joule`` of :func:`lmhdx.core3d.energy_budget`.
    Without conductivity, or with ``joule_heating`` off, nothing is deposited and
    nothing is divided.
    """
    grid, sigma = currents[0].grid, float(problem.flow.conductivity)
    if not (problem.joule_heating and sigma):
        return Field(jnp.zeros(grid.shape, dtype=currents[0].dtype), _CELL, grid)
    scalar = problem.flow.scalar_conditions
    squares = (
        face_average_adjoint(current.replace_data(current.data**2), axis, scalar[axis]).data
        for axis, current in enumerate(currents)
    )
    return Field(sum(squares) / sigma, _CELL, grid)


def _buoyancy(temperature: Field, problem: ThermalProblem) -> tuple[Field, Field, Field] | None:
    """Return the Boussinesq force density on the velocity faces, or ``None`` without buoyancy.

    The cell force ``-rho_0 beta (T - T_r) g`` is averaged onto each velocity
    face as :func:`lmhdx.core3d.face_lorentz_force` carries the Lorentz force,
    wrapping a periodic axis and copying at a wall, then zeroed on wall faces,
    where the velocity is not an unknown. ``None`` when ``beta`` or ``g`` is zero.
    """
    beta, gravity = float(problem.expansion_coefficient), problem.gravity
    if not (beta and any(gravity)):
        return None
    excess = -float(problem.flow.density) * beta * (temperature.data - float(problem.reference_temperature))
    force = tuple(temperature.replace_data(excess * g) for g in gravity)
    return cell_force_to_faces(force, problem.flow.scalar_conditions, zero_walls=True)


class ThermalState(NamedTuple):
    """What a thermal step advances: the staggered velocity, the cell temperature and the time.

    Its five leaves are arrays; grids and offsets are static.
    """

    velocity: tuple[Field, Field, Field]
    temperature: Field
    time: jnp.ndarray


class ThermalDiagnostics(NamedTuple):
    """The scalars of one step, in the problem's units.

    ``heat_balance_residual`` is ``storage_rate + conductive_heat_out +
    advective_heat_out - joule_power - external_power``, a power, where
    ``advective_heat_out`` is the enthalpy the flow carries out through an outlet
    less what it brings in through the inlet and through returning outlet faces
    (zero in a closed duct); ``heat_solve_residual`` is the
    largest local residual of the temperature equation, a temperature per time;
    ``advective_cfl`` is ``dt`` times the largest outgoing volume-flux rate of
    the old velocity; ``boussinesq_measure`` is the largest ``beta |T - T_r|`` of
    the old and new temperatures. ``finite`` says every value is a number, not
    that the step is accepted.

    At the outlet of an inflow-outflow axis, ``outlet_backflow_rate`` and
    ``outlet_outflow_rate`` are the volume rates returning and leaving, and
    ``outlet_backflow_enthalpy_in`` is ``C`` times the returning rate weighted by
    the reservoir temperature: what ``advective_heat_out`` counts in at the
    returning faces. ``backflow_boundary_valid`` is false where the flow returns
    under ``outlet_backflow="error"``, whose enthalpy in is then reported as zero.
    All four are zero, and the boundary valid, in a closed duct.

    ``buoyancy_courant`` is ``dt sqrt(beta |g| G)`` of the old temperature, the
    step over the buoyancy time scale, with ``G`` the largest absolute
    face-normal gradient on any face: one component, not a colocated gradient
    magnitude. It is zero without buoyancy and counts in ``finite``. The
    partitioned step is first order in it; in the Benard onset of #252 the
    critical Rayleigh number is about 15 % low near 0.2 and under 3 % near 0.025,
    a calibration of that case, not a general bound.
    """

    mass_residual: jnp.ndarray
    charge_residual: jnp.ndarray
    temperature_min: jnp.ndarray
    temperature_max: jnp.ndarray
    sensible_energy: jnp.ndarray
    joule_power: jnp.ndarray
    external_power: jnp.ndarray
    conductive_heat_out: jnp.ndarray
    advective_heat_out: jnp.ndarray
    storage_rate: jnp.ndarray
    heat_balance_residual: jnp.ndarray
    heat_solve_residual: jnp.ndarray
    advective_cfl: jnp.ndarray
    boussinesq_measure: jnp.ndarray
    finite: jnp.ndarray
    outlet_backflow_rate: jnp.ndarray
    outlet_outflow_rate: jnp.ndarray
    outlet_backflow_enthalpy_in: jnp.ndarray
    backflow_boundary_valid: jnp.ndarray
    buoyancy_courant: jnp.ndarray


def _buoyancy_courant(temperature: Field, problem: ThermalProblem) -> jnp.ndarray:
    """Return ``dt sqrt(beta |g| G)``, ``G`` the largest absolute face-normal gradient; no derivative flows."""
    strength = float(problem.expansion_coefficient) * float(np.linalg.norm(problem.gravity))
    if not strength:
        return jnp.zeros((), temperature.dtype)
    temperature = jax.lax.stop_gradient(temperature)
    steepest = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(face_gradient(temperature, axis, condition).data))
                for axis, condition in enumerate(problem.thermal_conditions)
            ]
        )
    )
    return float(problem.flow.dt) * jnp.sqrt(strength * steepest)


def initial_thermal_state(
    problem: ThermalProblem,
    temperature: float | np.ndarray,
    velocity: tuple[Field, Field, Field] | None = None,
    time: float = 0.0,
) -> ThermalState:
    """Return a state from an explicit temperature, a number or an array over the cells.

    The velocity is the flow at rest unless given; a given one must already meet
    the face constraints and be divergence free, as :func:`validate_thermal_inputs` checks.
    """
    grid, dtype = problem.flow.grid, jnp.result_type(float)
    data = jnp.broadcast_to(jnp.asarray(temperature, dtype=dtype), grid.shape)
    velocity = zero_velocity(problem.flow) if velocity is None else tuple(velocity)
    return ThermalState(velocity, Field(data, _CELL, grid), jnp.asarray(time, dtype=dtype))


def validate_thermal_inputs(
    problem: ThermalProblem,
    state: ThermalState,
    heat_source: Field,
    *,
    divergence_tolerance: float,
    backflow_temperature: float | np.ndarray | None = None,
    advective_cfl_limit: float | None = None,
    buoyancy_courant_limit: float | None = None,
) -> None:
    """Check a state and heat source on the host before a run, raising ``ValueError`` on the first fault.

    Grids, offsets and shapes; the core's compute dtype; finite values; the wall
    and periodic face constraints; a velocity divergence within
    ``divergence_tolerance``, in inverse time; with transport, a Courant number
    ``dt max(sum_f max(u_f . n_f, 0) A_f / V)`` within ``advective_cfl_limit``, by
    default the scheme's own, 0.8 for ``"upwind"``, inside its bound of one for no new
    extremum, and for ``"van_leer"`` 0.5, a tested envelope rather than a derived
    bound; and, at an outlet with transport, no returning face under
    ``outlet_backflow="error"``, or a finite, real ``backflow_temperature`` of the
    outlet's shape under ``"reservoir_flux"``; and, given ``buoyancy_courant_limit``,
    the start's ``buoyancy_courant`` (:class:`ThermalDiagnostics`) within it.
    Nothing is changed or returned.

    A held flow keeps its Courant number for the whole run. An evolving flow is
    checked at its start only, which a start from rest passes trivially; compare
    the ``advective_cfl`` each step of :func:`advance_thermal` reports with the
    limit, as :func:`lmhdx.io.run_thermal` does at every step.
    """
    if not (np.isfinite(divergence_tolerance) and divergence_tolerance >= 0.0):
        raise ValueError("divergence_tolerance must be nonnegative and finite")
    if buoyancy_courant_limit is not None and not (
        np.isfinite(buoyancy_courant_limit) and buoyancy_courant_limit > 0.0
    ):
        raise ValueError("buoyancy_courant_limit must be positive and finite")
    if advective_cfl_limit is not None and not (
        np.isfinite(advective_cfl_limit) and advective_cfl_limit > 0.0
    ):
        raise ValueError("advective_cfl_limit must be positive and finite")
    advective_cfl_limit = _SCHEMES[problem.advection] if advective_cfl_limit is None else advective_cfl_limit
    flow, dtype = problem.flow, jnp.result_type(float)
    _check_fields(flow, state.velocity, [(state.temperature, "temperature"), (heat_source, "heat source")])
    time = jnp.asarray(state.time)
    if time.shape or time.dtype != dtype or not np.isfinite(float(time)):
        raise ValueError(f"the time must be a finite {dtype} scalar")
    constrained = enforce_face_constraints(state.velocity, flow)
    if flow.open_axis is not None:
        constrained = with_inflow(constrained, flow)
    if not all(np.array_equal(a.data, b.data) for a, b in zip(constrained, state.velocity)):
        raise ValueError("the velocity does not meet the wall and periodic face constraints")
    if float(jnp.max(jnp.abs(divergence(state.velocity).data))) > divergence_tolerance:
        raise ValueError("the velocity divergence exceeds divergence_tolerance")
    if buoyancy_courant_limit is not None:
        buoyant = float(_buoyancy_courant(state.temperature, problem))
        if buoyant > buoyancy_courant_limit:
            raise ValueError(
                f"buoyancy_courant {buoyant:.4g} exceeds buoyancy_courant_limit {buoyancy_courant_limit}"
            )
    if problem.advection != "off":
        courant = float(flow.dt) * float(jnp.max(outgoing_rate(state.velocity)))
        if courant > advective_cfl_limit:
            raise ValueError(
                f"the Courant number {courant:.4g} exceeds advective_cfl_limit {advective_cfl_limit}"
            )
    if problem.outlet_backflow == "reservoir_flux":
        if not np.all(np.isfinite(np.asarray(_reservoir(problem, backflow_temperature, dtype)))):
            raise ValueError("backflow_temperature must be finite")
    else:
        _reservoir(problem, backflow_temperature, dtype)
        axis = flow.open_axis
        if (
            axis is not None
            and problem.advection != "off"
            and np.any(np.asarray(state.velocity[axis].data[(slice(None),) * axis + (-1,)]) < 0.0)
        ):
            raise ValueError("the flow returns through the outlet: use outlet_backflow='reservoir_flux'")


def _check_fields(flow: ChannelProblem, velocity, cells: list[tuple[Field, str]]) -> None:
    """Raise unless each field is a finite Field of the compute dtype at its offset on the flow's grid."""
    if len(velocity) != 3:
        raise ValueError("the velocity needs three components")
    dtype = jnp.result_type(float)
    fields = [(field, _CELL, name) for field, name in cells]
    fields += [(field, velocity_offset(c), f"velocity component {c}") for c, field in enumerate(velocity)]
    for field, offset, name in fields:
        if not (isinstance(field, Field) and field.grid == flow.grid and field.offset == offset):
            raise ValueError(f"the {name} must be a Field at offset {offset} on the problem's grid")
        if field.shape != flow.grid.offset_shape(offset) or field.dtype != dtype:
            raise ValueError(f"the {name} must have shape {flow.grid.offset_shape(offset)} and dtype {dtype}")
        if not np.all(np.isfinite(np.asarray(field.data))):
            raise ValueError(f"the {name} is not finite")


def build_thermal_step(
    problem: ThermalProblem, *, dtype: jnp.dtype | type | str, hold_flow: bool = False
) -> Callable[[ThermalState, Field], tuple[ThermalState, ThermalDiagnostics]]:
    """Factorize on the host and return the pure step ``(state, heat_source) -> (state, diagnostics)``.

    ``dtype`` must be the core's compute dtype, ``jnp.result_type(float)``:
    float64 with x64 enabled before setup, else float32. The step is bound to
    this problem's time step, properties, boundary data and precision; rebuild it
    when any of them changes. ``heat_source`` is the external heat per unit volume
    and time on the cells.

    Everything is taken from the old state, a first-order partitioned step: one
    electric solve gives the Lorentz force and the Joule heat, buoyancy comes from
    the old temperature, the momentum takes :func:`lmhdx.core3d.step`'s predictor
    and projection with the flow's implicit viscous factorizations, and the
    temperature takes its transport by the old velocity, then implicit conduction.

    ``hold_flow`` keeps the state's velocity, a steady flow the heat is carried by,
    without buoyancy, and prepares no viscous inverses. An inflow-outflow duct
    needs it, since the step does not advance such a flow: give it the physical
    velocity of :func:`lmhdx.axial.solve_open_duct`. The step is marked for
    :func:`load_attribution`: ``hold_flow``, and ``affine``, true for a held flow
    and ``"upwind"`` or ``"off"`` transport, where the temperature is affine in the heat source.

    Beyond the update, the diagnostics apply the temperature's Laplacian and wall
    gradients to the new temperature, for ``heat_solve_residual`` and
    ``conductive_heat_out``, and a divergence each to the new velocity and the
    currents, for ``mass_residual`` and ``charge_residual``, besides their
    reductions: no further solve.
    """
    compute = jnp.result_type(float)
    if jnp.dtype(dtype) != compute:
        raise ValueError(
            f"dtype {jnp.dtype(dtype)} is not the core's compute dtype {compute}; set x64 before setup"
        )
    flow, grid = problem.flow, problem.flow.grid
    if flow.open_axis is not None and not hold_flow:
        raise ValueError(
            "the flow of an inflow-outflow duct is held: pass hold_flow=True and its steady velocity"
        )
    if hold_flow and float(problem.expansion_coefficient) and any(problem.gravity):
        raise ValueError("a held flow takes no buoyancy")
    pressure = flow.factorization()
    viscous = None if hold_flow else flow.viscous_factorizations()
    if flow.conducting_walls:
        flow.potential_factorization()  # built once here, so tracing only looks it up
    diffusion, lift = problem.diffusion_factorization(), problem.boundary_lift()
    dt, alpha, capacity = float(flow.dt), problem.diffusivity, problem.heat_capacity
    reference, beta = float(problem.reference_temperature), float(problem.expansion_coefficient)
    upwind = problem.advection != "off"  # either scheme carries heat with the flow

    def total(data):
        return cell_inner_product(Field(data, _CELL, grid), Field(jnp.ones_like(data), _CELL, grid))

    def thermal_step(
        state: ThermalState, heat_source: Field, *, backflow_temperature: float | jnp.ndarray | None = None
    ) -> tuple[ThermalState, ThermalDiagnostics]:
        velocity, temperature = state.velocity, state.temperature
        reservoir = _reservoir(problem, backflow_temperature, compute)
        _, currents, field = face_currents(velocity, flow, pressure)
        joule = _joule_heating(currents, problem).data
        if hold_flow:
            moved = velocity
        else:
            body = face_lorentz_force(_lorentz_from_currents(currents, field, flow), flow)
            buoyancy = _buoyancy(temperature, problem)
            moved, _ = _advance_momentum(velocity, body, flow, pressure, viscous, extra_force=buoyancy)
        fluxes = _upwind_fluxes(temperature, velocity, problem, reservoir) if upwind else None
        transport = divergence(fluxes).data if upwind else 0.0
        carried = (
            _carried_out(fluxes, problem) if upwind and flow.open_axis is not None else jnp.zeros((), compute)
        )
        sources = (joule + heat_source.data) / capacity
        rhs = temperature.data + dt * (sources - transport + alpha * lift.data)
        heated = diffusion.solve(temperature.replace_data(rhs))
        change = heated.data - temperature.data
        conducted = alpha * laplacian(heated, problem.thermal_conditions).data
        storage, out = capacity * total(change) / dt, _heat_out(heated, problem)
        joule_power, external = total(joule), total(heat_source.data)
        excess = jnp.maximum(
            jnp.max(jnp.abs(temperature.data - reference)), jnp.max(jnp.abs(heated.data - reference))
        )
        values = (
            jnp.max(jnp.abs(divergence(moved).data)),
            jnp.max(jnp.abs(divergence(currents).data)),
            jnp.min(heated.data),
            jnp.max(heated.data),
            capacity * total(heated.data - reference),
            joule_power,
            external,
            out,
            carried,
            storage,
            storage + out + carried - joule_power - external,
            jnp.max(jnp.abs(change / dt + transport - conducted - sources)),
            dt * jnp.max(outgoing_rate(velocity)) if upwind else jnp.zeros((), compute),
            beta * excess,
        )
        new = ThermalState(moved, heated, state.time + dt)
        exchange = _outlet_exchange(velocity, reservoir, problem, compute)
        buoyant = _buoyancy_courant(temperature, problem)
        leaves = jax.tree_util.tree_leaves((new, values, exchange[:3], buoyant))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
        return new, ThermalDiagnostics(*values, finite, *exchange, buoyant)

    thermal_step.hold_flow = hold_flow  # load_attribution needs a flow the load cannot move
    thermal_step.affine = hold_flow and problem.advection in ("upwind", "off")  # and a transport linear in T
    return thermal_step


def advance_thermal(
    thermal_step: Callable[[ThermalState, Field], tuple[ThermalState, ThermalDiagnostics]],
    state: ThermalState,
    heat_source: Field,
    steps: int,
    *,
    checkpoint: bool = True,
    backflow_temperature: float | jnp.ndarray | None = None,
) -> tuple[ThermalState, ThermalDiagnostics]:
    """Take ``steps`` steps of a prepared step as one compiled scan; return the end state and the history.

    ``backflow_temperature`` goes to every step, constant over the call as the
    heat source is. The carry is the state alone and only the scalar diagnostics are stacked, one
    per step, so no field history is kept. ``steps`` is static and the heat source
    is constant over the call. ``checkpoint`` recomputes each step in reverse mode
    rather than storing its intermediates; a differentiated run still keeps one
    state per step.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    options = {} if backflow_temperature is None else {"backflow_temperature": backflow_temperature}

    def single(carry: ThermalState, _) -> tuple[ThermalState, ThermalDiagnostics]:
        return thermal_step(carry, heat_source, **options)

    body = jax.checkpoint(single) if checkpoint else single
    return jax.lax.scan(body, state, xs=None, length=steps)


_RECORD_SCHEMA, _RECORD_VERSION = "lmhdx.thermal.state", 1


def _described(problem: ThermalProblem) -> str:
    """The problem as plain data, JSON: what a record must be given again to resume bit for bit."""
    flow = problem.flow

    def condition(item: BoundaryCondition) -> dict:
        return {
            "kinds": list(item.kinds),
            "lower": np.asarray(item.lower, float).tolist(),
            "upper": np.asarray(item.upper, float).tolist(),
        }

    field = flow.magnetic_field
    if isinstance(field, ImposedField):  # its shapes, dtypes and placement as well as its bytes
        arrays = [*field.components, *(field.faces or ())]
        layout = repr([(array.shape, str(array.dtype)) for array in arrays] + [field.faces is None]).encode()
        field = hashlib.sha256(layout + b"".join(np.asarray(array).tobytes() for array in arrays)).hexdigest()
    names = (
        "density",
        "viscosity",
        "conductivity",
        "dt",
        "forcing",
        "wall_conductance",
        "advection",
        "precision",
    )
    thermal = ("specific_heat", "thermal_conductivity", "reference_temperature", "expansion_coefficient")
    thermal += ("gravity", "joule_heating", "advection", "outlet_backflow")
    return json.dumps(
        {
            "faces": [np.asarray(faces, float).tolist() for faces in flow.grid.faces],
            "flow": {name: getattr(flow, name) for name in names},
            "magnetic_field": field if isinstance(field, str) else [float(value) for value in field],
            "wall_layers": flow.wall_layers,
            "flow_conditions": [condition(item) for item in flow.conditions],
            "thermal": {name: getattr(problem, name) for name in thermal},
            "thermal_conditions": [condition(item) for item in problem.thermal_conditions],
        },
        sort_keys=True,
        default=float,
    )


def write_thermal_record(
    path: str | Path, problem: ThermalProblem, state: ThermalState, *, units: str, run: dict | None = None
) -> Path:
    """Write ``state`` as a versioned NPZ record, its staggered velocity, temperature and time exactly.

    The record holds its schema, version and ``units``, a label the caller
    declares (such as ``"SI"``) and :func:`load_thermal_record` checks, and the
    problem it belongs to, as plain data. ``run`` is the provenance of the run's
    other inputs, plain JSON data such as its heat source, whether its flow is held
    and an outlet's backflow temperature, kept so that a resume can check it.
    Returns the path written.
    """
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"velocity_{name}": np.asarray(field.data) for name, field in zip("xyz", state.velocity)}
    np.savez(
        path,
        schema=_RECORD_SCHEMA,
        version=_RECORD_VERSION,
        units=units,
        problem=_described(problem),
        run=json.dumps(run, sort_keys=True),
        temperature=np.asarray(state.temperature.data),
        time=np.asarray(state.time),
        **arrays,
    )
    return path


def load_thermal_record(
    path: str | Path, problem: ThermalProblem, *, units: str, run: dict | None = None
) -> ThermalState:
    """Return the state a :func:`write_thermal_record` record holds, raising ``ValueError`` unless it fits.

    The schema and version must be this one's, ``units`` the record's, and
    ``problem`` the one it was written with, in every value the step depends on.
    Every field must be finite, of the problem's shape and the compute dtype, and
    the time a finite scalar of that dtype; all is checked before a state is made.
    Given ``run``, the load is an exact resume and the record's run provenance must
    equal it; without, it is a state import, the start of another experiment.
    """
    dtype, grid = jnp.result_type(float), problem.flow.grid
    with np.load(Path(path), allow_pickle=False) as record:
        arrays = {name: np.asarray(record[name]) for name in record.files}
    names = ("schema", "version", "units", "problem", "run", "temperature", "time")
    missing = sorted(set(names + tuple(f"velocity_{name}" for name in "xyz")) - set(arrays))
    if missing:
        raise ValueError(f"{path} lacks {', '.join(missing)}")

    def scalar(name: str, kinds: str):
        value = arrays[name]
        if value.shape or value.dtype.kind not in kinds:
            raise ValueError(f"{path}: {name} is not a scalar of kind {kinds!r}")
        return value[()]

    if str(scalar("schema", "U")) != _RECORD_SCHEMA or int(scalar("version", "iu")) != _RECORD_VERSION:
        raise ValueError(f"{path} is not a {_RECORD_SCHEMA} record of version {_RECORD_VERSION}")
    if str(scalar("units", "U")) != units:
        raise ValueError(f"{path} is in units {str(arrays['units'])!r}, not {units!r}")
    written, expected = json.loads(str(scalar("problem", "U"))), json.loads(_described(problem))
    if written != expected:
        differing = sorted(key for key in expected if written.get(key) != expected[key])
        raise ValueError(f"{path} was written for another problem: {', '.join(differing)} differ")
    if run is not None and json.loads(str(scalar("run", "U"))) != json.loads(json.dumps(run)):
        raise ValueError(f"{path} was written by a run with other inputs: {str(arrays['run'])}")
    time = arrays["time"]
    if time.shape or time.dtype != dtype or not np.isfinite(time):
        raise ValueError(f"{path}: the time must be a finite {dtype} scalar")
    pieces = [(arrays[f"velocity_{name}"], velocity_offset(axis)) for axis, name in enumerate("xyz")]
    pieces.append((arrays["temperature"], _CELL))
    for data, offset in pieces:
        if data.shape != grid.offset_shape(offset) or data.dtype != dtype:
            raise ValueError(
                f"{path} holds a {data.dtype} array of shape {data.shape}, not {dtype} {grid.offset_shape(offset)}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{path} holds a field that is not finite")
    fields = [Field(jnp.asarray(data), offset, grid) for data, offset in pieces]
    return ThermalState(tuple(fields[:3]), fields[3], jnp.asarray(time))


class LoadAttribution(NamedTuple):
    """Where a load's heat goes, by difference at the last step of :func:`load_attribution`; each a power.

    ``imposed`` is the load's power. ``delivered`` is what it adds to the enthalpy
    leaving through the outlet, net of the inlet. ``conducted`` is what it adds to
    the conduction out: a prescribed wall flux cancels, so this is what leaks
    through the inlet and any prescribed-temperature wall, together. ``stored`` is
    what it still adds to storage, zero once steady, and ``residual`` is
    ``imposed - delivered - conducted - stored``: the two ledgers' residuals
    differenced, with the opposite sign, zero to their rounding. ``loaded`` is the
    load-on run's last diagnostics.
    """

    imposed: jnp.ndarray
    delivered: jnp.ndarray
    conducted: jnp.ndarray
    stored: jnp.ndarray
    residual: jnp.ndarray
    loaded: ThermalDiagnostics


def load_attribution(
    thermal_step: Callable[[ThermalState, Field], tuple[ThermalState, ThermalDiagnostics]],
    state: ThermalState,
    background: Field,
    load: Field,
    steps: int,
    *,
    backflow_temperature: float | jnp.ndarray | None = None,
) -> LoadAttribution:
    """Run ``steps`` steps with ``background + load`` and with ``background`` alone, and attribute the load.

    Both runs share the start, the held velocity and so its Joule heat, the
    inlet and reservoir temperatures and the walls, so their difference is the
    load's own response: neither Joule heat nor ``background`` is credited to
    it. That holds only where the temperature is affine in the heat source: a
    step of :func:`build_thermal_step` with ``hold_flow=True``, constant
    properties and ``"upwind"`` or ``"off"`` transport, marked ``affine``. A flow
    the load moves, by buoyancy, is not affine in it, nor is van Leer's face
    value, a nonlinear function of the temperature even on a held flow. Each is
    refused, and so is a wrapper of the step that drops its marks, such as a
    lambda; ``jax.jit`` and ``jax.checkpoint`` keep them.

    The load is a heat source on the cells. A heat flux ``q`` through a wall is
    part of the step's boundary data, so it is in both runs and cancels; to
    credit it, pass it as the source ``q A / V`` in the cells along that wall,
    with that wall's flux set to zero. With the transports accepted here the
    cell temperatures are then the same: a face flux enters its cell's equation
    exactly as that source.
    """
    if not hasattr(thermal_step, "affine"):
        raise ValueError(
            "load_attribution needs the step build_thermal_step returns; a wrapper loses its marks"
        )
    if not thermal_step.hold_flow:
        raise ValueError(
            "load_attribution needs a step built with hold_flow=True: a moving flow is not affine in the load"
        )
    if not thermal_step.affine:
        raise ValueError(
            "load_attribution needs upwind or no transport: a limited face value is not affine in the temperature,"
            " so on a held flow the difference of the runs still depends on the background"
        )
    options = {"backflow_temperature": backflow_temperature}
    on = advance_thermal(
        thermal_step, state, background.replace_data(background.data + load.data), steps, **options
    )[1]
    off = advance_thermal(thermal_step, state, background, steps, **options)[1]
    imposed, delivered, conducted, stored = (
        getattr(on, name)[-1] - getattr(off, name)[-1]
        for name in ("external_power", "advective_heat_out", "conductive_heat_out", "storage_rate")
    )
    last = jax.tree_util.tree_map(lambda history: history[-1], on)
    residual = imposed - delivered - conducted - stored
    return LoadAttribution(imposed, delivered, conducted, stored, residual, last)


def _reservoir(problem: ThermalProblem, value, dtype) -> jnp.ndarray | None:
    """Return the outlet's incoming temperature over its face: a number broadcasts, an array must match."""
    if problem.outlet_backflow != "reservoir_flux":
        if value is not None:
            raise ValueError("backflow_temperature is for outlet_backflow='reservoir_flux'")
        return None
    if value is None:
        raise ValueError("outlet_backflow='reservoir_flux' needs backflow_temperature")
    if jnp.iscomplexobj(value):
        raise ValueError("backflow_temperature must be real")
    axis, shape = problem.flow.open_axis, problem.flow.grid.shape
    face = shape[:axis] + shape[axis + 1 :]
    data = jnp.asarray(value, dtype=dtype)
    if data.shape not in ((), face):
        raise ValueError(f"backflow_temperature must be a number or an array of shape {face}")
    return jnp.broadcast_to(data, face)


def _outlet_exchange(velocity, reservoir, problem: ThermalProblem, dtype) -> tuple[jnp.ndarray, ...]:
    """Return the outlet's returning and leaving volume rates, the enthalpy the return brings in, and validity."""
    axis = problem.flow.open_axis
    if axis is None:
        zero = jnp.zeros((), dtype)
        return zero, zero, zero, jnp.asarray(True)
    end = (slice(None),) * axis + (-1,)
    normal = velocity[axis].data[end]
    area = jnp.asarray(problem.flow.grid.face_areas(axis)[end], dtype=normal.dtype)
    back = jnp.maximum(-normal, 0.0) * area
    brought = jnp.zeros((), dtype) if reservoir is None else problem.heat_capacity * jnp.sum(back * reservoir)
    carried = reservoir is None and problem.advection != "off"  # else nothing crosses with the flow
    valid = jnp.all(back == 0.0) if carried else jnp.asarray(True)
    return jnp.sum(back), jnp.sum(jnp.maximum(normal, 0.0) * area), brought, valid


def _carried_out(fluxes: tuple[Field, Field, Field], problem: ThermalProblem) -> jnp.ndarray:
    """Return the enthalpy carried out through the outlet less what enters at the inlet and returning faces."""
    axis = problem.flow.open_axis
    flux, areas = fluxes[axis].data, problem.flow.grid.face_areas(axis)
    ends = [(slice(None),) * axis + (end,) for end in (0, -1)]
    leaving, entering = (jnp.sum(flux[end] * jnp.asarray(areas[end], dtype=flux.dtype)) for end in ends[::-1])
    return problem.heat_capacity * (leaving - entering)


def _heat_out(temperature: Field, problem: ThermalProblem) -> jnp.ndarray:
    """Return the conductive heat leaving through the walls, from the wall gradients the conduction uses."""
    grid, conductivity = temperature.grid, float(problem.thermal_conductivity)
    out = jnp.zeros((), temperature.dtype)
    for axis, condition in enumerate(problem.thermal_conditions):
        if condition.is_periodic:
            continue
        gradient, areas = face_gradient(temperature, axis, condition).data, grid.face_areas(axis)
        for end, sign in ((0, 1.0), (-1, -1.0)):
            slab = (slice(None),) * axis + (end,)
            area = jnp.asarray(areas[slab], dtype=temperature.dtype)
            out = out + sign * conductivity * jnp.sum(gradient[slab] * area)
    return out


class DevelopedHeat(NamedTuple):
    """The thermally fully developed duct of :func:`developed_heat`, at its station ``x = 0``.

    ``temperature`` is ``theta`` on the cells. The wall values are taken on the
    wall faces of the cross-section, means weighted by face area over every
    transverse wall, heated or not: the wall temperature and the heat flux into
    the fluid. ``nusselt`` is ``q_w D_h / (k (T_w - T_b))``, with the hydraulic
    diameter ``D_h = 4 A / P``. With an internal source it is an apparent number
    for that source, not a coefficient of the duct alone. It is NaN, undefined,
    where ``|T_w - T_b|`` is within ``64 eps (L/h)^2`` of the largest temperature
    scale, with ``L/h`` the section's largest extent over its smallest cell, the
    conditioning of the solve; the scales are
    ``|T_w|``, ``|T_b|``, ``max |theta|``, and each heat input's own,
    ``D_h^2 max |rho c_p u gamma| / k``, ``D_h^2 max |q + q_J| / k`` and
    ``D_h max |q_w| / k``, so a difference cancelled to rounding is undefined;
    the temperatures and the balance stay valid.
    ``heat_balance_residual`` is ``rho c_p gamma Q`` less the heat entering per
    unit length, a power per length.

    With heat-flux walls only, ``flux_walls``, ``theta`` has an arbitrary
    constant, its zero mean, and :meth:`anchored_peaks` gives the physical
    temperatures.
    """

    temperature: Field
    axial_gradient: jnp.ndarray
    flow_rate: jnp.ndarray
    bulk_temperature: jnp.ndarray
    wall_temperature: jnp.ndarray
    peak_wall_temperature: jnp.ndarray
    wall_heat_flux: jnp.ndarray
    hydraulic_diameter: float
    nusselt: jnp.ndarray
    heat_balance_residual: jnp.ndarray
    flux_walls: bool

    def anchored_peaks(
        self, inlet_bulk_temperature: float | jnp.ndarray, length: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the hottest wall face and hottest cell over ``0 <= x <= length``, for heat-flux walls.

        The bulk temperature is ``inlet_bulk_temperature`` at ``x = 0``, so the
        temperature is ``T_in + gamma x + theta - theta_b``, the same for any
        constant added to ``theta``. It is linear in ``x``: each peak is at the end
        ``gamma`` rises toward. A prescribed wall temperature fixes the reference
        itself, so there both are NaN: use its station values. The profile is taken
        as developed from ``x = 0``, with no entrance region.
        """
        anchor = (
            inlet_bulk_temperature + jnp.maximum(self.axial_gradient * length, 0.0) - self.bulk_temperature
        )
        anchor = jnp.where(self.flux_walls, anchor, jnp.nan)
        return anchor + self.peak_wall_temperature, anchor + jnp.max(self.temperature.data)


def developed_heat(
    problem: ThermalProblem,
    velocity: tuple[Field, Field, Field],
    heat_source: Field,
    *,
    axial_gradient: float | jnp.ndarray | None = None,
) -> DevelopedHeat:
    """Solve the thermally fully developed temperature of a duct carrying ``velocity``.

    The duct is periodic along exactly one axis, the flow's, and the velocity
    and heat sources are uniform along it. Then ``T = gamma x + theta`` with
    ``k L(theta) = rho c_p u gamma - q - q_J``, one linear solve on the cells;
    the bulk temperature is weighted by the axial flow. With heat-flux walls only,
    ``gamma`` follows from the heat balance ``rho c_p gamma Q`` = heat in per unit
    length, and ``theta`` has zero mean. A prescribed-temperature wall takes in
    heat the solve decides, so ``axial_gradient`` must then be given, and that
    wall temperature rises along the duct at ``gamma`` too. A uniform axial heat
    input with a peripherally uniform wall temperature is the H1 condition; a
    uniform wall heat flux is H2. The flow rate must be positive, which
    :func:`validate_developed_inputs` checks with the rest of the reduction;
    otherwise the bulk temperature and Nu are NaN, as are ``gamma`` and the
    temperatures of heat-flux walls. Differentiable
    in the velocity, the heat source and ``axial_gradient``, and compiled whole
    by :func:`jax.jit`. With ``joule_heating`` off no current is formed.
    """
    flow, grid, conditions = problem.flow, problem.flow.grid, problem.thermal_conditions
    axial, capacity, k = (
        _developed_axis(problem, axial_gradient),
        problem.heat_capacity,
        float(problem.thermal_conductivity),
    )
    length = float(np.sum(grid.widths[axial]))

    def per_length(data):
        return cell_inner_product(Field(data, _CELL, grid), Field(jnp.ones_like(data), _CELL, grid)) / length

    speed = cell_average(velocity[axial], axial).data
    sources = heat_source.data
    if problem.joule_heating:  # otherwise no current is needed
        sources = _joule_heating(face_currents(velocity, flow)[1], problem).data + sources
    lift, flow_rate = problem.boundary_lift().data, per_length(speed)
    flowing = flow_rate > 0.0
    rate = jnp.where(flowing, flow_rate, 1.0)  # no division by zero, so no NaN in a derivative either
    if axial_gradient is None:  # heat-flux walls: k sum(V b) is the heat they let in
        gradient = jnp.where(
            flowing, (per_length(sources) + k * per_length(lift)) / (capacity * rate), jnp.nan
        )
    else:
        gradient = jnp.asarray(axial_gradient, dtype=speed.dtype)
    homogeneous = tuple(condition.homogeneous() for condition in conditions)
    factorization = fast_diagonal_poisson(grid, homogeneous, precision=flow.precision)
    theta = factorization.solve(Field((capacity * speed * gradient - sources) / k - lift, _CELL, grid))
    temperatures, fluxes, areas, wetted = [], [], [], 0.0
    for axis, condition in enumerate(conditions):
        if axis == axial:
            continue
        padded, normal = (
            pad(theta.data, axis, condition, grid=grid),
            face_gradient(theta, axis, condition).data,
        )
        for end, inward in ((0, -1.0), (-1, 1.0)):
            pair = (0, 1) if end == 0 else (-1, -2)
            slab = (slice(None),) * axis
            temperatures.append(0.5 * (padded[slab + (pair[0],)] + padded[slab + (pair[1],)]))
            fluxes.append(inward * k * normal[slab + (end,)])
            faces = grid.face_areas(axis)[slab + (end,)]  # host geometry, never traced
            areas.append(jnp.asarray(faces, dtype=speed.dtype))
            wetted += float(np.sum(faces))
    perimeter = wetted
    wall = sum(jnp.sum(area * value) for area, value in zip(areas, temperatures)) / perimeter
    flux = sum(jnp.sum(area * value) for area, value in zip(areas, fluxes)) / perimeter
    bulk = jnp.where(flowing, per_length(speed * theta.data) / rate, jnp.nan)
    diameter = 4.0 * float(np.sum(grid.cell_volumes())) / wetted
    difference, scales = wall - bulk, [jnp.abs(wall), jnp.abs(bulk), jnp.max(jnp.abs(theta.data))]
    scales += [diameter**2 * jnp.max(jnp.abs(term)) / k for term in (capacity * speed * gradient, sources)]
    scales += [diameter * jnp.max(jnp.abs(value)) / k for value in fluxes]
    # Rounding in the solve grows with its conditioning, (extent / smallest cell)^2 across the section.
    conditioning = max(
        (np.sum(grid.widths[axis]) / np.min(grid.widths[axis])) ** 2 for axis in range(3) if axis != axial
    )
    floor = 64.0 * jnp.finfo(difference.dtype).eps * conditioning
    resolved = jnp.abs(difference) > floor * jnp.max(jnp.stack(scales))
    entering = per_length(sources) + flux * perimeter / length
    return DevelopedHeat(
        theta,
        gradient,
        flow_rate,
        bulk,
        wall,
        jnp.max(jnp.stack([jnp.max(value) for value in temperatures])),
        flux,
        diameter,
        jnp.where(resolved, flux * diameter / (k * jnp.where(resolved, difference, 1.0)), jnp.nan),
        capacity * gradient * flow_rate - entering,
        not any(DIRICHLET in condition.kinds for condition in conditions),
    )


def _developed_axis(problem: ThermalProblem, axial_gradient) -> int:
    """Return the developed duct's axis: its one periodic axis, one cell long, the static checks of the reduction."""
    conditions = problem.thermal_conditions
    periodic = [axis for axis, condition in enumerate(conditions) if condition.is_periodic]
    if len(periodic) != 1:
        raise ValueError("a developed duct is periodic along exactly one axis, the flow's")
    if problem.flow.grid.shape[periodic[0]] != 1:
        raise ValueError("a developed duct has one cell along its axis")
    if any(DIRICHLET in condition.kinds for condition in conditions) == (axial_gradient is None):
        raise ValueError("give axial_gradient exactly when a wall prescribes its temperature")
    return periodic[0]


def validate_developed_inputs(
    problem: ThermalProblem,
    velocity: tuple[Field, Field, Field],
    heat_source: Field,
    *,
    flow_rate_tolerance: float,
    velocity_tolerance: float,
    axial_gradient: float | None = None,
) -> None:
    """Check a velocity and heat source for :func:`developed_heat` on the host, raising ``ValueError`` on the first fault.

    The duct is periodic along one axis, one cell long, with a finite
    ``axial_gradient`` given exactly when a wall prescribes its temperature; the
    fields are finite, of the compute dtype, at their offsets on the problem's
    grid. The flow is the reduction's single throughflow: a flow rate ``sum A u``
    above ``flow_rate_tolerance``, a volume rate; the same on both axial faces,
    and no transverse velocity or reversed axial velocity, beyond
    ``velocity_tolerance``, a speed. Nothing is changed or returned.
    """
    for name, value in (
        ("flow_rate_tolerance", flow_rate_tolerance),
        ("velocity_tolerance", velocity_tolerance),
    ):
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be nonnegative and finite")
    axial, grid = _developed_axis(problem, axial_gradient), problem.flow.grid
    if axial_gradient is not None and not np.all(np.isfinite(np.asarray(axial_gradient))):
        raise ValueError("axial_gradient must be finite")
    _check_fields(problem.flow, velocity, [(heat_source, "heat source")])
    across = max(float(jnp.max(jnp.abs(velocity[axis].data))) for axis in range(3) if axis != axial)
    if across > velocity_tolerance:
        raise ValueError(f"a transverse velocity of {across:.4g} exceeds velocity_tolerance")
    along = np.asarray(velocity[axial].data)
    ends = [along[(slice(None),) * axial + (end,)] for end in (0, -1)]
    if float(np.max(np.abs(ends[1] - ends[0]))) > velocity_tolerance:
        raise ValueError("the axial velocity changes along the axis beyond velocity_tolerance")
    if float(np.min(along)) < -velocity_tolerance:
        raise ValueError("the axial flow reverses beyond velocity_tolerance")
    flow_rate = float(
        np.sum(grid.face_areas(axial)[(slice(None),) * axial + (0,)] * along[(slice(None),) * axial + (0,)])
    )
    if not flow_rate > flow_rate_tolerance:
        raise ValueError(f"the flow rate {flow_rate:.4g} is not above flow_rate_tolerance")


class ThermalRun(NamedTuple):
    """A thermal run read by :func:`lmhdx.io.thermal_run_from_toml`: the problem, its start, and how to run and record it.

    ``provenance`` holds the run's other inputs, its heat source and held flow, kept in each record so that a
    resume can check them; ``limits`` the advective Courant and buoyancy limits each chunk's samples must meet
    (None: the scheme's own, or none); ``restart`` the record it started from and how, or None.
    """

    name: str
    units: str
    problem: ThermalProblem
    state: ThermalState
    heat_source: Field
    hold_flow: bool
    steps: int
    chunk: int
    divergence_tolerance: float
    directory: Path
    write_paraview: bool
    write_restart: bool
    provenance: dict
    limits: dict
    restart: dict | None
