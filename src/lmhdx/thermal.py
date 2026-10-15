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

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .core3d import (
    ChannelProblem,
    _advance_momentum,
    _lorentz_from_currents,
    enforce_face_constraints,
    face_currents,
    face_lorentz_force,
    velocity_offset,
    zero_velocity,
)
from .grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, pad
from .ops import (
    cell_inner_product,
    divergence,
    face_average,
    face_average_adjoint,
    face_gradient,
    laplacian,
    wall_insulated,
)
from .poisson import FastDiagonalHelmholtz, fast_diagonal_helmholtz, fast_diagonal_poisson
from .steady import with_inflow

__all__ = [
    "DevelopedHeat",
    "ThermalDiagnostics",
    "ThermalProblem",
    "ThermalState",
    "advance_thermal",
    "build_thermal_step",
    "developed_heat",
    "initial_thermal_state",
    "thermal_boundary",
    "validate_thermal_inputs",
]

_KINDS = {"temperature": DIRICHLET, "heat_flux": NEUMANN, "periodic": PERIODIC}
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
    where no periodic pressure balances its mean (ADR 0008, D32). ``advection`` is ``"upwind"``, or ``"off"`` for conduction alone,
    whatever the flow's own momentum transport.

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
        if self.advection not in ("upwind", "off"):
            raise ValueError(f"advection must be 'upwind' or 'off', got {self.advection!r}")
        if self.outlet_backflow not in ("error", "reservoir_flux"):
            raise ValueError(
                f"outlet_backflow must be 'error' or 'reservoir_flux', got {self.outlet_backflow!r}"
            )
        if self.outlet_backflow == "reservoir_flux" and (
            self.flow.open_axis is None or self.advection != "upwind"
        ):
            raise ValueError(
                "outlet_backflow='reservoir_flux' needs an inflow-outflow axis and upwind transport"
            )
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

    ``T_up`` is the temperature of the cell the face velocity comes from. The
    flow's scalar conditions pad periodic axes by wrapping and walls by copying;
    the copy is never used, since a wall face carries no advective flux. The
    inlet face brings in the inlet temperature, or takes out the first cell's
    against a backflow. The outlet face takes out the last cell's; where the flow
    returns there it brings in ``reservoir``, the outlet's incoming temperature.
    Without one, only an unvalidated step under ``outlet_backflow="error"``
    reaches a returning face; it takes the last cell's temperature and reports the
    boundary invalid.
    """
    scalar, opening = problem.flow.scalar_conditions, problem.flow.open_axis
    fluxes = []
    for axis, carrier in enumerate(velocity):
        padded = pad(temperature.data, axis, scalar[axis], grid=temperature.grid)
        lower, upper = (_sliced(padded, axis, start) for start in (0, 1))
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


def _outgoing_rate(velocity: tuple[Field, Field, Field]) -> jnp.ndarray:
    """Return each cell's outgoing volume flux over its volume, ``sum_f max(u_f . n_f, 0) A_f / V``.

    ``dt`` times its maximum is the transport's Courant number. On a Cartesian
    grid ``A_f / V`` is one over the cell's width along the face normal.
    """
    grid, total = velocity[0].grid, 0.0
    for axis, carrier in enumerate(velocity):
        lower, upper = (_sliced(carrier.data, axis, start) for start in (0, 1))
        shape = [-1 if position == axis else 1 for position in range(3)]
        widths = jnp.asarray(np.reshape(grid.widths[axis], shape), dtype=carrier.dtype)
        total = total + (jnp.maximum(upper, 0.0) + jnp.maximum(-lower, 0.0)) / widths
    return total


def _sliced(data: jnp.ndarray, axis: int, start: int) -> jnp.ndarray:
    """All but the last (``start`` 0) or the first (``start`` 1) entry along ``axis``."""
    return jax.lax.slice_in_dim(data, start, data.shape[axis] - 1 + start, axis=axis)


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
    scalar = problem.flow.scalar_conditions
    excess = -float(problem.flow.density) * beta * (temperature.data - float(problem.reference_temperature))
    faces = (
        face_average(temperature.replace_data(excess * g), axis, scalar[axis])
        for axis, g in enumerate(gravity)
    )
    return tuple(wall_insulated(face, axis, scalar[axis]) for axis, face in enumerate(faces))


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
    less what it brings in through an inlet (zero in a closed duct); ``heat_solve_residual`` is the
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
    advective_cfl_limit: float = 0.8,
) -> None:
    """Check a state and heat source on the host before a run, raising ``ValueError`` on the first fault.

    Grids, offsets and shapes; the core's compute dtype; finite values; the wall
    and periodic face constraints; a velocity divergence within
    ``divergence_tolerance``, in inverse time; with upwind transport, a Courant
    number ``dt max(sum_f max(u_f . n_f, 0) A_f / V)`` within ``advective_cfl_limit``;
    and, at an outlet, no returning
    face under ``outlet_backflow="error"``, or a finite ``backflow_temperature`` of
    the outlet's shape under ``"reservoir_flux"``. Nothing is changed or returned.

    A held flow keeps its Courant number for the whole run. An evolving flow is
    checked at its start only: its run is accepted on the ``advective_cfl`` it
    reports each step.
    """
    if not (np.isfinite(divergence_tolerance) and divergence_tolerance >= 0.0):
        raise ValueError("divergence_tolerance must be nonnegative and finite")
    if not (np.isfinite(advective_cfl_limit) and advective_cfl_limit > 0.0):
        raise ValueError("advective_cfl_limit must be positive and finite")
    flow, dtype = problem.flow, jnp.result_type(float)
    if len(state.velocity) != 3:
        raise ValueError("the velocity needs three components")
    fields = [(state.temperature, _CELL, "temperature"), (heat_source, _CELL, "heat source")]
    fields += [
        (field, velocity_offset(c), f"velocity component {c}") for c, field in enumerate(state.velocity)
    ]
    for field, offset, name in fields:
        if not (isinstance(field, Field) and field.grid == flow.grid and field.offset == offset):
            raise ValueError(f"the {name} must be a Field at offset {offset} on the problem's grid")
        if field.shape != flow.grid.offset_shape(offset) or field.dtype != dtype:
            raise ValueError(f"the {name} must have shape {flow.grid.offset_shape(offset)} and dtype {dtype}")
        if not np.all(np.isfinite(np.asarray(field.data))):
            raise ValueError(f"the {name} is not finite")
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
    if problem.advection == "upwind":
        courant = float(flow.dt) * float(jnp.max(_outgoing_rate(state.velocity)))
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
        if axis is not None and np.any(
            np.asarray(state.velocity[axis].data[(slice(None),) * axis + (-1,)]) < 0.0
        ):
            raise ValueError("the flow returns through the outlet: use outlet_backflow='reservoir_flux'")


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
    temperature takes upwind transport by the old velocity, then implicit conduction.

    ``hold_flow`` keeps the state's velocity, a steady flow the heat is carried by,
    without buoyancy. An inflow-outflow duct needs it, since the step does not
    advance such a flow: give it the physical velocity of
    :func:`lmhdx.axial.solve_open_duct`.
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
    pressure, viscous = flow.factorization(), flow.viscous_factorizations()
    if flow.conducting_walls:
        flow.potential_factorization()  # built once here, so tracing only looks it up
    diffusion, lift = problem.diffusion_factorization(), problem.boundary_lift()
    dt, alpha, capacity = float(flow.dt), problem.diffusivity, problem.heat_capacity
    reference, beta = float(problem.reference_temperature), float(problem.expansion_coefficient)
    upwind = problem.advection == "upwind"

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
            dt * jnp.max(_outgoing_rate(velocity)) if upwind else jnp.zeros((), compute),
            beta * excess,
        )
        new = ThermalState(moved, heated, state.time + dt)
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves((new, values))])
        )
        return new, ThermalDiagnostics(
            *values, finite, *_outlet_exchange(velocity, reservoir, problem, compute)
        )

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
    valid = jnp.asarray(True) if reservoir is not None else jnp.all(back == 0.0)
    return jnp.sum(back), jnp.sum(jnp.maximum(normal, 0.0) * area), brought, valid


def _carried_out(fluxes: tuple[Field, Field, Field], problem: ThermalProblem) -> jnp.ndarray:
    """Return the enthalpy the flow carries out through the outlet less what it brings in at the inlet."""
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
    wall faces of the cross-section, means weighted by face area: the wall
    temperature and the heat flux into the fluid. ``nusselt`` is
    ``q_w D_h / (k (T_w - T_b))``, with the hydraulic diameter ``D_h = 4 A / P``.
    ``heat_balance_residual`` is ``rho c_p gamma Q`` less the heat entering per
    unit length, a power per length.
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
    uniform wall heat flux is H2. The flow rate must be positive. Differentiable
    in the velocity, the heat source and ``axial_gradient``.
    """
    flow, grid, conditions = problem.flow, problem.flow.grid, problem.thermal_conditions
    periodic = [axis for axis, condition in enumerate(conditions) if condition.is_periodic]
    if len(periodic) != 1:
        raise ValueError("a developed duct is periodic along exactly one axis, the flow's")
    if any(DIRICHLET in condition.kinds for condition in conditions) == (axial_gradient is None):
        raise ValueError("give axial_gradient exactly when a wall prescribes its temperature")
    axial, capacity, k = periodic[0], problem.heat_capacity, float(problem.thermal_conductivity)
    length = float(np.sum(grid.widths[axial]))

    def per_length(data):
        return cell_inner_product(Field(data, _CELL, grid), Field(jnp.ones_like(data), _CELL, grid)) / length

    carrier = velocity[axial].data
    speed = 0.5 * (_sliced(carrier, axial, 0) + _sliced(carrier, axial, 1))
    sources = _joule_heating(face_currents(velocity, flow)[1], problem).data + heat_source.data
    lift, flow_rate = problem.boundary_lift().data, per_length(speed)
    if axial_gradient is None:  # heat-flux walls: k sum(V b) is the heat they let in
        gradient = (per_length(sources) + k * per_length(lift)) / (capacity * flow_rate)
    else:
        gradient = jnp.asarray(axial_gradient, dtype=speed.dtype)
    homogeneous = tuple(condition.homogeneous() for condition in conditions)
    factorization = fast_diagonal_poisson(grid, homogeneous, precision=flow.precision)
    theta = factorization.solve(Field((capacity * speed * gradient - sources) / k - lift, _CELL, grid))
    temperatures, fluxes, areas = [], [], []
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
            areas.append(jnp.asarray(grid.face_areas(axis)[slab + (end,)], dtype=speed.dtype))
    perimeter = sum(jnp.sum(area) for area in areas)
    wall = sum(jnp.sum(area * value) for area, value in zip(areas, temperatures)) / perimeter
    flux = sum(jnp.sum(area * value) for area, value in zip(areas, fluxes)) / perimeter
    bulk = per_length(speed * theta.data) / flow_rate
    diameter = (
        4.0 * float(np.sum(grid.cell_volumes())) / float(sum(np.sum(np.asarray(area)) for area in areas))
    )
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
        flux * diameter / (k * (wall - bulk)),
        capacity * gradient * flow_rate - entering,
    )
