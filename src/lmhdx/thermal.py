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
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .core3d import ChannelProblem
from .grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field
from .ops import laplacian
from .poisson import FastDiagonalHelmholtz, fast_diagonal_helmholtz

__all__ = ["ThermalProblem", "thermal_boundary"]

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
    where the flow's is. The flow must be on a Cartesian grid without an
    inflow-outflow axis.
    """

    flow: ChannelProblem
    thermal_conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    specific_heat: float
    thermal_conductivity: float

    def __post_init__(self) -> None:
        if not isinstance(self.flow, ChannelProblem):
            raise TypeError("flow must be a lmhdx.core3d.ChannelProblem")
        if self.flow.grid.is_polar or self.flow.open_axis is not None:
            raise ValueError("heat transport needs a Cartesian flow without an inflow-outflow axis")
        for name in ("specific_heat", "thermal_conductivity"):
            if not (np.isfinite(getattr(self, name)) and float(getattr(self, name)) > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("density", "dt"):
            if not np.isfinite(getattr(self.flow, name)):
                raise ValueError(f"the flow's {name} must be finite")
        if len(self.thermal_conditions) != 3:
            raise ValueError("heat transport needs one boundary condition per axis")
        shape = self.flow.grid.shape
        for axis, (condition, flow) in enumerate(zip(self.thermal_conditions, self.flow.conditions)):
            if not isinstance(condition, BoundaryCondition):
                raise TypeError(f"axis {axis} needs a lmhdx.grid.BoundaryCondition")
            if condition.is_periodic != flow.is_periodic:
                raise ValueError(f"axis {axis} must be periodic for heat exactly where it is for the flow")
            tangential = shape[:axis] + shape[axis + 1 :]
            for value in (condition.lower, condition.upper):
                if np.ndim(value) and np.shape(value) != tangential:
                    raise ValueError(f"axis {axis} boundary data must have shape {tangential}")

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
