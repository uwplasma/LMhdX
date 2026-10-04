"""Fully developed duct design: throughput, pumping power and their derivatives.

At fixed field, materials and geometry, flow is linear in force density:
``Q = G f``. One unit-drive solve gives ``G``, eliminating the drive from
fixed-flow optimization as ``f = Q_target / G`` with ``df/dQ = 1/G``.
For a uniform pressure-gradient drive over length ``L``, pressure drop is
``f L`` and hydraulic power is ``f L Q``. These are isothermal segment
quantities, excluding entry/exit losses, manifolds and thermal effects.

A ``CaseSpec`` is solved on the staggered core through
:mod:`lmhdx.fully_developed`, so both families share one solver. The
``channel_*`` functions are the :class:`~lmhdx.core3d.ChannelProblem`-native
counterparts, sharing the same linear response, reusing :class:`DuctResponse`,
:func:`pressure_drop` and :func:`hydraulic_power`, and the same segment-quantity
disclaimer above. They apply in the Stokes limit only: ``Q = G f`` holds
because the steady residual is affine in the drive when
:attr:`~lmhdx.core3d.ChannelProblem.advection` is ``"off"``, so
:func:`channel_flow_response` rejects any other value.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .core3d import ChannelProblem
from .fully_developed import channel_problem, solve_fully_developed_fields
from .specs import CaseSpec
from .steady import solve_steady_state

__all__ = [
    "DuctResponse",
    "channel_cross_section_weights",
    "channel_drive_for_flow_rate",
    "channel_fixed_flow_hydraulic_power",
    "channel_flow_rate",
    "channel_flow_response",
    "drive_for_flow_rate",
    "fluid_cell_areas",
    "hydraulic_power",
    "linear_flow_response",
    "pressure_drop",
    "volumetric_flow_rate",
]


def fluid_cell_areas(case: CaseSpec) -> jnp.ndarray:
    """Return the cross-section weights of the fluid mesh a case is solved on.

    That mesh is :func:`lmhdx.fully_developed.case_mesh`, the one the velocity of
    :func:`lmhdx.solve_fully_developed_fields` and :func:`lmhdx.solve` lives on.
    """
    return channel_cross_section_weights(channel_problem(case)).astype(case.dtype)


def channel_cross_section_weights(problem: ChannelProblem) -> jnp.ndarray:
    """Return the cross-section integration weights of a :class:`ChannelProblem`.

    Axis 0 is the flow axis of every channel this package builds (see
    :func:`lmhdx.core3d.duct_problem`), so a cell's weight is its transverse
    ``(y, z)`` area alone, independent of the axial spacing -- unlike
    :func:`fluid_cell_areas`, a channel carries no fluid mask, so every
    transverse cell counts.
    """
    _, dy, dz = problem.grid.widths
    return jnp.asarray(dy)[:, None] * jnp.asarray(dz)[None, :]


def volumetric_flow_rate(case: CaseSpec, velocity: jnp.ndarray) -> jnp.ndarray:
    """Integrate an axial velocity over the fluid cross-section."""
    areas = fluid_cell_areas(case).astype(velocity.dtype)
    if velocity.shape != areas.shape:
        raise ValueError(f"velocity shape {velocity.shape} does not match the mesh {areas.shape}")
    return jnp.sum(areas * velocity)


@dataclass(frozen=True)
class DuctResponse:
    """The linear throughput response of one duct at one field strength."""

    flow_per_unit_drive: jnp.ndarray
    magnetic_field_scale: jnp.ndarray

    def drive_for(self, target_flow_rate: float | jnp.ndarray) -> jnp.ndarray:
        """Return the drive that delivers ``target_flow_rate`` exactly."""
        return jnp.asarray(target_flow_rate) / self.flow_per_unit_drive


def linear_flow_response(case: CaseSpec, *, magnetic_field_scale: float | jnp.ndarray = 1.0) -> DuctResponse:
    """Measure ``G = Q(f = 1)``, the flow rate a unit drive produces.

    One solve determines the whole drive-to-flow relation because the problem is
    linear in the drive.
    """
    velocity, *_ = solve_fully_developed_fields(case, forcing=1.0, magnetic_field_scale=magnetic_field_scale)
    response = volumetric_flow_rate(case, velocity)
    return DuctResponse(response, jnp.asarray(magnetic_field_scale))


def drive_for_flow_rate(
    case: CaseSpec,
    target_flow_rate: float | jnp.ndarray,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the force density whose fully developed flow rate is ``target_flow_rate``."""
    return linear_flow_response(case, magnetic_field_scale=magnetic_field_scale).drive_for(target_flow_rate)


def pressure_drop(drive: float | jnp.ndarray, length: float) -> jnp.ndarray:
    """Return the pressure drop of a uniform pressure-gradient drive over ``length``."""
    if length <= 0.0:
        raise ValueError("length must be positive")
    return jnp.asarray(drive) * length


def hydraulic_power(drive: float | jnp.ndarray, flow_rate: float | jnp.ndarray, length: float) -> jnp.ndarray:
    """Return the isothermal hydraulic power of a fully developed segment.

    This is the pressure drop times the throughput. It excludes entry and exit
    losses, manifolds and every thermal effect, so it is a segment quantity and
    not a blanket pumping budget.
    """
    return pressure_drop(drive, length) * jnp.asarray(flow_rate)


def fixed_flow_hydraulic_power(
    case: CaseSpec,
    target_flow_rate: float,
    length: float,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the hydraulic power needed to hold a throughput at a given field.

    The drive is eliminated analytically, so this is a function of the design
    inputs alone and is differentiable through the solve.
    """
    response = linear_flow_response(case, magnetic_field_scale=magnetic_field_scale)
    drive = response.drive_for(target_flow_rate)
    return hydraulic_power(drive, target_flow_rate, length)


def channel_flow_rate(problem: ChannelProblem, velocity: jnp.ndarray) -> jnp.ndarray:
    """Integrate an axial velocity slice over a :class:`ChannelProblem` cross-section.

    ``velocity`` is the flow-axis component at one axial station -- every
    station carries the same value by periodicity -- shaped like
    :func:`channel_cross_section_weights`.
    """
    weights = channel_cross_section_weights(problem).astype(velocity.dtype)
    if velocity.shape != weights.shape:
        raise ValueError(f"velocity shape {velocity.shape} does not match the mesh {weights.shape}")
    return jnp.sum(weights * velocity)


def channel_flow_response(
    problem: ChannelProblem, *, magnetic_field_scale: float | jnp.ndarray = 1.0
) -> DuctResponse:
    """Measure ``G = Q(f = 1)`` on a :class:`ChannelProblem`, the flow a unit axial drive produces.

    One solve determines the whole drive-to-flow relation because the Stokes
    residual is affine in the drive; :attr:`ChannelProblem.advection` must be
    ``"off"``, since otherwise the residual carries ``-div(uu)`` and
    ``Q = G f`` breaks down.
    """
    if problem.advection != "off":
        raise ValueError(f"channel_flow_response requires advection='off', got {problem.advection!r}")
    solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), field_scale=magnetic_field_scale)
    response = channel_flow_rate(problem, solution.velocity[0].data[0])
    return DuctResponse(response, jnp.asarray(magnetic_field_scale))


def channel_drive_for_flow_rate(
    problem: ChannelProblem,
    target_flow_rate: float | jnp.ndarray,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the axial force density whose fully developed flow rate is ``target_flow_rate``."""
    return channel_flow_response(problem, magnetic_field_scale=magnetic_field_scale).drive_for(
        target_flow_rate
    )


def channel_fixed_flow_hydraulic_power(
    problem: ChannelProblem,
    target_flow_rate: float,
    length: float,
    *,
    magnetic_field_scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the hydraulic power needed to hold a throughput at a given field, on a ``ChannelProblem``.

    The drive is eliminated analytically, so this is a function of the design
    inputs alone and is differentiable through the solve.
    """
    response = channel_flow_response(problem, magnetic_field_scale=magnetic_field_scale)
    drive = response.drive_for(target_flow_rate)
    return hydraulic_power(drive, target_flow_rate, length)
