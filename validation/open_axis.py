"""How far a locally fully developed duct holds along a field that varies: the open-axis excess.

A duct design that sums fully developed pressure gradients station by station
(each at the local field) ignores the axial currents a varying field drives.
This check measures what that sum misses. A straight, insulated rectangular
duct with an inlet and an outlet (:mod:`lmhdx.axial`) runs through a field that
falls monotonically by a fraction ``delta`` of its mid value over
``|x| <= x0``. The 3-D pressure drop across the ramp is compared with the
fully developed gradient at the local field, integrated over the same length
on the same cross-section. The excess ``Delta p / Delta p_FD - 1`` is the
error of the station sum.

The steepest relative gradient of the field, per duct half-width, is
``gamma = delta * pi / (4 x0)``. The excess is reported against
``gamma * sqrt(Ha)``, the ratio of the side-layer thickness to the length over
which the field changes, which is the variable it collapses on across Hartmann
numbers (Stage 2 of the duct design study measured it at Ha 50 and 200).
"""

from __future__ import annotations

import time

import jax
import numpy as np

from lmhdx.axial import axial_faces, charge_balance, mass_balance, open_duct, pressure_drop, solve_open_duct
from lmhdx.core3d import ChannelProblem, ImposedField, fringe_field
from lmhdx.fully_developed import channel_flow_response
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces

__all__ = ["ramp_excess"]

_WALL = BoundaryCondition("neumann")


def ramp_excess(
    gamma_sqrt_ha: float,
    beta: float,
    ha_mid: float,
    *,
    delta: float = 0.2,
    cells: int = 24,
    cells_in_layer: int = 4,
    spacing: float = 0.25,
    window: float = 6.0,
    upstream: float = 15.0,
    downstream: float = 10.0,
) -> dict:
    """Return the 3-D excess (%) over the locally fully developed sum, and the solve's checks.

    The duct has half-width 1 along the field and ``beta`` across it. ``B_y(x)``
    falls by ``delta`` (relative to the mid-field Hartmann number ``ha_mid``)
    over ``|x| <= x0`` as :func:`lmhdx.core3d.fringe_field` (``B_y`` alone,
    divergence free) added to a uniform field, between buffers of ``upstream``
    and ``downstream`` half-widths; ``x0`` follows from ``gamma * sqrt(Ha)``.
    The excess is taken over ``[-x0 - window, x0 + window]``, with the fully
    developed gradient integrated by 24-point Gauss-Legendre quadrature over
    the ramp.
    """
    x0 = delta * np.pi * np.sqrt(ha_mid) / (4.0 * gamma_sqrt_ha)
    b_lo, d_b = ha_mid * (1.0 - delta / 2.0), ha_mid * delta
    ha_hi = b_lo + d_b
    y = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / ha_hi, cells_in_layer=cells_in_layer, max_ratio=None
    )
    z = wall_resolving_faces(
        cells,
        -beta,
        beta,
        layer_thickness=min(1.0 / np.sqrt(ha_mid), 0.25 * beta),
        cells_in_layer=cells_in_layer,
        max_ratio=None,
    )
    step = float(np.clip(spacing, x0 / 40.0, x0 / 5.0))  # 5-40 cells across the half-ramp
    grid = Grid(axial_faces(-x0 - upstream, x0 + downstream, (-x0 - 1.0, x0 + 1.0), step), y, z)
    ramp = fringe_field(grid, half_length=x0, strength=d_b, solenoidal=False)
    faces = (ramp.faces[0], ramp.faces[1] + b_lo, ramp.faces[2])
    components = (ramp.components[0], ramp.components[1] + b_lo, ramp.components[2])
    walled = ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), _WALL, _WALL),
        conductivity=1.0,
        magnetic_field=ImposedField(grid, components, faces),
        dt=1.0,
    )
    problem = open_duct(walled, 1.0)
    t0 = time.perf_counter()
    solution = solve_open_duct(problem)
    jax.block_until_ready(solution.velocity)
    elapsed = time.perf_counter() - t0
    xa, xb = -x0 - window, x0 + window
    dp = float(pressure_drop(solution.pressure, xa, xb))

    section = ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=walled.conditions,
        conductivity=1.0,
        magnetic_field=(0.0, ha_hi, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    q_fn = jax.jit(lambda s: channel_flow_response(section, magnetic_field_scale=s).flow_per_unit_drive)

    def gradient(b):
        """The fully developed ``|dp/dx|`` at field ``b`` and unit flow rate."""
        return 1.0 / float(q_fn(b / ha_hi))

    nodes, weights = np.polynomial.legendre.leggauss(24)
    s = x0 * nodes
    field = b_lo + d_b * (1.0 - np.sin(np.pi * s / (2.0 * x0))) / 2.0
    dp_fd = (
        gradient(ha_hi) * window
        + x0 * sum(w * gradient(b) for w, b in zip(weights, field))
        + gradient(b_lo) * window
    )
    return {
        "gamma_sqrt_ha": gamma_sqrt_ha,
        "beta": beta,
        "ha_mid": ha_mid,
        "delta": delta,
        "x0": x0,
        "cells": cells,
        "axial_cells": grid.shape[0],
        "spacing": step,
        "window": window,
        "upstream": upstream,
        "downstream": downstream,
        "dp": dp,
        "dp_fd": dp_fd,
        "excess_percent": 100.0 * (dp / dp_fd - 1.0),
        "iterations": int(solution.iterations),
        "residual": float(solution.residual_norm / solution.initial_residual_norm),
        "mass_balance": float(mass_balance(solution.velocity)),
        "charge_balance": float(charge_balance(solution, problem)),
        "elapsed_s": elapsed,
    }
