"""The 3-D validity map (step 0.7): where does the locally fully developed sum
stop matching a real 3-D solve?

Builds a periodic, insulated, square-or-beta* duct with a curl- and
divergence-free field modulation of fixed relative amplitude along the axis
(the same construction as ``examples/scratch/duct_opt_probes/
probe_axial_variation_3d.py``, generalised to a target ``gamma * sqrt(Ha)``),
solves it once as a genuine 3-D problem, and compares the result with the
locally fully developed (station) sum on the SAME axial stations. ``gamma =
a |dB/dx| / B`` is the field's relative change per half-width, at most
``delta k a`` here (stage_2_plan.txt 2.5; Tier 0 omitted the ``delta``, R1).
"""

from __future__ import annotations

import time

import jax
import numpy as np

from lmhdx.bc import PERIODIC, BoundaryCondition
from lmhdx.core3d import ChannelProblem, ImposedField
from lmhdx.design import channel_flow_response
from lmhdx.grid import Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_steady_state

_WALL = BoundaryCondition("neumann")
_DELTA = 0.10  # relative field-amplitude modulation, matching P11


def _lambda_over_a(gamma_sqrt_ha: float, ha: float) -> float:
    """``gamma = delta k a`` with ``k = 2 pi / lambda``; solve for lambda/a."""
    gamma = gamma_sqrt_ha / np.sqrt(ha)
    return 2.0 * np.pi * _DELTA / gamma


def _modulated_field(grid: Grid, ha_peak: float, lam_over_a: float, delta: float = _DELTA) -> ImposedField:
    k = 2.0 * np.pi / lam_over_a
    xf, yf = np.asarray(grid.x_faces)[:, None], np.asarray(grid.y_faces)[None, :]
    corners = -ha_peak * (xf + (delta / k) * np.sin(k * xf) * np.cosh(k * yf))
    fx = np.diff(corners, axis=1) / grid.widths[1][None, :]
    fy = -np.diff(corners, axis=0) / grid.widths[0][:, None]
    faces = tuple(np.repeat(v[:, :, None], grid.shape[2], axis=2) for v in (fx, fy))
    centres = (0.5 * (faces[0][:-1] + faces[0][1:]), 0.5 * (faces[1][:, :-1] + faces[1][:, 1:]))
    return ImposedField(grid, (*centres, np.zeros(grid.shape)), (*faces, np.zeros(grid.face_shape(2))))


def three_d_excess(
    gamma_sqrt_ha: float,
    beta: float,
    ha_base: float,
    *,
    cells: int = 24,
    cells_in_layer: int = 4,
    nx: int = 16,
) -> dict:
    """Return the 3-D excess pressure drop (%) over the locally fully developed sum.

    ``ha_base`` is the Hartmann number at the field's midplane value; the
    modulation raises it by up to ``1 + delta`` at the peak, which is what
    the mesh's Hartmann-layer clustering is built for.
    """
    ha_peak = ha_base * (1.0 + _DELTA * np.cosh(2.0 * np.pi / _lambda_over_a(gamma_sqrt_ha, ha_base)))
    lam_over_a = _lambda_over_a(gamma_sqrt_ha, ha_base)
    y = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / ha_peak, cells_in_layer=cells_in_layer, max_ratio=None
    )
    side = min(1.0 / np.sqrt(ha_base), 0.25 * beta)
    z = wall_resolving_faces(cells, -beta, beta, layer_thickness=side, cells_in_layer=cells_in_layer, max_ratio=None)
    grid = Grid(uniform_faces(nx, 0.0, lam_over_a), y, z)
    field = _modulated_field(grid, ha_base, lam_over_a)
    problem = ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), _WALL, _WALL),
        conductivity=1.0,
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    t0 = time.perf_counter()
    solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), linear_max_restarts=600)
    _, dy, dz = problem.grid.widths
    axial_velocity = np.asarray(solution.velocity[0].data)  # (nx, ny, nz)
    weights = np.asarray(dy)[:, None] * np.asarray(dz)[None, :]
    g_3d = float(np.mean([np.sum(weights * axial_velocity[i]) for i in range(nx)]))
    elapsed = time.perf_counter() - t0

    # locally fully developed sum on the same axial stations, same mesh/beta, uniform-field solves
    xc = np.asarray(grid.centers[0])
    stations = ha_base * (1.0 + _DELTA * np.cos(2.0 * np.pi / lam_over_a * xc)) / ha_peak
    fd_square = ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z),
        conditions=problem.conditions,
        conductivity=1.0,
        magnetic_field=(0.0, ha_peak, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
    q_fn = jax.jit(lambda s: channel_flow_response(fd_square, magnetic_field_scale=s).flow_per_unit_drive)
    q_stations = np.array([float(q_fn(s)) for s in stations])
    g_quasi = 1.0 / np.mean(1.0 / q_stations)

    # core-to-side-layer velocity ratio at the mid station, the M-shape diagnostic (M4, M5)
    mid = nx // 2
    u_mid = axial_velocity[mid]
    core = float(u_mid[u_mid.shape[0] // 2, u_mid.shape[1] // 2])
    side = float(np.max(u_mid[u_mid.shape[0] // 2, :]))

    return {
        "gamma_sqrt_ha": gamma_sqrt_ha,
        "beta": beta,
        "ha_base": ha_base,
        "lambda_over_a": lam_over_a,
        "nx": nx,
        "cells": cells,
        "g_3d": g_3d,
        "g_quasi_fd": g_quasi,
        "excess_percent": 100.0 * (g_quasi / g_3d - 1.0),
        "residual": float(solution.residual_norm),
        "core_over_side_velocity": core / side if side else float("nan"),
        "m_shape": core < 0.98 * side,
        "elapsed_s": elapsed,
    }


def validity_map(
    gamma_values: tuple[float, ...],
    beta_star: float,
    ha_base: float = 50.0,
    *,
    cells: int = 24,
    cells_in_layer: int = 4,
    refine_gamma: tuple[float, ...] = (),
) -> list[dict]:
    rows = []
    for gamma in gamma_values:
        for beta, label in ((1.0, "square"), (beta_star, "beta*")):
            row = three_d_excess(gamma, beta, ha_base, cells=cells, cells_in_layer=cells_in_layer, nx=16)
            row["case"] = label
            rows.append(row)
            if gamma in refine_gamma:
                fine = three_d_excess(
                    gamma, beta, ha_base, cells=cells, cells_in_layer=cells_in_layer, nx=32
                )
                fine["case"] = label
                fine["refinement_of"] = "nx16"
                rows.append(fine)
    return rows
