"""Mistrangelo et al. (2025): a heated conducting square duct in a fringing field (plan step 5.4, ADR 0009).

The frozen specification is ``src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml``. This module reads it,
derives the PbLi properties pinned to the printed dimensionless groups, and runs 5.4a's first check, Case 1's
developed flow in the uniform field. That flow is solved on a section one cell long, with the steel resolved in
cells on both transverse axes, which LMhdX supports today. The paper's developed pressure gradient, -0.0863 in
``sigma u0 B0^2`` (p. 8), then fixes the wall's conductivity ratio (:func:`infer_wall_ratio`). The ratio is
pinned in the specification, beside the measured value it differs from by 5.3 %.

``python -m validation.mistrangelo`` prints the check on three meshes, and at Matolich's measured ratio. The
reference curves it compares against are written by ``scripts/digitize_mistrangelo.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem
from lmhdx.fully_developed import channel_flow_rate
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_steady_state

__all__ = [
    "PIN_MESH",
    "REFERENCES",
    "REFERENCES_CASE2",
    "SPEC",
    "developed_gradient",
    "developed_section",
    "field",
    "infer_wall_ratio",
    "pinned_pbli",
    "section_flow",
    "spec",
]

_DATA = Path(__file__).resolve().parents[1] / "src" / "lmhdx" / "data" / "benchmarks"
SPEC = _DATA / "specs" / "mistrangelo2025.toml"
PIN_MESH = {"cells": 96, "cells_in_layer": 12, "wall_cells": 10}  # the mesh the wall's ratio is pinned on
_WALL = BoundaryCondition("neumann")


def spec() -> dict:
    """The frozen specification."""
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


def developed_gradient(record: dict | None = None) -> float:
    """The printed analytical developed gradient, as ``-dp/dx`` in ``sigma u0 B0^2`` (p. 8): 0.0863."""
    return -(record or spec())["printed"]["pressure_gradient_analytical"]


def field(x, record: dict | None = None):
    """``B_y(x)`` in tesla, eq. 1: ``B0/2 [1 - tanh((|x| - 0.7315)/0.1)]``, ``x`` in metres."""
    f = (record or spec())["field"]
    return 0.5 * f["peak_T"] * (1.0 - np.tanh((np.abs(x) - f["center_m"]) / f["width_m"]))


def pinned_pbli(record: dict | None = None) -> dict:
    """Martelli's PbLi with ``mu``, ``sigma`` and ``beta`` set to reproduce Case 1's printed Ha, Re and Gr.

    ``rho``, ``c_p`` and ``k`` are kept. With ``nu = a u0 / Re``: ``sigma = rho nu (Ha / (B0 a))^2`` and
    ``beta = Gr k nu^2 / (g q'' a^4)``. The specification records the result as ``[properties.pbli_pinned]``.
    """
    record = record or spec()
    pbli, case = dict(record["properties"]["pbli"]), record["cases"]["case1"]
    a, u0, q = record["geometry"]["half_width_m"], case["mean_velocity_m_per_s"], case["heat_flux_W_per_m2"]
    nu = a * u0 / case["reynolds"]
    pbli["dynamic_viscosity_Pa_s"] = pbli["density_kg_per_m3"] * nu
    pbli["electrical_conductivity_S_per_m"] = (
        pbli["dynamic_viscosity_Pa_s"] * (case["hartmann"] / (record["field"]["peak_T"] * a)) ** 2
    )
    pbli["expansion_coefficient_per_K"] = (
        case["grashof"]
        * pbli["thermal_conductivity_W_per_m_K"]
        * nu**2
        / (record["gravity"]["acceleration_m_per_s2"] * q * a**4)
    )
    return pbli


def developed_section(
    ratio: float, *, cells: int = 72, cells_in_layer: int = 9, wall_cells: int = 8, record: dict | None = None
) -> ChannelProblem:
    """Case 1's section in the uniform field, one cell long and periodic along the flow, unit half-width.

    Nondimensional as ``core3d.duct_problem``: density, viscosity and fluid conductivity one, the field
    ``Ha`` along ``y`` and a unit pressure gradient. Each wall is ``t_w / a`` thick in ``wall_cells`` equal cells
    of conductivity ``ratio``, insulated outside. Float64 is turned on, as the steady solve needs it.
    """
    enable_x64()
    record = record or spec()
    ha = record["cases"]["case1"]["hartmann"]
    thickness = record["geometry"]["wall_thickness_m"] / record["geometry"]["half_width_m"]
    across = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=cells_in_layer, max_ratio=None
    )
    along = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(ha), cells_in_layer=cells_in_layer, max_ratio=None
    )
    layer = ((float(ratio),) * wall_cells, (thickness / wall_cells,) * wall_cells)
    return ChannelProblem(
        Grid(uniform_faces(1, 0.0, 1.0), across, along),
        (BoundaryCondition(PERIODIC), _WALL, _WALL),
        magnetic_field=(0.0, ha, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_layers=(None, (layer, layer), (layer, layer)),
    )


def _peak(z, u) -> tuple[float, float]:
    """The maximum of ``u(z)`` and where it is: the parabola through the largest sample and its two neighbours."""
    i = int(np.clip(np.argmax(u), 1, len(u) - 2))
    c2, c1, c0 = np.polyfit(z[i - 1 : i + 2], u[i - 1 : i + 2], 2)
    at = -c1 / (2.0 * c2)
    return float(np.polyval((c2, c1, c0), at)), float(at)


def section_flow(ratio: float, *, record: dict | None = None, **mesh) -> dict:
    """The developed section's ``-dp/dx`` in ``sigma u0 B0^2``, and its profile ``u/u0`` along ``z`` at ``y = 0``.

    In the section's units ``sigma u0 B0^2`` is ``u0 Ha^2``, so the gradient is ``1 / (u0 Ha^2)``. The profile at
    ``y = 0`` is ``a + b y^2`` through the two rows nearest the midplane on each side, averaged across it; the jets'
    height and position are the parabola through each side's largest sample and its neighbours.
    """
    record = record or spec()
    problem = developed_section(ratio, record=record, **mesh)
    velocity = np.asarray(solve_steady_state(problem).velocity[0].data[0])
    ha = record["cases"]["case1"]["hartmann"]
    across, along = (np.asarray(f) for f in problem.grid.faces[1:])
    mean = float(channel_flow_rate(problem, velocity)) / float(np.ptp(across) * np.ptp(along))
    y, z = 0.5 * (across[1:] + across[:-1]), 0.5 * (along[1:] + along[:-1])
    (j1, j2), (k1, k2) = (
        np.argsort(np.where(y > 0, y, np.inf))[:2],
        np.argsort(np.where(y < 0, -y, np.inf))[:2],
    )
    inner, outer = 0.5 * (velocity[j1] + velocity[k1]), 0.5 * (velocity[j2] + velocity[k2])
    line = (inner * y[j2] ** 2 - outer * y[j1] ** 2) / (y[j2] ** 2 - y[j1] ** 2) / mean
    jets = [_peak(z[side], line[side]) for side in (z < 0, z > 0)]
    return {
        "gradient": 1.0 / (mean * ha**2),
        "z": z,
        "u_over_u0": line,
        "centre": float(np.interp(0.0, z, line)),
        "jet": float(np.mean([height for height, _ in jets])),
        "jet_z": float(np.mean([abs(at) for _, at in jets])),
    }


def infer_wall_ratio(
    start: tuple[float, float] = (1.18, 1.30), steps: int = 4, tolerance: float = 1e-10, record=None, **mesh
) -> float:
    """The wall's effective conductivity ratio at which the section meets the printed -0.0863, by the secant method.

    On ``PIN_MESH`` unless another mesh is given. The gradient is close to linear in the ratio, about 0.055 per
    unit on 72/9.
    """
    record, mesh = record or spec(), mesh or PIN_MESH
    target = developed_gradient(record)

    def residual(ratio):
        return section_flow(ratio, record=record, **mesh)["gradient"] - target

    (a, b), fa = start, residual(start[0])
    fb = residual(b)
    for _ in range(steps):
        if abs(fb) < tolerance or fb == fa:
            break
        a, fa, b = b, fb, b - fb * (b - a) / (fb - fa)
        fb = residual(b)
    return float(b)


# The reference curves of plan step 5.4a, written by ``scripts/digitize_mistrangelo.py`` (ADR 0009 D38).

REFERENCES = _DATA / "references" / "mistrangelo2025-case1.json"
REFERENCES_CASE2 = _DATA / "references" / "mistrangelo2025-case2.json"


def main() -> None:
    record = spec()
    ratio, target = record["properties"]["steel"]["conductivity_ratio_pinned"], developed_gradient(record)
    for cells, layer, walls in ((48, 6, 6), (72, 9, 8), (96, 12, 10)):
        flow = section_flow(ratio, record=record, cells=cells, cells_in_layer=layer, wall_cells=walls)
        print(f"{cells}/{layer}, {walls} wall cells: -dp/dx {flow['gradient']:.5f} (printed {target}); "
              f"jet {flow['jet']:.3f} u0 at |z|/a {flow['jet_z']:.4f}; centre {flow['centre']:.4f} u0")  # fmt: skip
    measured = record["properties"]["steel"]["electrical_conductivity_S_per_m"]
    measured /= pinned_pbli(record)["electrical_conductivity_S_per_m"]
    flow = section_flow(measured, record=record, **PIN_MESH)
    print(f"Matolich's ratio {measured:.4f} on 96/12: -dp/dx {flow['gradient']:.5f}")


if __name__ == "__main__":
    main()
