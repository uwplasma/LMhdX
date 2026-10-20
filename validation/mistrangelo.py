"""Mistrangelo et al. (2025): a heated conducting square duct in a fringing field (plan step 5.4, ADR 0009).

The frozen specification is ``src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml``. This module reads it,
derives the PbLi properties pinned to the printed dimensionless groups, and runs 5.4a's first check, Case 1's
developed flow in the uniform field. That flow is solved on a section one cell long, with the steel resolved in
cells on both transverse axes, which LMhdX supports today. The paper's developed pressure gradient, -0.0863 in
``sigma u0 B0^2`` (p. 8), then fixes the wall's conductivity ratio (:func:`infer_wall_ratio`). The ratio is
pinned in the specification, beside the measured value it differs from by 5.3 %.

``python -m validation.mistrangelo`` prints the check on three meshes.
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

__all__ = ["SPEC", "developed_section", "field", "infer_wall_ratio", "pinned_pbli", "section_flow", "spec"]

_DATA = Path(__file__).resolve().parents[1] / "src" / "lmhdx" / "data" / "benchmarks"
SPEC = _DATA / "specs" / "mistrangelo2025.toml"
DEVELOPED_GRADIENT = 0.0863  # printed, p. 8: the analytical developed flow, -dp/dx in sigma u0 B0^2
_WALL = BoundaryCondition("neumann")


def spec() -> dict:
    """The frozen specification."""
    return tomllib.loads(SPEC.read_text())


def field(x, record: dict | None = None):
    """``B_y(x)`` in tesla, eq. 1: ``B0/2 [1 - tanh((|x| - 0.7315)/0.1)]``, ``x`` in metres."""
    f = (record or spec())["field"]
    return 0.5 * f["peak_T"] * (1.0 - np.tanh((np.abs(x) - f["center_m"]) / f["width_m"]))


def pinned_pbli(record: dict | None = None) -> dict:
    """Martelli's PbLi with ``mu``, ``sigma`` and ``beta`` set to reproduce Case 1's printed Ha, Re and Gr.

    ``rho``, ``c_p`` and ``k`` are kept. With ``nu = a u0 / Re``: ``sigma = rho nu (Ha / (B0 a))^2`` and
    ``beta = Gr k nu^2 / (g q'' a^4)``.
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
        wall_layers=(None, (layer, layer), (layer, layer)),
    )


def section_flow(ratio: float, **mesh) -> dict:
    """The developed section's ``-dp/dx`` in ``sigma u0 B0^2``, and its profile ``u/u0`` along ``z`` at ``y = 0``.

    In the section's units ``sigma u0 B0^2`` is ``u0 Ha^2``, so the gradient is ``1 / (u0 Ha^2)``.
    """
    record = mesh.pop("record", None) or spec()
    problem = developed_section(ratio, record=record, **mesh)
    velocity = solve_steady_state(problem).velocity[0].data[0]
    ha = record["cases"]["case1"]["hartmann"]
    mean = float(channel_flow_rate(problem, velocity)) / 4.0
    y, z = (0.5 * (np.asarray(f)[1:] + np.asarray(f)[:-1]) for f in problem.grid.faces[1:])
    middle = int(np.argmin(np.abs(y)))
    return {"gradient": 1.0 / (mean * ha**2), "z": z, "u_over_u0": np.asarray(velocity)[middle] / mean}


def infer_wall_ratio(start: tuple[float, float] = (1.18, 1.30), steps: int = 3, **mesh) -> float:
    """The wall's conductivity ratio at which the section reproduces the printed -0.0863, by the secant method.

    The gradient is close to linear in the ratio (0.0826, 0.0892 and 0.0974 at 1.18, 1.30 and 1.45 on 72/9).
    """
    (a, b), residual = start, lambda r: section_flow(r, **mesh)["gradient"] - DEVELOPED_GRADIENT
    fa, fb = residual(a), residual(b)
    for _ in range(steps):
        a, fa, b = b, fb, b - fb * (b - a) / (fb - fa)
        fb = residual(b)
    return float(b)


def main() -> None:
    record = spec()
    ratio = record["properties"]["steel"]["conductivity_ratio_pinned"]
    for cells, layer, walls in ((48, 6, 6), (72, 9, 8), (96, 12, 10)):
        flow = section_flow(ratio, cells=cells, cells_in_layer=layer, wall_cells=walls)
        u = flow["u_over_u0"]
        print(f"{cells}/{layer}, {walls} wall cells: -dp/dx {flow['gradient']:.5f} (printed {DEVELOPED_GRADIENT}); "
              f"jet {u.max():.3f} u0 at |z|/a {abs(flow['z'][np.argmax(u)]):.3f}; centre {np.interp(0.0, flow['z'], u):.4f} u0")  # fmt: skip


if __name__ == "__main__":
    main()
