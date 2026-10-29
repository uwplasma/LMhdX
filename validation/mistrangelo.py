"""Mistrangelo et al. (2025): a heated conducting square duct in a fringing field (plan step 5.4, ADR 0009).

The frozen specification is ``src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml``. This module reads it,
derives the PbLi properties pinned to the printed dimensionless groups, and runs 5.4a's first check, Case 1's
developed flow in the uniform field. That flow is solved on a section one cell long, with the steel resolved in
cells on both transverse axes, which LMhdX supports today. The paper's developed pressure gradient, -0.0863 in
``sigma u0 B0^2`` (p. 8), then fixes the wall's conductivity ratio (:func:`infer_wall_ratio`). The ratio is
pinned in the specification, beside the measured value it differs from by 5.3 %.

``python -m validation.mistrangelo`` prints the check on three meshes, and at Matolich's measured ratio. The
reference curves it compares against are written by ``scripts/digitize_mistrangelo.py``.
``python -m validation.mistrangelo heated DIR`` runs step 5.4c, Case 1's heated zone with its developed flow
held, conjugate walls and no buoyancy, and writes its record in ``DIR`` (:func:`write_heated_zone`).
"""

from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem, SolidWall, zero_velocity
from lmhdx.fully_developed import channel_flow_rate
from lmhdx.grid import (
    CENTER,
    DIRICHLET,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.steady import solve_steady_state
from lmhdx.thermal import (
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    initial_thermal_state,
    thermal_boundary,
    validate_thermal_inputs,
)

__all__ = [
    "PIN_MESH",
    "REFERENCES",
    "REFERENCES_CASE2",
    "SPEC",
    "band_misfit",
    "developed_gradient",
    "developed_section",
    "field",
    "heated_zone",
    "heated_zone_comparison",
    "heated_zone_observables",
    "infer_wall_ratio",
    "pinned_pbli",
    "run_heated_zone",
    "section_flow",
    "spec",
    "write_heated_zone",
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


def band_misfit(x, y, figure: dict) -> float:
    """How far ``y(x)`` lies outside a figure's acceptance band: the codes' spread widened by the record's digitizing
    uncertainty in both axes (ADR 0009 D35). Arrays and numbers in, the largest excess out, zero inside.

    The band is interpolated in the plotted coordinates, as it was drawn and compacted: on a log axis in decades,
    where a value that is not positive and finite is refused, and the excess is in decades. Each edge is taken at its
    extreme over ``x`` plus or minus the x uncertainty (the window kept inside the band), then widened by the y
    uncertainty. Points outside the band's x range are refused.
    """
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    band = {key: np.asarray(band, float) for key, band in figure["band"].items()}
    if np.any((x < band["x"][0]) | (x > band["x"][-1])):
        raise ValueError(f"x outside Figure {figure['figure']}'s band, {band['x'][0]} to {band['x'][-1]}")
    lower, upper = band["lower"], band["upper"]
    if figure["y_scale"] == "log10":
        if not all(np.all(np.isfinite(v) & (v > 0.0)) for v in (lower, upper, y)):
            raise ValueError(f"Figure {figure['figure']}'s log axis takes positive, finite values only")
        y, lower, upper = np.log10(y), np.log10(lower), np.log10(upper)
    across, along = figure["uncertainty"]["x"], figure["uncertainty"]["y"]
    excess = 0.0
    for at, value in zip(x, y):
        ends = np.clip([at - across, at + across], band["x"][0], band["x"][-1])
        window = np.concatenate([ends, band["x"][(band["x"] > ends[0]) & (band["x"] < ends[1])]])
        low, high = np.min(np.interp(window, band["x"], lower)), np.max(np.interp(window, band["x"], upper))
        excess = max(excess, low - along - value, value - high - along)
    return float(excess)


def _peak(z, u) -> tuple[float, float]:
    """The maximum of ``u(z)`` and where it is: the parabola through the largest sample and its two neighbours."""
    i = int(np.clip(np.argmax(u), 1, len(u) - 2))
    c2, c1, c0 = np.polyfit(z[i - 1 : i + 2], u[i - 1 : i + 2], 2)
    at = -c1 / (2.0 * c2)
    return float(np.polyval((c2, c1, c0), at)), float(at)


def _midplane(values: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    """``values`` at ``y = 0`` along ``axis``: ``a + b y^2`` through the two rows nearest the midplane on each side,
    averaged across it, which an even cell count has no row on."""
    (j1, j2), (k1, k2) = (
        np.argsort(np.where(y > 0, y, np.inf))[:2],
        np.argsort(np.where(y < 0, -y, np.inf))[:2],
    )
    inner = 0.5 * (np.take(values, j1, axis=axis) + np.take(values, k1, axis=axis))
    outer = 0.5 * (np.take(values, j2, axis=axis) + np.take(values, k2, axis=axis))
    return (inner * y[j2] ** 2 - outer * y[j1] ** 2) / (y[j2] ** 2 - y[j1] ** 2)


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
    line = _midplane(velocity, y) / mean
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


# Step 5.4c: Case 1's heated zone with its developed flow held, conjugate walls and no buoyancy (reported).


def heated_zone(
    *,
    cells: int = 48,
    cells_in_layer: int = 6,
    wall_cells: int = 6,
    axial: int = 160,
    span: tuple[float, float] = (-0.4, 0.4),
    courant: float = 0.5,
    record: dict | None = None,
) -> tuple:
    """Step 5.4c's thermal problem and held velocity: Case 1's heated zone in SI units, temperatures in deg C.

    The flow is the developed section at the pinned wall ratio (:func:`developed_section`), scaled to the mean
    velocity ``u0`` and the half-width ``a``, and the same at every station of ``span``: no fringe and no buoyancy.
    The steel conducts and stores heat in ``wall_cells`` cells per wall; the fluid enters at 300 C, leaves with no
    conductive flux, and the outer bottom surface takes ``q''`` over ``|x| <= 0.3`` m and its full outer width.
    ``axial`` must put the heated zone's ends on faces. Van Leer transport at ``courant``; no Joule heat.
    """
    record = record or spec()
    geometry, case, heating = record["geometry"], record["cases"]["case1"], record["heating"]
    pbli, steel = pinned_pbli(record), record["properties"]["steel"]
    a, u0, q = geometry["half_width_m"], case["mean_velocity_m_per_s"], case["heat_flux_W_per_m2"]
    section = developed_section(
        steel["conductivity_ratio_pinned"],
        cells=cells,
        cells_in_layer=cells_in_layer,
        wall_cells=wall_cells,
        record=record,
    )
    profile = np.asarray(solve_steady_state(section).velocity[0].data[0])
    across, along = (np.asarray(faces) for faces in section.grid.faces[1:])
    inlet = u0 * profile * np.ptp(across) * np.ptp(along) / float(channel_flow_rate(section, profile))
    grid = Grid(uniform_faces(axial, *span), a * across, a * along)
    x = np.asarray(grid.centers[0])
    edges = np.asarray(grid.faces[0])
    if not all(np.any(np.isclose(edges, end)) for end in (heating["x_min_m"], heating["x_max_m"])):
        raise ValueError("the heated zone's ends must fall on faces of the axial mesh")
    rho, k = pbli["density_kg_per_m3"], pbli["thermal_conductivity_W_per_m_K"]
    flow = ChannelProblem(
        grid,
        (BoundaryCondition(DIRICHLET, lower=inlet, upper_kind=NEUMANN), _WALL, _WALL),
        density=rho,
        viscosity=pbli["dynamic_viscosity_Pa_s"] / rho,
        conductivity=0.0,  # the held flow carries the heat; with no Joule heat its currents are not needed
        dt=courant
        * float(np.min(np.diff(edges)))
        / float(np.max(inlet))
        * (1.0 - 1e-12),  # under, by rounding
    )
    k_w = steel["thermal_conductivity_W_per_m_K"]
    wall = SolidWall(
        geometry["wall_thickness_m"],
        wall_cells,
        k_w,
        steel["density_kg_per_m3"] * steel["heat_capacity_J_per_kg_K"],
    )
    heated = (x > heating["x_min_m"]) & (x < heating["x_max_m"])
    # Into the bottom wall, over its full outer width.
    outer = np.where(heated, -q, 0.0)[:, None] * np.ones(grid.shape[1] + 2 * wall_cells)
    t_in = heating["inlet_temperature_C"]
    conditions = (
        thermal_boundary("temperature", t_in, upper_kind="heat_flux", conductivity=k),
        thermal_boundary("heat_flux", conductivity=k_w),
        thermal_boundary("heat_flux", outer, 0.0, conductivity=k_w),
    )
    thermal = ThermalProblem(
        flow,
        conditions,
        pbli["heat_capacity_J_per_kg_K"],
        k,
        heating["reference_temperature_C"],
        joule_heating=False,
        advection="van_leer",
        walls=((None, None), (wall, wall), (wall, wall)),
    )
    rest = zero_velocity(flow)
    velocity = (rest[0].replace_data(jnp.broadcast_to(jnp.asarray(inlet), rest[0].shape)), *rest[1:])
    return thermal, velocity


def run_heated_zone(
    thermal: ThermalProblem, velocity, *, chunk: int = 200, tolerance: float = 1e-4, max_chunks: int = 100
) -> tuple:
    """March the heated zone from 300 C to its steady state, until the storage rate is ``tolerance`` of the heat
    conducted in; return the state and the last diagnostics of each chunk of ``chunk`` steps, with its seconds
    and whether the run has converged, which the last says."""
    thermal_step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    state = initial_thermal_state(thermal, thermal.reference_temperature, velocity)
    heat = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)
    validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-12)
    advance = jax.jit(lambda start: advance_thermal(thermal_step, start, heat, chunk, diagnostics=chunk))
    history = []
    for _ in range(max_chunks):
        begin = time.perf_counter()
        state, diagnostics = jax.block_until_ready(advance(state))
        history.append({name: float(np.asarray(value)[-1]) for name, value in diagnostics._asdict().items()})
        history[-1]["seconds"] = time.perf_counter() - begin
        last = history[-1]
        if not last["finite"]:
            raise FloatingPointError("the heated zone's run is not finite")
        last["converged"] = abs(last["storage_rate"]) <= tolerance * abs(last["conductive_heat_out"])
        if last["converged"]:
            break
    return state, history


def heated_zone_observables(thermal, state, record: dict | None = None) -> dict:
    """Case 1's observables of the heated zone, as the paper defines them (eqs. 6 and 7).

    ``T(z)`` at ``y = 0``, ``a + b y^2`` through the two rows either side, at ``x = -0.3`` and ``0`` (Figs. 9
    and 10); the bulk temperature of the fluid, weighted by the axial velocity (Fig. 17); ``T_w``, on the
    interface ``z = -a`` at ``y = 0``, where the two half cells' conductances meet; and
    ``Nu = q'' a / (k (T_w - T_b))`` (Figs. 18, 38), over the heated stations only (NaN elsewhere, where no
    flux defines it).
    """
    record = record or spec()
    a, q = record["geometry"]["half_width_m"], record["cases"]["case1"]["heat_flux_W_per_m2"]
    grid, block, heating = thermal.thermal_grid, thermal.fluid_block, record["heating"]
    temperature, x, y, z = np.asarray(state.temperature.data), *(np.asarray(c) for c in grid.centers)
    heated = (x > heating["x_min_m"]) & (x < heating["x_max_m"])
    midplane = _midplane(temperature, y, axis=1)
    u = np.asarray(state.velocity[0].data)
    u = 0.5 * (u[1:] + u[:-1])
    fluid = temperature[block]
    areas = np.outer(*(np.diff(np.asarray(f))[b] for f, b in zip(grid.faces[1:], block[1:])))
    bulk = np.sum(u * fluid * areas, axis=(1, 2)) / np.sum(u * areas, axis=(1, 2))
    k = float(thermal.thermal_conductivity)
    k_w = thermal.walls[2][0].thermal_conductivity
    first, widths = block[2].start, np.diff(np.asarray(grid.faces[2]))
    g_w, g_f = k_w / (0.5 * widths[first - 1]), k / (0.5 * widths[first])
    wall = (g_w * midplane[:, first - 1] + g_f * midplane[:, first]) / (g_w + g_f)
    station = {
        name: np.array([np.interp(at, x, midplane[:, j]) for j in range(z.size)])
        for name, at in (("-0.3", -0.3), ("0", 0.0))
    }
    return {
        "x": x,
        "z": z,
        "T_z": station,
        "T_b": bulk,
        "T_w": wall,
        "Nu": np.where(heated, q * a / (k * np.where(heated, wall - bulk, 1.0)), np.nan),
    }


def heated_zone_comparison(observables: dict, references: Path = REFERENCES) -> dict:
    """The heated zone's curves against Case 1's digitized references, Figures 9, 10, 17, 18 and 38.

    Per figure, at the band's points inside the zone: the fraction inside the agreeing codes' band, as drawn and
    widened by half a pixel of the figure (its bands are about a pixel wide), the largest distance outside it,
    in the figure's units and pixels, and the mean offset from its middle. Fig. 38 is one code's curves,
    OF_KIT's: the largest difference and the mean offset, in units and pixels.
    """
    figures = {entry["figure"]: entry for entry in json.loads(Path(references).read_text())["figures"]}
    x, z = observables["x"], observables["z"]
    ours = {9: (z, observables["T_z"]["-0.3"]), 10: (z, observables["T_z"]["0"])}
    ours |= {17: (x, observables["T_b"]), 18: (x, observables["Nu"])}
    result = {}
    for number, (at, values) in ours.items():
        band = {name: np.asarray(value) for name, value in figures[number]["band"].items()}
        pixel = figures[number]["per_pixel"][1]
        defined = np.isfinite(values)
        at, values = at[defined], values[defined]
        keep = (band["x"] >= at.min()) & (band["x"] <= at.max())
        mine, lower, upper = np.interp(band["x"][keep], at, values), band["lower"][keep], band["upper"][keep]
        outside = np.maximum(np.maximum(lower - mine, mine - upper), 0.0)
        result[str(number)] = {
            "points": int(keep.sum()),
            "inside": float(np.mean(outside == 0.0)),
            "inside_half_pixel": float(np.mean(outside <= 0.5 * pixel)),
            "largest_outside": float(outside.max()),
            "largest_outside_px": float(outside.max() / pixel),
            "where_largest": float(band["x"][keep][np.argmax(outside)]),
            "mean_offset": float(np.mean(mine - 0.5 * (lower + upper))),
        }
    for name in ("T_w", "T_b"):
        curve = {
            key: np.asarray(value)
            for key, value in figures[38]["curves"][f"{name} case1"].items()
            if key != "coverage"
        }
        keep = (curve["x"] >= x.min()) & (curve["x"] <= x.max())
        difference = np.interp(curve["x"][keep], x, observables[name]) - curve["y"][keep]
        result[f"38 {name}"] = {
            "points": int(keep.sum()),
            "largest_difference": float(np.max(np.abs(difference))),
            "largest_difference_px": float(np.max(np.abs(difference)) / figures[38]["per_pixel"][1]),
            "mean_offset": float(np.mean(difference)),
        }
    return result


def write_heated_zone(
    directory: Path, cells: int = 48, cells_in_layer: int = 6, wall_cells: int = 6, axial: int = 160
) -> Path:
    """Run step 5.4c to its steady state and write ``heated_zone.json`` in ``directory``.

    It holds the mesh, the step, the run's cost and last diagnostics, the curves of :func:`heated_zone_observables`
    and their :func:`heated_zone_comparison`.
    """
    begin = time.perf_counter()
    thermal, velocity = heated_zone(
        cells=cells, cells_in_layer=cells_in_layer, wall_cells=wall_cells, axial=axial
    )
    chunk, tolerance = 200, 1e-4
    state, history = run_heated_zone(thermal, velocity, chunk=chunk, tolerance=tolerance)
    seconds = time.perf_counter() - begin
    observables = heated_zone_observables(thermal, state)
    curves = {
        name: [float(v) if np.isfinite(v) else None for v in np.asarray(value)]
        for name, value in observables.items()
        if name != "T_z"
    }
    curves["T_z"] = {name: value.tolist() for name, value in observables["T_z"].items()}
    record = {
        "step": "5.4c",
        "mesh": {"cells": cells, "cells_in_layer": cells_in_layer, "wall_cells": wall_cells, "axial": axial},
        "cells": int(np.prod(thermal.thermal_grid.shape)),
        "dt_s": float(thermal.flow.dt),
        "chunks": len(history),
        "seconds": seconds,
        "seconds_per_step_warm": history[-1]["seconds"] / chunk,
        "chunk": chunk,
        "tolerance": tolerance,
        "converged": history[-1]["converged"],
        "backend": jax.default_backend(),
        "last": history[-1],
        "comparison": heated_zone_comparison(observables),
        "curves": curves,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "heated_zone.json"
    path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    return path


def main() -> None:
    if sys.argv[1:2] == ["heated"]:  # heated OUT [cells layer walls axial]: step 5.4c, its record in OUT
        print(write_heated_zone(Path(sys.argv[2]), *map(int, sys.argv[3:7])))
        return
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
