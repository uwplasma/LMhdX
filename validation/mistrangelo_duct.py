"""Mistrangelo et al. (2025), Cases 1 and 2 on the full duct (plan steps 5.4e and 5.4f, ADR 0009 D35).

The fringe, steel walls that carry current and conduct and store heat, inertia and buoyancy, with the flow
evolving through step 5.4d's open duct. :func:`case1` builds Case 1, :func:`case1_observables` gives each figure's
quantity as the paper defines it, :func:`case1_gate` applies the campaign record's rule and
:func:`case1_reported` compares the figures reported, not gated (``docs/validation/mistrangelo.md``).
:func:`case2` and :func:`case2_observables` do the same for Case 2, which is reported, not gated. The
specification, the PbLi properties pinned to it and the reference curves are :mod:`validation.mistrangelo`'s.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lmhdx import enable_x64
from lmhdx.axial import axial_faces, open_duct
from lmhdx.core3d import ChannelProblem, SolidWall, face_currents, velocity_offset, zero_velocity
from lmhdx.grid import PERIODIC, BoundaryCondition, Field, Grid, wall_resolving_faces
from lmhdx.thermal import ThermalProblem, initial_thermal_state, thermal_boundary
from validation.mistrangelo import REFERENCES, _midplane, band_misfit, field, pinned_pbli, spec

__all__ = [
    "CASE1_GATED",
    "CASE1_STATIONS",
    "CASE2_FIGURES",
    "case1",
    "case1_gate",
    "case1_observables",
    "case1_reported",
    "case2",
    "case2_observables",
]

_WALL = BoundaryCondition("neumann")


# Step 5.4e: Case 1, gated.


def case1(
    *,
    cells: int = 48,
    cells_in_layer: int = 6,
    wall_cells: int = 6,
    spacing: float = 0.005,
    span: tuple[float, float] = (-1.5, 1.5),
    core: tuple[float, float] = (-1.0, 1.0),
    dt: float = 0.01,
    record: dict | None = None,
    case: str = "case1",
) -> tuple:
    """Step 5.4e's coupled problem and its start: Case 1's full duct in SI units, temperatures in deg C.

    The field is eq. 1's, along ``y``. Each steel wall is ``wall_cells`` equal cells that carry current at the
    pinned conductivity ratio and conduct and store heat, insulated outside. The flow evolves (plan step 5.4d) with
    limited momentum transport and buoyancy, from the inlet's developed profile at ``u0`` carried along the duct.
    The heat is carried by van Leer transport; the PbLi enters at 300 C and leaves with no conductive flux, and the
    outer bottom surface takes ``q''`` over ``|x| <= 0.3`` m and its full outer width. The axial mesh has
    ``spacing`` over ``core`` and grows into the buffers out to ``span``. No Joule heat. ``case`` names the
    specification's case whose ``u0`` and ``q''`` it takes (:func:`case2`). Returns the problem and its start.
    """
    enable_x64()
    record = record or spec()
    geometry, case, heating = record["geometry"], record["cases"][case], record["heating"]
    pbli, steel = pinned_pbli(record), record["properties"]["steel"]
    a, t_w = geometry["half_width_m"], geometry["wall_thickness_m"]
    u0, q, ha = case["mean_velocity_m_per_s"], case["heat_flux_W_per_m2"], case["hartmann"]
    layers = {"cells_in_layer": cells_in_layer, "max_ratio": None}
    across = a * wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=1.0 / ha, **layers)
    along = a * wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=1.0 / np.sqrt(ha), **layers)
    edges = axial_faces(*span, core, spacing)
    if not all(np.any(np.isclose(edges, end)) for end in (heating["x_min_m"], heating["x_max_m"])):
        raise ValueError("the heated zone's ends must fall on faces of the axial mesh")
    grid = Grid(edges, across, along)
    rho, k, sigma = (
        pbli["density_kg_per_m3"],
        pbli["thermal_conductivity_W_per_m_K"],
        pbli["electrical_conductivity_S_per_m"],
    )
    k_w = steel["thermal_conductivity_W_per_m_K"]
    wall = SolidWall(
        t_w,
        wall_cells,
        k_w,
        steel["density_kg_per_m3"] * steel["heat_capacity_J_per_kg_K"],
        steel["conductivity_ratio_pinned"] * sigma,
    )
    walls = ((None, None), (wall, wall), (wall, wall))
    base = ChannelProblem(
        grid,
        (BoundaryCondition(PERIODIC), _WALL, _WALL),
        density=rho,
        viscosity=pbli["dynamic_viscosity_Pa_s"] / rho,
        conductivity=sigma,
        magnetic_field=(
            0.0,
            field(np.asarray(grid.centers[0]), record)[:, None, None] * np.ones(grid.shape),
            0.0,
        ),
        dt=dt,
        advection="limited",
        solid_walls=walls,
    )
    flow = open_duct(base, u0 * (2.0 * a) ** 2)
    x = np.asarray(grid.centers[0])
    heated = (x > heating["x_min_m"]) & (x < heating["x_max_m"])
    outer = np.where(heated, -q, 0.0)[:, None] * np.ones(
        grid.shape[1] + 2 * wall_cells
    )  # into the bottom wall
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
        expansion_coefficient=pbli["expansion_coefficient_per_K"],
        gravity=(0.0, 0.0, -record["gravity"]["acceleration_m_per_s2"]),
        advection="van_leer",
        walls=walls,
    )
    rest = zero_velocity(flow)
    inlet = jnp.asarray(flow.conditions[0].lower)
    velocity = (rest[0].replace_data(jnp.broadcast_to(inlet, rest[0].shape)), *rest[1:])
    return thermal, initial_thermal_state(thermal, t_in, velocity)


CASE1_GATED = (6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19)  # the campaign record's gated figures (ADR 0009 D35)
CASE1_STATIONS = ("0", "0.15", "0.3", "0.6", "1")  # Fig. 11's stations, x in metres, as its curves are keyed


def case1_observables(thermal, fields: dict, record: dict | None = None, case: str = "case1") -> dict:
    """Case 1's observables of time-averaged fields, as the paper defines them, by figure: ``{figure: (x, y)}``.

    ``fields`` holds the averaged velocity components ``u``, ``v`` and ``w`` on the flow's faces and ``T`` on
    ``thermal.thermal_grid``. Values at ``y = 0`` are ``a + b y^2`` through the two rows either side
    (:func:`_midplane`).
    - Figs. 6 and 7: ``T`` and ``u`` on the axis, against ``x``.
    - Figs. 9, 10, 12 and 13: ``T`` (walls included) and ``u`` against ``z``, at ``x = -0.3`` and ``0``.
    - Fig. 15: the potential, in microvolts, on the interface ``z = -a``, against ``x``. It is the potential of the
      averaged velocity, which it is linear in, gauged to zero mean over fluid and walls. The interface value is
      the first fluid cell's plus the current through the interface times its half cell's resistance. ``"15 z=+a"``
      is the same on ``z = +a``.
    - Fig. 16: the potential against ``z`` at ``x = 0``, in the fluid and on its two interfaces.
    - Figs. 17, 18 and 38: the bulk temperature (eq. 6), Nu (eq. 7, over the heated stations only) and ``T_w``.
    - Fig. 19: ``E_t`` (eq. 8), ``int(v^2 + w^2 dA) / (A u0^2)``.
    - Fig. 11: ``T`` against ``z`` at OF_KIT's five stations, keyed ``"11 x=0"`` to ``"11 x=1"``.
    - Fig. 8: the pressure on the axis against ``x``, in pascals, when ``fields`` holds ``p`` on the flow's cells.
    ``case`` names the specification's case whose ``q''`` and ``u0`` Nu and ``E_t`` take.
    """
    record = record or spec()
    a, q = record["geometry"]["half_width_m"], record["cases"][case]["heat_flux_W_per_m2"]
    u0, heating = record["cases"][case]["mean_velocity_m_per_s"], record["heating"]
    flow, tgrid, block = thermal.flow, thermal.thermal_grid, thermal.fluid_block
    grid = flow.grid
    xf, (xc, y, z) = np.asarray(grid.faces[0]), (np.asarray(c) for c in grid.centers)
    ty, tz = (np.asarray(c) for c in tgrid.centers[1:])
    u, v, w, temperature = (np.asarray(fields[k]) for k in ("u", "v", "w", "T"))
    fluid = temperature[block]
    out = {6: (xc, _midplane(_midplane(fluid, y, axis=1), z, axis=1))}
    out[7] = (xf, _midplane(_midplane(u, y, axis=1), z, axis=1))
    middle = _midplane(temperature, ty, axis=1)
    stations = ((9, -0.3), (10, 0.0)) + tuple((f"11 x={s}", float(s)) for s in CASE1_STATIONS)
    for number, at in stations:
        out[number] = (tz, np.array([np.interp(at, xc, middle[:, j]) for j in range(tz.size)]))
    if "p" in fields:
        out[8] = (xc, _midplane(_midplane(np.asarray(fields["p"]), y, axis=1), z, axis=1))
    stream = _midplane(u, y, axis=1)
    for number, at in ((12, -0.3), (13, 0.0)):
        out[number] = (z, np.array([np.interp(at, xf, stream[:, j]) for j in range(z.size)]))
    centred = 0.5 * (u[1:] + u[:-1])
    areas = np.outer(np.diff(np.asarray(grid.faces[1])), np.diff(np.asarray(grid.faces[2])))
    bulk = np.sum(centred * fluid * areas, axis=(1, 2)) / np.sum(centred * areas, axis=(1, 2))
    k, k_w = float(thermal.thermal_conductivity), thermal.walls[2][0].thermal_conductivity
    first, widths = block[2].start, np.diff(np.asarray(tgrid.faces[2]))
    g_w, g_f = k_w / (0.5 * widths[first - 1]), k / (0.5 * widths[first])
    wall = (g_w * middle[:, first - 1] + g_f * middle[:, first]) / (g_w + g_f)
    heated = (xc > heating["x_min_m"]) & (xc < heating["x_max_m"])
    out[17] = (xc, bulk)
    out[18] = (xc, np.where(heated, q * a / (k * np.where(heated, wall - bulk, 1.0)), np.nan))
    out["38 T_w"], out["38 T_b"] = (xc, wall), (xc, bulk)
    across = (0.5 * (v[:, 1:] + v[:, :-1])) ** 2 + (0.5 * (w[:, :, 1:] + w[:, :, :-1])) ** 2
    out[19] = (xc, np.sum(across * areas, axis=(1, 2)) / (np.sum(areas) * u0**2))
    velocity = tuple(
        Field(jnp.asarray(data), velocity_offset(axis), grid) for axis, data in enumerate((u, v, w))
    )
    potential, currents, _ = face_currents(velocity, flow)
    phi = _midplane(np.asarray(potential.data), y, axis=1)
    current = _midplane(np.asarray(currents[2].data), y, axis=1)
    half, sigma = 0.5 * np.diff(np.asarray(grid.faces[2])), float(flow.conductivity)
    lower = phi[:, 0] + current[:, 0] * half[0] / sigma  # the interface z = -a
    upper = phi[:, -1] - current[:, -1] * half[-1] / sigma  # and z = +a
    out[15], out["15 z=+a"] = (xc, 1e6 * lower), (xc, 1e6 * upper)
    section = np.concatenate([[lower], phi.T, [upper]]).T
    out[16] = (
        np.concatenate([[-a], z, [a]]),
        1e6 * np.array([np.interp(0.0, xc, section[:, j]) for j in range(section.shape[1])]),
    )
    return out


def _on_band(curve, figure: dict) -> tuple[np.ndarray, np.ndarray]:
    """A curve at the band's points that it covers, finite only, in the figure's plotted coordinate."""
    at, values = (np.asarray(c, float) for c in curve)
    finite = np.isfinite(values)
    at, values = at[finite], values[finite]
    x = np.asarray(figure["band"]["x"], float)
    x = x[(x >= at.min()) & (x <= at.max())]
    return x, np.interp(x, at, values)


def case1_reported(observables: dict, references: Path = REFERENCES) -> dict:
    """Case 1's reported figures, from :func:`case1_observables`, against their references; nothing is gated.

    - Fig. 8: the paper removes some codes' pressure offsets (p. 14), so the pressure meets the band after the one
      offset that best fits the band's middle. Reported: the offset, the shares inside the band as drawn and passing
      :func:`band_misfit`, the largest excess and where, and the drop over the band's span against the band's.
    - Figs. 11 and 38, OF_KIT's curves: the largest difference, where, and the mean offset. Fig. 38's are over the
      heated stations, with the largest also for ``|x| < 0.28`` m, away from the patch's edges.
    """
    figures = {entry["figure"]: entry for entry in json.loads(Path(references).read_text())["figures"]}
    result = {}
    if 8 in observables:
        x, ours = _on_band(observables[8], figures[8])
        band = {key: np.asarray(value, float) for key, value in figures[8]["band"].items()}
        lower, upper = (np.interp(x, band["x"], band[side]) for side in ("lower", "upper"))
        shifted = ours + np.mean(0.5 * (lower + upper) - ours)
        excess = np.array([band_misfit(xi, yi, figures[8]) for xi, yi in zip(x, shifted)])
        result["8"] = {
            "points": int(x.size),
            "offset": float(shifted[0] - ours[0]),
            "inside": float(np.mean((shifted >= lower) & (shifted <= upper))),
            "passing": float(np.mean(excess == 0.0)),
            "largest_excess": float(excess.max()),
            "where_largest": float(x[np.argmax(excess)]),
            "drop": float(ours[0] - ours[-1]),
            "band_drop": float(0.5 * (lower[0] + upper[0] - lower[-1] - upper[-1])),
        }
    curves = [(f"11 x={s}", 11, f"x={s}", None) for s in CASE1_STATIONS]
    curves += [(f"38 {name}", 38, f"{name} case1", 0.3) for name in ("T_w", "T_b")]
    for key, number, name, half in curves:
        cx, cy = (np.asarray(figures[number]["curves"][name][k], float) for k in ("x", "y"))
        at, values = (np.asarray(c, float) for c in observables[key])
        finite = np.isfinite(values)
        keep = (cx >= at[finite].min()) & (cx <= at[finite].max())
        if half is not None:
            keep &= np.abs(cx) <= half
        where = cx[keep]
        difference = np.interp(where, at[finite], values[finite]) - cy[keep]
        result[key] = {
            "points": int(keep.sum()),
            "largest_difference": float(np.max(np.abs(difference))),
            "where_largest": float(where[np.argmax(np.abs(difference))]),
            "mean_offset": float(np.mean(difference)),
        }
        if half is not None:
            result[key]["largest_away_from_edges"] = float(np.max(np.abs(difference[np.abs(where) < 0.28])))
    return result


def case1_gate(runs: dict, references: Path = REFERENCES) -> dict:
    """The campaign record's rule for Case 1, figure by figure (ADR 0009 D35; ``docs/validation/mistrangelo.md``).

    ``runs`` maps a run's name to ``{"mean": observables, "subs": [observables of the four sub-windows]}``, from
    :func:`case1_observables`, for M1L1, M2L1, M3L1, M1L2, M2L2, ``M1L1-dt2`` and ``M1L1-tol``. At each of a gated
    figure's band points that every run covers, in the plotted coordinate (decades on a log axis):
    - the excess is :func:`band_misfit`, outside the codes' spread widened by the digitizing uncertainty;
    - LMhdX's uncertainty ``U`` is the largest of the mesh changes (M1 to M2, M2 to M3), the outlet-length changes
      (on M1 and on M2), the time-step and tolerance changes, and twice the spread of a main run's four
      sub-window means;
    - the point passes on a run if its excess is at most ``U`` and ``U`` is at most half the band's width there,
      widened by the y uncertainty; a larger ``U`` is not resolved enough to gate.
    A figure holds if every point passes on every main run. Failures are listed, not dropped.
    """
    figures = {entry["figure"]: entry for entry in json.loads(Path(references).read_text())["figures"]}
    main = ("M1L1", "M2L1", "M3L1", "M1L2", "M2L2")
    pairs = {
        "mesh M1-M2": ("M1L1", "M2L1"),
        "mesh M2-M3": ("M2L1", "M3L1"),
        "outlet M1": ("M1L1", "M1L2"),
        "outlet M2": ("M2L1", "M2L2"),
        "time step": ("M1L1", "M1L1-dt2"),
        "tolerance": ("M1L1", "M1L1-tol"),
    }
    result = {}
    for number in CASE1_GATED:
        figure = figures[number]
        log = figure["y_scale"] == "log10"

        def plotted(values, log=log):
            return np.log10(np.maximum(values, 1e-300)) if log else values

        covered = [_on_band(runs[name]["mean"][number], figure)[0] for name in runs]
        x = np.asarray(figure["band"]["x"], float)
        x = x[np.all([np.isin(x, c) for c in covered], axis=0)]

        def at(observables, x=x, number=number):
            curve_x, curve_y = (np.asarray(c, float) for c in observables[number])
            finite = np.isfinite(curve_y)
            return np.interp(x, curve_x[finite], curve_y[finite])

        means = {name: at(runs[name]["mean"]) for name in runs}
        changes = {key: np.abs(plotted(means[b]) - plotted(means[a])) for key, (a, b) in pairs.items()}
        changes["statistics"] = np.max(
            [2.0 * np.ptp([plotted(at(sub)) for sub in runs[name]["subs"]], axis=0) for name in main], axis=0
        )
        uncertainty = np.max(list(changes.values()), axis=0)
        band = {key: np.asarray(value, float) for key, value in figure["band"].items()}
        edges = [plotted(np.interp(x, band["x"], band[side])) for side in ("lower", "upper")]
        width = 0.5 * (edges[1] - edges[0]) + figure["uncertainty"]["y"]
        resolved = uncertainty <= width
        verdict = {}
        for name in main:
            excess = np.array([band_misfit(xi, yi, figure) for xi, yi in zip(x, means[name])])
            passed = (excess <= uncertainty) & resolved
            worst = int(np.argmax(excess - uncertainty))
            verdict[name] = {
                "passed": int(passed.sum()),
                "failed": int((~passed & resolved).sum()),
                "worst_excess": float(excess[worst]),
                "worst_at": float(x[worst]),
                "worst_uncertainty": float(uncertainty[worst]),
            }
        result[number] = {
            "points": int(x.size),
            "unresolved": int((~resolved).sum()),
            "uncertainty_max": float(uncertainty.max()),
            "dominant": max(changes, key=lambda key: float(np.max(changes[key]))),
            "runs": verdict,
            "holds": bool(x.size and resolved.all() and all(v["passed"] == x.size for v in verdict.values())),
        }
    return result


# Step 5.4f: Case 2, reported.


def case2(**options) -> tuple:
    """Step 5.4f's problem and its start: :func:`case1`'s duct with Case 2's ``u0`` (0.01 m/s) and ``q''``
    (10 W/cm^2), both inferred (the specification). Its rolls about the field's axis make it time-dependent."""
    return case1(**options, case="case2")


# Case 2's figures and the same quantities' keys in case1_observables (step 5.4f).
CASE2_FIGURES = {23: 7, 24: 6, 26: 12, 27: 13, 28: 9, 29: 10, 30: 8, 31: 15, 32: "15 z=+a"}
CASE2_FIGURES |= {33: 16, 34: 17, 35: 19, 37: 18}


def case2_observables(thermal, fields: dict, record: dict | None = None) -> dict:
    """Case 2's observables by its figures (:data:`CASE2_FIGURES`), as :func:`case1_observables` defines them with
    Case 2's ``q''`` and ``u0``, and Fig. 38's ``"38 T_w"`` and ``"38 T_b"`` for its Case 2 curves."""
    same = case1_observables(thermal, fields, record, case="case2")
    out = {number: same[key] for number, key in CASE2_FIGURES.items() if key in same}
    out["38 T_w"], out["38 T_b"] = same["38 T_w"], same["38 T_b"]
    return out
