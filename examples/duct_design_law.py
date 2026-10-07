"""Find the aspect ratio that minimizes an insulated duct's pumping power at a fixed flow.

Edit the inputs below, then run ``python examples/duct_design_law.py``.

A straight, insulated rectangular duct of half-width ``a`` along the field and
``b = beta * a`` across it carries a fixed flow ``Q``. At a fixed cross-section
area its pumping power ``Q * Delta p`` is lowest at one aspect ratio ``beta*``,
deep along the field and narrow across it. Two results follow:

1. The design law. At a fixed area, with ``H`` the Hartmann number of the
   equal-area square duct, ``s* = beta* * sqrt(Ha*)`` tends to about 2.08 as
   ``H`` grows (``Ha*`` is the optimum's own, on its half-width ``a``), near the
   2.5 * 0.852 = 2.13 of the thin-layer friction law for insulated ducts.
2. One design. A PbLi duct running radially through an outboard blanket in a
   1/R toroidal field. The pumping power falls as the duct grows, so the
   smallest allowed velocity sets the area, which the sign of
   ``d Delta p / d ln A`` confirms. The aspect ratio is then the one free
   choice, and it is compared with the square duct of the same area.

Each fully developed solve gives the flow per unit drive ``q(Ha, beta)`` of a
unit duct. Each trial ``beta`` scales one mesh across the field, so the
discrete pressure drop is smooth in ``beta``, a quartic through five trials
locates its minimum, and every trial shares one compiled program. The
Hartmann number enters the solve as a field scale, so ``jax.grad`` gives the
derivative of the pressure drop in the area exactly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem
from lmhdx.fully_developed import channel_flow_rate, channel_flow_response
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_compiled

# Inputs: the design-law sweep, the duct and its field, and the mesh.
OUTPUT_DIR = Path("artifacts/examples/duct_design_law")
SQUARE_HARTMANN = (10.0, 30.0, 100.0, 300.0)  # H, the equal-area square duct's Hartmann number
FLOW_RATE = 2.4e-6  # m^3/s
MIN_VELOCITY = 0.010  # m/s, the smallest mean velocity allowed
VISCOSITY = 2.15e-3  # Pa s, PbLi at 573 K (Martelli, Venturini & Utili 2019)
CONDUCTIVITY = 8.84e5  # S/m, PbLi at 573 K
FIRST_WALL_FIELD = 0.5  # T, a laboratory-scale field, not a reactor's 4-10 T
FIRST_WALL_RADIUS = 12.1  # m, outboard midplane
BLANKET_DEPTH = 0.982  # m, the duct's radial run
STATIONS = 6  # equal lengths of the run, each at its local 1/R field
CELLS = 48  # cells along each direction of the cross-section
CELLS_IN_LAYER = 6  # of them in each wall layer
SEARCH_WIDTH = 0.1  # spacing of the five trials in ln(beta)


def unit_duct(beta: float, ha_mesh: float, centre: float) -> ChannelProblem:
    """A unit insulated duct of aspect ``beta``: the mesh built for ``centre`` and ``ha_mesh``, scaled to ``beta``."""
    y = wall_resolving_faces(
        CELLS, -1.0, 1.0, layer_thickness=1.0 / ha_mesh, cells_in_layer=CELLS_IN_LAYER, max_ratio=None
    )
    side = min(1.0 / np.sqrt(ha_mesh), 0.25 * centre)
    z = wall_resolving_faces(
        CELLS, -centre, centre, layer_thickness=side, cells_in_layer=CELLS_IN_LAYER, max_ratio=None
    )
    wall = BoundaryCondition("neumann")
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z * (beta / centre)),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, ha_mesh, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )


def pressure_drop(beta, area, fields, lengths, *, flow, mu, sigma, centre, slope=False, margin=1.15):
    """Return ``Delta p`` over stations of ``fields`` and ``lengths``; with ``slope``, also ``d Delta p / d ln(area)``.

    ``Delta p = sum L mu Q / (a^4 q(Ha, beta))`` with ``Ha = B a sqrt(sigma / mu)``
    and ``a = sqrt(area / (4 beta))``. The mesh is built for ``margin`` times the
    largest Hartmann number a trial around ``centre`` reaches. Every trial duct
    has the same mesh shape, so its solves share one compiled program; the
    slope instead differentiates the solve in its field scale with ``jax.grad``.
    """
    a = np.sqrt(area / (4.0 * beta))
    hartmann = np.asarray(fields) * a * np.sqrt(sigma / mu)
    ha_mesh = margin * max(fields) * np.sqrt(area / (4.0 * centre) * sigma / mu)
    problem = unit_duct(beta, ha_mesh, centre)
    response = jax.jit(
        jax.value_and_grad(
            lambda scale: channel_flow_response(problem, magnetic_field_scale=scale).flow_per_unit_drive
        )
    )
    drop = derivative = 0.0
    for ha, length in zip(hartmann, lengths):
        if slope:
            q, dq_dscale = (float(value) for value in response(ha / ha_mesh))
        else:
            station = replace(problem, magnetic_field=(0.0, float(ha), 0.0))
            q, dq_dscale = float(channel_flow_rate(station, solve_compiled(station).velocity[0].data[0])), 0.0
        gradient = mu * flow / (a**4 * q)
        drop += length * gradient
        # d ln a / d ln(area) = 1/2, and Ha is proportional to a at a fixed field.
        derivative += 0.5 * length * gradient * (-4.0 - dq_dscale * ha / (ha_mesh * q))
    return drop, derivative if slope else None


def best_aspect(drop_at, centre: float) -> float:
    """The minimum of a smooth ``drop_at(beta)``: a quartic through five trials in ``ln(beta)`` around ``centre``."""
    offsets = SEARCH_WIDTH * np.arange(-2, 3)
    coefficients = np.polyfit(offsets, [drop_at(centre * np.exp(x)) for x in offsets], 4)
    roots = np.roots(np.polyder(coefficients))
    convex = [
        root.real
        for root in roots
        if abs(root.imag) < 1e-12
        and abs(root.real) <= 2.0 * SEARCH_WIDTH
        and np.polyval(np.polyder(coefficients, 2), root.real) > 0.0
    ]
    if not convex:
        raise RuntimeError(f"no minimum within the trials around beta = {centre:g}; move the centre")
    return float(centre * np.exp(min(convex, key=abs)))


def law_guess(square_hartmann: float) -> float:
    """``beta`` from ``beta * sqrt(Ha) = 2.08`` with ``Ha = H / sqrt(beta)`` at a fixed area."""
    return float((2.08 / np.sqrt(square_hartmann)) ** (4.0 / 3.0))


# Run in float64: the optimum is a shallow minimum, which float32 solves would blur.
enable_x64()

# Run the design law: a unit mean velocity through area 4, so the square duct has a = 1 and Ha = H.
print(f"Design law over H = {SQUARE_HARTMANN}, {CELLS}^2 cells, five solves per H...", flush=True)
law = []
for square_hartmann in SQUARE_HARTMANN:
    centre = law_guess(square_hartmann)
    settings = dict(flow=4.0, mu=1.0, sigma=1.0, centre=centre)
    beta = best_aspect(lambda b: pressure_drop(b, 4.0, [square_hartmann], [1.0], **settings)[0], centre)
    optimum = pressure_drop(beta, 4.0, [square_hartmann], [1.0], **settings)[0]
    square = pressure_drop(1.0, 4.0, [square_hartmann], [1.0], **{**settings, "centre": 1.0})[0]
    ha_star = square_hartmann / np.sqrt(beta)
    law.append(
        {
            "H": square_hartmann,
            "beta_star": beta,
            "Ha_star": ha_star,
            "s_star": beta * np.sqrt(ha_star),
            "reduction": 1.0 - optimum / square,
        }
    )
    print(
        f"H {square_hartmann:5.0f}: beta* {beta:.5f}, Ha* {ha_star:7.1f}, "
        f"s* {law[-1]['s_star']:.4f}, {100 * law[-1]['reduction']:.1f}% below the square",
        flush=True,
    )

# Run the design: the stations of a radial run through a 1/R field, at the area MIN_VELOCITY sets.
edges = FIRST_WALL_RADIUS + BLANKET_DEPTH * np.linspace(0.0, 1.0, STATIONS + 1)
radii = 0.5 * (edges[:-1] + edges[1:])
fields = list(FIRST_WALL_FIELD * FIRST_WALL_RADIUS / radii)
lengths = [BLANKET_DEPTH / STATIONS] * STATIONS
area = FLOW_RATE / MIN_VELOCITY
print(
    f"Blanket duct: {STATIONS} stations at {min(fields):.3f}-{max(fields):.3f} T, area {1e4 * area:.2f} cm^2...",
    flush=True,
)
mean_field = float(np.mean(fields))
centre = law_guess(mean_field * np.sqrt(area / 4.0 * CONDUCTIVITY / VISCOSITY))
settings = dict(flow=FLOW_RATE, mu=VISCOSITY, sigma=CONDUCTIVITY, centre=centre)
beta = best_aspect(lambda b: pressure_drop(b, area, fields, lengths, **settings)[0], centre)
drop, slope = pressure_drop(beta, area, fields, lengths, slope=True, **settings)
square_drop = pressure_drop(1.0, area, fields, lengths, **{**settings, "centre": 1.0})[0]
if not slope < 0.0:
    raise RuntimeError("the pumping power rises with the area here: the velocity bound is not active")
a = float(np.sqrt(area / (4.0 * beta)))
design = {
    "flow_rate_m3_s": FLOW_RATE,
    "mean_velocity_m_s": MIN_VELOCITY,
    "area_m2": area,
    "station_fields_T": fields,
    "beta_star": beta,
    "half_width_along_field_m": a,
    "half_width_across_field_m": beta * a,
    "pressure_drop_Pa": drop,
    "pumping_power_W": FLOW_RATE * drop,
    "square_pumping_power_W": FLOW_RATE * square_drop,
    "reduction": 1.0 - drop / square_drop,
    "power_slope_in_ln_area_W": FLOW_RATE * slope,
}
print(
    f"design: beta* {beta:.5f}, a {1e3 * a:.2f} mm, b {1e3 * beta * a:.2f} mm, "
    f"W* {design['pumping_power_W']:.4e} W, {100 * design['reduction']:.1f}% below the square",
    flush=True,
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = {"mesh": {"cells": CELLS, "cells_in_layer": CELLS_IN_LAYER}, "design_law": law, "design": design}
(OUTPUT_DIR / "duct_design_law_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
figure, axis = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
axis.semilogx(SQUARE_HARTMANN, [row["s_star"] for row in law], "o-", label="solved optimum")
axis.axhline(2.13, ls="--", color="gray", label="thin-layer law 2.13")
axis.set(xlabel="H (equal-area square duct)", ylabel=r"$s^* = \beta^* \sqrt{Ha^*}$")
axis.legend()
figure.savefig(OUTPUT_DIR / "duct_design_law.png", dpi=150)
print(f"Wrote {OUTPUT_DIR / 'duct_design_law_summary.json'} and {OUTPUT_DIR / 'duct_design_law.png'}")
