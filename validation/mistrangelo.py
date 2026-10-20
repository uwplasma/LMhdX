"""Mistrangelo et al. (2025): a heated conducting square duct in a fringing field (plan step 5.4, ADR 0009).

The frozen specification is ``src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml``. This module reads it,
derives the PbLi properties pinned to the printed dimensionless groups, and runs 5.4a's first check, Case 1's
developed flow in the uniform field. That flow is solved on a section one cell long, with the steel resolved in
cells on both transverse axes, which LMhdX supports today. The paper's developed pressure gradient, -0.0863 in
``sigma u0 B0^2`` (p. 8), then fixes the wall's conductivity ratio (:func:`infer_wall_ratio`). The ratio is
pinned in the specification, beside the measured value it differs from by 5.3 %.

``python -m validation.mistrangelo`` prints the check on three meshes. ``python -m validation.mistrangelo digitize
DIR`` rewrites the reference curves from the paper's figure images in ``DIR`` (:func:`write_references`).
"""

from __future__ import annotations

import hashlib
import json
import sys
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
    "REFERENCES",
    "REFERENCES_CASE2",
    "SPEC",
    "developed_section",
    "digitize",
    "field",
    "infer_wall_ratio",
    "pinned_pbli",
    "section_flow",
    "spec",
    "write_references",
]

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


# The reference curves (plan step 5.4a, part 2). The paper's figures are xmgrace plots embedded as raster images;
# each is read from its image (page and xref in the PDF, SHA-256 recorded) by colour, one value per pixel column.
# The axes are calibrated on the major ticks drawn outside the frame, two labelled anchors per axis snapped to them.

REFERENCES = _DATA / "references" / "mistrangelo2025-case1.json"
REFERENCES_CASE2 = _DATA / "references" / "mistrangelo2025-case2.json"
SIX = ("FLUENT_CIEMAT", "FLUENT_UK", "CFX", "OF_ORNL", "OF_KIT", "COMSOL")
# Each colour as (hue window in degrees, saturation window, value window); black is unsaturated and dark.
COLOURS = {
    "FLUENT_CIEMAT": ((0, 360), (0.0, 0.35), (0.0, 0.30)),
    "FLUENT_UK": ((22, 50), (0.3, 1.0), (0.25, 1.0)),
    "OF_KIT": ((75, 160), (0.3, 1.0), (0.25, 1.0)),
    "OF_ORNL": ((200, 265), (0.3, 1.0), (0.25, 1.0)),
    "COMSOL": ((270, 325), (0.3, 1.0), (0.25, 1.0)),
    "CFX": ((335, 15), (0.3, 1.0), (0.25, 1.0)),
}
STATIONS = {  # Fig. 11, OF_KIT at five stations, in its own colours
    "x=0": ((0, 360), (0.0, 0.1), (0.0, 0.55)),
    "x=0.15": ((265, 291), (0.4, 1.0), (0.25, 1.0)),
    "x=0.3": ((165, 200), (0.4, 1.0), (0.25, 1.0)),
    "x=0.6": ((292, 330), (0.4, 1.0), (0.25, 1.0)),
    "x=1": ((340, 25), (0.12, 0.35), (0.55, 0.92)),
}
PAIRS = {"T_w": ((200, 265), (0.3, 1.0), (0.25, 1.0)), "T_b": ((75, 160), (0.3, 1.0), (0.25, 1.0))}  # Fig. 38
# Figure: page, xref, x anchors, y anchors (value, approximate pixel), the quantity, the codes the paper says agree
# (the band is their union) and where it says so. Options: "log" (y in decades), "circles" (the analytical
# solution's red markers split from CFX by shape), "rows" (pixel rows excluded from black: a dotted zero line),
# "branches" (two curves per colour, named lower then upper).
# fmt: off
FIGURES = {
    6: dict(page=8, xref=176, x=((-1.5, 85), (1.5, 730)), y=((310, 97), (300, 457)), quantity="T [C] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 8"),
    7: dict(page=9, xref=201, x=((-1.5, 84), (1.5, 693)), y=((0.07, 7), (0.02, 434)), quantity="u [m/s] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 8"),
    8: dict(page=9, xref=203, x=((-1.5, 79.5), (1.5, 692)), y=((1000, 7), (250, 327)), quantity="p [Pa] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 8"),
    9: dict(page=9, xref=205, x=((-0.025, 83), (0.025, 720.5)), y=((303, 56), (300, 390)), quantity="T [C] at x = -0.3 m, y = 0 against z [m]", agree=("FLUENT_UK", "CFX", "OF_KIT", "COMSOL"), cite="p. 8"),
    10: dict(page=9, xref=207, x=((-0.025, 84), (0.025, 720.5)), y=((325, 8), (300, 420)), quantity="T [C] at x = 0, y = 0 against z [m]", agree=SIX[:1] + SIX[2:], cite="p. 8"),
    11: dict(page=10, xref=235, x=((-0.025, 83), (0.025, 720.5)), y=((330, 8), (300, 454)), quantity="T [C] in the fluid at y = 0 against z [m], OF_KIT at five stations", agree=(), cite="p. 9", colours=STATIONS),
    12: dict(page=10, xref=237, x=((-0.025, 87), (0.025, 722)), y=((0.1, 7), (0.0, 451)), quantity="u [m/s] at x = -0.3 m, y = 0 against z [m]", agree=("FLUENT_CIEMAT", "OF_ORNL", "OF_KIT", "COMSOL"), cite="p. 9"),
    13: dict(page=10, xref=239, x=((-0.025, 88), (0.025, 722)), y=((0.1, 7), (0.0, 451)), quantity="u [m/s] at x = 0, y = 0 against z [m]", agree=("FLUENT_CIEMAT", "OF_ORNL", "OF_KIT", "COMSOL", "analytical"), cite="p. 9", circles=True),
    15: dict(page=11, xref=275, x=((-1.5, 95), (1.5, 731)), y=((0, 22), (-400, 446)), quantity="phi [uV] at y = 0 on the interface z = -0.023 m against x [m]", agree=SIX, cite="p. 10: most of the data fits well together"),
    16: dict(page=11, xref=277, x=((-0.025, 86), (0.025, 722)), y=((400, 7), (-400, 452.5)), quantity="phi [uV] at x = y = 0 against z [m]", agree=SIX[:1] + SIX[2:], cite="p. 10: FLUENT_UK deviates slightly", rows=(226, 234)),
    17: dict(page=11, xref=279, x=((-1.5, 87.5), (1.5, 731)), y=((312.5, 7), (300, 454)), quantity="bulk T [C], eq. 6, against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 10: FLUENT_UK non-physical"),
    18: dict(page=12, xref=309, x=((-0.3, 83), (0.3, 730)), y=((20, 8), (0, 468)), quantity="Nu, eq. 7, against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 10: FLUENT_UK as in the previous figures; Nu -> 2.45"),
    19: dict(page=12, xref=311, x=((-1.25, 68), (1.25, 726)), y=((1e-3, 33), (1e-7, 461.5)), quantity="E_t, eq. 8, against x [m]", agree=("FLUENT_CIEMAT", "OF_ORNL", "OF_KIT"), cite="as read: FLUENT_UK as in the previous figures (p. 10), CFX lower, its diffusion (p. 16); COMSOL not shown", log=True),
    38: dict(page=17, xref=486, x=((-1, 88.5), (1, 777)), y=((400, 44.5), (300, 481)), quantity="T_w at y = 0, z = -0.023 m and T_b [C] against x [m], OF_KIT, both cases (case1 dashed, below)", agree=(), cite="p. 15", colours=PAIRS, branches=("case1", "case2")),
}
ALL_BUT = {code: tuple(c for c in SIX if c != code) for code in SIX}
FIGURES_CASE2 = {
    23: dict(page=14, xref=368, x=((-1.5, 84), (1.5, 691)), y=((0.025, 7.5), (0.005, 434)), quantity="u [m/s] on y = z = 0 against x [m]", agree=("FLUENT_CIEMAT", "FLUENT_UK", "OF_KIT", "COMSOL"), cite="p. 13: CFX differs; OF_ORNL's fluctuations, numerical problems"),
    24: dict(page=14, xref=370, x=((-1.5, 81), (1.5, 692.5)), y=((375, 43), (285, 428.5)), quantity="T [C] on y = z = 0 against x [m]", agree=("FLUENT_CIEMAT", "FLUENT_UK", "OF_KIT", "COMSOL"), cite="p. 13: CFX's steeper rise; OF_ORNL's fluctuations"),
    26: dict(page=14, xref=374, x=((-0.025, 84.5), (0.025, 683)), y=((0.03, 7), (0.0, 427)), quantity="u [m/s] at x = -0.3 m, y = 0 against z [m]", agree=ALL_BUT["CFX"] + ("analytical",), cite="p. 13: all but CFX reproduce the analytical solution", circles=True),
    27: dict(page=14, xref=376, x=((-0.025, 84), (0.025, 684)), y=((0.04, 6), (0.0, 427)), quantity="u [m/s] at x = 0, y = 0 against z [m]", agree=ALL_BUT["CFX"], cite="p. 13: CFX's different profile in the heated zone"),
    28: dict(page=15, xref=408, x=((-0.025, 80.5), (0.025, 682)), y=((310, 7), (300, 390)), quantity="T [C] at x = -0.3 m, y = 0 against z [m]", agree=("OF_KIT", "COMSOL"), cite="p. 13: COMSOL and OF_KIT fit well together"),
    29: dict(page=15, xref=410, x=((-0.025, 78.5), (0.025, 681.5)), y=((420, 7), (300, 429)), quantity="T [C] at x = 0, y = 0 against z [m]", agree=ALL_BUT["CFX"], cite="p. 13: all predict the thicker layer; CFX's steeper rise, as read"),
    30: dict(page=15, xref=412, x=((-1.5, 81), (1.5, 692)), y=((300, 7), (0, 422)), quantity="p [Pa] on y = z = 0 against x [m], FLUENT_CIEMAT and CFX offsets removed", agree=SIX, cite="p. 14"),
    31: dict(page=15, xref=414, x=((-1.5, 90), (1.5, 693)), y=((20, 6), (-120, 414)), quantity="phi [uV] at y = 0 on z = -0.023 m against x [m]", agree=("CFX", "OF_ORNL", "OF_KIT", "COMSOL"), cite="p. 14: FLUENT's curves are instantaneous"),
    32: dict(page=15, xref=416, x=((-1.5, 83), (1.5, 693)), y=((120, 15), (-20, 427)), quantity="phi [uV] at y = 0 on z = 0.023 m against x [m]", agree=("CFX", "OF_ORNL", "OF_KIT", "COMSOL"), cite="p. 14: FLUENT's curves are instantaneous"),
    33: dict(page=16, xref=451, x=((-0.025, 82), (0.025, 683)), y=((150, 7), (-150, 427)), quantity="phi [uV] at x = y = 0 against z [m]", agree=ALL_BUT["CFX"], cite="p. 14: good agreement; CFX offset, as read", rows=(214, 221)),
    34: dict(page=16, xref=453, x=((-1, 86), (1, 700)), y=((380, 7), (300, 432)), quantity="bulk T [C], eq. 6, against x [m]", agree=("FLUENT_CIEMAT", "FLUENT_UK", "CFX", "OF_KIT"), cite="p. 14: OF_ORNL and COMSOL unphysical"),
    35: dict(page=16, xref=455, x=((-0.5, 284), (1, 723)), y=((1e-1, 12), (1e-8, 504)), quantity="E_t, eq. 8, against x [m]", agree=("FLUENT_CIEMAT", "FLUENT_UK", "OF_KIT"), cite="p. 15: CFX differs; OF_ORNL and COMSOL unphysical", log=True),
    37: dict(page=17, xref=484, x=((-0.3, 79), (0.3, 769)), y=((20, 9), (0, 496)), quantity="Nu, eq. 7, against x [m]", agree=("FLUENT_CIEMAT", "FLUENT_UK", "OF_KIT"), cite="as read: CFX differs and OF_ORNL's and COMSOL's bulk temperatures fluctuate (p. 15)"),
}
# fmt: on


def _hsv(a):
    top, low = a.max(2), a.min(2)
    delta = top - low
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        hue = np.where(
            top == r, ((g - b) / delta) % 6, np.where(top == g, (b - r) / delta + 2, (r - g) / delta + 4)
        )
    return 60.0 * np.nan_to_num(hue), np.where(top > 0, delta / np.maximum(top, 1e-9), 0.0), top


def _frame(a):
    """The plot frame: the outermost rows and columns dark across most of the image."""
    dark = a.max(2) < 0.7
    rows, cols = np.flatnonzero(dark.mean(1) > 0.6), np.flatnonzero(dark.mean(0) > 0.6)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _ticks(a, frame, axis):
    """Major tick positions, the longest runs dark within 13 px outside the bottom or the left frame line."""
    top, bottom, left, right = frame
    dark = a.max(2) < 0.45
    lo = max(top - 2, 0)
    length = (
        dark[bottom + 1 : bottom + 14, left - 2 : right + 3].sum(0)
        if axis == "x"
        else dark[lo : bottom + 3, left - 13 : left].sum(1)
    )
    on = np.append(length >= 0.75 * length.max(), False)
    edges = np.diff(np.concatenate([[0], on.astype(int)]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (starts + ends - 1) / 2.0 + (left - 2 if axis == "x" else lo)


def _calibration(anchors, ticks, log=False):
    """Snap two (value, approximate pixel) anchors to the detected ticks; the map and the other ticks' misfit."""
    (v0, p0), (v1, p1) = ((np.log10(v) if log and v > 0 else v, p) for v, p in anchors)
    q0, q1 = (float(ticks[np.argmin(np.abs(ticks - p))]) for p in (p0, p1))
    if max(abs(q0 - p0), abs(q1 - p1)) > 4.0:
        raise ValueError(f"no tick within 4 px of {p0} or {p1}: {ticks}")
    step = abs(q1 - q0) / max(1, round(abs(q1 - q0) / np.median(np.diff(np.sort(ticks)))))
    misfit = [abs((t - q0) / step - round((t - q0) / step)) * step for t in ticks]
    on = [m for m in misfit if m < 0.25 * step]
    return (v0, q0, (v1 - v0) / (q1 - q0)), {
        "anchors": [[v0, q0], [v1, q1]],
        "ticks": len(on),
        "misfit_px": max(on),
    }


def _masks(a, colours, frame, legend, relax=0.0):
    hue, sat, val = _hsv(a)
    top, bottom, left, right = frame
    masks = {}
    for name, ((h0, h1), (s0, s1), (v0, v1)) in colours.items():
        inside = (hue >= h0) & (hue <= h1) if h0 < h1 else (hue >= h0) | (hue <= h1)
        mask = inside & (sat >= (s0 - relax if s0 >= 0.3 else s0)) & (sat <= s1) & (val >= v0) & (val <= v1)
        margin = 3 if s1 <= 0.35 and v1 <= 0.55 else 2  # black runs along the 2-px frame lines
        keep = np.zeros_like(mask)
        keep[top + margin : bottom - margin + 1, left + margin : right - margin + 1] = True
        if legend is not None:
            keep[legend[0] : legend[1], legend[2] : legend[3]] = False
        masks[name] = mask & keep
    return masks


def _legend(a, frame):
    """The legend's frame: dark runs longer than 80 px (rows) and 40 px (columns) inside the plot frame, padded."""
    top, bottom, left, right = frame
    dark = a[top + 4 : bottom - 3, left + 4 : right - 3].max(2) < 0.6

    def longest(m):
        padded = np.pad(m.astype(int), ((0, 0), (1, 1)))
        return np.array(
            [
                max(
                    [0]
                    + [
                        e - s
                        for s, e in zip(np.flatnonzero(np.diff(r) == 1), np.flatnonzero(np.diff(r) == -1))
                    ]
                )
                for r in padded
            ]
        )

    rows, cols = np.flatnonzero(longest(dark) > 80), np.flatnonzero(longest(dark.T) > 40)
    if len(rows) < 2 or len(cols) < 2:
        return None
    return (rows.min() + top + 1, rows.max() + top + 8, cols.min() + left + 1, cols.max() + left + 8)


def _circles(mask):
    """Open-circle markers (rings 4-10 px across, with a hole) split from the rest of a mask."""
    from scipy import ndimage

    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    rings = np.zeros_like(mask)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        part = labels[sl] == k
        h, w = part.shape
        if (
            4 <= h <= 10
            and 4 <= w <= 10
            and part.sum() < 0.75 * h * w
            and ndimage.binary_fill_holes(part).sum() > part.sum()
        ):
            rings[sl] |= part
    return rings, mask & ~rings


def _branches(mask):
    """Split a mask holding two curves of one colour into the lower and the upper in value, column by column.

    A column with one cluster of pixels goes to the nearer of the two tracks, or to both where they coincide.
    """
    low, high = np.zeros_like(mask), np.zeros_like(mask)
    track = None
    for c in np.flatnonzero(mask.any(0)):
        rows = np.flatnonzero(mask[:, c])
        clusters = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
        if len(clusters) > 1:
            high[clusters[0], c], low[clusters[-1], c] = True, True  # rows grow downward: the first is higher
            track = (np.median(clusters[-1]), np.median(clusters[0]))
            continue
        row = np.median(rows)
        if track is None or abs(track[0] - track[1]) <= 3:
            low[rows, c] = high[rows, c] = True
            track = (row, row)
        elif abs(row - track[0]) < abs(row - track[1]):
            low[rows, c], track = True, (row, track[1])
        else:
            high[rows, c], track = True, (track[0], row)
    return low, high


def digitize(image: Path, figure: int, table: dict | None = None) -> dict:
    """One figure's curves, each the median row per pixel column, and the band of the codes that agree."""
    from PIL import Image

    spec_, data = (table or FIGURES)[figure], Path(image).read_bytes()
    a = np.asarray(Image.open(image).convert("RGB")).astype(float) / 255.0
    frame = _frame(a)
    legend = _legend(a, frame)
    (x0, px0, sx), cx = _calibration(spec_["x"], _ticks(a, frame, "x"))
    (y0, py0, sy), cy = _calibration(spec_["y"], _ticks(a, frame, "y"), spec_.get("log", False))
    relax = 0.1 if Path(image).suffix.lower() in (".jpg", ".jpeg") else 0.0  # JPEG washes thin lines out
    masks = _masks(a, spec_.get("colours", COLOURS), frame, legend, relax)
    if spec_.get("circles"):
        masks["analytical"], masks["CFX"] = _circles(masks["CFX"])
    for name in list(masks) if spec_.get("branches") else ():
        low, high = _branches(masks.pop(name))
        masks[f"{name} {spec_['branches'][0]}"], masks[f"{name} {spec_['branches'][1]}"] = low, high
    for name in masks:
        if "rows" in spec_ and name == "FLUENT_CIEMAT":
            masks[name][slice(*spec_["rows"])] = False

    def values(cols, rows):
        y = y0 + sy * (rows - py0)
        return x0 + sx * (cols - px0), (10.0**y if spec_.get("log") else y)

    def sample(x, *ys, points):
        stride = max(1, int(np.ceil(len(x) / points)))
        return [[float(f"{v:.5g}") for v in arr[::stride]] for arr in (x, *ys)]

    curves = {}
    for name, mask in masks.items():
        cols = np.flatnonzero(mask.any(0))
        rows = np.array([np.median(np.flatnonzero(mask[:, c])) for c in cols])
        x, y = values(cols, rows)
        curves[name] = dict(
            zip(("x", "y"), sample(x, y, points=60)), coverage=round(len(cols) / (frame[3] - frame[2]), 3)
        )
    record = {"figure": figure, "quantity": spec_["quantity"], "page": spec_["page"], "xref": spec_["xref"],
              "sha256": hashlib.sha256(data).hexdigest(), "pixels": list(a.shape[:2]), "frame": list(frame),
              "legend": None if legend is None else [int(v) for v in legend], "calibration": {"x": cx, "y": cy},
              "per_pixel": [abs(sx), abs(sy)], "agree": list(spec_["agree"]), "agree_source": spec_["cite"], "curves": curves}  # fmt: skip
    if spec_["agree"]:
        union = np.any([masks[name] for name in spec_["agree"]], axis=0)
        cols = np.flatnonzero(union.any(0))
        lower = np.array([np.flatnonzero(union[:, c]).max() for c in cols], float)
        upper = np.array([np.flatnonzero(union[:, c]).min() for c in cols], float)
        x, lo = values(cols, lower)
        record["band"] = dict(zip(("x", "lower", "upper"), sample(x, lo, values(cols, upper)[1], points=100)))
    return record


def write_references(images: Path) -> tuple[Path, Path]:
    """Digitize every figure from ``images`` (``x<xref>.png`` or ``.jpeg``, from the PDF) into the two files."""
    for case, table, path in (("case1", FIGURES, REFERENCES), ("case2", FIGURES_CASE2, REFERENCES_CASE2)):
        figures = []
        for figure, spec_ in table.items():
            record = digitize(next(Path(images).glob(f"x{spec_['xref']}.*")), figure, table)
            curves, band = record.pop("curves"), record.pop("band", None)
            lines = [f"  {json.dumps(name)}: {json.dumps(curve)}" for name, curve in curves.items()]
            tail = f',\n "band": {json.dumps(band)}' if band else ""
            figures.append(json.dumps(record)[:-1] + ', "curves": {\n' + ",\n".join(lines) + "}" + tail + "}")
        head = f'{{"source_id": "mistrangelo-2025", "case": "{case}", "figures": [\n'
        path.write_text(head + ",\n".join(figures) + "\n]}\n")
    return REFERENCES, REFERENCES_CASE2


def main() -> None:
    if sys.argv[1:2] == ["digitize"]:
        print(write_references(Path(sys.argv[2])))
        return
    record = spec()
    ratio = record["properties"]["steel"]["conductivity_ratio_pinned"]
    for cells, layer, walls in ((48, 6, 6), (72, 9, 8), (96, 12, 10)):
        flow = section_flow(ratio, cells=cells, cells_in_layer=layer, wall_cells=walls)
        u = flow["u_over_u0"]
        print(f"{cells}/{layer}, {walls} wall cells: -dp/dx {flow['gradient']:.5f} (printed {DEVELOPED_GRADIENT}); "
              f"jet {u.max():.3f} u0 at |z|/a {abs(flow['z'][np.argmax(u)]):.3f}; centre {np.interp(0.0, flow['z'], u):.4f} u0")  # fmt: skip


if __name__ == "__main__":
    main()
