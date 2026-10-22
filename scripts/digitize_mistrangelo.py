"""Digitize the figures of Mistrangelo et al. (2025) into the compact references of plan step 5.4a (ADR 0009 D38).

``python scripts/digitize_mistrangelo.py DIR`` rewrites ``mistrangelo2025-case1.json`` from the paper's figure
images in ``DIR``, and writes every code's curve to ``DIR/mistrangelo2025-case1-curves.json``, a release asset. They are extracted from the journal's PDF (doi:10.1088/1741-4326/ae0800) with ``pymupdf``,
``Document.extract_image(xref)``, as ``x<xref>.png`` or ``.jpeg``, and are not in the repository, so this script
is not part of the package or of ``validation/``. ``benchmarks/provenance.json`` records its SHA-256 beside each
reference file's.

A reference keeps, per figure, the image's hash, the calibration of both axes, the codes that agree and where the
paper says so (``agree_basis``: "the paper", or "my reading" where it does not), and the band of those codes, with
the curves that are not one of the six codes. Each is simplified to within half a pixel and rounded to a tenth of
one. Each axis carries its calibration misfit in its own units, and the record its digitizing uncertainty, which a
test uses as its tolerance. ``digitize`` returns every code's curve in full, for a figure read again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REFERENCES = (
    Path(__file__).resolve().parents[1] / "src/lmhdx/data/benchmarks/references/mistrangelo2025-case1.json"
)

# The reference curves (plan step 5.4a). The paper's figures are xmgrace plots embedded as raster images, extracted
# with ``pymupdf`` (``Document.extract_image(xref)``) as ``x<xref>.png`` or ``.jpeg``; each record carries the image's
# SHA-256. A figure is read by colour, one value per pixel column. Each axis is calibrated on the major ticks drawn
# outside the frame: two labelled anchors label the detected ticks, every tick inside the frame must land on the axis's
# label lattice (``step`` apart; decades on a log axis), and the map is their least-squares fit.

SIX = ("FLUENT_CIEMAT", "FLUENT_UK", "CFX", "OF_ORNL", "OF_KIT", "COMSOL")
DASH = 15  # px: gaps in one code's curve up to a dash period are closed before the band is taken
# The digitizing uncertainty, per axis: a drawn line or ring is read within 1.5 px and the compact record keeps it
# within 0.6 px (both from test_the_digitizer_reads_a_drawn_figure), plus the axis calibration's own misfit.
READ_PX, COMPACT_PX = 1.5, 0.6
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
# Figure: the PDF page (the journal's is one less), xref, each axis as two (value, approximate pixel) anchors and the
# label step, the quantity, the codes the paper says agree (the band is their spread) with where it says so ("as
# read" where it does not), and the x window of the band. Options: "log" (y in decades), "colours", "omit" (codes not
# separable, with the reason), "circles" (the analytical solution's red rings split from CFX by shape), "rows" (pixel
# rows excluded from black: a dotted zero line).
# fmt: off
FIGURES = {
    6: dict(pdf_page=8, xref=176, x=((-1.5, 85), (1.5, 730), 0.5), y=((310, 97), (300, 457), 2.5), quantity="T [C] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 7: good agreement for -0.75 < x < 0.8 m; FLUENT_UK deviates", window=(-0.75, 0.8)),
    7: dict(pdf_page=9, xref=201, x=((-1.5, 84), (1.5, 693), 0.5), y=((0.07, 7), (0.02, 434), 0.01), quantity="u [m/s] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="as read, in Fig. 6's window (p. 7); FLUENT_UK's core velocity deviates (p. 8)", window=(-0.75, 0.8)),
    8: dict(pdf_page=9, xref=203, x=((-1.5, 79.5), (1.5, 692), 0.5), y=((1000, 7), (250, 327), 250), quantity="p [Pa] on y = z = 0 against x [m]", agree=SIX[:1] + SIX[2:], cite="p. 8: all agree but FLUENT_UK"),
    9: dict(pdf_page=9, xref=205, x=((-0.025, 83), (0.025, 720.5), 0.01), y=((303, 56), (300, 390), 1), quantity="T [C] at x = -0.3 m, y = 0 against z [m]", agree=("FLUENT_UK", "CFX", "OF_KIT", "COMSOL"), cite="p. 8: OF_ORNL and FLUENT_CIEMAT deviate, their axial resolution"),
    10: dict(pdf_page=9, xref=207, x=((-0.025, 84), (0.025, 720.5), 0.01), y=((325, 8), (300, 420), 5), quantity="T [C] at x = 0, y = 0 against z [m]", agree=SIX[:1] + SIX[2:], cite="as read: FLUENT_UK lower"),
    11: dict(pdf_page=10, xref=235, x=((-0.025, 83), (0.025, 720.5), 0.01), y=((330, 8), (300, 454), 5), quantity="T [C] in the fluid at y = 0 against z [m], OF_KIT at five stations", agree=(), cite="p. 8 (text), p. 9 (caption)", colours=STATIONS),
    12: dict(pdf_page=10, xref=237, x=((-0.025, 87), (0.025, 722), 0.01), y=((0.1, 7), (0.0, 451), 0.02), quantity="u [m/s] at x = -0.3 m, y = 0 against z [m]", agree=("FLUENT_CIEMAT", "OF_ORNL", "OF_KIT", "COMSOL"), cite="as read: FLUENT_UK and CFX deviate, as p. 9 says of Fig. 13"),
    13: dict(pdf_page=10, xref=239, x=((-0.025, 88), (0.025, 722), 0.01), y=((0.1, 7), (0.0, 451), 0.02), quantity="u [m/s] at x = 0, y = 0 against z [m]", agree=("FLUENT_CIEMAT", "OF_ORNL", "OF_KIT", "COMSOL", "analytical"), cite="p. 9: the analytical solution agrees with FLUENT_CIEMAT, OF_KIT, OF_ORNL and COMSOL", circles=True, omit={"CFX": "not separable from the analytical solution's red markers"}),
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


def _ticks(a, frame, axis, dark=0.45):
    """Major tick positions, the longest runs darker than ``dark`` within 13 px outside the bottom or left frame line."""
    top, bottom, left, right = frame
    darker = a.max(2) < dark
    lo = max(top - 2, 0)
    length = (
        darker[bottom + 1 : bottom + 14, left - 2 : right + 3].sum(0)
        if axis == "x"
        else darker[lo : bottom + 3, left - 13 : left].sum(1)
    )
    on = np.append(length >= 0.75 * length.max(), False)
    edges = np.diff(np.concatenate([[0], on.astype(int)]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (starts + ends - 1) / 2.0 + (left - 2 if axis == "x" else lo)


def _calibration(axis, ticks, ends, log=False):
    """The axis's map, fitted to every major tick inside the frame on its label lattice.

    The two (value, approximate pixel) anchors are snapped to detected ticks and label the others. A tick inside the
    frame that does not land within a quarter step of the lattice, or fewer than three ticks that do, is an error, so
    a wrong anchor value fails here; a tick on a frame line counts only where it lands on the lattice.
    """
    (v0, p0), (v1, p1), step = axis
    if log:
        if min(v0, v1) <= 0:
            raise ValueError("a log axis needs positive anchors")
        v0, v1 = np.log10(v0), np.log10(v1)
    q0, q1 = (float(ticks[np.argmin(np.abs(ticks - p))]) for p in (p0, p1))
    if max(abs(q0 - p0), abs(q1 - p1)) > 4.0:
        raise ValueError(f"no tick within 4 px of {p0} or {p1}: {ticks}")
    slope = (v1 - v0) / (q1 - q0)
    edge = np.array([min(abs(t - e) for e in ends) <= 2.0 for t in ticks])
    labels = v0 + slope * (np.asarray(ticks) - q0)
    lattice = np.round(labels / step) * step
    on = np.abs(labels - lattice) <= 0.25 * step
    if not on[~edge].all() or on.sum() < 3:  # a frame-edge tick counts only where it lands on the lattice
        raise ValueError(f"ticks inside the frame off the {step} lattice or too few: {ticks}, {labels}")
    fit = np.polyfit(np.asarray(ticks)[on], lattice[on], 1)
    misfit = float(np.max(np.abs(np.polyval(fit, np.asarray(ticks)[on]) - lattice[on])) / abs(fit[0]))
    found = {"anchors": [[v0, q0], [v1, q1]], "step": step, "ticks": int(on.sum()), "misfit_px": misfit}
    return (float(fit[1]), float(fit[0])), found


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
        steps = (np.diff(r) for r in padded)
        return np.array([max([0, *(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))]) for d in steps])

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
        if 4 <= h <= 10 and 4 <= w <= 10 and part.sum() < 0.75 * h * w:
            if ndimage.binary_fill_holes(part).sum() > part.sum():
                rings[sl] |= part
    return rings, mask & ~rings


def _curve(mask, gap=0):
    """The median row of ``mask`` in each column, with gaps up to ``gap`` columns closed by linear interpolation."""
    cols = np.flatnonzero(mask.any(0))
    rows = np.array([np.median(np.flatnonzero(mask[:, c])) for c in cols])
    if gap < 2 or len(cols) < 2:
        return cols, rows
    full = np.arange(cols[0], cols[-1] + 1)
    after = np.clip(np.searchsorted(cols, full), 1, len(cols) - 1)
    keep = np.isin(full, cols) | (cols[after] - cols[after - 1] <= gap)
    return full[keep], np.interp(full[keep], cols, rows)


def _sample(points, *arrays):
    """Every ``ceil(n / points)``-th entry and the last, at five significant figures."""
    n = len(arrays[0])
    index = np.unique(np.r_[np.arange(0, n, max(1, int(np.ceil(n / points)))), n - 1]) if n else np.arange(0)
    return [[float(f"{v:.5g}") for v in np.asarray(arr)[index]] for arr in arrays]


def digitize(image: Path, figure: int, table: dict | None = None) -> dict:
    """One figure's curves, each the median row per pixel column, and the band of the codes that agree, in full."""
    from PIL import Image

    spec_, data = (table or FIGURES)[figure], Path(image).read_bytes()
    a = np.asarray(Image.open(image).convert("RGB")).astype(float) / 255.0
    jpeg = Path(image).suffix.lower() in (".jpg", ".jpeg")  # JPEG washes thin lines and ticks out
    frame = _frame(a)
    legend = _legend(a, frame)
    log = spec_.get("log", False)
    dark = 0.6 if jpeg else 0.45
    (x0, sx), cx = _calibration(spec_["x"], _ticks(a, frame, "x", dark), frame[2:])
    (y0, sy), cy = _calibration(spec_["y"], _ticks(a, frame, "y", dark), frame[:2], log)
    masks = _masks(a, spec_.get("colours", COLOURS), frame, legend, 0.1 if jpeg else 0.0)
    if spec_.get("circles"):
        masks["analytical"], masks["CFX"] = _circles(masks["CFX"])
    if "rows" in spec_:
        masks["FLUENT_CIEMAT"][slice(*spec_["rows"])] = False
    omitted = dict(spec_.get("omit", {}))

    def values(cols, rows):
        y = y0 + sy * np.asarray(rows, float)
        return x0 + sx * np.asarray(cols, float), (10.0**y if log else y)

    curves = {}
    for name, mask in masks.items():
        if name in omitted:
            continue
        cols, rows = _curve(mask)
        coverage = round(len(cols) / (frame[3] - frame[2]), 3)
        if coverage < 0.01:
            omitted[name] = "not visible: under 1 % of the columns"
            continue
        curves[name] = dict(zip(("x", "y"), _sample(60, *values(cols, rows))), coverage=coverage)
    record = {"figure": figure, "quantity": spec_["quantity"], "pdf_page": spec_["pdf_page"], "xref": spec_["xref"],
              "sha256": hashlib.sha256(data).hexdigest(), "pixels": list(a.shape[:2]), "frame": list(frame),
              "legend": None if legend is None else [int(v) for v in legend], "calibration": {"x": cx, "y": cy},
              "y_scale": "log10" if log else "linear", "per_pixel": [abs(sx), abs(sy)], "agree": list(spec_["agree"]),
              "agree_source": spec_["cite"], "window": spec_.get("window"), "omitted": omitted, "curves": curves}  # fmt: skip
    agree = [name for name in spec_["agree"] if name in curves]
    if agree:
        rows = {}
        for name in agree:
            for col, row in zip(*_curve(masks[name], DASH)):
                rows.setdefault(int(col), []).append(row)
        cols = np.array(sorted(rows))
        if spec_.get("window"):
            x = values(cols, np.zeros(len(cols)))[0]
            cols = cols[(x >= spec_["window"][0]) & (x <= spec_["window"][1])]
        lower = [max(rows[c]) for c in cols]  # rows grow downward: the largest row is the lowest value
        upper = [min(rows[c]) for c in cols]
        x, lo = values(cols, lower)
        band = _sample(100, x, lo, values(cols, upper)[1], [len(rows[c]) for c in cols])
        record["band"] = dict(zip(("x", "lower", "upper", "codes"), band))
    return record


def _image(images: Path, xref: int) -> Path:
    found = sorted(Path(images).glob(f"x{xref}.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"need exactly one image x{xref}.* in {images}, found {found}")
    return found[0]


def _simplify(cols, *rows, tolerance=0.5):
    """The columns to keep so that every series stays within ``tolerance`` px of its linear interpolation."""
    cols, rows = np.asarray(cols, float), [np.asarray(r, float) for r in rows]
    keep, stack = {0, len(cols) - 1}, [(0, len(cols) - 1)]
    while stack:
        i, j = stack.pop()
        if j > i + 1:
            t = (cols[i + 1 : j] - cols[i]) / (cols[j] - cols[i])
            error = np.max([np.abs(r[i + 1 : j] - r[i] - t * (r[j] - r[i])) for r in rows], axis=0)
            if error.max() > tolerance:
                k = i + 1 + int(np.argmax(error))
                keep.add(k)
                stack += [(i, k), (k, j)]
    return np.array(sorted(keep))


def _compact(record: dict) -> dict:
    """The record as shipped: the band and the curves that are not one of the six codes (the analytical markers,
    stations, branches), each simplified to within half a pixel and rounded to a tenth of one; the per-code curves
    are dropped, since this script regenerates them."""
    (sx, sy), log = record["per_pixel"], record["y_scale"] == "log10"

    def pixels(values):
        return np.log10(values) / sy if log else np.asarray(values) / sy

    def rounded(values, step):
        digits = max(0, int(np.ceil(-np.log10(step / 10.0))))
        return [round(float(v), digits) if not log else float(f"{v:.5g}") for v in values]

    out = {k: v for k, v in record.items() if k not in ("curves", "band")}
    out["calibration"] = {axis: dict(fit) for axis, fit in record["calibration"].items()}
    spread = {}
    for axis, scale in zip("xy", record["per_pixel"]):
        fit = out["calibration"][axis]
        fit["misfit"] = float(f"{fit['misfit_px'] * scale:.3g}")  # in the axis's units, decades on a log axis
        spread[axis] = READ_PX + COMPACT_PX + fit["misfit_px"]
    out["uncertainty"] = {
        axis: float(f"{spread[axis] * scale:.3g}") for axis, scale in zip("xy", record["per_pixel"])
    }
    out["uncertainty"]["px"] = {axis: round(value, 2) for axis, value in spread.items()}
    out["agree_basis"] = "my reading" if "as read" in record["agree_source"] else "the paper"
    out["curves"] = {}
    for name, curve in record["curves"].items():
        if name in SIX:
            continue
        index = _simplify(np.asarray(curve["x"]) / sx, pixels(curve["y"]))
        out["curves"][name] = {"x": rounded(np.asarray(curve["x"])[index], sx),
                               "y": rounded(np.asarray(curve["y"])[index], sy), "coverage": curve["coverage"]}  # fmt: skip
    if "band" in record:
        band = {k: np.asarray(v) for k, v in record["band"].items()}
        index = _simplify(band["x"] / sx, pixels(band["lower"]), pixels(band["upper"]), 2.0 * band["codes"])
        out["band"] = {"x": rounded(band["x"][index], sx), "lower": rounded(band["lower"][index], sy),
                       "upper": rounded(band["upper"][index], sy), "codes": [int(c) for c in band["codes"][index]]}  # fmt: skip
    return out


def write_references(
    images: Path, path: Path = REFERENCES, figures: dict = FIGURES, case: str = "case1"
) -> Path:
    """Digitize every figure from ``images`` (``x<xref>.png`` or ``.jpeg``, from the PDF) into ``path``, compact."""
    records = [
        _compact(digitize(_image(images, spec_["xref"]), figure, figures))
        for figure, spec_ in figures.items()
    ]
    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    head = json.dumps(
        {"source_id": "mistrangelo-2025", "case": case, "digitizer": "scripts/digitize_mistrangelo.py"}
    )
    path.write_text(head[:-1] + ',"figures":[\n' + ",\n".join(lines) + "\n]}\n")
    return path


def write_curves(images: Path, path: Path, figures: dict = FIGURES, case: str = "case1") -> Path:
    """Every code's curve, each figure's calibration and image hash: the extraction's intermediate, which is a
    release asset beside the figure images (CC BY 4.0) rather than a file of the package."""
    records = [digitize(_image(images, spec_["xref"]), figure, figures) for figure, spec_ in figures.items()]
    head = {"source_id": "mistrangelo-2025", "case": case, "digitizer": "scripts/digitize_mistrangelo.py"}
    path.write_text(json.dumps({**head, "figures": records}, indent=1) + "\n")
    return path


def main() -> None:
    images = Path(sys.argv[1])
    print(write_references(images))
    print(write_curves(images, images / "mistrangelo2025-case1-curves.json"))


if __name__ == "__main__":
    main()
