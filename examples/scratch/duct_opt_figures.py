"""Paper-ready figures and tables for the Stage 2 duct-optimization proof of
concept (stage_2_plan.txt Section 5B). Reads ONLY artifacts/duct_opt/results.json
(measures nothing itself, per plan Section 6/2.2's rule) and writes, per item,
a vector PDF, a 300-dpi PNG, a CSV twin of the plotted data, and (for tables)
a LaTeX booktabs file. Also writes captions.md.

Run with:
    .venv/bin/python examples/scratch/duct_opt_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "artifacts" / "duct_opt"
FIG_DIR = DATA_DIR / "figures"
TAB_DIR = DATA_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ house style
# Palette: dataviz skill reference instance (light mode, white page), validated
# 2026-09-24 with scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a"
# --mode light --surface #ffffff --pairs all (all checks pass; aqua below 3:1
# contrast, so it always carries a marker + direct label here).
COLOR = {"R_outboard": "#2a78d6", "R_inboard": "#eb6834", "P": "#1baf7a", "square": "#898781"}
MARKER = {"R_outboard": "o", "R_inboard": "s", "P": "^", "square": "D"}
LABEL = {"R_outboard": "Case R, Outboard", "R_inboard": "Case R, Inboard", "P": "Case P, Poloidal"}
LINESTYLE = {
    "R_outboard": "-",
    "R_inboard": "-",
    "P": "--",
    "square": ":",
}  # a 2nd style once >2 lines share an axis
SEQ_BLUE = plt.cm.colors.LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#0d366b"])
DIVERGING = plt.cm.colors.LinearSegmentedColormap.from_list("div_br", ["#0d366b", "#f0efec", "#8a1414"])
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"

MM_PER_IN = 25.4
SINGLE_COL = 90.0 / MM_PER_IN
DOUBLE_COL = 190.0 / MM_PER_IN

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK_MUTED,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.0,
        "lines.markeredgecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed as Type 42, not a bitmap font
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "axes.axisbelow": True,
    }
)

CAPTIONS: list[str] = []


def load_results() -> dict:
    with (DATA_DIR / "results.json").open() as fh:
        return json.load(fh)


def _panel_letter(ax, letter: str, dx: float = -0.18) -> None:
    """``dx`` moves the letter further left (more negative) for panels whose y-axis label
    is long enough to reach past the default offset -- e.g. a two-line/wide unit label."""
    ax.text(
        dx, 1.0, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left"
    )


def _value_ticks(ax, axis: str, values, fmt: str = "{:g}") -> None:
    """Add minor ticks labeled with the exact swept values (e.g. the sampled
    V_min or Ha^* points), layered on top of the existing major-decade grid so
    the reader can read a point's coordinate directly without disturbing the
    house-style grid spacing, which stays tied to the major (decade) ticks.
    Values that already coincide with an existing major tick (e.g. a sweep
    point that lands exactly on a decade) are skipped so the two labels don't
    print on top of each other."""
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    major_locs = axis_obj.get_majorticklocs()
    vals = sorted(set(round(float(v), 12) for v in values))
    vals = [v for v in vals if v > 0 and not any(np.isclose(v, m, rtol=1e-6) for m in major_locs)]
    axis_obj.set_minor_locator(mticker.FixedLocator(vals))
    axis_obj.set_minor_formatter(mticker.FixedFormatter([fmt.format(v) for v in vals]))
    ax.tick_params(axis=axis, which="minor", labelsize=6)
    ax.grid(visible=False, which="minor", axis=axis)


def _value_grid(ax, axis: str, values, fmt: str = "{:g}") -> None:
    """Replace the default log-decade major ticks outright with ticks at
    exactly the swept values, so the grid lines land on the data points
    themselves. Use this instead of _value_ticks when a sweep point sits
    close enough to a decade (e.g. Ha*=1228 next to 10^3) that the two
    labels would visually collide even though they aren't equal."""
    vals = sorted(set(round(float(v), 12) for v in values))
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    axis_obj.set_major_locator(mticker.FixedLocator(vals))
    axis_obj.set_major_formatter(mticker.FixedFormatter([fmt.format(v) for v in vals]))
    axis_obj.set_minor_locator(mticker.NullLocator())


def axis_label(name: str, symbol: str = "", unit: str = "") -> str:
    """ "Label, $Symbol$ [unit]" (house style): descriptive name in title case,
    the mathematical symbol in italics via mathtext, the unit in brackets.
    ``unit=""`` for a dimensionless quantity omits the bracket entirely;
    ``symbol=""`` omits the comma-symbol clause (e.g. for a bare percentage).
    """
    label = name
    if symbol:
        label += f", ${symbol}$"
    if unit:
        label += f" [{unit}]"
    return label


def save_figure(fig, name: str, rows: list[dict], caption: str) -> None:
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    csv_path = FIG_DIR / f"{name}.csv"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    if rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    CAPTIONS.append(f"**{name}.** {caption}")
    print(f"  wrote {pdf_path.name}, {png_path.name}, {csv_path.name}")


def save_table(name: str, header: list[str], rows: list[list], caption: str) -> None:
    csv_path = TAB_DIR / f"{name}.csv"
    tex_path = TAB_DIR / f"{name}.tex"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    align = "l" + "r" * (len(header) - 1)
    with tex_path.open("w") as fh:
        fh.write(f"% {caption}\n")
        fh.write("\\begin{tabular}{" + align + "}\n\\toprule\n")
        fh.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for row in rows:
            fh.write(" & ".join(str(cell) for cell in row) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    CAPTIONS.append(f"**{name}.** {caption}")
    print(f"  wrote {csv_path.name}, {tex_path.name}")


# --------------------------------------------------------------------- figures
def figure_f1(results: dict) -> None:
    """Four panels, laid out 2x2 so each has room to breathe:
    (a) a locator -- where the two orientations sit in the machine;
    (b) a zoom on the outboard blanket showing Case R and Case P together,
        in their true relative orientation (radial vs. poloidal), which a
        single R-Z plot could not do without placing Case P at a meaningless
        R = 0 (a mistake in the first version of this figure, caught in
        review: it drew Case P as a bar at the machine axis, where nothing
        physical is);
    (c) the duct cross-section convention;
    (d) the field magnitude each case's duct actually sees.
    """
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.56))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1], hspace=0.55, wspace=0.32)
    demo = results["inputs"]["demo_geometry"]
    b_fw = results["inputs"]["B_fw_T"]
    R0 = demo["R0_m"]
    r_out_fw, r_in_fw = demo["R_outboard_fw_m"], demo["R_inboard_fw_m"]
    depth_out, depth_in = demo["blanket_depth_outboard_m"], demo["blanket_depth_inboard_m"]
    Rc = r_out_fw + 0.5 * depth_out  # Case P duct centre, mid-depth of the outboard blanket

    # (a) locator: where the whole machine sits, and which part panel (b) zooms into
    ax = fig.add_subplot(gs[0, 0])
    a_minor = 0.5 * (r_out_fw - r_in_fw)
    ax.add_patch(Circle((R0, 0), a_minor, fill=False, ec=INK_SECONDARY, lw=0.75, ls=(0, (3, 2))))
    ax.text(R0, a_minor + 0.15, "Plasma Boundary", color=INK_SECONDARY, fontsize=6, ha="center", va="bottom")
    ax.add_patch(
        Wedge((0, 0), r_out_fw + depth_out, -6, 6, width=depth_out, color=COLOR["R_outboard"], alpha=0.35)
    )
    ax.add_patch(Wedge((0, 0), r_in_fw, -6, 6, width=depth_in, color=COLOR["R_inboard"], alpha=0.35))
    zoom = plt.Rectangle(
        (r_out_fw - 0.3, -1.3), depth_out + 0.6, 2.6, fill=False, ec=INK, lw=0.75, ls=(0, (2, 1.5))
    )
    ax.add_patch(zoom)
    # labels sit BELOW their shapes, on white, not on top of the same-colored fill (low
    # contrast there) -- both pushed clear of the wedges and the dashed zoom box
    ax.text(
        r_out_fw + depth_out / 2,
        -1.9,
        "Outboard\nBlanket",
        color=COLOR["R_outboard"],
        fontsize=6,
        ha="center",
    )
    ax.text(
        r_in_fw - depth_in / 2, -1.15, "Inboard\nBlanket", color=COLOR["R_inboard"], fontsize=6, ha="center"
    )
    ax.annotate(
        "zoom in (b)",
        xy=(r_out_fw + depth_out + 0.3, 1.3),
        xytext=(r_out_fw + depth_out + 1.0, 2.6),
        fontsize=6,
        ha="left",
        arrowprops=dict(arrowstyle="->", lw=0.6, color=INK),
    )
    ax.set_xlim(0, r_out_fw + depth_out + 2.6)
    ax.set_ylim(-3.6, 4.3)
    # not aspect='equal' here: this panel is already schematic (see caption), and letting it
    # fill its grid cell avoids the dead space a locked aspect ratio left around it
    ax.set_xlabel(axis_label("Major Radius", "R", "m"))
    ax.set_ylabel(axis_label("Height", "Z", "m"))
    ax.grid(False)
    _panel_letter(ax, "a")

    # (b) the zoom: Case R (radial) and Case P (poloidal) drawn in their real relative
    # orientation, both sitting inside the outboard blanket band
    ax = fig.add_subplot(gs[0, 1])
    ax.add_patch(
        plt.Rectangle(
            (r_out_fw, -1.3), depth_out, 2.6, fill=True, fc=COLOR["R_outboard"], alpha=0.12, ec="none"
        )
    )
    # thinner than the first version: these bars mark ORIENTATION in space, not a solved
    # 3-D volume (see caption -- the actual solves are 2-D cross-sections at stations
    # along each axis, "locally fully developed"; only the F7 validity check is true 3-D)
    ax.plot(
        [r_out_fw, r_out_fw + depth_out], [0, 0], color=COLOR["R_outboard"], lw=2.2, solid_capstyle="round"
    )
    ax.plot([Rc, Rc], [-0.8, 0.8], color=COLOR["P"], lw=2.2, solid_capstyle="round")
    ax.annotate(
        # anchored at a QUARTER of the way along the blue bar, not its midpoint -- the
        # midpoint coincides with Rc, the green bar's x-position, which sent this arrow
        # straight down alongside it (the same mistake fixed for the Case P arrow above)
        "Case R (Radial)",
        xy=(r_out_fw + depth_out * 0.25, 0.08),
        xytext=(r_out_fw + depth_out * 0.25, 1.55),
        color=COLOR["R_outboard"],
        fontsize=6.5,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", lw=0.6, color=COLOR["R_outboard"]),
    )
    # approach horizontally, at the SAME height as the target point, so the arrow only
    # touches the green line at its tip instead of running alongside it
    ax.annotate(
        "Case P (Poloidal)",
        xy=(Rc, 0.5),
        xytext=(r_out_fw + depth_out + 0.55, 0.5),
        color=COLOR["P"],
        fontsize=6.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.6, color=COLOR["P"]),
    )
    ax.annotate(
        "",
        xy=(r_out_fw - 0.05, -1.55),
        xytext=(r_out_fw + 0.7, -1.55),
        arrowprops=dict(arrowstyle="->", lw=0.5, color=INK_MUTED),
    )
    ax.text(r_out_fw + 0.75, -1.55, "toward plasma", fontsize=5.5, color=INK_MUTED, ha="left", va="center")
    ax.set_xlim(r_out_fw - 0.3, r_out_fw + depth_out + 1.7)
    ax.set_ylim(-1.8, 2.2)
    # not aspect='equal' here either, for the same reason as panel (a) -- see caption
    ax.set_xlabel(axis_label("Major Radius", "R", "m"))
    ax.set_ylabel(axis_label("Height", "Z", "m"))
    ax.grid(False)
    _panel_letter(ax, "b")

    # (c) duct cross-section convention (aspect exaggerated for legibility, not to scale)
    ax = fig.add_subplot(gs[1, 0])
    half_a, half_b = 1.0, 0.55
    ax.add_patch(plt.Rectangle((-half_a, -half_b), 2 * half_a, 2 * half_b, fill=False, ec=INK, lw=1.0))
    # b (height, across B): dimension line to the right of the duct
    ax.annotate("", xy=(1.35, -half_b), xytext=(1.35, half_b), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(1.45, 0, "$b$", va="center", ha="left", fontsize=7)
    # a (width, along B): dimension line below the duct
    ax.annotate("", xy=(half_a, -0.85), xytext=(-half_a, -0.85), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(0, -1.02, "$a$", ha="center", va="top", fontsize=7)
    # B direction: vertical arrow through the centre, label directly to its right at mid-height
    ax.annotate(
        "",
        xy=(0, half_b * 0.75),
        xytext=(0, -half_b * 0.75),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=INK),
    )
    ax.text(0.12, 0, "$B$", fontsize=7, va="center", ha="left")
    # Hartmann walls (top, normal to B): callout above, with an arrowhead so it reads as
    # "pointing at the wall", not a stray line
    ax.annotate(
        "Hartmann Wall\n(Thickness $\\sim a/Ha$)",
        xy=(0.35, half_b),
        xytext=(0.35, 1.05),
        ha="center",
        va="bottom",
        fontsize=6,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", lw=0.6, color=INK_SECONDARY),
    )
    # side walls: callout to the LEFT (the b-dimension arrow already occupies the right side)
    ax.annotate(
        "Side Wall\n($\\sim a/\\sqrt{Ha}$)",
        xy=(-half_a, 0.25),
        xytext=(-1.95, 0.55),
        ha="right",
        va="center",
        fontsize=6,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", lw=0.6, color=INK_SECONDARY),
    )
    # generous, EQUAL spacing top to bottom: the "a" label, the not-to-scale note, and the
    # optimum callout are laid out from one shared gap rather than three guessed offsets
    row_gap = 0.75
    a_label_y = -1.02
    note_y = a_label_y - row_gap
    ax.text(0, note_y, "(schematic aspect; not to scale)", ha="center", fontsize=5.5, color=INK_MUTED)

    # the optimum, drawn TO SCALE below the labelled schematic (its own row, clear of
    # every callout above it), so the elongation along B is a one-glance check against
    # F2's design law rather than something to infer from a number in a table
    # (plan default case, Case R outboard at V_min = 10 mm/s)
    beta_star = next(
        r["beta"] for r in results["case_r_outboard"]["pareto"] if abs(r["V_min"] - 0.010) < 1e-9
    )
    opt_half_a, opt_half_b = 0.75, 0.75 * beta_star
    opt_label_y = note_y - row_gap
    box_top = opt_label_y - 0.3
    opt_cy = box_top - opt_half_b
    ax.text(0, opt_label_y, f"Optimum, $\\beta^*\\approx{beta_star:.2f}$ (to scale)", ha="center", fontsize=6)
    ax.add_patch(
        plt.Rectangle(
            (-opt_half_a, opt_cy - opt_half_b),
            2 * opt_half_a,
            2 * opt_half_b,
            fill=True,
            fc=COLOR["R_outboard"],
            ec=INK,
            lw=1.0,
            alpha=0.3,
        )
    )

    ax.set_xlim(-2.3, 1.9)
    ax.set_ylim(opt_cy - opt_half_b - 0.3, 1.4)
    # not aspect='equal' here either -- same reasoning as (a)/(b): this panel is already
    # explicitly "not to scale", and a locked aspect only left dead space around it
    ax.axis("off")
    _panel_letter(ax, "c")

    # (d) field magnitude each case's duct actually sees
    ax = fig.add_subplot(gs[1, 1])
    xs_out = np.linspace(0, depth_out, 30)
    ax.plot(
        xs_out,
        b_fw * r_out_fw / (r_out_fw + xs_out),
        color=COLOR["R_outboard"],
        ls=LINESTYLE["R_outboard"],
        label=LABEL["R_outboard"],
    )
    xs_in = np.linspace(0, depth_in, 30)
    ax.plot(
        xs_in,
        b_fw * r_in_fw / (r_in_fw - xs_in),
        color=COLOR["R_inboard"],
        ls=LINESTYLE["R_inboard"],
        label=LABEL["R_inboard"],
    )
    b_center = b_fw * r_out_fw / Rc
    zs = np.linspace(-0.02, 0.02, 30)
    ax.plot(
        zs / 0.02 * 0.982 * 0.05 + 0.5 * depth_out,
        b_center * Rc / (Rc + zs),
        color=COLOR["P"],
        ls=LINESTYLE["P"],
        label=LABEL["P"] + " (cross-duct)",
    )
    ax.set_xlabel(axis_label("Distance Into Blanket / Across Duct", "x", "m"))
    ax.set_ylabel(axis_label("Magnetic Field Magnitude", "|B|", "T"))
    # a FIGURE-level legend (not attached to this axes), sitting in the wide dead space
    # between row 0 (panels a/b) and row 1 (panels c/d) that panel (b) leaves open --
    # attaching it to ax directly shrank panel (d)'s own axes to make room for it, which
    # pushed the x-axis label off the bottom of the figure (a caught mistake)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="center", bbox_to_anchor=(0.75, 0.46), ncol=1, fontsize=6.5
    )
    _panel_letter(ax, "d", dx=-0.34)  # a longer y-label than the other panels needs more room

    caption = (
        f"Problem geometry. (a) Locator: a schematic tokamak cross-section (illustrative, "
        f"not an equilibrium), $R_0={R0:.1f}$ m, first-wall radii {r_in_fw:.1f}/{r_out_fw:.1f} "
        f"m (inboard/outboard), blanket depths {depth_in:.3f}/{depth_out:.3f} m (Federici et "
        "al. 2019, Nucl. Fusion 59:066013 [M1]); the dashed box is panel (b)'s extent. "
        "(b) Case R (radial) and Case P (poloidal) drawn in their true relative orientation "
        "inside the outboard blanket; the bars mark ORIENTATION, not a solved 3-D volume -- "
        "every optimization result in this study comes from a series of independent 2-D "
        "cross-section solves at stations along each duct's axis (\"locally fully "
        'developed"), not a single 3-D solve; Fig. F7 checks that approximation against '
        "genuine 3-D solves. Case P's height is illustrative, not a sourced poloidal module "
        "length (plan Section 0.6). (c) Duct cross-section convention: $a$ along $B$ "
        "(Hartmann walls), $b$ across it (side walls); the shaded rectangle is the actual "
        f"optimum (Case R outboard, default $V_\\mathrm{{min}}$), drawn to scale. "
        f"(d) $|B|$ along the two Case R ducts and across the Case P duct at "
        f"$B_\\mathrm{{fw}}={b_fw:.1f}$ T (lab-scale reference, plan Section 4.4)."
    )
    save_figure(fig, "F1_geometry", [], caption)


def figure_f2(results: dict) -> None:
    rows = results["design_law"]["rows"]
    high = results["design_law"].get("high_ha", [])
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, SINGLE_COL * 1.35), sharex=True)
    ha_star = np.array([r["Ha_star"] for r in rows])
    s_star = np.array([r["s_star"] for r in rows])
    # s* = beta^(3/4) H^(1/2) at fixed area, so a relative GCI on beta is 3/4 of it on s*
    gci_err = np.array(
        [0.75 * r["gci"]["beta"]["gci_fine"] * r["s_star"] if r.get("gci") else 0.0 for r in rows]
    )
    ref_ha = [r["spectral"]["Ha_star"] for r in rows + high]
    ref_s = [r["spectral"]["s_star"] for r in rows + high]

    ax = axes[0]
    ax.errorbar(
        ha_star,
        s_star,
        yerr=gci_err,
        fmt="o",
        color=COLOR["R_outboard"],
        ecolor=INK_MUTED,
        capsize=2,
        label="Core, 48/6 Cells",
        markeredgecolor="white",
        zorder=3,
    )
    ax.plot(
        [r["Ha_star"] for r in high],
        [r["s_star"] for r in high],
        "o",
        color=COLOR["R_outboard"],
        mfc="none",
        mec=COLOR["R_outboard"],
        mew=1.3,
        markersize=7,
        label="Core, Reactor-Scale Ha",
        zorder=3,
    )
    ax.plot(ref_ha, ref_s, "s", color=COLOR["R_inboard"], markersize=4, label="Spectral Reference", zorder=2)
    ax.axhline(2.13, color=INK_SECONDARY, ls="--", lw=1.0)
    ax.annotate("Shercliff Formula, 2.13 [M3]", (ha_star[1], 2.135), fontsize=6, color=INK_SECONDARY)
    ax.set_ylim(1.97, 2.15)
    ax.set_xscale("log")
    ax.set_ylabel(axis_label("Optimal Aspect Ratio", "\\beta^*\\sqrt{Ha^*}"))
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    _panel_letter(ax, "a", dx=-0.22)

    ax = axes[1]
    ax.plot(ha_star, [100 * r["reduction"] for r in rows], "o-", color=COLOR["R_outboard"])
    ax.plot(
        [r["Ha_star"] for r in high],
        [100 * r["reduction"] for r in high],
        "o",
        color=COLOR["R_outboard"],
        mfc="none",
        mec=COLOR["R_outboard"],
        mew=1.3,
        markersize=7,
    )
    ax.set_xscale("log")
    ax.set_xlabel(axis_label("Hartmann Number at the Optimum", "Ha^*"))
    ax.set_ylabel(axis_label("Pressure-Drop Reduction vs. Square Duct", unit="%"))
    _panel_letter(ax, "b", dx=-0.26)
    for ax in axes:
        ax.grid(True, which="major", axis="x")

    rows_csv = [
        {
            "H": r["H"],
            "Ha_star": r["Ha_star"],
            "beta_star_48_6": r["beta_star"],
            "s_star_48_6": r["s_star"],
            "s_star_reference": r["spectral"]["s_star"],
            "reduction_pct": 100 * r["reduction"],
            "gci_beta": r["gci"]["beta"]["gci_fine"] if r.get("gci") else None,
        }
        for r in rows + high
    ]
    gci_points = "$, $".join(f"{r['Ha_star']:.0f}" for r in rows if r.get("gci"))
    caption = (
        "The design law. (a) The optimal aspect ratio $\\beta^*\\sqrt{Ha^*}$ at fixed area, from the core (filled, "
        "48/6 cells; error bars are the three-mesh GCI at the observed order, 32/4, 48/6, 72/9 cells, at "
        f"$Ha^*={gci_points}$) and from the independent spectral "
        "reference (squares), with the reactor-scale points $Ha^*\\approx2{,}400$ and $6{,}100$ (hollow, 48/6 "
        "cells). The reference tends to $\\approx2.08$; Shercliff's formula as given in [M3] gives 2.13. "
        "The optimum at fixed flow rate and area was also computed by Nishio et al. 2025 for Ha up to $6.5\\times10^4$ "
        "(plan, novelty gate). (b) The pumping-power reduction over a square duct of equal area."
    )
    save_figure(fig, "F2_design_law", rows_csv, caption)


def figure_f3(results: dict) -> None:
    cases_ = [("case_r_outboard", "R_outboard"), ("case_r_inboard", "R_inboard"), ("case_p", "P")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.36), sharex=True, sharey=True)
    csv_rows = []
    for ax, (key, tag) in zip(axes, cases_):
        case = results[key]
        landscape = case["landscape"]
        betas = sorted(set(row["beta"] for row in landscape))
        vs = sorted(set(row["V"] for row in landscape))
        W = np.array(
            [[next(r["W"] for r in landscape if r["beta"] == b and r["V"] == v) for b in betas] for v in vs]
        )
        # normalize each row (each V) by its own max over beta, so the valley shape -- the
        # whole point of the figure -- is visible at every V, not swamped by the ~100x range
        # of W across V (an earlier version normalized by one fixed row and showed a flat
        # panel below V ~ 0.1; caught and fixed, noted in stage_2_plan.txt)
        cs = ax.contourf(betas, vs, W / W.max(axis=1, keepdims=True), levels=8, cmap=SEQ_BLUE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        v_min_default = [row["V_min"] for row in case["pareto"]]
        for row in case["pareto"]:
            ax.axhline(row["V_min"], color=INK, lw=0.4, alpha=0.5)
        _value_ticks(ax, "y", v_min_default, fmt="{:g}")
        if "demo_optimizer" in case:
            path = case["demo_optimizer"]["path"]
            Q = results["inputs"]["Q_m3s"]
            ws_path = [path[0]["w_center"]] + [p["w_next"] for p in path]
            us_path = [p["u"] for p in path] + [path[-1]["u"]]
            betas_path = np.exp(ws_path)
            v_path = [Q / np.exp(u) for u in us_path]
            ax.plot(
                betas_path,
                v_path,
                color=COLOR[tag],
                lw=1.0,
                marker="o",
                markersize=3,
                markeredgecolor="white",
            )
            ax.plot(
                betas_path[0], v_path[0], marker="s", color=COLOR[tag], markeredgecolor="white", markersize=6
            )
            ax.plot(
                betas_path[-1],
                v_path[-1],
                marker="*",
                color=COLOR[tag],
                markeredgecolor="white",
                markersize=9,
            )
        ax.set_xlabel(axis_label("Aspect Ratio", "\\beta=b/a"))
        ax.set_title(LABEL[tag], fontsize=7, color=COLOR[tag])
        for row in landscape:
            csv_rows.append({"case": tag, "beta": row["beta"], "V": row["V"], "W": row["W"]})
    axes[0].set_ylabel(axis_label("Mean Velocity", "V=Q/A", "m/s"))
    fig.colorbar(
        cs, ax=axes, shrink=0.6, label="$W(\\beta)\\,/\\,\\max_\\beta W$ (Per Row)", location="right"
    )
    # the demo-optimizer path markers (square/star) had no legend entry in the first version
    # -- readable only from the caption, which a reader glancing at the figure alone would
    # miss; proxy handles give them one directly on the figure
    marker_handles = [
        Line2D(
            [],
            [],
            color=COLOR["R_outboard"],
            marker="s",
            ls="none",
            markeredgecolor="white",
            markersize=6,
            label="Optimizer Start",
        ),
        Line2D(
            [],
            [],
            color=COLOR["R_outboard"],
            marker="*",
            ls="none",
            markeredgecolor="white",
            markersize=9,
            label="Optimizer End (Optimum)",
        ),
    ]
    axes[0].legend(handles=marker_handles, frameon=False, loc="upper left", fontsize=5.5)
    caption = (
        "Pumping-power landscapes over aspect ratio and mean velocity, for the "
        "three cases at their default $V_\\mathrm{min}$. Horizontal lines mark "
        "the $V_\\mathrm{min}$ values of the Pareto sweep (Fig. F5); the square and "
        "star mark the start and converged end of the demonstration optimizer's "
        "path (Case R outboard only, plan step 0.4; see also Fig. F4)."
    )
    save_figure(fig, "F3_landscape", csv_rows, caption)


def figure_f4(results: dict) -> None:
    demo = results["case_r_outboard"]["demo_optimizer"]
    path = demo["path"]
    fig, axes = plt.subplots(
        2, 1, figsize=(SINGLE_COL, SINGLE_COL * 1.25), sharex=True, gridspec_kw={"hspace": 0.2}
    )
    fig.get_layout_engine().set(h_pad=0.08)
    W0 = path[0]["W"]
    rounds = [p["round"] for p in path]
    ax = axes[0]
    ax.plot(rounds, [p["W"] / W0 for p in path], "o-", color=COLOR["R_outboard"])
    ax.set_ylabel(axis_label("Pumping Power (Normalized)", "W/W_0"))
    _panel_letter(ax, "a", dx=-0.30)
    ax = axes[1]
    step = [abs(p["w_next"] - p["w_center"]) for p in path]
    ax.plot(rounds, step, "o-", color=COLOR["R_outboard"])
    ax.axhline(1e-3, color=INK_SECONDARY, ls="--", lw=1.0)
    ax.annotate("Stopping Tolerance", (rounds[-1] * 0.5, 1.3e-3), fontsize=6, color=INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_xlabel(axis_label("Outer Round"))
    ax.set_ylabel(axis_label("Bracket Step Size", "|\\Delta w|"))
    _panel_letter(ax, "b", dx=-0.24)
    rows_csv = [
        {
            "round": p["round"],
            "W": p["W"],
            "w_center": p["w_center"],
            "w_next": p["w_next"],
            "interior": p["interior"],
        }
        for p in path
    ]
    caption = (
        "Convergence of the demonstration optimizer (Case R, outboard, default "
        "$V_\\mathrm{min}$): a block-coordinate scheme alternating an exact, "
        "mesh-free area optimum with a quadratic-fit bracket search in aspect "
        "ratio (Nocedal \\& Wright ch. 8 [O7]), starting from a square duct at "
        "the geometric-mean velocity."
    )
    save_figure(fig, "F4_convergence", rows_csv, caption)


def figure_f5(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.1, SINGLE_COL * 1.1))
    rows_csv = []
    all_v = sorted(
        {row["V_min"] for key in ("case_r_outboard", "case_r_inboard") for row in results[key]["pareto"]}
    )
    for key, tag in [("case_r_outboard", "R_outboard"), ("case_r_inboard", "R_inboard"), ("case_p", "P")]:
        pareto = results[key]["pareto"]
        v = [row["V_min"] for row in pareto]
        W = [row["W"] for row in pareto]
        beta = [row["beta"] for row in pareto]
        axes[0].plot(v, W, marker=MARKER[tag], color=COLOR[tag], ls=LINESTYLE[tag], label=LABEL[tag])
        axes[1].plot(v, beta, marker=MARKER[tag], color=COLOR[tag], ls=LINESTYLE[tag], label=LABEL[tag])
        for row in pareto:
            rows_csv.append(
                {
                    "case": tag,
                    "V_min": row["V_min"],
                    "W": row["W"],
                    "beta": row["beta"],
                    "Ha_max": max(s["Ha"] for s in row["validity"]["stations"]),
                    "gamma_leq_0.2_all": all(s["gamma_sqrt_Ha_leq_0.2"] for s in row["validity"]["stations"]),
                    "ReHa_leq_200_all": all(s["Re_over_Ha_leq_200"] for s in row["validity"]["stations"]),
                }
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(axis_label("Minimum Velocity Bound", "V_{\\min}", "m/s"))
        ax.legend(frameon=False, loc="best")
        _value_ticks(ax, "x", all_v, fmt="{:g}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(axis_label("Optimal Pumping Power", "W^*", "W"))
    _panel_letter(axes[0], "a")
    axes[1].set_ylabel(axis_label("Optimal Aspect Ratio", "\\beta^*"))
    axes[1].set_yscale("log")
    _panel_letter(axes[1], "b")
    caption = (
        "Pareto fronts over the minimum-velocity bound $V_\\mathrm{min}$, the "
        "epsilon-constraint proxy for heat removal (Haimes, Lasdon & Wismer 1971 "
        "[O1]; Miettinen 1999 [O4]). (a) Optimal pumping power. (b) Optimal aspect "
        f"ratio. Hartmann numbers reach {max(r['Ha_max'] for r in rows_csv):.0f} at fixed $Q$ (not capped); "
        f"{sum(not (r['gamma_leq_0.2_all'] and r['ReHa_leq_200_all']) for r in rows_csv)} of "
        f"{len(rows_csv)} points fail $\\gamma\\sqrt{{Ha}}\\le0.2$ or $Re/Ha\\le200$ at some station "
        "(Table T4)."
    )
    save_figure(fig, "F5_pareto", rows_csv, caption)


def figure_f6(results: dict) -> None:
    """Case P's exact 1/R cross-duct field against the uniform-field model, in two panels (one y-axis each)."""
    cp_corr = results["case_p_correction"]
    designs = [("Square Duct", cp_corr["square"]), ("Optimum", cp_corr["at_optimum"])]
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.1, SINGLE_COL * 0.95))
    quantities = [
        (
            "flow_ratio",
            "Flow-Rate Change vs. Uniform Field",
            lambda d: 100 * (d["flow_ratio"] - 1.0),
            "{:+.1e}%",
            "P",
        ),
        (
            "flow_centroid_fraction",
            "Flow-Centroid Shift, Fraction of $b$",
            lambda d: 100 * d["flow_centroid_fraction"],
            "{:+.3f}%",
            "R_outboard",
        ),
    ]
    rows_csv = []
    for ax, (_, name, value, fmt, tag) in zip(axes, quantities):
        values = [value(d) for _, d in designs]
        bars = ax.bar(range(2), values, width=0.5, color=COLOR[tag])
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                fmt.format(v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=6.5,
            )
        ax.axhline(0.0, color=INK_SECONDARY, lw=0.6)
        ax.set_xticks(range(2))
        ax.set_xticklabels([f"{label}\n($R_c/a$={d['Rc_over_a']:.0f})" for label, d in designs], fontsize=6.5)
        ax.set_ylabel(axis_label(name, unit="%"))
        ax.margins(y=0.25)
        rows_csv += [
            {"quantity": name, "design": label, "value_pct": v} for (label, _), v in zip(designs, values)
        ]
    _panel_letter(axes[0], "a", dx=-0.3)
    _panel_letter(axes[1], "b", dx=-0.3)
    caption = (
        "Case P's exact curl-free $1/R$ cross-duct field (Petrykowski & Walker 1984 [M7]) against the "
        "uniform-field model the optimization uses, at the equal-area square and at the optimum ($R_c/a=629$). "
        "(a) The flow-rate change, of order $(a/R_c)^2$. (b) The shift of the flow centroid across the duct, weighted "
        "by the cell areas (R2), positive toward the weak-field (outboard) side. It is positive for the square "
        "duct, as in P10, and negative for the optimum; the optimum's sign is mesh-converged (32/4 to 96/12 cells) "
        "and not yet explained. Both are negligible for $\\Delta p$ at the tokamak radius ratio used here."
    )
    save_figure(fig, "F6_case_p_correction", rows_csv, caption)


def figure_f7(results: dict) -> None:
    ramp = results["ramp_excess"]["rows"]
    periodic = [r for r in results["validity_map"]["rows"] if not r.get("refinement_of")]
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.1, SINGLE_COL * 1.05), sharey=True)
    series = [("Open Duct, Ha 50", 50.0, "R_outboard"), ("Open Duct, Ha 200", 200.0, "R_inboard")]
    rows_csv = []
    for ax, case, periodic_case, title in (
        (axes[0], "square", "square", "Square Duct"),
        (axes[1], "beta_star", "beta*", "Design-Law Aspect ($\\beta^*=2.09/\\sqrt{Ha}$)"),
    ):
        for label, ha, key in series:
            pts = [r for r in ramp if r["case"] == case and r["ha_mid"] == ha]
            base = [r for r in pts if r["variant"] == "base"]
            ax.plot(
                [r["gamma_sqrt_ha"] for r in base],
                [r["excess_percent"] for r in base],
                marker=MARKER[key],
                color=COLOR[key],
                label=label,
                markeredgecolor="white",
                zorder=3,
            )
            refined = [r for r in pts if r["variant"] != "base"]
            ax.plot(
                [r["gamma_sqrt_ha"] for r in refined],
                [r["excess_percent"] for r in refined],
                ls="none",
                marker=MARKER[key],
                mfc="none",
                mec=COLOR[key],
                markersize=8,
                zorder=2,
            )
            rows_csv += [
                {
                    "series": label,
                    "case": case,
                    **{
                        k: r[k]
                        for k in (
                            "variant",
                            "gamma_sqrt_ha",
                            "excess_percent",
                            "x0",
                            "axial_cells",
                            "cells",
                            "iterations",
                        )
                    },
                }
                for r in pts
            ]
        per = [r for r in periodic if r["case"] == periodic_case]
        ax.plot(
            [r["gamma_sqrt_ha"] for r in per],
            [abs(r["excess_percent"]) for r in per],
            marker="^",
            color=COLOR["P"],
            lw=1.0,
            label="Periodic, Ha 50",
            markeredgecolor="white",
            zorder=3,
        )
        rows_csv += [
            {
                "series": "Periodic, Ha 50",
                "case": case,
                "variant": f"nx{r['nx']}",
                "gamma_sqrt_ha": r["gamma_sqrt_ha"],
                "excess_percent": r["excess_percent"],
            }
            for r in per
        ]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color=INK_SECONDARY, lw=0.6)
        ax.annotate("1% Threshold", (0.021, 1.15), fontsize=6, color=INK_SECONDARY)
        ax.axvspan(0.5, 2.0, color=INK_MUTED, alpha=0.15)
        ax.annotate("Reactor Estimate\n[M4, M5]", (0.52, 0.0025), fontsize=6, color=INK_SECONDARY)
        ax.set_xlabel(axis_label("Field-Gradient Parameter", "\\gamma\\sqrt{Ha}"))
        ax.set_title(title, fontsize=7)
        ax.legend(frameon=False, loc="upper left", fontsize=6)
    axes[0].set_ylabel(axis_label("3-D Excess Pressure Drop", unit="%"))
    _panel_letter(axes[0], "a")
    _panel_letter(axes[1], "b")
    summary = results["ramp_excess"]["summary"]
    caption = (
        "The validity map: 3-D excess pressure drop over the locally fully developed sum, against "
        "$\\gamma\\sqrt{Ha}$ with $\\gamma=a|dB/dx|/B$ (Tier 0 mislabelled it, R1). Filled: an open duct "
        "in a monotone 20 % sine ramp of $B_y$ (TM-228's field, buffers of 15 and 10 half-widths, the 1.9d "
        "method, Stokes limit). Hollow rings: 72/9 cross-section cells and half the axial spacing. Triangles: "
        "the periodic-modulation construction of Tier 0 at Ha 50 (10 % amplitude, 24 cells); its flattening below "
        "0.01 % is taken to be that mesh's floor, not physics. "
        f"{len(summary['mesh_flagged'])} of {len(summary['mesh_changes'])} refinements move the excess by more "
        f"than {100 * results['meta']['exits']['i_mesh_change_max']:.0f} %. The excess "
        f"{'collapses' if summary['collapsed'] else 'does not collapse'} on $\\gamma\\sqrt{{Ha}}$ across Ha 50 "
        "and 200 (Table T5); the shaded band is the reactor estimate."
    )
    save_figure(fig, "F7_validity_map", rows_csv, caption)


TILT_H_STYLE = {
    30.0: ("D", 0.35),
    100.0: ("s", 0.55),
    300.0: ("o", 0.78),
    1000.0: ("^", 1.0),
}  # marker, blue-ramp level


def _tilt_law_arrays(results: dict) -> dict:
    """Per H: kappa, tilt, beta*/beta0, dp*/dp0 and the aligned-shape/square ratio, all on the finest mesh."""
    out = {}
    for entry in results["tilt_law"]:
        first = entry["rows"][0]["meshes"][-1]
        rows = [(r["tilt"], r["kappa"], r["meshes"][-1]) for r in entry["rows"][1:]]
        out[entry["H"]] = {
            "Ha_star_0": first["Ha_star"],
            "beta0": first["beta_star"],
            "tilt": np.array([t for t, _, _ in rows]),
            "kappa": np.array([k for _, k, _ in rows]),
            "G": np.array([m["beta_star"] / first["beta_star"] for _, _, m in rows]),
            "P": np.array([m["dp_star"] / first["dp_star"] for _, _, m in rows]),
            "aligned_over_square": np.array([m["dp_at_aligned_beta"] / m["dp_square"] for _, _, m in rows]),
            "reduction": np.array([m["reduction"] for _, _, m in rows]),
        }
    return out


def figure_f9(results: dict) -> None:
    data = _tilt_law_arrays(results)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.36))
    kappa = np.geomspace(0.1, 30.0, 200)
    rows_csv = []
    for H, d in data.items():
        marker, level = TILT_H_STYLE[H]
        color = SEQ_BLUE(level)
        label = f"$Ha^*_0={d['Ha_star_0']:.0f}$"
        axes[0].plot(
            d["kappa"], d["G"], marker=marker, color=color, ls="none", label=label, markeredgecolor="white"
        )
        axes[1].plot(d["kappa"], d["P"], marker=marker, color=color, ls="none", markeredgecolor="white")
        axes[2].plot(
            d["tilt"],
            d["aligned_over_square"],
            marker=marker,
            color=color,
            ls="-",
            lw=1.0,
            markeredgecolor="white",
        )
        rows_csv += [
            {
                "H": H,
                "Ha_star_0": d["Ha_star_0"],
                "tilt": t,
                "kappa": k,
                "beta_ratio": g,
                "dp_ratio": p,
                "aligned_shape_over_square": a,
                "reduction": r,
            }
            for t, k, g, p, a, r in zip(
                d["tilt"], d["kappa"], d["G"], d["P"], d["aligned_over_square"], d["reduction"]
            )
        ]
    s0 = 2.08  # the reference's high-Ha limit of beta* sqrt(Ha*)
    axes[0].plot(kappa, np.sqrt(1 + (kappa / s0) ** 2), ls="--", color=INK_SECONDARY, lw=1.0)
    axes[0].annotate(
        "$\\sqrt{1+(\\kappa/2.08)^2}$",
        (0.32, 0.55),
        xycoords="axes fraction",
        fontsize=6.5,
        color=INK_SECONDARY,
    )
    axes[1].plot(
        kappa, np.sqrt(1 + 0.1221 * kappa**2 / (1 + kappa / 3.011)), ls="--", color=INK_SECONDARY, lw=1.0
    )
    axes[1].annotate(
        "$\\sqrt{1+0.122\\kappa^2/(1+\\kappa/3.01)}$",
        (0.05, 0.86),
        xycoords="axes fraction",
        fontsize=6.5,
        color=INK_SECONDARY,
    )
    axes[2].axhline(1.0, color=INK_SECONDARY, lw=0.6)
    axes[2].annotate(
        "Square Duct Wins Above", (0.03, 0.92), xycoords="axes fraction", fontsize=6.5, color=INK_SECONDARY
    )
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(axis_label("Tilt Parameter", "\\kappa=\\sqrt{Ha^*_0}\\,B_p/B_T"))
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(axis_label("Field Tilt", "B_p/B_T"))
    axes[0].set_ylabel(axis_label("Aspect-Ratio Shift", "\\beta^*/\\beta^*_0"))
    axes[1].set_ylabel(axis_label("Pressure-Drop Penalty", "\\Delta p^*/\\Delta p^*_0"))
    axes[2].set_ylabel(axis_label("Aligned Shape vs. Square", "\\Delta p/\\Delta p_\\mathrm{sq}"))
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    for ax, letter in zip(axes, "abc"):
        _panel_letter(ax, letter, dx=-0.27)
    caption = (
        "The tilt-aware design law: the fixed-area optimum of a duct in a field tilted by $B_p/B_T$ in the cross-section "
        "(insulating walls, 32/4, 48/6 and 72/9 cells; the finest mesh is plotted, and 48/6 to 72/9 moves "
        "$\\beta^*$ by at most 0.7 % and $\\Delta p^*$ by at most 0.7 %). (a) The optimal aspect ratio relative to the aligned "
        "one and (b) the penalty of the retuned optimum collapse on $\\kappa=\\sqrt{Ha^*_0}\\,B_p/B_T$ across "
        "$Ha^*_0=58$-$6{,}127$ to within 5 % (dashed: $\\beta^{*2}=\\beta_0^{*2}+(B_p/B_T)^2$ and a two-parameter fit "
        "of the penalty, both fitted here). (c) Keeping the aligned optimum's shape in a tilted field, against the "
        "equal-area square, which barely responds to tilt (0.1-0.5 % at $B_p/B_T\\le0.2$): the aligned shape loses to "
        "the square above the crossing."
    )
    save_figure(fig, "F9_tilt_law", rows_csv, caption)


def figure_f8(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.1, SINGLE_COL * 1.05))
    tt = results["case_r_outboard"]["pareto"][0]["taylor_test"]
    ax = axes[0]
    ax.plot(
        tt["h"],
        tt["remainder_zeroth"],
        "o-",
        color=COLOR["square"],
        label=f"0th Order (Slope {tt['slope_zeroth']:.2f})",
    )
    ax.plot(
        tt["h"],
        tt["remainder_first"],
        "o-",
        color=COLOR["R_outboard"],
        label=f"1st Order (Slope {tt['slope_first']:.2f})",
    )
    h = np.array(tt["h"])
    ax.plot(h, h / h[0] * tt["remainder_zeroth"][0], ":", color=INK_MUTED, lw=0.75)
    ax.plot(h, (h / h[0]) ** 2 * tt["remainder_first"][0], ":", color=INK_MUTED, lw=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(axis_label("Perturbation Size", "h"))
    ax.set_ylabel(axis_label("Taylor Remainder"))
    ax.legend(frameon=False, fontsize=6)
    _panel_letter(ax, "a")
    # major decade ticks (1e-5..1e-2) already label 4 of the 7 sampled h; the
    # in-between points sit at exactly 3x each decade, so mark those too
    _value_ticks(ax, "x", [hv for hv in h if not np.isclose(np.log10(hv) % 1, 0)], fmt="{:.0e}")

    ax = axes[1]
    verify = results["verify"]
    for key, tag in (("R_out", "R_outboard"), ("R_in", "R_inboard")):
        points = sorted((v["V_min"], v) for k, v in verify.items() if k.startswith(key + "_"))
        v_min = [p[0] for p in points]
        d_beta = [abs(p[1]["meshes"][2]["beta_star"] / p[1]["meshes"][1]["beta_star"] - 1.0) for p in points]
        d_w = [abs(p[1]["meshes"][2]["value_star"] / p[1]["meshes"][1]["value_star"] - 1.0) for p in points]
        ax.plot(v_min, d_w, marker=MARKER[tag], color=COLOR[tag], label=f"{LABEL[tag]}, $W^*$")
        ax.plot(
            v_min,
            d_beta,
            marker=MARKER[tag],
            color=COLOR[tag],
            mfc="none",
            ls="--",
            label=f"{LABEL[tag]}, $\\beta^*$",
        )
    ax.axhline(0.01, color=INK_SECONDARY, lw=0.6)
    ax.annotate("1% ($W^*$)", (v_min[0], 0.0112), fontsize=6, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _value_ticks(ax, "x", v_min, fmt="{:g}")
    ax.set_xlabel(axis_label("Minimum Velocity Bound", "V_{\\min}", "m/s"))
    ax.set_ylabel(axis_label("Relative Change After Re-Optimizing"))
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    _panel_letter(ax, "b")

    rows_csv = [
        {"h": hv, "remainder_zeroth": z, "remainder_first": f}
        for hv, z, f in zip(tt["h"], tt["remainder_zeroth"], tt["remainder_first"])
    ]
    caption = (
        "Verification. (a) Taylor remainder test (Farrell et al. 2013 [O10]) "
        "at Case R outboard's default optimum: the zeroth-order remainder falls "
        "as $O(h)$ and the gradient-corrected remainder as $O(h^2)$ (dotted "
        "guides), confirming the exact gradient in the area. (b) The change of $W^*$ (solid) and $\\beta^*$ "
        "(dashed) when each Pareto point is re-optimized on the finer mesh (48/6 to 72/9 cells, a ratio of 1.5); "
        "the exit limits are 1 % and 3 %."
    )
    save_figure(fig, "F8_verification", rows_csv, caption)


# ---------------------------------------------------------------------- tables
def table_t1(results: dict) -> None:
    pbli = results["inputs"]["pbli"]
    demo = results["inputs"]["demo_geometry"]
    # demo["source"] carries a citation plus a one-time caveat about the geometry;
    # repeating the full string on 5 rows just bloats the table, so the citation
    # goes in each row and the caveat is stated once, in the caption, instead
    demo_citation, _, demo_caveat = demo["source"].partition("; ")
    rows = [
        ["PbLi Temperature", f"{pbli['temperature_K']:.0f}", "K", pbli["source"]],
        ["Conductivity $\\sigma$", f"{pbli['sigma']:.3e}", "S/m", pbli["source"]],
        ["Density $\\rho$", f"{pbli['rho']:.0f}", "kg/m$^3$", pbli["source"]],
        ["Dynamic Viscosity $\\mu$", f"{pbli['mu']:.3e}", "Pa s", pbli["source"]],
        ["Specific Heat $c_p$", f"{pbli['cp']:.1f}", "J/(kg K)", pbli["source"]],
        [
            "First-Wall Field $B_\\mathrm{fw}$",
            f"{results['inputs']['B_fw_T']:.2f}",
            "T",
            "Plan 4.4, Lab-Scale",
        ],
        ["$R_0$", f"{demo['R0_m']:.1f}", "m", demo_citation],
        ["Outboard First Wall", f"{demo['R_outboard_fw_m']:.1f}", "m", demo_citation],
        ["Inboard First Wall", f"{demo['R_inboard_fw_m']:.1f}", "m", demo_citation],
        ["Outboard Blanket Depth", f"{demo['blanket_depth_outboard_m']:.3f}", "m", demo_citation],
        ["Inboard Blanket Depth", f"{demo['blanket_depth_inboard_m']:.3f}", "m", demo_citation],
        ["Target Flow Rate $Q$", f"{results['inputs']['Q_m3s']:.2e}", "m$^3$/s", "Plan 4.4 (Worked Example)"],
        [
            "Default $V_\\mathrm{min}$",
            f"{results['inputs']['V_min_default_ms'] * 1000:.1f}",
            "mm/s",
            "Plan 2.4",
        ],
        ["$V_\\mathrm{max}$", f"{results['inputs']['V_max_default_ms']:.1f}", "m/s", "Plan 2.4"],
        [
            "Aspect Bounds $\\beta$",
            f"[{results['inputs']['beta_bounds'][0]}, {results['inputs']['beta_bounds'][1]}]",
            "$-$",
            "No Blanket-Space Box (User Decision, 2026-09-24)",
        ],
    ]
    save_table(
        "T1_inputs",
        ["Quantity", "Value", "Unit", "Source"],
        rows,
        "Inputs and properties, frozen before any result was examined (CONTRIBUTING campaign "
        f"discipline). $R_0$ through the two blanket depths: {demo_caveat}.",
    )


_BOUND_LABEL = {"V_min": "$V_\\mathrm{min}$", "V_max": "$V_\\mathrm{max}$", "interior": "interior"}


def table_t2(results: dict) -> None:
    rows = []
    for key, tag in [("case_r_outboard", "R Outboard"), ("case_r_inboard", "R Inboard"), ("case_p", "P")]:
        for row in results[key]["pareto"]:
            if abs(row["V_min"] - results["inputs"]["V_min_default_ms"]) < 1e-12:
                rows.append(
                    [
                        tag,
                        f"{row['a_m'] * 1000:.2f}",
                        f"{row['b_m'] * 1000:.2f}",
                        f"{row['beta']:.4f}",
                        f"{row['W']:.4e}",
                        _BOUND_LABEL.get(row["active_u_bound"], row["active_u_bound"]),
                        f"{row['u_multiplier']:.2e}",
                        f"{row['dW_dw_final']:.2e}",
                    ]
                )
    save_table(
        "T2_optima",
        [
            "Case",
            "$a^*$ (mm)",
            "$b^*$ (mm)",
            "$\\beta^*$",
            "$W^*$ (W)",
            "Active Bound",
            "Multiplier",
            "$dW/dw$",
        ],
        rows,
        f"Optima at the default $V_\\mathrm{{min}}={results['inputs']['V_min_default_ms'] * 1000:.0f}$ mm/s.",
    )


def table_t3(results: dict) -> None:
    rows = []
    for r in results["design_law"]["rows"] + results["design_law"].get("high_ha", []):
        gci = f"{100 * r['gci']['beta']['gci_fine']:.2f}" if r.get("gci") else "$-$"
        rows.append(
            [
                f"{r['H']:.0f}",
                f"{r['Ha_star']:.1f}",
                f"{r['beta_star']:.5f}",
                f"{r['s_star']:.4f}",
                f"{r['spectral']['beta_star']:.5f}",
                f"{r['spectral']['s_star']:.4f}",
                f"{100 * r['reduction']:.1f}",
                gci,
            ]
        )
    save_table(
        "T3_design_law",
        [
            "$H$",
            "$Ha^*$",
            "$\\beta^*$ (48/6)",
            "$s^*$ (48/6)",
            "$\\beta^*$ (ref.)",
            "$s^*$ (ref.)",
            "Reduction (\\%)",
            "$\\beta^*$ GCI (\\%)",
        ],
        rows,
        "The design-law data behind Fig. F2: fixed area, core on the scaled-mesh family and the spectral reference.",
    )


_STATION_LABEL = {
    **{f"station {i}": f"Station {i}" for i in range(6)},
    "uniform (cross-duct 1/R neglected)": "Uniform (Cross-Duct 1/R Neglected)",
}


def table_t4(results: dict) -> None:
    rows = []
    for key, tag in [("case_r_outboard", "R Outboard"), ("case_r_inboard", "R Inboard"), ("case_p", "P")]:
        for row in results[key]["pareto"]:
            for s in row["validity"]["stations"]:
                rows.append(
                    [
                        tag,
                        f"{row['V_min'] * 1000:.1f}",
                        _STATION_LABEL.get(s["label"], s["label"]),
                        f"{s['Ha']:.1f}",
                        f"{s['Re']:.2f}",
                        f"{s['Re_over_Ha']:.3f}",
                        f"{s['Re_over_sqrtHa']:.2f}",
                        f"{s['N_interaction']:.1f}",
                        f"{s['gamma_sqrt_Ha']:.3f}",
                        "Pass" if s["gamma_sqrt_Ha_leq_0.2"] and s["Re_over_Ha_leq_200"] else "LIMIT",
                    ]
                )
    save_table(
        "T4_validity",
        [
            "Case",
            "$V_\\mathrm{min}$ (mm/s)",
            "Station",
            "$Ha$",
            "$Re$",
            "$Re/Ha$",
            "$Re/\\sqrt{Ha}$",
            "$N$",
            "$\\gamma\\sqrt{Ha}$",
            "Status",
        ],
        rows,
        "Validity of the laminar, inertialess, isothermal model at every optimum and station (plan Section 2.5). "
        "Status: $Re/Ha\\le200$ and $\\gamma\\sqrt{Ha}\\le0.2$; $Ha$ is not capped.",
    )


def table_t5(results: dict) -> None:
    rows = []
    for key, tag in [("case_r_outboard", "R Outboard"), ("case_r_inboard", "R Inboard"), ("case_p", "P")]:
        for row in results[key]["pareto"]:
            tt = row["taylor_test"]
            rows.append(
                [
                    tag,
                    f"{row['V_min'] * 1000:.1f}",
                    f"{tt['slope_zeroth']:.2f}",
                    f"{tt['slope_first']:.2f}",
                    f"{row['finer_mesh']['W_relative_change']:.2e}",
                ]
            )
    save_table(
        "T5_verification",
        [
            "Case",
            "$V_\\mathrm{min}$ (mm/s)",
            "Taylor Slope (0th)",
            "Taylor Slope (1st)",
            "Finer-Mesh $|\\Delta W|/W$",
        ],
        rows,
        "Verification summary at every Pareto point: Taylor-test slopes (expect 1, 2) and the finer-mesh check.",
    )


def table_t10(results: dict) -> None:
    rows = []
    for H, d in _tilt_law_arrays(results).items():
        for i in range(len(d["tilt"])):
            rows.append(
                [
                    f"{d['Ha_star_0']:.0f}",
                    f"{d['tilt'][i]:g}",
                    f"{d['kappa'][i]:.2f}",
                    f"{d['beta0'] * d['G'][i]:.5f}",
                    f"{d['G'][i]:.3f}",
                    f"{d['P'][i]:.3f}",
                    f"{100 * d['reduction'][i]:.1f}",
                    f"{d['aligned_over_square'][i]:.3f}",
                ]
            )
    save_table(
        "T10_tilt_law",
        [
            "$Ha^*_0$",
            "$B_p/B_T$",
            "$\\kappa$",
            "$\\beta^*$",
            "$\\beta^*/\\beta^*_0$",
            "$\\Delta p^*/\\Delta p^*_0$",
            "Reduction vs. Square (\\%)",
            "Aligned Shape / Square",
        ],
        rows,
        "The tilt-aware design law on the finest mesh (72/9 cells): the optimum in a tilted field, its penalty, its "
        "reduction over the equal-area square in the same field, and the aligned optimum's shape against the square.",
    )


def table_t8(results: dict) -> None:
    rows = []
    for key, r in sorted(results["verify"].items(), key=lambda kv: (kv[0].split("_")[0], kv[1]["V_min"])):
        m, s, g = r["meshes"], r["spectral"], r["gci"]
        rows.append(
            [
                key.rsplit("_", 1)[0].replace("_", " "),
                f"{1000 * r['V_min']:g}",
                f"{m[0]['beta_star']:.5f}",
                f"{m[1]['beta_star']:.5f}",
                f"{m[2]['beta_star']:.5f}",
                f"{s['beta_star']:.5f}",
                f"{g['beta']['order']:.2f}",
                f"{100 * g['beta']['gci_fine']:.2f}",
                f"{100 * (m[2]['value_star'] / m[1]['value_star'] - 1):+.2f}",
                f"{100 * s['W_fv_over_spectral']:+.2f}",
                f"{m[1]['dW_dw_rel_h']:+.1e}",
                f"{m[1]['dW_dw_rel_richardson']:+.1e}",
            ]
        )
    save_table(
        "T8_reoptimized",
        [
            "Case",
            "$V_\\mathrm{min}$ (mm/s)",
            "$\\beta^*$ 32/4",
            "$\\beta^*$ 48/6",
            "$\\beta^*$ 72/9",
            "$\\beta^*$ ref.",
            "Order",
            "GCI (\\%)",
            "$W^*$ change (\\%)",
            "Core vs. ref. $W$ (\\%)",
            "$dW/dw$ at $h$",
            "$dW/dw$ Rich.",
        ],
        rows,
        "Every Pareto optimum re-optimized on three meshes of constant refinement ratio 1.5 (the O12 scaled-mesh family), "
        "the observed order and GCI of $\\beta^*$, the change of $W^*$ from 48/6 to 72/9, the core's $W$ against the "
        "spectral reference at the core's own $\\beta^*$, and the stationarity $|dW/dw|/W$ at the registered step "
        "$h=0.01$ and Richardson-extrapolated.",
    )


def table_t9(results: dict) -> None:
    rows = [
        [
            r["case"].replace("_", " "),
            r["design"],
            f"{r['beta']:.4f}",
            f"{r['Ha']:.1f}",
            f"{r['B_p_over_B_T']:g}",
            f"{r['W_over_aligned']:.4f}",
        ]
        for r in results["tilt"]
    ]
    save_table(
        "T9_tilt",
        ["Case", "Design", "$\\beta$", "$Ha$", "$B_p/B_T$", "$W/W_\\mathrm{aligned}$"],
        rows,
        "Pumping power of the default optimum and the equal-area square in a field tilted in the cross-section "
        "(exit (j)); one mid-run station, 48/6 cells (tilt-aware mesh; from 32/4 to 96/12 the optimum's ratios move by 0.02 % and 0.0002 %, the square's by 0.02 % and 0.06 %).",
    )


def table_t7(results: dict) -> None:
    ramp = results["ramp_excess"]["rows"]
    rows = []
    for r in (r for r in ramp if r["variant"] == "base"):
        change = {
            v["variant"]: abs(v["excess_percent"] - r["excess_percent"]) / abs(r["excess_percent"])
            for v in ramp
            if v["variant"] != "base"
            and (v["case"], v["ha_mid"], v["gamma_sqrt_ha"]) == (r["case"], r["ha_mid"], r["gamma_sqrt_ha"])
        }
        rows.append(
            [
                "Square" if r["case"] == "square" else "Design Law",
                f"{r['ha_mid']:.0f}",
                f"{r['beta']:.3f}",
                f"{r['gamma_sqrt_ha']:g}",
                f"{r['x0']:.3g}",
                f"{r['excess_percent']:.4f}",
                f"{100 * change['cross_72_9']:.1f}" if change else "-",
                f"{100 * change['half_spacing']:.1f}" if change else "-",
                f"{r['iterations']}",
                f"{r['elapsed_s']:.0f}",
            ]
        )
    save_table(
        "T7_ramp",
        [
            "Duct",
            "$Ha$",
            "$\\beta$",
            "$\\gamma\\sqrt{Ha}$",
            "$x_0/a$",
            "Excess (\\%)",
            "72/9 Change (\\%)",
            "Half-Spacing Change (\\%)",
            "CG Iterations",
            "Solve (s)",
        ],
        rows,
        "Open-duct ramp study (5A.B): the 3-D excess over the locally fully developed sum, the change of that "
        "excess under the two refinements (relative to itself), and the cost of each base solve on a CPU.",
    )


def table_t6(results: dict) -> None:
    meta = results["meta"]
    rows = [
        [
            "Landscape (3 Cases)",
            f"{sum(results[k]['landscape_time_s'] for k in ('case_r_outboard', 'case_r_inboard', 'case_p')):.0f}",
        ],
        ["Demo Optimizer", f"{results['case_r_outboard']['demo_optimizer']['time_s']:.0f}"],
        [
            "Pareto Sweeps (3 Cases)",
            f"{sum(sum(r['time_s'] for r in results[k]['pareto']) for k in ('case_r_outboard', 'case_r_inboard', 'case_p')):.0f}",
        ],
        ["Design-Law Sweep", f"{results['design_law']['time_s']:.0f}"],
        ["Validity Map", f"{results['validity_map']['time_s']:.0f}"],
        ["Total", f"{meta['total_time_s']:.0f}"],
    ]
    save_table(
        "T6_cost",
        ["Stage", "Wall Time (s)"],
        rows,
        f"Cost, {meta['host']}, JAX {meta['jax_version']}, SOLVAX {meta['solvax_version']}, "
        f"git {meta['git_sha'][:8]}. Over the stages: "
        f"{sum(c['meshes_built'] for c in meta['stage_cost'].values())} meshes built and "
        f"{sum(c['compiles'] for c in meta['stage_cost'].values())} XLA compiles "
        "(measured per stage process, R7).",
    )


def main():
    results = load_results()
    print("figures:")
    figure_f1(results)
    figure_f2(results)
    figure_f3(results)
    figure_f4(results)
    figure_f5(results)
    figure_f6(results)
    figure_f7(results)
    figure_f8(results)
    figure_f9(results)
    print("tables:")
    table_t1(results)
    table_t2(results)
    table_t3(results)
    table_t4(results)
    table_t5(results)
    table_t6(results)
    table_t7(results)
    table_t8(results)
    table_t9(results)
    table_t10(results)
    with (DATA_DIR / "captions.md").open("w") as fh:
        fh.write("# Draft figure and table captions\n\n" + "\n\n".join(CAPTIONS) + "\n")
    print(f"wrote {DATA_DIR / 'captions.md'}")


if __name__ == "__main__":
    main()
