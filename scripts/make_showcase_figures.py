"""Regenerate the README gallery assets from the public LMhdX API.

Run ``python scripts/make_showcase_figures.py``. Outputs go to ``docs/_static``:
a 256^2 quasi-2D turbulence animation and poster, a Hunt-duct Hartmann-number
sweep, the validation ladder of the staggered core against the spectral
reference, the pipe, and the heat-transfer set: three benchmarks against the
literature (developed Nusselt numbers, the thermal entrance, magnetoconvection
onset) and two 3-D movies (convection with and without a field, a heated duct).
The duct, pipe and Q2D figures take about three minutes on a laptop CPU, the
thermal benchmarks about eight; the convection movie takes three minutes on a
laptop GPU and about a quarter of an hour on its CPU. The small thermal solves
are pinned to the CPU, where they run faster than on a GPU. Sizes are kept under
the documented Git media budget; production-resolution movies are release assets.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

import lmhdx  # noqa: E402
from lmhdx.axial import open_duct  # noqa: E402
from lmhdx.core3d import ChannelProblem, duct_problem, zero_velocity  # noqa: E402
from lmhdx.grid import (  # noqa: E402
    CENTER,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.io import extract_midplane_profile  # noqa: E402
from lmhdx.steady import solve_steady_state  # noqa: E402
from lmhdx.thermal import (  # noqa: E402
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    developed_heat,
    initial_thermal_state,
    thermal_boundary,
    validate_thermal_inputs,
)

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"


def q2d_turbulence(n: int = 256, steps: int = 3000, stride: int = 50, seed: int = 3) -> None:
    """Decaying Q2D turbulence: band-limited random vorticity, weak friction."""
    length = 2.0 * np.pi
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.fftfreq(n, d=1.0 / n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k0 = 12.0
    amplitude = np.where(k > 0, (k / k0) ** 4 * np.exp(-2.0 * (k / k0) ** 2), 0.0)
    phase = np.exp(2j * np.pi * np.asarray(jax.random.uniform(jax.random.PRNGKey(seed), (n, n))))
    vorticity = np.fft.ifftn(amplitude * phase).real
    vorticity *= 4.0 / np.sqrt(np.mean(vorticity**2))
    problem = lmhdx.Q2DProblem(
        jnp.asarray(vorticity, dtype=jnp.float32),
        length=(length, length),
        viscosity=2.0e-4,
        hartmann_friction=2.0e-2,
        dt=2.0e-3,
        steps=steps,
        history_stride=stride,
        energy_budget_tolerance=5.0e-2,
    )
    result = lmhdx.solve(problem)
    frames = np.asarray(result.vorticity_history)
    limit = float(np.percentile(np.abs(frames), 99.5))
    print(
        f"q2d: {result.status}, {frames.shape[0]} frames, energy defect "
        f"{float(result.diagnostics.energy_budget_residual):.2e}"
    )

    cmap = plt.get_cmap("RdBu_r")
    images = []
    for frame in frames[::2]:
        rgba = cmap(0.5 + 0.5 * np.clip(frame.T[::-1] / limit, -1.0, 1.0))
        images.append(
            Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8)).resize((320, 320), Image.LANCZOS)
        )
    images[0].save(
        STATIC / "q2d_turbulence_256.webp",
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        quality=52,
        method=6,
    )

    def spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k2 = kx**2 + ky**2
        psi_hat = np.divide(np.fft.fftn(field), k2, out=np.zeros((n, n), complex), where=k2 > 0)
        energy_k = 0.5 * np.abs(psi_hat) ** 2 * k2 / n**4
        bins = np.arange(1, n // 2)
        return bins, np.array([energy_k[(k >= b - 0.5) & (k < b + 0.5)].sum() for b in bins])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    snapshots = (0, len(frames) // 3, len(frames) - 1)
    for axis, index in zip(axes[:3], snapshots, strict=True):
        axis.imshow(
            frames[index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0, length, 0, length),
        )
        axis.set_title(f"t = {index * stride * problem.dt:.1f}")
        axis.set_xticks([])
        axis.set_yticks([])
    for index, style in zip(snapshots, (":", "--", "-"), strict=True):
        bins, energy = spectrum(frames[index])
        axes[3].loglog(bins, energy, style, color="k", label=f"t = {index * stride * problem.dt:.1f}")
    guide = np.arange(8, 60)
    axes[3].loglog(guide, 3.0e-1 * guide**-3.0, color="tab:red", lw=1, label="$k^{-3}$")
    axes[3].set_xlabel("k")
    axes[3].set_ylabel("E(k)")
    axes[3].set_title("Energy spectrum")
    axes[3].legend(frameon=False)
    _save_webp(fig, STATIC / "q2d_turbulence_poster.webp")


def hunt_sweep(hartmann_numbers: tuple[float, ...] = (20.0, 100.0, 500.0, 1000.0), cells: int = 72) -> None:
    """Hunt duct profiles versus Ha: side-layer jets and their Ha^-1/2 thickness."""
    profiles, maps, peaks = {}, {}, []
    for ha in hartmann_numbers:
        solution = lmhdx.solve(lmhdx.make_hunt_case(ha=ha, ny=cells, nz=cells))
        if not solution.converged:
            raise RuntimeError(f"Hunt Ha={ha:g} ended with {solution.status}")
        profile = extract_midplane_profile(solution, axis="z")
        z, u = np.asarray(profile["z"]), np.asarray(profile["u"])
        profiles[ha] = (z, u / (np.trapezoid(u, z) / (z[-1] - z[0])))
        maps[ha] = (
            np.asarray(solution.mesh.y_centers),
            np.asarray(solution.mesh.z_centers),
            np.asarray(solution.state.u),
        )
        half = len(u) // 2
        peaks.append((ha, 1.0 + z[np.argmax(u[:half])]))
        print(f"hunt Ha={ha:g}: {solution.steps} steps, jet at wall distance {peaks[-1][1]:.4f}")

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9])
    ax_profile, ax_map, ax_scaling = (fig.add_subplot(grid[i]) for i in range(3))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(hartmann_numbers)))
    for (ha, (z, u)), color in zip(profiles.items(), colors, strict=True):
        ax_profile.plot(z, u, color=color, lw=1.8, label=f"Ha = {ha:g}")
    ax_profile.set_xlabel("z / a")
    ax_profile.set_ylabel("u / ⟨u⟩")
    ax_profile.set_title("Hunt duct midplane profiles: side-layer jets grow with Ha")
    ax_profile.set_xlim(-1, 1)
    ax_profile.legend(frameon=False)
    ha_max = hartmann_numbers[-1]
    y, z, u = maps[ha_max]
    image = ax_map.pcolormesh(z, y, u / np.max(u), cmap="magma", shading="auto", vmin=0, vmax=1)
    ax_map.set_title(f"u / max u at Ha = {ha_max:g}, c = 0.05, {cells}² fluid cells")
    ax_map.set_xlabel("z / a")
    ax_map.set_ylabel("y / a")
    ax_map.set_aspect("equal")
    fig.colorbar(image, ax=ax_map, shrink=0.85)
    ha_values, distances = (np.array(v) for v in zip(*peaks, strict=True))
    ax_scaling.loglog(ha_values, distances, "o", color="k", label="jet maximum, LMhdX")
    guide = np.array([15.0, 1500.0])
    ax_scaling.loglog(
        guide,
        distances[1] * (guide / ha_values[1]) ** -0.5,
        "--",
        color="tab:red",
        label=r"$\propto Ha^{-1/2}$",
    )
    ax_scaling.set_xlabel("Ha")
    ax_scaling.set_ylabel("distance of jet maximum from side wall / a")
    ax_scaling.set_title("Side-layer thickness")
    ax_scaling.legend(frameon=False)
    _save_webp(fig, STATIC / "hunt_side_layers.webp")


def validation_ladder(
    hartmann_numbers: tuple[float, ...] = (20.0, 100.0, 300.0),
    conductances: tuple[float, ...] = (0.0, 0.027, 0.1),
    cells: int = 48,
) -> None:
    """Rows 2-4 of the validation ladder, solved by the staggered core.

    Every curve here is the steady solve of `lmhdx.steady` on a wall-resolving
    mesh (conjugate gradients for insulating walls, Newton-Krylov for the
    conducting ones), and every reference is `validation.shercliff`, which
    shares no operator, mesh or solver with the package. The point of the middle
    panel is that the error does not grow with the field: the Hartmann layer is
    resolved rather than tolerated.
    """
    from validation.shercliff import duct_flow

    # The gates here are parts in a thousand; float32 cannot express them.
    lmhdx.enable_x64()
    profiles, errors, refinement = {}, {}, {}
    for hartmann in hartmann_numbers:
        for conductance in conductances:
            problem = _duct_problem(cells, hartmann, conductance)
            rate, velocity = _steady_duct(problem)
            exact = _reference_rate(hartmann, conductance)
            errors[(hartmann, conductance)] = abs(rate - exact) / exact
            if conductance == 0.0:
                profiles[hartmann] = (
                    np.asarray(problem.grid.centers[1]),
                    np.asarray(velocity)[0][:, cells // 2],
                )
            print(f"ladder Ha={hartmann:g} c={conductance:g}: {rate:.6g} against {exact:.6g}")
    for count in (24, 32, 40, 48):
        problem = _duct_problem(count, 20.0, 0.0)
        rate, _ = _steady_duct(problem)
        refinement[count] = abs(rate - _reference_rate(20.0, 0.0)) / _reference_rate(20.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(hartmann_numbers)))
    for (hartmann, (y, u)), color in zip(profiles.items(), colors, strict=True):
        half = y < 0.0
        axes[0].semilogx(
            (1.0 + y[half]) * hartmann, u[half] / np.max(u), color=color, lw=1.8, label=f"Ha = {hartmann:g}"
        )
        nodes, reference, _ = duct_flow(hartmann, 48)
        middle = reference[:, reference.shape[1] // 2]
        inside = nodes < 0.0
        axes[0].semilogx(
            (1.0 + nodes[inside]) * hartmann,
            middle[inside] / np.max(middle),
            "o",
            ms=2.5,
            color=color,
            alpha=0.5,
        )
    wall = np.logspace(-1.5, 1.2, 60)
    axes[0].semilogx(wall, 1.0 - np.exp(-wall), "--", color="tab:red", lw=1.2, label=r"$1-e^{-\xi}$")
    axes[0].set_xlabel(r"$\xi = (1 + y/a)\,Ha$   (wall distance in layer widths)")
    axes[0].set_ylabel("u / max u   (lines LMhdX, points spectral)")
    axes[0].set_title("Hartmann layers collapse", fontsize=11)
    axes[0].set_xlim(3.0e-2, 20.0)
    axes[0].legend(frameon=False, loc="lower right")

    markers = {0.0: "o", 0.027: "s", 0.1: "^"}
    for conductance in conductances:
        values = [errors[(hartmann, conductance)] for hartmann in hartmann_numbers]
        axes[1].loglog(
            hartmann_numbers,
            values,
            markers[conductance] + "-",
            lw=1.4,
            label=f"c = {conductance:g}" if conductance else "insulating",
        )
    axes[1].axhline(0.02, color="tab:red", ls="--", lw=1, label="2 % gate")
    axes[1].set_xlabel("Ha")
    axes[1].set_ylabel("relative error in Q / A")
    axes[1].set_title(f"Flow rate vs the spectral reference, {cells}² cells", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8)

    counts = np.array(sorted(refinement))
    values = np.array([refinement[count] for count in counts])
    axes[2].loglog(counts, values, "o-", color="k", label="LMhdX, Ha = 20")
    axes[2].loglog(
        counts, values[0] * (counts / counts[0]) ** -2.0, "--", color="tab:red", label="second order"
    )
    axes[2].set_xlabel("cells per transverse direction")
    axes[2].set_ylabel("relative error in Q / A")
    axes[2].set_title("Mesh convergence, insulating Ha = 20", fontsize=11)
    axes[2].legend(frameon=False)
    _save_webp(fig, STATIC / "validation_ladder.webp")


def _duct_problem(cells: int, hartmann: float, conductance: float):
    from lmhdx.core3d import duct_problem

    return duct_problem(hartmann=hartmann, cells=cells, wall_conductance=conductance)


def _steady_duct(problem) -> tuple[float, np.ndarray]:
    from lmhdx.steady import solve_steady_state

    velocity = solve_steady_state(problem, pseudo_step=1.0e3, linear_restart=600).velocity[0].data
    volumes = np.asarray(problem.grid.cell_volumes())[0]
    rate = float((np.asarray(velocity)[0] * volumes).sum() / volumes.sum())
    return rate, np.asarray(velocity)


def _reference_rate(hartmann: float, conductance: float) -> float:
    from validation.shercliff import flow_rate

    return flow_rate(hartmann, 48, hartmann_wall=conductance)


def pipe_sweep(hartmann_numbers: tuple[float, ...] = (0.0, 20.0, 100.0, 400.0)) -> None:
    """A circular pipe across the Hartmann range: core, layers and flow rate."""
    from lmhdx.pipe import flow_rate, pipe_problem, solve_pipe

    lmhdx.enable_x64()
    profiles, rates, maps = {}, {}, None
    for hartmann in hartmann_numbers:
        problem = pipe_problem(hartmann=hartmann, radial=64, azimuthal=128)
        velocity, _ = solve_pipe(problem)
        radius = np.asarray(problem.grid.centers[0])
        angle = np.asarray(problem.grid.centers[1])
        values = np.asarray(velocity.data)[:, :, 0]
        rates[hartmann] = flow_rate(velocity)
        along = np.argmin(np.abs(angle - 0.0))
        across = np.argmin(np.abs(angle - 0.5 * np.pi))
        scale = np.max(values)
        profiles[hartmann] = (radius, values[:, along] / scale, values[:, across] / scale)
        if hartmann == hartmann_numbers[-1]:
            maps = (
                np.asarray(problem.grid.faces[0]),
                np.asarray(problem.grid.faces[1]),
                values / scale,
            )
        print(f"pipe Ha={hartmann:g}: Q/A = {rates[hartmann]:.6g}")

    fig = plt.figure(figsize=(13.5, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.0])
    ax_profile, ax_map, ax_rate = (fig.add_subplot(grid[index]) for index in range(3))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(hartmann_numbers)))
    for (hartmann, (radius, along, across)), color in zip(profiles.items(), colors, strict=True):
        ax_profile.plot(radius, along, color=color, lw=1.8, label=f"Ha = {hartmann:g}")
        ax_profile.plot(radius, across, color=color, lw=1.2, ls="--")
    ax_profile.set_xlabel("r / a")
    ax_profile.set_ylabel("u / max u")
    ax_profile.set_title("Radial profiles: solid along B, dashed across it", fontsize=11)
    ax_profile.set_xlim(0.0, 1.0)
    ax_profile.legend(frameon=False, fontsize=9)

    radial_faces, azimuthal_faces, values = maps
    mesh_angle, mesh_radius = np.meshgrid(azimuthal_faces, radial_faces)
    image = ax_map.pcolormesh(
        mesh_radius * np.cos(mesh_angle),
        mesh_radius * np.sin(mesh_angle),
        values,
        cmap="magma",
        shading="flat",
    )
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("distance along B / a")
    ax_map.set_ylabel("distance across B / a")
    ax_map.set_title(f"u / max u at Ha = {hartmann_numbers[-1]:g}", fontsize=11)
    fig.colorbar(image, ax=ax_map, shrink=0.85)

    values = np.array([rates[hartmann] for hartmann in hartmann_numbers])
    finite = np.array(hartmann_numbers) > 0.0
    ax_rate.loglog(np.array(hartmann_numbers)[finite], values[finite], "o-", color="k", label="LMhdX")
    guide = np.array([10.0, 600.0])
    reference = np.array(hartmann_numbers)[finite][1]
    ax_rate.loglog(
        guide,
        values[finite][1] * (guide / reference) ** -1.0,
        "--",
        color="tab:red",
        label=r"$\propto Ha^{-1}$",
    )
    ax_rate.axhline(0.125, color="tab:blue", ls=":", lw=1.2, label="Poiseuille, Ha = 0")
    ax_rate.set_xlabel("Ha")
    ax_rate.set_ylabel("Q / A")
    ax_rate.set_title("Flow rate at fixed pressure gradient", fontsize=11)
    ax_rate.legend(frameon=False, fontsize=9)
    _save_webp(fig, STATIC / "pipe_flow.webp")


# Heat transfer (lmhdx.thermal): three benchmarks against exact solutions and the literature, two 3-D movies.
# Journal style: STIX type, ticks inside boxed axes, panel letters. The series colors pass a check for
# color-vision deficiency on adjacent pairs; exact and published references are drawn in ink.
SERIES, INK, MUTED = ("#2a78d6", "#eb6834", "#1baf7a"), "#1a1a1a", "#6b6a66"
OPEN = {"mfc": "white", "mew": 1.6}  # open markers for LMhdX over a reference line
SOLID = {"projection": "3d", "computed_zorder": False}  # 3-D axes painted back to front
ROW = {"right": 0.985, "top": 0.9, "bottom": 0.14}  # a row of three panels
PAPER = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
    **{f"{a}tick.direction": "in" for a in "xy"},
    **{f"{a}tick.minor.visible": True for a in "xy"},
    "xtick.top": True,
    "ytick.right": True,
}
_WRAP = BoundaryCondition(PERIODIC)


def thermal_developed() -> None:
    """Developed Nusselt numbers against exact series, Shah & London's fit and the Hartmann closed form."""
    from validation.thermal import h1_series, plate_nusselt, shah_london_h1

    lmhdx.enable_x64()
    aspects, counts = np.array([1.0, 0.75, 0.5, 1 / 3, 0.25, 1 / 6, 0.125]), np.array([8, 16, 32, 64])
    duct_fields = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    plate_fields = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0])

    def nusselt(flow, slug=False):
        return float(_h1(flow, slug)[0].nusselt)

    with jax.default_device(jax.devices("cpu")[0]):  # small sections: the CPU beats a GPU's launch latency
        rectangles = np.array([nusselt(_rectangle(32, aspect)) for aspect in aspects])
        squares = {hartmann: _h1(duct_problem(hartmann=hartmann, cells=48)) for hartmann in duct_fields}
        plates = np.array([_heated_plates(128, hartmann) for hartmann in plate_fields])
        errors = {
            "Square H1, Poiseuille": [nusselt(_rectangle(n)) / h1_series(1, 1) - 1 for n in counts],
            "Square H1, slug": [nusselt(_rectangle(n), True) / h1_series(1, 1, True) - 1 for n in counts],
            "Plates, Ha = 10": [_heated_plates(n, 10.0, False) / plate_nusselt(10.0) - 1 for n in counts],
        }
    square = np.array([float(result.nusselt) for result, _ in squares.values()])
    worst = np.max(np.abs(rectangles / [h1_series(1.0, aspect) for aspect in aspects] - 1.0))
    print(f"developed: rectangles within {worst:.2%} of the series; square H1 Nu {np.round(square, 3)}")
    relative = np.round(plates / plate_nusselt(plate_fields) - 1, 4)
    print(f"developed: plates against the closed form {relative}")

    with plt.rc_context(PAPER):
        fig = plt.figure(figsize=(13.2, 8.4))
        outer = fig.add_gridspec(2, 1, hspace=0.3, left=0.05, right=0.985, top=0.95, bottom=0.075)
        top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.18)
        bottom = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
        views = {0.0: "H1 square duct, no field", 100.0: "The same duct at Ha = 100"}
        labels = (r"$y/a$  (along $\mathbf{B}$)", "$z/a$", r"$u/\bar{u}$")
        for index, (hartmann, title) in enumerate(views.items()):
            result, velocity = squares[hartmann]
            theta, u = np.asarray(result.temperature.data)[0], np.asarray(velocity[0].data)[0]
            axis = _axes_3d(fig, top[index], labels, (26, -60))
            mesh = np.meshgrid(*result.temperature.grid.centers[1:], indexing="ij")
            _paint(axis, *mesh, u / np.mean(u), plt.get_cmap("inferno")(theta / np.min(theta)))
            axis.set(zlim=(0.0, 2.2), zticks=[0, 1, 2])
            axis.set_title(f"{title}:  $Nu_{{H1}}$ = {float(result.nusselt):.2f}", pad=0)
            _panel(axis, "ab"[index])
        label = r"Height: axial velocity $u/\bar{u}$.   Color: temperature $(T_w - T)/(T_w - T_{center})$."
        _colorbar(fig, (0.14, 0.535, 0.36, 0.012), "inferno", Normalize(0.0, 1.0), label)

        axis = fig.add_subplot(top[2])
        alpha = np.linspace(0.02, 1.0, 200)
        exact, fit = [h1_series(1.0, a) for a in alpha], shah_london_h1(alpha)
        axis.plot(alpha, exact, color=INK, lw=1.4, label="Exact double-sine series")
        axis.plot(alpha, fit, "--", color=SERIES[1], lw=1.4, label="Shah & London (1978) fit")
        axis.plot(aspects, rectangles, "o", color=SERIES[0], ms=6.5, label="LMhdX", **OPEN)
        axis.axhline(140.0 / 17.0, color=MUTED, lw=0.8, ls=":")
        axis.text(0.5, 140 / 17 + 0.08, "Parallel plates, 140/17", ha="center", color=MUTED, size=9)
        axis.set(xlim=(0, 1.02), ylim=(3.2, 8.75), xlabel="Aspect ratio $b/a$", ylabel="$Nu_{H1}$")
        axis.set_title(f"Rectangular ducts, 32 cells across $b$: within {100 * worst:.2f} % of exact")
        axis.legend(loc="center right", bbox_to_anchor=(1.0, 0.6))
        _panel(axis, "c")

        axis = fig.add_subplot(bottom[0])
        line = np.logspace(-1, 3.3, 300)
        axis.semilogx(line, plate_nusselt(line), color=INK, lw=1.4, label="Hartmann plates, closed form")
        axis.semilogx(plate_fields, plates, "s", color=SERIES[1], label="Hartmann plates, LMhdX", **OPEN)
        axis.semilogx(duct_fields[1:], square[1:], "o-", color=SERIES[0], label="Square duct, LMhdX", **OPEN)
        for value, text in (
            (140 / 17, "Plates, Ha → 0: 140/17"),
            (12.0, "Plates, slug flow: 12"),
            (h1_series(1.0, 1.0), "Square, Ha → 0: 3.608"),
            (h1_series(1.0, 1.0, True), "Square, slug flow: 7.114"),
        ):
            offset, align = (0.1, "bottom") if "slug" in text else (-0.1, "top")
            axis.axhline(value, color=MUTED, lw=0.8, ls=":")
            axis.text(0.115, value + offset, text, color=MUTED, size=8.5, va=align)
        axis.set(xlim=(0.1, 2000), ylim=(2.8, 13.2), xlabel="Hartmann number Ha")
        axis.set_ylabel(r"Nu   (uniform heat flux along the duct, $D_h$-based)")
        axis.set_title("A transverse field flattens the core and raises Nu toward slug flow")
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
        _panel(axis, "d")

        axis = fig.add_subplot(bottom[1])
        for (label, values), marker, color in zip(errors.items(), "o^s", (SERIES[0], SERIES[2], SERIES[1])):
            axis.loglog(counts, np.abs(values), marker + "-", color=color, label=label, **OPEN)
        axis.loglog([7.0, 80.0], 0.3 * (np.array([7.0, 80.0]) / 10.0) ** -2.0, "--", color=INK, lw=1.0)
        axis.text(25, 0.3 * 2.5**-2 * 1.3, "Second order", rotation=-31, size=9, color=INK)
        axis.set_xticks(counts, [str(count) for count in counts])
        axis.set_xticks([], minor=True)
        axis.set(xlabel="Cells across the duct", ylabel="Relative error in Nu")
        axis.set_title("Mesh convergence against the exact values")
        axis.legend(loc="lower left")
        _panel(axis, "e")
        _save_webp(fig, STATIC / "thermal_developed_duct.webp", dpi=110, palette=True)


def thermal_entrance(hartmann_numbers: tuple[float, ...] = (0.0, 10.0, 50.0)) -> None:
    """The thermal entrance of heated Hartmann plates, an open duct against the Graetz eigen-expansion."""
    from validation.thermal import graetz

    lmhdx.enable_x64()
    with jax.default_device(jax.devices("cpu")[0]):
        cases = {hartmann: _entrance(hartmann) for hartmann in hartmann_numbers}
        long = _entrance(0.0, peclet=1600.0, extent=0.03)  # the expansion leaves out axial conduction
    errors = {ha: (xplus, nusselt / graetz(ha, xplus) - 1) for ha, (xplus, nusselt, *_) in cases.items()}
    for hartmann, (xplus, error) in errors.items():
        within = [f"{np.max(np.abs(error[xplus > start])):.2%} past x+ = {start:g}" for start in (1e-3, 1e-2)]
        print(f"entrance Ha {hartmann:g}: within {', '.join(within)}")

    with plt.rc_context(PAPER):
        fig = plt.figure(figsize=(13.2, 4.3))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.2, 0.9], wspace=0.28, left=0.02, **ROW)
        xplus, _, theta, y = cases[10.0]
        keep = (xplus >= 2e-4) & (xplus <= 0.06)
        labels = (r"$\log_{10} x^+$", "$y/a$", r"$k(T - T_b)/(q\,a)$")
        axis = _axes_3d(fig, grid[0], labels, (24, -122), ((-3, -2), (-1, 0, 1)))
        colors = plt.get_cmap("RdBu_r")(0.5 + 0.5 * theta[keep] / np.max(np.abs(theta[keep])))
        _paint(axis, *np.meshgrid(np.log10(xplus[keep]), y, indexing="ij"), theta[keep], colors)
        axis.set_title("Temperature above the bulk, Ha = 10", pad=0)
        _panel(axis, "a")

        axis = fig.add_subplot(grid[1])
        line = np.logspace(-4.3, -0.9, 300)
        for color, marker, (hartmann, (xplus, nusselt, *_)) in zip(SERIES, "os^", cases.items()):
            axis.semilogx(line, graetz(hartmann, line), color=color, lw=1.4)
            axis.semilogx(xplus[3::6], nusselt[3::6], marker, color=color, label=f"Ha = {hartmann:g}", **OPEN)
        axis.semilogx(line, graetz(np.inf, line), "--", color=INK, lw=1.1, label="Slug flow, Ha → ∞")
        axis.semilogx([], [], color=MUTED, lw=1.4, label="Eigen-expansion (lines)")
        axis.semilogx([], [], "o", color=MUTED, label="LMhdX open duct (symbols)", **OPEN)
        axis.set(xlim=(1.5e-4, 0.12), ylim=(6, 42), xlabel=r"$x^+ = x/(D_h\,Pe)$", ylabel="Local $Nu_x$")
        axis.set_title("Thermal entrance of heated Hartmann plates")
        axis.legend(loc="upper right")
        _panel(axis, "b")

        axis = fig.add_subplot(grid[2])
        for color, (hartmann, (xplus, error)) in zip(SERIES, errors.items()):
            axis.semilogx(xplus, 100 * error, color=color, label=f"Ha = {hartmann:g}")
        error = 100 * (long[1] / graetz(0.0, long[0]) - 1)
        axis.semilogx(long[0], error, "--", color=SERIES[0], label="Ha = 0, Pe = 1600")
        axis.axhline(0.0, color=MUTED, lw=0.8)
        axis.set(xlim=(2e-4, 0.12), ylim=(-1, 4), xlabel="$x^+$", ylabel="$Nu_x$ against the expansion (%)")
        axis.set_title("Pe = 400 unless marked: axial conduction")
        axis.legend(loc="upper right")
        _panel(axis, "c")
        _save_webp(fig, STATIC / "thermal_entrance.webp", dpi=110, palette=True)


def thermal_onset(chandrasekhar_numbers: tuple[float, ...] = (0.0, 10.0, 100.0, 300.0, 1000.0)) -> None:
    """Rayleigh-Benard onset in a vertical field against Chandrasekhar's eigenproblem."""
    from validation.thermal import CHANDRASEKHAR_TABLE, chandrasekhar_onset

    lmhdx.enable_x64()
    curve = np.array([(q, *chandrasekhar_onset(q)) for q in np.concatenate(([0.0], np.logspace(0, 5, 31)))])
    rows, modes = [], {}
    with jax.default_device(jax.devices("cpu")[0]):
        for q in chandrasekhar_numbers:
            exact, k = chandrasekhar_onset(q)
            coarse, onsets = 0.05 / (q + np.sqrt(exact)), []  # resolves 1/Q and the buoyancy time
            for dt in (coarse, coarse / 2.0):
                (below, _), (above, modes[q]) = (_onset_growth(f * exact, dt, q, k) for f in (0.97, 1.03))
                onsets.append(0.97 * exact - below * 0.06 * exact / (above - below))
            onsets.append(2.0 * onsets[1] - onsets[0])  # first order in dt
            rows.append((q, exact, k, *onsets))
            print(f"onset Q {q:g}: Ra_c {exact:.1f}; LMhdX at dt, dt/2, Richardson {np.round(onsets, 1)}")
    rows = np.array(rows)

    with plt.rc_context(PAPER):
        fig = plt.figure(figsize=(13.2, 4.4))
        outer = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 0.95], wspace=0.3, left=0.06, **ROW)
        axis = fig.add_subplot(outer[0])
        axis.plot(curve[:, 0], curve[:, 1], color=INK, lw=1.4, label="Chandrasekhar eigenproblem")
        strong = np.logspace(1.5, 5, 50)
        axis.plot(strong, np.pi**2 * strong, "--", color=MUTED, lw=1, label=r"$\pi^2 Q$, strong-field limit")
        table = np.transpose(CHANDRASEKHAR_TABLE)
        axis.plot(*table, "D", color=INK, mfc="none", ms=8, label="Chandrasekhar (1961) table")
        axis.plot(rows[:, 0], rows[:, 5], "o", color=SERIES[0], label=r"LMhdX, extrapolated in $\Delta t$")
        axis.set_xscale("symlog", linthresh=1.0, linscale=0.35)
        axis.set_yscale("log")
        axis.set(xlim=(0, 1.2e5), ylim=(1.2e3, 2e6), xlabel="Chandrasekhar number $Q = Ha^2$")
        axis.set_ylabel("Critical Rayleigh number $Ra_c$")
        worst = np.max(np.abs(rows[:, 5] / rows[:, 1] - 1.0))
        axis.set_title(f"A vertical field delays convection: LMhdX within {100 * worst:.1f} %")
        axis.legend(loc="upper left")
        _panel(axis, "a")

        inner = outer[1].subgridspec(2, 1, hspace=0.42)
        for row, q in enumerate((chandrasekhar_numbers[0], chandrasekhar_numbers[-1])):
            axis = fig.add_subplot(inner[row])
            _draw_roll(axis, modes[q], width=4.0)
            exact = rows[list(rows[:, 0]).index(q)]
            title = f"Critical roll at Q = {q:g}:  $a_c$ = {exact[2]:.2f},  $Ra_c$ = {exact[1]:,.0f}"
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("$x/d$" if row else "")
            if not row:
                axis.set_xticklabels([])
                axis.text(-0.13, 1.08, "(b)", transform=axis.transAxes, fontsize=12, fontweight="bold")

        axis = fig.add_subplot(outer[2])
        steps = (r"$\Delta t_0$", r"$\Delta t_0/2$", r"Richardson in $\Delta t$")
        for color, column, marker, label in zip(SERIES[::-1], (3, 4, 5), "s^o", steps):
            values = 100 * (rows[:, column] / rows[:, 1] - 1)
            axis.plot(rows[:, 0], values, marker + "-", color=color, label=label, **OPEN)
        axis.axhline(0.0, color=MUTED, lw=0.8)
        axis.set_xscale("symlog", linthresh=10.0, linscale=0.35)
        axis.set(xlim=(0, 1500), xlabel="$Q$", ylabel="$Ra_c$ against the eigenproblem (%)")
        axis.set_title(r"First order in the time step, $\Delta t_0 = 0.05/(Q + Ra_c^{1/2})$")
        axis.legend(loc="center right", bbox_to_anchor=(1.0, 0.6))
        _panel(axis, "c")
        _save_webp(fig, STATIC / "thermal_magnetoconvection.webp", dpi=110, palette=True)


def thermal_convection(rayleigh: float = 1.0e4, prandtl: float = 0.1, chandrasekhar: float = 1000.0) -> None:
    """A liquid-metal layer heated from below: cells without a field, rolls along a horizontal one.

    Three minutes on a laptop GPU (an RTX 3060, float64), about a quarter of an hour on its CPU.
    """
    lmhdx.enable_x64()
    # Free-fall units, with d, U_ff = sqrt(g beta Delta T d) and rho one; Q = sigma B^2 d^2 / (rho nu).
    field = np.sqrt(chandrasekhar * np.sqrt(prandtl / rayleigh))
    titles = ("No magnetic field", f"Horizontal field along $x$,  Q = {chandrasekhar:g}")
    runs = {title: _convection(strength, rayleigh, prandtl) for title, strength in zip(titles, (0.0, field))}
    cmap, norm = plt.get_cmap("RdBu_r"), Normalize(-0.5, 0.5)
    shape = runs[titles[0]][0].shape
    faces = [np.linspace(0.0, extent, count + 1) for extent, count in zip((4.0, 4.0, 1.0), shape[1:])]
    label = (
        rf"$(T - T_{{mean}})/\Delta T$ on the walls and the mid-plane.  "
        rf"Ra = $10^{{{np.log10(rayleigh):.0f}}}$, Pr = {prandtl:g}."
    )
    frames = []
    with plt.rc_context(PAPER):
        for index in range(3, shape[0], 2):  # every 5 d/U_ff, from when the instability has grown
            fig = plt.figure(figsize=(8.0, 4.1), dpi=160)
            for column, (title, (temperatures, nusselt)) in enumerate(runs.items()):
                axis = fig.add_axes((0.5 * column, 0.16, 0.5, 0.68), **SOLID)
                _cutaway(axis, temperatures[index], faces, cmap, norm, cut=shape[3] // 2)
                axis.set_box_aspect((4, 4, 1.3), zoom=1.05)
                axis.view_init(elev=30, azim=-58)
                axis.set_axis_off()
                middle = 0.25 + 0.5 * column
                fig.text(middle, 0.95, title, ha="center", va="top", fontsize=11.5)
                fig.text(middle, 0.875, f"Nu = {nusselt[index]:.2f}", ha="center", va="top", color=MUTED)
            axis.quiver(0.6, -0.8, 0.0, 2.4, 0.0, 0.0, color=INK, linewidth=1.5, arrow_length_ratio=0.15)
            axis.text(3.2, -0.95, 0.0, r"$\mathbf{B}$", color=INK, fontsize=12)
            fig.text(0.5, 0.95, f"t = {2.5 * (index + 1):.0f} $d/U_{{ff}}$", ha="center", va="top")
            _colorbar(fig, (0.32, 0.115, 0.36, 0.022), cmap, norm, label, (-0.5, 0, 0.5))
            frames.append(_rgb(fig))
    _animate(frames, "thermal_convection_3d", width=720, duration=160)


def thermal_heated_duct(hartmann: float = 1000.0, peclet: float = 2000.0) -> None:
    """A duct heated through one side wall from a cold inlet: a Hunt duct's side-layer jet keeps it cool."""
    lmhdx.enable_x64()
    cases = {0.0: _heated_duct(0.0, hartmann, peclet), 0.05: _heated_duct(0.05, hartmann, peclet)}
    peaks = {conductance: float(case["wall"][-1].max()) for conductance, case in cases.items()}
    jet, cooler = cases[0.05]["inlet"].max(), 1 - peaks[0.05] / peaks[0.0]
    finer = {c: float(_heated_duct(c, hartmann, peclet, cells=40)["wall"][-1].max()) for c in cases}
    print(f"heated duct: jet {jet:.2f} x mean; hottest kT_w/(qa) {peaks}, {cooler:.1%} lower")
    print(f"  on 40 cells across instead of 32: {finer}, {1 - finer[0.05] / finer[0.0]:.1%} lower")
    cmap, norm = plt.get_cmap("YlOrRd"), Normalize(0.0, max(peaks.values()))
    titles = {0.0: ("Insulating walls", "Insulating")}
    titles[0.05] = ("Conducting Hartmann walls, c = 0.05", "c = 0.05")
    frames = []
    with plt.rc_context(PAPER):
        for frame, time in enumerate(cases[0.0]["time"]):
            fig = plt.figure(figsize=(9.6, 4.6), dpi=150)
            profiles, walls = fig.add_axes((0.67, 0.6, 0.31, 0.32)), fig.add_axes((0.67, 0.12, 0.31, 0.32))
            for row, (color, (conductance, case)) in enumerate(zip(SERIES, cases.items())):
                axis = fig.add_axes((0.02, 0.47 - 0.45 * row, 0.6, 0.52), **SOLID)
                _duct_cutaway(axis, case, frame, cmap, norm, labels=row == 0)
                fig.text(0.03, 0.955 - 0.47 * row, titles[conductance][0], fontsize=11, va="top")
                x, z = (0.5 * (face[1:] + face[:-1]) for face in case["faces"][::2])
                profiles.plot(z, case["inlet"][case["inlet"].shape[0] // 2], color=color)
                walls.plot(x, case["wall"][-1].max(axis=1), color=color, lw=0.8, alpha=0.35)
                walls.plot(x, case["wall"][frame].max(axis=1), color=color, label=titles[conductance][1])
            _colorbar(fig, (0.215, 0.1, 0.24, 0.02), cmap, norm, r"$kT/(q\,a)$")
            profiles.set(xlim=(-1, 1), xlabel="$z/a$  (heated wall at $-1$)", ylabel=r"$u/\bar u$ at $y = 0$")
            profiles.set_title(f"Ha = {hartmann:g}: side-layer jets in the Hunt duct", fontsize=10)
            walls.set(xlim=(0, x[-1]), ylim=(0, 1.08 * norm.vmax), xlabel="Distance from the inlet $x/a$")
            walls.set_ylabel(r"Hottest $kT_w/(q\,a)$")
            walls.set_title(rf"Heated wall, Pe = {peclet:g},  t = {time:.0f} $a/\bar u$", fontsize=10)
            walls.legend(loc="upper left")
            frames.append(_rgb(fig))
    _animate(frames, "thermal_heated_duct", width=760, duration=150, quality=70)  # keeps the gradient


def _developed_flow(grid, hartmann: float = 0.0, periodic_span: bool = False):
    wall = BoundaryCondition(NEUMANN)
    conditions = (_WRAP, wall, _WRAP if periodic_span else wall)
    return ChannelProblem(grid, conditions, magnetic_field=(0.0, hartmann, 0.0), forcing=(1.0, 0.0, 0.0))


def _rectangle(cells: int, aspect: float = 1.0):
    """A duct ``[-1, 1] x [-aspect, aspect]`` with ``cells`` uniform cells across its short side."""
    across = (uniform_faces(round(cells / aspect), -1.0, 1.0), uniform_faces(cells, -aspect, aspect))
    return _developed_flow(Grid(uniform_faces(1, 0.0, 1.0), *across))


def _h1(flow, slug: bool = False):
    """The developed H1 duct on ``flow``'s steady velocity, or a uniform one: a peripherally uniform wall
    temperature rising at a unit gradient along the duct, without Joule heat."""
    if slug:
        rest = zero_velocity(flow)
        velocity = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])
    else:
        velocity = solve_steady_state(flow, pseudo_step=1.0e3, linear_restart=600).velocity
    wall = thermal_boundary("temperature")
    thermal = ThermalProblem(flow, (_WRAP, wall, wall), 1.0, 1.0, 0.0, joule_heating=False)
    zero = Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid)
    return developed_heat(thermal, velocity, zero, axial_gradient=1.0), velocity


def _heated_plates(cells: int, hartmann: float, stretched: bool = True) -> float:
    """Plates at ``y = +-a`` each taking in a unit flux, carrying LMhdX's own steady Hartmann flow."""
    across = (
        wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=1 / hartmann, cells_in_layer=6, max_ratio=None)
        if stretched
        else uniform_faces(cells, -1.0, 1.0)
    )
    grid = Grid(uniform_faces(1, 0.0, 1.0), across, uniform_faces(1, 0.0, 1.0))
    # Periodic along z for the Hartmann profile; then the same velocity between insulated z ends one cell
    # apart, since the developed duct has a single periodic axis.
    solved = solve_steady_state(_developed_flow(grid, hartmann, periodic_span=True)).velocity
    flow = _developed_flow(grid)
    plates, ends = (thermal_boundary("heat_flux", q, q, conductivity=1.0) for q in (-1.0, 0.0))
    thermal = ThermalProblem(flow, (_WRAP, plates, ends), 1.0, 1.0, 0.0, joule_heating=False)
    rest = zero_velocity(flow)
    velocity = (rest[0].replace_data(solved[0].data), *rest[1:])
    result = developed_heat(thermal, velocity, Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid))
    return 4.0 / float(result.peak_wall_temperature - result.bulk_temperature)  # q = k = 1, D_h = 4a


def _entrance(hartmann: float, peclet: float = 400.0, cells: int = 64, extent: float = 0.12):
    """Plates heated from an inlet at T = 0, carrying LMhdX's developed Hartmann flow held in an open duct.

    Mean velocity, half gap and rho c_p are one, so ``k = 4/Pe``. Returns x+, the local Nu,
    ``k(T - T_b)/(q a)`` and the cell centres across the gap.
    """
    conductivity, length = 4.0 / peclet, extent * 4.0 * peclet
    widths = [0.04]
    while sum(widths) < length:
        widths.append(1.04 * widths[-1])
    axial = np.concatenate(([0.0], np.cumsum(widths))) * length / sum(widths)
    layer = 1.0 / hartmann if hartmann > 10.0 else 0.1
    across = wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=layer, cells_in_layer=6, max_ratio=None)
    grid = Grid(axial, across, uniform_faces(1, 0.0, 1.0))
    walls = (thermal_boundary("heat_flux", -1.0, -1.0, conductivity=conductivity), _WRAP)
    end = 1.5 * length + 3.0 / conductivity  # the transit and three diffusion times across the gap
    temperature, _, inlet = _held_heat(grid, hartmann, 0.0, walls, conductivity, end)
    temperature, heights, speed = temperature[-1][:, :, 0], np.diff(across), inlet[:, 0]
    bulk = temperature @ (speed * heights) / np.sum(speed * heights)
    wall = 0.5 * (temperature[:, 0] + temperature[:, -1]) + 0.25 * (heights[0] + heights[-1]) / conductivity
    xplus = 0.5 * (axial[1:] + axial[:-1]) / (4.0 * peclet)
    nusselt = 4.0 / (conductivity * (wall - bulk))
    return xplus, nusselt, conductivity * (temperature - bulk[:, None]), grid.centers[1]


def _heated_duct(
    conductance: float, hartmann: float, peclet: float, length: float = 24.0, cells: int = 32
) -> dict:
    """A square duct whose side wall z = -a takes in a unit flux: ``kT/(q a)`` at 40 frames to t = 60 a/u.

    ``cells`` across the section, on ``duct_problem``'s layer-resolving mesh, and twice as many along it.
    """
    section = duct_problem(hartmann=hartmann, cells=cells, wall_conductance=conductance).grid
    grid = Grid(uniform_faces(2 * cells, 0.0, length), section.y_faces, section.z_faces)
    conductivity = 2.0 / peclet  # D_h = 2a
    walls = tuple(thermal_boundary("heat_flux", q, 0.0, conductivity=conductivity) for q in (0.0, -1.0))
    temperature, times, inlet = _held_heat(grid, hartmann, conductance, walls, conductivity, 60.0, frames=40)
    temperature, faces = conductivity * temperature, [np.asarray(face) for face in grid.faces]
    wall = temperature[:, :, :, 0] + 0.5 * (faces[2][1] - faces[2][0])  # the unit flux over half a cell
    return {"temperature": temperature, "wall": wall, "time": times, "inlet": inlet, "faces": faces}


def _held_heat(grid, hartmann, c, walls, k, end, frames=1):
    """Carry heat from a cold inlet on LMhdX's developed profile, mean velocity one, held in an open duct.

    The field is along y with wall conductance ``c`` on the walls normal to it; ``walls`` are the thermal
    conditions of the two transverse axes and ``k`` the conductivity. Van Leer transport; Joule heat left
    out, so the walls' flux is the only heat. Returns ``frames`` temperatures evenly spaced up to time
    ``end``, their times, and the inlet profile.
    """
    wall = BoundaryCondition(NEUMANN)
    conditions = (_WRAP, wall, _WRAP if walls[1].is_periodic else wall)
    duct = ChannelProblem(grid, conditions, magnetic_field=(0, hartmann, 0), wall_conductance=(0, c, 0))
    flow = open_duct(duct, float(np.sum(grid.face_areas(0)[0])))
    inlet = np.asarray(flow.conditions[0].lower)
    rest = zero_velocity(flow)
    velocity = (rest[0].replace_data(jnp.broadcast_to(jnp.asarray(inlet)[None], rest[0].shape)), *rest[1:])
    inflow = thermal_boundary("temperature", 0.0, upper_kind="heat_flux", conductivity=k)
    dt = 0.45 * float(np.min(np.diff(grid.faces[0])) / inlet.max())  # van Leer's Courant limit is 0.5
    flow = dataclasses.replace(flow, dt=dt)
    thermal = ThermalProblem(flow, (inflow, *walls), 1.0, k, 0.0, joule_heating=False, advection="van_leer")
    state = initial_thermal_state(thermal, 0.0, velocity)
    heat = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-12)
    step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    steps = max(1, round(end / frames / dt))
    run = jax.jit(lambda start: advance_thermal(step, start, heat, steps, checkpoint=False))
    temperatures, times = [], []
    for _ in range(frames):
        state, history = run(state)
        if not bool(history.finite.all()):
            raise RuntimeError("the held-flow heat run lost finiteness")
        temperatures.append(np.asarray(state.temperature.data))
        times.append(float(state.time))
    storage = abs(float(history.storage_rate[-1])) / float(history.advective_heat_out[-1])
    print(f"held flow Ha {hartmann:g}, c {c:g}: {grid.shape} cells, final storage/outflow {storage:.1e}")
    return np.array(temperatures), np.array(times), inlet


def _onset_growth(rayleigh: float, dt: float, chandrasekhar: float, wavenumber: float) -> tuple[float, dict]:
    """The critical roll's growth rate in ``kappa/d^2`` with ``d = nu = kappa = 1``, and its shape.

    Power iteration on the time stepper: the perturbation of the conducting state starts at 1e-6, linear, and
    is scaled back to that size every half time unit, far above the solver's round-off in the hydrostatic
    balance; the rate is that of the last of four. The field ``sigma B^2 = Q`` is vertical.
    """
    hartmann = np.sqrt(chandrasekhar)
    height = (  # at Q >= 100 the mesh resolves the Hartmann layers
        wall_resolving_faces(32, 0.0, 1.0, layer_thickness=1 / hartmann, cells_in_layer=5, max_ratio=None)
        if chandrasekhar >= 100.0
        else uniform_faces(32, 0.0, 1.0)
    )
    grid = Grid(uniform_faces(24, 0.0, 2.0 * np.pi / wavenumber), uniform_faces(1, 0.0, 1.0), height)
    conditions = (_WRAP, _WRAP, BoundaryCondition(NEUMANN))
    flow = ChannelProblem(grid, conditions, magnetic_field=(0.0, 0.0, hartmann), dt=dt)
    plates, buoyancy = thermal_boundary("temperature", 1.0, 0.0), {"gravity": (0.0, 0.0, -rayleigh)}
    buoyancy["expansion_coefficient"] = 1.0
    thermal = ThermalProblem(flow, (_WRAP, _WRAP, plates), 1.0, 1.0, 0.0, **buoyancy)
    x, z = np.asarray(grid.centers[0])[:, None, None], np.asarray(grid.centers[2])[None, None, :]
    base = jnp.asarray(np.broadcast_to(1.0 - z, grid.shape))
    state = initial_thermal_state(thermal, base + 1e-6 * np.cos(wavenumber * x) * np.sin(np.pi * z))
    size = float(jnp.linalg.norm(state.temperature.data - base))
    step = build_thermal_step(thermal, dtype=jnp.float64)
    heat = state.temperature.replace_data(jnp.zeros(grid.shape))
    run = jax.jit(lambda start: advance_thermal(step, start, heat, round(0.5 / dt), checkpoint=False))
    for _ in range(4):  # converged to five digits by the third
        state, _ = run(state)
        perturbation = state.temperature.data - base
        grown = float(jnp.linalg.norm(perturbation)) / size
        velocity = tuple(component.replace_data(component.data / grown) for component in state.velocity)
        state = state._replace(
            velocity=velocity, temperature=state.temperature.replace_data(base + perturbation / grown)
        )
    u, w = (np.asarray(state.velocity[c].data)[:, 0, :] for c in (0, 2))
    mode = {
        "x": np.asarray(grid.faces[0]),
        "z": np.asarray(grid.faces[2]),
        "theta": np.asarray(perturbation)[:, 0, :],
        "u": 0.5 * (u[:-1] + u[1:]),  # at the cell centres
        "w": 0.5 * (w[:, :-1] + w[:, 1:]),
    }
    return float(np.log(grown) / 0.5), mode


def _draw_roll(axis, mode: dict, width: float) -> None:
    """The roll's temperature and streamlines, tiled periodically to ``width``."""
    copies = int(np.ceil(width / mode["x"][-1]))
    theta, u, w = (np.tile(mode[name], (copies, 1)) for name in ("theta", "u", "w"))
    period = mode["x"][-1]
    x = np.concatenate([mode["x"][:-1] + k * period for k in range(copies)] + [[copies * period]])
    axis.pcolormesh(x, mode["z"], theta.T / np.max(np.abs(theta)), cmap="RdBu_r", vmin=-1, vmax=1)
    heights, centres = 0.5 * (mode["z"][1:] + mode["z"][:-1]), 0.5 * (x[1:] + x[:-1])
    even = np.linspace(heights[0], heights[-1], 40)  # streamplot needs an even grid
    u, w = (np.array([np.interp(even, heights, column) for column in field]) for field in (u, w))
    axis.streamplot(centres, even, u.T, w.T, color=INK, linewidth=0.6, density=0.9, arrowsize=0.6)
    axis.set(xlim=(0, width), ylim=(0, 1), yticks=[0, 1], ylabel="$z/d$", aspect="equal")


def _convection(field: float, rayleigh: float, prandtl: float, cells: int = 64, end: float = 150.0):
    """A periodic 4 x 4 x 1 layer between rigid isothermal plates in free-fall units, from a perturbed rest.

    A horizontal field ``field`` along x. Every 2.5 time units: the temperature and the Nusselt number at the
    hot plate. Van Leer heat transport and limited momentum transport, both inside their Courant limits.
    """
    dt, every = 0.0125, 2.5
    grid = Grid(uniform_faces(cells, 0, 4), uniform_faces(cells, 0, 4), uniform_faces(cells // 2, 0, 1))
    conditions = (_WRAP, _WRAP, BoundaryCondition(NEUMANN))
    motion = {"viscosity": np.sqrt(prandtl / rayleigh), "dt": dt, "advection": "limited"}
    flow = ChannelProblem(grid, conditions, magnetic_field=(field, 0.0, 0.0), **motion)
    plates, diffusivity = thermal_boundary("temperature", 0.5, -0.5), 1.0 / np.sqrt(rayleigh * prandtl)
    buoyancy = {"expansion_coefficient": 1.0, "gravity": (0.0, 0.0, -1.0), "advection": "van_leer"}
    thermal = ThermalProblem(flow, (_WRAP, _WRAP, plates), 1.0, diffusivity, 0.0, **buoyancy)
    z = np.asarray(grid.centers[2])[None, None, :]
    noise = np.random.default_rng(7).standard_normal(grid.shape)
    state = initial_thermal_state(thermal, 0.5 - z + 1e-3 * noise * np.sin(np.pi * z))
    heat = state.temperature.replace_data(jnp.zeros(grid.shape))
    validate_thermal_inputs(thermal, state, heat, divergence_tolerance=1e-12)
    step = build_thermal_step(thermal, dtype=jnp.float64)
    run = jax.jit(lambda start: advance_thermal(step, start, heat, round(every / dt), checkpoint=False))
    temperatures, nusselt, courant = [], [], 0.0
    for _ in range(round(end / every)):
        state, history = run(state)
        courant = max(courant, float(history.advective_cfl.max()))
        if not bool(history.finite.all()) or courant > 0.5:
            raise RuntimeError(f"the convection run left its envelope: Courant {courant:.3f}")
        temperatures.append(np.asarray(state.temperature.data, dtype=np.float32))
        nusselt.append(float(np.mean(0.5 - temperatures[-1][:, :, 0])) * cells)  # over half a cell
    print(f"convection, field {field:.3g}: largest Courant {courant:.3f}, Nu at the end {nusselt[-1]:.3f}")
    return np.array(temperatures), np.array(nusselt)


def _panel(axis, letter: str) -> None:
    """A journal-style panel letter at the top left."""
    style = {"transform": axis.transAxes, "fontsize": 12, "fontweight": "bold"}
    if hasattr(axis, "text2D"):
        axis.text2D(0.0, 0.97, f"({letter})", **style)
    else:
        axis.text(-0.02, 1.03, f"({letter})", ha="right", **style)


def _axes_3d(fig, spec, labels, view, ticks=((-1, 0, 1), (-1, 0, 1))):
    """3-D axes on a light box, with the vertical label upright whichever side the view puts it on."""
    axis = fig.add_subplot(spec, projection="3d")
    axis.zaxis.set_rotate_label(False)
    axis.set(xlabel=labels[0], ylabel=labels[1], xticks=ticks[0], yticks=ticks[1])
    axis.set_zlabel(labels[2], rotation=90, labelpad=2)
    axis.view_init(*view)
    axis.tick_params(pad=-1)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set(facecolor=(0.97, 0.97, 0.96, 1.0), edgecolor="0.8")
    return axis


def _paint(axis, x, y, z, colors) -> None:
    """One colored quad per cell: ``x, y, z`` are the corner grids, ``colors`` one RGBA per quad."""
    style = {"rstride": 1, "cstride": 1, "linewidth": 0, "antialiased": False, "shade": False}
    axis.plot_surface(x, y, z, facecolors=colors, **style)


def _face(axis, normal: int, position: float, u, v, values, cmap, norm) -> None:
    """Paint ``values`` on the plane ``normal = position``; ``u`` and ``v`` are the faces along the others."""
    mesh = list(np.meshgrid(u, v, indexing="ij"))
    mesh.insert(normal, np.full_like(mesh[0], position))
    _paint(axis, *mesh, cmap(norm(values)))


def _cutaway(axis, temperature, faces, cmap, norm, cut: int) -> None:
    """A box cut open at its ``cut``-th face along z: four side walls and the slice, back to front."""
    x, y, z = faces[0], faces[1], faces[2][: cut + 1]
    for normal, position, u, v, values in (
        (0, x[0], y, z, temperature[0, :, :cut]),
        (1, y[-1], x, z, temperature[:, -1, :cut]),
        (2, z[-1], x, y, temperature[:, :, cut - 1]),
        (0, x[-1], y, z, temperature[-1, :, :cut]),
        (1, y[0], x, z, temperature[:, 0, :cut]),
    ):
        _face(axis, normal, position, u, v, values, cmap, norm)
    axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(0.0, 2.0 * z[-1]))


def _duct_cutaway(axis, case: dict, frame: int, cmap, norm, labels: bool) -> None:
    """The duct cut open toward the viewer: the heated side wall behind, the bottom wall, the outlet section.

    Display axes: x along the duct, ``-z`` into the page (the heated wall z = -a behind), the field's y up.
    """
    x, y, z = case["faces"]
    depth, temperature = -z[::-1], case["temperature"][frame][:, :, ::-1]
    _face(axis, 1, 1.0, x, y, case["wall"][frame], cmap, norm)
    _face(axis, 2, -1.0, x, depth, temperature[:, 0, :], cmap, norm)
    _face(axis, 0, x[-1], depth, y, temperature[-1].T, cmap, norm)
    for start, end in (((0, -1, -1), (0, 1, -1)), ((0, 1, -1), (0, 1, 1)), ((0, 1, 1), (x[-1], 1, 1))):
        axis.plot(*zip(start, end), color=MUTED, lw=0.7)
    if labels:
        axis.text(x[-1] * 0.5, 1.0, 1.25, "Heated wall", color=MUTED, fontsize=9, ha="center")
        axis.text2D(0.0, 0.42, "Flow →", transform=axis.transAxes, color=INK, fontsize=10)
        axis.text2D(0.0, 0.55, r"↑ $\mathbf{B}$", transform=axis.transAxes, color=INK, fontsize=11)
    axis.set(xlim=(0, x[-1]), ylim=(-1, 1), zlim=(-1, 1))
    axis.set_box_aspect((x[-1] / 2.4, 2, 2), zoom=1.12)
    axis.view_init(elev=22, azim=-62)
    axis.set_axis_off()


def _colorbar(fig, rect, cmap, norm, label: str, ticks=None) -> None:
    mappable = plt.cm.ScalarMappable(norm, cmap)
    bar = fig.colorbar(mappable, cax=fig.add_axes(rect), orientation="horizontal", ticks=ticks)
    bar.set_label(label, fontsize=9.5)
    bar.ax.tick_params(labelsize=8.5, direction="out", top=False)
    bar.ax.minorticks_off()


def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _animate(frames: list, name: str, width: int, duration: int, quality: int = 45) -> None:
    """An animated WebP, each frame downsampled to ``width`` pixels for smooth edges; ``minimize_size`` stores
    only what changes between frames. The movie is a release asset, written to the untracked
    ``docs/_static/generated``; its last frame, 480 px wide, is the tracked preview."""
    height = round(frames[0].shape[0] * width / frames[0].shape[1])
    images = [Image.fromarray(frame).resize((width, height), Image.LANCZOS) for frame in frames]
    options = {"duration": duration, "loop": 0, "quality": quality, "method": 6, "minimize_size": True}
    path = STATIC / "generated" / f"{name}.webp"
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], **options)
    last = Image.fromarray(frames[-1])
    last.resize((480, round(last.height * 480 / last.width)), Image.LANCZOS).save(
        STATIC / f"{name}_preview.webp", quality=80, method=6
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path.name}: {len(images)} frames, {path.stat().st_size / 1024:.0f} KiB, sha256 {digest}")


RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def benchmark_scaling() -> None:
    """Time per step against problem size, from the JSON the benchmark harness writes.

    Nothing here is measured by this script: it reads
    ``benchmarks/results/*.json`` so that a figure and the number it draws come
    from the same run, on the machine named in the file. A missing device or
    precision is simply absent from the plot rather than interpolated.
    """
    reports = sorted(RESULTS.glob("*.json"))
    if not reports:
        raise SystemExit(f"no benchmark JSON in {RESULTS}; run scripts/run_benchmarks.py first")
    # (case, device, mode) -> {cells: (median, low, high)} seconds per step, where the
    # mode is fp32, fp64, or mixed (float32 solves with float64 correction).
    series: dict = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        environment = report["environment"]
        for entry in report["cases"]:
            # The host-sync and sharding cases report their own numbers; this
            # figure is about time against size on one device.
            median = entry.get("seconds_per_step")
            if not entry.get("accepted") or not isinstance(median, float):
                continue
            mode = (
                "fp32" if not environment["x64"] else "mixed" if entry.get("precision") == "mixed" else "fp64"
            )
            low, high = entry.get("seconds_per_step_ci95", (median, median))
            key = (entry["case"], environment["platform"].upper(), mode)
            series.setdefault(key, {})[entry["cells"]] = (median, low, high)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    styles = {"fp64": "-", "fp32": "--", "mixed": ":"}
    colors = {"CPU": "tab:blue", "GPU": "tab:red"}
    for index, case in enumerate(("core3d_advance", "q2d_evolve")):
        for (name, device, mode), points in sorted(series.items()):
            if name != case:
                continue
            cells = np.array(sorted(points))
            median, low, high = (np.array(values) for values in zip(*(points[n] for n in cells), strict=True))
            axes[index].errorbar(
                cells,
                median,
                yerr=(median - low, high - median),
                fmt=styles[mode],
                marker="o",
                ms=4,
                capsize=2,
                color=colors[device],
                label=f"{device}, {mode}",
            )
        axes[index].set_xscale("log")
        axes[index].set_yscale("log")
        axes[index].set_xlabel("cells")
        axes[index].set_ylabel("seconds per step (median, 95 % CI)")
        axes[index].legend(frameon=False, fontsize=8)
    axes[0].set_title("Staggered 3-D core", fontsize=11)
    axes[1].set_title("Quasi-2D evolution", fontsize=11)

    def fastest(case: str, device: str, modes: tuple[str, ...]) -> dict:
        """Per size, the fastest median among the given modes on one device."""
        times = [series.get((case, device, mode), {}) for mode in modes]
        return {n: min(t[n][0] for t in times if n in t) for n in set().union(*times)}

    # Each device in its fastest float64-accurate mode, and float32 against float32.
    comparisons = (
        ("core3d_advance", ("fp64", "mixed"), "-", "tab:purple", "3-D core, float64-accurate"),
        ("core3d_advance", ("fp32",), "--", "tab:purple", "3-D core, fp32"),
        ("q2d_evolve", ("fp64",), "-", "tab:green", "Q2D, fp64"),
        ("q2d_evolve", ("fp32",), "--", "tab:green", "Q2D, fp32"),
    )
    for case, modes, style, color, label in comparisons:
        cpu, gpu = fastest(case, "CPU", modes), fastest(case, "GPU", modes)
        shared = sorted(set(cpu) & set(gpu))
        if shared:
            axes[2].loglog(
                shared, [cpu[n] / gpu[n] for n in shared], style, marker="o", ms=4, color=color, label=label
            )
    low, high = axes[2].get_ylim()
    axes[2].set_ylim(min(low, 5.0), max(high, 30.0))
    for target, name in ((10.0, "3-D core target"), (20.0, "Q2D target")):
        axes[2].axhline(target, color="k", ls=":", lw=1)
        axes[2].annotate(name, (axes[2].get_xlim()[0], target), fontsize=7, va="bottom")
    plain = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}x")
    axes[2].yaxis.set_major_formatter(plain)
    axes[2].yaxis.set_minor_formatter(plain)
    axes[2].tick_params(axis="y", which="minor", labelsize=7)
    axes[2].set_xlabel("cells")
    axes[2].set_ylabel("GPU speed-up over CPU")
    axes[2].set_title("Targets: 10x (3-D core), 20x (Q2D)", fontsize=11)
    axes[2].legend(frameon=False, fontsize=8)
    _save_webp(fig, STATIC / "device_scaling.webp", dpi=100)


def _save_webp(fig: plt.Figure, path: Path, dpi: int = 120, palette: bool = False) -> None:
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    image = Image.open(png).convert("RGB")
    if palette:
        # 128 colors, losslessly, to keep the Git media budget. Median cut keeps a near-white apart from the page.
        image = image.quantize(128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        image.save(path, lossless=True, quality=100, method=6)
    else:
        image.save(path, quality=82, method=6)
    png.unlink()
    plt.close(fig)
    print(f"wrote {path.name}: {path.stat().st_size / 1024:.0f} KiB")


THERMAL = {
    "developed": thermal_developed,
    "entrance": thermal_entrance,
    "onset": thermal_onset,
    "convection": thermal_convection,
    "heated-duct": thermal_heated_duct,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        choices=("q2d", "hunt", "ladder", "pipe", "scaling", *THERMAL),
        help="Regenerate one asset group.",
    )
    args = parser.parse_args()
    STATIC.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "q2d"):
        q2d_turbulence()
    if args.only in (None, "hunt"):
        hunt_sweep()
    if args.only in (None, "ladder"):
        validation_ladder()
    if args.only in (None, "pipe"):
        pipe_sweep()
    for name, figure in THERMAL.items():
        if args.only in (None, name):
            figure()
    if args.only == "scaling":
        benchmark_scaling()


if __name__ == "__main__":
    main()
