"""Outputs, evidence reports and the command line.

A solved case is written as ParaView files, midplane and centreline profiles,
NPZ archives, restart bundles and figures; the analytical, conservation and
benchmark reports (Hartmann acceptance, profile metrics, solver benchmarks) read
the same solutions; and ``lmhdx`` (:func:`main`) runs a named or TOML case, a
validation or a benchmark from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .cases import (
    Diagnostics,
    MHDState,
    RestartLogInfo,
    RunConfig,
    Solution,
    StreamingSolverLogger,
    StructuredMesh,
    _load_toml,
    default_log_path,
    load_run_config,
    make_hartmann_case,
    make_hunt_case,
    make_shercliff_case,
)
from .fully_developed import case_mesh, solve_fully_developed, solve_fully_developed_transient
from .ops import cell_average


@dataclass(frozen=True)
class AnalyticComparison:
    coordinate: jnp.ndarray
    simulated: jnp.ndarray
    reference: jnp.ndarray
    l2_error: float
    linf_error: float


@dataclass(frozen=True)
class ProfileSymmetry:
    axis: str
    mean_abs_error: float
    max_abs_error: float


@dataclass(frozen=True)
class AcceptanceReport:
    case_name: str
    l2_error: float
    linf_error: float
    l2_threshold: float
    linf_threshold: float
    passed_l2: bool
    passed_linf: bool
    passed: bool


# A history the solver does not record (an interface the model has no wall for,
# a potential solve the core does not run) is reported as None, never as zero.
_SUMMARY_HISTORIES = (
    ("potential_residual", "potential_residual_history"),
    ("potential_iterations_used", "potential_iterations_history"),
    ("mean_velocity", "mean_velocity_history"),
    ("applied_forcing", "applied_forcing_history"),
    ("linear_residual", "linear_residual_history"),
    ("linear_iterations_used", "linear_iterations_history"),
    ("volumetric_flow_rate", "volumetric_flow_rate_history"),
    ("mean_current_magnitude", "mean_current_magnitude_history"),
    ("lorentz_power", "lorentz_power_history"),
    ("div_current_max", "div_current_max_history"),
    ("charge_balance_residual", "charge_balance_residual_history"),
    ("gauge_residual", "gauge_residual_history"),
    ("interface_current_residual", "interface_current_residual_history"),
)


def hartmann_analytic_profile(y: jnp.ndarray, ha: float) -> jnp.ndarray:
    denom = jnp.cosh(ha) - 1.0
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    return 1.0 - (jnp.cosh(ha * y) - 1.0) / denom


def _exact_coordinate_index(
    coordinate: jnp.ndarray, *, target: float = 0.0, tolerance: float = 1.0e-12
) -> int | None:
    coordinate = jnp.asarray(coordinate, dtype=float)
    matches = np.where(np.abs(np.asarray(coordinate, dtype=float) - float(target)) <= tolerance)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _midplane_indices(coordinate: jnp.ndarray) -> tuple[int, int, float]:
    if coordinate.size == 1:
        return 0, 0, 0.0
    center = _exact_coordinate_index(coordinate)
    if center is not None:
        return center, center, 0.0
    upper = max(1, min(int(jnp.searchsorted(coordinate, 0.0)), coordinate.size - 1))
    lower = upper - 1
    span = float(coordinate[upper] - coordinate[lower])
    weight = 0.5 if abs(span) <= 1.0e-12 else -float(coordinate[lower]) / span
    return lower, upper, min(max(weight, 0.0), 1.0)


def _midplane_values(
    field: jnp.ndarray, coordinate: jnp.ndarray, fixed_axis: int
) -> tuple[jnp.ndarray, tuple[int, int]]:
    lower, upper, weight = _midplane_indices(coordinate)
    lower_values = jnp.take(field, lower, axis=fixed_axis)
    if lower == upper:
        return lower_values, (lower, upper)
    upper_values = jnp.take(field, upper, axis=fixed_axis)
    return (1.0 - weight) * lower_values + weight * upper_values, (lower, upper)


def _midplane_field(
    solution: Solution,
    field: jnp.ndarray,
    axis: str,
    fluid_only: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if axis == "y":
        coordinate, fixed_coordinate, fixed_axis = (
            solution.mesh.y_centers,
            solution.mesh.z_centers,
            1,
        )
    elif axis == "z":
        coordinate, fixed_coordinate, fixed_axis = (
            solution.mesh.z_centers,
            solution.mesh.y_centers,
            0,
        )
    else:
        raise ValueError(f"Unsupported axis {axis}")
    values, (lower, upper) = _midplane_values(field, fixed_coordinate, fixed_axis)
    if not fluid_only or solution.mesh.fluid_mask is None:
        return coordinate, values
    mask = jnp.take(solution.mesh.fluid_mask, lower, axis=fixed_axis)
    if lower != upper:
        mask &= jnp.take(solution.mesh.fluid_mask, upper, axis=fixed_axis)
    return coordinate[mask], values[mask]


def extract_centerline(solution: Solution) -> dict[str, jnp.ndarray]:
    return extract_midplane_profile(solution, axis="y")


def extract_midplane_profile(
    solution: Solution, axis: str = "y", fluid_only: bool = False
) -> dict[str, jnp.ndarray]:
    phi = getattr(solution.state, "phi", jnp.zeros_like(solution.state.u))
    coordinate, velocity = _midplane_field(solution, solution.state.u, axis, fluid_only)
    _, potential = _midplane_field(solution, phi, axis, fluid_only)
    return {axis: coordinate, "u": velocity, "phi": potential}


def compare_profile_to_reference(
    coordinate: jnp.ndarray,
    simulated: jnp.ndarray,
    reference: jnp.ndarray,
) -> AnalyticComparison:
    diff = simulated - reference
    l2 = float(jnp.sqrt(jnp.mean(diff**2)))
    linf = float(jnp.max(jnp.abs(diff)))
    return AnalyticComparison(
        coordinate=coordinate,
        simulated=simulated,
        reference=reference,
        l2_error=l2,
        linf_error=linf,
    )


def symmetry_metrics(profile: jnp.ndarray, axis: str) -> ProfileSymmetry:
    mirrored = jnp.flip(profile)
    diff = profile - mirrored
    return ProfileSymmetry(
        axis=axis,
        mean_abs_error=float(jnp.mean(jnp.abs(diff))),
        max_abs_error=float(jnp.max(jnp.abs(diff))),
    )


def profile_sign_changes(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> int:
    signs = jnp.where(jnp.abs(profile) <= tolerance, 0.0, jnp.sign(profile))
    left = signs[:-1]
    right = signs[1:]
    transitions = (left * right) < 0.0
    return int(jnp.sum(transitions))


def negative_fraction(profile: jnp.ndarray, *, tolerance: float = 1e-12) -> float:
    return float(jnp.mean((profile < -tolerance).astype(float)))


def duct_profile_metrics(solution: Solution) -> dict[str, float]:
    y_profile = extract_midplane_profile(solution, axis="y")["u"]
    z_profile = extract_midplane_profile(solution, axis="z")["u"]
    y_sym = symmetry_metrics(y_profile, axis="y")
    z_sym = symmetry_metrics(z_profile, axis="z")
    return {
        "symmetry_y_mean_abs_error": y_sym.mean_abs_error,
        "symmetry_y_max_abs_error": y_sym.max_abs_error,
        "symmetry_z_mean_abs_error": z_sym.mean_abs_error,
        "symmetry_z_max_abs_error": z_sym.max_abs_error,
        "centerline_y_sign_changes": float(profile_sign_changes(y_profile)),
        "centerline_z_sign_changes": float(profile_sign_changes(z_profile)),
        "centerline_y_negative_fraction": negative_fraction(y_profile),
        "centerline_z_negative_fraction": negative_fraction(z_profile),
        "u_max": float(jnp.max(solution.state.u)),
        "u_mean": float(jnp.mean(solution.state.u)),
    }


def validation_summary(
    solution: Solution, case_name: str, ha: float | None = None
) -> dict[str, float | str | None]:
    payload: dict[str, float | str | None] = {
        "case": case_name,
        "time": solution.state.time,
        "residual": solution.state.residual,
    }
    for name, attribute in _SUMMARY_HISTORIES:
        history = getattr(solution.diagnostics, attribute)
        payload[name] = float(history[-1]) if history.size else None
    payload.update(duct_profile_metrics(solution))
    if case_name.startswith("hartmann") and ha is not None:
        comparison = hartmann_validation(solution, ha)
        payload["l2_error"] = comparison.l2_error
        payload["linf_error"] = comparison.linf_error
    return payload


def hartmann_validation(solution: Solution, ha: float) -> AnalyticComparison:
    profile = extract_centerline(solution)
    half_width = 0.5 * float(solution.mesh.y_faces[-1] - solution.mesh.y_faces[0])
    scale_y = half_width if half_width > 0.0 else float(jnp.max(jnp.abs(profile["y"])))
    coordinate = profile["y"] / max(scale_y, 1.0e-12)
    u = profile["u"]
    scale = jnp.max(jnp.abs(u))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = u / scale
    reference = hartmann_analytic_profile(coordinate, ha)
    return compare_profile_to_reference(coordinate, normalized, reference)


def hartmann_acceptance(
    solution: Solution,
    ha: float,
    *,
    l2_threshold: float,
    linf_threshold: float,
) -> AcceptanceReport:
    comparison = hartmann_validation(solution, ha)
    passed_l2 = comparison.l2_error <= l2_threshold
    passed_linf = comparison.linf_error <= linf_threshold
    return AcceptanceReport(
        case_name=f"hartmann_ha{int(ha)}",
        l2_error=comparison.l2_error,
        linf_error=comparison.linf_error,
        l2_threshold=float(l2_threshold),
        linf_threshold=float(linf_threshold),
        passed_l2=passed_l2,
        passed_linf=passed_linf,
        passed=passed_l2 and passed_linf,
    )


def write_profile_csv(path: str | Path, data: dict[str, jnp.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    rows = zip(*(jnp.asarray(data[key]).tolist() for key in keys))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)
    return path


def _write_json(payload: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _comparison_payload(comparison: AnalyticComparison, axis_name: str = "coordinate") -> dict[str, object]:
    return {
        axis_name: jnp.asarray(comparison.coordinate).tolist(),
        "simulated": jnp.asarray(comparison.simulated).tolist(),
        "reference": jnp.asarray(comparison.reference).tolist(),
        "l2_error": comparison.l2_error,
        "linf_error": comparison.linf_error,
    }


def write_analytic_comparison(
    comparison: AnalyticComparison, path: str | Path, axis_name: str = "coordinate"
) -> Path:
    return _write_json(_comparison_payload(comparison, axis_name), path)


def write_acceptance_report(report: AcceptanceReport, path: str | Path) -> Path:
    return _write_json(vars(report), path)


def write_metrics_json(metrics: dict[str, float | str], path: str | Path) -> Path:
    return _write_json(metrics, path)


def benchmark_solver(
    repeats: int = 3, ha: float = 20.0, ny: int = 48, nz: int = 48
) -> dict[str, float | str]:
    from .fully_developed import solve_fully_developed

    case = make_hartmann_case(ha=ha, ny=ny, nz=nz)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        solution = solve_fully_developed(case)
        jax.block_until_ready((solution.fields.u, solution.fields.phi))
        timings.append(time.perf_counter() - start)
    cold = timings[0]
    warm_samples = np.asarray(timings[1:] or timings)
    warm = float(np.median(warm_samples))
    return {
        "case": case.name,
        "ha": ha,
        "ny": float(ny),
        "nz": float(nz),
        "repeats": float(repeats),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_cv": float(np.std(warm_samples, ddof=1) / warm) if warm_samples.size > 1 else 0.0,
        "mean_seconds": sum(timings) / len(timings),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }


def write_benchmark_report(report: dict[str, float | str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


# Solution files, restart bundles and figures.

_DIAGNOSTIC_FIELDS = tuple(item.name for item in fields(Diagnostics))


def _portable_path(path: str | Path) -> str:
    candidate, base = Path(path), Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


@dataclass(frozen=True)
class RestartBundle:
    path: Path
    state: MHDState
    diagnostics: Diagnostics
    metadata: dict[str, object]
    y_faces: np.ndarray
    z_faces: np.ndarray
    geometry_kind: str


def _array_text(array: jnp.ndarray) -> str:
    return " ".join(f"{float(value):.12e}" for value in jnp.ravel(array))


def write_vtr(path: str | Path, faces, **cells) -> Path:
    """Write named cell arrays as a ParaView rectilinear grid; returns ``path`` with the ``.vtr`` suffix.

    ``faces`` is the three axes' face coordinates, and each array is ``(nx, ny, nz)`` or has a trailing
    component axis of three; the cells are written with x varying fastest, as VTK orders them.
    """
    target = Path(path).with_suffix(".vtr")
    target.parent.mkdir(parents=True, exist_ok=True)
    extent = " ".join(f"0 {np.size(axis) - 1}" for axis in faces)

    def entry(name, data, extra=""):
        return (
            f'<DataArray type="Float64" Name="{name}"{extra} format="ascii">{_array_text(data)}</DataArray>'
        )

    def cell(name, data):
        data = np.asarray(data)
        vector = data.ndim == 4
        data = np.transpose(data, (2, 1, 0, 3) if vector else (2, 1, 0))
        return entry(name, data, ' NumberOfComponents="3"' if vector else "")

    target.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
        f'<RectilinearGrid WholeExtent="{extent}">\n<Piece Extent="{extent}">\n<CellData>\n'
        + "\n".join(cell(name, data) for name, data in cells.items())
        + "\n</CellData>\n<Coordinates>\n"
        + "\n".join(entry(name, axis) for name, axis in zip("XYZ", faces))
        + "\n</Coordinates>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n"
    )
    return target


def write_pvd(entries: list[tuple[float, str]], out_dir: str | Path, name: str = "series") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.pvd"
    datasets = "\n".join(
        f'<DataSet timestep="{time:.8f}" group="" part="0" file="{filename}"/>' for time, filename in entries
    )
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "<Collection>\n"
        f"{datasets}\n"
        "</Collection>\n"
        "</VTKFile>\n"
    )
    target.write_text(content)
    return target


def write_paraview(solution: Solution, out_dir: str | Path) -> list[Path]:
    mesh, state = solution.mesh, solution.state
    cells = {name: getattr(state, name)[None] for name in ("u", "phi", "jy", "jz", "lorentz_x")}
    paths = [
        write_vtr(Path(out_dir) / solution.case_name, (mesh.x_faces, mesh.y_faces, mesh.z_faces), **cells)
    ]
    paths.append(write_pvd([(solution.state.time, paths[0].name)], out_dir, name=solution.case_name))
    return paths


def write_solution_npz(solution: Solution, case, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The solution covers the fluid alone, so its material arrays are the fluid's, uniform.
    fluid = next(region for region in case.regions if region.kind == "fluid")
    shape = np.shape(solution.state.u)
    metadata = {
        "case": solution.case_name,
        "time": float(solution.state.time),
        "residual": float(solution.state.residual),
        "description": "LMhdX solution dump",
        "geometry_kind": case.geometry.kind,
        "notes": case.notes,
        "restart_capable": True,
    }
    diag = solution.diagnostics
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        y_centers=np.asarray(solution.mesh.y_centers),
        z_centers=np.asarray(solution.mesh.z_centers),
        y_faces=np.asarray(solution.mesh.y_faces),
        z_faces=np.asarray(solution.mesh.z_faces),
        u=np.asarray(solution.state.u),
        phi=np.asarray(solution.state.phi),
        jy=np.asarray(solution.state.jy),
        jz=np.asarray(solution.state.jz),
        lorentz_x=np.asarray(solution.state.lorentz_x),
        state_time=np.asarray(float(solution.state.time)),
        state_residual=np.asarray(float(solution.state.residual)),
        conductivity=np.full(shape, fluid.conductivity),
        density=np.full(shape, fluid.density or 1.0),
        viscosity=np.full(shape, fluid.viscosity or 1.0),
        fluid_mask=np.ones(shape, dtype=bool),
        **{name: np.asarray(getattr(diag, name)) for name in _DIAGNOSTIC_FIELDS},
    )
    return path


def write_restart_npz(solution: Solution, case, path: str | Path) -> Path:
    return write_solution_npz(solution, case, path)


def _load_optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        return np.zeros((0,), dtype=float)
    return np.asarray(data[key])


def load_restart_bundle(path: str | Path) -> RestartBundle:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        state_time = float(data["state_time"]) if "state_time" in data else float(metadata.get("time", 0.0))
        if "state_residual" in data:
            state_residual = float(data["state_residual"])
        else:
            residual_history = _load_optional_array(data, "residual_history")
            state_residual = (
                float(residual_history[-1]) if residual_history.size else float(metadata.get("residual", 0.0))
            )
        state = MHDState(
            u=jnp.asarray(data["u"]),
            phi=jnp.asarray(data["phi"]),
            jy=jnp.asarray(data["jy"]),
            jz=jnp.asarray(data["jz"]),
            lorentz_x=jnp.asarray(data["lorentz_x"]),
            time=state_time,
            residual=state_residual,
        )
        diagnostics = Diagnostics(
            **{name: jnp.asarray(_load_optional_array(data, name)) for name in _DIAGNOSTIC_FIELDS}
        )
        return RestartBundle(
            path=path,
            state=state,
            diagnostics=diagnostics,
            metadata=metadata,
            y_faces=np.asarray(data["y_faces"]),
            z_faces=np.asarray(data["z_faces"]),
            geometry_kind=str(metadata.get("geometry_kind", "unknown")),
        )


def validate_restart_bundle(
    bundle: RestartBundle, *, mesh: StructuredMesh, geometry_kind: str, case_name: str
) -> None:
    if bundle.geometry_kind not in {"unknown", geometry_kind}:
        raise ValueError(
            f"Restart geometry_kind {bundle.geometry_kind!r} does not match current case geometry {geometry_kind!r}"
        )
    if bundle.state.u.shape != mesh.yz_shape:
        raise ValueError(
            f"Restart field shape {bundle.state.u.shape!r} does not match current mesh shape {mesh.yz_shape!r}"
        )
    if bundle.y_faces.shape != np.asarray(mesh.y_faces).shape or not np.allclose(
        bundle.y_faces, np.asarray(mesh.y_faces)
    ):
        raise ValueError("Restart y_faces do not match the current mesh")
    if bundle.z_faces.shape != np.asarray(mesh.z_faces).shape or not np.allclose(
        bundle.z_faces, np.asarray(mesh.z_faces)
    ):
        raise ValueError("Restart z_faces do not match the current mesh")
    restart_case = str(bundle.metadata.get("case", case_name))
    if restart_case != case_name:
        metadata_name = bundle.metadata.get("case")
        if metadata_name is not None:
            raise ValueError(f"Restart case {metadata_name!r} does not match current case name {case_name!r}")


def write_solution_outputs(
    solution: Solution,
    case,
    out_dir: str | Path,
    *,
    write_npz: bool = True,
    write_plots: bool = False,
) -> dict[str, list[Path]]:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[Path]] = {"paraview": [], "csv": [], "npz": [], "plots": []}

    if case.output.write_paraview:
        payload["paraview"] = write_paraview(solution, out_dir)
    if case.output.write_csv_profiles:
        payload["csv"] = [
            write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution)),
            write_profile_csv(
                out_dir / f"{case.name}_midplane_y.csv",
                extract_midplane_profile(solution, axis="y", fluid_only=True),
            ),
            write_profile_csv(
                out_dir / f"{case.name}_midplane_z.csv",
                extract_midplane_profile(solution, axis="z", fluid_only=True),
            ),
        ]
    if write_npz and case.output.write_npz:
        payload["npz"] = [write_solution_npz(solution, case, out_dir / f"{case.name}_results.npz")]
    if write_plots and case.output.write_plots:
        payload["plots"] = write_case_overview_plots(solution, out_dir, case_title=case.name)
    return payload


def _load_matplotlib() -> None:
    global plt, colors
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors


def _set_plot_style() -> None:
    _load_matplotlib()
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "grid.color": "#4f4f4f",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
        }
    )


def _prepare_plot_output(out_dir: str | Path) -> Path:
    """Load plotting dependencies, apply the house style, and create output."""

    _set_plot_style()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_figure_pair(fig, out_dir: Path, stem: str) -> list[Path]:
    """Save one figure as PNG and PDF, then release its Matplotlib state."""

    from matplotlib import pyplot

    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    pyplot.close(fig)
    return paths


def _plot_field(ax: plt.Axes, solution: Solution, field: jnp.ndarray, *, title: str, cmap: str) -> None:
    _load_matplotlib()
    mesh = solution.mesh
    field_min = float(jnp.min(field))
    field_max = float(jnp.max(field))
    if field_min >= 0.0:
        cmap = "magma"
        norm = colors.Normalize(vmin=field_min, vmax=max(field_max, field_min + 1e-12))
    elif field_max <= 0.0:
        cmap = "magma_r"
        norm = colors.Normalize(vmin=min(field_min, field_max - 1e-12), vmax=field_max)
    else:
        vmax = float(jnp.max(jnp.abs(field)))
        vmax = max(vmax, 1e-12)
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.pcolormesh(
        mesh.z_faces,
        mesh.y_faces,
        field,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_profile(
    ax: plt.Axes,
    coordinate: jnp.ndarray,
    values: jnp.ndarray,
    *,
    axis_name: str,
    title: str,
    reference_coordinate: jnp.ndarray | None = None,
    reference_values: jnp.ndarray | None = None,
    reference_label: str | None = None,
) -> None:
    coord_scale = float(jnp.max(jnp.abs(coordinate)))
    coord_scale = coord_scale if coord_scale > 0.0 else 1.0
    value_scale = float(jnp.max(jnp.abs(values)))
    value_scale = value_scale if value_scale > 0.0 else 1.0
    ax.plot(coordinate / coord_scale, values / value_scale, color="#0f766e", label="LMhdX")
    if reference_coordinate is not None and reference_values is not None:
        ref_coord_scale = float(jnp.max(jnp.abs(reference_coordinate)))
        ref_coord_scale = ref_coord_scale if ref_coord_scale > 0.0 else 1.0
        ref_value_scale = float(jnp.max(jnp.abs(reference_values)))
        ref_value_scale = ref_value_scale if ref_value_scale > 0.0 else 1.0
        ax.plot(
            reference_coordinate / ref_coord_scale,
            reference_values / ref_value_scale,
            color="#b45309",
            linestyle="--",
            label=reference_label or "Reference",
        )
    ax.set_title(title)
    ax.set_xlabel(f"Normalized {axis_name}")
    ax.set_ylabel("Normalized velocity")
    ax.set_xlim(-1.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))


def write_case_overview_plots(
    solution: Solution,
    out_dir: str | Path,
    *,
    case_title: str,
    y_reference_coordinate: jnp.ndarray | None = None,
    y_reference_values: jnp.ndarray | None = None,
    z_reference_coordinate: jnp.ndarray | None = None,
    z_reference_values: jnp.ndarray | None = None,
    reference_label: str = "Reference",
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16, y=1.02)

    _plot_field(axes[0, 0], solution, solution.state.u, title="Velocity u", cmap="RdBu_r")
    _plot_field(
        axes[0, 1],
        solution,
        solution.state.phi,
        title="Electric potential φ",
        cmap="PuOr_r",
    )
    _plot_profile(
        axes[1, 0],
        y_profile["y"],
        y_profile["u"],
        axis_name="y",
        title="Midplane y profile",
        reference_coordinate=y_reference_coordinate,
        reference_values=y_reference_values,
        reference_label=reference_label,
    )
    _plot_profile(
        axes[1, 1],
        z_profile["z"],
        z_profile["u"],
        axis_name="z",
        title="Midplane z profile",
        reference_coordinate=z_reference_coordinate,
        reference_values=z_reference_values,
        reference_label=reference_label,
    )

    overview_paths = _save_figure_pair(fig, out_dir, "overview")

    diagnostics_paths: list[Path] = []
    if solution.diagnostics.time_history.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        time_history = solution.diagnostics.time_history
        axes[0].plot(
            time_history,
            solution.diagnostics.u_max_history,
            color="#1d4ed8",
            label="max |u|",
        )
        if solution.diagnostics.current_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.current_max_history,
                color="#b91c1c",
                label="max |J|",
            )
        if solution.diagnostics.lorentz_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.lorentz_max_history,
                color="#6d28d9",
                label="max |J×B|",
            )
        axes[0].set_title("Trace magnitudes")
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("magnitude")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        axes[1].plot(
            time_history,
            solution.diagnostics.residual_history,
            color="#0f766e",
            label="velocity residual",
        )
        if solution.diagnostics.potential_residual_history.size:
            axes[1].plot(
                time_history,
                solution.diagnostics.potential_residual_history,
                color="#b45309",
                label="potential residual",
            )
        axes[1].set_title("Solver residuals")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("residual")
        axes[1].set_yscale("log")
        axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        diagnostics_paths = _save_figure_pair(fig, out_dir, "diagnostics")

    return [*overview_paths, *diagnostics_paths]


# The command line (formerly ``lmhdx.io``).


def _build_case(args: argparse.Namespace):
    builders = {"hartmann": make_hartmann_case, "shercliff": make_shercliff_case, "hunt": make_hunt_case}
    if args.case not in builders:
        raise ValueError(args.case)
    geometry = {key: getattr(args, key) for key in ("width", "height", "ny", "nz") if hasattr(args, key)}
    return builders[args.case](ha=args.ha, output_dir=args.output, **geometry)


def _solve_case_with_optional_logger(
    case,
    *,
    solve_mode: str,
    logger=None,
    initial_state=None,
    initial_diagnostics=None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
):
    if solve_mode == "transient":
        return solve_fully_developed_transient(
            case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    start_time = 0.0 if initial_state is None else float(initial_state.time)
    return solve_fully_developed(case, logger=logger, start_time=start_time)


def _runtime_summary(
    solution,
    case,
    out_dir: Path,
    outputs: dict[str, list[Path]],
    *,
    restart_info: dict[str, object] | None = None,
) -> dict[str, object]:
    diag = getattr(solution, "diagnostics", None)
    geometry = getattr(getattr(case, "geometry", None), "kind", "unknown")
    u_field = getattr(solution.state, "u", jnp.asarray([0.0]))

    def _latest(name: str) -> float | None:
        history = getattr(diag, name, jnp.asarray([]))
        return float(history[-1]) if getattr(history, "size", 0) else None

    summary = {
        "case": case.name,
        "geometry": geometry,
        "solver_kind": getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless"),
        "solver_mode": getattr(getattr(case, "solver", None), "mode", "steady"),
        "converged": getattr(solution, "converged", None),
        "status": getattr(solution, "status", "not_recorded"),
        "steps": int(getattr(solution, "steps", 0)),
        "time": float(solution.state.time),
        "residual": float(solution.state.residual),
        "u_max": float(jnp.max(jnp.abs(u_field))),
        "potential_residual": _latest("potential_residual_history"),
        "potential_iterations_used": _latest("potential_iterations_history"),
        "linear_residual": _latest("linear_residual_history"),
        "linear_iterations_used": _latest("linear_iterations_history"),
        "volumetric_flow_rate": _latest("volumetric_flow_rate_history"),
        "mean_current_magnitude": _latest("mean_current_magnitude_history"),
        "lorentz_power": _latest("lorentz_power_history"),
        "div_current_max": _latest("div_current_max_history"),
        "charge_balance_residual": _latest("charge_balance_residual_history"),
        "gauge_residual": _latest("gauge_residual_history"),
        "interface_current_residual": _latest("interface_current_residual_history"),
        "output": _portable_path(out_dir),
        "generated_files": {
            key: [_portable_path(path) for path in paths] for key, paths in outputs.items() if paths
        },
    }
    if restart_info is not None:
        summary["restart"] = restart_info
    return summary


def _write_run_summary(summary: dict[str, object], case, out_dir: Path) -> Path | None:
    if not getattr(case.output, "write_json_summary", True):
        return None
    path = out_dir / f"{case.name}_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def _summary_exit_code(summary: dict[str, object]) -> int:
    """Return failure for a recorded unconverged steady result or a rejected thermal run."""

    if summary.get("solver_mode") == "thermal":
        return 0 if summary.get("accepted") else 2
    return 2 if summary.get("solver_mode") == "steady" and summary.get("converged") is False else 0


def _run_config(config: RunConfig) -> dict[str, object]:
    case = config.case
    output_dir = getattr(case, "output_dir", None)
    if output_dir is None and getattr(case, "output", None) is not None:
        output_dir = getattr(case.output, "directory", None)
    out_dir = Path(output_dir) if output_dir else Path.cwd() / "out" / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = StreamingSolverLogger(config.logging) if config.logging.enabled else None
    log_handle = None
    log_path: Path | None = None
    if logger is not None:
        log_path = default_log_path(out_dir, case.name)
        log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        logger.add_stream(log_handle)
    solve_start = time.perf_counter()
    initial_state = None
    initial_diagnostics = None
    restart_log_info = RestartLogInfo(enabled=False)
    restart_summary: dict[str, object] | None = None
    if config.restart.enabled:
        if config.restart.path is None:
            raise ValueError("Restart is enabled but no restart.path was provided")
        restart_bundle = load_restart_bundle(config.restart.path)
        validate_restart_bundle(
            restart_bundle,
            mesh=case_mesh(case),
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )
        initial_state = restart_bundle.state
        initial_diagnostics = restart_bundle.diagnostics
        restart_log_info = RestartLogInfo(
            enabled=True,
            path=str(restart_bundle.path),
            start_time=float(restart_bundle.state.time),
            reset_histories=config.restart.reset_histories,
        )
        restart_summary = {
            "enabled": True,
            "input": str(restart_bundle.path),
            "start_time": float(restart_bundle.state.time),
            "reset_histories": bool(config.restart.reset_histories),
        }
    try:
        solution = _solve_case_with_optional_logger(
            case,
            solve_mode=case.solver.mode,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=bool(config.restart.enabled and not config.restart.reset_histories),
            restart_info=restart_log_info,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
    outputs = write_solution_outputs(
        solution,
        case,
        out_dir,
        write_npz=getattr(case.output, "write_npz", True),
        write_plots=getattr(case.output, "write_plots", False),
    )
    if config.restart.write_restart:
        restart_filename = config.restart.restart_filename or f"{case.name}_restart.npz"
        restart_path = write_restart_npz(solution, case, out_dir / restart_filename)
        outputs.setdefault("restart", []).append(restart_path)
        if restart_summary is None:
            restart_summary = {"enabled": False}
        restart_summary["output"] = _portable_path(restart_path)
    if log_path is not None:
        outputs.setdefault("log", []).append(log_path)
    if config.input_path is not None and getattr(case.output, "copy_input_file", True):
        copied_input = out_dir / config.input_path.name
        shutil.copy2(config.input_path, copied_input)
        outputs.setdefault("input", []).append(copied_input)
    summary = _runtime_summary(solution, case, out_dir, outputs, restart_info=restart_summary)
    summary["execution_seconds"] = time.perf_counter() - solve_start
    summary_path = _write_run_summary(summary, case, out_dir)
    if summary_path is not None:
        outputs.setdefault("json", []).append(summary_path)
        summary["generated_files"]["json"] = [_portable_path(summary_path)]
    print(json.dumps(summary, indent=2))
    return summary


def _thermal_rejection(history, limits: dict, done: int) -> dict | None:
    """Return the first sample of a chunk's ``history`` outside the run's ``limits``, its step and reasons, or None.

    Every sample is checked, so a violation inside a chunk is not hidden by a valid last one; a NaN fails.
    """
    checks = {
        "not finite": ~np.asarray(history.finite),
        "the flow returns through the outlet": ~np.asarray(history.backflow_boundary_valid),
    }
    for name in ("advective_cfl", "buoyancy_courant"):
        if limits[name] is not None:
            checks[f"{name} over {limits[name]}"] = ~(np.asarray(getattr(history, name)) <= limits[name])
    failing = {reason: np.flatnonzero(mask) for reason, mask in checks.items()}
    samples = [int(found[0]) for found in failing.values() if found.size]
    if not samples:
        return None
    sample = min(samples)
    reasons = [reason for reason, found in failing.items() if sample in found]
    values = {
        name: float(np.asarray(getattr(history, name))[sample])
        for name in ("advective_cfl", "buoyancy_courant")
    }
    return {"step": done + sample + 1, "reasons": reasons, **values}


_LARGEST = ("heat_balance_residual", "advective_cfl", "buoyancy_courant", "boussinesq_measure")


def _fold_chunk(extremes: dict, history) -> dict:
    """The summary's finite flag, largest magnitude of each of ``_LARGEST`` and last sample, with one chunk more.

    Only floats are kept, so a run's memory does not grow with its step count. A NaN, once seen, stays, as a
    maximum over the whole history would give.
    """
    largest = {}
    for name in _LARGEST:
        chunk = np.max(np.abs(np.asarray(getattr(history, name))))  # NaN if any sample is
        largest[name] = float(np.maximum(extremes["largest"].get(name, chunk), chunk))  # and NaN stays
    return {
        "finite": extremes["finite"] and bool(np.all(np.asarray(history.finite))),
        "last": {name: float(np.asarray(values)[-1]) for name, values in history._asdict().items()},
        "largest": largest,
    }


def run_thermal(run) -> dict[str, object]:
    """Run a :class:`lmhdx.thermal.ThermalRun` in compiled chunks, accepting each before recording it.

    Every sample of a chunk must be finite, keep a valid outlet, and stay within the scheme's advective Courant
    limit (upwind 0.8, van Leer 0.5, none for conduction alone) and a configured ``buoyancy_courant_limit``; the start is held
    to the same limits. ``[run] advective_cfl_limit`` replaces the scheme's, recorded as an override. An accepted
    chunk writes the record ``{name}_{step:06d}.npz`` and the frame ``{name}_{step:06d}.vtr``, ``step`` counted
    from the record's own time on a restart, gathered in ``{name}.pvd`` from step 0 and in
    ``{name}_from_{first:06d}.pvd`` from a restart, so a resume keeps the earlier series. The first rejected
    chunk stops the run and writes only ``{name}_{step:06d}_rejected.npz``, whose run provenance names the
    reasons, so it imports but does not resume. ``thermal_summary.json`` holds ``accepted``, the first rejected
    sample (its step, reasons and Courant numbers), the limits, the last diagnostics
    and the time they are at (the end of the rejected chunk on a rejection, otherwise of the run) and the extremes over the
    run of the ledger, the Courant numbers and the Boussinesq measure, which no limit applies to.
    """
    from .thermal import (
        _SCHEMES,
        advance_thermal,
        build_thermal_step,
        validate_thermal_inputs,
        write_thermal_record,
    )

    problem, state, heat = run.problem, run.state, run.heat_source
    grid = problem.flow.grid
    scheme = _SCHEMES[problem.advection]
    courant = run.limits["advective_cfl"] if problem.advection != "off" else None
    limits = {
        "advective_cfl": scheme if courant is None else courant,
        "advective_cfl_override": courant is not None and courant != scheme,
        "buoyancy_courant": run.limits["buoyancy_courant"],
    }
    start_limits = {"buoyancy_courant_limit": limits["buoyancy_courant"]}
    if limits["advective_cfl"] is not None:
        start_limits["advective_cfl_limit"] = limits["advective_cfl"]
    validate_thermal_inputs(
        problem, state, heat, divergence_tolerance=run.divergence_tolerance, **start_limits
    )
    thermal_step = build_thermal_step(problem, dtype=jnp.result_type(float), hold_flow=run.hold_flow)
    chunk = jax.jit(lambda start: advance_thermal(thermal_step, start, heat, run.chunk))
    first, frames, rejected = round(float(state.time) / float(problem.flow.dt)), [], None
    accepted, done = state, first
    extremes = {"finite": True, "last": {}, "largest": {}}
    for index in range(run.steps // run.chunk):
        state, history = chunk(accepted)
        extremes = _fold_chunk(extremes, history)  # the chunk's arrays are then dropped
        rejected, label = _thermal_rejection(history, limits, done), f"{run.name}_{done + run.chunk:06d}"
        if rejected is not None:
            rejected["chunk"] = index
            if run.write_restart:
                provenance = {**run.provenance, "rejected": rejected["reasons"]}
                write_thermal_record(
                    run.directory / f"{label}_rejected", problem, state, units=run.units, run=provenance
                )
            break
        accepted, done = state, done + run.chunk
        if run.write_restart:
            write_thermal_record(run.directory / label, problem, state, units=run.units, run=run.provenance)
        if run.write_paraview:
            centred = [cell_average(field, axis).data for axis, field in enumerate(state.velocity)]
            cells = {"temperature": state.temperature.data, "velocity": jnp.stack(centred, axis=-1)}
            frames.append((float(state.time), write_vtr(run.directory / label, grid.faces, **cells).name))
    if frames:
        write_pvd(frames, run.directory, name=run.name if first == 0 else f"{run.name}_from_{first:06d}")
    summary = {
        "case": run.name,
        "solver_mode": "thermal",
        "units": run.units,
        "accepted": rejected is None,
        "rejected": rejected,
        "limits": limits,
        "restart": run.restart,
        "steps": done - first,
        "time": float(state.time),
        **extremes,
    }
    run.directory.mkdir(parents=True, exist_ok=True)
    (run.directory / "thermal_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


_TOML_TYPES = {
    "table": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}
_TOML_SCHEMA = {  # each table's keys and TOML types, "name:type"; "vector" is an array of three numbers
    "": "schema_version:integer case:table solver:table grid:table flow:table thermal:table run:table"
    " output:table restart:table",
    "case": "name:string units:string",
    "solver": "kind:string",
    "grid": "x:table y:table z:table",
    "axis": "cells:integer lower:number upper:number layer_thickness:number cells_in_layer:integer",
    "flow": "conditions:array advection:string initial_velocity:string dt:number density:number viscosity:number"
    " conductivity:number magnetic_field:vector forcing:vector wall_conductance:vector",
    "thermal": "joule_heating:boolean advection:string hold_flow:boolean gravity:vector x:table y:table z:table"
    " specific_heat:number thermal_conductivity:number reference_temperature:number"
    " expansion_coefficient:number initial_temperature:number heat_source:number",
    "wall": "kind:string lower:number upper:number",
    "run": "steps:integer chunk:integer divergence_tolerance:number advective_cfl_limit:number"
    " buoyancy_courant_limit:number",
    "output": "directory:string write_paraview:boolean write_restart:boolean",
    "restart": "path:string mode:string",
}


def _toml_table(table, schema: str, where: str, required: str = "") -> dict:
    """Return a copy of ``table`` with every key ``schema`` knows, of its TOML type, and every ``required`` one."""
    kinds = dict(item.split(":") for item in _TOML_SCHEMA[schema].split())
    if not isinstance(table, dict):
        raise ValueError(f"[{where}] must be a table")
    unknown, missing = sorted(set(table) - set(kinds)), [key for key in required.split() if key not in table]
    if unknown or missing:
        raise ValueError(f"[{where}] has unknown keys {unknown}" if unknown else f"[{where}] lacks {missing}")

    def typed(kind, value):
        if kind == "vector":
            return (
                isinstance(value, list) and len(value) == 3 and all(typed("number", item) for item in value)
            )
        return isinstance(value, _TOML_TYPES[kind]) and (kind == "boolean" or not isinstance(value, bool))

    for key, value in table.items():
        if not typed(kinds[key], value):
            label = "array of three numbers" if kinds[key] == "vector" else kinds[key]
            raise ValueError(f"[{where}] {key} must be a TOML {label}, not {value!r}")
    return {key: tuple(value) if kinds[key] == "vector" else value for key, value in table.items()}


def _toml_grid(table: dict):
    """Build the :class:`lmhdx.grid.Grid` of ``[grid]``: each axis ``{cells, lower, upper}``, wall-resolving
    with ``layer_thickness`` (and ``cells_in_layer``)."""
    from .grid import Grid, uniform_faces, wall_resolving_faces

    faces = []
    for axis in "xyz":
        spec = _toml_table(table[axis], "axis", f"grid.{axis}", "cells lower upper")
        ends, layer = (spec["cells"], spec["lower"], spec["upper"]), spec.get("layer_thickness")
        if layer is None and "cells_in_layer" in spec:
            raise ValueError(f"[grid.{axis}] cells_in_layer needs layer_thickness")
        faces.append(
            uniform_faces(*ends)
            if layer is None
            else wall_resolving_faces(
                *ends, layer_thickness=layer, cells_in_layer=spec.get("cells_in_layer", 4), max_ratio=None
            )
        )
    return Grid(*faces)


def _toml_flow(table: dict, grid):
    """Build the :class:`lmhdx.core3d.ChannelProblem` of ``[flow]`` and its ``initial_velocity``: ``conditions``
    are ``"periodic"`` or ``"wall"`` per axis (an inflow-outflow axis is not read from TOML yet)."""
    from .core3d import ChannelProblem
    from .grid import NEUMANN, PERIODIC
    from .grid import BoundaryCondition as Boundary

    kinds, table = {"periodic": PERIODIC, "wall": NEUMANN}, dict(table)
    conditions, start = table.pop("conditions"), table.pop("initial_velocity", "rest")
    if len(conditions) != 3 or any(kind not in kinds for kind in conditions):
        raise ValueError(
            '[flow] conditions are three of "periodic" and "wall"; an open axis is not read from TOML yet'
        )
    if start not in ("rest", "steady"):
        raise ValueError('[flow] initial_velocity is "rest" or "steady"')
    return (
        ChannelProblem(grid, tuple(Boundary(kinds[kind]) for kind in conditions), **table),
        conditions,
        start,
    )


def thermal_run_from_toml(root: dict, base: str | Path = ".", *, version: int = 1):
    """Read a thermal run from a parsed TOML document with ``[solver] kind = "thermal"``; ``base`` resolves paths.

    The tables are those of ``docs/physics/thermal.md``, with the Python names as keys. Each refuses an unknown
    or missing key and a value of the wrong TOML type (neither converted) before the run is built;
    ``schema_version`` is 1, the only one, and may be left out. A wall axis, and only that, has a table
    ``[thermal.y]`` for :func:`lmhdx.thermal.thermal_boundary`. ``[restart]`` is a record of
    :func:`lmhdx.thermal.write_thermal_record`: ``"resume"``, the default, continues the run that wrote it,
    ``"import"`` starts from its state.
    """
    from .grid import PERIODIC, Field
    from .grid import BoundaryCondition as Boundary
    from .steady import solve_steady_state
    from .thermal import (
        _CELL,
        ThermalProblem,
        ThermalRun,
        initial_thermal_state,
        load_thermal_record,
        thermal_boundary,
    )

    base, tables = Path(base), _toml_table(root, "", "top level")
    if tables.get("schema_version", version) != version:
        raise ValueError(f"schema_version must be {version}")
    required = {"case": "name units", "solver": "kind", "flow": "conditions dt", "run": "steps"}
    required |= {"thermal": "specific_heat thermal_conductivity reference_temperature", "output": "directory"}
    case, solver, flow, run, thermal, output, restart = (
        _toml_table(tables.get(name, {}), name, name, keys)
        for name, keys in (required | {"restart": ""}).items()
    )
    if solver["kind"] != "thermal":
        raise ValueError('a thermal run needs [solver] kind = "thermal"')
    flow, conditions, start = _toml_flow(
        flow, _toml_grid(_toml_table(tables.get("grid", {}), "grid", "grid", "x y z"))
    )
    hold_flow, temperature, source = (
        thermal.pop(key, value)
        for key, value in (("hold_flow", False), ("initial_temperature", 0.0), ("heat_source", 0.0))
    )
    boundaries = []
    for axis, kind in zip("xyz", conditions):
        if (axis in thermal) == (kind == "periodic"):
            raise ValueError(f"[thermal.{axis}] is needed for a wall axis and refused for a periodic one")
        if kind == "periodic":
            boundaries.append(Boundary(PERIODIC))
            continue
        wall = _toml_table(thermal.pop(axis), "wall", f"thermal.{axis}", "kind")
        if wall["kind"] not in ("temperature", "heat_flux"):
            raise ValueError(f'[thermal.{axis}] kind is "temperature" or "heat_flux"')
        ends, k = (wall.get("lower", 0.0), wall.get("upper", 0.0)), thermal["thermal_conductivity"]
        boundaries.append(thermal_boundary(wall["kind"], *ends, conductivity=k))
    steps, chunk = run["steps"], run.get("chunk", run["steps"])
    if steps < 1 or chunk < 1 or steps % chunk:
        raise ValueError("[run] steps must be a positive multiple of chunk")
    restart = {"mode": "resume", **restart} if restart else None
    if restart is not None and ("path" not in restart or restart["mode"] not in ("resume", "import")):
        raise ValueError('[restart] needs a path, and mode "resume" or "import"')
    problem = ThermalProblem(flow, tuple(boundaries), **thermal)
    provenance = {"heat_source": float(source), "hold_flow": hold_flow}
    if restart is not None:
        resume = provenance if restart["mode"] == "resume" else None
        state = load_thermal_record(base / restart["path"], problem, units=case["units"], run=resume)
    else:
        velocity = solve_steady_state(flow).velocity if start == "steady" else None
        state = initial_thermal_state(problem, temperature, velocity)
    heat = Field(jnp.full(flow.grid.shape, source, dtype=jnp.result_type(float)), _CELL, flow.grid)
    limits = {key: run.get(f"{key}_limit") for key in ("advective_cfl", "buoyancy_courant")}
    return ThermalRun(
        case["name"], case["units"], problem, state, heat, hold_flow, steps, chunk,
        float(run.get("divergence_tolerance", 1e-10)), base / output["directory"],
        output.get("write_paraview", True), output.get("write_restart", True), provenance, limits, restart,
    )  # fmt: skip


def run_from_toml(path: str | Path) -> dict[str, object]:
    root = _load_toml(path)
    if root.get("solver", {}).get("kind") == "thermal":
        return run_thermal(thermal_run_from_toml(root, Path(path).resolve().parent))
    config = load_run_config(path)
    return _run_config(config)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    # A TOML case runs directly (`lmhdx case.toml`); `run case.toml` is kept for old scripts.
    toml = argv[1:2] if argv[:1] == ["run"] else argv[:1]
    if toml and Path(toml[0]).suffix == ".toml":
        return _summary_exit_code(run_from_toml(toml[0]))

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="lmx" if sys.argv and Path(sys.argv[0]).name == "lmx" else "lmhdx",
        description="Run and validate differentiable inductionless MHD cases.",
        epilog="A TOML case may also be passed directly: lmhdx CASE.toml",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", title="commands", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a named built-in case.",
        description="Run a fully developed duct case.",
        formatter_class=formatter,
    )
    run_parser.add_argument(
        "case",
        choices=[
            "hartmann",
            "shercliff",
            "hunt",
        ],
        help="Built-in case or solver family.",
    )
    run_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    run_parser.add_argument("--output", default="./out", help="Output directory.")
    run_parser.add_argument(
        "--mode",
        choices=["steady", "transient"],
        default="steady",
        help="Solve mode for fully developed cases.",
    )
    run_parser.add_argument("--plots", action="store_true", help="Write summary plots.")
    run_parser.add_argument("--quiet", action="store_true", help="Disable solver logging.")
    geometry = run_parser.add_argument_group("geometry and resolution")
    geometry.add_argument("--width", type=float, default=2.0, help="Duct width.")
    geometry.add_argument("--height", type=float, default=2.0, help="Duct height.")
    geometry.add_argument("--ny", type=int, default=48, help="Cross-stream y cells.")
    geometry.add_argument("--nz", type=int, default=48, help="Cross-stream z cells.")

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Time a bounded Hartmann solve.",
        description="Measure cold and warm runtime for a portable Hartmann case.",
        formatter_class=formatter,
    )
    bench_parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions.")
    bench_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    bench_parser.add_argument("--ny", type=int, default=48, help="Mesh y cells.")
    bench_parser.add_argument("--nz", type=int, default=48, help="Mesh z cells.")
    bench_parser.add_argument("--output", default="", help="Optional JSON report path.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Solve a duct case and write validation metrics.",
        description="Solve a duct case and write profiles, diagnostics, and Hartmann acceptance metrics.",
        formatter_class=formatter,
    )
    validate_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"], help="Validation case.")
    validate_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    validate_parser.add_argument("--output", default="./out", help="Output directory.")
    validate_parser.add_argument(
        "--hartmann-l2-threshold", type=float, default=0.05, help="Hartmann L2 gate."
    )
    validate_parser.add_argument(
        "--hartmann-linf-threshold", type=float, default=0.1, help="Hartmann Linf gate."
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        payload = benchmark_solver(repeats=args.repeats, ha=args.ha, ny=args.ny, nz=args.nz)
        if args.output:
            write_benchmark_report(payload, args.output)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "validate":
        case = _build_case(args)
        solution = _solve_case_with_optional_logger(case, solve_mode="steady", logger=None)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_paraview(solution, out_dir)
        write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
        z_profile = extract_midplane_profile(solution, axis="z")
        write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", z_profile)
        payload = validation_summary(solution, case.name, ha=args.ha)
        payload.update(
            converged=getattr(solution, "converged", None),
            status=getattr(solution, "status", "not_recorded"),
            steps=int(getattr(solution, "steps", 0)),
        )
        if args.case == "hartmann":
            comparison = hartmann_validation(solution, args.ha)
            write_analytic_comparison(comparison, out_dir / f"{case.name}_analytic.json", axis_name="y")
            acceptance = hartmann_acceptance(
                solution,
                args.ha,
                l2_threshold=args.hartmann_l2_threshold,
                linf_threshold=args.hartmann_linf_threshold,
            )
            write_acceptance_report(acceptance, out_dir / f"{case.name}_acceptance.json")
            payload["accepted"] = float(acceptance.passed)
            payload["acceptance_l2_threshold"] = acceptance.l2_threshold
            payload["acceptance_linf_threshold"] = acceptance.linf_threshold
        write_metrics_json(payload, out_dir / f"{case.name}_metrics.json")
        print(json.dumps(payload, indent=2))
        return 2 if getattr(solution, "converged", None) is False else 0

    case = _build_case(args)
    case = replace(
        case,
        solver=replace(case.solver, mode=args.mode),
        output=replace(
            case.output,
            directory=args.output,
            write_npz=True,
            write_json_summary=True,
            write_plots=args.plots,
        ),
    )
    config = RunConfig(case=case)
    logging = config.logging
    if args.quiet:
        logging = replace(logging, enabled=False)
    if logging is not config.logging:
        config = replace(config, logging=logging)
    return _summary_exit_code(_run_config(config))
