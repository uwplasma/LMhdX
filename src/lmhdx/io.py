from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh
from .specs import Diagnostics, MHDState, Solution
from .validation import extract_midplane_profile

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


def enable_compilation_cache(
    cache_dir: str | Path | None = None,
    *,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = -1,
    share_across_values: bool = False,
) -> Path:
    """Enable JAX's persistent compilation cache before a heavy compile.

    With ``share_across_values`` the arrays a solve closes over (the field, the
    factorizations) are passed to the compiled program as arguments rather than
    embedded in it, where the installed JAX supports it, so a new Hartmann
    number or field on the same mesh reuses the cached executable: on an RTX
    A4000 a new Hartmann number then compiled in 0.45 s against 4.4 s. The
    arguments are handed over on every call, which slowed a warm 48-cell solve
    from 6.0 to 8–10 ms on a CPU, so it suits sweeps over new problems rather
    than loops that reuse one compiled function; the default keeps them embedded.
    """

    target = Path(cache_dir or (Path.home() / ".cache" / "lmhdx" / "jax_compilation"))
    target.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(target))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    if share_across_values and "jax_use_simplified_jaxpr_constants" in jax.config.values:
        jax.config.update("jax_use_simplified_jaxpr_constants", True)
    return target


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


def _rectilinear_points(mesh: StructuredMesh) -> str:
    return (
        f"<Coordinates>\n"
        f'<DataArray type="Float64" Name="X" format="ascii">{_array_text(mesh.x_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Y" format="ascii">{_array_text(mesh.y_faces)}</DataArray>\n'
        f'<DataArray type="Float64" Name="Z" format="ascii">{_array_text(mesh.z_faces)}</DataArray>\n'
        f"</Coordinates>"
    )


def _cell_data(solution: Solution) -> str:
    fields = {
        "u": solution.state.u,
        "phi": solution.state.phi,
        "jy": solution.state.jy,
        "jz": solution.state.jz,
        "lorentz_x": solution.state.lorentz_x,
    }
    arrays = []
    for name, field in fields.items():
        cell = field[None, :, :]
        arrays.append(
            f'<DataArray type="Float64" Name="{name}" format="ascii">{_array_text(cell)}</DataArray>'
        )
    return "<CellData>\n" + "\n".join(arrays) + "\n</CellData>"


def write_vtr(solution: Solution, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{solution.case_name}.vtr"
    mesh = solution.mesh
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
        f'<RectilinearGrid WholeExtent="0 {mesh.nx} 0 {mesh.ny} 0 {mesh.nz}">\n'
        f'<Piece Extent="0 {mesh.nx} 0 {mesh.ny} 0 {mesh.nz}">\n'
        f"{_cell_data(solution)}\n"
        f"{_rectilinear_points(mesh)}\n"
        "</Piece>\n"
        "</RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    target.write_text(content)
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
    paths = []
    paths.append(write_vtr(solution, out_dir))
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
    from .validation import (
        extract_centerline,
        extract_midplane_profile,
        write_profile_csv,
    )

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
